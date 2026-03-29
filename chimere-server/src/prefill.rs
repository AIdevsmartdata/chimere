//! # Batch Prompt Prefill — Chimère Engine
//!
//! Accelerated prefill for Qwen3.5: instead of processing prompt tokens one by
//! one (N forward passes), this module batches all linear projections into a
//! single QMatMul call per layer, then runs only the cheap sequential recurrence.
//!
//! ## Speedup rationale
//!
//! For a prompt of N tokens, the old approach cost:
//!   N × (64 matmuls per GDN layer + 4 matmuls per attn layer)
//!
//! With batch prefill:
//!   1 matmul call per weight matrix (GPU batched, ~N× faster for large N)
//!   + N sequential recurrence iterations (cheap — no matmuls, only tensor ops
//!     that fit in SRAM)
//!   + Full causal self-attention for attention layers (O(N²) but cheap vs QMatMul)
//!
//! ## Architecture reference
//!
//! - 64 main layers: 48 GDN (layers where `(i+1) % 4 != 0`) + 16 attention
//! - hidden_size = 5120, num_attention_heads = 24, num_kv_heads = 4, head_dim = 256
//! - GDN: ssm_dt_rank = 48, ssm_d_state = 128, ssm_n_group = 16, ssm_d_inner = 6144
//! - conv_channels = 10240, conv_kernel = 4
//! - MRoPE for attention (rope_sections [11, 11, 10, 0])
//!
//! ## Entry point
//!
//! Call `Qwen35Model::forward_prefill(tokens, state)` — implemented as a method
//! on `Qwen35Model` in `qwen35_model.rs` (needs private field access).
//! This module provides the shared helper functions used by that implementation.

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Public helper: causal attention mask
// ---------------------------------------------------------------------------

/// Build a lower-triangular causal mask for self-attention.
///
/// Returns a `[seq_len, seq_len]` tensor where:
/// - Entries on and below the diagonal are `0.0` (positions may attend)
/// - Entries above the diagonal are `-inf` (future positions are masked)
///
/// This mask is added to the raw attention scores **before** softmax, so
/// adding `-inf` makes the corresponding softmax weight exactly zero.
///
/// The mask is created in F32 and can be cast with `.to_dtype()` if needed.
/// For the attention computation in `forward_prefill` it is used as F32.
///
/// # Arguments
/// - `seq_len` — number of tokens in the prompt
/// - `device`  — target device (must match query/key tensors)
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // Fast path: single-token sequences don't need masking
    if seq_len == 1 {
        return Tensor::zeros((1, 1), DType::F32, device);
    }

    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        // Token i cannot attend to tokens j > i (future)
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_data, (seq_len, seq_len), device)
}

// ---------------------------------------------------------------------------
// Internal helper: tile groups (matches ggml_repeat_4d tiling convention)
// ---------------------------------------------------------------------------

/// Tile a `[N, n_group, d_state]` tensor to `[N, dt_rank, d_state]` using
/// the same tiling convention as `ggml_repeat_4d` / `llama.cpp`.
///
/// `ggml_repeat_4d` tiles the repetitions *across* the group dimension:
///   output = cat([src, src, src, ...], dim=1)
///
/// This produces `[s0,s1,...,s15, s0,s1,...,s15, s0,s1,...,s15]`
/// NOT `[s0,s0,s0, s1,s1,s1, ...]` which `expand+reshape` would give.
///
/// # Arguments
/// - `x`       — `[N, n_group, d_state]`
/// - `repeats` — how many times to tile (`dt_rank / n_group`)
pub(crate) fn tile_groups(x: &Tensor, repeats: usize) -> Result<Tensor> {
    let parts: Vec<&Tensor> = (0..repeats).map(|_| x).collect();
    Tensor::cat(&parts, 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_causal_mask_shape() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device)?;
        assert_eq!(mask.dims(), &[4, 4], "Mask shape should be [seq_len, seq_len]");
        Ok(())
    }

    #[test]
    fn test_causal_mask_values() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 4;
        let mask = create_causal_mask(seq_len, &device)?;
        let data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        for i in 0..seq_len {
            for j in 0..seq_len {
                let val = data[i * seq_len + j];
                if j <= i {
                    assert!(
                        val == 0.0,
                        "mask[{i},{j}] should be 0.0 (attend), got {val}"
                    );
                } else {
                    assert!(
                        val.is_infinite() && val < 0.0,
                        "mask[{i},{j}] should be -inf (mask), got {val}"
                    );
                }
            }
        }

        println!("Causal mask [4,4] validated: lower-triangular 0.0, upper-triangular -inf");
        Ok(())
    }

    #[test]
    fn test_causal_mask_single_token() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(1, &device)?;
        assert_eq!(mask.dims(), &[1, 1]);
        let val: f32 = mask.flatten_all()?.get(0)?.to_scalar()?;
        assert_eq!(val, 0.0, "Single-token mask should be 0.0");
        println!("Single-token causal mask: OK");
        Ok(())
    }

    #[test]
    fn test_tile_groups_shape_and_values() -> Result<()> {
        let device = Device::Cpu;
        // Build a [2, 2, 3] tensor with known values
        // token 0: groups [[1,2,3],[4,5,6]]
        // token 1: groups [[7,8,9],[10,11,12]]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0,  // tok0, grp0
            4.0, 5.0, 6.0,  // tok0, grp1
            7.0, 8.0, 9.0,  // tok1, grp0
            10.0, 11.0, 12.0, // tok1, grp1
        ];
        let x = Tensor::from_vec(data, (2, 2, 3), &device)?;

        // repeats=3 → [2, 6, 3]
        let tiled = tile_groups(&x, 3)?;
        assert_eq!(tiled.dims(), &[2, 6, 3], "Tiled shape should be [N, n_group*repeats, d_state]");

        // ggml_repeat_4d tiling: cat([grp0,grp1, grp0,grp1, grp0,grp1], dim=1)
        // token 0, head 0 (first copy of grp0): [1,2,3]
        // token 0, head 1 (first copy of grp1): [4,5,6]
        // token 0, head 2 (second copy of grp0): [1,2,3]
        // ... etc
        let flat: Vec<f32> = tiled.flatten_all()?.to_vec1()?;
        // tok0, head0 = [1,2,3]
        assert_eq!(&flat[0..3], &[1.0, 2.0, 3.0], "tok0,head0 should be grp0");
        // tok0, head1 = [4,5,6]
        assert_eq!(&flat[3..6], &[4.0, 5.0, 6.0], "tok0,head1 should be grp1");
        // tok0, head2 = [1,2,3] (second copy of grp0 = tiling, not interleaving)
        assert_eq!(&flat[6..9], &[1.0, 2.0, 3.0], "tok0,head2 should be grp0 again (tiled)");

        println!("tile_groups [2,2,3] × 3 → [2,6,3]: tiling verified");
        Ok(())
    }

    /// Verify that the causal mask integrates correctly with softmax:
    /// masked positions should receive ~0 weight after softmax.
    #[test]
    fn test_causal_mask_with_softmax() -> Result<()> {
        use candle_nn::ops::softmax;
        use candle_core::D;

        let device = Device::Cpu;
        let seq_len = 3;

        // Arbitrary score matrix [1, 1, seq_len, seq_len]
        let scores_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
        ];
        let scores = Tensor::from_vec(scores_data, (1, 1, seq_len, seq_len), &device)?;
        let mask = create_causal_mask(seq_len, &device)?
            .reshape((1, 1, seq_len, seq_len))?;

        let masked = (&scores + &mask)?;
        let weights = softmax(&masked, D::Minus1)?;
        let w: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        // Token 0 can only attend to position 0
        assert!(w[1] < 1e-6, "weight[0,1] should be ~0 (masked), got {}", w[1]);
        assert!(w[2] < 1e-6, "weight[0,2] should be ~0 (masked), got {}", w[2]);
        assert!((w[0] - 1.0).abs() < 1e-5, "weight[0,0] should be 1.0, got {}", w[0]);

        // Token 2 can attend to all 3 positions
        let row2 = &w[6..9];
        let sum: f32 = row2.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row 2 weights should sum to 1.0, got {}", sum);
        // All weights should be positive (no masking for token 2)
        for (j, &wij) in row2.iter().enumerate() {
            assert!(wij > 0.0, "weight[2,{j}] should be >0, got {wij}");
        }

        println!("Causal mask + softmax: masking verified, weights sum to 1.0");
        Ok(())
    }
}
