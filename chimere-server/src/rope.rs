//! # Rotary Position Embeddings (RoPE) — Chimère Engine
//!
//! Standard Llama-style Rotary Position Embedding (RoPE) as described in
//! "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//! (Su et al., 2021, https://arxiv.org/abs/2104.09864).
//!
//! ## How RoPE Works
//!
//! Instead of adding positional information to token embeddings, RoPE rotates
//! query and key vectors by a position-dependent angle. For a vector split
//! into pairs (x1, x2), the rotation is:
//!
//! ```text
//! x1' = x1 * cos(m * θ_i) - x2 * sin(m * θ_i)
//! x2' = x1 * sin(m * θ_i) + x2 * cos(m * θ_i)
//! ```
//!
//! where m is the token position and θ_i = base^(-2i/d) is the frequency
//! for dimension pair i.
//!
//! ## Why RoPE for Chimère
//!
//! The 12 sparse GQA layers in Chimère's 3:1 hybrid (3 GatedDeltaNet : 1 GQA)
//! need positional encoding. RoPE is the natural choice because:
//!
//! - Relative positions emerge naturally from dot products: RoPE(q,m)·RoPE(k,n)
//!   depends only on (m-n), giving implicit relative attention.
//! - Long-context extrapolation (with NTK or YaRN scaling, future work).
//! - Used in Qwen3-Coder-Next, Llama 3, Mistral — well-validated.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let rope = RotaryEmbedding::new(128, 4096, 500000.0, DType::F32, &device)?;
//! let x = Tensor::randn(0f32, 1f32, (2, 16, 8, 128), &device)?;
//! let x_rotated = rope.apply(&x, 0)?;  // offset=0 for first chunk
//! ```

use candle_core::{DType, Device, Result, Tensor};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// RotaryEmbedding
// ---------------------------------------------------------------------------

/// Precomputed cosine and sine tables for Rotary Position Embedding.
///
/// The tables are computed once at construction and reused for all forward
/// passes. They cover positions 0..max_seq_len and half the head dimension
/// (since each pair (x1, x2) shares one (cos, sin) value).
pub struct RotaryEmbedding {
    /// Cosine cache: [max_seq_len, head_dim/2]
    cos_cache: Tensor,
    /// Sine cache: [max_seq_len, head_dim/2]
    sin_cache: Tensor,
    /// Full head dimension (must be even)
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Create a new RoPE module with precomputed cos/sin tables.
    ///
    /// # Arguments
    ///
    /// - `head_dim`: Dimension per attention head (must be even)
    /// - `max_seq_len`: Maximum sequence length to precompute for
    /// - `theta`: Base frequency (10000.0 for Llama, 500000.0 for Llama 3 / Qwen3)
    /// - `dtype`: Target dtype for the cache tensors
    /// - `device`: Target device
    ///
    /// # Frequency Formula
    ///
    /// For dimension index i in 0..head_dim/2:
    ///   freq_i = 1.0 / (theta ^ (2*i / head_dim))
    ///
    /// Positions t = 0..max_seq_len are outer-producted with freqs to get
    /// the angle matrix freqs[t, i] = t * freq_i.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        assert!(
            head_dim % 2 == 0,
            "head_dim must be even for RoPE, got {}",
            head_dim
        );

        let half_dim = head_dim / 2;

        // Compute inverse frequencies in f32: freq_i = 1 / (theta ^ (2i / head_dim))
        // i ranges from 0 to half_dim - 1
        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                (1.0 / theta.powf(exponent)) as f32
            })
            .collect();

        // freq tensor: [half_dim]
        let inv_freq = Tensor::from_vec(inv_freqs, (half_dim,), &Device::Cpu)?;

        // Position indices: [max_seq_len]
        let positions: Vec<f32> = (0..max_seq_len).map(|t| t as f32).collect();
        let t = Tensor::from_vec(positions, (max_seq_len,), &Device::Cpu)?;

        // Outer product: freqs[pos, i] = t[pos] * inv_freq[i]
        // t: [max_seq_len, 1] * inv_freq: [1, half_dim] -> [max_seq_len, half_dim]
        let t_col = t.unsqueeze(1)?;           // [max_seq_len, 1]
        let freq_row = inv_freq.unsqueeze(0)?; // [1, half_dim]
        let freqs = t_col.broadcast_mul(&freq_row)?; // [max_seq_len, half_dim]

        // Precompute cos and sin in f32, then cast to target dtype
        let cos_cache = freqs.cos()?.to_dtype(dtype)?.to_device(device)?;
        let sin_cache = freqs.sin()?.to_dtype(dtype)?.to_device(device)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Apply RoPE to an input tensor.
    ///
    /// # Arguments
    ///
    /// - `x`: Input tensor with shape [batch, seq_len, num_heads, head_dim]
    /// - `offset`: Starting position index (for KV-cache / chunked inference).
    ///             Use 0 for full-sequence encoding.
    ///
    /// # Returns
    ///
    /// Rotated tensor with the same shape as `x`.
    ///
    /// # Rotation Formula
    ///
    /// Split x along head_dim into x1 (first half) and x2 (second half):
    ///
    /// ```text
    /// out = cat(x1*cos - x2*sin, x1*sin + x2*cos, dim=-1)
    /// ```
    ///
    /// The cos/sin tensors are sliced from the cache for positions
    /// offset..offset+seq_len and reshaped to [1, seq_len, 1, half_dim]
    /// for broadcasting over batch and num_heads.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, seq_len, num_heads, head_dim) = x.dims4()?;
        assert_eq!(
            head_dim, self.head_dim,
            "Input head_dim {} does not match RoPE head_dim {}",
            head_dim, self.head_dim
        );

        let half_dim = self.head_dim / 2;

        // Split x into first and second halves along the last dim
        // x1: [batch, seq_len, num_heads, half_dim]
        // x2: [batch, seq_len, num_heads, half_dim]
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // Slice cos/sin cache for current positions: offset..offset+seq_len
        // Shape: [seq_len, half_dim]
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        // Reshape for broadcast over batch and num_heads:
        // [seq_len, half_dim] -> [1, seq_len, 1, half_dim]
        let cos = cos.reshape((1, seq_len, 1, half_dim))?;
        let sin = sin.reshape((1, seq_len, 1, half_dim))?;

        // Expand to [batch, seq_len, num_heads, half_dim] for the multiply
        let cos = cos.expand((batch, seq_len, num_heads, half_dim))?;
        let sin = sin.expand((batch, seq_len, num_heads, half_dim))?;

        // Apply rotation:
        //   out1 = x1 * cos - x2 * sin
        //   out2 = x1 * sin + x2 * cos
        let out1 = (x1.mul(&cos)? - x2.mul(&sin)?)?;
        let out2 = (x1.mul(&sin)? + x2.mul(&cos)?)?;

        // Concatenate along the last dimension: [batch, seq_len, num_heads, head_dim]
        Tensor::cat(&[&out1, &out2], 3)
    }
}

// ---------------------------------------------------------------------------
// MRoPE (Multi-Resolution Rotary Position Embedding)
// ---------------------------------------------------------------------------
//
// Qwen3.5 uses a partial, multi-dimensional RoPE over only the first `n_rot`
// dims of each head (partial_rotary_factor = 0.25 → n_rot = 64 out of 256).
// Those 64 dims (32 pairs) are split across up to 4 sections, each encoding a
// different positional axis:
//
//   Section 0 (11 pairs = 22 dims): temporal / token position
//   Section 1 (11 pairs = 22 dims): image height  (0 for text)
//   Section 2 (10 pairs = 20 dims): image width   (0 for text)
//   Section 3 ( 0 pairs):           unused
//
// Per-pair frequencies are shared across sections (global pair index `g`):
//
//   g = section_offset + pair_within_section
//   theta_g = rope_theta ^ (-2 * g / n_rot)
//
// For each token, a 4-element position vector `[t, h, w, _]` is supplied.
// The rotation applied to section `s` uses `positions[token][s]` as the
// scalar position; this makes section 0 encode time and sections 1-2 encode
// 2-D spatial coordinates, all with independent position counters.
//
// For text-only inference every token has position `[token_idx, 0, 0, 0]`.

/// Default maximum position for the precomputed cos/sin cache.
/// 65536 covers 64K context; uses ~1.1 MB total (3 sections, f32, cos+sin).
const MROPE_MAX_POS: usize = 65536;

/// Multi-Resolution Rotary Position Embedding for Qwen3.5.
///
/// Only the first `n_rot` dimensions of each attention head are rotated;
/// the remaining `head_dim - n_rot` dimensions are passed through unchanged.
///
/// ## Performance
///
/// The cos/sin tables are precomputed at construction time (CPU, f32) for
/// positions 0..MROPE_MAX_POS.  On the first call to `apply()`, they are
/// converted to the input dtype and transferred to the input device (GPU).
/// Subsequent calls reuse the device cache — no per-token cos/sin computation,
/// no CPU→GPU transfer.
pub struct MRoPE {
    /// Number of dimension pairs per section, e.g. `[11, 11, 10, 0]`.
    sections: Vec<usize>,
    /// Total rotary dimensions (sum of sections × 2), e.g. 64 for Qwen3.5.
    n_rot: usize,
    /// Full attention head dimension, e.g. 256.
    head_dim: usize,
    /// RoPE base frequency, e.g. 10_000_000.0.
    #[allow(dead_code)]
    theta: f64,
    /// Precomputed per-pair inverse-frequency values (kept for tests/debug).
    ///
    /// `freqs[s][i]` = `rope_theta ^ (-2 * global_pair / n_rot)`
    /// where `global_pair = section_offsets[s] + i`.
    #[allow(dead_code)]
    freqs: Vec<Vec<f64>>,
    /// Per-section precomputed cos tables on CPU (f32).
    /// `cos_cpu[s]` has shape `[MROPE_MAX_POS, n_pairs_in_section]`.
    /// Sections with 0 pairs have a placeholder empty tensor.
    cos_cpu: Vec<Tensor>,
    /// Per-section precomputed sin tables on CPU (f32).
    sin_cpu: Vec<Tensor>,
    /// Lazily-initialised device cache: `(cos_dev, sin_dev)` per section,
    /// converted to the model's dtype and placed on the compute device.
    /// Initialised on first `apply()` call.
    device_cache: OnceLock<(Vec<Tensor>, Vec<Tensor>)>,
}

impl MRoPE {
    /// Build a new MRoPE module.
    ///
    /// # Arguments
    ///
    /// - `head_dim`  — full head dimension (e.g. 256)
    /// - `n_rot`     — how many of the leading dims receive RoPE (e.g. 64)
    /// - `sections`  — dimension-pair count for each positional axis (sum must
    ///                 equal `n_rot / 2`)
    /// - `theta`     — RoPE base frequency (e.g. 10_000_000.0 for Qwen3.5)
    ///
    /// # Panics
    ///
    /// Panics if `n_rot` is odd, `n_rot > head_dim`, or if the section pair
    /// sum does not equal `n_rot / 2`.
    pub fn new(head_dim: usize, n_rot: usize, sections: &[usize; 4], theta: f64) -> Self {
        assert!(n_rot % 2 == 0, "n_rot must be even, got {}", n_rot);
        assert!(
            n_rot <= head_dim,
            "n_rot ({}) must be ≤ head_dim ({})",
            n_rot,
            head_dim
        );
        let n_pairs = n_rot / 2;
        let section_sum: usize = sections.iter().sum();
        assert_eq!(
            section_sum, n_pairs,
            "sections sum ({}) must equal n_rot/2 ({})",
            section_sum, n_pairs
        );

        // Build global pair offset for each section start.
        let mut global_offset = 0usize;
        let mut freqs: Vec<Vec<f64>> = Vec::with_capacity(sections.len());

        for &n_pairs_in_section in sections.iter() {
            let section_freqs: Vec<f64> = (0..n_pairs_in_section)
                .map(|i| {
                    let g = global_offset + i;
                    // theta_g = theta ^ (-2*g / n_rot)
                    let exponent = (2 * g) as f64 / n_rot as f64;
                    theta.powf(-exponent)
                })
                .collect();
            freqs.push(section_freqs);
            global_offset += n_pairs_in_section;
        }

        // -----------------------------------------------------------------
        // Precompute cos/sin tables on CPU for positions 0..MROPE_MAX_POS.
        // For each section s with n_pairs_s pairs:
        //   cos_cpu[s] has shape [MROPE_MAX_POS, n_pairs_s]
        //   sin_cpu[s] has shape [MROPE_MAX_POS, n_pairs_s]
        // -----------------------------------------------------------------
        let max_pos = MROPE_MAX_POS;
        let mut cos_cpu: Vec<Tensor> = Vec::with_capacity(sections.len());
        let mut sin_cpu: Vec<Tensor> = Vec::with_capacity(sections.len());

        for (s, &n_pairs_s) in sections.iter().enumerate() {
            if n_pairs_s == 0 {
                // Placeholder: empty 2-D tensor so indexing stays consistent.
                let empty = Tensor::zeros((max_pos, 0), DType::F32, &Device::Cpu)
                    .expect("Failed to create empty MRoPE cache tensor");
                cos_cpu.push(empty.clone());
                sin_cpu.push(empty);
                continue;
            }

            // Build [max_pos * n_pairs_s] flat buffer, then reshape.
            let mut cos_data: Vec<f32> = Vec::with_capacity(max_pos * n_pairs_s);
            let mut sin_data: Vec<f32> = Vec::with_capacity(max_pos * n_pairs_s);

            for pos in 0..max_pos {
                let pos_f64 = pos as f64;
                for pair_i in 0..n_pairs_s {
                    let angle = pos_f64 * freqs[s][pair_i];
                    cos_data.push(angle.cos() as f32);
                    sin_data.push(angle.sin() as f32);
                }
            }

            let cos_t = Tensor::from_vec(cos_data, (max_pos, n_pairs_s), &Device::Cpu)
                .expect("Failed to build MRoPE cos cache");
            let sin_t = Tensor::from_vec(sin_data, (max_pos, n_pairs_s), &Device::Cpu)
                .expect("Failed to build MRoPE sin cache");
            cos_cpu.push(cos_t);
            sin_cpu.push(sin_t);
        }

        Self {
            sections: sections.to_vec(),
            n_rot,
            head_dim,
            theta,
            freqs,
            cos_cpu,
            sin_cpu,
            device_cache: OnceLock::new(),
        }
    }

    /// Apply MRoPE to a query or key tensor.
    ///
    /// # Arguments
    ///
    /// - `x`         — shape `[seq_len, num_heads, head_dim]`
    /// - `positions` — one `[usize; 4]` per token; for text use
    ///                 `MRoPE::text_positions(seq_len, offset)`.
    ///
    /// # Returns
    ///
    /// Tensor with the same shape as `x`. Only the first `n_rot` dims along
    /// the last axis are rotated; the remaining `head_dim - n_rot` dims are
    /// copied unchanged.
    ///
    /// # Layout
    ///
    /// Inside the rotary region the sections are laid out contiguously:
    ///
    /// ```text
    /// dims [0 .. 2*s0]               — section 0 (temporal)
    /// dims [2*s0 .. 2*(s0+s1)]       — section 1 (height)
    /// dims [2*(s0+s1) .. n_rot]      — section 2 (width)
    /// dims [n_rot .. head_dim]       — untouched pass-through
    /// ```
    pub fn apply(&self, x: &Tensor, positions: &[[usize; 4]]) -> Result<Tensor> {
        let (seq_len, num_heads, head_dim) = x.dims3()?;
        assert_eq!(
            head_dim, self.head_dim,
            "Input head_dim {} ≠ MRoPE head_dim {}",
            head_dim, self.head_dim
        );
        assert_eq!(
            positions.len(),
            seq_len,
            "positions.len() ({}) must equal seq_len ({})",
            positions.len(),
            seq_len
        );

        let device = x.device();
        let dtype = x.dtype();

        // Lazily initialise device cache on first call: transfer CPU f32
        // tables to the target dtype + device.  Subsequent calls are free.
        let (cos_dev, sin_dev) = self.device_cache.get_or_init(|| {
            let cos: Vec<Tensor> = self.cos_cpu.iter().map(|t| {
                t.to_dtype(dtype).unwrap().to_device(device).unwrap()
            }).collect();
            let sin: Vec<Tensor> = self.sin_cpu.iter().map(|t| {
                t.to_dtype(dtype).unwrap().to_device(device).unwrap()
            }).collect();
            (cos, sin)
        });

        // -----------------------------------------------------------------
        // Reconstruct the rotary region section by section, then
        // concatenate with the pass-through tail.
        // -----------------------------------------------------------------

        let mut rotated_sections: Vec<Tensor> = Vec::with_capacity(self.sections.len());
        let mut dim_offset = 0usize;

        for (s, &n_pairs) in self.sections.iter().enumerate() {
            if n_pairs == 0 {
                continue;
            }
            let section_dims = n_pairs * 2;

            // Slice the input for this section: [seq_len, num_heads, section_dims]
            let x_sec = x.narrow(2, dim_offset, section_dims)?;
            let x1 = x_sec.narrow(2, 0, n_pairs)?;
            let x2 = x_sec.narrow(2, n_pairs, n_pairs)?;

            // ---------------------------------------------------------
            // Gather cos/sin from precomputed device cache.
            //
            // Fast path: if seq_len == 1, use narrow (zero-copy view).
            // General path: build an index tensor and use index_select.
            // Both avoid any cos/sin computation and any CPU→GPU transfer
            // (the cache is already on device).
            // ---------------------------------------------------------
            let (cos_tok, sin_tok) = if seq_len == 1 {
                // Single-token decode: just a view into the cache row.
                let pos = positions[0][s];
                assert!(
                    pos < MROPE_MAX_POS,
                    "Position {} exceeds MRoPE cache size {}",
                    pos, MROPE_MAX_POS
                );
                let c = cos_dev[s].narrow(0, pos, 1)?;   // [1, n_pairs]
                let sn = sin_dev[s].narrow(0, pos, 1)?;
                (c, sn)
            } else {
                // Multi-token: detect sequential pattern to use narrow,
                // otherwise fall back to index_select.
                let first_pos = positions[0][s];
                let is_sequential = positions.iter().enumerate().all(|(i, p)| p[s] == first_pos + i);

                if is_sequential {
                    let end = first_pos + seq_len;
                    assert!(
                        end <= MROPE_MAX_POS,
                        "Position range {}..{} exceeds MRoPE cache size {}",
                        first_pos, end, MROPE_MAX_POS
                    );
                    let c = cos_dev[s].narrow(0, first_pos, seq_len)?;
                    let sn = sin_dev[s].narrow(0, first_pos, seq_len)?;
                    (c, sn)
                } else if positions.iter().all(|p| p[s] == positions[0][s]) {
                    // All tokens share the same position for this section
                    // (common for sections 1,2 in text-only mode where pos=0).
                    let pos = positions[0][s];
                    assert!(
                        pos < MROPE_MAX_POS,
                        "Position {} exceeds MRoPE cache size {}",
                        pos, MROPE_MAX_POS
                    );
                    let c = cos_dev[s].narrow(0, pos, 1)?;   // [1, n_pairs]
                    let sn = sin_dev[s].narrow(0, pos, 1)?;
                    // Will broadcast over seq_len via expand below.
                    (c, sn)
                } else {
                    // General case: arbitrary per-token positions.
                    let indices: Vec<u32> = positions.iter().map(|p| {
                        assert!(
                            p[s] < MROPE_MAX_POS,
                            "Position {} exceeds MRoPE cache size {}",
                            p[s], MROPE_MAX_POS
                        );
                        p[s] as u32
                    }).collect();
                    let idx = Tensor::from_vec(indices, (seq_len,), device)?;
                    let c = cos_dev[s].index_select(&idx, 0)?;   // [seq_len, n_pairs]
                    let sn = sin_dev[s].index_select(&idx, 0)?;
                    (c, sn)
                }
            };

            // cos_tok/sin_tok: [L, n_pairs] where L is 1 or seq_len.
            // Reshape for broadcast over num_heads: [L, 1, n_pairs]
            let cos_tok = cos_tok.unsqueeze(1)?;
            let sin_tok = sin_tok.unsqueeze(1)?;

            // Expand to [seq_len, num_heads, n_pairs] — free if already matching.
            let cos_tok = cos_tok.expand((seq_len, num_heads, n_pairs))?;
            let sin_tok = sin_tok.expand((seq_len, num_heads, n_pairs))?;

            // Rotation: [out1, out2] = [x1*cos - x2*sin, x1*sin + x2*cos]
            let out1 = (x1.mul(&cos_tok)? - x2.mul(&sin_tok)?)?;
            let out2 = (x1.mul(&sin_tok)? + x2.mul(&cos_tok)?)?;

            // Concatenate pairs back: [seq_len, num_heads, section_dims]
            let rotated = Tensor::cat(&[&out1, &out2], 2)?;
            rotated_sections.push(rotated);

            dim_offset += section_dims;
        }

        // Pass-through tail: dims [n_rot .. head_dim]
        let pass_through_dims = self.head_dim - self.n_rot;
        if pass_through_dims > 0 {
            let tail = x.narrow(2, self.n_rot, pass_through_dims)?;
            rotated_sections.push(tail);
        }

        Tensor::cat(&rotated_sections, 2)
    }

    /// Generate text-only position vectors: `[[offset, 0, 0, 0], [offset+1, 0, 0, 0], ...]`.
    ///
    /// Only section 0 (temporal) gets a non-zero position; spatial sections
    /// 1 and 2 stay at 0, which makes their rotation the identity (cos=1, sin=0).
    pub fn text_positions(seq_len: usize, offset: usize) -> Vec<[usize; 4]> {
        (0..seq_len)
            .map(|i| [offset + i, 0, 0, 0])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Verify that RoPE preserves the shape of the input tensor.
    #[test]
    fn test_rope_shape_preserved() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 16;
        let max_seq_len = 64;
        let theta = 10000.0f64;
        let batch = 2;
        let seq_len = 8;
        let num_heads = 4;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, theta, DType::F32, &device)?;

        // Random input: [batch, seq_len, num_heads, head_dim]
        let x = Tensor::randn(0f32, 1f32, (batch, seq_len, num_heads, head_dim), &device)?;
        let y = rope.apply(&x, 0)?;

        assert_eq!(
            x.shape(),
            y.shape(),
            "RoPE must preserve tensor shape: input={:?}, output={:?}",
            x.shape(),
            y.shape()
        );

        println!(
            "Shape preserved: {:?} -> {:?}",
            x.shape(),
            y.shape()
        );
        Ok(())
    }

    /// Verify that the precomputed frequencies are correct.
    ///
    /// - freq[0] corresponds to i=0: 1 / theta^(0/head_dim) = 1 / theta^0 = 1.0
    ///   so cos[1, 0] = cos(1.0 * 1.0) = cos(1.0)
    /// - freq[half_dim-1] corresponds to the last pair: 1 / theta^((head_dim-2)/head_dim)
    ///
    /// We verify by extracting the cache values and checking against analytical values.
    #[test]
    fn test_rope_frequencies_correct() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;   // half_dim = 4
        let max_seq_len = 16;
        let theta = 10000.0f64;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, theta, DType::F32, &device)?;

        let half_dim = head_dim / 2; // 4

        // freq[0] = 1 / theta^(0 / head_dim) = 1 / theta^0 = 1.0
        // At position t=1: angle = 1.0 * 1.0 = 1.0 -> cos(1.0)
        let freq_0 = 1.0f64 / theta.powf(0.0 / head_dim as f64); // = 1.0
        let expected_cos_pos1_dim0 = freq_0.cos() as f32;

        // Extract cos_cache[1, 0] (position=1, dim_index=0)
        let actual_cos = rope
            .cos_cache
            .narrow(0, 1, 1)?  // [1, half_dim]
            .narrow(1, 0, 1)?  // [1, 1]
            .squeeze(0)?       // [1]
            .squeeze(0)?       // scalar []
            .to_scalar::<f32>()?;

        assert!(
            (actual_cos - expected_cos_pos1_dim0).abs() < 1e-5,
            "freq[0] mismatch: expected cos(1.0)={:.6}, got {:.6}",
            expected_cos_pos1_dim0,
            actual_cos
        );

        // freq[half_dim-1]: last frequency pair
        // i = half_dim - 1, exponent = 2*(half_dim-1) / head_dim = (head_dim-2)/head_dim
        let last_i = half_dim - 1;
        let last_exponent = (2 * last_i) as f64 / head_dim as f64;
        let freq_last = 1.0f64 / theta.powf(last_exponent);
        // At position t=1: angle = 1.0 * freq_last -> cos(freq_last)
        let expected_cos_pos1_last = freq_last.cos() as f32;

        let actual_cos_last = rope
            .cos_cache
            .narrow(0, 1, 1)?        // [1, half_dim]
            .narrow(1, last_i, 1)?   // [1, 1]
            .squeeze(0)?             // [1]
            .squeeze(0)?             // scalar []
            .to_scalar::<f32>()?;

        assert!(
            (actual_cos_last - expected_cos_pos1_last).abs() < 1e-5,
            "freq[last] mismatch: expected {:.6}, got {:.6}",
            expected_cos_pos1_last,
            actual_cos_last
        );

        // Cross-check: at position 0, all cos values should be 1.0 (cos(0) = 1)
        // and all sin values should be 0.0 (sin(0) = 0)
        let cos_pos0: Vec<f32> = rope.cos_cache.narrow(0, 0, 1)?.squeeze(0)?.to_vec1()?;
        let sin_pos0: Vec<f32> = rope.sin_cache.narrow(0, 0, 1)?.squeeze(0)?.to_vec1()?;

        for (i, &c) in cos_pos0.iter().enumerate() {
            assert!(
                (c - 1.0f32).abs() < 1e-6,
                "cos at position 0, dim {i} should be 1.0, got {c}"
            );
        }
        for (i, &s) in sin_pos0.iter().enumerate() {
            assert!(
                s.abs() < 1e-6,
                "sin at position 0, dim {i} should be 0.0, got {s}"
            );
        }

        println!(
            "Frequency check passed: freq[0]={:.6}, freq[last]={:.6}",
            freq_0, freq_last
        );
        Ok(())
    }

    /// Verify that applying RoPE with different offsets produces different results.
    ///
    /// For a non-zero input, offset=5 encodes positions 5..5+seq_len while
    /// offset=0 encodes positions 0..seq_len. These must differ (unless the
    /// input is zero everywhere, which is statistically impossible for random).
    #[test]
    fn test_rope_offset() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 16;
        let max_seq_len = 128;
        let theta = 10000.0f64;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let offset = 5usize;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, theta, DType::F32, &device)?;

        // Use a non-trivial constant input (all ones) so we can see positional effects
        let x = Tensor::ones((batch, seq_len, num_heads, head_dim), DType::F32, &device)?;

        let y_offset0 = rope.apply(&x, 0)?;
        let y_offset5 = rope.apply(&x, offset)?;

        // Compute mean absolute difference between the two outputs
        let diff = (&y_offset5 - &y_offset0)?.abs()?.mean_all()?.to_scalar::<f32>()?;

        assert!(
            diff > 1e-4,
            "offset=5 and offset=0 must produce different outputs for non-trivial input, \
             but mean abs diff = {:.8}",
            diff
        );

        // Additional check: at position 0 with offset=0, cos=1 and sin=0 for all dims,
        // so the first token of y_offset0 should equal the first token of x.
        // (RoPE at position 0 is the identity.)
        let x_tok0 = x.narrow(1, 0, 1)?;       // [1, 1, num_heads, head_dim]
        let y_tok0 = y_offset0.narrow(1, 0, 1)?;
        let identity_diff = (&y_tok0 - &x_tok0)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;

        assert!(
            identity_diff < 1e-5,
            "RoPE at position 0 must be identity, but max diff = {:.8}",
            identity_diff
        );

        println!(
            "Offset test passed: mean |y[offset=5] - y[offset=0]| = {:.6}",
            diff
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // MRoPE tests
    // -----------------------------------------------------------------------

    /// Qwen3.5 MRoPE parameters used across multiple tests.
    fn qwen35_mrope() -> MRoPE {
        MRoPE::new(
            256,                    // head_dim
            64,                     // n_rot
            &[11, 11, 10, 0],       // sections (pairs per axis)
            10_000_000.0,           // theta
        )
    }

    /// The sections must sum to n_rot / 2 = 32 pairs.
    #[test]
    fn test_mrope_new() {
        let mrope = qwen35_mrope();
        let pair_sum: usize = mrope.sections.iter().sum();
        assert_eq!(pair_sum, mrope.n_rot / 2,
            "sections sum ({pair_sum}) must equal n_rot/2 ({})", mrope.n_rot / 2);
        assert_eq!(pair_sum, 32, "Qwen3.5: 11+11+10+0 = 32 pairs");

        // Verify freqs table shape.
        assert_eq!(mrope.freqs.len(), 4, "freqs must have one entry per section");
        assert_eq!(mrope.freqs[0].len(), 11);
        assert_eq!(mrope.freqs[1].len(), 11);
        assert_eq!(mrope.freqs[2].len(), 10);
        assert_eq!(mrope.freqs[3].len(),  0);

        println!("test_mrope_new passed: pair_sum={pair_sum}");
    }

    /// `text_positions(5, 10)` must return `[[10,0,0,0], [11,0,0,0], ..., [14,0,0,0]]`.
    #[test]
    fn test_mrope_text_positions() {
        let pos = MRoPE::text_positions(5, 10);
        assert_eq!(pos.len(), 5);
        for (i, p) in pos.iter().enumerate() {
            assert_eq!(p[0], 10 + i, "token {i}: temporal position mismatch");
            assert_eq!(p[1], 0, "token {i}: height must be 0 for text");
            assert_eq!(p[2], 0, "token {i}: width must be 0 for text");
            assert_eq!(p[3], 0, "token {i}: unused axis must be 0");
        }
        println!("test_mrope_text_positions passed: {:?}", pos);
    }

    /// MRoPE must preserve the shape `[seq_len, num_heads, head_dim]`.
    #[test]
    fn test_mrope_shape_preserved() -> Result<()> {
        let device = Device::Cpu;
        let mrope = qwen35_mrope();

        let seq_len = 7;
        let num_heads = 8;
        let head_dim = 256;

        let x = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &device)?;
        let positions = MRoPE::text_positions(seq_len, 0);
        let y = mrope.apply(&x, &positions)?;

        assert_eq!(x.shape(), y.shape(),
            "Shape must be preserved: {:?} → {:?}", x.shape(), y.shape());
        println!("test_mrope_shape_preserved passed: {:?}", y.shape());
        Ok(())
    }

    /// At position `[0,0,0,0]` all angles are 0 → cos=1, sin=0 → rotation is
    /// the identity, so the rotary dims must be unchanged.
    #[test]
    fn test_mrope_position_zero() -> Result<()> {
        let device = Device::Cpu;
        let mrope = qwen35_mrope();

        let seq_len = 3;
        let num_heads = 4;
        let head_dim = 256;

        let x = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &device)?;

        // All tokens at position 0.
        let positions = vec![[0usize; 4]; seq_len];
        let y = mrope.apply(&x, &positions)?;

        // The rotary region (first n_rot=64 dims) must be unchanged.
        let x_rot = x.narrow(2, 0, mrope.n_rot)?;
        let y_rot = y.narrow(2, 0, mrope.n_rot)?;
        let max_diff = (&y_rot - &x_rot)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;

        assert!(
            max_diff < 1e-5,
            "Position [0,0,0,0] must be identity on rotary dims; max diff = {max_diff:.2e}"
        );
        println!("test_mrope_position_zero passed: max_diff={max_diff:.2e}");
        Ok(())
    }

    /// For text tokens, only section 0 (temporal) has a non-zero position.
    /// Sections 1 and 2 must therefore apply identity (position 0 → cos=1, sin=0).
    ///
    /// Concretely: the dims for sections 1 and 2 (indices 22..64) must equal
    /// the corresponding input dims, while section 0 dims (0..22) differ from
    /// a non-trivial input at non-zero position.
    #[test]
    fn test_mrope_text_only_section0() -> Result<()> {
        let device = Device::Cpu;
        let mrope = qwen35_mrope();
        // sections: [11, 10, 10, 0] pairs → in dims: s0=[0..22], s1=[22..44], s2=[44..64]

        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 256;

        let x = Tensor::ones((seq_len, num_heads, head_dim), DType::F32, &device)?;

        // Text positions starting at token 5 (non-zero to avoid accidental identity).
        let positions = MRoPE::text_positions(seq_len, 5);
        let y = mrope.apply(&x, &positions)?;

        // ----- Section 1 dims (22..44): position=0 → identity -----
        let s1_start = mrope.sections[0] * 2; // 22
        let s1_len   = mrope.sections[1] * 2; // 22
        let x_s1 = x.narrow(2, s1_start, s1_len)?;
        let y_s1 = y.narrow(2, s1_start, s1_len)?;
        let s1_diff = (&y_s1 - &x_s1)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            s1_diff < 1e-5,
            "Section 1 (height, pos=0) must be identity; max diff = {s1_diff:.2e}"
        );

        // ----- Section 2 dims (44..64): position=0 → identity -----
        let s2_start = s1_start + s1_len; // 44
        let s2_len   = mrope.sections[2] * 2; // 20
        let x_s2 = x.narrow(2, s2_start, s2_len)?;
        let y_s2 = y.narrow(2, s2_start, s2_len)?;
        let s2_diff = (&y_s2 - &x_s2)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            s2_diff < 1e-5,
            "Section 2 (width, pos=0) must be identity; max diff = {s2_diff:.2e}"
        );

        // ----- Section 0 dims (0..22): position=5..9 → must be rotated -----
        let s0_len = mrope.sections[0] * 2; // 22
        let x_s0 = x.narrow(2, 0, s0_len)?;
        let y_s0 = y.narrow(2, 0, s0_len)?;
        let s0_diff = (&y_s0 - &x_s0)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            s0_diff > 1e-4,
            "Section 0 (temporal, pos≥5) must be rotated; max diff = {s0_diff:.6}"
        );

        println!(
            "test_mrope_text_only_section0 passed: s0_diff={s0_diff:.4}, \
             s1_diff={s1_diff:.2e}, s2_diff={s2_diff:.2e}"
        );
        Ok(())
    }

    /// Two calls with different position arrays must produce different outputs.
    #[test]
    fn test_mrope_different_positions() -> Result<()> {
        let device = Device::Cpu;
        let mrope = qwen35_mrope();

        let seq_len = 6;
        let num_heads = 4;
        let head_dim = 256;

        let x = Tensor::randn(0f32, 1f32, (seq_len, num_heads, head_dim), &device)?;

        let pos_a = MRoPE::text_positions(seq_len, 0);
        let pos_b = MRoPE::text_positions(seq_len, 100);

        let ya = mrope.apply(&x, &pos_a)?;
        let yb = mrope.apply(&x, &pos_b)?;

        let mean_diff = (&yb - &ya)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(
            mean_diff > 1e-4,
            "Different positions must produce different outputs; mean diff = {mean_diff:.2e}"
        );
        println!("test_mrope_different_positions passed: mean_diff={mean_diff:.4}");
        Ok(())
    }
}
