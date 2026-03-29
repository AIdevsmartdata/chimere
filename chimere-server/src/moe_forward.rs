//! MoE (Mixture of Experts) forward pass for Qwen3.5-35B-A3B.
//!
//! Routes each token to the top-K experts via a learned gate,
//! computes SwiGLU FFN for each selected expert, and combines
//! outputs with the always-active shared expert.
//!
//! ## Architecture
//!
//! - 256 routed experts per layer, top-8 routing
//! - Each expert: SwiGLU FFN with hidden=512, input/output=2048
//! - 1 shared expert (always active): same architecture
//! - Router: linear projection [2048] → [256] logits → softmax → top-8
//!
//! ## Tensor shapes (GGUF naming)
//!
//! ```text
//! ffn_gate_inp.weight        [2048, 256]       — router gate (F32)
//! ffn_gate_inp_shexp.weight  [2048]            — shared expert scale (F32)
//! ffn_gate_exps.weight       [2048, 512, 256]  — 256 expert gate projections stacked
//! ffn_up_exps.weight         [2048, 512, 256]  — 256 expert up projections stacked
//! ffn_down_exps.weight       [512, 2048, 256]  — 256 expert down projections stacked
//! ffn_gate_shexp.weight      [2048, 512]       — shared expert gate
//! ffn_up_shexp.weight        [2048, 512]       — shared expert up
//! ffn_down_shexp.weight      [512, 2048]       — shared expert down
//! ```
//!
//! ## v0 tradeoffs
//!
//! - Per-expert slicing via `narrow(2, i, 1).squeeze(2)` creates views (no copy),
//!   but runs a separate matmul per expert. Future: batch all 8 expert matmuls.
//! - Top-K selection moves data to CPU (`.to_vec1()`). This is a ~0.1 ms
//!   GPU→CPU sync point. Future: CUDA top-K kernel.
//! - Correctness first; optimisation later.

use candle_core::{DType, Result, Tensor, D};
use candle_core::quantized::QMatMul;
use candle_core::Module;
use candle_nn::ops::softmax;

// ---------------------------------------------------------------------------
// MoeWeights
// ---------------------------------------------------------------------------

/// Weights for one MoE layer.
///
/// Routed-expert tensors are stored as regular `Tensor`s because their 3D
/// shape (`[hidden, expert_ffn, num_experts]`) is not supported by `QMatMul`.
/// They are dequantized to F32 at load time and sliced per expert at runtime.
///
/// Shared-expert weights use `QMatMul` so they can stay quantized on GPU
/// (Q8_0, Q4K, etc.) and benefit from on-the-fly GPU dequant during matmul.
pub struct MoeWeights {
    /// Router gate: [hidden_size, num_experts] — F32.
    pub router: Tensor,
    /// Shared expert scale/bias applied to shared expert output: [hidden_size] — F32.
    pub shared_gate: Tensor,
    /// Stacked gate projections for all routed experts.
    /// Shape: [hidden_size, expert_ffn, num_experts].
    pub gate_exps: Tensor,
    /// Stacked up-projections for all routed experts.
    /// Shape: [hidden_size, expert_ffn, num_experts].
    pub up_exps: Tensor,
    /// Stacked down-projections for all routed experts.
    /// Shape: [expert_ffn, hidden_size, num_experts].
    pub down_exps: Tensor,
    /// Shared expert gate projection: [hidden_size, expert_ffn] (quantized on GPU).
    pub gate_shexp: QMatMul,
    /// Shared expert up-projection: [hidden_size, expert_ffn] (quantized on GPU).
    pub up_shexp: QMatMul,
    /// Shared expert down-projection: [expert_ffn, hidden_size] (quantized on GPU).
    pub down_shexp: QMatMul,
}

// ---------------------------------------------------------------------------
// Top-K helpers
// ---------------------------------------------------------------------------

/// Select the top-`k` entries from `probs` by value.
///
/// Returns `(indices, renormalised_weights)`.  Weights are renormalised so
/// they sum to 1.0, implementing the standard top-K probability rescaling
/// used by Qwen3.5 and DeepSeek-V2.
pub fn topk_probs(probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(probs.len());
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    // Sort descending by probability.
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    let weight_sum: f32 = top.iter().map(|(_, w)| w).sum();
    let scale = if weight_sum > 1e-12 {
        1.0 / weight_sum
    } else {
        1.0 / k as f32
    };

    let ids: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top.iter().map(|(_, w)| w * scale).collect();

    (ids, weights)
}

// ---------------------------------------------------------------------------
// SwiGLU helper (operates on plain Tensors — used for routed expert slices)
// ---------------------------------------------------------------------------

/// SwiGLU: `down( silu(gate(x)) * up(x) )`
///
/// All weight matrices are 2D and already extracted for a single expert.
///
/// # Arguments
/// - `x`: [1, hidden_size]
/// - `gate_w`: [hidden_size, expert_ffn] — gate projection weight
/// - `up_w`: [hidden_size, expert_ffn] — up projection weight
/// - `down_w`: [expert_ffn, hidden_size] — down projection weight
///
/// Weight layout follows GGUF convention:
/// - `gate_exps / up_exps` are stored as `[hidden, expert_ffn, num_experts]`,
///   so a sliced expert gives `[hidden, expert_ffn]` → matmul is `x @ gate_w`.
/// - `down_exps` is stored as `[expert_ffn, hidden, num_experts]`,
///   so a sliced expert gives `[expert_ffn, hidden]` → matmul is `intermediate @ down_w`.
///
/// # Returns
/// [1, hidden_size]
fn swiglu_expert(
    x: &Tensor,
    gate_w: &Tensor,
    up_w: &Tensor,
    down_w: &Tensor,
) -> Result<Tensor> {
    // [1, hidden] @ [hidden, expert_ffn] → [1, expert_ffn]
    let gate_out = x.matmul(gate_w)?;
    let up_out = x.matmul(up_w)?;

    // SiLU via Tensor::silu() — fused CUDA kernel when on GPU.
    let activated = gate_out.silu()?;
    let intermediate = (activated * up_out)?;

    // down_w is [expert_ffn, hidden]: [1, expert_ffn] @ [expert_ffn, hidden] → [1, hidden]
    intermediate.matmul(down_w)
}

// ---------------------------------------------------------------------------
// MoE forward pass
// ---------------------------------------------------------------------------

/// MoE forward pass for a single token.
///
/// # Arguments
/// - `hidden`: [1, hidden_size] — input hidden state (must be F32 on device).
/// - `weights`: MoE weights for this layer.
/// - `top_k`: number of routed experts to activate (8 for Qwen3.5-35B-A3B).
///
/// # Returns
/// `[1, hidden_size]` — weighted sum of routed expert outputs plus shared expert output.
///
/// # Algorithm
///
/// ```text
/// logits      = hidden @ router          [1, num_experts]
/// probs       = softmax(logits, dim=-1)
/// (ids, w)    = top_k(probs)
/// routed_out  = Σᵢ  wᵢ · swiglu(hidden, gate_exps[:,:,i], up_exps[:,:,i], down_exps[:,:,i])
/// shared_out  = swiglu_qmatmul(hidden, gate_shexp, up_shexp, down_shexp)
/// scale       = sigmoid(shared_gate)     [1, hidden_size] — per-dim blending
/// output      = routed_out + scale ⊙ shared_out
/// ```
pub fn moe_forward(hidden: &Tensor, weights: &MoeWeights, top_k: usize) -> Result<Tensor> {
    let hidden_size = hidden.dim(1)?;
    let device = hidden.device();

    // ------------------------------------------------------------------
    // 1. Router: compute expert scores
    //    router: [hidden_size, num_experts]  (stored as weight matrix)
    //    hidden @ router → [1, num_experts]
    // ------------------------------------------------------------------
    let router_logits = hidden.matmul(&weights.router)?; // [1, num_experts]

    // ------------------------------------------------------------------
    // 2. Softmax + top-K selection (CPU sync point — ~0.1 ms)
    // ------------------------------------------------------------------
    let router_probs = softmax(&router_logits, D::Minus1)?; // [1, num_experts]
    let probs_vec: Vec<f32> = router_probs.squeeze(0)?.to_vec1()?;

    let (expert_ids, expert_weights) = topk_probs(&probs_vec, top_k);

    // ------------------------------------------------------------------
    // 3. Routed expert outputs: weighted sum
    // ------------------------------------------------------------------
    let mut combined = Tensor::zeros((1, hidden_size), DType::F32, device)?;

    for (&expert_idx, &weight) in expert_ids.iter().zip(expert_weights.iter()) {
        // Slice weights for expert `expert_idx` from the 3D stacked tensors.
        //
        // gate_exps: [hidden_size, expert_ffn, num_experts]
        //   → narrow(2, i, 1) → [hidden_size, expert_ffn, 1]
        //   → squeeze(2)      → [hidden_size, expert_ffn]
        let gate_w = weights.gate_exps.narrow(2, expert_idx, 1)?.squeeze(2)?;
        let up_w = weights.up_exps.narrow(2, expert_idx, 1)?.squeeze(2)?;
        // down_exps: [expert_ffn, hidden_size, num_experts]
        //   → slice → [expert_ffn, hidden_size]
        let down_w = weights.down_exps.narrow(2, expert_idx, 1)?.squeeze(2)?;

        let expert_out = swiglu_expert(hidden, &gate_w, &up_w, &down_w)?; // [1, hidden_size]

        // Accumulate: combined += weight * expert_out
        combined = (combined + expert_out.affine(weight as f64, 0.0)?)?;
    }

    // ------------------------------------------------------------------
    // 4. Shared expert (always active, uses QMatMul for GPU quantised matmul)
    // ------------------------------------------------------------------
    // QMatMul::forward stores weight as W^T internally; convention is the same
    // as candle_nn::Linear::forward — input @ W^T.
    let sh_gate = weights.gate_shexp.forward(hidden)?; // [1, expert_ffn]
    let sh_up = weights.up_shexp.forward(hidden)?;     // [1, expert_ffn]
    let sh_activated = sh_gate.silu()?;
    let sh_intermediate = (sh_activated * sh_up)?;
    let shared_out = weights.down_shexp.forward(&sh_intermediate)?; // [1, hidden_size]

    // ------------------------------------------------------------------
    // 5. Per-dim sigmoid gate controls how much of the shared expert to blend.
    //    shared_gate: [hidden_size] → unsqueeze → [1, hidden_size] → sigmoid
    // ------------------------------------------------------------------
    let shared_scale = candle_nn::ops::sigmoid(&weights.shared_gate.unsqueeze(0)?)?; // [1, hidden_size]
    let gated_shared = (shared_out * shared_scale)?;

    // ------------------------------------------------------------------
    // 6. Combine routed + gated shared expert
    // ------------------------------------------------------------------
    let output = (&combined + &gated_shared)?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    // Small dimensions for fast CPU tests.
    const HIDDEN: usize = 16;
    const EXPERT_FFN: usize = 8;
    const NUM_EXPERTS: usize = 4;
    const TOP_K: usize = 2;

    /// Build a MoeWeights with random F32 weights and QMatMul::Tensor shared experts.
    fn make_weights(device: &Device) -> Result<MoeWeights> {
        // Router: [hidden, num_experts]
        let router = Tensor::randn(0.0f32, 0.02, (HIDDEN, NUM_EXPERTS), device)?;

        // Shared expert scale: [hidden]
        let shared_gate = Tensor::randn(0.0f32, 0.02, HIDDEN, device)?;

        // Stacked routed experts.
        let gate_exps =
            Tensor::randn(0.0f32, 0.02, (HIDDEN, EXPERT_FFN, NUM_EXPERTS), device)?;
        let up_exps =
            Tensor::randn(0.0f32, 0.02, (HIDDEN, EXPERT_FFN, NUM_EXPERTS), device)?;
        let down_exps =
            Tensor::randn(0.0f32, 0.02, (EXPERT_FFN, HIDDEN, NUM_EXPERTS), device)?;

        // Shared expert weights as QMatMul::Tensor (F32 path — GPU native for Q8_0 etc).
        // QMatMul::Tensor wraps a plain Tensor; QMatMul::forward transposes internally.
        // Weight shapes follow QMatMul convention: [out_features, in_features].
        let gate_shexp =
            QMatMul::Tensor(Tensor::randn(0.0f32, 0.02, (EXPERT_FFN, HIDDEN), device)?);
        let up_shexp =
            QMatMul::Tensor(Tensor::randn(0.0f32, 0.02, (EXPERT_FFN, HIDDEN), device)?);
        // down_shexp maps expert_ffn → hidden: [hidden, expert_ffn] in QMatMul convention.
        let down_shexp =
            QMatMul::Tensor(Tensor::randn(0.0f32, 0.02, (HIDDEN, EXPERT_FFN), device)?);

        Ok(MoeWeights {
            router,
            shared_gate,
            gate_exps,
            up_exps,
            down_exps,
            gate_shexp,
            up_shexp,
            down_shexp,
        })
    }

    // -----------------------------------------------------------------------
    // 1. Output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_forward_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let weights = make_weights(&device)?;

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN), &device)?;
        let out = moe_forward(&hidden, &weights, TOP_K)?;

        assert_eq!(
            out.dims(),
            &[1, HIDDEN],
            "moe_forward output shape should be [1, {HIDDEN}], got {:?}",
            out.dims()
        );

        println!("test_moe_forward_output_shape: {:?}", out.dims());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 2. Top-K selection picks the correct expert indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_topk_probs_selects_highest() {
        // Known probability distribution — experts 1 and 3 dominate.
        let probs = vec![0.05f32, 0.60, 0.10, 0.25];
        let (ids, weights) = topk_probs(&probs, 2);

        assert_eq!(ids, vec![1, 3], "top-2 should be experts 1 and 3, got {:?}", ids);

        // Weights must sum to 1.0.
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalised weights must sum to 1.0, got {:.6}",
            sum
        );

        // Relative proportions preserved: w[0]/w[1] ≈ 0.60/0.25 = 2.4
        let ratio = weights[0] / weights[1];
        let expected_ratio = 0.60 / 0.25;
        assert!(
            (ratio - expected_ratio).abs() < 1e-4,
            "weight ratio should be {:.4}, got {:.4}",
            expected_ratio,
            ratio
        );

        println!("test_topk_probs: ids={:?}, weights={:?}", ids, weights);
    }

    #[test]
    fn test_topk_probs_k1() {
        let probs = vec![0.1f32, 0.8, 0.05, 0.05];
        let (ids, weights) = topk_probs(&probs, 1);

        assert_eq!(ids, vec![1], "top-1 should be expert 1");
        assert!(
            (weights[0] - 1.0).abs() < 1e-5,
            "single expert weight should be 1.0, got {:.6}",
            weights[0]
        );
    }

    #[test]
    fn test_topk_probs_all_equal() {
        // Uniform — any k experts, all equal weight.
        let probs = vec![0.25f32; 4];
        let (ids, weights) = topk_probs(&probs, 2);

        assert_eq!(ids.len(), 2);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "uniform top-2 weights must sum to 1.0, got {:.6}",
            sum
        );
        // Both weights should be 0.5.
        for w in &weights {
            assert!(
                (w - 0.5).abs() < 1e-5,
                "uniform weights should each be 0.5, got {:.6}",
                w
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Weight renormalisation sums to 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_topk_renorm_sums_to_one() {
        // Asymmetric distribution.
        let probs = vec![0.01f32, 0.50, 0.30, 0.07, 0.12];
        for k in 1..=5 {
            let (_, weights) = topk_probs(&probs, k);
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "k={k}: renormalised weights sum = {:.6}, expected 1.0",
                sum
            );
        }
        println!("test_topk_renorm_sums_to_one: all k=1..5 passed");
    }

    // -----------------------------------------------------------------------
    // 4. Zero input → finite output (no NaN / Inf)
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_forward_zero_input_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let weights = make_weights(&device)?;

        // All-zeros hidden state — gate logits will be zero → uniform softmax.
        let hidden = Tensor::zeros((1, HIDDEN), DType::F32, &device)?;
        let out = moe_forward(&hidden, &weights, TOP_K)?;

        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for (i, v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "output[{i}] = {v} is not finite (NaN or Inf)"
            );
        }

        println!("test_moe_forward_zero_input_is_finite: max_abs={:.4e}",
            vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 5. top_k=1 and top_k=num_experts both work correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_forward_extreme_k() -> Result<()> {
        let device = Device::Cpu;
        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN), &device)?;

        // k=1: only the best expert.
        let weights1 = make_weights(&device)?;
        let out1 = moe_forward(&hidden, &weights1, 1)?;
        assert_eq!(out1.dims(), &[1, HIDDEN], "k=1: wrong shape");

        // k=NUM_EXPERTS: all experts active.
        let weights_all = make_weights(&device)?;
        let out_all = moe_forward(&hidden, &weights_all, NUM_EXPERTS)?;
        assert_eq!(out_all.dims(), &[1, HIDDEN], "k=all: wrong shape");

        println!("test_moe_forward_extreme_k: k=1 and k={NUM_EXPERTS} both produce [1, {HIDDEN}]");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 6. Different hidden states produce different outputs (not a no-op)
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_forward_not_constant() -> Result<()> {
        let device = Device::Cpu;
        let weights = make_weights(&device)?;

        let h1 = Tensor::randn(0.0f32, 1.0, (1, HIDDEN), &device)?;
        let h2 = Tensor::randn(0.0f32, 1.0, (1, HIDDEN), &device)?;

        let o1 = moe_forward(&h1, &weights, TOP_K)?;
        let o2 = moe_forward(&h2, &weights, TOP_K)?;

        // Outputs should differ (with overwhelmingly high probability for random inputs).
        let diff: f32 = (&o1 - &o2)?.abs()?.sum_all()?.to_scalar()?;
        assert!(
            diff > 1e-6,
            "Two different hidden states should produce different outputs, diff={:.2e}",
            diff
        );

        println!("test_moe_forward_not_constant: L1 diff = {:.4}", diff);
        Ok(())
    }
}
