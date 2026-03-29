//! # Expert FFN (SwiGLU + MoE) — Chimère Engine
//!
//! This module provides the feed-forward expert layers used inside the
//! Mixture of Experts blocks of the Chimère architecture.
//!
//! ## SwiGLU Experts
//!
//! Each expert is a two-gate FFN:
//!
//! ```text
//! out = down_proj( silu(gate_proj(x)) * up_proj(x) )
//! ```
//!
//! SiLU (Swish-1): `silu(x) = x * sigmoid(x)`
//!
//! This is the standard FFN used in LLaMA, Qwen, Mistral, etc.
//!
//! ## MoeBlock
//!
//! The `MoeBlock` wraps a set of `SwiGluExpert` instances under a common
//! `MoeRouter`. It applies pre-norm (RMSNorm), routes each token to the
//! appropriate expert(s), and accumulates weighted outputs. The shared
//! expert (always active) is added on top.
//!
//! Residual connections are NOT applied here — the caller (model layer)
//! is responsible for the skip connection, following the pre-norm
//! convention used in GatedDeltaNetLayer.
//!
//! ## Integration
//!
//! ```text
//! token → RmsNormLight → MoeRouter → weighted expert sum → output
//!                                 ↘ shared expert ↗
//! ```

use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::moe_router::{MoeRouter, RoutingDecision};

// ---------------------------------------------------------------------------
// SiLU / Sigmoid helpers (local, keep expert.rs self-contained)
// ---------------------------------------------------------------------------

/// Sigmoid activation: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let one = Tensor::ones(x.shape(), x.dtype(), x.device())?;
    let denom = (&one + &exp_neg)?;
    one.div(&denom)
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = sigmoid(x)?;
    x.mul(&sig)
}

// ---------------------------------------------------------------------------
// SwiGluExpert
// ---------------------------------------------------------------------------

/// A single SwiGLU feed-forward expert.
///
/// Computes: `down_proj( silu(gate_proj(x)) * up_proj(x) )`
///
/// All projections are bias-free, following the convention used in
/// Qwen3, LLaMA, and the Chimère reference architecture.
pub struct SwiGluExpert {
    /// [intermediate_size, hidden_size]
    gate_proj: Linear,
    /// [intermediate_size, hidden_size]
    up_proj: Linear,
    /// [hidden_size, intermediate_size]
    down_proj: Linear,
}

impl SwiGluExpert {
    /// Create a new SwiGLU expert.
    ///
    /// # Arguments
    /// - `hidden_size`: Input and output dimension.
    /// - `intermediate_size`: Inner (expanded) dimension.
    /// - `vb`: VarBuilder scoped to this expert's prefix.
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass.
    ///
    /// # Input
    /// `x`: any shape ending in `hidden_size`, e.g. `[batch, seq, hidden]`
    ///
    /// # Output
    /// Same shape as input.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?; // [..., intermediate_size]
        let up = self.up_proj.forward(x)?; // [..., intermediate_size]
        let activated = (silu(&gate)? * up)?; // SwiGLU gating
        self.down_proj.forward(&activated) // [..., hidden_size]
    }
}

// ---------------------------------------------------------------------------
// RmsNormLight
// ---------------------------------------------------------------------------

/// Lightweight RMSNorm without a `VarBuilder` dependency.
///
/// Used for pre-norm inside `MoeBlock`. The weight tensor is passed
/// directly so callers can manage it however they like (e.g. from
/// a `VarBuilder` or hand-crafted ones-tensor for tests).
///
/// Formula:
/// ```text
/// rms   = sqrt( mean(x^2, dim=-1) + eps )
/// x_out = (x / rms) * weight
/// ```
pub struct RmsNormLight {
    weight: Tensor,
    eps: f64,
}

impl RmsNormLight {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Apply RMSNorm over the last dimension.
    ///
    /// Works for tensors of any rank `>= 1`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rank = x.rank();
        let last_dim = rank - 1;
        let hidden_size = x.dim(last_dim)? as f64;

        // mean(x^2, dim=-1, keepdim=True)
        // candle lacks mean_keepdim, so: sum_keepdim / n
        let sum_sq = x.sqr()?.sum_keepdim(last_dim)?;
        let mean_sq = (sum_sq / hidden_size)?;

        // rms = sqrt(mean_sq + eps)
        // affine(1.0, eps) adds eps elementwise: rms_sq + eps
        let rms = mean_sq.affine(1.0, self.eps)?.sqrt()?;

        // x_norm = x / rms  (broadcast over last dim)
        let x_norm = x.broadcast_div(&rms)?;

        // scale by learned weight
        x_norm.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// MoeBlock
// ---------------------------------------------------------------------------

/// Mixture-of-Experts block with optional shared expert and entropy-adaptive
/// routing.
///
/// # Pre-norm architecture
///
/// `MoeBlock` applies `RmsNormLight` to the input before dispatching to
/// experts.  The residual (`x + output`) is handled by the caller, consistent
/// with `GatedDeltaNetLayer`.
///
/// # Expert dispatch
///
/// When routed experts exist, tokens are processed one at a time (batch=1
/// assumed for now).  Each token receives a `RoutingDecision` from `MoeRouter`
/// and its output is the weighted sum of the selected experts' outputs.
/// The shared expert, if present, is always added unconditionally.
///
/// When `routed_experts` is empty, the router is **never called** (avoids
/// softmax over an empty tensor).  Only the shared expert is run.
pub struct MoeBlock {
    /// Always-active expert (optional). Added to routed output when present.
    shared_expert: Option<SwiGluExpert>,
    /// Pool of selectable experts. May be empty.
    routed_experts: Vec<SwiGluExpert>,
    /// Router that selects which routed experts to activate per token.
    /// `None` when `routed_experts` is empty.
    router: Option<MoeRouter>,
    /// Pre-norm applied to the input before FFN.
    norm: RmsNormLight,
}

impl MoeBlock {
    /// Construct a `MoeBlock`.
    ///
    /// # Arguments
    /// - `shared_expert`: Optional always-active expert.
    /// - `routed_experts`: Pool of selectable experts (may be empty).
    /// - `router`: Router instance. Should be `None` iff `routed_experts` is empty.
    /// - `norm_weight`: RMSNorm scale parameter tensor `[hidden_size]`.
    /// - `eps`: RMSNorm epsilon.
    pub fn new(
        shared_expert: Option<SwiGluExpert>,
        routed_experts: Vec<SwiGluExpert>,
        router: Option<MoeRouter>,
        norm_weight: Tensor,
        eps: f64,
    ) -> Self {
        Self {
            shared_expert,
            routed_experts,
            router,
            norm: RmsNormLight::new(norm_weight, eps),
        }
    }

    /// Forward pass over a sequence.
    ///
    /// # Arguments
    /// - `x`: Input tensor `[batch, seq_len, hidden_size]`.
    /// - `state_metrics`: Optional DeltaNet state metrics used by the router
    ///   to modulate adaptive-K selection.
    ///
    /// # Returns
    /// Output tensor with the same shape as `x`.
    /// **No residual** is applied — callers are responsible for `x + output`.
    pub fn forward(
        &self,
        x: &Tensor,
        state_metrics: Option<&crate::StateMetrics>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = x.dims3()?;

        // Pre-norm
        let x_norm = self.norm.forward(x)?;

        // --- Fast path: no routed experts ---
        if self.routed_experts.is_empty() {
            return match &self.shared_expert {
                Some(expert) => expert.forward(&x_norm),
                None => Tensor::zeros((batch, seq_len, hidden_size), x.dtype(), x.device()),
            };
        }

        // --- Full MoE path ---
        // We iterate token-by-token (batch=1 assumed, seq_len tokens).
        // Each token gets a RoutingDecision from the router.

        let router = self
            .router
            .as_ref()
            .expect("router must be Some when routed_experts is non-empty");

        let mut token_outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // token_norm: [batch=1, 1, hidden_size] → squeeze → [hidden_size]
            let token_norm = x_norm.narrow(1, t, 1)?.squeeze(1)?.squeeze(0)?; // [hidden_size]

            // Get routing decision for this token
            let decision: RoutingDecision =
                router.route_token(&token_norm, state_metrics)?;

            // Weighted sum of selected expert outputs
            // Each expert sees the token as [1, 1, hidden_size] to preserve
            // the 3-D contract of SwiGluExpert::forward.
            let token_3d = token_norm.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, hidden_size]

            let mut routed_out: Option<Tensor> = None;
            for (&expert_idx, &weight) in decision
                .expert_ids
                .iter()
                .zip(decision.expert_weights.iter())
            {
                let expert_out = self.routed_experts[expert_idx].forward(&token_3d)?; // [1, 1, hidden_size]
                let weighted = expert_out.affine(weight as f64, 0.0)?;
                routed_out = Some(match routed_out {
                    None => weighted,
                    Some(acc) => acc.add(&weighted)?,
                });
            }

            // If the router selected no experts (edge case), use zeros
            let mut token_out = match routed_out {
                Some(t) => t,
                None => Tensor::zeros((1, 1, hidden_size), x.dtype(), x.device())?,
            };

            // Add shared expert output unconditionally
            if let Some(shared) = &self.shared_expert {
                let shared_out = shared.forward(&token_3d)?; // [1, 1, hidden_size]
                token_out = (&token_out + &shared_out)?;
            }

            // token_out: [1, 1, hidden_size] → [hidden_size]
            token_outputs.push(token_out.squeeze(0)?.squeeze(0)?);
        }

        // Stack all tokens: [seq_len, hidden_size] → [batch, seq_len, hidden_size]
        let output = Tensor::stack(&token_outputs, 0)?; // [seq_len, hidden_size]
        output.unsqueeze(0) // [1, seq_len, hidden_size]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    const HIDDEN: usize = 64;
    const INTER: usize = 128;

    fn make_vb(device: &Device) -> (VarMap, VarBuilder<'static>) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, device);
        (vm, vb)
    }

    // -----------------------------------------------------------------------
    // 1. SwiGluExpert — shape preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_swiglu_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (_vm, vb) = make_vb(&device);

        let expert = SwiGluExpert::new(HIDDEN, INTER, vb)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, HIDDEN), &device)?;
        let out = expert.forward(&x)?;

        assert_eq!(
            out.dims(),
            &[1, 4, HIDDEN],
            "SwiGluExpert should preserve shape [1, 4, {HIDDEN}], got {:?}",
            out.dims()
        );

        println!("test_swiglu_shapes: output shape {:?}", out.dims());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 2. SwiGluExpert — SiLU math on a known tensor
    // -----------------------------------------------------------------------

    #[test]
    fn test_swiglu_math() -> Result<()> {
        // For a zero-weight expert on a known input, gate_proj(x) = 0,
        // up_proj(x) = 0, so silu(0) * 0 = 0, down_proj(0) = 0.
        // We verify the computation chain with a non-trivial silu check.

        let device = Device::Cpu;

        // Manually verify silu: silu(1.0) = 1.0 * sigmoid(1.0) ~ 0.7311
        // Use a 1-D tensor of length 1 and extract via to_vec1.
        let one = Tensor::new(&[1.0f32], &device)?;
        let silu_one_vec: Vec<f32> = silu(&one)?.to_vec1()?;
        let val: f32 = silu_one_vec[0];
        let expected = 1.0f32 / (1.0 + (-1.0f32).exp()); // sigmoid(1)
        assert!(
            (val - expected).abs() < 1e-5,
            "silu(1.0) = {:.6}, expected {:.6}",
            val,
            expected
        );

        // silu(0.0) should be 0.0
        let zero = Tensor::new(&[0.0f32], &device)?;
        let silu_zero: f32 = silu(&zero)?.to_vec1::<f32>()?[0];
        assert!(
            silu_zero.abs() < 1e-6,
            "silu(0.0) should be 0, got {silu_zero}"
        );

        // Now run an expert with a structured small input to verify the chain.
        // Use tiny dimensions so we can check by hand.
        let h = 4usize;
        let mid = 8usize;
        let (_vm, vb) = make_vb(&device);
        let expert = SwiGluExpert::new(h, mid, vb)?;

        // All-zeros input → all projections output zeros → output must be zeros
        let x_zero = Tensor::zeros((1, 1, h), DType::F32, &device)?;
        let out_zero = expert.forward(&x_zero)?;
        let vals_zero: Vec<f32> = out_zero.flatten_all()?.to_vec1()?;
        let max_abs: f32 = vals_zero.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "Expert on zero input must produce zeros (got max_abs={max_abs})"
        );

        println!(
            "test_swiglu_math: silu(1.0)={:.6}, silu(0.0)={:.6}, zero-input max={:.2e}",
            val, silu_zero, max_abs
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 3. MoeBlock — single shared expert, no routed experts
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_block_single_expert() -> Result<()> {
        // MoeBlock with 1 shared expert and 0 routed experts.
        // Should behave exactly like running SwiGluExpert directly (after norm).
        let device = Device::Cpu;
        let (_vm_shared, vb_shared) = make_vb(&device);
        let (_vm_block, _vb_block) = make_vb(&device);

        let shared = SwiGluExpert::new(HIDDEN, INTER, vb_shared.pp("shared"))?;

        let norm_w = Tensor::ones(HIDDEN, DType::F32, &device)?;
        let block = MoeBlock::new(Some(shared), vec![], None, norm_w, 1e-6);

        let x = Tensor::randn(0.0f32, 1.0, (1, 6, HIDDEN), &device)?;
        let out = block.forward(&x, None)?;

        assert_eq!(
            out.dims(),
            &[1, 6, HIDDEN],
            "MoeBlock (shared only) should output [1, 6, {HIDDEN}], got {:?}",
            out.dims()
        );

        // Output should not be identical to input (norm + FFN transforms it)
        let diff: f32 = (&out - &x)?.abs()?.sum_all()?.to_scalar()?;
        assert!(diff > 0.0, "Output should differ from input after FFN");

        println!(
            "test_moe_block_single_expert: output {:?}, diff_sum={:.4}",
            out.dims(),
            diff
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 4. MoeBlock — no experts at all → return zeros
    // -----------------------------------------------------------------------

    #[test]
    fn test_moe_block_no_experts() -> Result<()> {
        let device = Device::Cpu;

        let norm_w = Tensor::ones(HIDDEN, DType::F32, &device)?;
        let block = MoeBlock::new(None, vec![], None, norm_w, 1e-6);

        let x = Tensor::randn(0.0f32, 1.0, (1, 3, HIDDEN), &device)?;
        let out = block.forward(&x, None)?;

        assert_eq!(
            out.dims(),
            &[1, 3, HIDDEN],
            "MoeBlock (no experts) should output [1, 3, {HIDDEN}]"
        );

        let vals_out: Vec<f32> = out.abs()?.flatten_all()?.to_vec1()?;
        let max_abs: f32 = vals_out.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "MoeBlock with no experts must return zeros, got max_abs={max_abs}"
        );

        println!("test_moe_block_no_experts: max_abs={:.2e}", max_abs);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 5. RmsNormLight — unit input × ones weight = weight values
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_light() -> Result<()> {
        // RMSNorm of a constant vector [c, c, ..., c]:
        //   rms = sqrt(c^2 + eps) ≈ c (for large c)
        //   x_norm ≈ [1, 1, ..., 1]
        //   output = weight * x_norm ≈ weight
        //
        // With weight = ones and input = [1, 1, ..., 1]:
        //   rms = sqrt(1 + eps) ≈ 1
        //   output ≈ [1, 1, ..., 1]

        let device = Device::Cpu;
        let dim = 8usize;

        let weight = Tensor::ones(dim, DType::F32, &device)?;
        let norm = RmsNormLight::new(weight.clone(), 1e-6);

        // All-ones input: rms = sqrt(mean(1^2) + eps) ≈ 1.0
        let x = Tensor::ones((1, 1, dim), DType::F32, &device)?;
        let out = norm.forward(&x)?;

        let vals: Vec<f32> = out.squeeze(0)?.squeeze(0)?.to_vec1()?;
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - 1.0f32).abs() < 1e-4,
                "RmsNorm(ones, weight=ones)[{i}] = {v:.6}, expected ~1.0"
            );
        }

        // With weight = 2*ones: output should be ~2.0
        let weight2 = weight.affine(2.0, 0.0)?;
        let norm2 = RmsNormLight::new(weight2, 1e-6);
        let out2 = norm2.forward(&x)?;
        let vals2: Vec<f32> = out2.squeeze(0)?.squeeze(0)?.to_vec1()?;
        for (i, &v) in vals2.iter().enumerate() {
            assert!(
                (v - 2.0f32).abs() < 1e-4,
                "RmsNorm(ones, weight=2)[{i}] = {v:.6}, expected ~2.0"
            );
        }

        println!(
            "test_rms_norm_light: weight=1 → {:?}, weight=2 → {:?}",
            &vals[..4],
            &vals2[..4]
        );
        Ok(())
    }
}
