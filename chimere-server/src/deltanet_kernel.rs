//! CUDA kernel dispatch for chimere-deltanet.
//!
//! This module re-exports kernel functions from `crate::kernels::*` for
//! backward compatibility. The actual implementations live in the
//! `src/kernels/` module tree.
//!
//! ## Module structure
//!
//! - `kernels::iq3s_gemv` — IQ3_S GEMV + Q8_1 quantize + dequant
//! - `kernels::deltanet_step` — Fused DeltaNet state update
//! - `kernels::elementwise` — Fused elementwise ops (rms_norm, silu, etc.)
//! - `kernels::q5k_gemv` — Q5_K GEMV kernels (F32 and dp4a)
//! - `kernels::topk_softmax` — GPU top-K softmax for MoE routing
//! - `kernels::fused_moe` — Fused MoE IQ3_S kernel
//! - `kernels::nvrtc_compile` — Shared NVRTC compilation helpers

// ---------------------------------------------------------------------------
// Re-exports from kernels::iq3s_gemv
// ---------------------------------------------------------------------------

// Dequant functions (not dispatched — always from iq3s_gemv)
pub use crate::kernels::iq3s_gemv::{
    dequant_iq3s_at_offset,
    dequant_iq3s_gpu,
    dequant_iq3s_into,
};

// GEMV + quantize functions — dispatched via mod.rs (v3 when CHIMERE_IQ3S_V3=1)
pub use crate::kernels::{
    gemv_iq3s_fused,
    gemv_iq3s_fused_at_offset,
    gemv_iq3s_fused_at_offset_q8,
    gemv_iq3s_q8_batched,
    gemv_iq3s_q8_precomputed,
    quantize_f32_to_q8_1_gpu,
};

// ---------------------------------------------------------------------------
// Re-exports from kernels::deltanet_step
// ---------------------------------------------------------------------------

pub use crate::kernels::deltanet_step::deltanet_step_fused;

// ---------------------------------------------------------------------------
// Re-exports from kernels::elementwise
// ---------------------------------------------------------------------------

pub use crate::kernels::elementwise::{
    fused_beta_alpha_gate_tensor,
    fused_rms_norm_silu_gate_tensor,
    raw_argmax,
    raw_rms_norm,
    raw_silu_mul,
};

// ---------------------------------------------------------------------------
// Re-exports from kernels::q5k_gemv
// ---------------------------------------------------------------------------

pub use crate::kernels::q5k_gemv::{
    gemv_q5k_fused,
    gemv_q5k_from_tensor,
    gemv_q5k_q8_from_tensor,
    gemv_q5k_q8_dual_from_tensor,
};

// ---------------------------------------------------------------------------
// Re-exports from kernels::topk_softmax
// ---------------------------------------------------------------------------

pub use crate::kernels::topk_softmax::gpu_topk_softmax;

// ---------------------------------------------------------------------------
// Re-exports from kernels::fused_moe
// ---------------------------------------------------------------------------

pub use crate::kernels::fused_moe::fused_moe_iq3s;
