//! CUDA kernel modules for chimere-deltanet.
//!
//! This module contains all custom CUDA kernels, organized by functionality:
//! - `cubin_loader`: Pre-compiled cubin from build.rs (nvcc at build time)
//! - `nvrtc_compile`: Shared NVRTC compilation helpers
//! - `iq3s_gemv`: IQ3_S GEMV + Q8_1 quantization + dequantization
//! - `deltanet_step`: Fused DeltaNet state update kernel
//! - `elementwise`: Fused elementwise kernels (rms_norm, silu, sigmoid, etc.)
//! - `q5k_gemv`: Q5_K GEMV kernels
//! - `fused_moe`: Fused MoE IQ3_S kernel
//! - `topk_softmax`: GPU top-K softmax kernel
//! - `gqa_attention`: Fused GQA attention (score + softmax + output, no KV expand)
//! - `flash_attn`: Flash Attention decode (F16 KV, online softmax, CHIMERE_FLASH_ATTN=1)
//! - `attention_raw`: Raw attention kernels (MRoPE, deinterleave, per-head norm, gate)
//! - `gemv_q8_0`: Q8_0 GEMV for lm_head (F32 + dp4a variants)
//! - `fused_ops`: In-place fused kernels (silu_mul, add, scale_add, zero, rmsnorm)
//! - `ggml_gpu`: ggml MMVQ GPU GEMV via FFI (IQ3_S, Q5_K, Q8_0, Q4_K, Q6_K)

pub mod attention_raw;
pub mod cubin_loader;
pub mod flash_attn;
pub mod gemv_q8_0;
pub mod fused_ops;
pub mod nvrtc_compile;
pub mod iq3s_gemv;
pub mod iq3s_gemv_v3;
pub mod deltanet_step;
pub mod elementwise;
pub mod fused_gate_up;
pub mod fused_moe;
pub mod gqa_attention;
pub mod kv_q8_0;
pub mod q5k_gemv;
pub mod q5k_mmvq_ggml;
pub mod ggml_gpu;
pub mod raw_qmatmul;
pub mod topk_softmax;

// Re-export raw attention kernels (Phase 4)
pub use attention_raw::{
    RawMRoPETables,
    raw_mrope_apply,
    raw_mrope_apply_multimodal,
    raw_deinterleave_q_gate,
    raw_rms_norm_per_head,
    raw_sigmoid_gate_mul,
    raw_kv_append,
    // Batch kernels (V2-2 prefill)
    raw_deinterleave_q_gate_batch,
    raw_rms_norm_per_head_batch,
    raw_mrope_apply_batch,
    // Flash Attention causal prefill
    flash_attn_causal_prefill,
    flash_attn_causal_prefill_from_batch,
};

// Re-export Q5_K dual GEMV
pub use q5k_gemv::gemv_q5k_q8_dual_from_tensor;

// Re-export ggml Q5_K MMVQ (cubin-only, toggle: CHIMERE_GGML_Q5K=1)
pub use q5k_mmvq_ggml::{GgmlQ5KBuffers, ggml_q5k_gemv, ggml_q5k_gemv_buffered, ggml_q5k_tensor_forward, ggml_quantize_q8_1};

// Re-export ggml GPU MMVQ via FFI (toggle: CHIMERE_GGML_GPU=1)
pub use ggml_gpu::{GgmlGpuBuffers, GgmlGpuQuantType, ggml_gpu_gemv_iq3s, ggml_gpu_gemv_q5k, ggml_gpu_gemv_q8_0, ggml_gpu_tensor_forward};

// Re-export batched MoE elementwise kernels
pub use elementwise::{
    raw_f32_gemv,
    raw_fused_conv1d_silu_update,
    raw_silu_mul_batched,
    raw_weighted_combine,
};

/// IQ3_S block size: 110 bytes per 256 elements (QK_K=256).
/// Canonical definition — used by iq3s_gemv, iq3s_gemv_v3, and fused_gate_up.
pub const IQ3S_BLOCK_BYTES: usize = 110;

// Re-export public IQ3_S functions — dispatches to v3 when CHIMERE_IQ3S_V3=1
//
// When v3 is disabled (default), these re-export directly from iq3s_gemv.
// When v3 is enabled, wrapper functions dispatch to iq3s_gemv_v3.
// Non-GEMV functions (dequant, quantize) always come from iq3s_gemv (unchanged).

pub use iq3s_gemv::{
    dequant_iq3s_at_offset,
    dequant_iq3s_gpu,
    dequant_iq3s_into,
};

// Re-export the v3 fused gate+up for callers who want it explicitly
pub use iq3s_gemv_v3::fused_gate_up_silu_iq3s_v3;

/// Dispatch IQ3_S GEMV: v3 (shared-mem Q8_1 + 2 rows/block) when toggled on.
#[inline(always)]
pub fn gemv_iq3s_fused(
    weights: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    input: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_fused(weights, input, output, rows, cols, dev)
    } else {
        iq3s_gemv::gemv_iq3s_fused(weights, input, output, rows, cols, dev)
    }
}

#[inline(always)]
pub fn gemv_iq3s_fused_at_offset(
    base: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    byte_offset: usize,
    input: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_fused_at_offset(base, byte_offset, input, output, rows, cols, dev)
    } else {
        iq3s_gemv::gemv_iq3s_fused_at_offset(base, byte_offset, input, output, rows, cols, dev)
    }
}

#[inline(always)]
pub fn gemv_iq3s_fused_at_offset_q8(
    base: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    byte_offset: usize,
    q8_buf: &candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_fused_at_offset_q8(base, byte_offset, q8_buf, output, rows, cols, dev)
    } else {
        iq3s_gemv::gemv_iq3s_fused_at_offset_q8(base, byte_offset, q8_buf, output, rows, cols, dev)
    }
}

#[inline(always)]
pub fn gemv_iq3s_q8_precomputed(
    weights: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    q8_input: &candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_q8_precomputed(weights, q8_input, output, rows, cols, dev)
    } else {
        iq3s_gemv::gemv_iq3s_q8_precomputed(weights, q8_input, output, rows, cols, dev)
    }
}

#[inline(always)]
pub fn gemv_iq3s_q8_batched(
    all_weights: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    q8_input: &candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    expert_ids_buf: &candle_core::cuda_backend::cudarc::driver::CudaSlice<i32>,
    cols: usize,
    rows: usize,
    top_k: usize,
    expert_stride: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_q8_batched(all_weights, q8_input, output, expert_ids_buf, cols, rows, top_k, expert_stride, dev)
    } else {
        iq3s_gemv::gemv_iq3s_q8_batched(all_weights, q8_input, output, expert_ids_buf, cols, rows, top_k, expert_stride, dev)
    }
}

#[inline(always)]
pub fn gemv_iq3s_q8_batched_multi_input(
    all_weights: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    all_q8_inputs: &candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    expert_ids: &candle_core::cuda_backend::cudarc::driver::CudaSlice<i32>,
    cols: usize,
    rows: usize,
    top_k: usize,
    expert_stride: usize,
    q8_stride: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::gemv_iq3s_q8_batched_multi_input(all_weights, all_q8_inputs, output, expert_ids, cols, rows, top_k, expert_stride, q8_stride, dev)
    } else {
        iq3s_gemv::gemv_iq3s_q8_batched_multi_input(all_weights, all_q8_inputs, output, expert_ids, cols, rows, top_k, expert_stride, q8_stride, dev)
    }
}

#[inline(always)]
pub fn quantize_f32_to_q8_1_gpu(
    input: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    n_elements: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::quantize_f32_to_q8_1_gpu(input, output, n_elements, dev)
    } else {
        iq3s_gemv::quantize_f32_to_q8_1_gpu(input, output, n_elements, dev)
    }
}

#[inline(always)]
pub fn quantize_f32_to_q8_1_batched_gpu(
    input_all: &candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    output_all: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    n_per_expert: usize,
    top_k: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    if iq3s_gemv_v3::is_v3() {
        iq3s_gemv_v3::quantize_f32_to_q8_1_batched_gpu(input_all, output_all, n_per_expert, top_k, dev)
    } else {
        iq3s_gemv::quantize_f32_to_q8_1_batched_gpu(input_all, output_all, n_per_expert, top_k, dev)
    }
}
