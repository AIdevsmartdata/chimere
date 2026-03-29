//! GPU top-K softmax kernel for MoE routing.
//!
//! Computes softmax over `num_experts` logits, then selects the top-K
//! experts with renormalised weights. All on GPU -- only 64 bytes
//! (8 i32 indices + 8 f32 weights) are transferred back to CPU.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// -----------------------------------------------------------------------
// CUDA source
// -----------------------------------------------------------------------

const TOPK_KERNEL_SRC: &str = r#"
extern "C" __global__ void topk_softmax(
    const float* __restrict__ logits,
    int*         __restrict__ top_indices,
    float*       __restrict__ top_weights,
    int num_experts,
    int top_k
) {
    extern __shared__ char smem[];
    float* probs = (float*)smem;

    int tid = threadIdx.x;

    float val = (tid < num_experts) ? logits[tid] : -1.0e30f;

    float warp_max = val;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));

    if ((tid & 31) == 0)
        probs[tid >> 5] = warp_max;
    __syncthreads();

    float max_val = 0.0f;
    if (tid == 0) {
        max_val = probs[0];
        int n_warps = (num_experts + 31) >> 5;
        for (int w = 1; w < n_warps; w++)
            max_val = fmaxf(max_val, probs[w]);
        probs[0] = max_val;
    }
    __syncthreads();
    max_val = probs[0];
    __syncthreads();

    float exp_val = (tid < num_experts) ? expf(val - max_val) : 0.0f;
    if (tid < num_experts)
        probs[tid] = exp_val;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < num_experts)
            probs[tid] += probs[tid + stride];
        __syncthreads();
    }
    float sum_exp = probs[0];
    __syncthreads();

    if (tid < num_experts)
        probs[tid] = (sum_exp > 0.0f) ? (exp_val / sum_exp) : (1.0f / (float)num_experts);
    __syncthreads();

    if (tid == 0) {
        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int   best_idx = 0;
            float best_val = -1.0f;
            for (int i = 0; i < num_experts; i++) {
                if (probs[i] > best_val) {
                    best_val = probs[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
            top_weights[k] = best_val;
            weight_sum     += best_val;
            probs[best_idx] = -1.0f;
        }
        if (weight_sum > 0.0f) {
            float inv = 1.0f / weight_sum;
            for (int k = 0; k < top_k; k++)
                top_weights[k] *= inv;
        }
    }
}
"#;

// -----------------------------------------------------------------------
// PTX cache
// -----------------------------------------------------------------------

static TOPK_PTX: OnceLock<String> = OnceLock::new();

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// GPU top-K softmax: compute softmax over `num_experts` logits, extract
/// top-K indices and renormalised weights entirely on GPU.
///
/// # Arguments
/// - `logits`: [num_experts] F32 logits on GPU (CudaView — borrowed from Candle storage).
/// - `top_indices`: pre-allocated [top_k] i32 buffer for output indices.
/// - `top_weights`: pre-allocated [top_k] f32 buffer for output weights.
/// - `num_experts`: total number of experts (e.g. 256).
/// - `top_k`: number of experts to select (e.g. 8).
/// - `dev`: CUDA device handle.
pub fn gpu_topk_softmax(
    logits: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    top_indices: &mut CudaSlice<i32>,
    top_weights: &mut CudaSlice<f32>,
    num_experts: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "topk_softmax", "chimere_topk_softmax_v1",
        TOPK_KERNEL_SRC, &TOPK_PTX,
    )?;

    // One block, num_experts threads (padded to next power of 2, min 256).
    let block_size = (num_experts as u32).next_power_of_two().max(256).min(1024);
    // Shared memory: num_experts floats for probs (max 256*4 = 1 KB).
    let smem = (num_experts as u32) * 4;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: smem,
    };

    let num_experts_i32 = num_experts as i32;
    let top_k_i32 = top_k as i32;

    let mut builder = _stream.launch_builder(&func);
    builder.arg(logits);
    builder.arg(top_indices);
    builder.arg(top_weights);
    builder.arg(&num_experts_i32);
    builder.arg(&top_k_i32);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("topk_softmax launch: {e}")))?;

    Ok(())
}
