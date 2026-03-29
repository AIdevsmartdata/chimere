//! Fused in-place CUDA kernels to reduce kernel launch overhead.
//!
//! These kernels replace Candle tensor operations that allocate new tensors
//! with in-place GPU operations on pre-allocated scratch buffers.
//!
//! - `silu_mul_inplace`: gate[i] = silu(gate[i]) * up[i]  (replaces 3 ops)
//! - `add_inplace_f32`: a[i] += b[i]  (replaces Candle add + alloc)
//! - `scale_add_inplace`: accum[i] += scale * x[i]  (weighted expert accumulation)
//! - `zero_f32`: out[i] = 0  (fast buffer clear)
//! - `rmsnorm_f32`: fused RMSNorm with shared-memory reduction
//! - `fused_add_residual_rmsnorm_batch`: batch residual-add + RMSNorm for N tokens (prefill)
//! - `rmsnorm_f32_batch`: batch RMSNorm for N tokens (prefill, no residual-add)
//!
//! All kernels target sm_120 via NVRTC and use the cubin-first load path.

use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

const FUSED_OPS_KERNEL_SRC: &str = r#"
// =====================================================================
// Kernel 1: silu_mul_inplace
// out[i] = silu(gate[i]) * up[i]  -- replaces 3 ops (silu, mul, alloc)
// Gate buffer is modified in-place.
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void silu_mul_inplace(
    float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));  // silu(g)
        gate[i] = s * up[i];
    }
}

// =====================================================================
// Kernel 2: add_inplace_f32
// a[i] += b[i]  -- replaces Candle add which allocates a new tensor
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// =====================================================================
// Kernel 3: scale_add_inplace
// accum[i] += scale * x[i]  -- for weighted expert accumulation
// Eliminates separate scale + add + 2 allocs per expert.
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void scale_add_inplace(
    float* __restrict__ accum,
    const float* __restrict__ x,
    float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] += scale * x[i];
}

// =====================================================================
// Kernel 4: zero_f32
// out[i] = 0  -- zero a buffer (faster than cudaMemset for small buffers,
// avoids host-side synchronization overhead)
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void zero_f32(
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

// =====================================================================
// Kernel 6: fused_add_residual_rmsnorm
// residual[i] = hidden[i] + residual[i]  (accumulate)
// normed[i] = (residual[i] / rms) * weight[i]  (normalize)
// Two passes: first accumulate + sum_sq, then normalize.
// Grid: (1, 1, 1), Block: (256, 1, 1), SharedMem: 256 * sizeof(float)
// =====================================================================
extern "C" __global__ void fused_add_residual_rmsnorm(
    const float* __restrict__ hidden,
    float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ normed,
    int n,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Pass 1: accumulate residual += hidden, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = hidden[i] + residual[i];
        residual[i] = val;
        sum_sq += val * val;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    // Pass 2: normalize and scale
    for (int i = tid; i < n; i += blockDim.x) {
        normed[i] = residual[i] * inv_rms * weight[i];
    }
}

// =====================================================================
// Kernel 5: rmsnorm_f32
// Fused RMSNorm: out[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
// Uses shared memory reduction (one block per normalization).
// Grid: (1, 1, 1), Block: (256, 1, 1), SharedMem: 256 * sizeof(float)
// =====================================================================
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Step 1: compute sum of squares (stride loop for n > blockDim.x)
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: normalize and scale (stride loop)
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// =====================================================================
// Kernel 7: fused_split_norm_expand_scale
// Fuses GDN steps 9-12 into a single launch:
//   - Read Q, K from conv_output (split by offset)
//   - L2-normalize Q and K per group
//   - Expand from n_group to dt_rank heads (TILED layout)
//   - Scale expanded Q by 1/sqrt(d_state)
//
// Grid: (n_group, 1, 1) — one block per group
// Block: (d_state, 1, 1) — one thread per element (d_state=128)
// Shared mem: 2 * d_state floats (for Q and K reductions)
//
// Replaces 5 kernel launches (2x L2 norm + 2x expand + 1x scale)
// with a single launch per GDN layer.
// =====================================================================
extern "C" __global__ void fused_split_norm_expand_scale(
    const float* __restrict__ conv_output,   // [conv_channels] input
    float* __restrict__ q_scaled,            // [dt_rank * d_state] output
    float* __restrict__ k_expanded,          // [dt_rank * d_state] output
    int key_dim,       // n_group * d_state = offset to K in conv_output
    int d_state,       // 128
    int n_group,       // 16
    int repeats,       // dt_rank / n_group
    float eps,         // 1e-6
    float q_scale      // 1.0 / sqrt(d_state)
) {
    extern __shared__ float sdata[];
    // sdata[0..blockDim.x-1] for Q reduction
    // sdata[blockDim.x..2*blockDim.x-1] for K reduction

    int g = blockIdx.x;   // group index
    int j = threadIdx.x;  // element index within group
    int base = g * d_state;

    // Read Q and K values for this group
    float q_val = 0.0f;
    float k_val = 0.0f;
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    if (j < d_state) {
        q_val = conv_output[base + j];               // Q at offset 0
        k_val = conv_output[key_dim + base + j];     // K at offset key_dim
        q_sq = q_val * q_val;
        k_sq = k_val * k_val;
    }

    // Parallel reduction for Q sum-of-squares
    sdata[j] = q_sq;
    sdata[blockDim.x + j] = k_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) {
            sdata[j] += sdata[j + s];
            sdata[blockDim.x + j] += sdata[blockDim.x + j + s];
        }
        __syncthreads();
    }

    if (j >= d_state) return;

    // L2 norm: norm = sqrt(sum_sq + eps), matching raw_l2_norm_groups exactly
    float q_norm = sqrtf(sdata[0] + eps);
    float k_norm = sqrtf(sdata[blockDim.x] + eps);
    float q_normed = q_val / q_norm;
    float k_normed = k_val / k_norm;

    // Expand to TILED layout + scale Q
    // TILED: head h maps to group (h % n_group)
    // For each repeat r, the output head index is (r * n_group + g)
    // Output offset for head h, element j: h * d_state + j
    for (int r = 0; r < repeats; r++) {
        int head = r * n_group + g;  // TILED: [g0..g15, g0..g15, ...]
        int out_idx = head * d_state + j;
        q_scaled[out_idx]   = q_normed * q_scale;
        k_expanded[out_idx] = k_normed;
    }
}

// =====================================================================
// Kernel 8: fused_add_residual_rmsnorm_batch
// Batch version of Kernel 6 for N-token prefill.
// Grid: (N, 1, 1) — one block per token
// Block: (256, 1, 1) — threads cooperate on one row
// SharedMem: 256 * sizeof(float)
//
// For each token t:
//   residual[t*H + i] += hidden[t*H + i]
//   normed[t*H + i] = rmsnorm(residual[t*H..]) * weight[i]
//
// Uses IDENTICAL reduction logic (shared-memory tree) as the single-token
// kernel to guarantee bit-identical results per row.
// =====================================================================
extern "C" __global__ void fused_add_residual_rmsnorm_batch(
    const float * __restrict__ hidden,
    float * __restrict__ residual,
    const float * __restrict__ weight,
    float * __restrict__ normed,
    int H,
    float eps
) {
    extern __shared__ float shared[];
    int t = blockIdx.x;
    int tid = threadIdx.x;

    const float* h_row = hidden + t * H;
    float* r_row = residual + t * H;
    float* n_row = normed + t * H;

    // Pass 1: accumulate residual += hidden, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float val = h_row[i] + r_row[i];
        r_row[i] = val;
        sum_sq += val * val;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    // Reduction (identical to single-token kernel)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)H + eps);
    float inv_rms = 1.0f / rms;

    // Pass 2: normalize and scale
    for (int i = tid; i < H; i += blockDim.x) {
        n_row[i] = r_row[i] * inv_rms * weight[i];
    }
}

// =====================================================================
// Kernel 9: rmsnorm_f32_batch
// Batch version of Kernel 5 for N-token prefill (standalone, no residual-add).
// Grid: (N, 1, 1), Block: (256, 1, 1), SharedMem: 256 * sizeof(float)
//
// For each token t:
//   out[t*H + i] = (x[t*H + i] / rms) * weight[i]
// =====================================================================
extern "C" __global__ void rmsnorm_f32_batch(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    float * __restrict__ out,
    int H,
    float eps
) {
    extern __shared__ float shared[];
    int t = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + t * H;
    float* o_row = out + t * H;

    // Step 1: compute sum of squares (stride loop)
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    // Reduction (identical to single-token kernel)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)H + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: normalize and scale
    for (int i = tid; i < H; i += blockDim.x) {
        o_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

// =====================================================================
// Kernel: weighted_combine_gpu
// Phase 1.2: GPU-resident weighted combination of expert outputs.
// Reads expert weights from GPU memory (no CPU sync needed).
//
// accum[h] = sum_k(weights[k] * expert_outputs[k * hidden_size + h])
//
// Grid: (ceil(hidden_size/256), 1, 1), Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void weighted_combine_gpu(
    const float * __restrict__ expert_outputs,
    const float * __restrict__ weights,
    float * __restrict__ accum,
    int hidden_size,
    int top_k
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden_size) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        sum += weights[k] * expert_outputs[k * hidden_size + h];
    }
    accum[h] = sum;
}
"#;

// ---------------------------------------------------------------------------
// PTX compilation (cached, NVRTC via cubin-first path)
// ---------------------------------------------------------------------------

static PTX_CACHE: OnceLock<String> = OnceLock::new();

fn load_func(
    dev: &CudaDevice,
    fn_name: &str,
) -> Result<(
    candle_core::cuda_backend::cudarc::driver::CudaFunction,
    std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
)> {
    super::nvrtc_compile::get_or_load_func(
        dev,
        fn_name,
        "chimere_fused_ops",
        FUSED_OPS_KERNEL_SRC,
        &PTX_CACHE,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused SiLU-multiply in-place: `gate[i] = silu(gate[i]) * up[i]`.
///
/// Replaces 3 separate operations (silu activation, element-wise multiply,
/// output allocation) with a single kernel launch that writes back into the
/// gate buffer.
///
/// `gate` is modified in-place. `up` is read-only.
/// Both must have at least `n` elements.
pub fn silu_mul_inplace(
    gate: &mut CudaSlice<f32>,
    up: &CudaView<'_, f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "silu_mul_inplace")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(gate);
    builder.arg(up);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul_inplace launch: {e}")))?;
    Ok(())
}

/// In-place add: `a[i] += b[i]`.
///
/// Replaces Candle's `Tensor::add` which allocates a new output tensor.
/// `a` is modified in-place. `b` is read-only.
/// Both must have at least `n` elements.
pub fn add_inplace_f32(
    a: &mut CudaSlice<f32>,
    b: &CudaView<'_, f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "add_inplace_f32")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(a);
    builder.arg(b);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("add_inplace_f32 launch: {e}")))?;
    Ok(())
}

/// Scaled accumulate in-place: `accum[i] += scale * x[i]`.
///
/// Used for weighted expert accumulation in MoE layers. Replaces a separate
/// scale kernel + add kernel + 2 tensor allocations with a single fused launch.
///
/// `accum` is modified in-place. `x` is read-only.
/// Both must have at least `n` elements.
pub fn scale_add_inplace(
    accum: &mut CudaSlice<f32>,
    x: &CudaView<'_, f32>,
    scale: f32,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "scale_add_inplace")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(accum);
    builder.arg(x);
    builder.arg(&scale);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("scale_add_inplace launch: {e}")))?;
    Ok(())
}

/// Zero a float buffer: `out[i] = 0.0`.
///
/// Faster than `cudaMemset` for small buffers because it avoids host-side
/// synchronization overhead (cudaMemset is a CUDA runtime call, this is a
/// kernel on the same stream).
///
/// `out` must have at least `n` elements.
pub fn zero_f32(
    out: &mut CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "zero_f32")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(out);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("zero_f32 launch: {e}")))?;
    Ok(())
}

/// Fused RMSNorm: `out[i] = (x[i] / rms) * weight[i]`
/// where `rms = sqrt(mean(x^2) + eps)`.
///
/// Uses shared-memory parallel reduction for the sum-of-squares computation.
/// Single block launch (256 threads) handles vectors up to any size via
/// stride loops.
///
/// `x`: input vector, at least `n` elements.
/// `weight`: per-element scale weights, at least `n` elements.
/// `out`: output buffer, at least `n` elements.
/// `eps`: small constant for numerical stability (typically 1e-6).
pub fn rmsnorm_f32(
    x: &CudaView<'_, f32>,
    weight: &CudaView<'_, f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "rmsnorm_f32")?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4, // 256 floats for reduction
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(x);
    builder.arg(weight);
    builder.arg(out);
    builder.arg(&n_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rmsnorm_f32 launch: {e}")))?;
    Ok(())
}

/// Variant of `rmsnorm_f32` that accepts `CudaSlice` inputs (owned buffers)
/// instead of `CudaView` (borrowed slices). Useful when the caller has
/// pre-allocated scratch buffers.
pub fn rmsnorm_f32_slices(
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "rmsnorm_f32")?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(x);
    builder.arg(weight);
    builder.arg(out);
    builder.arg(&n_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rmsnorm_f32 launch: {e}")))?;
    Ok(())
}

/// Variant of `silu_mul_inplace` that accepts a `CudaSlice` for `up`
/// (owned buffer) instead of `CudaView` (borrowed slice).
pub fn silu_mul_inplace_slices(
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "silu_mul_inplace")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(gate);
    builder.arg(up);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul_inplace launch: {e}")))?;
    Ok(())
}

/// Variant of `add_inplace_f32` that accepts a `CudaSlice` for `b`
/// (owned buffer) instead of `CudaView` (borrowed slice).
pub fn add_inplace_f32_slices(
    a: &mut CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "add_inplace_f32")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(a);
    builder.arg(b);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("add_inplace_f32 launch: {e}")))?;
    Ok(())
}

/// Fused residual-add + RMSNorm in a single kernel launch.
///
/// Computes:
///   residual[i] = hidden[i] + residual[i]   (accumulate into residual)
///   normed[i] = rmsnorm(residual) * weight[i]
///
/// Replaces 2 kernel launches (add_inplace + rmsnorm) + 1 pointer swap with
/// a single launch. Called 2× per layer (pre-attention + pre-FFN) = 80→40 launches.
///
/// After this call:
/// - `residual` contains the accumulated hidden state
/// - `normed` contains the normalized state for the next sub-layer
/// - `hidden` is stale (will be overwritten by the next sub-layer)
pub fn fused_add_residual_rmsnorm(
    hidden: &CudaSlice<f32>,
    residual: &mut CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    normed: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "fused_add_residual_rmsnorm")?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4, // 256 floats for reduction
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(hidden);
    builder.arg(residual);
    builder.arg(weight);
    builder.arg(normed);
    builder.arg(&n_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_add_residual_rmsnorm launch: {e}")))?;
    Ok(())
}

/// Batch fused residual-add + RMSNorm for N tokens.
///
/// For each token `t` in `0..n_tokens`:
///   `residual[t*H..] += hidden[t*H..]`
///   `normed[t*H..] = rmsnorm(residual[t*H..]) * weight`
///
/// Grid: `(n_tokens, 1, 1)`, Block: `(256, 1, 1)`.
/// Uses IDENTICAL shared-memory tree reduction as the single-token variant
/// so results are bit-identical per row.
///
/// After this call:
/// - `residual` contains the accumulated hidden state for all tokens
/// - `normed` contains the normalized state for the next sub-layer
/// - `hidden` is stale (will be overwritten by the next sub-layer)
pub fn fused_add_residual_rmsnorm_batch(
    hidden_batch: &CudaSlice<f32>,
    residual_batch: &mut CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    normed_batch: &mut CudaSlice<f32>,
    hidden_size: usize,
    n_tokens: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }
    let (func, stream) = load_func(dev, "fused_add_residual_rmsnorm_batch")?;
    let hs_i32 = hidden_size as i32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4, // 256 floats for reduction
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(hidden_batch);
    builder.arg(residual_batch);
    builder.arg(weight);
    builder.arg(normed_batch);
    builder.arg(&hs_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_add_residual_rmsnorm_batch launch: {e}")))?;
    Ok(())
}

/// Batch RMSNorm for N tokens (standalone, no residual-add).
///
/// For each token `t` in `0..n_tokens`:
///   `out[t*H..] = rmsnorm(x[t*H..]) * weight`
///
/// Grid: `(n_tokens, 1, 1)`, Block: `(256, 1, 1)`.
/// Uses IDENTICAL shared-memory tree reduction as the single-token `rmsnorm_f32`
/// so results are bit-identical per row.
///
/// Useful for the final norm before lm_head during batch prefill.
pub fn rmsnorm_f32_batch(
    x_batch: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out_batch: &mut CudaSlice<f32>,
    hidden_size: usize,
    n_tokens: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }
    let (func, stream) = load_func(dev, "rmsnorm_f32_batch")?;
    let hs_i32 = hidden_size as i32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4, // 256 floats for reduction
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(x_batch);
    builder.arg(weight);
    builder.arg(out_batch);
    builder.arg(&hs_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rmsnorm_f32_batch launch: {e}")))?;
    Ok(())
}

/// Variant of `scale_add_inplace` that accepts a `CudaSlice` for `x`
/// (owned buffer) instead of `CudaView` (borrowed slice).
pub fn scale_add_inplace_slices(
    accum: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    scale: f32,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "scale_add_inplace")?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(accum);
    builder.arg(x);
    builder.arg(&scale);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("scale_add_inplace launch: {e}")))?;
    Ok(())
}

/// Phase 1.2: GPU-resident weighted combination of expert outputs.
///
/// accum[h] = sum_k(weights[k] * expert_outputs[k * hidden_size + h])
///
/// Expert weights are read from GPU memory — zero CPU sync.
pub fn weighted_combine_gpu(
    expert_outputs: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    accum: &mut CudaSlice<f32>,
    hidden_size: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "weighted_combine_gpu")?;
    let blocks = ((hidden_size as u32) + 255) / 256;
    let hs_i32 = hidden_size as i32;
    let tk_i32 = top_k as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(expert_outputs);
    builder.arg(weights);
    builder.arg(accum);
    builder.arg(&hs_i32);
    builder.arg(&tk_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("weighted_combine_gpu launch: {e}")))?;
    Ok(())
}

/// Fused split + L2-norm + expand + scale for GDN steps 9-12.
///
/// Reads Q and K from `conv_output` (at offsets 0 and `key_dim`),
/// L2-normalizes each group, expands from `n_group` to `dt_rank` heads
/// (TILED layout matching `expand_groups_kernel`), and scales Q by
/// `1/sqrt(d_state)`.
///
/// Replaces 5 kernel launches (2x L2 norm + 2x expand + 1x scale) with
/// a single launch. Grid: (n_group, 1, 1), Block: (block_size, 1, 1).
///
/// # Arguments
/// - `conv_output`: full conv output buffer [conv_channels], Q at [0..key_dim], K at [key_dim..key_dim*2]
/// - `q_scaled`: output buffer [dt_rank * d_state], expanded + scaled Q
/// - `k_expanded`: output buffer [dt_rank * d_state], expanded K
/// - `key_dim`: n_group * d_state (offset to K in conv_output)
/// - `d_state`: head dimension (128)
/// - `n_group`: number of groups (16)
/// - `repeats`: dt_rank / n_group (expansion factor)
/// - `eps`: L2 norm epsilon (1e-6)
pub fn fused_split_norm_expand_scale(
    conv_output: &CudaSlice<f32>,
    q_scaled: &mut CudaSlice<f32>,
    k_expanded: &mut CudaSlice<f32>,
    key_dim: usize,
    d_state: usize,
    n_group: usize,
    repeats: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, stream) = load_func(dev, "fused_split_norm_expand_scale")?;
    let block_size = d_state.next_power_of_two().max(32).min(1024) as u32;
    let key_dim_i32 = key_dim as i32;
    let d_state_i32 = d_state as i32;
    let n_group_i32 = n_group as i32;
    let repeats_i32 = repeats as i32;
    let q_scale = 1.0f32 / (d_state as f32).sqrt();
    let cfg = LaunchConfig {
        grid_dim: (n_group as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4 * 2, // 2x block_size floats for Q and K reductions
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(conv_output);
    builder.arg(q_scaled);
    builder.arg(k_expanded);
    builder.arg(&key_dim_i32);
    builder.arg(&d_state_i32);
    builder.arg(&n_group_i32);
    builder.arg(&repeats_i32);
    builder.arg(&eps);
    builder.arg(&q_scale);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(
            format!("fused_split_norm_expand_scale launch: {e}")))?;
    Ok(())
}
