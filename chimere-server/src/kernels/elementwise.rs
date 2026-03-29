//! Elementwise CUDA kernels for chimere-deltanet.
//!
//! Small fused kernels: rms_norm, silu, sigmoid, softplus, mul, silu_mul,
//! add, weighted_add, argmax, bias_softplus_mul, sigmoid_mul, f32_gemv.
//!
//! Also fused GDN elementwise kernels:
//!   fused_beta_alpha_gate, fused_rms_norm_silu_gate, fused_conv1d_silu_update.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, DevicePtr, DeviceSlice, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Result, Tensor};
use std::sync::OnceLock;

// -----------------------------------------------------------------------
// Basic elementwise kernels
// -----------------------------------------------------------------------

const ELEMWISE_KERNEL_SRC: &str = r#"
extern "C" __global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n,
    float eps
) {
    __shared__ float sdata[256];
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

extern "C" __global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C" __global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = logf(1.0f + expf(input[i]));
    }
}

extern "C" __global__ void mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] * b[i];
}

extern "C" __global__ void silu_mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float silu_x = x / (1.0f + expf(-x));
        output[i] = silu_x * b[i];
    }
}

extern "C" __global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}

extern "C" __global__ void weighted_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float weight,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += input[i] * weight;
}

extern "C" __global__ void argmax_kernel(
    const float* __restrict__ input,
    int* __restrict__ output,
    int n
) {
    __shared__ float sval[256];
    __shared__ int sidx[256];
    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = sidx[0];
}

extern "C" __global__ void bias_softplus_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i] + bias[i];
        output[i] = logf(1.0f + expf(x)) * scale[i];
    }
}

extern "C" __global__ void sigmoid_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        output[i] += g * input[i];
    }
}

// Shared-memory GEMV: y = A @ x, A is [rows, cols] row-major.
// Grid: (rows, 1, 1), Block: (256, 1, 1).
// Each block computes one output element y[row].
// The input vector x is loaded into shared memory once per block,
// eliminating redundant global memory reads (was 256*2048 = 512K reads,
// now 2048 reads + shared broadcast).
extern "C" __global__ void f32_gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    extern __shared__ float sx[];       // shared x[cols]
    __shared__ float sred[256];         // reduction scratch

    int row = blockIdx.x;
    if (row >= rows) return;

    // Cooperatively load x into shared memory
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        sx[j] = x[j];
    }
    __syncthreads();

    // Each thread computes a partial dot product
    float acc = 0.0f;
    const float* a_row = A + (long long)row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        acc += a_row[j] * sx[j];
    }

    // Block-level reduction
    sred[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) y[row] = sred[0];
}

// Batched silu_mul: processes top_k experts in a single launch.
// Grid: (ceil(expert_ffn/256), top_k, 1)  Block: (256, 1, 1)
// gate_all/up_all/inter_all are [top_k * expert_ffn] contiguous.
extern "C" __global__ void silu_mul_batched_kernel(
    const float* __restrict__ gate_all,
    const float* __restrict__ up_all,
    float* __restrict__ inter_all,
    int expert_ffn,
    int top_k
) {
    int k = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= expert_ffn || k >= top_k) return;
    int idx = k * expert_ffn + i;
    float g = gate_all[idx];
    float s = g / (1.0f + expf(-g));
    inter_all[idx] = s * up_all[idx];
}

// Batched weighted combine: replaces top_k weighted_add launches with 1.
// Grid: (ceil(hidden_size/256), 1, 1)  Block: (256, 1, 1)
// expert_outs is [top_k * hidden_size], weights is [top_k].
// combined[i] = sum_k( weights[k] * expert_outs[k * hidden_size + i] )
extern "C" __global__ void weighted_combine_kernel(
    const float* __restrict__ expert_outs,
    const float* __restrict__ weights,
    float* __restrict__ combined,
    int hidden_size,
    int top_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        sum += weights[k] * expert_outs[k * hidden_size + i];
    }
    combined[i] = sum;
}

// L2-normalize groups: input[g*d..g*d+d] /= ||input[g*d..g*d+d]||_2
// Grid: (n_group, 1, 1), Block: (blockDim, 1, 1) where blockDim >= d
extern "C" __global__ void l2_norm_groups_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int d_state,
    int n_group,
    float eps
) {
    int g = blockIdx.x;
    int j = threadIdx.x;
    extern __shared__ char smem_raw[];
    float* sdata = (float*)smem_raw;
    int base = g * d_state;
    float val = 0.0f;
    float sq = 0.0f;
    if (j < d_state) {
        val = input[base + j];
        sq = val * val;
    }
    sdata[j] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) sdata[j] += sdata[j + s];
        __syncthreads();
    }
    if (j >= d_state) return;
    float norm = sqrtf(sdata[0] + eps);
    output[base + j] = val / norm;
}

// Expand groups: repeat entire tensor 'repeats' times (TILED layout).
// input: [n_group * d_state], output: [n_group * repeats * d_state]
// Layout: [g0, g1, ..., g(n-1), g0, g1, ..., g(n-1), ...]
// This matches Candle's Tensor::repeat([1, repeats, 1]) behavior.
// Grid: (ceil(total/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void expand_groups_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int d_state,
    int n_group,
    int repeats
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_group * repeats * d_state;
    if (i >= total) return;
    // output layout: [repeat0: g0..g(n-1), repeat1: g0..g(n-1), ...]
    // head = i / d_state  (0..n_group*repeats-1)
    // group = head % n_group  (tiled: wraps around after n_group)
    int head = i / d_state;
    int dim = i % d_state;
    int group = head % n_group;
    output[i] = input[group * d_state + dim];
}

// Scale: output[i] = input[i] * scale
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * scale;
}

// Add: output[i] = a[i] + b[i]  (for residual connections)
// Already exists as add_kernel above, but this is in-place variant:
// output[i] += input[i]
extern "C" __global__ void add_inplace_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += input[i];
}
"#;

static ELEMWISE_PTX: OnceLock<String> = OnceLock::new();

/// Public accessor for the compiled add_inplace_kernel PTX.
/// Used by `AddCudaSliceInplace` to launch the same kernel as `raw_add_inplace`.
pub fn get_add_inplace_ptx() -> &'static str {
    super::nvrtc_compile::compile_and_cache(ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX)
}

// -----------------------------------------------------------------------
// Fused GDN elementwise kernels
// -----------------------------------------------------------------------

const FUSED_GDN_KERNEL_SRC: &str = r#"
extern "C" __global__ void fused_beta_alpha_gate(
    const float* __restrict__ beta_proj,
    const float* __restrict__ alpha_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ beta_out,
    float* __restrict__ gate_exp_out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    beta_out[i] = 1.0f / (1.0f + expf(-beta_proj[i]));
    float ab = alpha_proj[i] + dt_bias[i];
    float sp = ab > 20.0f ? ab : logf(1.0f + expf(ab));
    gate_exp_out[i] = expf(sp * ssm_a[i]);
}

// Batch variant: processes N * dt_rank elements.
// dt_bias[j] and ssm_a[j] are indexed via modulo (shared across tokens).
// NOTE: beta_proj/beta_out and alpha_proj/gate_exp_out MAY alias (in-place),
// so __restrict__ is NOT used on those pairs. Each thread reads [i] then
// writes [i], so no cross-thread hazard — but the compiler must not reorder
// the read past the write within the same thread.
extern "C" __global__ void fused_beta_alpha_gate_batch(
    const float * beta_proj,               // [N * dt_rank] (may alias beta_out)
    const float * alpha_proj,              // [N * dt_rank] (may alias gate_exp_out)
    const float * __restrict__ dt_bias,    // [dt_rank] shared
    const float * __restrict__ ssm_a,      // [dt_rank] shared
    float * beta_out,                      // [N * dt_rank] (may alias beta_proj)
    float * gate_exp_out,                  // [N * dt_rank] (may alias alpha_proj)
    int total_n,   // N * dt_rank
    int dt_rank    // for modulo
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_n) return;
    int j = i % dt_rank;
    // Read inputs to registers FIRST to ensure correctness when aliased.
    float bp = beta_proj[i];
    float ap = alpha_proj[i];
    float bias = dt_bias[j];
    float a = ssm_a[j];
    // Compute and write outputs.
    beta_out[i] = 1.0f / (1.0f + expf(-bp));
    float ab = ap + bias;
    float sp = ab > 20.0f ? ab : logf(1.0f + expf(ab));
    gate_exp_out[i] = expf(sp * a);
}

extern "C" __global__ void fused_rms_norm_silu_gate(
    const float* __restrict__ ssm_out,
    const float* __restrict__ weight,
    const float* __restrict__ z,
    float* __restrict__ output,
    int D,
    float eps
) {
    const int g = blockIdx.x;
    const int j = threadIdx.x;
    extern __shared__ char smem[];
    float* sdata = (float*)smem;
    const int base = g * D;
    float val = 0.0f;
    float sq = 0.0f;
    if (j < D) {
        val = ssm_out[base + j];
        sq = val * val;
    }
    sdata[j] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) sdata[j] += sdata[j + s];
        __syncthreads();
    }
    if (j >= D) return;
    float rms = sqrtf(sdata[0] / (float)D + eps);
    float inv_rms = 1.0f / rms;
    float normed = val * inv_rms * weight[j];
    float zv = z[base + j];
    float silu_z = zv / (1.0f + expf(-zv));
    output[base + j] = normed * silu_z;
}
"#;

static FUSED_GDN_PTX: OnceLock<String> = OnceLock::new();

// -----------------------------------------------------------------------
// Fused conv1d + SiLU + state update kernel
// -----------------------------------------------------------------------

const FUSED_CONV1D_SILU_KERNEL_SRC: &str = r#"
// Fused conv1d + state update + SiLU for GDN SSM layers.
// For single-token inference: conv_kernel=4, so it's a 4-tap FIR filter.
//
// Grid: (ceil(channels/256), 1, 1)
// Block: (256, 1, 1)
//
extern "C" __global__ void fused_conv1d_silu_update(
    const float* __restrict__ conv_state,    // [channels, conv_kernel-1]
    const float* __restrict__ new_input,     // [channels]
    const float* __restrict__ conv_weight,   // [channels, conv_kernel]
    float* __restrict__ output,              // [channels] (after silu)
    float* __restrict__ new_state,           // [channels, conv_kernel-1]
    int channels,
    int conv_kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int km1 = conv_kernel - 1;

    // 1. Build the full conv window [conv_kernel] for this channel
    //    = [state[ch,0], state[ch,1], ..., state[ch,km1-1], new_input[ch]]
    float sum = 0.0f;
    for (int k = 0; k < km1; k++) {
        float s = conv_state[ch * km1 + k];
        sum += s * conv_weight[ch * conv_kernel + k];
        // 4. Update state: shift left (drop oldest, keep last km1 values)
        //    new_state[ch, k] = window[k+1]
        if (k < km1 - 1)
            new_state[ch * km1 + k] = conv_state[ch * km1 + k + 1];
        else
            new_state[ch * km1 + k] = new_input[ch];
    }
    sum += new_input[ch] * conv_weight[ch * conv_kernel + km1];

    // 3. SiLU activation: x * sigmoid(x)
    float silu = sum / (1.0f + expf(-sum));
    output[ch] = silu;
}
"#;

static FUSED_CONV1D_SILU_PTX: OnceLock<String> = OnceLock::new();

// -----------------------------------------------------------------------
// Helper: extract CudaDevice from Candle Device
// -----------------------------------------------------------------------

fn get_cuda_dev(device: &Device) -> Result<&CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d),
        _ => candle_core::bail!("elementwise kernel: expected CUDA device"),
    }
}

// -----------------------------------------------------------------------
// Public API: basic elementwise ops
// -----------------------------------------------------------------------

/// RMS Norm: output[i] = input[i] * inv_rms * weight[i]
pub fn raw_rms_norm(
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "rms_norm_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(weight);
    builder.arg(output);
    builder.arg(&n_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;
    Ok(())
}

/// Fused silu(gate) * up -> output. CudaSlice inputs.
pub fn raw_silu_mul(
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "silu_mul_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(gate);
    builder.arg(up);
    builder.arg(output);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul launch: {e}")))?;
    Ok(())
}

/// GPU argmax over `n` float values. Writes single i32 result to `output`.
pub fn raw_argmax(
    logits: &CudaSlice<f32>,
    output: &mut CudaSlice<i32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "argmax_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(logits);
    builder.arg(output);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("argmax launch: {e}")))?;
    Ok(())
}

// -----------------------------------------------------------------------
// Public API: F32 GEMV (raw CudaSlice interface)
// -----------------------------------------------------------------------

/// F32 GEMV: y = A @ x, where A is [rows, cols] row-major and x is [cols].
/// Output y is [rows], written into a pre-allocated CudaSlice.
///
/// Uses shared-memory kernel: 1 block per row, 256 threads/block.
/// Input vector x is loaded into shared memory once per block, then each
/// thread computes a partial dot product with block-level reduction.
/// For [256x2048] MoE router: eliminates 512K redundant global reads.
///
/// This is used for the MoE router matmul (hidden [2048] x router [2048,256])
/// to avoid Candle tensor allocation overhead.
pub fn raw_f32_gemv(
    a: &CudaView<'_, f32>,     // [rows * cols] row-major
    x: &CudaView<'_, f32>,     // [cols]
    y: &mut CudaSlice<f32>,    // [rows] output
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "f32_gemv_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    // 1 block per row, 256 threads per block.
    // Shared memory: cols floats for input vector + 256 floats for reduction.
    let shared_bytes = (cols as u32) * 4; // sx[] only; sred[256] is statically allocated
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(a);
    builder.arg(x);
    builder.arg(y);
    builder.arg(&rows_i32);
    builder.arg(&cols_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("f32_gemv launch: {e}")))?;
    Ok(())
}

// -----------------------------------------------------------------------
// Public API: batched MoE elementwise ops
// -----------------------------------------------------------------------

/// Batched silu_mul across `top_k` experts in a single kernel launch.
///
/// gate_all, up_all, inter_all are contiguous buffers of size [top_k * expert_ffn].
/// Computes: inter_all[k*expert_ffn + i] = silu(gate_all[...]) * up_all[...]
/// for all k in 0..top_k and i in 0..expert_ffn.
pub fn raw_silu_mul_batched(
    gate_all: &CudaSlice<f32>,
    up_all: &CudaSlice<f32>,
    inter_all: &mut CudaSlice<f32>,
    expert_ffn: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "silu_mul_batched_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let blocks_x = ((expert_ffn as u32) + 255) / 256;
    let expert_ffn_i32 = expert_ffn as i32;
    let top_k_i32 = top_k as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, top_k as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(gate_all);
    builder.arg(up_all);
    builder.arg(inter_all);
    builder.arg(&expert_ffn_i32);
    builder.arg(&top_k_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul_batched launch: {e}")))?;
    Ok(())
}

/// Batched weighted combine of `top_k` expert outputs into a single vector.
///
/// expert_outs is [top_k * hidden_size], weights is [top_k].
/// combined[i] = sum_k( weights[k] * expert_outs[k * hidden_size + i] )
///
/// Replaces `top_k` separate weighted_add launches with 1 kernel launch.
pub fn raw_weighted_combine(
    expert_outs: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    combined: &mut CudaSlice<f32>,
    hidden_size: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "weighted_combine_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let blocks = ((hidden_size as u32) + 255) / 256;
    let hidden_size_i32 = hidden_size as i32;
    let top_k_i32 = top_k as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(expert_outs);
    builder.arg(weights);
    builder.arg(combined);
    builder.arg(&hidden_size_i32);
    builder.arg(&top_k_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("weighted_combine launch: {e}")))?;
    Ok(())
}

// -----------------------------------------------------------------------
// Public API: raw GDN SSM elementwise kernels (CudaSlice interface)
// -----------------------------------------------------------------------

/// L2-normalize groups of vectors: output[g,j] = input[g,j] / ||input[g,:]||_2
/// Input/output are flat [n_group * d_state]. Each group is normalized independently.
pub fn raw_l2_norm_groups(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    d_state: usize,
    n_group: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "l2_norm_groups_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let block_size = d_state.next_power_of_two().max(32).min(1024) as u32;
    let d_state_i32 = d_state as i32;
    let n_group_i32 = n_group as i32;
    let cfg = LaunchConfig {
        grid_dim: (n_group as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&d_state_i32);
    builder.arg(&n_group_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("l2_norm_groups launch: {e}")))?;
    Ok(())
}

/// Expand groups: repeat each of n_group vectors 'repeats' times (tiled layout).
/// Input: [n_group * d_state], Output: [n_group * repeats * d_state].
pub fn raw_expand_groups(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    d_state: usize,
    n_group: usize,
    repeats: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "expand_groups_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let total = n_group * repeats * d_state;
    let blocks = ((total as u32) + 255) / 256;
    let d_state_i32 = d_state as i32;
    let n_group_i32 = n_group as i32;
    let repeats_i32 = repeats as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&d_state_i32);
    builder.arg(&n_group_i32);
    builder.arg(&repeats_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("expand_groups launch: {e}")))?;
    Ok(())
}

/// Scale: output[i] = input[i] * scale
pub fn raw_scale(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "scale_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&n_i32);
    builder.arg(&scale);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("scale launch: {e}")))?;
    Ok(())
}

/// In-place add: output[i] += input[i]
pub fn raw_add_inplace(
    output: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "add_inplace_kernel", "chimere_elemwise_v4", ELEMWISE_KERNEL_SRC, &ELEMWISE_PTX,
    )?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(output);
    builder.arg(input);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("add_inplace launch: {e}")))?;
    Ok(())
}

// -----------------------------------------------------------------------
// Public API: fused GDN elementwise ops (Tensor interface)
// -----------------------------------------------------------------------

/// Fused beta/alpha/gate computation for GDN layers.
///
/// Computes in one kernel launch:
///   beta_out[i] = sigmoid(beta_proj[i])
///   gate_exp_out[i] = exp(softplus(alpha_proj[i] + dt_bias[i]) * ssm_a[i])
///
/// Returns (beta, gate_exp) as Tensors with same shape as input.
pub fn fused_beta_alpha_gate_tensor(
    beta_proj: &Tensor,
    alpha_proj: &Tensor,
    dt_bias: &Tensor,
    ssm_a: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle_core::Storage;

    let device = beta_proj.device();
    let dev = get_cuda_dev(device)?;
    let n = beta_proj.elem_count();

    // Extract CudaViews from contiguous tensors (hold storage guards)
    let bp = beta_proj.contiguous()?;
    let ap = alpha_proj.contiguous()?;
    let db = dt_bias.contiguous()?;
    let sa = ssm_a.contiguous()?;

    let (bp_stor, bp_lay) = bp.storage_and_layout();
    let bp_cuda = match &*bp_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let bp_view = bp_cuda.as_cuda_slice::<f32>()?.slice(bp_lay.start_offset()..);

    let (ap_stor, ap_lay) = ap.storage_and_layout();
    let ap_cuda = match &*ap_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let ap_view = ap_cuda.as_cuda_slice::<f32>()?.slice(ap_lay.start_offset()..);

    let (db_stor, db_lay) = db.storage_and_layout();
    let db_cuda = match &*db_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let db_view = db_cuda.as_cuda_slice::<f32>()?.slice(db_lay.start_offset()..);

    let (sa_stor, sa_lay) = sa.storage_and_layout();
    let sa_cuda = match &*sa_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let sa_view = sa_cuda.as_cuda_slice::<f32>()?.slice(sa_lay.start_offset()..);

    let stream = dev.cuda_stream();
    let mut beta_out: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .map_err(|e| candle_core::Error::Msg(format!("alloc beta_out: {e}")))?;
    let mut gate_out: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .map_err(|e| candle_core::Error::Msg(format!("alloc gate_out: {e}")))?;

    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_beta_alpha_gate", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;

    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(&bp_view);
    builder.arg(&ap_view);
    builder.arg(&db_view);
    builder.arg(&sa_view);
    builder.arg(&mut beta_out);
    builder.arg(&mut gate_out);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_beta_alpha_gate launch: {e}")))?;

    // Drop storage guards
    drop(bp_stor); drop(ap_stor); drop(db_stor); drop(sa_stor);

    // Wrap results as Tensors with the same shape as beta_proj
    let shape = beta_proj.shape().clone();
    let beta_tensor = {
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            beta_out, dev.clone(),
        );
        Tensor::from_storage(
            Storage::Cuda(storage),
            shape.clone(),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let gate_tensor = {
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            gate_out, dev.clone(),
        );
        Tensor::from_storage(
            Storage::Cuda(storage),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
    };

    Ok((beta_tensor, gate_tensor))
}

/// Raw CudaSlice variant of fused_beta_alpha_gate — zero Tensor allocation.
/// Writes directly to pre-allocated output buffers.
pub fn raw_fused_beta_alpha_gate(
    beta_proj: &CudaSlice<f32>,
    alpha_proj: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    ssm_a: &CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    gate_exp_out: &mut CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_beta_alpha_gate", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;
    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(beta_proj);
    builder.arg(alpha_proj);
    builder.arg(dt_bias);
    builder.arg(ssm_a);
    builder.arg(beta_out);
    builder.arg(gate_exp_out);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("raw_fused_beta_alpha_gate launch: {e}")))?;
    Ok(())
}

/// Batch variant of fused_beta_alpha_gate — processes N tokens in one kernel launch.
///
/// Reads from contiguous [n_tokens * dt_rank] input buffers and writes results
/// back to same-sized output buffers. dt_bias and ssm_a are [dt_rank] and
/// indexed via modulo (shared across all tokens).
///
/// Can operate in-place: beta_proj_batch and beta_out_batch may alias
/// (same for alpha/gate_exp), since the kernel reads before writing per-element.
pub fn raw_fused_beta_alpha_gate_batch(
    beta_proj_batch: &CudaSlice<f32>,   // [n_tokens * dt_rank]
    alpha_proj_batch: &CudaSlice<f32>,  // [n_tokens * dt_rank]
    dt_bias: &CudaSlice<f32>,           // [dt_rank]
    ssm_a: &CudaSlice<f32>,             // [dt_rank]
    beta_out_batch: &mut CudaSlice<f32>,// [n_tokens * dt_rank]
    gate_exp_batch: &mut CudaSlice<f32>,// [n_tokens * dt_rank]
    dt_rank: usize,
    n_tokens: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let total_n = n_tokens * dt_rank;
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_beta_alpha_gate_batch", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;
    let blocks = ((total_n as u32) + 255) / 256;
    let total_n_i32 = total_n as i32;
    let dt_rank_i32 = dt_rank as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(beta_proj_batch);
    builder.arg(alpha_proj_batch);
    builder.arg(dt_bias);
    builder.arg(ssm_a);
    builder.arg(beta_out_batch);
    builder.arg(gate_exp_batch);
    builder.arg(&total_n_i32);
    builder.arg(&dt_rank_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("raw_fused_beta_alpha_gate_batch launch: {e}")))?;
    Ok(())
}

/// In-place batch variant: reads from beta_batch/alpha_batch and writes results
/// back to the SAME buffers. Safe because the CUDA kernel reads each element
/// before writing (per-thread, no cross-element dependency).
///
/// Uses raw device pointers for the "input" aliases to bypass Rust's borrow
/// checker (same pattern as `deltanet_step_fused_raw_inplace`).
pub fn raw_fused_beta_alpha_gate_batch_inplace(
    beta_batch: &mut CudaSlice<f32>,    // [n_tokens * dt_rank] in/out
    alpha_batch: &mut CudaSlice<f32>,   // [n_tokens * dt_rank] in/out
    dt_bias: &CudaSlice<f32>,           // [dt_rank]
    ssm_a: &CudaSlice<f32>,            // [dt_rank]
    dt_rank: usize,
    n_tokens: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DeviceSlice;

    let total_n = n_tokens * dt_rank;
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_beta_alpha_gate_batch", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;
    let blocks = ((total_n as u32) + 255) / 256;
    let total_n_i32 = total_n as i32;
    let dt_rank_i32 = dt_rank as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Extract raw device pointers for the "input" read aliases.
    // SAFETY: kernel reads beta_proj[i] before writing beta_out[i] per-element,
    // no cross-element dependency. Same pattern as deltanet_step in-place state.
    let beta_in_ptr = {
        let view: CudaView<'_, f32> = beta_batch.slice(..);
        let (ptr, _sync) = view.device_ptr(view.stream());
        ptr
    };
    let alpha_in_ptr = {
        let view: CudaView<'_, f32> = alpha_batch.slice(..);
        let (ptr, _sync) = view.device_ptr(view.stream());
        ptr
    };

    unsafe {
        _stream
            .launch_builder(&func)
            .arg(&beta_in_ptr)       // input: beta_proj (raw ptr)
            .arg(&alpha_in_ptr)      // input: alpha_proj (raw ptr)
            .arg(dt_bias)
            .arg(ssm_a)
            .arg(beta_batch)         // output: beta_out (same buffer, via &mut)
            .arg(alpha_batch)        // output: gate_exp_out (same buffer, via &mut)
            .arg(&total_n_i32)
            .arg(&dt_rank_i32)
            .launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("raw_fused_beta_alpha_gate_batch_inplace launch: {e}")))?;
    Ok(())
}

/// Raw CudaSlice variant of fused_rms_norm_silu_gate — zero Tensor allocation.
/// output[g*D+j] = rms_norm(ssm_out[g*D+j], weight[j]) * silu(z[g*D+j])
pub fn raw_fused_rms_norm_silu_gate(
    ssm_out: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    z: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    groups: usize,
    d: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_rms_norm_silu_gate", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;
    let block_size = d.next_power_of_two().max(32).min(1024) as u32;
    let d_i32 = d as i32;
    let cfg = LaunchConfig {
        grid_dim: (groups as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(ssm_out);
    builder.arg(norm_weight);
    builder.arg(z);
    builder.arg(output);
    builder.arg(&d_i32);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("raw_fused_rms_norm_silu_gate launch: {e}")))?;
    Ok(())
}

/// Fused RMSNorm + SiLU gate for GDN output.
///
/// output[g, j] = rms_norm(ssm_out[g, j], weight[j]) * silu(z[g, j])
///
/// `hidden` is [groups, D], `norm_weight` is [D], `gate` is [1, groups*D].
/// Returns [1, groups*D].
pub fn fused_rms_norm_silu_gate_tensor(
    hidden: &Tensor,
    norm_weight: &Tensor,
    gate: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    use candle_core::Storage;

    let device = hidden.device();
    let dev = get_cuda_dev(device)?;

    // hidden is [1, groups, D] or [groups, D]
    let hidden_c = hidden.contiguous()?;
    let dims = hidden_c.dims();
    let (groups, d) = if dims.len() == 3 {
        (dims[1], dims[2])
    } else if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        candle_core::bail!("fused_rms_norm_silu_gate: expected 2D or 3D tensor, got {:?}", dims);
    };

    let nw = norm_weight.contiguous()?;
    let gate_c = gate.contiguous()?;

    let (h_stor, h_lay) = hidden_c.storage_and_layout();
    let h_cuda = match &*h_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let h_view = h_cuda.as_cuda_slice::<f32>()?.slice(h_lay.start_offset()..);

    let (w_stor, w_lay) = nw.storage_and_layout();
    let w_cuda = match &*w_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let w_view = w_cuda.as_cuda_slice::<f32>()?.slice(w_lay.start_offset()..);

    let (g_stor, g_lay) = gate_c.storage_and_layout();
    let g_cuda = match &*g_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let g_view = g_cuda.as_cuda_slice::<f32>()?.slice(g_lay.start_offset()..);

    let total = groups * d;
    let stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(total)
        .map_err(|e| candle_core::Error::Msg(format!("alloc fused_rms_out: {e}")))?;

    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_rms_norm_silu_gate", "chimere_fused_gdn_elem_v2",
        FUSED_GDN_KERNEL_SRC, &FUSED_GDN_PTX,
    )?;

    let block_size = d.next_power_of_two().max(32).min(1024) as u32;
    let d_i32 = d as i32;
    let eps_f32 = eps as f32;
    let cfg = LaunchConfig {
        grid_dim: (groups as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(&h_view);
    builder.arg(&w_view);
    builder.arg(&g_view);
    builder.arg(&mut output);
    builder.arg(&d_i32);
    builder.arg(&eps_f32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_rms_norm_silu_gate launch: {e}")))?;

    drop(h_stor); drop(w_stor); drop(g_stor);

    let out_shape = candle_core::Shape::from_dims(&[1, total]);
    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
        output, dev.clone(),
    );
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        out_shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

// -----------------------------------------------------------------------
// Public API: fused conv1d + SiLU + state update (CudaSlice interface)
// -----------------------------------------------------------------------

/// Fused depthwise conv1d + SiLU activation + conv state update.
///
/// Replaces 9 Candle Tensor ops (reshape, cat, narrow, contiguous, squeeze,
/// mul, sum, unsqueeze, silu) with a single kernel launch.
///
/// For single-token inference in GDN SSM layers:
/// - Builds a [channels, conv_kernel] window from the old state + new input
/// - Computes depthwise 1D convolution (dot product per channel)
/// - Applies SiLU activation
/// - Updates the sliding window state (shift left, append new input)
///
/// # Arguments
/// - `conv_state`: Old conv state, `[channels * (conv_kernel - 1)]` flat.
/// - `new_input`: New input column, `[channels]`.
/// - `conv_weight`: Depthwise conv weights, `[channels * conv_kernel]` flat.
/// - `output`: Pre-allocated output buffer, `[channels]`.
/// - `new_state`: Pre-allocated buffer for updated state, `[channels * (conv_kernel - 1)]`.
///   May alias `conv_state` since the kernel reads old state before writing new.
/// - `channels`: Number of channels (conv_channels = 8192 for Qwen3.5-35B-A3B).
/// - `conv_kernel`: Kernel size (4 for Qwen3.5).
/// - `dev`: CUDA device.
pub fn raw_fused_conv1d_silu_update(
    conv_state: &CudaView<'_, f32>,
    new_input: &CudaView<'_, f32>,
    conv_weight: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    new_state: &mut CudaSlice<f32>,
    channels: usize,
    conv_kernel: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_conv1d_silu_update", "chimere_fused_conv1d_silu_v1",
        FUSED_CONV1D_SILU_KERNEL_SRC, &FUSED_CONV1D_SILU_PTX,
    )?;
    let blocks = ((channels as u32) + 255) / 256;
    let channels_i32 = channels as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(conv_state);
    builder.arg(new_input);
    builder.arg(conv_weight);
    builder.arg(output);
    builder.arg(new_state);
    builder.arg(&channels_i32);
    builder.arg(&conv_kernel_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_conv1d_silu_update launch: {e}")))?;
    Ok(())
}

// -----------------------------------------------------------------------
// GDN scratch buffers — pre-allocated, zero cudaMalloc per token
// -----------------------------------------------------------------------

/// Pre-allocated GPU scratch buffers for the raw GDN SSM path.
/// Allocated once at model load time, reused every layer every token.
/// Eliminates 11 cudaMalloc calls per GDN layer (330/token total).
pub struct GdnScratchBuffers {
    pub conv_output: CudaSlice<f32>,    // [conv_channels]
    pub new_state: CudaSlice<f32>,      // [conv_channels * (conv_kernel-1)]
    pub q_split: CudaSlice<f32>,        // [key_dim]
    pub k_split: CudaSlice<f32>,        // [key_dim]
    pub q_normed: CudaSlice<f32>,       // [key_dim]
    pub k_normed: CudaSlice<f32>,       // [key_dim]
    pub q_expanded: CudaSlice<f32>,     // [dt_rank * d_state]
    pub k_expanded: CudaSlice<f32>,     // [dt_rank * d_state]
    pub q_scaled: CudaSlice<f32>,       // [dt_rank * d_state]
    pub v_copy: CudaSlice<f32>,         // [value_dim]
}

impl GdnScratchBuffers {
    /// Allocate scratch buffers for Qwen3.5-35B-A3B dimensions.
    pub fn new(
        conv_channels: usize,
        conv_kernel: usize,
        key_dim: usize,
        value_dim: usize,
        dt_rank: usize,
        d_state: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        let km1 = conv_kernel - 1;
        let expanded = dt_rank * d_state;
        let a = |n: usize, name: &str| -> Result<CudaSlice<f32>> {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| candle_core::Error::Msg(format!("alloc {name}: {e}")))
        };
        Ok(Self {
            conv_output: a(conv_channels, "conv_output")?,
            new_state: a(conv_channels * km1, "new_state")?,
            q_split: a(key_dim, "q_split")?,
            k_split: a(key_dim, "k_split")?,
            q_normed: a(key_dim, "q_normed")?,
            k_normed: a(key_dim, "k_normed")?,
            q_expanded: a(expanded, "q_expanded")?,
            k_expanded: a(expanded, "k_expanded")?,
            q_scaled: a(expanded, "q_scaled")?,
            v_copy: a(value_dim, "v_copy")?,
        })
    }
}

// -----------------------------------------------------------------------
// Tensor-level wrapper: fused conv1d+silu + split + L2 norm + expand
// -----------------------------------------------------------------------

/// Fused conv1d+silu + split QKV + L2 norm + expand groups — replaces ~27 Candle ops.
///
/// Takes QKV projection output (Candle Tensor) and conv state, runs fused conv1d+silu,
/// splits into Q/K/V, L2-normalizes Q and K per group, expands to dt_rank heads,
/// and scales Q by 1/sqrt(d_state).
///
/// Returns `(conv_activated, q_scaled, k_expanded, v_3d, new_conv_state)` as Candle Tensors.
///
/// # Shapes
/// - `qkv`: `[1, conv_channels]` — QKV projection output
/// - `conv_state`: `[1, conv_channels, conv_kernel-1]` — sliding window state
/// - `conv_weight`: `[conv_channels, conv_kernel]` — depthwise conv weights
/// - Returns: q_scaled `[1, dt_rank, d_state]`, k_expanded `[1, dt_rank, d_state]`,
///   v_3d `[1, dt_rank, d_state]`, new_conv_state `[1, conv_channels, conv_kernel-1]`
pub fn fused_conv_split_norm_expand_tensor(
    qkv: &Tensor,
    conv_state: &Tensor,
    conv_weight: &Tensor,
    key_dim: usize,        // n_group * d_state
    value_dim: usize,      // dt_rank * d_state
    conv_channels: usize,  // key_dim * 2 + value_dim
    conv_kernel: usize,    // 4
    n_group: usize,        // 16
    d_state: usize,        // 128
    dt_rank: usize,        // 32
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    use candle_core::Storage;

    let device = qkv.device();
    let dev = get_cuda_dev(device)?;
    let stream = dev.cuda_stream();
    let km1 = conv_kernel - 1;

    // --- Extract CudaSlice from Candle tensors ---
    // IMPORTANT: use exact-range slices (start..start+n) NOT open-ended (start..)
    // to avoid copying extra data from shared storage buffers.
    let qkv_n = qkv.elem_count();
    let qkv_c = qkv.contiguous()?;
    let (qkv_stor, qkv_lay) = qkv_c.storage_and_layout();
    let qkv_cuda = match &*qkv_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let qkv_off = qkv_lay.start_offset();
    let qkv_view = qkv_cuda.as_cuda_slice::<f32>()?.slice(qkv_off..qkv_off + qkv_n);

    let cs_n = conv_state.elem_count();
    let conv_state_c = conv_state.contiguous()?;
    let (cs_stor, cs_lay) = conv_state_c.storage_and_layout();
    let cs_cuda = match &*cs_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let cs_off = cs_lay.start_offset();
    let cs_view = cs_cuda.as_cuda_slice::<f32>()?.slice(cs_off..cs_off + cs_n);

    let cw_n = conv_weight.elem_count();
    let conv_weight_c = conv_weight.contiguous()?;
    let (cw_stor, cw_lay) = conv_weight_c.storage_and_layout();
    let cw_cuda = match &*cw_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let cw_off = cw_lay.start_offset();
    let cw_view = cw_cuda.as_cuda_slice::<f32>()?.slice(cw_off..cw_off + cw_n);

    // --- Allocate output buffers ---
    let mut conv_output: CudaSlice<f32> = stream.alloc_zeros(conv_channels)
        .map_err(|e| candle_core::Error::Msg(format!("alloc conv_output: {e}")))?;
    let mut new_state: CudaSlice<f32> = stream.alloc_zeros(conv_channels * km1)
        .map_err(|e| candle_core::Error::Msg(format!("alloc new_state: {e}")))?;

    // --- Step 1: fused conv1d + silu + state update (1 kernel launch) ---
    // conv_state is [1, conv_channels, km1] = [conv_channels * km1] flat
    // qkv is [1, conv_channels] = the new input column
    // Pass CudaViews directly — the GPU kernel just reads from the device pointer,
    // it doesn't care about Rust ownership. No dtod copies needed.
    raw_fused_conv1d_silu_update(
        &cs_view, &qkv_view, &cw_view,
        &mut conv_output, &mut new_state,
        conv_channels, conv_kernel, dev,
    )?;

    // --- Step 2+3: split Q/K/V views and L2 normalize (zero-copy, Phase 2.3) ---
    // Q = [0..key_dim], K = [key_dim..key_dim*2], V = [key_dim*2..key_dim*2+value_dim]
    let q_view = conv_output.slice(0..key_dim);
    let k_view = conv_output.slice(key_dim..key_dim * 2);

    let mut q_normed: CudaSlice<f32> = stream.alloc_zeros(key_dim)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q_normed: {e}")))?;
    let mut k_normed: CudaSlice<f32> = stream.alloc_zeros(key_dim)
        .map_err(|e| candle_core::Error::Msg(format!("alloc k_normed: {e}")))?;

    raw_l2_norm_groups(&q_view, &mut q_normed, d_state, n_group, eps as f32, dev)?;
    raw_l2_norm_groups(&k_view, &mut k_normed, d_state, n_group, eps as f32, dev)?;

    // --- Step 4: expand groups (2 kernel launches) ---
    let repeats = dt_rank / n_group;
    let expanded_size = dt_rank * d_state;
    let mut q_expanded: CudaSlice<f32> = stream.alloc_zeros(expanded_size)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q_expanded: {e}")))?;
    let mut k_expanded: CudaSlice<f32> = stream.alloc_zeros(expanded_size)
        .map_err(|e| candle_core::Error::Msg(format!("alloc k_expanded: {e}")))?;

    raw_expand_groups(&q_normed.slice(..), &mut q_expanded, d_state, n_group, repeats, dev)?;
    raw_expand_groups(&k_normed.slice(..), &mut k_expanded, d_state, n_group, repeats, dev)?;

    // --- Step 5: scale Q by 1/sqrt(d_state) (1 kernel launch) ---
    let scale = 1.0f32 / (d_state as f32).sqrt();
    let mut q_scaled: CudaSlice<f32> = stream.alloc_zeros(expanded_size)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q_scaled: {e}")))?;
    raw_scale(&q_expanded.slice(..), &mut q_scaled, expanded_size, scale, dev)?;

    // --- Drop storage guards ---
    drop(qkv_stor);
    drop(cs_stor);
    drop(cw_stor);

    // --- Wrap results as Candle Tensors ---
    let wrap = |slice: CudaSlice<f32>, shape: &[usize]| -> Result<Tensor> {
        let s = candle_core::Shape::from_dims(shape);
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(slice, dev.clone());
        Ok(Tensor::from_storage(
            Storage::Cuda(storage), s,
            candle_core::op::BackpropOp::none(), false,
        ))
    };

    let q_scaled_t = wrap(q_scaled, &[1, dt_rank, d_state])?;
    let k_expanded_t = wrap(k_expanded, &[1, dt_rank, d_state])?;
    // V: copy the slice to an owned buffer
    let mut v_owned: CudaSlice<f32> = stream.alloc_zeros(value_dim)
        .map_err(|e| candle_core::Error::Msg(format!("alloc v_owned: {e}")))?;
    stream.memcpy_dtod(&conv_output.slice(key_dim * 2..key_dim * 2 + value_dim), &mut v_owned)
        .map_err(|e| candle_core::Error::Msg(format!("dtod v: {e}")))?;
    let v_3d_t = wrap(v_owned, &[1, dt_rank, d_state])?;
    let new_state_t = wrap(new_state, &[1, conv_channels, km1])?;

    Ok((q_scaled_t, k_expanded_t, v_3d_t, new_state_t))
}

/// Same as `fused_conv_split_norm_expand_tensor` but uses pre-allocated scratch buffers.
/// Zero cudaMalloc per call — all temporaries live in `scratch`.
/// The output Tensors still need fresh CudaSlice allocations (4 total) because they
/// are consumed by the DeltaNet kernel which takes ownership.
pub fn fused_conv_split_norm_expand_with_scratch(
    qkv: &Tensor,
    conv_state: &Tensor,
    conv_weight: &Tensor,
    key_dim: usize,
    value_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    n_group: usize,
    d_state: usize,
    dt_rank: usize,
    eps: f64,
    scratch: &mut GdnScratchBuffers,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    use candle_core::Storage;

    let device = qkv.device();
    let dev = get_cuda_dev(device)?;
    let stream = dev.cuda_stream();
    let km1 = conv_kernel - 1;

    // Extract CudaViews from Candle tensors (exact-range slices)
    let qkv_n = qkv.elem_count();
    let qkv_c = qkv.contiguous()?;
    let (qkv_stor, qkv_lay) = qkv_c.storage_and_layout();
    let qkv_cuda = match &*qkv_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let qkv_off = qkv_lay.start_offset();
    let qkv_view = qkv_cuda.as_cuda_slice::<f32>()?.slice(qkv_off..qkv_off + qkv_n);

    let cs_n = conv_state.elem_count();
    let conv_state_c = conv_state.contiguous()?;
    let (cs_stor, cs_lay) = conv_state_c.storage_and_layout();
    let cs_cuda = match &*cs_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let cs_off = cs_lay.start_offset();
    let cs_view = cs_cuda.as_cuda_slice::<f32>()?.slice(cs_off..cs_off + cs_n);

    let cw_n = conv_weight.elem_count();
    let conv_weight_c = conv_weight.contiguous()?;
    let (cw_stor, cw_lay) = conv_weight_c.storage_and_layout();
    let cw_cuda = match &*cw_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let cw_off = cw_lay.start_offset();
    let cw_view = cw_cuda.as_cuda_slice::<f32>()?.slice(cw_off..cw_off + cw_n);

    // Step 1: fused conv1d + silu + state update
    // Pass CudaViews directly — zero dtod copies. The GPU kernel just reads
    // from the device pointer; Rust ownership is irrelevant at the CUDA level.
    raw_fused_conv1d_silu_update(
        &cs_view, &qkv_view, &cw_view,
        &mut scratch.conv_output, &mut scratch.new_state,
        conv_channels, conv_kernel, dev,
    )?;

    // Step 2+3: split Q/K views and L2 normalize (zero-copy, Phase 2.3)
    let q_view = scratch.conv_output.slice(0..key_dim);
    let k_view = scratch.conv_output.slice(key_dim..key_dim * 2);

    raw_l2_norm_groups(&q_view, &mut scratch.q_normed, d_state, n_group, eps as f32, dev)?;
    raw_l2_norm_groups(&k_view, &mut scratch.k_normed, d_state, n_group, eps as f32, dev)?;

    // Step 4: expand groups
    let repeats = dt_rank / n_group;
    raw_expand_groups(&scratch.q_normed.slice(..), &mut scratch.q_expanded, d_state, n_group, repeats, dev)?;
    raw_expand_groups(&scratch.k_normed.slice(..), &mut scratch.k_expanded, d_state, n_group, repeats, dev)?;

    // Step 5: scale Q
    let scale = 1.0f32 / (d_state as f32).sqrt();
    let expanded_size = dt_rank * d_state;
    raw_scale(&scratch.q_expanded.slice(..), &mut scratch.q_scaled, expanded_size, scale, dev)?;

    // Drop storage guards before wrapping outputs
    drop(qkv_stor);
    drop(cs_stor);
    drop(cw_stor);

    // Wrap results as Candle Tensors (4 allocs for output ownership — unavoidable)
    let wrap = |src: &CudaSlice<f32>, n: usize, shape: &[usize]| -> Result<Tensor> {
        let mut dst = stream.alloc_zeros(n)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;
        stream.memcpy_dtod(&src.slice(..n), &mut dst)
            .map_err(|e| candle_core::Error::Msg(format!("dtod output: {e}")))?;
        let s = candle_core::Shape::from_dims(shape);
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok(Tensor::from_storage(Storage::Cuda(storage), s, candle_core::op::BackpropOp::none(), false))
    };

    let q_t = wrap(&scratch.q_scaled, expanded_size, &[1, dt_rank, d_state])?;
    let k_t = wrap(&scratch.k_expanded, expanded_size, &[1, dt_rank, d_state])?;
    // V: wrap directly from conv_output view (zero-copy split, Phase 2.3)
    let v_t = {
        let v_view = scratch.conv_output.slice(key_dim * 2..key_dim * 2 + value_dim);
        let mut dst = stream.alloc_zeros(value_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc v output: {e}")))?;
        stream.memcpy_dtod(&v_view, &mut dst)
            .map_err(|e| candle_core::Error::Msg(format!("dtod v output: {e}")))?;
        let s = candle_core::Shape::from_dims(&[1, dt_rank, d_state]);
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Tensor::from_storage(Storage::Cuda(storage), s, candle_core::op::BackpropOp::none(), false)
    };
    let ns_t = wrap(&scratch.new_state, conv_channels * km1, &[1, conv_channels, km1])?;

    Ok((q_t, k_t, v_t, ns_t))
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that fused conv1d+silu+update matches the separate operations:
    ///   1. Build window = cat(conv_state, new_input)
    ///   2. conv_out[ch] = dot(window[ch,:], weight[ch,:])
    ///   3. output[ch] = silu(conv_out[ch])
    ///   4. new_state = window shifted left (drop column 0, keep 1..conv_kernel-1 + new_input)
    #[test]
    fn test_fused_conv1d_silu_update_matches_reference() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[CONV1D_SILU] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let channels = 8192usize;
        let conv_kernel = 4usize;
        let km1 = conv_kernel - 1; // 3

        // Generate deterministic test data
        let conv_state_h: Vec<f32> = (0..channels * km1)
            .map(|i| ((i as f32) * 0.01 - 1.0).sin())
            .collect();
        let new_input_h: Vec<f32> = (0..channels)
            .map(|i| ((i as f32) * 0.037 + 0.5).cos())
            .collect();
        let conv_weight_h: Vec<f32> = (0..channels * conv_kernel)
            .map(|i| ((i as f32) * 0.0023 - 0.3).sin() * 0.5)
            .collect();

        // --- Reference: compute on CPU ---
        let mut ref_output = vec![0.0f32; channels];
        let mut ref_new_state = vec![0.0f32; channels * km1];
        for ch in 0..channels {
            // Build window: [state[ch,0], state[ch,1], state[ch,2], new_input[ch]]
            let mut window = [0.0f32; 4];
            for k in 0..km1 {
                window[k] = conv_state_h[ch * km1 + k];
            }
            window[km1] = new_input_h[ch];

            // Dot product with weight
            let mut sum = 0.0f32;
            for k in 0..conv_kernel {
                sum += window[k] * conv_weight_h[ch * conv_kernel + k];
            }

            // SiLU
            let silu = sum / (1.0 + (-sum).exp());
            ref_output[ch] = silu;

            // Update state: shift left
            for k in 0..km1 {
                ref_new_state[ch * km1 + k] = window[k + 1];
            }
        }

        // --- GPU: run fused kernel ---
        let stream = cuda_dev.cuda_stream();

        let mut conv_state_gpu: CudaSlice<f32> = stream
            .alloc_zeros(channels * km1)
            .map_err(|e| candle_core::Error::Msg(format!("alloc conv_state: {e}")))?;
        cuda_dev
            .memcpy_htod(&conv_state_h, &mut conv_state_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload conv_state: {e}")))?;

        let mut new_input_gpu: CudaSlice<f32> = stream
            .alloc_zeros(channels)
            .map_err(|e| candle_core::Error::Msg(format!("alloc new_input: {e}")))?;
        cuda_dev
            .memcpy_htod(&new_input_h, &mut new_input_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload new_input: {e}")))?;

        let mut conv_weight_gpu: CudaSlice<f32> = stream
            .alloc_zeros(channels * conv_kernel)
            .map_err(|e| candle_core::Error::Msg(format!("alloc conv_weight: {e}")))?;
        cuda_dev
            .memcpy_htod(&conv_weight_h, &mut conv_weight_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload conv_weight: {e}")))?;

        let mut output_gpu: CudaSlice<f32> = stream
            .alloc_zeros(channels)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;

        let mut new_state_gpu: CudaSlice<f32> = stream
            .alloc_zeros(channels * km1)
            .map_err(|e| candle_core::Error::Msg(format!("alloc new_state: {e}")))?;

        raw_fused_conv1d_silu_update(
            &conv_state_gpu.slice(..),
            &new_input_gpu.slice(..),
            &conv_weight_gpu.slice(..),
            &mut output_gpu,
            &mut new_state_gpu,
            channels,
            conv_kernel,
            cuda_dev,
        )?;

        // Read back results
        let gpu_output: Vec<f32> = output_gpu
            .stream()
            .clone()
            .clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback output: {e}")))?;

        let gpu_new_state: Vec<f32> = new_state_gpu
            .stream()
            .clone()
            .clone_dtoh(&new_state_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback new_state: {e}")))?;

        // --- Compare ---
        let mut max_output_err = 0.0f32;
        for ch in 0..channels {
            let err = (gpu_output[ch] - ref_output[ch]).abs();
            if err > max_output_err {
                max_output_err = err;
            }
        }
        eprintln!(
            "[CONV1D_SILU] channels={channels}, conv_kernel={conv_kernel}, max_output_err={max_output_err:.2e}"
        );
        assert!(
            max_output_err < 1e-4,
            "Output mismatch: max_err={max_output_err:.6e} (tolerance 1e-4)"
        );

        let mut max_state_err = 0.0f32;
        for i in 0..channels * km1 {
            let err = (gpu_new_state[i] - ref_new_state[i]).abs();
            if err > max_state_err {
                max_state_err = err;
            }
        }
        eprintln!(
            "[CONV1D_SILU] max_state_err={max_state_err:.2e}"
        );
        assert!(
            max_state_err < 1e-6,
            "State mismatch: max_err={max_state_err:.6e} (tolerance 1e-6)"
        );

        eprintln!("[CONV1D_SILU] PASS: fused kernel matches reference");
        Ok(())
    }
}
