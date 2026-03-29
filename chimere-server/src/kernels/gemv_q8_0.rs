//! Q8_0 GEMV kernel for chimere-deltanet's lm_head projection.
//!
//! Projects hidden[2048] -> logits[vocab_size=248320] using Q8_0-quantized weights.
//! This is the single most impactful GEMV in the model — it runs every token
//! and is the largest weight matrix (248320 x 2048 = ~524 MB in Q8_0).
//!
//! ## Q8_0 format
//!
//! Each block encodes 32 elements in 34 bytes:
//!   - 2 bytes: f16 scale `d`
//!   - 32 bytes: int8 quantized values `qs[0..31]`
//!
//! Dequantization: `x[i] = qs[i] * d`
//!
//! ## Kernel design
//!
//! - Grid: one block per output row (up to 248320 blocks)
//! - Block: 64 threads (2 warps) — optimal for 64 Q8_0 blocks per row
//!   (2048 / 32 = 64), so each thread handles exactly 1 block
//! - Uses dp4a integer dot products after on-the-fly Q8_1 quantization
//!   of the F32 input, for ~2x throughput vs naive F32 dot product
//! - Warp reduction via __shfl_xor_sync, then shared memory cross-warp
//!
//! ## Variants
//!
//! - `gemv_q8_0_f32`: F32 input, pure floating-point dot product (simple, correct)
//! - `gemv_q8_0_q8`: F32 input quantized to Q8_1 on-the-fly, dp4a integer dot (fast)
//! - `gemv_q8_0_from_tensor`: Candle Tensor API wrapper around gemv_q8_0_f32
//! - `gemv_q8_0_q8_from_tensor`: Candle Tensor API wrapper around gemv_q8_0_q8

use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Result, Tensor};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Q8_0 block: 34 bytes per 32 elements (2-byte f16 scale + 32 int8 values)
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_ELEMS: usize = 32;

/// Q8_1 block: 36 bytes per 32 elements (2-byte f16 d + 2-byte f16 s + 32 int8)
const Q8_1_BLOCK_BYTES: usize = 36;

// ---------------------------------------------------------------------------
// CUDA source: F32 Q8_0 GEMV (simple, correct)
// ---------------------------------------------------------------------------

const Q8_0_F32_GEMV_SRC: &str = r#"
// Software f16 <-> f32 conversion (NVRTC-safe, no cuda_fp16.h needed)
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant =  h        & 0x3FFu;
    if (exp == 0u) {
        // Subnormal
        float r = (float)mant * (1.0f / (1024.0f * 16384.0f));
        return sign ? -r : r;
    } else if (exp == 31u) {
        // Inf/NaN
        unsigned int f32 = (sign << 31) | 0x7F800000u | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    } else {
        // Normal
        unsigned int f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    }
}

// Q8_0 GEMV: one block per output row, 64 threads (= 2 warps).
// With ncols=2048, nblocks_per_row=64, each thread handles exactly 1 Q8_0 block.
//
// Weight layout: row-major, each row = nblocks_per_row * 34 contiguous bytes.
// Block layout: [f16 d (2 bytes)] [int8 qs[32] (32 bytes)] = 34 bytes total.
extern "C" __global__ void gemv_q8_0_f32(
    const unsigned char* __restrict__ weights,  // Q8_0 [nrows, nblocks*34]
    const float*         __restrict__ input,    // F32 [ncols]
    float*               __restrict__ output,   // F32 [nrows]
    int                               ncols,    // number of columns (e.g. 2048)
    int                               nrows     // number of rows (e.g. 248320)
) {
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int nblocks = ncols / 32;
    const int bytes_per_row = nblocks * 34;

    const unsigned char* row_data = weights + (long long)row * (long long)bytes_per_row;

    float sum = 0.0f;

    // Each thread processes one or more Q8_0 blocks
    for (int b = tid; b < nblocks; b += blockDim.x) {
        const unsigned char* block = row_data + b * 34;

        // Read f16 scale (little-endian, 2 bytes at offset 0)
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        float d = f16_to_f32(d_bits);

        // 32 int8 quantized values at offset 2
        const signed char* qs = (const signed char*)(block + 2);

        // Dot product with F32 input
        float block_sum = 0.0f;
        int base = b * 32;

        // Unrolled by 8 for ILP
        #pragma unroll 8
        for (int j = 0; j < 32; j++) {
            block_sum += (float)qs[j] * input[base + j];
        }

        sum += d * block_sum;
    }

    // --- Reduction ---
    // Warp reduction via butterfly shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction via shared memory (2 warps)
    __shared__ float warp_sums[2];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid == 0) {
        output[row] = warp_sums[0] + warp_sums[1];
    }
}
"#;

// ---------------------------------------------------------------------------
// CUDA source: Q8_1+dp4a Q8_0 GEMV (fast path)
// ---------------------------------------------------------------------------

const Q8_0_Q8_GEMV_SRC: &str = r#"
// Software f16 <-> f32 conversion (NVRTC-safe)
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant =  h        & 0x3FFu;
    if (exp == 0u) {
        float r = (float)mant * (1.0f / (1024.0f * 16384.0f));
        return sign ? -r : r;
    } else if (exp == 31u) {
        unsigned int f32 = (sign << 31) | 0x7F800000u | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    } else {
        unsigned int f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    }
}

__device__ __forceinline__ unsigned short f32_to_f16_bits(float x) {
    unsigned int u = *(unsigned int*)&x;
    unsigned int sign = (u >> 31) & 1u;
    int exp_f32 = (int)((u >> 23) & 0xFFu) - 127;
    unsigned int mant_f32 = u & 0x7FFFFFu;
    unsigned short h;
    if (exp_f32 >= 16) h = (unsigned short)((sign << 15) | 0x7C00u);
    else if (exp_f32 < -14) h = (unsigned short)(sign << 15);
    else {
        int e16 = exp_f32 + 15;
        h = (unsigned short)((sign << 15) | ((unsigned int)e16 << 10) | (mant_f32 >> 13));
    }
    return h;
}

// Q8_1 block: {f16 d, f16 s, int8 qs[32]} = 36 bytes
// d = scale, s = d * sum(qs) (used for min correction in Q4/Q5, not needed for Q8_0)
struct block_q8_1 {
    unsigned short d_bits;
    unsigned short s_bits;
    signed char qs[32];
};

// Quantize one block of 32 f32 values to Q8_1
__device__ void quantize_q8_1_block(
    const float* __restrict__ x,
    struct block_q8_1* b
) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > amax) amax = ax;
    }
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    b->d_bits = f32_to_f16_bits(scale);

    int sum = 0;
    for (int i = 0; i < 32; i++) {
        float fq = x[i] * inv_scale;
        int q = (int)(fq + (fq >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        b->qs[i] = (signed char)q;
        sum += q;
    }
    b->s_bits = f32_to_f16_bits(scale * (float)sum);
}

// GPU-side quantization kernel: F32 -> Q8_1
extern "C" __global__ void quantize_f32_to_q8_1_q80(
    const float* __restrict__ input,
    unsigned char* __restrict__ q8_out,
    int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x; blk < n_blocks; blk += blockDim.x * gridDim.x) {
        quantize_q8_1_block(&input[blk * 32], (struct block_q8_1*)(q8_out + blk * 36));
    }
}

// Q8_0 x Q8_1 dot product for one Q8_0 block against one Q8_1 block.
// Uses dp4a (int8 dot product with int32 accumulator) for throughput.
//
// Q8_0 block: [f16 d_w (2B)] [int8 qs_w[32] (32B)] = 34B
// Q8_1 block: [f16 d_x (2B)] [f16 s_x (2B)] [int8 qs_x[32] (32B)] = 36B
//
// dot = d_w * d_x * sum_i(qs_w[i] * qs_x[i])
// (Q8_0 has no min term, so the Q8_1 s field is unused)
__device__ __forceinline__ float dot_q8_0_q8_1(
    const unsigned char* __restrict__ bq0,  // Q8_0 block (34 bytes)
    const struct block_q8_1* __restrict__ bq1  // Q8_1 block (36 bytes)
) {
    // Weight scale
    unsigned short d_w_bits = (unsigned short)bq0[0] | ((unsigned short)bq0[1] << 8);

    // Weight quantized values as int32 (4 int8 packed)
    const int* qs_w = (const int*)(bq0 + 2);

    // Input quantized values as int32
    const int* qs_x = (const int*)(bq1->qs);

    // dp4a: 8 iterations x 4 int8 multiplies = 32 elements
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sumi = __dp4a(qs_w[i], qs_x[i], sumi);
    }

    // d_w * d_x * sumi
    float d_w = f16_to_f32(d_w_bits);
    float d_x = f16_to_f32(bq1->d_bits);

    return d_w * d_x * (float)sumi;
}

// Q8_0 GEMV with dp4a: one block per output row, 64 threads (2 warps).
// Expects pre-quantized Q8_1 input.
//
// For ncols=2048: nblocks=64, 64 threads => 1 block/thread (perfect).
extern "C" __global__ void gemv_q8_0_q8(
    const unsigned char* __restrict__ weights,  // Q8_0 [nrows, nblocks*34]
    float*               __restrict__ output,   // F32 [nrows]
    int                               ncols,    // e.g. 2048
    int                               nrows,    // e.g. 248320
    const unsigned char* __restrict__ q8_input  // Q8_1 quantized input [nblocks*36]
) {
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int nblocks = ncols / 32;
    const int bytes_per_row = nblocks * 34;

    const unsigned char* row_data = weights + (long long)row * (long long)bytes_per_row;
    const struct block_q8_1* q8_blocks = (const struct block_q8_1*)q8_input;

    float sum = 0.0f;

    for (int b = tid; b < nblocks; b += blockDim.x) {
        sum += dot_q8_0_q8_1(row_data + b * 34, &q8_blocks[b]);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction (2 warps)
    __shared__ float warp_sums[2];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid == 0) {
        output[row] = warp_sums[0] + warp_sums[1];
    }
}
"#;

// ---------------------------------------------------------------------------
// PTX caches
// ---------------------------------------------------------------------------

static Q8_0_F32_PTX: OnceLock<String> = OnceLock::new();
fn get_q8_0_f32_ptx() -> &'static str {
    super::nvrtc_compile::compile_and_cache(Q8_0_F32_GEMV_SRC, &Q8_0_F32_PTX)
}

static Q8_0_Q8_PTX: OnceLock<String> = OnceLock::new();
fn get_q8_0_q8_ptx() -> &'static str {
    super::nvrtc_compile::compile_and_cache(Q8_0_Q8_GEMV_SRC, &Q8_0_Q8_PTX)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_cuda_dev(device: &Device) -> Result<&CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d),
        _ => candle_core::bail!("Q8_0 GEMV requires CUDA device"),
    }
}

/// Calculate Q8_0 row stride in bytes.
#[inline]
fn q8_0_row_bytes(ncols: usize) -> usize {
    (ncols / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
}

/// Calculate Q8_1 buffer size for a vector of `n` elements.
#[inline]
fn q8_1_buf_bytes(n: usize) -> usize {
    (n / Q8_0_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// Public API: Low-level (CudaView / CudaSlice)
// ---------------------------------------------------------------------------

/// Q8_0 GEMV with F32 input: weights[nrows, ncols] @ input[ncols] -> output[nrows].
///
/// Uses pure floating-point dot product (no quantization of input).
/// Simple and correct; use `gemv_q8_0_q8` for higher throughput.
///
/// `weights`: raw Q8_0 bytes (nrows * ncols/32 * 34 bytes).
/// `input`: F32 input vector [ncols].
/// `output`: pre-allocated F32 output buffer [nrows].
pub fn gemv_q8_0_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(ncols % Q8_0_BLOCK_ELEMS == 0, "ncols must be multiple of 32");

    let (func, stream) = super::nvrtc_compile::get_or_load_func(
        dev, "gemv_q8_0_f32", "chimere_q8_0_gemv_v1",
        Q8_0_F32_GEMV_SRC, &Q8_0_F32_PTX,
    )?;

    let ncols_i32 = ncols as i32;
    let nrows_i32 = nrows as i32;
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 2 * 4, // 2 warp sums
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(weights);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&ncols_i32);
    builder.arg(&nrows_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("gemv_q8_0_f32 launch: {e}")))?;

    Ok(())
}

/// Same as `gemv_q8_0_f32` but accepts `&CudaSlice` instead of `&CudaView`.
pub fn gemv_q8_0_f32_slices(
    weights: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let w_view = weights.slice(0..);
    let i_view = input.slice(0..);
    gemv_q8_0_f32(&w_view, &i_view, output, nrows, ncols, dev)
}

/// Q8_0 GEMV with dp4a: quantizes input to Q8_1 on-the-fly, then uses
/// integer dot products for ~2x throughput over the F32 path.
///
/// `weights`: raw Q8_0 bytes (nrows * ncols/32 * 34 bytes).
/// `input`: F32 input vector [ncols].
/// `output`: pre-allocated F32 output buffer [nrows].
pub fn gemv_q8_0_q8(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(ncols % Q8_0_BLOCK_ELEMS == 0, "ncols must be multiple of 32");

    let stream = dev.cuda_stream();

    // Step 1: Quantize input F32 -> Q8_1 on GPU
    let n_blocks = ncols / Q8_0_BLOCK_ELEMS;
    let q8_bytes = q8_1_buf_bytes(ncols);
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8_1 buf: {e}")))?;

    {
        let ptx = get_q8_0_q8_ptx();
        let quant_func = dev.get_or_load_custom_func(
            "quantize_f32_to_q8_1_q80",
            "chimere_q8_0_q8_gemv_v1",
            ptx,
        )?;
        let grid = ((n_blocks as u32) + 255) / 256;
        let n_blocks_i32 = n_blocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = quant_func.builder();
        builder.arg(input);
        builder.arg(&mut q8_buf);
        builder.arg(&n_blocks_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1_q80 launch: {e}")))?;
    }

    // Step 2: dp4a GEMV
    {
        let ptx = get_q8_0_q8_ptx();
        let gemv_func = dev.get_or_load_custom_func(
            "gemv_q8_0_q8",
            "chimere_q8_0_q8_gemv_v1",
            ptx,
        )?;
        let ncols_i32 = ncols as i32;
        let nrows_i32 = nrows as i32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 2 * 4, // 2 warp sums
        };
        let mut builder = gemv_func.builder();
        builder.arg(weights);
        builder.arg(output);
        builder.arg(&ncols_i32);
        builder.arg(&nrows_i32);
        builder.arg(&q8_buf);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("gemv_q8_0_q8 launch: {e}")))?;
    }

    Ok(())
}

/// Same as `gemv_q8_0_q8` but accepts `&CudaSlice` instead of `&CudaView`.
pub fn gemv_q8_0_q8_slices(
    weights: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let w_view = weights.slice(0..);
    let i_view = input.slice(0..);
    gemv_q8_0_q8(&w_view, &i_view, output, nrows, ncols, dev)
}

/// Q8_0 GEMV with dp4a using a pre-quantized Q8_1 input buffer.
///
/// This avoids re-quantizing the input when it is reused across multiple
/// GEMV calls (e.g. lm_head + MTP head sharing the same hidden state).
pub fn gemv_q8_0_q8_precomputed(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(ncols % Q8_0_BLOCK_ELEMS == 0, "ncols must be multiple of 32");

    let ptx = get_q8_0_q8_ptx();
    let gemv_func = dev.get_or_load_custom_func(
        "gemv_q8_0_q8",
        "chimere_q8_0_q8_gemv_v1",
        ptx,
    )?;
    let ncols_i32 = ncols as i32;
    let nrows_i32 = nrows as i32;
    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 2 * 4,
    };
    let mut builder = gemv_func.builder();
    builder.arg(weights);
    builder.arg(output);
    builder.arg(&ncols_i32);
    builder.arg(&nrows_i32);
    builder.arg(q8_input);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("gemv_q8_0_q8_precomputed launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: Candle Tensor wrappers
// ---------------------------------------------------------------------------

/// Q8_0 GEMV with F32 input from Candle Tensors.
///
/// `weights_raw`: flat U8 Tensor of raw Q8_0 bytes on GPU.
/// `input`: 1-D F32 Tensor [ncols].
/// Returns Tensor [nrows].
pub fn gemv_q8_0_f32_from_tensor(
    weights_raw: &Tensor,
    input: &Tensor,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<Tensor> {
    use candle_core::Storage;
    let dev = get_cuda_dev(device)?;

    let wt = weights_raw.contiguous()?;
    let it = input.contiguous()?;

    let (w_stor, w_lay) = wt.storage_and_layout();
    let w_cuda = match &*w_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("weights not on CUDA"),
    };
    let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(w_lay.start_offset()..);

    let (i_stor, i_lay) = it.storage_and_layout();
    let i_cuda = match &*i_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("input not on CUDA"),
    };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(i_lay.start_offset()..);

    let stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(nrows)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8_0 output: {e}")))?;

    gemv_q8_0_f32(&w_view, &i_view, &mut output, nrows, ncols, dev)?;

    drop(w_stor);
    drop(i_stor);

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[nrows]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Q8_0 GEMV with dp4a from Candle Tensors (fast path).
///
/// Quantizes the F32 input to Q8_1 on GPU, then uses dp4a integer dot products.
/// `weights_raw`: flat U8 Tensor of raw Q8_0 bytes on GPU.
/// `input`: 1-D F32 Tensor [ncols].
/// Returns Tensor [nrows].
pub fn gemv_q8_0_q8_from_tensor(
    weights_raw: &Tensor,
    input: &Tensor,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<Tensor> {
    use candle_core::Storage;
    let dev = get_cuda_dev(device)?;

    let wt = weights_raw.contiguous()?;
    let it = input.contiguous()?;

    let (w_stor, w_lay) = wt.storage_and_layout();
    let w_cuda = match &*w_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("weights not on CUDA"),
    };
    let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(w_lay.start_offset()..);

    let (i_stor, i_lay) = it.storage_and_layout();
    let i_cuda = match &*i_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("input not on CUDA"),
    };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(i_lay.start_offset()..);

    let stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(nrows)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8_0_q8 output: {e}")))?;

    gemv_q8_0_q8(&w_view, &i_view, &mut output, nrows, ncols, dev)?;

    drop(w_stor);
    drop(i_stor);

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[nrows]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Quantize F32 input to Q8_1 on GPU (for use with `gemv_q8_0_q8_precomputed`).
///
/// Returns a CudaSlice<u8> containing the Q8_1-encoded input.
pub fn quantize_input_q8_1(
    input: &CudaView<'_, f32>,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<CudaSlice<u8>> {
    assert!(ncols % Q8_0_BLOCK_ELEMS == 0, "ncols must be multiple of 32");

    let n_blocks = ncols / Q8_0_BLOCK_ELEMS;
    let q8_bytes = q8_1_buf_bytes(ncols);
    let stream = dev.cuda_stream();
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8_1 buf: {e}")))?;

    let ptx = get_q8_0_q8_ptx();
    let quant_func = dev.get_or_load_custom_func(
        "quantize_f32_to_q8_1_q80",
        "chimere_q8_0_q8_gemv_v1",
        ptx,
    )?;
    let grid = ((n_blocks as u32) + 255) / 256;
    let n_blocks_i32 = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = quant_func.builder();
    builder.arg(input);
    builder.arg(&mut q8_buf);
    builder.arg(&n_blocks_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1 launch: {e}")))?;

    Ok(q8_buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify Q8_0 row byte calculation.
    #[test]
    fn test_q8_0_row_bytes() {
        assert_eq!(q8_0_row_bytes(2048), 2048 / 32 * 34);  // 64 * 34 = 2176
        assert_eq!(q8_0_row_bytes(32), 34);
    }

    /// Verify Q8_1 buffer size calculation.
    #[test]
    fn test_q8_1_buf_bytes() {
        assert_eq!(q8_1_buf_bytes(2048), 2048 / 32 * 36);  // 64 * 36 = 2304
        assert_eq!(q8_1_buf_bytes(32), 36);
    }
}
