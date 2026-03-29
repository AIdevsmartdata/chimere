//! Q5_K GEMV kernels for chimere-deltanet.
//!
//! Two variants:
//!   1. F32 input: `gemv_q5k` — dequantize Q5_K on the fly, dot with F32 input.
//!   2. Q8_1+dp4a: `gemv_q5k_q8` — quantize input to Q8_1, use integer dp4a.
//!
//! Both produce one output row per thread block (grid = out_features blocks).

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::{Device, Result, Tensor};
use std::sync::OnceLock;

// -----------------------------------------------------------------------
// CUDA source: F32 Q5_K GEMV
// -----------------------------------------------------------------------

const Q5K_GEMV_SRC: &str = r#"
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

__device__ void get_scale_min_k4(
    int j, const unsigned char* scales, float* sc_out, float* mn_out
) {
    if (j < 4) {
        *sc_out = (float)(scales[j]   & 63);
        *mn_out = (float)(scales[j+4] & 63);
    } else {
        *sc_out = (float)((scales[j+4] & 0xF)  | ((scales[j-4] >> 6) << 4));
        *mn_out = (float)((scales[j+4] >> 4)    | ((scales[j-0] >> 6) << 4));
    }
}

extern "C" __global__ void gemv_q5k(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    int                               in_features
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int BLOCK_ELEMS = 256;
    const int BLOCK_BYTES = 176;
    const int n_blocks = in_features / BLOCK_ELEMS;
    const unsigned char* row_data = weights + (long long)row * (long long)n_blocks * BLOCK_BYTES;
    float sum = 0.0f;
    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const unsigned char* block = row_data + block_idx * BLOCK_BYTES;
        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        const unsigned char* scales = block + 4;
        const unsigned char* qh     = block + 16;
        const unsigned char* qs     = block + 48;
        int base_elem = block_idx * BLOCK_ELEMS;
        for (int j = 0; j < 8; j++) {
            float sc, mn;
            get_scale_min_k4(j, scales, &sc, &mn);
            float d_sc = d * sc;
            float d_mn = dmin * mn;
            for (int k = 0; k < 32; k++) {
                int global_elem = base_elem + j * 32 + k;
                int qs_byte_idx = (j / 2) * 32 + k;
                int low4 = (j & 1) ? (((int)qs[qs_byte_idx] >> 4) & 0xF)
                                    : ( (int)qs[qs_byte_idx]       & 0xF);
                int high1 = ((int)qh[k] >> j) & 1;
                int q5    = low4 | (high1 << 4);
                float w = d_sc * (float)q5 - d_mn;
                sum += w * input[global_elem];
            }
        }
    }
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[row] = sdata[0];
}
"#;

// -----------------------------------------------------------------------
// CUDA source: Q8_1+dp4a Q5_K GEMV
// -----------------------------------------------------------------------

const Q5K_Q8_GEMV_SRC: &str = r#"
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant =  h        & 0x3FFu;
    if (exp == 0u) { float r = (float)mant * (1.0f / (1024.0f * 16384.0f)); return sign ? -r : r; }
    else if (exp == 31u) { unsigned int f32 = (sign << 31) | 0x7F800000u | (mant << 13); float r; *(unsigned int*)&r = f32; return r; }
    else { unsigned int f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13); float r; *(unsigned int*)&r = f32; return r; }
}
__device__ __forceinline__ unsigned short f32_to_f16_bits(float x) {
    unsigned int u = *(unsigned int*)&x;
    unsigned int sign = (u >> 31) & 1u;
    int exp_f32 = (int)((u >> 23) & 0xFFu) - 127;
    unsigned int mant_f32 = u & 0x7FFFFFu;
    unsigned short h;
    if (exp_f32 >= 16) h = (unsigned short)((sign << 15) | 0x7C00u);
    else if (exp_f32 < -14) h = (unsigned short)(sign << 15);
    else { int e16 = exp_f32 + 15; h = (unsigned short)((sign << 15) | ((unsigned int)e16 << 10) | (mant_f32 >> 13)); }
    return h;
}
struct block_q8_1_q5k { unsigned short d_bits; unsigned short s_bits; signed char qs[32]; };
__device__ void quantize_q8_1_block_q5k(const float* __restrict__ x, struct block_q8_1_q5k* b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) { float ax = x[i] < 0.0f ? -x[i] : x[i]; if (ax > amax) amax = ax; }
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    b->d_bits = f32_to_f16_bits(scale);
    int sum = 0;
    for (int i = 0; i < 32; i++) {
        float fq = x[i] * inv_scale;
        int q = (int)(fq + (fq >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127; if (q < -127) q = -127;
        b->qs[i] = (signed char)q; sum += q;
    }
    b->s_bits = f32_to_f16_bits(scale * (float)sum);
}
extern "C" __global__ void quantize_f32_to_q8_1_q5k(
    const float* __restrict__ input, unsigned char* __restrict__ q8_out, int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x; blk < n_blocks; blk += blockDim.x * gridDim.x)
        quantize_q8_1_block_q5k(&input[blk * 32], (struct block_q8_1_q5k*)(q8_out + blk * 36));
}
__device__ __forceinline__ float dot_q5k_q8_subblock(
    const unsigned char* __restrict__ blk, const struct block_q8_1_q5k* __restrict__ q8b, int sg
) {
    unsigned short d_bits    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
    unsigned short dmin_bits = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
    float d5    = f16_to_f32(d_bits);
    float dmin5 = f16_to_f32(dmin_bits);
    const unsigned char* scales = blk + 4;
    float sc_f, mn_f;
    if (sg < 4) { sc_f = (float)(scales[sg] & 63); mn_f = (float)(scales[sg+4] & 63); }
    else { sc_f = (float)((scales[sg+4] & 0xF) | ((scales[sg-4] >> 6) << 4)); mn_f = (float)((scales[sg+4] >> 4) | ((scales[sg] >> 6) << 4)); }
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    float d_q8 = f16_to_f32(q8b->d_bits);
    float s_q8 = f16_to_f32(q8b->s_bits);
    const int* q8_qs_int = (const int*)(q8b->qs);
    int sumi = 0;
    for (int l = 0; l < 8; l++) {
        int k_base = l * 4; unsigned int v = 0;
        for (int b = 0; b < 4; b++) {
            int k = k_base + b; int qs_byte_idx = (sg / 2) * 32 + k;
            int low4 = (sg & 1) ? (((int)qs[qs_byte_idx] >> 4) & 0xF) : ((int)qs[qs_byte_idx] & 0xF);
            int high1 = ((int)qh[k] >> sg) & 1; int q5 = low4 | (high1 << 4);
            v |= ((unsigned int)(q5 & 0xFF)) << (b * 8);
        }
        sumi = __dp4a((int)v, q8_qs_int[l], sumi);
    }
    return d5 * sc_f * d_q8 * (float)sumi - dmin5 * mn_f * s_q8;
}
extern "C" __global__ void gemv_q5k_q8(
    const unsigned char* __restrict__ weights, float* __restrict__ output,
    int cols, const unsigned char* __restrict__ q8_input
) {
    const int row = blockIdx.x; const int tid = threadIdx.x;
    const int warp_id = tid >> 5; const int lane_id = tid & 31;
    const int n_q5k_blocks = cols >> 8; const int Q5K_BYTES = 176;
    extern __shared__ char smem[]; float* warp_sums = (float*)smem;
    const struct block_q8_1_q5k* q8_blocks = (const struct block_q8_1_q5k*)q8_input;
    const unsigned char* row_weights = weights + (long long)row * (long long)n_q5k_blocks * Q5K_BYTES;
    float sum = 0.0f;
    const int total_subblocks = n_q5k_blocks * 8;
    for (int sg_idx = tid; sg_idx < total_subblocks; sg_idx += 128) {
        int kbx = sg_idx >> 3; int sg = sg_idx & 7;
        sum += dot_q5k_q8_subblock(row_weights + kbx * Q5K_BYTES, &q8_blocks[kbx * 8 + sg], sg);
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (tid == 0) { output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]; }
}

// ---------------------------------------------------------------------------
// Dual GEMV: blockIdx.y selects projection (0 = A, 1 = B).
// Both projections share the same Q8_1 quantized input — avoids redundant
// quantization and saves 1 kernel launch per GDN layer.
// Grid: (max(rows_a, rows_b), 2, 1),  Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_q5k_q8_dual(
    const unsigned char* __restrict__ weights_a,   // Q5_K [rows_a, cols]
    const unsigned char* __restrict__ weights_b,   // Q5_K [rows_b, cols]
    float*               __restrict__ output_a,    // [rows_a]
    float*               __restrict__ output_b,    // [rows_b]
    int                               cols,
    int                               rows_a,
    int                               rows_b,
    const unsigned char* __restrict__ q8_input     // shared Q8_1 quantized input
) {
    const int proj = blockIdx.y;         // 0 = A (qkv), 1 = B (gate)
    const int row  = blockIdx.x;
    const int rows = (proj == 0) ? rows_a : rows_b;
    if (row >= rows) return;

    const unsigned char* weights = (proj == 0) ? weights_a : weights_b;
    float* output = (proj == 0) ? output_a : output_b;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_q5k_blocks = cols >> 8;
    const int Q5K_BYTES = 176;
    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;
    const struct block_q8_1_q5k* q8_blocks = (const struct block_q8_1_q5k*)q8_input;
    const unsigned char* row_weights = weights + (long long)row * (long long)n_q5k_blocks * Q5K_BYTES;
    float sum = 0.0f;
    const int total_subblocks = n_q5k_blocks * 8;
    for (int sg_idx = tid; sg_idx < total_subblocks; sg_idx += 128) {
        int kbx = sg_idx >> 3;
        int sg  = sg_idx & 7;
        sum += dot_q5k_q8_subblock(row_weights + kbx * Q5K_BYTES, &q8_blocks[kbx * 8 + sg], sg);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    }
}
"#;

// -----------------------------------------------------------------------
// PTX caches
// -----------------------------------------------------------------------

static Q5K_F32_PTX: OnceLock<String> = OnceLock::new();
static Q5K_Q8_PTX: OnceLock<String> = OnceLock::new();

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn get_cuda_dev(device: &Device) -> Result<&candle_core::cuda_backend::CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d),
        _ => candle_core::bail!("Q5_K GEMV requires CUDA device"),
    }
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Q5_K GEMV with F32 input using CudaSlice pointers.
/// Returns a new Tensor of shape [out_features].
pub fn gemv_q5k_fused(
    weights: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    let dev = get_cuda_dev(device)?;
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "gemv_q5k", "chimere_q5k_gemv_v2", Q5K_GEMV_SRC, &Q5K_F32_PTX,
    )?;

    let stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(out_features)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q5k output: {e}")))?;

    let in_f_i32 = in_features as i32;
    let cfg = LaunchConfig {
        grid_dim: (out_features as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(weights);
    builder.arg(input);
    builder.arg(&mut output);
    builder.arg(&in_f_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("gemv_q5k launch: {e}")))?;

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[out_features]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Q5_K GEMV with F32 input from Candle Tensors.
/// `weights_raw`: flat U8 Tensor of raw Q5_K bytes on GPU.
/// `input`: 1D F32 Tensor [in_features]. Returns Tensor [out_features].
pub fn gemv_q5k_from_tensor(
    weights_raw: &Tensor,
    input: &Tensor,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    use candle_core::Storage;
    let wt = weights_raw.contiguous()?;
    let it = input.contiguous()?;

    let (w_stor, w_lay) = wt.storage_and_layout();
    let w_cuda = match &*w_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(w_lay.start_offset()..);

    let (i_stor, i_lay) = it.storage_and_layout();
    let i_cuda = match &*i_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(i_lay.start_offset()..);

    let dev = get_cuda_dev(device)?;
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "gemv_q5k", "chimere_q5k_gemv_v2", Q5K_GEMV_SRC, &Q5K_F32_PTX,
    )?;

    let stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(out_features)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q5k output: {e}")))?;

    let in_f_i32 = in_features as i32;
    let cfg = LaunchConfig {
        grid_dim: (out_features as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = _stream.launch_builder(&func);
    builder.arg(&w_view);
    builder.arg(&i_view);
    builder.arg(&mut output);
    builder.arg(&in_f_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("gemv_q5k_from_tensor launch: {e}")))?;

    drop(w_stor); drop(i_stor);

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[out_features]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Q5_K GEMV with Q8_1+dp4a quantized input from Candle Tensors.
/// Quantizes F32 `input` to Q8_1 on GPU, then uses dp4a integer dot products.
/// Returns Tensor [out_features].
pub fn gemv_q5k_q8_from_tensor(
    weights_raw: &Tensor,
    input: &Tensor,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    use candle_core::Storage;
    let dev = get_cuda_dev(device)?;

    let wt = weights_raw.contiguous()?;
    let it = input.contiguous()?;

    let (w_stor, w_lay) = wt.storage_and_layout();
    let w_cuda = match &*w_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(w_lay.start_offset()..);

    let (i_stor, i_lay) = it.storage_and_layout();
    let i_cuda = match &*i_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(i_lay.start_offset()..);

    // Quantize input to Q8_1 format on GPU
    let n_blocks = in_features / 32;
    let q8_bytes = n_blocks * 36;
    let stream = dev.cuda_stream();
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8 buf: {e}")))?;

    {
        let (quant_func, _stream) = super::nvrtc_compile::get_or_load_func(
            dev, "quantize_f32_to_q8_1_q5k", "chimere_q5k_q8_gemv_v1",
            Q5K_Q8_GEMV_SRC, &Q5K_Q8_PTX,
        )?;
        let grid = ((n_blocks as u32) + 255) / 256;
        let n_blocks_i32 = n_blocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = _stream.launch_builder(&quant_func);
        builder.arg(&i_view);
        builder.arg(&mut q8_buf);
        builder.arg(&n_blocks_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_q5k launch: {e}")))?;
    }

    // Run the dp4a GEMV kernel
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(out_features)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q5k_q8 output: {e}")))?;

    {
        let (gemv_func, _stream) = super::nvrtc_compile::get_or_load_func(
            dev, "gemv_q5k_q8", "chimere_q5k_q8_gemv_v1",
            Q5K_Q8_GEMV_SRC, &Q5K_Q8_PTX,
        )?;
        let cols_i32 = in_features as i32;
        let cfg = LaunchConfig {
            grid_dim: (out_features as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 4 * 4, // 4 warp sums
        };
        let mut builder = _stream.launch_builder(&gemv_func);
        builder.arg(&w_view);
        builder.arg(&mut output);
        builder.arg(&cols_i32);
        builder.arg(&q8_buf);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("gemv_q5k_q8 launch: {e}")))?;
    }

    drop(w_stor); drop(i_stor);

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[out_features]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Dual Q5_K GEMV: runs two Q5_K-times-F32-input projections in a single kernel
/// launch, sharing one Q8_1 quantization pass on the input.
///
/// Returns `(output_a, output_b)` where each is a 1-D Tensor `[out_features_*]`.
///
/// This saves one kernel launch (4-5 us) per GDN layer and avoids reading +
/// quantizing the input twice.
pub fn gemv_q5k_q8_dual_from_tensor(
    weights_a_raw: &Tensor,    // Q5_K raw bytes, projection A (e.g. qkv [8192, 2048])
    weights_b_raw: &Tensor,    // Q5_K raw bytes, projection B (e.g. gate [4096, 2048])
    input: &Tensor,            // F32 [in_features]
    out_features_a: usize,     // rows of projection A (e.g. 8192)
    out_features_b: usize,     // rows of projection B (e.g. 4096)
    in_features: usize,        // columns, shared (e.g. 2048)
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    use candle_core::Storage;
    let dev = get_cuda_dev(device)?;

    let wa = weights_a_raw.contiguous()?;
    let wb = weights_b_raw.contiguous()?;
    let it = input.contiguous()?;

    let (wa_stor, wa_lay) = wa.storage_and_layout();
    let wa_cuda = match &*wa_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let wa_view = wa_cuda.as_cuda_slice::<u8>()?.slice(wa_lay.start_offset()..);

    let (wb_stor, wb_lay) = wb.storage_and_layout();
    let wb_cuda = match &*wb_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let wb_view = wb_cuda.as_cuda_slice::<u8>()?.slice(wb_lay.start_offset()..);

    let (i_stor, i_lay) = it.storage_and_layout();
    let i_cuda = match &*i_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(i_lay.start_offset()..);

    let stream = dev.cuda_stream();

    // --- Step 1: Quantize input to Q8_1 (shared, done once) ---
    let n_blocks = in_features / 32;
    let q8_bytes = n_blocks * 36;
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("alloc q8 dual buf: {e}")))?;

    {
        let (quant_func, _stream) = super::nvrtc_compile::get_or_load_func(
            dev, "quantize_f32_to_q8_1_q5k", "chimere_q5k_q8_gemv_v1",
            Q5K_Q8_GEMV_SRC, &Q5K_Q8_PTX,
        )?;
        let grid = ((n_blocks as u32) + 255) / 256;
        let n_blocks_i32 = n_blocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = _stream.launch_builder(&quant_func);
        builder.arg(&i_view);
        builder.arg(&mut q8_buf);
        builder.arg(&n_blocks_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("quantize_q8 dual launch: {e}")))?;
    }

    // --- Step 2: Dual GEMV kernel ---
    let mut output_a: CudaSlice<f32> = stream
        .alloc_zeros(out_features_a)
        .map_err(|e| candle_core::Error::Msg(format!("alloc dual output_a: {e}")))?;
    let mut output_b: CudaSlice<f32> = stream
        .alloc_zeros(out_features_b)
        .map_err(|e| candle_core::Error::Msg(format!("alloc dual output_b: {e}")))?;

    {
        let (dual_func, _stream) = super::nvrtc_compile::get_or_load_func(
            dev, "gemv_q5k_q8_dual", "chimere_q5k_q8_gemv_v1",
            Q5K_Q8_GEMV_SRC, &Q5K_Q8_PTX,
        )?;
        let max_rows = out_features_a.max(out_features_b) as u32;
        let cols_i32 = in_features as i32;
        let rows_a_i32 = out_features_a as i32;
        let rows_b_i32 = out_features_b as i32;
        let cfg = LaunchConfig {
            grid_dim: (max_rows, 2, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 4 * 4, // 4 warp sums
        };
        let mut builder = _stream.launch_builder(&dual_func);
        builder.arg(&wa_view);
        builder.arg(&wb_view);
        builder.arg(&mut output_a);
        builder.arg(&mut output_b);
        builder.arg(&cols_i32);
        builder.arg(&rows_a_i32);
        builder.arg(&rows_b_i32);
        builder.arg(&q8_buf);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("gemv_q5k_q8_dual launch: {e}")))?;
    }

    drop(wa_stor); drop(wb_stor); drop(i_stor);

    let stor_a = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output_a, dev.clone());
    let t_a = Tensor::from_storage(
        candle_core::Storage::Cuda(stor_a),
        candle_core::Shape::from_dims(&[out_features_a]),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let stor_b = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output_b, dev.clone());
    let t_b = Tensor::from_storage(
        candle_core::Storage::Cuda(stor_b),
        candle_core::Shape::from_dims(&[out_features_b]),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok((t_a, t_b))
}
