//! ggml-compatible Q5_K MMVQ kernel wrapper.
//!
//! Calls the ggml `vec_dot_q5_K_q8_1` kernel compiled in `chimere_kernels.cubin`
//! by the other agent. This gives numerical parity with llama.cpp's Q5_K GEMV.
//!
//! ## Kernel signatures (in cubin)
//!
//! - `ggml_quantize_q8_1(const float* x, void* vy, int64_t kx, int64_t kx_padded)`
//! - `ggml_mul_mat_vec_q5_K_q8_1(const void* vx, const void* vy, float* dst,
//!       int ncols_x, int nrows_x, int nrows_y, int nrows_dst)`
//!
//! ## Toggle
//!
//! `CHIMERE_GGML_Q5K=1` to use this path instead of Candle's built-in kernel.
//!
//! ## Block sizes
//!
//! - Q5_K: 176 bytes per 256 elements (QK_K=256)
//! - Q8_1:  36 bytes per 32 elements  (QK8_1=32)
//!
//! ## Launch config
//!
//! - quantize_q8_1: grid=((ncols_padded+255)/256, 1, 1), block=(256, 1, 1)
//! - mul_mat_vec:   grid=(nrows, 1, 1), block=(32, 4, 1), shared_mem=384

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Q8_1: 36 bytes per 32 elements.
const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_ELEMS: usize = 32;

/// ggml MMVQ shared memory size (384 bytes, from ggml source).
const MMVQ_SHARED_MEM: u32 = 384;

/// Padding alignment matching ggml's MATRIX_ROW_PADDING.
const MATRIX_ROW_PADDING: usize = 512;

fn pad(n: usize, align: usize) -> usize {
    (n + align - 1) / align * align
}

// ---------------------------------------------------------------------------
// Kernel names (must match extern "C" in chimere_kernels.cu)
// ---------------------------------------------------------------------------

const QUANTIZE_FUNC: &str = "ggml_quantize_q8_1";
const MMVQ_FUNC: &str = "ggml_mul_mat_vec_q5_K_q8_1";

/// Dummy NVRTC source — these kernels require cuda_fp16.h so they MUST come
/// from the cubin. If the cubin is missing, we emit a clear error rather than
/// trying (and failing) to compile via NVRTC.
const DUMMY_SRC: &str = r#"
// This kernel is only available via the pre-compiled cubin.
// It requires cuda_fp16.h which is not available in NVRTC.
extern "C" __global__ void ggml_quantize_q8_1(void) {}
extern "C" __global__ void ggml_mul_mat_vec_q5_K_q8_1(void) {}
"#;

static DUMMY_PTX: OnceLock<String> = OnceLock::new();

/// Module name for NVRTC fallback (should never actually be used).
const MODULE_NAME: &str = "chimere_ggml_q5k_mmvq_v1";

// ---------------------------------------------------------------------------
// Kernel loader
// ---------------------------------------------------------------------------

/// Load a ggml Q5_K kernel function from the cubin.
///
/// Falls back to NVRTC only for compilation (but the dummy source will produce
/// non-functional kernels — the cubin path is required for correctness).
fn load_func(
    dev: &CudaDevice,
    fn_name: &str,
) -> Result<(candle_core::cuda_backend::cudarc::driver::CudaFunction,
            std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>)> {
    let result = super::nvrtc_compile::get_or_load_func(
        dev, fn_name, MODULE_NAME, DUMMY_SRC, &DUMMY_PTX,
    );

    // If the function was not found in the cubin, give a clear error.
    match &result {
        Ok(_) => {}
        Err(_) => {
            if !super::cubin_loader::has_cubin() {
                return Err(candle_core::Error::Msg(
                    format!(
                        "ggml Q5_K kernel '{fn_name}' requires the pre-compiled cubin \
                         (chimere_kernels.cubin). Build with nvcc or unset CHIMERE_GGML_Q5K."
                    ),
                ));
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Pre-allocated buffers
// ---------------------------------------------------------------------------

/// Pre-allocated buffers for ggml Q5_K MMVQ operations.
///
/// One instance is shared across all layers (buffers are reused serially).
/// Allocate once at model init with the maximum dimensions.
pub struct GgmlQ5KBuffers {
    /// Q8_1 quantized input: `pad(ncols, 512) * 36 / 32` bytes.
    pub q8_input: CudaSlice<u8>,
    /// Output buffer: `max_nrows` floats.
    pub output: CudaSlice<f32>,
}

impl GgmlQ5KBuffers {
    /// Allocate buffers for the given max dimensions.
    ///
    /// `max_ncols`: maximum input vector length (e.g. 2048 for hidden_size).
    /// `max_nrows`: maximum output rows (e.g. 8192 for QKV projection).
    pub fn new(max_ncols: usize, max_nrows: usize, dev: &CudaDevice) -> Result<Self> {
        let ncols_padded = pad(max_ncols, MATRIX_ROW_PADDING);
        // Q8_1: 36 bytes per 32 elements
        let q8_size = (ncols_padded / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
        let q8_input = dev
            .alloc_zeros::<u8>(q8_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ggml q8_input: {e}")))?;
        let output = dev
            .alloc_zeros::<f32>(max_nrows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ggml output: {e}")))?;
        Ok(Self { q8_input, output })
    }
}

// ---------------------------------------------------------------------------
// Public API: Q8_1 quantization (ggml layout)
// ---------------------------------------------------------------------------

/// Quantize f32 input to Q8_1 format on GPU using ggml's kernel.
///
/// Writes to `q8_output`. The buffer must be at least
/// `pad(ncols, 512) * 36 / 32` bytes.
///
/// The ggml `quantize_q8_1` kernel signature is:
///   `(const float* x, void* vy, int64_t kx, int64_t kx_padded)`
///
/// where `kx` is the real column count and `kx_padded` is padded to 512.
pub fn ggml_quantize_q8_1(
    input: &CudaView<'_, f32>,
    q8_output: &mut CudaSlice<u8>,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // Kernel signature: (const float* x, void* vy, int ncols)
    // Each block quantizes 32 elements (1 Q8_1 block). Grid = ncols/32.
    let nblocks_q8 = (ncols + Q8_1_BLOCK_ELEMS - 1) / Q8_1_BLOCK_ELEMS;

    let (func, stream) = load_func(dev, QUANTIZE_FUNC)?;

    let cfg = LaunchConfig {
        grid_dim: (nblocks_q8 as u32, 1, 1),
        block_dim: (Q8_1_BLOCK_ELEMS as u32, 1, 1), // 32 threads per block
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(input);          // const float* x
    builder.arg(q8_output);      // void* vy
    builder.arg(&ncols_i32);     // int ncols
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("ggml_quantize_q8_1 launch: {e}")))?;

    // Sync to catch async kernel errors
    use candle_core::backend::BackendDevice;
    dev.synchronize().map_err(|e| candle_core::Error::Msg(format!("ggml_quantize_q8_1 sync: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: ggml Q5_K MMVQ
// ---------------------------------------------------------------------------

/// Compute y = W @ x where W is Q5_K `[nrows, ncols]` and x is f32 `[ncols]`.
///
/// Uses ggml's MMVQ kernel for numerical parity with llama.cpp.
///
/// `weight_q5k`: raw Q5_K bytes on GPU (flat, row-major).
/// `q8_input`:   pre-quantized Q8_1 input (from `ggml_quantize_q8_1`).
/// `output`:     `[nrows]` output buffer.
///
/// The ggml kernel signature is:
///   `(const void* vx, const void* vy, float* dst,
///     int ncols_x, int nrows_x, int nrows_y, int nrows_dst)`
///
/// For single-vector GEMV: `nrows_x = nrows`, `nrows_y = pad(ncols)/QK8_1`,
/// `nrows_dst = nrows`.
pub fn ggml_q5k_gemv(
    weight_q5k: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let _ncols_padded = pad(ncols, MATRIX_ROW_PADDING);

    let (func, stream) = load_func(dev, MMVQ_FUNC)?;

    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (32, 4, 1),                  // 128 threads
        shared_mem_bytes: MMVQ_SHARED_MEM,       // 384 bytes
    };

    // Kernel signature: (const void* vx, const void* vy, float* dst, int ncols, int nrows)
    let ncols_i32 = ncols as i32;
    let nrows_i32 = nrows as i32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(weight_q5k);     // const void* vx
    builder.arg(q8_input);       // const void* vy
    builder.arg(output);         // float* dst
    builder.arg(&ncols_i32);    // int ncols
    builder.arg(&nrows_i32);    // int nrows
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("ggml_mul_mat_vec_q5_K_q8_1 launch: {e}")))?;

    // Sync to catch async kernel errors
    use candle_core::backend::BackendDevice;
    dev.synchronize().map_err(|e| candle_core::Error::Msg(format!("ggml_mul_mat_vec_q5_K sync: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Convenience: quantize + GEMV in one call
// ---------------------------------------------------------------------------

/// Quantize f32 input to Q8_1 and run ggml Q5_K GEMV, writing to `output`.
///
/// Combines `ggml_quantize_q8_1` + `ggml_q5k_gemv` with a single Q8_1 buffer.
pub fn ggml_q5k_gemv_f32(
    weight_q5k: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    q8_buf: &mut CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_quantize_q8_1(input, q8_buf, ncols, dev)?;
    ggml_q5k_gemv(weight_q5k, q8_buf, output, nrows, ncols, dev)
}

// ---------------------------------------------------------------------------
// High-level: from Tensor, with pre-allocated GgmlQ5KBuffers
// ---------------------------------------------------------------------------

/// Run ggml Q5_K GEMV using pre-allocated buffers and Candle Tensor inputs.
///
/// 1. Quantizes the f32 input to Q8_1 into `bufs.q8_input`.
/// 2. Runs the ggml MMVQ kernel, writing to `bufs.output`.
///
/// After this call, the result is in `bufs.output[0..nrows]`.
/// The caller is responsible for copying the output out if needed.
pub fn ggml_q5k_gemv_buffered(
    weight_raw: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlQ5KBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;
    ggml_q5k_gemv(weight_raw, &bufs.q8_input, &mut bufs.output, nrows, ncols, dev)
}

// ---------------------------------------------------------------------------
// Convenience: Tensor → Tensor (allocates temp buffers per call)
// ---------------------------------------------------------------------------

/// Run ggml Q5_K GEMV from raw Tensor inputs, returning a Tensor result.
///
/// Extracts CUDA slices from Candle Tensors, allocates temp Q8_1 + output buffers,
/// runs quantize + GEMV, and wraps the result as `[1, nrows]` Tensor.
///
/// This is less efficient than the buffered path (allocates each call) but
/// convenient for one-off calls like lm_head or shared expert projections.
pub fn ggml_q5k_tensor_forward(
    raw_weight: &candle_core::Tensor,  // Q5_K raw bytes on GPU (flat U8)
    input: &candle_core::Tensor,       // [1, ncols] or [ncols] F32
    nrows: usize,
    ncols: usize,
) -> Result<candle_core::Tensor> {
    use candle_core::{Device, Storage, Tensor};

    let inp_flat = input.flatten_all()?.contiguous()?;
    let raw_c = raw_weight.contiguous()?;

    let (w_stor, w_lay) = raw_c.storage_and_layout();
    let w_cuda = match &*w_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("ggml_q5k: weight not CUDA") };
    let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(
        w_lay.start_offset()..w_lay.start_offset() + raw_c.elem_count());

    let (i_stor, i_lay) = inp_flat.storage_and_layout();
    let i_cuda = match &*i_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("ggml_q5k: input not CUDA") };
    let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(
        i_lay.start_offset()..i_lay.start_offset() + inp_flat.elem_count());

    let Device::Cuda(cuda_dev) = inp_flat.device() else {
        candle_core::bail!("ggml_q5k requires CUDA");
    };

    let q8_size = ((ncols + 31) / 32) * 36;
    let mut q8_buf = cuda_dev.alloc_zeros::<u8>(q8_size)
        .map_err(|e| candle_core::Error::Msg(format!("ggml_q5k q8 alloc: {e}")))?;
    let mut out_buf = cuda_dev.alloc_zeros::<f32>(nrows)
        .map_err(|e| candle_core::Error::Msg(format!("ggml_q5k out alloc: {e}")))?;

    ggml_q5k_gemv_f32(&w_view, &i_view, &mut q8_buf, &mut out_buf, nrows, ncols, cuda_dev)?;

    let out_tensor = Tensor::from_storage(
        Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            out_buf, cuda_dev.clone())),
        candle_core::Shape::from_dims(&[1, nrows]),
        candle_core::op::BackpropOp::none(), false);
    drop(w_stor); drop(i_stor);
    Ok(out_tensor)
}
