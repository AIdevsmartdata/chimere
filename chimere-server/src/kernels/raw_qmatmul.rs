//! Raw QMatMul — calls Candle's quantized GEMV kernel with pre-allocated buffers.
//!
//! Candle's QMatMul::forward() does 2 cudaMalloc per call (Q8_1 quantization buffer
//! + output buffer) and wraps the result in Arc<Storage>. Over 150 calls per token,
//! this adds ~0.6ms of allocation overhead + ~0.3ms of Rust dispatch.
//!
//! This module calls the SAME Candle kernel (`mul_mat_vec_q5_K_q8_1_cuda1`) directly,
//! but with pre-allocated buffers. Zero cudaMalloc per call.
//!
//! Toggle: CHIMERE_RAW_QMATMUL=1

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;

// Candle's QUANTIZED module — contains mul_mat_vec_q5_K_q8_1_cuda1 and quantize_q8_1
fn get_quantized_module() -> &'static candle_core::cuda_backend::kernels::Module {
    &candle_core::cuda_backend::kernels::QUANTIZED
}

/// Pre-allocated buffers for raw QMatMul calls.
/// One instance is shared across all layers (buffers are reused).
pub struct RawQMatMulBuffers {
    /// Q8_1 quantization of the input vector.
    /// Size: ceil(max_cols / 256) * 256 * 36 / 32 bytes.
    /// For hidden_size=2048: 2048 * 36 / 32 = 2304 bytes.
    pub q8_input: CudaSlice<u8>,
    /// Output buffer for the largest projection.
    /// Size: max(all output sizes) floats.
    /// For qkv=8192: 8192 * 4 = 32 KB.
    pub output: CudaSlice<f32>,
    /// Candle quantize_q8_1 function handle (cached).
    _quantize_func_cached: bool,
}

// Padding helper (from Candle)
const MATRIX_ROW_PADDING: usize = 512;
fn pad(n: usize, align: usize) -> usize {
    (n + align - 1) / align * align
}

impl RawQMatMulBuffers {
    /// Allocate buffers for the given max dimensions.
    pub fn new(max_input_cols: usize, max_output_rows: usize, dev: &CudaDevice) -> Result<Self> {
        let ncols_padded = pad(max_input_cols, MATRIX_ROW_PADDING);
        // Q8_1: 36 bytes per 32 elements
        let q8_size = ncols_padded * 36 / 32;
        let q8_input = dev.alloc_zeros::<u8>(q8_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8_input: {e}")))?;
        let output = dev.alloc_zeros::<f32>(max_output_rows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;
        Ok(Self {
            q8_input,
            output,
            _quantize_func_cached: false,
        })
    }
}

/// Quantize an f32 input vector to Q8_1 using Candle's quantize kernel.
/// Writes to `buf.q8_input`. No allocation.
pub fn raw_quantize_q8_1(
    input: &CudaView<'_, f32>,
    buf: &mut RawQMatMulBuffers,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let module = get_quantized_module();
    let func = dev.get_or_load_func("quantize_q8_1", module)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let block_num = (ncols_padded + 255) / 256;

    let cfg = LaunchConfig {
        grid_dim: (block_num as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;
    let ncols_padded_i32 = ncols_padded as i32;
    let mut builder = func.builder();
    builder.arg(input);
    builder.arg(&mut buf.q8_input);
    builder.arg(&ncols_i32);
    builder.arg(&ncols_padded_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1 launch: {e}")))?;
    Ok(())
}

/// Run Q5_K GEMV using Candle's compiled kernel, with pre-allocated buffers.
/// Writes result to `buf.output[0..nrows]`. No allocation.
///
/// `weight_data`: raw Q5_K bytes on GPU (from QTensor storage) — CudaView allows
/// passing either a full CudaSlice (via `.as_view()`) or a sub-slice.
/// `ncols`: input dimension (e.g. 2048 for hidden_size).
/// `nrows`: output dimension (e.g. 8192 for QKV projection).
pub fn raw_q5k_gemv_candle(
    weight_data: &CudaView<'_, u8>,
    buf: &mut RawQMatMulBuffers,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let module = get_quantized_module();
    let kernel_name = "mul_mat_vec_q5_K_q8_1_cuda1";
    let func = dev.get_or_load_func(kernel_name, module)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;
    let nrows_i32 = nrows as i32;
    let mut builder = func.builder();
    builder.arg(weight_data);
    builder.arg(&buf.q8_input);
    builder.arg(&mut buf.output);
    builder.arg(&ncols_i32);     // ncols_x
    builder.arg(&nrows_i32);     // nrows_x
    builder.arg(&ncols_padded);  // nrows_y (padded input cols)
    builder.arg(&nrows_i32);     // nrows_dst
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("{kernel_name} launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unbundled variants — separate Q8_1 input and output buffers
// ---------------------------------------------------------------------------
//
// These variants decouple the Q8_1 quantized input and the GEMV output from
// the `RawQMatMulBuffers` struct, allowing callers to route outputs to
// specific pre-allocated scratch buffers without an extra memcpy.

/// Quantize an f32 input vector to Q8_1, writing to an arbitrary u8 output buffer.
///
/// Same kernel as `raw_quantize_q8_1` but the destination is caller-provided
/// instead of `buf.q8_input`. The output buffer must be at least
/// `pad(ncols, 512) * 36 / 32` bytes.
pub fn raw_quantize_q8_1_to(
    input: &CudaView<'_, f32>,
    q8_out: &mut CudaSlice<u8>,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let module = get_quantized_module();
    let func = dev.get_or_load_func("quantize_q8_1", module)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let block_num = (ncols_padded + 255) / 256;

    let cfg = LaunchConfig {
        grid_dim: (block_num as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;
    let ncols_padded_i32 = ncols_padded as i32;
    let mut builder = func.builder();
    builder.arg(input);
    builder.arg(q8_out);
    builder.arg(&ncols_i32);
    builder.arg(&ncols_padded_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1_to launch: {e}")))?;
    Ok(())
}

/// Run Q5_K GEMV with separate Q8_1 input and f32 output buffers.
///
/// Same Candle kernel as `raw_q5k_gemv_candle` but callers provide their own
/// pre-quantized Q8_1 input (`q8_in`) and destination f32 buffer (`output`),
/// instead of routing through `RawQMatMulBuffers`.
///
/// This avoids a device-to-device memcpy when the GEMV output needs to land
/// in a specific scratch buffer rather than `buf.output`.
pub fn raw_q5k_gemv_to(
    weight_data: &CudaView<'_, u8>,
    q8_in: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let module = get_quantized_module();
    let kernel_name = "mul_mat_vec_q5_K_q8_1_cuda1";
    let func = dev.get_or_load_func(kernel_name, module)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };

    let ncols_i32 = ncols as i32;
    let nrows_i32 = nrows as i32;
    let mut builder = func.builder();
    builder.arg(weight_data);
    builder.arg(q8_in);
    builder.arg(output);
    builder.arg(&ncols_i32);     // ncols_x
    builder.arg(&nrows_i32);     // nrows_x
    builder.arg(&ncols_padded);  // nrows_y (padded input cols)
    builder.arg(&nrows_i32);     // nrows_dst
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("raw_q5k_gemv_to launch: {e}")))?;
    Ok(())
}
