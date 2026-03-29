//! FFI bindings to ggml's optimized CUDA GEMV kernels (MMVQ).
//!
//! Replaces chimere's naive CUDA GEMV kernels with ggml's optimized ones
//! for a ~4x compute speedup. The wrapper constructs ggml's internal
//! `mmvq_args` struct and calls the pre-compiled MMVQ kernel launchers.
//!
//! ## Supported quant types
//!
//! - **IQ3_S**: 110 bytes/256 elements -- MoE experts in custom-mix GGUF
//! - **Q5_K**: 176 bytes/256 elements -- attention projections (Q/K/V/O)
//! - **Q8_0**: 34 bytes/32 elements -- embeddings, lm_head
//! - **Q4_K**: 144 bytes/256 elements
//! - **Q6_K**: 210 bytes/256 elements
//!
//! ## Q8_1 input format
//!
//! All MMVQ kernels expect the input vector quantized as Q8_1:
//! - 36 bytes per 32 elements (f16 scale + f16 sum + 32 int8 quants)
//! - The sum field is used for types with min values (Q4_K, Q5_K)
//!
//! ## Toggle
//!
//! `CHIMERE_GGML_GPU=1` to enable the ggml MMVQ path (otherwise falls back
//! to chimere's cubin kernels).
//!
//! ## Usage
//!
//! ```ignore
//! use chimere_deltanet::kernels::ggml_gpu::*;
//!
//! // Pre-allocate buffers once at model init
//! let bufs = GgmlGpuBuffers::new(2048, 8192, &cuda_dev)?;
//!
//! // Per-token: quantize input + GEMV
//! ggml_gpu_gemv_iq3s(&weight_view, &input_view, &mut bufs, nrows, ncols, &cuda_dev)?;
//! // Result is in bufs.output[0..nrows]
//! ```

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, DevicePtr, DeviceSlice};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Q8_1: 36 bytes per 32 elements.
pub const Q8_1_BLOCK_BYTES: usize = 36;
pub const Q8_1_BLOCK_ELEMS: usize = 32;

/// ggml MATRIX_ROW_PADDING.
pub const MATRIX_ROW_PADDING: usize = 512;

pub fn pad(n: usize, align: usize) -> usize {
    (n + align - 1) / align * align
}

// ---------------------------------------------------------------------------
// Toggle check
// ---------------------------------------------------------------------------

/// Check if the ggml GPU MMVQ path is enabled.
pub fn is_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CHIMERE_GGML_GPU").is_ok())
}

// ---------------------------------------------------------------------------
// Global stream override
// ---------------------------------------------------------------------------
//
// When the device uses a non-blocking stream (CudaDevice::new_with_stream),
// ALL ggml FFI calls must also run on that stream. Otherwise they execute on
// the NULL (legacy default) stream, causing race conditions with cudarc kernels.
//
// Set once at init via `set_global_stream()`, used by all FFI call sites.

use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_STREAM: AtomicUsize = AtomicUsize::new(0);

/// Set the global CUDA stream for all ggml FFI calls.
///
/// # Safety
/// The raw stream pointer must remain valid for the lifetime of the program.
pub unsafe fn set_global_stream(stream: *mut std::ffi::c_void) {
    GLOBAL_STREAM.store(stream as usize, Ordering::Release);
}

/// Get the global CUDA stream (NULL if not set).
#[inline]
fn global_stream() -> *mut std::ffi::c_void {
    GLOBAL_STREAM.load(Ordering::Acquire) as *mut std::ffi::c_void
}

// ---------------------------------------------------------------------------
// FFI declarations (from ggml_cuda_gemv.cu, linked via libggml_cuda_gemv.a)
// ---------------------------------------------------------------------------

#[cfg(feature = "ggml_cuda_gemv")]
extern "C" {
    fn chimere_mmvq_iq3s(
        weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        stream: *mut std::ffi::c_void,  // cudaStream_t = NULL for default
    );

    fn chimere_mmvq_q5k(
        weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q8_0(
        weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q4k(
        weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q6k(
        weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_quantize_q8_1(
        input: *const f32,
        output: *mut std::ffi::c_void,
        ncols: i64,
        ncols_padded: i64,
        stream: *mut std::ffi::c_void,
    );

    // Fused F32->Q8_1->MMVQ: one FFI call instead of two
    fn chimere_mmvq_q5k_f32(
        weights: *const std::ffi::c_void,
        input_f32: *const f32,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        ncols_padded: i32,
        q8_scratch: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_iq3s_f32(
        weights: *const std::ffi::c_void,
        input_f32: *const f32,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        ncols_padded: i32,
        q8_scratch: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q8_0_f32(
        weights: *const std::ffi::c_void,
        input_f32: *const f32,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        ncols_padded: i32,
        q8_scratch: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q4k_f32(
        weights: *const std::ffi::c_void,
        input_f32: *const f32,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        ncols_padded: i32,
        q8_scratch: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_mmvq_q6k_f32(
        weights: *const std::ffi::c_void,
        input_f32: *const f32,
        output: *mut f32,
        ncols: i32,
        nrows: i32,
        ncols_padded: i32,
        q8_scratch: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    );

    // Phase 1.2: Batched MMVQ with GPU-resident expert indirection
    fn chimere_mmvq_iq3s_batched(
        all_weights: *const std::ffi::c_void,
        input_q8: *const std::ffi::c_void,
        output: *mut f32,
        expert_ids: *const i32,
        ncols: i32,
        nrows: i32,
        nrows_y_padded: i32,
        expert_stride: i64,
        input_stride: i64,
        top_k: i32,
        stream: *mut std::ffi::c_void,
    );

    fn chimere_quantize_q8_1_batched(
        input: *const f32,
        output: *mut std::ffi::c_void,
        ncols: i64,
        ncols_padded: i64,
        batch: i32,
        stream: *mut std::ffi::c_void,
    );
}

// ---------------------------------------------------------------------------
// Helper: extract raw CUdeviceptr from cudarc slices
// ---------------------------------------------------------------------------

/// Extract a raw pointer from a CudaView (read-only).
///
/// CUdeviceptr is u64 on x86_64. We cast it to a C pointer for FFI.
/// The cudarc stream ordering guarantee is maintained because we use
/// the default CUDA stream (NULL) for all FFI calls.
///
/// Uses the slice's own stream reference (via deref coercion from
/// `&Arc<CudaStream>` to `&CudaStream`) to avoid an Arc clone/drop
/// cycle on every call.
fn view_as_ptr<T>(view: &CudaView<'_, T>) -> *const std::ffi::c_void {
    let (ptr, _sync) = view.device_ptr(view.stream());
    ptr as *const std::ffi::c_void
}

/// Extract a raw pointer from a CudaSlice (read-only).
fn slice_as_ptr<T>(slice: &CudaSlice<T>) -> *const std::ffi::c_void {
    let (ptr, _sync) = slice.device_ptr(slice.stream());
    ptr as *const std::ffi::c_void
}

/// Extract a mutable raw pointer from a CudaSlice (for output).
fn slice_as_mut_ptr<T>(slice: &mut CudaSlice<T>) -> *mut f32 {
    let (ptr, _sync) = slice.device_ptr(slice.stream());
    ptr as *mut f32
}

/// Extract a mutable raw void pointer from a CudaSlice.
fn slice_as_mut_void_ptr<T>(slice: &mut CudaSlice<T>) -> *mut std::ffi::c_void {
    let (ptr, _sync) = slice.device_ptr(slice.stream());
    ptr as *mut std::ffi::c_void
}

/// Extract a const f32 pointer from a CudaView<f32>.
fn view_as_f32_ptr(view: &CudaView<'_, f32>) -> *const f32 {
    let (ptr, _sync) = view.device_ptr(view.stream());
    ptr as *const f32
}

// ---------------------------------------------------------------------------
// Pre-allocated GPU buffers
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for ggml MMVQ operations.
///
/// One instance shared across all layers (reused serially during inference).
/// Allocate once at model init with the maximum dimensions.
pub struct GgmlGpuBuffers {
    /// Q8_1 quantized input buffer (normed hidden state for gate+up GEMVs).
    pub q8_input: CudaSlice<u8>,
    /// Second Q8_1 buffer for intermediates (SwiGLU output for down GEMV).
    pub q8_input_b: CudaSlice<u8>,
    /// F32 output buffer.
    pub output: CudaSlice<f32>,
    /// Padded column count for the Q8_1 buffer.
    pub max_ncols_padded: usize,

    // Phase 1.2: Batched MoE buffers (zero-sync path)
    /// Batched gate GEMV output: [top_k * expert_ffn]
    pub batched_gate: CudaSlice<f32>,
    /// Batched up GEMV output: [top_k * expert_ffn]
    pub batched_up: CudaSlice<f32>,
    /// Batched down GEMV output: [top_k * hidden_size]
    pub batched_down: CudaSlice<f32>,
    /// Batched Q8_1 for SwiGLU intermediates: [top_k * q8_row_bytes(expert_ffn)]
    pub batched_q8_inter: CudaSlice<u8>,

    // Phase 3.1: Stream override for CUDA Graph capture.
    //
    // When set, all ggml FFI calls use this stream instead of NULL.
    // This allows the ggml kernels to be captured into a CUDA Graph
    // alongside cudarc kernels that run on the same stream.
    //
    // The raw pointer is a CUstream (= *mut c_void). NULL = default stream.
    // Set via `set_stream_override()`, cleared via `clear_stream_override()`.
    //
    // Wrapped in SendPtr to maintain Send-ability of GgmlGpuBuffers
    // (raw pointers don't implement Send, but CUDA stream handles are
    // safe to share across threads when properly synchronized).
    stream_override: SendPtr,
}

/// Wrapper for a raw CUDA stream pointer that implements Send.
///
/// CUDA stream handles are safe to share across threads as long as
/// the stream is not destroyed while in use. This wrapper is only used
/// for the stream override in GgmlGpuBuffers, which is set/cleared
/// within a single forward pass and never outlives the stream.
#[derive(Clone, Copy)]
struct SendPtr(*mut std::ffi::c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn null() -> Self { Self(std::ptr::null_mut()) }
    fn is_null(&self) -> bool { self.0.is_null() }
    fn as_ptr(&self) -> *mut std::ffi::c_void { self.0 }
}

impl GgmlGpuBuffers {
    /// Allocate buffers for the given max dimensions.
    ///
    /// `max_ncols`: maximum input vector length (e.g. 2048 for hidden_size).
    /// `max_nrows`: maximum output rows (e.g. 8192 for QKV projection).
    pub fn new(max_ncols: usize, max_nrows: usize, dev: &CudaDevice) -> Result<Self> {
        Self::with_moe(max_ncols, max_nrows, 0, 0, 0, dev)
    }

    /// Allocate buffers including Phase 1.2 batched MoE buffers.
    ///
    /// `expert_ffn`: expert intermediate dimension (512).
    /// `hidden_size`: model hidden dimension (2048).
    /// `top_k`: number of selected experts (8).
    pub fn with_moe(
        max_ncols: usize,
        max_nrows: usize,
        expert_ffn: usize,
        hidden_size: usize,
        top_k: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        let ncols_padded = pad(max_ncols, MATRIX_ROW_PADDING);
        let q8_size = (ncols_padded / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
        let q8_input = dev
            .alloc_zeros::<u8>(q8_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ggml_gpu q8_input: {e}")))?;
        let q8_input_b = dev
            .alloc_zeros::<u8>(q8_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ggml_gpu q8_input_b: {e}")))?;
        let output = dev
            .alloc_zeros::<f32>(max_nrows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ggml_gpu output: {e}")))?;

        // Phase 1.2: batched MoE buffers (zero if not using MoE)
        let batched_gate = dev
            .alloc_zeros::<f32>(top_k.max(1) * expert_ffn.max(1))
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_gate: {e}")))?;
        let batched_up = dev
            .alloc_zeros::<f32>(top_k.max(1) * expert_ffn.max(1))
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_up: {e}")))?;
        let batched_down = dev
            .alloc_zeros::<f32>(top_k.max(1) * hidden_size.max(1))
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_down: {e}")))?;
        // Q8_1 for top_k intermediate vectors of expert_ffn elements each
        let eff_padded = pad(expert_ffn.max(1), MATRIX_ROW_PADDING);
        let q8_inter_per_expert = (eff_padded / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
        let batched_q8_inter = dev
            .alloc_zeros::<u8>(top_k.max(1) * q8_inter_per_expert)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_q8_inter: {e}")))?;

        Ok(Self {
            q8_input,
            q8_input_b,
            output,
            max_ncols_padded: ncols_padded,
            batched_gate,
            batched_up,
            batched_down,
            batched_q8_inter,
            stream_override: SendPtr::null(),
        })
    }
}

impl GgmlGpuBuffers {
    /// Set the stream override for CUDA Graph capture.
    ///
    /// When set, all ggml FFI calls will use the given raw CUstream instead of
    /// the CUDA default stream (NULL). This allows ggml kernels to be captured
    /// into a CUDA Graph alongside cudarc kernels on the same stream.
    ///
    /// # Safety
    ///
    /// The caller must ensure the raw CUstream pointer is valid for the duration
    /// of all FFI calls made while the override is active.
    pub unsafe fn set_stream_override(&mut self, raw_stream: *mut std::ffi::c_void) {
        self.stream_override = SendPtr(raw_stream);
    }

    /// Clear the stream override, reverting to the CUDA default stream (NULL).
    pub fn clear_stream_override(&mut self) {
        self.stream_override = SendPtr::null();
    }

    /// Get the active stream pointer (per-buffer override > global > NULL).
    #[inline]
    pub fn active_stream(&self) -> *mut std::ffi::c_void {
        let s = self.stream_override.as_ptr();
        if !s.is_null() { s } else { global_stream() }
    }
}

// ---------------------------------------------------------------------------
// Q8_1 quantization
// ---------------------------------------------------------------------------

/// Quantize F32 input to Q8_1 on GPU using ggml's kernel.
///
/// The Q8_1 output buffer must be at least `pad(ncols, 512) / 32 * 36` bytes.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_quantize_q8_1(
    input: &CudaView<'_, f32>,
    q8_output: &mut CudaSlice<u8>,
    ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i64;
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_void_ptr(q8_output);
    unsafe {
        chimere_quantize_q8_1(
            in_ptr,
            out_ptr,
            ncols as i64,
            ncols_padded,
            global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_quantize_q8_1(
    _input: &CudaView<'_, f32>,
    _q8_output: &mut CudaSlice<u8>,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Quantize F32 input to Q8_1 on a specific CUDA stream.
///
/// Same as `ggml_gpu_quantize_q8_1` but launches on the given stream instead
/// of the CUDA default stream. Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_quantize_q8_1_on_stream(
    input: &CudaView<'_, f32>,
    q8_output: &mut CudaSlice<u8>,
    ncols: usize,
    _dev: &CudaDevice,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i64;
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_void_ptr(q8_output);
    unsafe {
        chimere_quantize_q8_1(
            in_ptr,
            out_ptr,
            ncols as i64,
            ncols_padded,
            stream,
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_quantize_q8_1_on_stream(
    _input: &CudaView<'_, f32>,
    _q8_output: &mut CudaSlice<u8>,
    _ncols: usize,
    _dev: &CudaDevice,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// GEMV wrappers
// ---------------------------------------------------------------------------

/// IQ3_S GEMV using ggml's optimized MMVQ kernel.
///
/// Quantizes input to Q8_1 and runs ggml's MMVQ kernel.
/// Result is written to `bufs.output[0..nrows]`.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlGpuBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_gpu_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(&bufs.q8_input);
    let out_ptr = slice_as_mut_ptr(&mut bufs.output);

    unsafe {
        chimere_mmvq_iq3s(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _bufs: &mut GgmlGpuBuffers,
    _nrows: usize,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q5_K GEMV using ggml's optimized MMVQ kernel.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlGpuBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_gpu_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(&bufs.q8_input);
    let out_ptr = slice_as_mut_ptr(&mut bufs.output);

    unsafe {
        chimere_mmvq_q5k(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _bufs: &mut GgmlGpuBuffers,
    _nrows: usize,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q8_0 GEMV using ggml's optimized MMVQ kernel.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q8_0(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlGpuBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_gpu_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(&bufs.q8_input);
    let out_ptr = slice_as_mut_ptr(&mut bufs.output);

    unsafe {
        chimere_mmvq_q8_0(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q8_0(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _bufs: &mut GgmlGpuBuffers,
    _nrows: usize,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q4_K GEMV using ggml's optimized MMVQ kernel.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q4k(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlGpuBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_gpu_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(&bufs.q8_input);
    let out_ptr = slice_as_mut_ptr(&mut bufs.output);

    unsafe {
        chimere_mmvq_q4k(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q4k(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _bufs: &mut GgmlGpuBuffers,
    _nrows: usize,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q6_K GEMV using ggml's optimized MMVQ kernel.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q6k(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    bufs: &mut GgmlGpuBuffers,
    nrows: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    ggml_gpu_quantize_q8_1(input, &mut bufs.q8_input, ncols, dev)?;

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(&bufs.q8_input);
    let out_ptr = slice_as_mut_ptr(&mut bufs.output);

    unsafe {
        chimere_mmvq_q6k(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q6k(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _bufs: &mut GgmlGpuBuffers,
    _nrows: usize,
    _ncols: usize,
    _dev: &CudaDevice,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// Raw GEMV (pre-quantized Q8_1 input, no buffer management)
// ---------------------------------------------------------------------------

/// IQ3_S GEMV with pre-quantized Q8_1 input (no F32->Q8_1 step).
///
/// For callers who quantize once and reuse for multiple weight matrices.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_q8(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);

    unsafe {
        chimere_mmvq_iq3s(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_q8(
    _weights: &CudaView<'_, u8>,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q5_K GEMV with pre-quantized Q8_1 input.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k_q8(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);

    unsafe {
        chimere_mmvq_q5k(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k_q8(
    _weights: &CudaView<'_, u8>,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Q5_K GEMV with pre-quantized Q8_1 input on a specific CUDA stream.
///
/// Same as `ggml_gpu_gemv_q5k_q8` but launches on the given stream instead
/// of the CUDA default stream. Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k_q8_on_stream(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(weights);
    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);

    unsafe {
        chimere_mmvq_q5k(w_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, stream);
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k_q8_on_stream(
    _weights: &CudaView<'_, u8>,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _nrows: usize,
    _ncols: usize,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// Fused F32->Q8_1->MMVQ wrappers (1 FFI call = 2 CUDA kernels on same stream)
//
// These eliminate one Rust->C->CUDA round trip per GEMV by combining the
// quantize and MMVQ steps into a single extern "C" call. The Q8_1 scratch
// buffer is written internally and can be reused by subsequent q8-input
// GEMV calls (e.g. quantize once, then run gate+up+beta+alpha).
// ---------------------------------------------------------------------------

/// Fused Q5_K GEMV from F32 input: quantize + MMVQ in one FFI call.
///
/// Takes F32 input directly, quantizes to Q8_1 into `q8_scratch`, then
/// runs ggml's MMVQ Q5_K kernel. Both CUDA kernels serialize on the
/// same stream -- no extra sync needed.
///
/// The Q8_1 scratch buffer is populated as a side effect, so subsequent
/// calls using `ggml_gpu_gemv_q5k_q8` can reuse it without re-quantizing.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_q5k_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k_f32(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Fused Q5_K GEMV from F32 input on a specific CUDA stream.
///
/// Same as `ggml_gpu_gemv_q5k_f32` but launches on the given stream.
/// Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k_f32_on_stream(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_q5k_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, stream,
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k_f32_on_stream(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Fused IQ3_S GEMV from F32 input: quantize + MMVQ in one FFI call.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_iq3s_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_f32(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Fused Q8_0 GEMV from F32 input: quantize + MMVQ in one FFI call.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q8_0_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_q8_0_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q8_0_f32(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Fused Q4_K GEMV from F32 input: quantize + MMVQ in one FFI call.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q4k_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_q4k_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q4k_f32(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Fused Q6_K GEMV from F32 input: quantize + MMVQ in one FFI call.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q6k_f32(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    q8_scratch: &mut CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i32;

    let w_ptr = view_as_ptr(weights);
    let in_ptr = view_as_f32_ptr(input);
    let out_ptr = slice_as_mut_ptr(output);
    let q8_ptr = slice_as_mut_void_ptr(q8_scratch);

    unsafe {
        chimere_mmvq_q6k_f32(
            w_ptr, in_ptr, out_ptr,
            ncols as i32, nrows as i32, ncols_padded,
            q8_ptr, global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q6k_f32(
    _weights: &CudaView<'_, u8>,
    _input: &CudaView<'_, f32>,
    _output: &mut CudaSlice<f32>,
    _q8_scratch: &mut CudaSlice<u8>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// Raw-pointer GEMV variants (cached CUdeviceptr, zero Arc/SyncOnDrop overhead)
//
// These take pre-extracted raw CUDA pointers instead of CudaView/CudaSlice.
// Used by the cudarc forward path with CachedWeightPtrs to eliminate the
// ~1-3µs per-call overhead of device_ptr()/Arc clone/event tracking.
//
// SAFETY: Callers must ensure pointers are valid GPU memory from the same
// device, and that the underlying CudaSlice outlives the pointer.
// ---------------------------------------------------------------------------

/// Q5_K GEMV with raw cached weight pointer and pre-quantized Q8_1 input.
///
/// Skips `device_ptr()` for the weight tensor (uses cached raw pointer).
/// The Q8_1 input and output still use `slice_as_ptr`/`slice_as_mut_ptr`
/// since those are scratch buffers that may be reallocated.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_q5k_q8_cached(
    weights_ptr: *const std::ffi::c_void,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);

    unsafe {
        chimere_mmvq_q5k(weights_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_q5k_q8_cached(
    _weights_ptr: *const std::ffi::c_void,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// IQ3_S GEMV with raw cached weight pointer and pre-quantized Q8_1 input.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_q8_cached(
    weights_ptr: *const std::ffi::c_void,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);

    unsafe {
        chimere_mmvq_iq3s(weights_ptr, q8_ptr, out_ptr, ncols as i32, nrows as i32, nrows_y, global_stream());
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_q8_cached(
    _weights_ptr: *const std::ffi::c_void,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _nrows: usize,
    _ncols: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Batched IQ3_S GEMV with raw cached weight pointer and expert indirection.
///
/// Same as `ggml_gpu_gemv_iq3s_batched` but takes a pre-extracted weight pointer
/// instead of a CudaView.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_batched_cached(
    all_weights_ptr: *const std::ffi::c_void,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    nrows: usize,
    ncols: usize,
    expert_stride: usize,
    input_stride: usize,
    top_k: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);
    let ids_ptr = {
        let (ptr, _sync) = expert_ids.device_ptr(expert_ids.stream());
        ptr as *const i32
    };

    unsafe {
        chimere_mmvq_iq3s_batched(
            all_weights_ptr,
            q8_ptr,
            out_ptr,
            ids_ptr,
            ncols as i32,
            nrows as i32,
            nrows_y,
            expert_stride as i64,
            input_stride as i64,
            top_k as i32,
            global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_batched_cached(
    _all_weights_ptr: *const std::ffi::c_void,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _expert_ids: &CudaSlice<i32>,
    _nrows: usize,
    _ncols: usize,
    _expert_stride: usize,
    _input_stride: usize,
    _top_k: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// Phase 1.2: Batched MMVQ with GPU-resident expert indirection
// ---------------------------------------------------------------------------

/// Batched IQ3_S GEMV with expert indirection — zero CPU sync.
///
/// Dispatches `top_k` expert GEMVs in a single kernel launch.
/// Expert IDs are read directly from GPU memory.
///
/// `all_weights`: all expert weights stacked (e.g. gate_exps_raw).
/// `q8_input`: Q8_1 quantized input.
/// `output`: F32 output [top_k * nrows], contiguous per expert.
/// `expert_ids`: int32 expert indices on GPU [top_k].
/// `expert_stride`: byte stride between experts in weight tensor.
/// `input_stride`: byte stride between per-expert inputs (0 = shared input).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_batched(
    all_weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    nrows: usize,
    ncols: usize,
    expert_stride: usize,
    input_stride: usize,
    top_k: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(all_weights);
    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);
    let ids_ptr = {
        let (ptr, _sync) = expert_ids.device_ptr(expert_ids.stream());
        ptr as *const i32
    };

    unsafe {
        chimere_mmvq_iq3s_batched(
            w_ptr,
            q8_ptr,
            out_ptr,
            ids_ptr,
            ncols as i32,
            nrows as i32,
            nrows_y,
            expert_stride as i64,
            input_stride as i64,
            top_k as i32,
            global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_batched(
    _all_weights: &CudaView<'_, u8>,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _expert_ids: &CudaSlice<i32>,
    _nrows: usize,
    _ncols: usize,
    _expert_stride: usize,
    _input_stride: usize,
    _top_k: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Batched Q8_1 quantization: quantize `batch` contiguous vectors at once.
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_quantize_q8_1_batched(
    input: &CudaSlice<f32>,
    q8_output: &mut CudaSlice<u8>,
    ncols: usize,
    batch: usize,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i64;
    let in_ptr = {
        let (ptr, _sync) = input.device_ptr(input.stream());
        ptr as *const f32
    };
    let out_ptr = slice_as_mut_void_ptr(q8_output);
    unsafe {
        chimere_quantize_q8_1_batched(
            in_ptr,
            out_ptr,
            ncols as i64,
            ncols_padded,
            batch as i32,
            global_stream(),
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_quantize_q8_1_batched(
    _input: &CudaSlice<f32>,
    _q8_output: &mut CudaSlice<u8>,
    _ncols: usize,
    _batch: usize,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// Phase 3.1: _on_stream variants for batched MoE (CUDA Graph capture)
// ---------------------------------------------------------------------------

/// Batched IQ3_S GEMV on a specific CUDA stream.
///
/// Same as `ggml_gpu_gemv_iq3s_batched` but launches on the given stream
/// instead of the global stream. Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_batched_on_stream(
    all_weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    nrows: usize,
    ncols: usize,
    expert_stride: usize,
    input_stride: usize,
    top_k: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let w_ptr = view_as_ptr(all_weights);
    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);
    let ids_ptr = {
        let (ptr, _sync) = expert_ids.device_ptr(expert_ids.stream());
        ptr as *const i32
    };

    unsafe {
        chimere_mmvq_iq3s_batched(
            w_ptr,
            q8_ptr,
            out_ptr,
            ids_ptr,
            ncols as i32,
            nrows as i32,
            nrows_y,
            expert_stride as i64,
            input_stride as i64,
            top_k as i32,
            stream,
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_batched_on_stream(
    _all_weights: &CudaView<'_, u8>,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _expert_ids: &CudaSlice<i32>,
    _nrows: usize,
    _ncols: usize,
    _expert_stride: usize,
    _input_stride: usize,
    _top_k: usize,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Batched IQ3_S GEMV with raw cached weight pointer on a specific CUDA stream.
///
/// Same as `ggml_gpu_gemv_iq3s_batched_cached` but launches on the given stream.
/// Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_gemv_iq3s_batched_cached_on_stream(
    all_weights_ptr: *const std::ffi::c_void,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    nrows: usize,
    ncols: usize,
    expert_stride: usize,
    input_stride: usize,
    top_k: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let nrows_y = (ncols_padded / Q8_1_BLOCK_ELEMS) as i32;

    let q8_ptr = slice_as_ptr(q8_input);
    let out_ptr = slice_as_mut_ptr(output);
    let ids_ptr = {
        let (ptr, _sync) = expert_ids.device_ptr(expert_ids.stream());
        ptr as *const i32
    };

    unsafe {
        chimere_mmvq_iq3s_batched(
            all_weights_ptr,
            q8_ptr,
            out_ptr,
            ids_ptr,
            ncols as i32,
            nrows as i32,
            nrows_y,
            expert_stride as i64,
            input_stride as i64,
            top_k as i32,
            stream,
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_gemv_iq3s_batched_cached_on_stream(
    _all_weights_ptr: *const std::ffi::c_void,
    _q8_input: &CudaSlice<u8>,
    _output: &mut CudaSlice<f32>,
    _expert_ids: &CudaSlice<i32>,
    _nrows: usize,
    _ncols: usize,
    _expert_stride: usize,
    _input_stride: usize,
    _top_k: usize,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

/// Batched Q8_1 quantization on a specific CUDA stream.
///
/// Same as `ggml_gpu_quantize_q8_1_batched` but launches on the given stream.
/// Required for CUDA Graph capture (Phase 3.1).
#[cfg(feature = "ggml_cuda_gemv")]
pub fn ggml_gpu_quantize_q8_1_batched_on_stream(
    input: &CudaSlice<f32>,
    q8_output: &mut CudaSlice<u8>,
    ncols: usize,
    batch: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING) as i64;
    let in_ptr = {
        let (ptr, _sync) = input.device_ptr(input.stream());
        ptr as *const f32
    };
    let out_ptr = slice_as_mut_void_ptr(q8_output);
    unsafe {
        chimere_quantize_q8_1_batched(
            in_ptr,
            out_ptr,
            ncols as i64,
            ncols_padded,
            batch as i32,
            stream,
        );
    }
    Ok(())
}

#[cfg(not(feature = "ggml_cuda_gemv"))]
pub fn ggml_gpu_quantize_q8_1_batched_on_stream(
    _input: &CudaSlice<f32>,
    _q8_output: &mut CudaSlice<u8>,
    _ncols: usize,
    _batch: usize,
    _stream: *mut std::ffi::c_void,
) -> Result<()> {
    candle_core::bail!("ggml CUDA GEMV not compiled -- need nvcc and ggml source")
}

// ---------------------------------------------------------------------------
// High-level: Tensor -> Tensor convenience
// ---------------------------------------------------------------------------

/// Run ggml MMVQ GEMV from Candle Tensors, returning a Tensor result.
///
/// Dispatches to the appropriate quant type based on `quant_type`.
/// Allocates temp buffers per call -- use the buffered API for hot paths.
pub fn ggml_gpu_tensor_forward(
    raw_weight: &candle_core::Tensor,
    input: &candle_core::Tensor,
    nrows: usize,
    ncols: usize,
    quant_type: GgmlGpuQuantType,
) -> Result<candle_core::Tensor> {
    use candle_core::{Device, Storage, Tensor};

    let inp_flat = input.flatten_all()?.contiguous()?;
    let raw_c = raw_weight.contiguous()?;

    let (w_stor, w_lay) = raw_c.storage_and_layout();
    let w_cuda = match &*w_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("ggml_gpu: weight not on CUDA"),
    };
    let w_view = w_cuda
        .as_cuda_slice::<u8>()?
        .slice(w_lay.start_offset()..w_lay.start_offset() + raw_c.elem_count());

    let (i_stor, i_lay) = inp_flat.storage_and_layout();
    let i_cuda = match &*i_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("ggml_gpu: input not on CUDA"),
    };
    let i_view = i_cuda
        .as_cuda_slice::<f32>()?
        .slice(i_lay.start_offset()..i_lay.start_offset() + inp_flat.elem_count());

    let Device::Cuda(cuda_dev) = inp_flat.device() else {
        candle_core::bail!("ggml_gpu requires CUDA device");
    };

    let mut bufs = GgmlGpuBuffers::new(ncols, nrows, cuda_dev)?;

    match quant_type {
        GgmlGpuQuantType::IQ3_S => ggml_gpu_gemv_iq3s(&w_view, &i_view, &mut bufs, nrows, ncols, cuda_dev)?,
        GgmlGpuQuantType::Q5_K => ggml_gpu_gemv_q5k(&w_view, &i_view, &mut bufs, nrows, ncols, cuda_dev)?,
        GgmlGpuQuantType::Q8_0 => ggml_gpu_gemv_q8_0(&w_view, &i_view, &mut bufs, nrows, ncols, cuda_dev)?,
        GgmlGpuQuantType::Q4_K => ggml_gpu_gemv_q4k(&w_view, &i_view, &mut bufs, nrows, ncols, cuda_dev)?,
        GgmlGpuQuantType::Q6_K => ggml_gpu_gemv_q6k(&w_view, &i_view, &mut bufs, nrows, ncols, cuda_dev)?,
    }

    let out_tensor = Tensor::from_storage(
        Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            bufs.output,
            cuda_dev.clone(),
        )),
        candle_core::Shape::from_dims(&[1, nrows]),
        candle_core::op::BackpropOp::none(),
        false,
    );
    drop(w_stor);
    drop(i_stor);
    Ok(out_tensor)
}

/// Quant type selector for ggml GPU GEMV dispatch.
#[derive(Debug, Clone, Copy)]
pub enum GgmlGpuQuantType {
    IQ3_S,
    Q5_K,
    Q8_0,
    Q4_K,
    Q6_K,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Minimal test: compare ggml GPU MMVQ IQ3_S GEMV against chimere's naive kernel.
    ///
    /// Loads expert 0's gate weights from the real GGUF, creates an all-1.0 input,
    /// runs both kernels, and checks that outputs match within tolerance.
    ///
    /// Run with:
    ///   LD_LIBRARY_PATH={IKLLAMACPP_DIR}/build_sm120/ggml/src \
    ///   CHIMERE_GGML_GPU=1 CUDA_COMPUTE_CAP=89 \
    ///   cargo test --release --features server test_ggml_gpu_vs_naive_single_gemv -- --nocapture
    #[test]
    fn test_ggml_gpu_vs_naive_single_gemv() {
        let gguf_path = "{HOME}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            eprintln!("[SKIP] GGUF not found at {}", gguf_path);
            return;
        }

        // 1. Acquire CUDA device
        let device = Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => {
                eprintln!("[SKIP] No CUDA device available");
                return;
            }
        };

        let stream = cuda_dev.cuda_stream();

        // 2. Open GGUF and load the expert gate tensor
        let gguf = crate::gguf_loader::GgufFile::open(gguf_path).unwrap();
        let (raw, _n_elements, shape) = gguf.load_tensor_u8("blk.0.ffn_gate_exps.weight", &device).unwrap();

        eprintln!("[INFO] Tensor shape (experts, nrows, ncols): {:?}", shape);
        eprintln!("[INFO] Raw tensor elem_count (total bytes): {}", raw.elem_count());

        // Shape = (256 experts, 512 rows, 2048 cols)
        let n_experts = shape.0;
        let nrows = shape.1;      // 512 (expert_ffn / intermediate)
        let ncols = shape.2;      // 2048 (hidden_size)

        // IQ3_S: 110 bytes per 256 elements
        // expert_bytes = nrows * (ncols / 256) * 110
        let expert_bytes = nrows * (ncols / 256) * 110;
        let expected_total = n_experts * expert_bytes;
        eprintln!("[INFO] n_experts={}, nrows={}, ncols={}", n_experts, nrows, ncols);
        eprintln!("[INFO] expert_bytes={}, expected_total={}, actual_total={}",
                  expert_bytes, expected_total, raw.elem_count());

        assert_eq!(raw.elem_count(), expected_total,
            "Total bytes mismatch: {} expected vs {} actual", expected_total, raw.elem_count());

        // 3. Extract expert 0 (first expert_bytes of the tensor)
        let expert_0 = raw.narrow(0, 0, expert_bytes).unwrap().contiguous().unwrap();

        // Get CudaView for expert 0 weights
        let (e0_stor, e0_lay) = expert_0.storage_and_layout();
        let e0_cuda = match &*e0_stor {
            candle_core::Storage::Cuda(c) => c,
            _ => panic!("expert_0 not on CUDA"),
        };
        let e0_slice = e0_cuda.as_cuda_slice::<u8>().unwrap();
        let e0_offset = e0_lay.start_offset();
        let e0_view = e0_slice.slice(e0_offset..e0_offset + expert_bytes);

        // 4. Create input: all 1.0 on GPU
        let input_host: Vec<f32> = vec![1.0; ncols];
        let mut input_gpu: CudaSlice<f32> = stream.alloc_zeros(ncols)
            .map_err(|e| candle_core::Error::Msg(format!("alloc input: {e}"))).unwrap();
        cuda_dev.memcpy_htod(&input_host, &mut input_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload input: {e}"))).unwrap();
        let input_view = input_gpu.slice(..);

        // Print pointer addresses for debugging
        eprintln!("[PTR] weights: {:?}", view_as_ptr(&e0_view));
        eprintln!("[PTR] input:   {:?}", view_as_ptr(&input_view));

        // 5. Run naive chimere kernel
        let mut result_naive: CudaSlice<f32> = stream.alloc_zeros(nrows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc naive output: {e}"))).unwrap();

        crate::kernels::gemv_iq3s_fused(&e0_view, &input_view, &mut result_naive, nrows, ncols, &cuda_dev)
            .expect("naive gemv_iq3s_fused failed");

        // Synchronize to ensure naive kernel is complete
        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after naive kernel");
        }

        let naive_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&result_naive)
            .map_err(|e| candle_core::Error::Msg(format!("readback naive: {e}"))).unwrap();

        // 6. Run ggml GPU MMVQ kernel
        let mut bufs = GgmlGpuBuffers::new(ncols, nrows, &cuda_dev).unwrap();

        eprintln!("[INFO] Q8_1 buffer size: {} bytes, max_ncols_padded: {}",
                  bufs.q8_input.len(), bufs.max_ncols_padded);

        // Check Q8_1 buffer is large enough
        let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
        let q8_needed = (ncols_padded / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
        eprintln!("[INFO] ncols_padded={}, q8_needed={}, q8_allocated={}",
                  ncols_padded, q8_needed, bufs.q8_input.len());
        assert!(bufs.q8_input.len() >= q8_needed,
            "Q8_1 buffer too small: need {} but only {} allocated", q8_needed, bufs.q8_input.len());

        ggml_gpu_gemv_iq3s(&e0_view, &input_view, &mut bufs, nrows, ncols, &cuda_dev)
            .expect("ggml_gpu_gemv_iq3s failed");

        // Synchronize to ensure ggml kernel is complete
        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after ggml kernel");
        }

        let ggml_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&bufs.output)
            .map_err(|e| candle_core::Error::Msg(format!("readback ggml: {e}"))).unwrap();

        // Only compare the first nrows elements (output buffer may be larger)
        let ggml_out = &ggml_cpu[..nrows];

        // 7. Diagnostics
        eprintln!("\n=== RESULTS ===");
        eprintln!("Naive first 10: {:?}", &naive_cpu[..10.min(nrows)]);
        eprintln!("Ggml  first 10: {:?}", &ggml_out[..10.min(nrows)]);

        let has_nan_naive = naive_cpu.iter().any(|x| x.is_nan());
        let has_nan_ggml = ggml_out.iter().any(|x| x.is_nan());
        let all_zero_naive = naive_cpu.iter().all(|&x| x == 0.0);
        let all_zero_ggml = ggml_out.iter().all(|&x| x == 0.0);

        eprintln!("Naive  NaN: {}, all_zero: {}", has_nan_naive, all_zero_naive);
        eprintln!("Ggml   NaN: {}, all_zero: {}", has_nan_ggml, all_zero_ggml);

        if has_nan_ggml {
            panic!("BUG: ggml produces NaN -- check Q8_1 format or mmvq_args construction");
        }
        if all_zero_ggml {
            panic!("BUG: ggml produces all zeros -- check pointer passing or kernel dispatch");
        }
        if has_nan_naive {
            eprintln!("WARNING: naive kernel also produces NaN -- suspicious");
        }

        // 8. Element-wise comparison
        let max_diff = naive_cpu.iter().zip(ggml_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = naive_cpu.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = max_diff / (max_val + 1e-10);

        eprintln!("Max abs diff:  {:.6}", max_diff);
        eprintln!("Max abs value: {:.6}", max_val);
        eprintln!("Relative err:  {:.6}", rel_err);

        // Also show some per-element diffs
        let mut diffs: Vec<(usize, f32, f32, f32)> = naive_cpu.iter().zip(ggml_out.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        eprintln!("\nTop 5 largest diffs (idx, naive, ggml, |diff|):");
        for (idx, naive_val, ggml_val, diff) in diffs.iter().take(5) {
            eprintln!("  [{:4}] naive={:12.6}, ggml={:12.6}, diff={:.6}", idx, naive_val, ggml_val, diff);
        }

        assert!(rel_err < 0.02,
            "Relative error too high: {:.6} (max_diff={:.6}, max_val={:.6}). \
             Kernels produce different results.", rel_err, max_diff, max_val);

        eprintln!("\nPASS: ggml GPU MMVQ matches naive kernel (rel_err={:.6})", rel_err);
    }

    /// Compare ggml GPU MMVQ Q5_K GEMV against chimere's naive Q5_K kernel.
    ///
    /// Loads `blk.0.attn_gate.weight` (Q5_K, [2048, 4096] = 4096 rows, 2048 cols),
    /// creates an all-0.01 input, and checks that both kernels produce matching output.
    ///
    /// Run with:
    ///   LD_LIBRARY_PATH={IKLLAMACPP_DIR}/build_sm120/ggml/src \
    ///   CHIMERE_GGML_GPU=1 CUDA_COMPUTE_CAP=89 \
    ///   cargo test --release --features server test_ggml_gpu_vs_naive_q5k -- --nocapture
    #[test]
    fn test_ggml_gpu_vs_naive_q5k() {
        let gguf_path = "{HOME}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            eprintln!("[SKIP] GGUF not found at {}", gguf_path);
            return;
        }

        // 1. Acquire CUDA device
        let device = Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => {
                eprintln!("[SKIP] No CUDA device available");
                return;
            }
        };

        let stream = cuda_dev.cuda_stream();

        // 2. Open GGUF and load Q5_K tensor (2D)
        let gguf = crate::gguf_loader::GgufFile::open(gguf_path).unwrap();
        let tensor_name = "blk.0.attn_gate.weight";
        let (raw, _n_elements, dims) = gguf.load_tensor_u8_any(tensor_name, &device).unwrap();

        // Q5_K shape [2048, 4096] => dims_rev = [4096, 2048] => nrows=4096, ncols=2048
        let nrows = dims[0];
        let ncols = dims[1];
        eprintln!("[Q5K] Tensor '{}' dims: {:?} => nrows={}, ncols={}", tensor_name, dims, nrows, ncols);

        // Q5_K: 176 bytes per 256 elements
        let expected_bytes = nrows * (ncols / 256) * 176;
        eprintln!("[Q5K] expected_bytes={}, actual_bytes={}", expected_bytes, raw.elem_count());
        assert_eq!(raw.elem_count(), expected_bytes,
            "Q5_K bytes mismatch: {} expected vs {} actual", expected_bytes, raw.elem_count());

        // Get CudaView for weight bytes
        let raw_c = raw.contiguous().unwrap();
        let (w_stor, w_lay) = raw_c.storage_and_layout();
        let w_cuda = match &*w_stor {
            candle_core::Storage::Cuda(c) => c,
            _ => panic!("weight not on CUDA"),
        };
        let w_slice = w_cuda.as_cuda_slice::<u8>().unwrap();
        let w_offset = w_lay.start_offset();
        let w_view = w_slice.slice(w_offset..w_offset + expected_bytes);

        // 3. Create input: all 0.01 on GPU
        let input_host: Vec<f32> = vec![0.01; ncols];
        let mut input_gpu: CudaSlice<f32> = stream.alloc_zeros(ncols)
            .map_err(|e| candle_core::Error::Msg(format!("alloc input: {e}"))).unwrap();
        cuda_dev.memcpy_htod(&input_host, &mut input_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload input: {e}"))).unwrap();
        let input_view = input_gpu.slice(..);

        // 4. Run naive chimere Q5_K kernel (q5k_mmvq_ggml::ggml_q5k_gemv_f32)
        let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
        let q8_size = (ncols_padded / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES;
        let mut q8_buf: CudaSlice<u8> = stream.alloc_zeros(q8_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8_buf: {e}"))).unwrap();
        let mut result_naive: CudaSlice<f32> = stream.alloc_zeros(nrows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc naive output: {e}"))).unwrap();

        crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv_f32(
            &w_view, &input_view, &mut q8_buf, &mut result_naive, nrows, ncols, &cuda_dev,
        ).expect("naive ggml_q5k_gemv_f32 failed");

        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after naive Q5K kernel");
        }

        let naive_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&result_naive)
            .map_err(|e| candle_core::Error::Msg(format!("readback naive: {e}"))).unwrap();

        // 5. Run ggml GPU MMVQ Q5_K kernel
        let mut bufs = GgmlGpuBuffers::new(ncols, nrows, &cuda_dev).unwrap();

        ggml_gpu_gemv_q5k(&w_view, &input_view, &mut bufs, nrows, ncols, &cuda_dev)
            .expect("ggml_gpu_gemv_q5k failed");

        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after ggml Q5K kernel");
        }

        let ggml_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&bufs.output)
            .map_err(|e| candle_core::Error::Msg(format!("readback ggml: {e}"))).unwrap();
        let ggml_out = &ggml_cpu[..nrows];

        // 6. Diagnostics
        eprintln!("\n=== Q5_K RESULTS ===");
        eprintln!("Naive first 10: {:?}", &naive_cpu[..10.min(nrows)]);
        eprintln!("Ggml  first 10: {:?}", &ggml_out[..10.min(nrows)]);

        let has_nan_naive = naive_cpu.iter().any(|x| x.is_nan());
        let has_nan_ggml = ggml_out.iter().any(|x| x.is_nan());
        let all_zero_naive = naive_cpu.iter().all(|&x| x == 0.0);
        let all_zero_ggml = ggml_out.iter().all(|&x| x == 0.0);

        eprintln!("Naive  NaN: {}, all_zero: {}", has_nan_naive, all_zero_naive);
        eprintln!("Ggml   NaN: {}, all_zero: {}", has_nan_ggml, all_zero_ggml);

        if has_nan_ggml {
            panic!("BUG: Q5_K ggml produces NaN");
        }
        if all_zero_ggml {
            panic!("BUG: Q5_K ggml produces all zeros");
        }

        // 7. Element-wise comparison
        let max_diff = naive_cpu.iter().zip(ggml_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = naive_cpu.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = max_diff / (max_val + 1e-10);

        eprintln!("Max abs diff:  {:.6}", max_diff);
        eprintln!("Max abs value: {:.6}", max_val);
        eprintln!("Relative err:  {:.6}", rel_err);

        let mut diffs: Vec<(usize, f32, f32, f32)> = naive_cpu.iter().zip(ggml_out.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        eprintln!("\nTop 5 largest diffs (idx, naive, ggml, |diff|):");
        for (idx, naive_val, ggml_val, diff) in diffs.iter().take(5) {
            eprintln!("  [{:4}] naive={:12.6}, ggml={:12.6}, diff={:.6}", idx, naive_val, ggml_val, diff);
        }

        assert!(rel_err < 0.02,
            "Q5_K relative error too high: {:.6} (max_diff={:.6}, max_val={:.6}). \
             Kernels produce different results.", rel_err, max_diff, max_val);

        eprintln!("\nPASS: Q5_K ggml GPU MMVQ matches naive kernel (rel_err={:.6})", rel_err);
    }

    /// Compare ggml GPU MMVQ Q8_0 GEMV against chimere's naive Q8_0 kernel.
    ///
    /// Loads `blk.3.attn_v.weight` (Q8_0, [2048, 512] = 512 rows, 2048 cols),
    /// creates an all-0.01 input, and checks that both kernels produce matching output.
    ///
    /// Run with:
    ///   LD_LIBRARY_PATH={IKLLAMACPP_DIR}/build_sm120/ggml/src \
    ///   CHIMERE_GGML_GPU=1 CUDA_COMPUTE_CAP=89 \
    ///   cargo test --release --features server test_ggml_gpu_vs_naive_q8_0 -- --nocapture
    #[test]
    fn test_ggml_gpu_vs_naive_q8_0() {
        let gguf_path = "{HOME}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            eprintln!("[SKIP] GGUF not found at {}", gguf_path);
            return;
        }

        // 1. Acquire CUDA device
        let device = Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => {
                eprintln!("[SKIP] No CUDA device available");
                return;
            }
        };

        let stream = cuda_dev.cuda_stream();

        // 2. Open GGUF and load Q8_0 tensor (2D)
        let gguf = crate::gguf_loader::GgufFile::open(gguf_path).unwrap();
        let tensor_name = "blk.3.attn_v.weight";
        let (raw, _n_elements, dims) = gguf.load_tensor_u8_any(tensor_name, &device).unwrap();

        // Q8_0 shape [2048, 512] => dims_rev = [512, 2048] => nrows=512, ncols=2048
        let nrows = dims[0];
        let ncols = dims[1];
        eprintln!("[Q8_0] Tensor '{}' dims: {:?} => nrows={}, ncols={}", tensor_name, dims, nrows, ncols);

        // Q8_0: 34 bytes per 32 elements
        let expected_bytes = nrows * (ncols / 32) * 34;
        eprintln!("[Q8_0] expected_bytes={}, actual_bytes={}", expected_bytes, raw.elem_count());
        assert_eq!(raw.elem_count(), expected_bytes,
            "Q8_0 bytes mismatch: {} expected vs {} actual", expected_bytes, raw.elem_count());

        // Get CudaView for weight bytes
        let raw_c = raw.contiguous().unwrap();
        let (w_stor, w_lay) = raw_c.storage_and_layout();
        let w_cuda = match &*w_stor {
            candle_core::Storage::Cuda(c) => c,
            _ => panic!("weight not on CUDA"),
        };
        let w_slice = w_cuda.as_cuda_slice::<u8>().unwrap();
        let w_offset = w_lay.start_offset();
        let w_view = w_slice.slice(w_offset..w_offset + expected_bytes);

        // 3. Create input: all 0.01 on GPU
        let input_host: Vec<f32> = vec![0.01; ncols];
        let mut input_gpu: CudaSlice<f32> = stream.alloc_zeros(ncols)
            .map_err(|e| candle_core::Error::Msg(format!("alloc input: {e}"))).unwrap();
        cuda_dev.memcpy_htod(&input_host, &mut input_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload input: {e}"))).unwrap();
        let input_view = input_gpu.slice(..);

        // 4. Run naive chimere Q8_0 kernel (gemv_q8_0::gemv_q8_0_f32)
        let mut result_naive: CudaSlice<f32> = stream.alloc_zeros(nrows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc naive output: {e}"))).unwrap();

        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
            &w_view, &input_view, &mut result_naive, nrows, ncols, &cuda_dev,
        ).expect("naive gemv_q8_0_f32 failed");

        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after naive Q8_0 kernel");
        }

        let naive_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&result_naive)
            .map_err(|e| candle_core::Error::Msg(format!("readback naive: {e}"))).unwrap();

        // 5. Run ggml GPU MMVQ Q8_0 kernel
        let mut bufs = GgmlGpuBuffers::new(ncols, nrows, &cuda_dev).unwrap();

        ggml_gpu_gemv_q8_0(&w_view, &input_view, &mut bufs, nrows, ncols, &cuda_dev)
            .expect("ggml_gpu_gemv_q8_0 failed");

        {
            use candle_core::backend::BackendDevice;
            cuda_dev.synchronize().expect("sync after ggml Q8_0 kernel");
        }

        let ggml_cpu: Vec<f32> = stream.clone()
            .clone_dtoh(&bufs.output)
            .map_err(|e| candle_core::Error::Msg(format!("readback ggml: {e}"))).unwrap();
        let ggml_out = &ggml_cpu[..nrows];

        // 6. Diagnostics
        eprintln!("\n=== Q8_0 RESULTS ===");
        eprintln!("Naive first 10: {:?}", &naive_cpu[..10.min(nrows)]);
        eprintln!("Ggml  first 10: {:?}", &ggml_out[..10.min(nrows)]);

        let has_nan_naive = naive_cpu.iter().any(|x| x.is_nan());
        let has_nan_ggml = ggml_out.iter().any(|x| x.is_nan());
        let all_zero_naive = naive_cpu.iter().all(|&x| x == 0.0);
        let all_zero_ggml = ggml_out.iter().all(|&x| x == 0.0);

        eprintln!("Naive  NaN: {}, all_zero: {}", has_nan_naive, all_zero_naive);
        eprintln!("Ggml   NaN: {}, all_zero: {}", has_nan_ggml, all_zero_ggml);

        if has_nan_ggml {
            panic!("BUG: Q8_0 ggml produces NaN");
        }
        if all_zero_ggml {
            panic!("BUG: Q8_0 ggml produces all zeros");
        }

        // 7. Element-wise comparison
        let max_diff = naive_cpu.iter().zip(ggml_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val = naive_cpu.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = max_diff / (max_val + 1e-10);

        eprintln!("Max abs diff:  {:.6}", max_diff);
        eprintln!("Max abs value: {:.6}", max_val);
        eprintln!("Relative err:  {:.6}", rel_err);

        let mut diffs: Vec<(usize, f32, f32, f32)> = naive_cpu.iter().zip(ggml_out.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, *a, *b, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        eprintln!("\nTop 5 largest diffs (idx, naive, ggml, |diff|):");
        for (idx, naive_val, ggml_val, diff) in diffs.iter().take(5) {
            eprintln!("  [{:4}] naive={:12.6}, ggml={:12.6}, diff={:.6}", idx, naive_val, ggml_val, diff);
        }

        assert!(rel_err < 0.02,
            "Q8_0 relative error too high: {:.6} (max_diff={:.6}, max_val={:.6}). \
             Kernels produce different results.", rel_err, max_diff, max_val);

        eprintln!("\nPASS: Q8_0 ggml GPU MMVQ matches naive kernel (rel_err={:.6})", rel_err);
    }
}
