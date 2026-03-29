//! Q8_0 quantization/dequantization for KV cache.
//!
//! Q8_0 block: 34 bytes per 32 elements.
//! Layout: `[f16 scale (2 bytes), i8 qs[32] (32 bytes)]`
//!
//! Compared to F16 (2 bytes/element = 64 bytes per 32 elements), Q8_0 uses
//! 34 bytes per 32 elements = 53% of F16 memory (~47% savings).
//!
//! Two paths are provided:
//! - **CPU path** (`quantize_q8_0_cpu` / `dequantize_q8_0_cpu`): Used for
//!   CPU tensors and as a fallback.
//! - **GPU path** (`quantize_q8_0_gpu` / `dequantize_q8_0_gpu`): CUDA kernels
//!   via NVRTC for zero-copy quantize/dequant on GPU.

use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use half::f16;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Q8_0 block: 32 elements per block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
/// Q8_0 block: 34 bytes per block (2 byte f16 scale + 32 byte i8 values).
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Return the Q8_0 byte size for `n_elements` f32 values (must be multiple of 32).
pub fn q8_0_byte_size(n_elements: usize) -> usize {
    debug_assert!(
        n_elements % Q8_0_BLOCK_ELEMS == 0,
        "n_elements ({n_elements}) must be a multiple of {Q8_0_BLOCK_ELEMS}"
    );
    (n_elements / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// CPU quantization: F32 -> Q8_0
// ---------------------------------------------------------------------------

/// Quantize `input` (f32 slice) to Q8_0 format.
///
/// `input.len()` must be a multiple of 32.
/// Returns a `Vec<u8>` of `q8_0_byte_size(input.len())` bytes.
pub fn quantize_q8_0_cpu(input: &[f32]) -> Vec<u8> {
    let n = input.len();
    assert!(
        n % Q8_0_BLOCK_ELEMS == 0,
        "quantize_q8_0_cpu: input length ({n}) must be a multiple of {Q8_0_BLOCK_ELEMS}"
    );
    let n_blocks = n / Q8_0_BLOCK_ELEMS;
    let mut out = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];

    for blk in 0..n_blocks {
        let src = &input[blk * Q8_0_BLOCK_ELEMS..(blk + 1) * Q8_0_BLOCK_ELEMS];
        let dst = &mut out[blk * Q8_0_BLOCK_BYTES..(blk + 1) * Q8_0_BLOCK_BYTES];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        for &x in src {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        // Store scale as f16 (little-endian)
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_bits().to_le_bytes();
        dst[0] = scale_bytes[0];
        dst[1] = scale_bytes[1];

        // Quantize values
        for i in 0..Q8_0_BLOCK_ELEMS {
            let fq = src[i] * inv_scale;
            let q = fq.round().clamp(-127.0, 127.0) as i8;
            dst[2 + i] = q as u8;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// CPU dequantization: Q8_0 -> F32
// ---------------------------------------------------------------------------

/// Dequantize Q8_0 bytes back to f32.
///
/// `data` must contain exactly `q8_0_byte_size(n_elements)` bytes.
/// Returns a `Vec<f32>` of `n_elements` values.
pub fn dequantize_q8_0_cpu(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(
        n_elements % Q8_0_BLOCK_ELEMS == 0,
        "dequantize_q8_0_cpu: n_elements ({n_elements}) must be a multiple of {Q8_0_BLOCK_ELEMS}"
    );
    let n_blocks = n_elements / Q8_0_BLOCK_ELEMS;
    assert!(
        data.len() >= n_blocks * Q8_0_BLOCK_BYTES,
        "dequantize_q8_0_cpu: data too short ({} bytes for {n_blocks} blocks)",
        data.len()
    );

    let mut result = Vec::with_capacity(n_elements);

    for blk in 0..n_blocks {
        let block = &data[blk * Q8_0_BLOCK_BYTES..(blk + 1) * Q8_0_BLOCK_BYTES];
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        for i in 0..Q8_0_BLOCK_ELEMS {
            let q = block[2 + i] as i8;
            result.push(q as f32 * d);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// CUDA kernel source for Q8_0 quantize + dequantize
// ---------------------------------------------------------------------------

const KV_Q8_0_KERNEL_SRC: &str = r#"
// f16 -> f32 via hardware instruction
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    float f;
    asm("{ .reg .b16 tmp; mov.b16 tmp, %1; cvt.f32.f16 %0, tmp; }"
        : "=f"(f) : "h"(h));
    return f;
}

// f32 -> f16 (round-to-nearest-even)
__device__ __forceinline__ unsigned short f32_to_f16_bits(float x) {
    unsigned short h;
    asm("{ .reg .b16 tmp; cvt.rn.f16.f32 tmp, %1; mov.b16 %0, tmp; }" : "=h"(h) : "f"(x));
    return h;
}

// Q8_0 block: 34 bytes = 2 (f16 d) + 32 (i8 qs)
struct block_q8_0 {
    unsigned short d_bits;
    signed char qs[32];
};

// =====================================================================
// F32 -> Q8_0 quantization kernel
// =====================================================================
// One thread per block of 32 elements.
// input:  [n_blocks * 32] f32
// q8_out: [n_blocks * 34] u8

extern "C" __global__ void quantize_f32_to_q8_0(
    const float* __restrict__ input,
    unsigned char* __restrict__ q8_out,
    int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x;
         blk < n_blocks;
         blk += blockDim.x * gridDim.x)
    {
        const float* x = &input[blk * 32];
        struct block_q8_0* b = (struct block_q8_0*)(q8_out + blk * 34);

        // Find max absolute value
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = x[i] < 0.0f ? -x[i] : x[i];
            if (ax > amax) amax = ax;
        }

        float scale = amax / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        b->d_bits = f32_to_f16_bits(scale);

        for (int i = 0; i < 32; i++) {
            float fq = x[i] * inv_scale;
            int q = (int)(fq + (fq >= 0.0f ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            b->qs[i] = (signed char)q;
        }
    }
}

// =====================================================================
// Q8_0 -> F32 dequantization kernel
// =====================================================================
// One thread per block of 32 elements.
// q8_in:  [n_blocks * 34] u8
// output: [n_blocks * 32] f32

extern "C" __global__ void dequantize_q8_0_to_f32(
    const unsigned char* __restrict__ q8_in,
    float* __restrict__ output,
    int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x;
         blk < n_blocks;
         blk += blockDim.x * gridDim.x)
    {
        const struct block_q8_0* b =
            (const struct block_q8_0*)(q8_in + blk * 34);
        float d = f16_to_f32(b->d_bits);
        float* dst = &output[blk * 32];

        for (int i = 0; i < 32; i++) {
            dst[i] = (float)(b->qs[i]) * d;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// PTX compilation (cached, NVRTC)
// ---------------------------------------------------------------------------

static PTX_CACHE: OnceLock<String> = OnceLock::new();

const QUANTIZE_FUNC: &str = "quantize_f32_to_q8_0";
const DEQUANT_FUNC: &str = "dequantize_q8_0_to_f32";

fn load_func(dev: &CudaDevice, fn_name: &str) -> Result<(candle_core::cuda_backend::cudarc::driver::CudaFunction,
            std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>)> {
    super::nvrtc_compile::get_or_load_func(dev, fn_name, "chimere_kv_q8_0", KV_Q8_0_KERNEL_SRC, &PTX_CACHE)
}

// ---------------------------------------------------------------------------
// GPU quantization: F32 -> Q8_0
// ---------------------------------------------------------------------------

/// Quantize `n_elements` f32 values to Q8_0 on GPU.
///
/// `input`: CudaView of at least `n_elements` f32 values.
/// `output`: CudaSlice<u8> of at least `q8_0_byte_size(n_elements)` bytes.
/// `n_elements` must be a multiple of 32.
pub fn quantize_q8_0_gpu(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<u8>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(
        n_elements % Q8_0_BLOCK_ELEMS == 0,
        "n_elements must be multiple of 32"
    );
    let n_blocks = n_elements / Q8_0_BLOCK_ELEMS;
    if n_blocks == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev, QUANTIZE_FUNC)?;

    let threads = 256u32;
    let blocks = ((n_blocks as u32) + threads - 1) / threads;
    let n_blocks_i32 = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&n_blocks_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("quantize_f32_to_q8_0 launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU dequantization: Q8_0 -> F32
// ---------------------------------------------------------------------------

/// Dequantize `n_elements` values from Q8_0 to f32 on GPU.
///
/// `input`: CudaView<u8> of at least `q8_0_byte_size(n_elements)` bytes.
/// `output`: CudaSlice<f32> of at least `n_elements` f32 values.
/// `n_elements` must be a multiple of 32.
pub fn dequantize_q8_0_gpu(
    input: &CudaView<'_, u8>,
    output: &mut CudaSlice<f32>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(
        n_elements % Q8_0_BLOCK_ELEMS == 0,
        "n_elements must be multiple of 32"
    );
    let n_blocks = n_elements / Q8_0_BLOCK_ELEMS;
    if n_blocks == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev, DEQUANT_FUNC)?;

    let threads = 256u32;
    let blocks = ((n_blocks as u32) + threads - 1) / threads;
    let n_blocks_i32 = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&n_blocks_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("dequantize_q8_0_to_f32 launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level Candle Tensor API
// ---------------------------------------------------------------------------

/// Quantize a Candle Tensor (F32, on GPU or CPU) to Q8_0 bytes.
///
/// Returns a flat `Vec<u8>` containing the quantized data.
/// The tensor is flattened before quantization; element count must be multiple of 32.
pub fn quantize_tensor_q8_0(tensor: &candle_core::Tensor) -> Result<Vec<u8>> {
    let n = tensor.elem_count();
    assert!(
        n % Q8_0_BLOCK_ELEMS == 0,
        "tensor element count ({n}) must be a multiple of {Q8_0_BLOCK_ELEMS}"
    );

    // Pull to CPU F32 for quantization
    let flat = tensor
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_device(&candle_core::Device::Cpu)?;
    let data = flat.to_vec1::<f32>()?;
    Ok(quantize_q8_0_cpu(&data))
}

/// Dequantize Q8_0 bytes to a Candle Tensor on the given device.
///
/// Returns a tensor of shape `shape` with dtype F32.
pub fn dequantize_bytes_q8_0(
    data: &[u8],
    n_elements: usize,
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<candle_core::Tensor> {
    let f32_data = dequantize_q8_0_cpu(data, n_elements);
    let t = candle_core::Tensor::from_vec(f32_data, shape, &candle_core::Device::Cpu)?;
    if matches!(device, candle_core::Device::Cpu) {
        Ok(t)
    } else {
        t.to_device(device)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_roundtrip_identity() {
        // Quantize then dequantize a known vector; verify round-trip accuracy.
        let n = 64; // 2 blocks of 32
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();

        let q8 = quantize_q8_0_cpu(&input);
        assert_eq!(q8.len(), q8_0_byte_size(n));

        let output = dequantize_q8_0_cpu(&q8, n);
        assert_eq!(output.len(), n);

        // Q8_0 quantization error: max ~1/127 relative + f16 scale rounding.
        // For values near zero (small |x|/max), the step size is max/127,
        // so relative error can be large. Use absolute tolerance for small values.
        for i in 0..n {
            let err = (input[i] - output[i]).abs();
            // Scale = max_abs / 127; quantization step = scale.
            // Max absolute error per element ~= 0.5 * scale = 0.5 * max_abs / 127.
            // For this test, max_abs ~ 3.2, so max step ~ 0.025, max abs error ~ 0.013.
            let max_err = if input[i].abs() < 0.1 {
                0.05 // absolute tolerance for near-zero values
            } else {
                input[i].abs() * 0.03 // 3% relative tolerance (covers f16 scale rounding)
            };
            assert!(
                err <= max_err,
                "element {i}: input={}, output={}, err={err}, max_err={max_err}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_q8_0_zeros() {
        let input = vec![0.0f32; 32];
        let q8 = quantize_q8_0_cpu(&input);
        let output = dequantize_q8_0_cpu(&q8, 32);
        assert!(output.iter().all(|&v| v == 0.0), "zeros should round-trip exactly");
    }

    #[test]
    fn test_q8_0_large_values() {
        // Values near the extremes of f16 range
        let mut input = vec![0.0f32; 32];
        input[0] = 1000.0;
        input[1] = -1000.0;
        input[31] = 500.0;

        let q8 = quantize_q8_0_cpu(&input);
        let output = dequantize_q8_0_cpu(&q8, 32);

        // Scale = 1000/127 ~ 7.87, so quantization step is ~7.87
        // For value 1000: q = 127, dequant = 127 * 7.87 = ~1000
        assert!(
            (output[0] - 1000.0).abs() < 10.0,
            "expected ~1000, got {}",
            output[0]
        );
        assert!(
            (output[1] + 1000.0).abs() < 10.0,
            "expected ~-1000, got {}",
            output[1]
        );
    }

    #[test]
    fn test_q8_0_byte_size() {
        assert_eq!(q8_0_byte_size(32), 34);
        assert_eq!(q8_0_byte_size(64), 68);
        assert_eq!(q8_0_byte_size(1024), 1024 / 32 * 34);
    }

    #[test]
    fn test_q8_0_memory_savings() {
        // Verify the expected memory savings vs F16
        let n_elements = 4096; // typical cache dimension
        let f16_bytes = n_elements * 2;
        let q8_bytes = q8_0_byte_size(n_elements);
        let savings = 1.0 - (q8_bytes as f64 / f16_bytes as f64);
        assert!(
            savings > 0.45 && savings < 0.50,
            "Expected ~47% savings, got {:.1}%",
            savings * 100.0
        );
        eprintln!(
            "Q8_0 memory: {} bytes vs F16 {} bytes ({:.1}% savings)",
            q8_bytes,
            f16_bytes,
            savings * 100.0
        );
    }
}
