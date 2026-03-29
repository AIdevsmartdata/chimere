//! # Scratch Buffer Pool — Chimere Engine
//!
//! Pre-allocated GPU buffers for the MoE shared expert path, eliminating
//! ~9 cudaMalloc/cudaFree per layer × 39 layers = 351 allocations per token.
//!
//! ## Problem
//!
//! Candle allocates a new CudaStorage for every tensor operation. The shared
//! expert path (gate, up, silu, mul, down, sigmoid, broadcast_mul) produces
//! ~9 intermediate tensors per MoE layer, each requiring cudaMalloc.
//!
//! ## Solution
//!
//! Pre-allocate fixed-size CudaSlice buffers at model init time:
//! - `shexp_gate_buf` / `shexp_up_buf`: [expert_ffn] = [512] for gate/up GEMV output
//! - `shexp_inter_buf`: [expert_ffn] = [512] for silu(gate) * up
//! - `shexp_out_buf`: [hidden_size] = [2048] for down GEMV output
//! - `shexp_gate_logit`: [1] for sigmoid gate scalar
//! - `shexp_q5k_bufs`: GgmlQ5KBuffers for Q8_1 input quantization (reused)
//!
//! Total: ~16 KB of GPU memory. Trivial vs model's 15 GB.
//!
//! ## Toggle
//!
//! Set `CHIMERE_SCRATCH_POOL=1` to enable. When disabled, falls back to
//! the existing Candle-based shared expert path.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Result};

use crate::kernels::q5k_mmvq_ggml::GgmlQ5KBuffers;

/// Check (once) whether the scratch pool is enabled via CHIMERE_SCRATCH_POOL env var.
pub fn scratch_pool_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        let enabled = std::env::var("CHIMERE_SCRATCH_POOL").is_ok();
        if enabled {
            eprintln!("[SCRATCH_POOL] Enabled — shared expert path uses pre-allocated buffers");
        }
        enabled
    });
    *ENABLED
}

/// Pre-allocated scratch buffers for the MoE shared expert forward pass.
///
/// These buffers are reused across all 39 MoE layers — each layer overwrites
/// them completely before reading.  Not snapshotted (stateless scratch).
pub struct ScratchPool {
    /// Shared expert gate GEMV output: [expert_ffn] = [512]
    pub shexp_gate_buf: CudaSlice<f32>,
    /// Shared expert up GEMV output: [expert_ffn] = [512]
    pub shexp_up_buf: CudaSlice<f32>,
    /// Shared expert SwiGLU intermediate: silu(gate) * up, [expert_ffn] = [512]
    pub shexp_inter_buf: CudaSlice<f32>,
    /// Shared expert down GEMV output: [hidden_size] = [2048]
    pub shexp_out_buf: CudaSlice<f32>,
    /// Sigmoid gate scalar: [1]
    pub shexp_gate_logit: CudaSlice<f32>,
    /// Pre-allocated Q5_K GEMV buffers (Q8_1 input + output, reused per projection).
    /// Max dimensions: ncols=2048 (hidden_size input), nrows=2048 (down_shexp output).
    pub shexp_q5k_bufs: GgmlQ5KBuffers,

    // Dimensions (stored for kernel launches)
    pub hidden_size: usize,
    pub expert_ffn: usize,
}

impl ScratchPool {
    /// Allocate all scratch buffers for the shared expert path.
    ///
    /// `hidden_size`: model hidden dimension (2048 for Qwen3.5-35B-A3B).
    /// `expert_ffn`: expert FFN hidden dimension (512 for Qwen3.5-35B-A3B).
    pub fn new(hidden_size: usize, expert_ffn: usize, device: &Device) -> Result<Self> {
        let Device::Cuda(dev) = device else {
            candle_core::bail!("ScratchPool::new: device must be CUDA");
        };

        let shexp_gate_buf = dev
            .alloc_zeros::<f32>(expert_ffn)
            .map_err(|e| candle_core::Error::Msg(format!("scratch shexp_gate: {e}")))?;
        let shexp_up_buf = dev
            .alloc_zeros::<f32>(expert_ffn)
            .map_err(|e| candle_core::Error::Msg(format!("scratch shexp_up: {e}")))?;
        let shexp_inter_buf = dev
            .alloc_zeros::<f32>(expert_ffn)
            .map_err(|e| candle_core::Error::Msg(format!("scratch shexp_inter: {e}")))?;
        let shexp_out_buf = dev
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| candle_core::Error::Msg(format!("scratch shexp_out: {e}")))?;
        let shexp_gate_logit = dev
            .alloc_zeros::<f32>(1)
            .map_err(|e| candle_core::Error::Msg(format!("scratch shexp_gate_logit: {e}")))?;

        // Q5_K GEMV buffers: max ncols = max(hidden_size, expert_ffn) = 2048
        // max nrows = max(expert_ffn, hidden_size) = 2048
        let shexp_q5k_bufs = GgmlQ5KBuffers::new(hidden_size, hidden_size, dev)?;

        let total_bytes = (expert_ffn * 3 + hidden_size + 1) * 4; // f32
        eprintln!(
            "[SCRATCH_POOL] Allocated shared expert buffers: {:.1} KB \
             (gate={} up={} inter={} out={} + Q5K GEMV bufs)",
            total_bytes as f64 / 1024.0,
            expert_ffn, expert_ffn, expert_ffn, hidden_size,
        );

        Ok(Self {
            shexp_gate_buf,
            shexp_up_buf,
            shexp_inter_buf,
            shexp_out_buf,
            shexp_gate_logit,
            shexp_q5k_bufs,
            hidden_size,
            expert_ffn,
        })
    }

    /// Run the shared expert forward pass using scratch buffers (zero cudaMalloc).
    ///
    /// This replaces the Candle-based shared expert path with raw CUDA kernel calls
    /// that write directly into pre-allocated buffers.
    ///
    /// # Arguments
    /// - `hidden_view`: input hidden state [hidden_size] as CudaView
    /// - `gate_raw`: Q5_K raw bytes for shared gate projection
    /// - `up_raw`: Q5_K raw bytes for shared up projection
    /// - `down_raw`: Q5_K raw bytes for shared down projection
    /// - `gate_inp_shexp`: shared expert scale/bias vector [hidden_size] as CudaView
    /// - `combined_buf`: routed expert combined output [hidden_size] — added in-place
    /// - `dev`: CUDA device handle
    ///
    /// # Returns
    /// Nothing — the result is written into `self.shexp_out_buf` which contains
    /// `routed + sigmoid_gate * shared_expert_out`. The caller wraps it into a Tensor.
    pub fn shared_expert_forward(
        &mut self,
        hidden_view: &CudaView<'_, f32>,
        gate_raw: &CudaView<'_, u8>,
        up_raw: &CudaView<'_, u8>,
        down_raw: &CudaView<'_, u8>,
        gate_inp_shexp: &CudaView<'_, f32>,
        combined_buf: &CudaSlice<f32>,
        dev: &CudaDevice,
    ) -> Result<()> {
        let hidden_size = self.hidden_size;
        let expert_ffn = self.expert_ffn;

        // --- 1. Gate projection: Q5_K GEMV [hidden_size -> expert_ffn] ---
        // Quantize hidden to Q8_1 (reused for gate AND up projections)
        crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
            hidden_view,
            &mut self.shexp_q5k_bufs.q8_input,
            hidden_size,
            dev,
        )?;

        // Gate GEMV: gate_raw [expert_ffn, hidden_size] @ hidden -> shexp_gate_buf [expert_ffn]
        crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
            gate_raw,
            &self.shexp_q5k_bufs.q8_input,
            &mut self.shexp_gate_buf,
            expert_ffn,
            hidden_size,
            dev,
        )?;

        // --- 2. Up projection: Q5_K GEMV [hidden_size -> expert_ffn] ---
        // Q8_1 input is the same (hidden), already quantized above.
        crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
            up_raw,
            &self.shexp_q5k_bufs.q8_input,
            &mut self.shexp_up_buf,
            expert_ffn,
            hidden_size,
            dev,
        )?;

        // --- 3. SwiGLU: silu(gate) * up -> inter ---
        crate::kernels::elementwise::raw_silu_mul(
            &self.shexp_gate_buf,
            &self.shexp_up_buf,
            &mut self.shexp_inter_buf,
            expert_ffn,
            dev,
        )?;

        // --- 4. Down projection: Q5_K GEMV [expert_ffn -> hidden_size] ---
        // Need to quantize the intermediate to Q8_1 (different size: expert_ffn=512)
        crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
            &self.shexp_inter_buf.slice(..),
            &mut self.shexp_q5k_bufs.q8_input,
            expert_ffn,
            dev,
        )?;

        crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
            down_raw,
            &self.shexp_q5k_bufs.q8_input,
            &mut self.shexp_out_buf,
            hidden_size,
            expert_ffn,
            dev,
        )?;

        // --- 5. Sigmoid gate: dot(hidden, gate_inp_shexp) -> scalar ---
        // gate_logit = hidden @ gate_inp_shexp (both [hidden_size])
        // This is a simple dot product -> one scalar.
        raw_dot_product(
            hidden_view,
            gate_inp_shexp,
            &mut self.shexp_gate_logit,
            hidden_size,
            dev,
        )?;

        // --- 6. Fused: shexp_out *= sigmoid(gate_logit), then shexp_out += combined ---
        // This replaces: sigmoid(gate_logit) -> broadcast_mul -> add
        // with a single fused kernel: out[i] = out[i] * sigmoid(gate_logit[0]) + combined[i]
        raw_sigmoid_gate_add_inplace(
            &mut self.shexp_out_buf,
            &self.shexp_gate_logit,
            combined_buf,
            hidden_size,
            dev,
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// New CUDA kernels for the scratch pool
// ---------------------------------------------------------------------------

/// CUDA kernel source for scratch pool operations.
const SCRATCH_KERNEL_SRC: &str = r#"
// Dot product: output[0] = sum(a[i] * b[i]) for i in 0..n
// Uses shared-memory parallel reduction.
// Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void dot_product_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    __shared__ float sdata[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += a[i] * b[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = sdata[0];
}

// Fused sigmoid-gate-add: out[i] = out[i] * sigmoid(gate[0]) + combined[i]
// `gate` is a single-element buffer containing the raw logit (pre-sigmoid).
// Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ combined,
    int n
) {
    float scale = 1.0f / (1.0f + expf(-gate[0]));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = output[i] * scale + combined[i];
    }
}
"#;

use std::sync::OnceLock;

static SCRATCH_PTX: OnceLock<String> = OnceLock::new();

fn get_scratch_ptx() -> &'static str {
    crate::kernels::nvrtc_compile::compile_and_cache(SCRATCH_KERNEL_SRC, &SCRATCH_PTX)
}

/// Dot product: output[0] = sum(a[i] * b[i])
///
/// Computes the dot product of two [n]-element vectors into a single scalar.
/// Used for the shared expert gate logit: `hidden @ gate_inp_shexp`.
pub fn raw_dot_product(
    a: &CudaView<'_, f32>,
    b: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let ptx = get_scratch_ptx();
    let func = dev.get_or_load_custom_func("dot_product_kernel", "chimere_scratch_v1", ptx)?;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(a);
    builder.arg(b);
    builder.arg(output);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("dot_product launch: {e}")))?;
    Ok(())
}

/// Fused sigmoid-gate-add: out[i] = out[i] * sigmoid(gate[0]) + combined[i]
///
/// Combines three operations (sigmoid, broadcast_mul, add) into one kernel:
/// 1. Read the scalar gate logit from `gate[0]`
/// 2. Compute `scale = sigmoid(gate[0])`
/// 3. For each element: `output[i] = output[i] * scale + combined[i]`
///
/// Replaces 3 Candle tensor operations (3 cudaMalloc) with 0 allocations.
pub fn raw_sigmoid_gate_add_inplace(
    output: &mut CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    combined: &CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let ptx = get_scratch_ptx();
    let func = dev.get_or_load_custom_func("sigmoid_gate_add_kernel", "chimere_scratch_v1", ptx)?;
    let n_i32 = n as i32;
    let blocks = ((n as u32) + 255) / 256;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(output);
    builder.arg(gate);
    builder.arg(combined);
    builder.arg(&n_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("sigmoid_gate_add launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    /// Helper: get Candle CUDA device and cudarc CudaDevice reference.
    fn get_cuda_device() -> Option<candle_core::Device> {
        candle_core::Device::new_cuda(0).ok()
    }

    fn cuda_dev(device: &candle_core::Device) -> &candle_core::cuda_backend::CudaDevice {
        match device {
            candle_core::Device::Cuda(d) => d,
            _ => panic!("not CUDA"),
        }
    }

    /// Helper: create a GPU CudaSlice from a Vec<f32> using alloc_zeros + add_inplace.
    fn to_gpu_slice(data: &[f32], device: &candle_core::Device) -> Result<CudaSlice<f32>> {
        let dev = cuda_dev(device);
        let n = data.len();
        // Upload via Candle Tensor, then copy into a fresh CudaSlice via a kernel
        let t = Tensor::from_slice(data, n, device)?;
        let (stor, lay) = t.storage_and_layout();
        let cuda_stor = match &*stor {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("not CUDA"),
        };
        let src_slice = cuda_stor.as_cuda_slice::<f32>()?;
        // Allocate output and copy via add_inplace (output starts at 0, so 0+src = src)
        let mut owned = dev.alloc_zeros::<f32>(n)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        crate::kernels::elementwise::raw_add_inplace(&mut owned, &src_slice, n, dev)?;
        drop(stor);
        Ok(owned)
    }

    /// Helper: read a GPU CudaSlice back to CPU.
    fn from_gpu_slice(slice: &CudaSlice<f32>, n: usize, device: &candle_core::Device) -> Result<Vec<f32>> {
        // Wrap as Tensor and read back
        let _ = device; // just for the signature
        let t_clone = slice.try_clone()
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let dev_arc = match device {
            candle_core::Device::Cuda(d) => d.clone(),
            _ => candle_core::bail!("not CUDA"),
        };
        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(t_clone, dev_arc);
        let t = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            candle_core::Shape::from_dims(&[n]),
            candle_core::op::BackpropOp::none(), false);
        t.to_device(&candle_core::Device::Cpu)?.to_vec1()
    }

    #[test]
    fn test_dot_product_kernel() -> Result<()> {
        let device = match get_cuda_device() {
            Some(d) => d,
            None => { eprintln!("Skipping: no CUDA device"); return Ok(()); }
        };
        let dev = cuda_dev(&device);

        let n = 2048usize;
        let a_data: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
        let b_data: Vec<f32> = vec![1.0f32; n];

        let a_gpu = to_gpu_slice(&a_data, &device)?;
        let b_gpu = to_gpu_slice(&b_data, &device)?;
        let mut out_gpu = dev.alloc_zeros::<f32>(1)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        raw_dot_product(&a_gpu.slice(..), &b_gpu.slice(..), &mut out_gpu, n, dev)?;

        let result = from_gpu_slice(&out_gpu, 1, &device)?;

        let expected = (n * (n + 1) / 2) as f32;
        let err = (result[0] - expected).abs() / expected;
        assert!(
            err < 1e-4,
            "dot product: expected {}, got {}, rel_err={:.6}",
            expected, result[0], err
        );
        eprintln!("test_dot_product_kernel: expected={} got={} rel_err={:.2e}", expected, result[0], err);
        Ok(())
    }

    #[test]
    fn test_sigmoid_gate_add_kernel() -> Result<()> {
        let device = match get_cuda_device() {
            Some(d) => d,
            None => { eprintln!("Skipping: no CUDA device"); return Ok(()); }
        };
        let dev = cuda_dev(&device);

        let n = 2048usize;
        let mut out_gpu = to_gpu_slice(&vec![1.0f32; n], &device)?;
        let gate_gpu = to_gpu_slice(&vec![0.0f32; 1], &device)?;
        let combined_gpu = to_gpu_slice(&vec![2.0f32; n], &device)?;

        raw_sigmoid_gate_add_inplace(&mut out_gpu, &gate_gpu, &combined_gpu, n, dev)?;

        let result = from_gpu_slice(&out_gpu, n, &device)?;

        for i in 0..n {
            let expected = 1.0f32 * 0.5 + 2.0;
            let err = (result[i] - expected).abs();
            assert!(
                err < 1e-5,
                "sigmoid_gate_add[{}]: expected {}, got {}, err={}",
                i, expected, result[i], err
            );
        }
        eprintln!("test_sigmoid_gate_add_kernel: all {} elements correct (expected 2.5)", n);

        // Test with non-zero gate: gate = [2.0], sigmoid(2) ~= 0.8808
        let mut out_gpu2 = to_gpu_slice(&vec![3.0f32; n], &device)?;
        let gate_gpu2 = to_gpu_slice(&vec![2.0f32; 1], &device)?;
        let combined_gpu2 = to_gpu_slice(&vec![1.0f32; n], &device)?;

        raw_sigmoid_gate_add_inplace(&mut out_gpu2, &gate_gpu2, &combined_gpu2, n, dev)?;

        let result2 = from_gpu_slice(&out_gpu2, n, &device)?;

        let sigmoid_2 = 1.0 / (1.0 + (-2.0f32).exp());
        let expected2 = 3.0 * sigmoid_2 + 1.0;
        let err2 = (result2[0] - expected2).abs();
        assert!(
            err2 < 1e-5,
            "sigmoid_gate_add (gate=2.0): expected {:.6}, got {:.6}, err={:.2e}",
            expected2, result2[0], err2
        );
        eprintln!("test_sigmoid_gate_add_kernel (gate=2.0): expected={:.6} got={:.6}", expected2, result2[0]);
        Ok(())
    }

    #[test]
    fn test_scratch_pool_allocation() -> Result<()> {
        let device = candle_core::Device::new_cuda(0);
        let device = match device {
            Ok(d) => d,
            Err(_) => { eprintln!("Skipping: no CUDA device"); return Ok(()); }
        };

        let pool = ScratchPool::new(2048, 512, &device)?;
        assert_eq!(pool.hidden_size, 2048);
        assert_eq!(pool.expert_ffn, 512);
        eprintln!("test_scratch_pool_allocation: ok");
        Ok(())
    }
}
