//! Raw cudarc forward pass — bypasses Candle for the MoE hot path.
//!
//! Extracts GPU pointers from Candle tensors at init time, then runs
//! the forward pass using cuBLAS GEMV + custom CUDA kernels directly.
//! Zero tensor allocation per token = zero dispatch overhead.
//!
//! ## Design
//!
//! Candle's per-token overhead (~52 ms / 74% of 70 ms) comes from:
//!   - Tensor allocation (Arc, Shape, Layout, Storage) per operation
//!   - Dispatch through the op-graph machinery
//!   - Multiple GPU→CPU sync points (top-K routing via `.to_vec1()`)
//!
//! This module bypasses all of that by:
//!   1. Pre-allocating all intermediate GPU buffers once at model load time.
//!   2. Keeping raw `CudaSlice<f32>` handles — no Candle wrappers at runtime.
//!   3. Running CUDA kernels directly via cudarc's `LaunchConfig` / `builder()`.
//!
//! ## Integration plan (incremental)
//!
//! - v0 (this file): struct + alloc + stubs. Compiles, zero runtime cost.
//! - v1: cuBLAS GEMV for router logits (removes the single hottest QMatMul).
//! - v2: CUDA top-K kernel to eliminate the GPU→CPU sync in expert selection.
//! - v3: Batched expert GEMM (all 8 experts in one cuBLAS call).
//! - v4: Full token loop without any Candle tensor allocation.

use candle_core::{Device, Module, Result, Tensor};
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::cuda_backend::CudaDevice;

// Q8_1 buffer size helpers.
// Q8_1 format: 36 bytes per block of 32 elements.
// q8_size(n) returns the byte count needed to hold Q8_1 quantization of n f32 values.
// n must be a multiple of 32.
fn q8_size(n_elements: usize) -> usize {
    debug_assert!(n_elements % 32 == 0, "n_elements must be multiple of 32");
    (n_elements / 32) * 36
}

// ---------------------------------------------------------------------------
// Model dimensions (Qwen3.5-35B-A3B)
// ---------------------------------------------------------------------------
//
// These are compile-time constants matching `Qwen35Config::qwen35_35b_a3b()`.
// If the config changes, update these constants and recompile.

/// Transformer hidden dimension (embedding size).
const HIDDEN_SIZE: usize = 2048;
/// Number of routed MoE experts per layer.
const NUM_EXPERTS: usize = 256;
/// Top-K experts selected per token.
const EXPERTS_PER_TOKEN: usize = 8;
/// Hidden dimension inside each expert's SwiGLU FFN.
const EXPERT_FFN_HIDDEN: usize = 512;
/// Vocabulary size (for the final lm_head projection).
const VOCAB_SIZE: usize = 248320;
/// SSM inner dimension (for GDN layers).
const SSM_D_INNER: usize = 4096;
/// QKV projection output size: num_heads * head_dim = 16 * 256 = 4096.
const QKV_SIZE: usize = 4096;

// ---------------------------------------------------------------------------
// RawGpuBuffers
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers reused every token. Zero allocation at runtime.
///
/// All fields are `CudaSlice<f32>` — raw device memory managed by cudarc.
/// They are never freed during the forward pass; they live as long as the model.
///
/// Buffer sizing rationale:
///   - `hidden`          : one token's embedding vector [HIDDEN_SIZE]
///   - `normed`          : RMSNorm(hidden)              [HIDDEN_SIZE]
///   - `qkv_buf`         : Q / K / V projection output  [QKV_SIZE] each, reused
///   - `gate_buf`        : router logits before softmax  [NUM_EXPERTS]
///   - `ssm_out_buf`     : GDN/SSM layer output          [HIDDEN_SIZE]
///   - `expert_gate_buf` : SwiGLU gate projection        [EXPERT_FFN_HIDDEN]
///   - `expert_up_buf`   : SwiGLU up projection          [EXPERT_FFN_HIDDEN]
///   - `expert_inter_buf`: gate * silu(up) intermediate  [EXPERT_FFN_HIDDEN]
///   - `expert_out_buf`  : down projection output        [HIDDEN_SIZE]
///   - `combined_buf`    : accumulated expert outputs    [HIDDEN_SIZE]
///   - `logits_buf`      : lm_head output logits         [VOCAB_SIZE]
///   - `router_probs_buf`: softmax router probabilities  [NUM_EXPERTS]
///   - `q8_hidden_buf`   : Q8_1 of normed_ffn input      [(HIDDEN_SIZE/32)*36 bytes]
///   - `q8_inter_buf`    : Q8_1 of expert intermediate   [(EXPERT_FFN_HIDDEN/32)*36 bytes]
pub struct RawGpuBuffers {
    pub hidden: CudaSlice<f32>,
    pub normed: CudaSlice<f32>,
    pub qkv_buf: CudaSlice<f32>,
    pub gate_buf: CudaSlice<f32>,
    pub ssm_out_buf: CudaSlice<f32>,
    pub expert_gate_buf: CudaSlice<f32>,
    pub expert_up_buf: CudaSlice<f32>,
    pub expert_inter_buf: CudaSlice<f32>,
    pub expert_out_buf: CudaSlice<f32>,
    pub combined_buf: CudaSlice<f32>,
    pub logits_buf: CudaSlice<f32>,
    pub router_probs_buf: CudaSlice<f32>,
    /// Q8_1 quantized buffer for the normed FFN hidden state.
    /// Size: (HIDDEN_SIZE / 32) * 36 bytes = 2304 bytes for hidden=2048.
    /// Quantized once per MoE layer; reused for all gate+up GEMVs.
    pub q8_hidden_buf: CudaSlice<u8>,
    /// Q8_1 quantized buffer for the expert intermediate (after silu_mul).
    /// Size: (EXPERT_FFN_HIDDEN / 32) * 36 bytes = 576 bytes for ffn=512.
    /// Quantized once per expert; used for the down GEMV.
    pub q8_inter_buf: CudaSlice<u8>,
    /// Batched Q8_1 intermediates for all top_k experts (fused v2 path).
    /// Size: EXPERTS_PER_TOKEN * (EXPERT_FFN_HIDDEN / 32) * 36 = 8 * 576 = 4608 bytes.
    /// Each slot `[k * q8_stride .. (k+1) * q8_stride]` holds the Q8_1 quantization
    /// of expert k's intermediate (after fused gate+up+silu).
    pub batched_q8_inter: CudaSlice<u8>,
    /// GPU-side output of the top-K softmax kernel: selected expert indices.
    /// Size: EXPERTS_PER_TOKEN * sizeof(i32) = 32 bytes for top_k=8.
    /// Written by `gpu_topk_softmax`, then dtoh'd (64 bytes total with top_wt).
    pub topk_indices_buf: CudaSlice<i32>,
    /// GPU-side output of the top-K softmax kernel: renormalised expert weights.
    /// Size: EXPERTS_PER_TOKEN * sizeof(f32) = 32 bytes for top_k=8.
    pub topk_weights_buf: CudaSlice<f32>,
    /// Batched gate GEMV output: [top_k * expert_ffn] = 8 * 512 = 4096 floats.
    /// Written by `gemv_iq3s_q8_batched` for all gate projections in one launch.
    pub batched_gate_out: CudaSlice<f32>,
    /// Batched up GEMV output: [top_k * expert_ffn] = 8 * 512 = 4096 floats.
    /// Written by `gemv_iq3s_q8_batched` for all up projections in one launch.
    pub batched_up_out: CudaSlice<f32>,
    /// Batched silu_mul intermediate: [top_k * expert_ffn] = 8 * 512 = 4096 floats.
    /// Written by `raw_silu_mul_batched` (1 launch for all experts).
    pub batched_inter_out: CudaSlice<f32>,
    /// Batched expert down-projection outputs: [top_k * hidden_size] = 8 * 2048 = 16384 floats.
    /// Each expert's down GEMV result is copied here; then `raw_weighted_combine` reduces them.
    pub batched_expert_outs: CudaSlice<f32>,
    /// GDN SSM scratch buffers — pre-allocated for CHIMERE_RAW_SSM=1.
    /// Eliminates 11 cudaMalloc per GDN layer (330/token → 0).
    pub gdn_scratch: Option<crate::kernels::elementwise::GdnScratchBuffers>,
    /// Raw QMatMul buffers — pre-allocated for CHIMERE_RAW_QMATMUL=1.
    /// Eliminates ~2 cudaMalloc per QMatMul call (Q8_1 input + output buffer).
    /// Shared across all 3 GDN projections (attn_qkv, attn_gate, ssm_out).
    pub qmatmul_bufs: Option<crate::kernels::raw_qmatmul::RawQMatMulBuffers>,
}

impl RawGpuBuffers {
    /// Allocate all buffers for the Qwen3.5-35B-A3B MoE model.
    ///
    /// Must be called once at model load time with the CUDA device handle.
    /// The device is extracted from a Candle `Device::Cuda` via pattern match.
    ///
    /// Total allocation: ~1.5 MB (negligible vs model weights at ~15 GB).
    pub fn new(device: &Device) -> Result<Self> {
        let Device::Cuda(dev) = device else {
            candle_core::bail!("RawGpuBuffers::new: device must be CUDA");
        };

        // All buffers initialised to zero. Because these are pre-allocated and
        // always fully overwritten before being read, the zero-init cost is
        // paid once at startup (not per token).
        let hidden = dev
            .alloc_zeros::<f32>(HIDDEN_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc hidden: {e}")))?;
        let normed = dev
            .alloc_zeros::<f32>(HIDDEN_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc normed: {e}")))?;
        // qkv_buf is reused for Q, K, and V in sequence — size = max(QKV_SIZE, HIDDEN_SIZE)
        let qkv_buf = dev
            .alloc_zeros::<f32>(QKV_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc qkv_buf: {e}")))?;
        let gate_buf = dev
            .alloc_zeros::<f32>(NUM_EXPERTS)
            .map_err(|e| candle_core::Error::Msg(format!("alloc gate_buf: {e}")))?;
        let ssm_out_buf = dev
            .alloc_zeros::<f32>(SSM_D_INNER)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ssm_out_buf: {e}")))?;
        let expert_gate_buf = dev
            .alloc_zeros::<f32>(EXPERT_FFN_HIDDEN)
            .map_err(|e| candle_core::Error::Msg(format!("alloc expert_gate_buf: {e}")))?;
        let expert_up_buf = dev
            .alloc_zeros::<f32>(EXPERT_FFN_HIDDEN)
            .map_err(|e| candle_core::Error::Msg(format!("alloc expert_up_buf: {e}")))?;
        let expert_inter_buf = dev
            .alloc_zeros::<f32>(EXPERT_FFN_HIDDEN)
            .map_err(|e| candle_core::Error::Msg(format!("alloc expert_inter_buf: {e}")))?;
        let expert_out_buf = dev
            .alloc_zeros::<f32>(HIDDEN_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc expert_out_buf: {e}")))?;
        let combined_buf = dev
            .alloc_zeros::<f32>(HIDDEN_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc combined_buf: {e}")))?;
        let logits_buf = dev
            .alloc_zeros::<f32>(VOCAB_SIZE)
            .map_err(|e| candle_core::Error::Msg(format!("alloc logits_buf: {e}")))?;
        let router_probs_buf = dev
            .alloc_zeros::<f32>(NUM_EXPERTS)
            .map_err(|e| candle_core::Error::Msg(format!("alloc router_probs_buf: {e}")))?;

        // Q8_1 quantization buffers for amortized quantization in moe_ffn_raw.
        // Size: (n / 32) * 36 bytes.
        let q8_hidden_size = q8_size(HIDDEN_SIZE);      // 2304 bytes for hidden=2048
        let q8_inter_size  = q8_size(EXPERT_FFN_HIDDEN); //  576 bytes for ffn=512
        let q8_hidden_buf = dev
            .alloc_zeros::<u8>(q8_hidden_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8_hidden_buf: {e}")))?;
        let q8_inter_buf = dev
            .alloc_zeros::<u8>(q8_inter_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8_inter_buf: {e}")))?;

        // Batched Q8_1 intermediates for fused v2 path:
        // 8 experts * (512 / 32) * 36 = 8 * 576 = 4608 bytes.
        let batched_q8_inter_size = EXPERTS_PER_TOKEN * q8_inter_size;
        let batched_q8_inter = dev
            .alloc_zeros::<u8>(batched_q8_inter_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_q8_inter: {e}")))?;

        // Top-K output buffers: 8 indices (i32) + 8 weights (f32) = 64 bytes total.
        // Written by gpu_topk_softmax; then 64 bytes dtoh'd back to CPU.
        let topk_indices_buf = dev
            .alloc_zeros::<i32>(EXPERTS_PER_TOKEN)
            .map_err(|e| candle_core::Error::Msg(format!("alloc topk_indices_buf: {e}")))?;
        let topk_weights_buf = dev
            .alloc_zeros::<f32>(EXPERTS_PER_TOKEN)
            .map_err(|e| candle_core::Error::Msg(format!("alloc topk_weights_buf: {e}")))?;

        // Batched GEMV output buffers: [top_k * expert_ffn] floats each.
        // Used by gemv_iq3s_q8_batched to write all gate (or up) projections in one launch.
        let batched_size = EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN;  // 8 * 512 = 4096
        let batched_gate_out = dev
            .alloc_zeros::<f32>(batched_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_gate_out: {e}")))?;
        let batched_up_out = dev
            .alloc_zeros::<f32>(batched_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_up_out: {e}")))?;

        // Batched silu_mul intermediate: same size as batched gate/up outputs.
        let batched_inter_out = dev
            .alloc_zeros::<f32>(batched_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_inter_out: {e}")))?;

        // Batched expert outputs for weighted_combine: [top_k * hidden_size].
        let batched_expert_outs_size = EXPERTS_PER_TOKEN * HIDDEN_SIZE;  // 8 * 2048 = 16384
        let batched_expert_outs = dev
            .alloc_zeros::<f32>(batched_expert_outs_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_expert_outs: {e}")))?;

        // GDN scratch buffers (only if CHIMERE_RAW_SSM is set)
        let gdn_scratch = if std::env::var("CHIMERE_RAW_SSM").is_ok() {
            let key_dim = 16 * 128;      // n_group * d_state = 2048
            let value_dim = 32 * 128;    // dt_rank * d_state = 4096
            let conv_channels = key_dim * 2 + value_dim;  // 8192
            Some(crate::kernels::elementwise::GdnScratchBuffers::new(
                conv_channels, 4, key_dim, value_dim, 32, 128, dev,
            )?)
        } else {
            None
        };

        // Raw QMatMul buffers (only if CHIMERE_RAW_QMATMUL is set)
        // Eliminates ~2 cudaMalloc per QMatMul call for 3 GDN projections × 30 layers.
        // max_input_cols = max(HIDDEN_SIZE, SSM_D_INNER) = 4096 (ssm_out input is value_dim=4096)
        // max_output_rows = max(conv_channels=8192, value_dim=4096, hidden_size=2048) = 8192
        let qmatmul_bufs = if std::env::var("CHIMERE_RAW_QMATMUL").is_ok() {
            let max_input_cols = SSM_D_INNER; // 4096 — largest input dim (ssm_out)
            let max_output_rows = 8192;       // conv_channels (attn_qkv)
            Some(crate::kernels::raw_qmatmul::RawQMatMulBuffers::new(
                max_input_cols, max_output_rows, dev,
            )?)
        } else {
            None
        };

        Ok(Self {
            hidden,
            normed,
            qkv_buf,
            gate_buf,
            ssm_out_buf,
            expert_gate_buf,
            expert_up_buf,
            expert_inter_buf,
            expert_out_buf,
            combined_buf,
            logits_buf,
            router_probs_buf,
            q8_hidden_buf,
            q8_inter_buf,
            batched_q8_inter,
            topk_indices_buf,
            topk_weights_buf,
            batched_gate_out,
            batched_up_out,
            batched_inter_out,
            batched_expert_outs,
            gdn_scratch,
            qmatmul_bufs,
        })
    }

    /// Total GPU memory used by these buffers, in bytes.
    pub fn bytes_allocated(&self) -> usize {
        let f32_bytes = (HIDDEN_SIZE         // hidden
            + HIDDEN_SIZE    // normed
            + QKV_SIZE       // qkv_buf
            + NUM_EXPERTS    // gate_buf
            + SSM_D_INNER    // ssm_out_buf
            + EXPERT_FFN_HIDDEN  // expert_gate_buf
            + EXPERT_FFN_HIDDEN  // expert_up_buf
            + EXPERT_FFN_HIDDEN  // expert_inter_buf
            + HIDDEN_SIZE    // expert_out_buf
            + HIDDEN_SIZE    // combined_buf
            + VOCAB_SIZE     // logits_buf
            + NUM_EXPERTS    // router_probs_buf
            + EXPERTS_PER_TOKEN  // topk_weights_buf
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN  // batched_gate_out
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN  // batched_up_out
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN  // batched_inter_out
            + EXPERTS_PER_TOKEN * HIDDEN_SIZE)        // batched_expert_outs
            * std::mem::size_of::<f32>();
        let u8_bytes = q8_size(HIDDEN_SIZE)       // q8_hidden_buf
            + q8_size(EXPERT_FFN_HIDDEN)           // q8_inter_buf
            + EXPERTS_PER_TOKEN * q8_size(EXPERT_FFN_HIDDEN);  // batched_q8_inter
        let i32_bytes = EXPERTS_PER_TOKEN          // topk_indices_buf
            * std::mem::size_of::<i32>();
        // qmatmul_bufs: Q8_1 input + f32 output (when enabled)
        let qmatmul_bytes = if self.qmatmul_bufs.is_some() {
            // Q8_1 input: padded to 512 alignment, (n/32)*36 bytes
            // max_input_cols = SSM_D_INNER = 4096 → ((4096+511)/512*512) = 4096 → 4096*36/32 = 4608 bytes
            let ncols_padded = (SSM_D_INNER + 511) / 512 * 512;
            let q8_in = (ncols_padded / 32) * 36;
            // f32 output: 8192 floats = 32768 bytes
            let f32_out = 8192 * std::mem::size_of::<f32>();
            q8_in + f32_out
        } else {
            0
        };
        f32_bytes + u8_bytes + i32_bytes + qmatmul_bytes
    }
}

// (extract_gpu_ptr, RawForwardArgs, forward_token_raw removed — dead code)

// ---------------------------------------------------------------------------
// MoE FFN raw forward — zero Candle Tensor allocation on the hot path
// ---------------------------------------------------------------------------

/// Softmax over a CPU slice in-place.
///
/// Numerically stable: subtract max before exp, then normalise.
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Select top-`k` entries from `probs` by value.
///
/// Returns `(expert_idx, renormalised_weight)` pairs sorted by weight descending.
/// Weights are renormalised so they sum to 1.0.
fn topk_cpu(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(probs.len());
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    // Partial sort: bring top-k to the front.
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top = indexed[..k].to_vec();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let weight_sum: f32 = top.iter().map(|(_, w)| w).sum();
    let scale = if weight_sum > 1e-12 {
        1.0 / weight_sum
    } else {
        1.0 / k as f32
    };
    top.iter().map(|(i, w)| (*i, w * scale)).collect()
}

// (dtoh_f32 removed — dead code)

/// Zero a `CudaSlice<f32>` using GPU-side `cuMemsetD8Async`.
///
/// This avoids the CPU allocation + PCIe memcpy_htod that the old implementation
/// performed 40 times per token (once per MoE layer).
fn zero_slice(slice: &mut CudaSlice<f32>, _n: usize, dev: &CudaDevice) -> Result<()> {
    dev.cuda_stream()
        .memset_zeros(slice)
        .map_err(|e| candle_core::Error::Msg(format!("zero_slice memset: {e}")))
}

/// Direct MoE FFN using NVRTC GEMV + IQ3_S kernel — zero Candle Tensor allocation
/// on the hot path (router + expert accumulation).
///
/// ## Operation sequence
///
/// 1. **Router GEMV**: `router_probs_buf = router_weight @ hidden`
///    (F32 row-major GEMV via NVRTC `f32_gemv_kernel`)
/// 2. **Softmax + top-K** on CPU (256 floats = 1 KB transfer via dtoh).
/// 3. **For each selected expert** (`top_k` experts, default 8):
///    a. IQ3_S dequant gate/up projections into fresh Tensors (one allocation each)
///    b. Dequanted gate/up → two F32 GEMV → `expert_gate_buf`, `expert_up_buf`
///    c. Fused `silu_mul` → `expert_inter_buf`
///    d. IQ3_S dequant down projection → one fresh Tensor
///    e. F32 GEMV down → `expert_out_buf`
///    f. `weighted_add`: `combined_buf += weight * expert_out_buf`
/// 4. **Shared expert** (always active, F32 weights on GPU):
///    a. F32 GEMV gate+up → `expert_gate_buf`, `expert_up_buf` (reuse scratch)
///    b. Fused `silu_mul` → `expert_inter_buf`
///    c. F32 GEMV down → `expert_out_buf`
///    d. `sigmoid_mul_acc`: `combined_buf += sigmoid(sh_gate_bias) * expert_out_buf`
/// 5. Write `combined_buf` → `output`.
///
/// ## Allocation note
///
/// The IQ3_S dequant step (step 3a, 3d) still allocates one F32 `CudaSlice`
/// per expert per projection (6 allocations × top_k experts = 48 for top-8).
/// This is a known limitation: `dequant_iq3s_at_offset` writes to a freshly
/// allocated slice to wrap into a Candle Tensor.  A future pass can add a
/// `dequant_iq3s_into_slice` variant that writes into a pre-allocated buffer,
/// eliminating these too.  The router overhead (the hottest path: one call per
/// token regardless of expert count) is now fully allocation-free.
///
/// ## Dimensions (Qwen3.5-35B-A3B defaults)
///
/// | Symbol         | Value  | Description                          |
/// |----------------|--------|--------------------------------------|
/// | `hidden_size`  | 2048   | Transformer hidden dim               |
/// | `expert_ffn`   | 512    | Expert FFN hidden dim (SwiGLU)       |
/// | `num_experts`  | 256    | Total routed experts                 |
/// | `top_k`        | 8      | Experts selected per token           |
/// Raw MoE FFN for routed experts only (shared expert handled by caller in Candle).
/// All weight inputs are CudaView — borrowed from Candle storage guards, zero copy.
/// Result is written to `buffers.combined_buf`.
pub fn moe_ffn_raw(
    // Input (CudaView — borrowed from Candle Tensor storage guard)
    hidden: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,

    // Pre-selected experts (computed by Candle router — correct)
    selected_experts: &[(usize, f32)],  // (expert_idx, weight) already softmax'd + top-K'd

    // Expert weights (IQ3_S raw bytes, CudaView into the full stacked tensor)
    gate_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    up_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    down_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    expert_bytes_gate: usize,
    expert_bytes_up: usize,
    expert_bytes_down: usize,
    _expert_elements_gate: usize,
    _expert_elements_up: usize,
    _expert_elements_down: usize,

    // Pre-allocated scratch buffers (reused every token)
    buffers: &mut RawGpuBuffers,

    // Dimensions
    hidden_size: usize,
    expert_ffn: usize,
    _num_experts: usize,
    top_k: usize,

    // CUDA device
    dev: &CudaDevice,
    _device: &Device,
) -> Result<()> {
    use crate::deltanet_kernel::{
        quantize_f32_to_q8_1_gpu,
        gemv_iq3s_fused_at_offset_q8,
        gemv_iq3s_q8_batched,
    };
    use crate::kernels::{
        raw_silu_mul_batched,
        raw_weighted_combine,
        gemv_iq3s_q8_batched_multi_input,
        quantize_f32_to_q8_1_batched_gpu,
    };

    // -----------------------------------------------------------------------
    // 1. Router done by Candle caller — selected_experts passed in directly.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // 2. Zero the combined accumulator buffer.
    // -----------------------------------------------------------------------
    zero_slice(&mut buffers.combined_buf, hidden_size, dev)?;

    // -----------------------------------------------------------------------
    // 3. Quantize the shared hidden input to Q8_1 ONCE.
    //
    //    The gate and up projections of all top_k experts all read the same
    //    `hidden` (normed_ffn) vector.
    // -----------------------------------------------------------------------
    quantize_f32_to_q8_1_gpu(hidden, &mut buffers.q8_hidden_buf, hidden_size, dev)?;

    // -----------------------------------------------------------------------
    // 4. Expert IDs are already on GPU from gpu_topk_softmax
    //    (written to buffers.topk_indices_buf). No upload needed.
    //    Use the top_k parameter (always 8) — gpu_topk_softmax always
    //    selects exactly top_k experts.
    // -----------------------------------------------------------------------
    let actual_top_k = top_k;

    // -----------------------------------------------------------------------
    // Toggle: CHIMERE_FUSED_MOE_V2=1 activates the 5-launch MoE FFN path.
    //
    // v2 path (5 launches, matching ik_llama):
    //   1. Q8_1 quantize hidden (done above, 1 launch)
    //   2. Fused gate+up+silu (1 launch)
    //   3. Batched Q8_1 quantize all experts' intermediates (1 launch)
    //   4. Batched down multi-input (1 launch)
    //   5. Weighted combine (1 launch)
    //
    // v1 path (fallback, ~29 launches):
    //   batched gate (1) + batched up (1) + batched silu_mul (1)
    //   + per-expert (quantize + down + dtod) (8 × 3 = 24)
    //   + weighted combine (1)
    // -----------------------------------------------------------------------
    let use_fused_v2 = {
        use once_cell::sync::Lazy;
        static V2: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FUSED_MOE_V2").is_ok());
        *V2
    };

    if use_fused_v2 {
        // ===================================================================
        // FUSED V2 PATH: 5 kernel launches (matching ik_llama)
        // ===================================================================

        // -------------------------------------------------------------------
        // Step 2: Fused gate+up+silu (1 launch replaces 3)
        //
        //   For each expert k and output row r:
        //     batched_inter_out[k * expert_ffn + r] =
        //       silu(gate_dot(expert_ids[k], r)) * up_dot(expert_ids[k], r)
        // -------------------------------------------------------------------
        crate::kernels::fused_gate_up::fused_gate_up_silu_iq3s(
            gate_exps,
            up_exps,
            &buffers.q8_hidden_buf,
            &mut buffers.batched_inter_out,
            &buffers.topk_indices_buf,
            hidden_size,
            expert_ffn,
            actual_top_k,
            expert_bytes_gate,  // gate and up have same stride
            dev,
        )?;

        // -------------------------------------------------------------------
        // Step 3: Batched Q8_1 quantize all experts' intermediates (1 launch)
        //
        //   batched_inter_out is [top_k * expert_ffn] contiguous f32.
        //   batched_q8_inter is [top_k * q8_stride] contiguous u8.
        //   Single kernel launch quantizes all top_k vectors in parallel,
        //   writing directly to the correct Q8_1 slot per expert.
        //   Replaces 8 quantize launches + 8 dtod copies with 1 launch.
        // -------------------------------------------------------------------
        let q8_inter_stride = q8_size(expert_ffn);  // 576 bytes per expert
        quantize_f32_to_q8_1_batched_gpu(
            &buffers.batched_inter_out,
            &mut buffers.batched_q8_inter,
            expert_ffn,
            actual_top_k,
            dev,
        )?;

        // -------------------------------------------------------------------
        // Step 4: Batched down GEMV with different Q8_1 input per expert
        //   (1 launch replaces 8 individual down GEMVs + 8 dtod copies)
        //
        //   output[k * hidden_size .. (k+1) * hidden_size] =
        //     down_weights[expert_ids[k]] @ batched_q8_inter[k * q8_stride..]
        // -------------------------------------------------------------------
        gemv_iq3s_q8_batched_multi_input(
            down_exps,
            &buffers.batched_q8_inter,
            &mut buffers.batched_expert_outs,
            &buffers.topk_indices_buf,
            expert_ffn,       // cols (down proj input = expert FFN hidden)
            hidden_size,      // rows (down proj output = transformer hidden)
            actual_top_k,
            expert_bytes_down,
            q8_inter_stride,
            dev,
        )?;

        // -------------------------------------------------------------------
        // Step 5: Weighted combine (1 launch)
        //   Weights are already on GPU from gpu_topk_softmax. No upload needed.
        // -------------------------------------------------------------------

        raw_weighted_combine(
            &buffers.batched_expert_outs,
            &buffers.topk_weights_buf,
            &mut buffers.combined_buf,
            hidden_size,
            actual_top_k,
            dev,
        )?;

    } else {
        // ===================================================================
        // V1 FALLBACK PATH: ~29 launches (existing flow)
        // ===================================================================

        // -------------------------------------------------------------------
        // 5. Batched gate GEMV: 1 launch for all top_k experts' gate projection.
        //    Output: batched_gate_out[k * expert_ffn .. (k+1) * expert_ffn]
        // -------------------------------------------------------------------
        gemv_iq3s_q8_batched(
            gate_exps,
            &buffers.q8_hidden_buf,
            &mut buffers.batched_gate_out,
            &buffers.topk_indices_buf,
            hidden_size,      // cols
            expert_ffn,       // rows
            actual_top_k,
            expert_bytes_gate, // expert_stride in bytes
            dev,
        )?;

        // -------------------------------------------------------------------
        // 6. Batched up GEMV: 1 launch for all top_k experts' up projection.
        //    Output: batched_up_out[k * expert_ffn .. (k+1) * expert_ffn]
        // -------------------------------------------------------------------
        gemv_iq3s_q8_batched(
            up_exps,
            &buffers.q8_hidden_buf,
            &mut buffers.batched_up_out,
            &buffers.topk_indices_buf,
            hidden_size,      // cols
            expert_ffn,       // rows
            actual_top_k,
            expert_bytes_up,  // expert_stride in bytes
            dev,
        )?;

        // -------------------------------------------------------------------
        // 7. Batched silu_mul: 1 launch instead of top_k separate launches.
        //
        //    gate_all = batched_gate_out [top_k * expert_ffn]
        //    up_all   = batched_up_out   [top_k * expert_ffn]
        //    inter_all= batched_inter_out[top_k * expert_ffn]  (output)
        // -------------------------------------------------------------------
        raw_silu_mul_batched(
            &buffers.batched_gate_out,
            &buffers.batched_up_out,
            &mut buffers.batched_inter_out,
            expert_ffn,
            actual_top_k,
            dev,
        )?;

        // -------------------------------------------------------------------
        // 8. Per-expert: Q8_1 quantize intermediate + down GEMV.
        //
        //    Each expert's intermediate (after silu_mul) must be quantized and
        //    then projected through the down matrix individually (different
        //    expert weights). Results are collected into batched_expert_outs.
        // -------------------------------------------------------------------
        for (k, &(expert_idx, _weight)) in selected_experts.iter().enumerate() {
            // Quantize the k-th slice of batched_inter_out to Q8_1.
            let inter_view = buffers.batched_inter_out.slice(k * expert_ffn..);
            quantize_f32_to_q8_1_gpu(&inter_view, &mut buffers.q8_inter_buf, expert_ffn, dev)?;

            // Down: [hidden_size, expert_ffn] → expert_out_buf [hidden_size]
            gemv_iq3s_fused_at_offset_q8(
                down_exps, expert_idx * expert_bytes_down,
                &buffers.q8_inter_buf,
                &mut buffers.expert_out_buf,
                hidden_size, expert_ffn, dev)?;

            // Copy expert_out_buf → batched_expert_outs[k * hidden_size..(k+1) * hidden_size]
            // Device-to-device copy via candle's CudaDevice.
            {
                let mut dst = buffers.batched_expert_outs.slice_mut(
                    k * hidden_size..(k + 1) * hidden_size,
                );
                dev.memcpy_dtod(&buffers.expert_out_buf, &mut dst)?;
            }
        }

        // -------------------------------------------------------------------
        // 9. Upload expert weights to GPU and batched weighted combine.
        //
        //    topk_weights_buf already exists on GPU. Upload the softmax'd
        //    weights from selected_experts, then 1 kernel launch to compute:
        //    combined_buf[i] = sum_k(weights[k] * expert_outs[k * hidden + i])
        // -------------------------------------------------------------------
        {
            let weights_cpu: Vec<f32> = selected_experts.iter().map(|(_, w)| *w).collect();
            dev.memcpy_htod(&weights_cpu, &mut buffers.topk_weights_buf)
                .map_err(|e| candle_core::Error::Msg(format!("upload topk_weights: {e}")))?;
        }

        raw_weighted_combine(
            &buffers.batched_expert_outs,
            &buffers.topk_weights_buf,
            &mut buffers.combined_buf,
            hidden_size,
            actual_top_k,
            dev,
        )?;
    }

    // Shared expert is handled by the caller in Candle (0.02ms/layer, trivial).
    // Result of routed experts is in buffers.combined_buf.
    Ok(())
}

/// Fused MoE FFN: single kernel launch for all 8 experts.
///
/// Replaces 24 individual GEMV launches + 8 silu_mul + 8 weighted_add
/// with a single `fused_moe_iq3s` kernel launch.
/// Output is in `buffers.combined_buf`.
pub fn moe_ffn_fused(
    hidden: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    _selected_experts: &[(usize, f32)],
    gate_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    up_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    down_exps: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, u8>,
    expert_bytes_gate: usize,
    expert_bytes_up: usize,
    expert_bytes_down: usize,
    buffers: &mut RawGpuBuffers,
    hidden_size: usize,
    expert_ffn: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // Zero the output accumulator
    zero_slice(&mut buffers.combined_buf, hidden_size, dev)?;

    // GPU-resident path: expert_ids and weights are already on GPU
    // from gpu_topk_softmax (written to buffers.topk_indices_buf/topk_weights_buf).
    // No CPU round-trip needed.
    crate::kernels::fused_moe::fused_moe_iq3s_gpu_resident(
        hidden,
        gate_exps,
        up_exps,
        down_exps,
        &buffers.topk_indices_buf,
        &buffers.topk_weights_buf,
        &mut buffers.combined_buf,
        hidden_size,
        expert_ffn,
        expert_bytes_gate,
        expert_bytes_up,
        expert_bytes_down,
        top_k,
        dev,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Raw GDN Layer Forward — zero Candle Tensor operations
// ---------------------------------------------------------------------------
//
// Replaces `forward_gdn_layer_moe` with direct CudaSlice kernel calls.
// Pre-allocated buffers are reused across layers (zero cudaMalloc).
//
// See docs/raw-forward-v4-design.md Section 4.2 for the full call sequence.

use candle_core::cuda_backend::cudarc::driver::CudaView;

/// Q8_1 buffer size: pad ncols to 512 alignment, then (n/32)*36 bytes.
fn q8_buf_size(ncols: usize) -> usize {
    let padded = (ncols + 511) / 512 * 512;
    (padded / 32) * 36
}

/// Pre-allocated scratch buffers for one raw GDN layer forward pass.
///
/// Allocated once at model load time, reused every GDN layer every token.
/// All buffers are fully overwritten before being read — the zero-init cost
/// is paid once at startup.
///
/// Buffer dimensions are for Qwen3.5-35B-A3B:
///   hidden_size=2048, conv_channels=8192, value_dim=4096, key_dim=2048,
///   dt_rank=32, d_state=128, n_group=16.
pub struct RawGdnLayerBuffers {
    // -- Norm output --
    /// RMSNorm(hidden) -> [hidden_size]
    pub normed: CudaSlice<f32>,

    // -- Q8_1 quantization scratch --
    /// Q8_1 of normed hidden (2048 floats) — reused for qkv/gate/beta/alpha GEMVs
    pub q8_hidden: CudaSlice<u8>,
    /// Q8_1 of gated SSM output (4096 floats) — for ssm_out projection
    pub q8_gated: CudaSlice<u8>,

    // -- SSM projection outputs (written by Q5K GEMV) --
    /// QKV projection: [conv_channels=8192]
    pub qkv_buf: CudaSlice<f32>,
    /// Gate (z) projection: [value_dim=4096]
    pub gate_buf: CudaSlice<f32>,
    /// Beta projection before sigmoid: [dt_rank=32]
    pub beta_proj: CudaSlice<f32>,
    /// Alpha projection before softplus: [dt_rank=32]
    pub alpha_proj: CudaSlice<f32>,

    // -- Fused beta/alpha/gate outputs --
    /// sigmoid(beta_proj): [dt_rank=32]
    pub beta_out: CudaSlice<f32>,
    /// exp(softplus(alpha+bias)*a): [dt_rank=32]
    pub gate_exp_out: CudaSlice<f32>,

    // -- Conv1d + split + norm + expand --
    /// Conv1d + SiLU output: [conv_channels=8192]
    pub conv_output: CudaSlice<f32>,
    /// Q split from conv_output: [key_dim=2048]
    pub q_split: CudaSlice<f32>,
    /// K split from conv_output: [key_dim=2048]
    pub k_split: CudaSlice<f32>,
    /// L2-normed Q per group: [key_dim=2048]
    pub q_normed: CudaSlice<f32>,
    /// L2-normed K per group: [key_dim=2048]
    pub k_normed: CudaSlice<f32>,
    /// Q expanded to dt_rank heads: [value_dim=4096]
    pub q_expanded: CudaSlice<f32>,
    /// K expanded to dt_rank heads: [value_dim=4096]
    pub k_expanded: CudaSlice<f32>,
    /// Q scaled by 1/sqrt(d_state): [value_dim=4096]
    pub q_scaled: CudaSlice<f32>,
    /// V copied from conv_output: [value_dim=4096]
    pub v_copy: CudaSlice<f32>,

    // -- DeltaNet output --
    /// DeltaNet step output: [value_dim=4096]
    pub ssm_output: CudaSlice<f32>,

    // -- Post-SSM path --
    /// Fused RMSNorm + SiLU gate: [value_dim=4096]
    pub gated: CudaSlice<f32>,
    /// SSM output projection result: [hidden_size=2048]
    pub projected: CudaSlice<f32>,

    // -- Conv state temp --
    /// Temporary buffer for new conv state: [conv_channels * (conv_kernel-1)]
    /// Written by fused_conv1d_silu_update, then copied back to the layer's conv_state.
    pub new_conv_state: CudaSlice<f32>,

    // -- FFN path --
    /// RMSNorm for FFN input: [hidden_size=2048]
    pub normed_ffn: CudaSlice<f32>,
}

/// Double-buffer for DeltaNet recurrent state.
///
/// The deltanet_step kernel reads state_in and writes state_out. Rust's borrow
/// checker prevents simultaneous `&` and `&mut` access to the same CudaSlice.
/// This struct provides safe alternating access via `current()` / `next_mut()` / `swap()`.
///
/// Each buffer is [dt_rank * d_state * d_state] = [32 * 128 * 128] = 524,288 f32 = 2 MB.
pub struct GdnStateDoubleBuffer {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    current_is_a: bool,
}

impl GdnStateDoubleBuffer {
    /// Allocate a double buffer, zeroed.
    pub fn new(state_size: usize, dev: &CudaDevice) -> Result<Self> {
        let a = dev.alloc_zeros::<f32>(state_size)
            .map_err(|e| candle_core::Error::Msg(format!("GdnStateDoubleBuffer alloc a: {e}")))?;
        let b = dev.alloc_zeros::<f32>(state_size)
            .map_err(|e| candle_core::Error::Msg(format!("GdnStateDoubleBuffer alloc b: {e}")))?;
        Ok(Self { a, b, current_is_a: true })
    }

    /// Get both the current (read) and next (write) state simultaneously.
    ///
    /// This returns `(&current, &mut next)` without triggering Rust's borrow
    /// checker, because `a` and `b` are disjoint fields.
    pub fn current_and_next(&mut self) -> (&CudaSlice<f32>, &mut CudaSlice<f32>) {
        if self.current_is_a {
            (&self.a, &mut self.b)
        } else {
            (&self.b, &mut self.a)
        }
    }

    /// Swap: what was `next` becomes `current`.
    pub fn swap(&mut self) {
        self.current_is_a = !self.current_is_a;
    }
}

impl RawGdnLayerBuffers {
    /// Allocate all scratch buffers for one GDN layer forward pass.
    pub fn new(config: &crate::config::Qwen35Config, dev: &CudaDevice) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let key_dim = config.ssm_n_group * config.ssm_d_state;
        let value_dim = config.ssm_dt_rank * config.ssm_d_state;
        let conv_channels = key_dim * 2 + value_dim;
        let conv_kernel = config.ssm_conv_kernel;
        let dt_rank = config.ssm_dt_rank;

        let af = |n: usize, name: &str| -> Result<CudaSlice<f32>> {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| candle_core::Error::Msg(format!("alloc {name}({n}): {e}")))
        };
        let au = |n: usize, name: &str| -> Result<CudaSlice<u8>> {
            dev.alloc_zeros::<u8>(n)
                .map_err(|e| candle_core::Error::Msg(format!("alloc {name}({n}): {e}")))
        };

        Ok(Self {
            normed: af(hidden_size, "normed")?,
            q8_hidden: au(q8_buf_size(hidden_size), "q8_hidden")?,
            q8_gated: au(q8_buf_size(value_dim), "q8_gated")?,
            qkv_buf: af(conv_channels, "qkv_buf")?,
            gate_buf: af(value_dim, "gate_buf")?,
            beta_proj: af(dt_rank, "beta_proj")?,
            alpha_proj: af(dt_rank, "alpha_proj")?,
            beta_out: af(dt_rank, "beta_out")?,
            gate_exp_out: af(dt_rank, "gate_exp_out")?,
            conv_output: af(conv_channels, "conv_output")?,
            q_split: af(key_dim, "q_split")?,
            k_split: af(key_dim, "k_split")?,
            q_normed: af(key_dim, "q_normed")?,
            k_normed: af(key_dim, "k_normed")?,
            q_expanded: af(value_dim, "q_expanded")?,
            k_expanded: af(value_dim, "k_expanded")?,
            q_scaled: af(value_dim, "q_scaled")?,
            v_copy: af(value_dim, "v_copy")?,
            ssm_output: af(value_dim, "ssm_output")?,
            gated: af(value_dim, "gated")?,
            projected: af(hidden_size, "projected")?,
            new_conv_state: af(conv_channels * (conv_kernel - 1), "new_conv_state")?,
            normed_ffn: af(hidden_size, "normed_ffn")?,
        })
    }

    /// Total GPU memory used by these buffers, in bytes.
    pub fn bytes_allocated(&self) -> usize {
        // Quick estimate based on known Qwen3.5-35B-A3B dimensions.
        // f32: normed(2048) + qkv(8192) + gate(4096) + beta(32)*2 + gate_exp(32)*2
        //    + conv_output(8192) + q_split(2048) + k_split(2048) + q_normed(2048)
        //    + k_normed(2048) + q_expanded(4096) + k_expanded(4096) + q_scaled(4096)
        //    + v_copy(4096) + ssm_output(4096) + gated(4096) + projected(2048)
        //    + normed_ffn(2048)
        // u8: q8_hidden(2304) + q8_gated(4608)
        // Total f32 elements: 2048 + 8192 + 4096 + 32*4 + 8192 + 2048*4 + 4096*6 + 2048*2 = ~57K
        // Total: ~228 KB f32 + ~7 KB u8 = ~235 KB per GDN buffer set
        0 // Placeholder — not critical for functionality
    }
}

// ---------------------------------------------------------------------------
// RawForwardBufs — bundles all per-token scratch for the raw forward path
// ---------------------------------------------------------------------------

/// All pre-allocated buffers needed by `raw_forward_token`, bundled together.
///
/// Stored as `Option<RawForwardBufs>` on `GdnRecurrentState` and lazily
/// initialised on first use when `CHIMERE_RAW_FORWARD=1`.
/// These are stateless scratch (not snapshotted with MTP branching).
pub struct RawForwardBufs {
    /// Per-layer scratch buffers reused across GDN layers (single set, not per-layer).
    pub gdn_bufs: RawGdnLayerBuffers,
    /// Per-GDN-layer double-buffered DeltaNet recurrent state.
    pub gdn_state_dbufs: Vec<GdnStateDoubleBuffer>,
    /// Per-GDN-layer conv1d sliding-window state (raw CudaSlice mirror).
    pub conv_state_slices: Vec<CudaSlice<f32>>,
}

impl RawForwardBufs {
    /// Allocate all buffers for the raw forward path.
    ///
    /// - `gdn_bufs`: one set of scratch buffers (reused across layers).
    /// - `gdn_state_dbufs`: one double-buffer per GDN layer.
    /// - `conv_state_slices`: one conv state per GDN layer, sized
    ///   `[conv_channels * (conv_kernel - 1)]`.
    pub fn new(config: &crate::config::Qwen35Config, dev: &CudaDevice) -> Result<Self> {
        let num_gdn = config.num_gdn_layers();
        let gdn_bufs = RawGdnLayerBuffers::new(config, dev)?;

        // DeltaNet state size per layer: dt_rank * d_state * d_state
        let state_size = config.ssm_dt_rank * config.ssm_d_state * config.ssm_d_state;
        let mut gdn_state_dbufs = Vec::with_capacity(num_gdn);
        for i in 0..num_gdn {
            gdn_state_dbufs.push(
                GdnStateDoubleBuffer::new(state_size, dev)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("RawForwardBufs: GdnStateDoubleBuffer[{i}]: {e}")))?
            );
        }

        // Conv state: conv_channels * (conv_kernel - 1) f32 per GDN layer
        let key_dim = config.ssm_n_group * config.ssm_d_state;
        let value_dim = config.ssm_dt_rank * config.ssm_d_state;
        let conv_channels = key_dim * 2 + value_dim;
        let conv_buf_len = if config.ssm_conv_kernel > 0 {
            config.ssm_conv_kernel - 1
        } else {
            0
        };
        let conv_state_size = conv_channels * conv_buf_len;
        let mut conv_state_slices = Vec::with_capacity(num_gdn);
        for i in 0..num_gdn {
            conv_state_slices.push(
                dev.alloc_zeros::<f32>(conv_state_size)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("RawForwardBufs: conv_state[{i}]({conv_state_size}): {e}")))?
            );
        }

        let total_mb = (
            // rough estimate: state_size*4*2 per dbuf + conv_state_size*4 per layer
            num_gdn * (state_size * 4 * 2 + conv_state_size * 4)
        ) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[RAW_FORWARD] Allocated RawForwardBufs: {} GDN layers, \
             state_size={}, conv_state_size={}, ~{:.1} MB",
            num_gdn, state_size, conv_state_size, total_mb
        );

        Ok(Self {
            gdn_bufs,
            gdn_state_dbufs,
            conv_state_slices,
        })
    }
}

/// Raw GDN layer forward — zero Candle Tensor operations.
///
/// Replaces `forward_gdn_layer_moe` with direct CudaSlice kernel calls.
/// Pre-allocated buffers are reused across layers (zero cudaMalloc).
///
/// The call sequence per GDN layer:
///   1.  rms_norm(hidden -> normed)
///   2.  quantize_q8_1(normed -> q8_hidden)   [ONCE, reused for 4 projections]
///   3.  q5k_gemv(w_qkv, q8_hidden -> qkv_buf)        [conv_channels output]
///   4.  q5k_gemv(w_gate, q8_hidden -> gate_buf)       [value_dim output]
///   5.  q5k_gemv(w_beta, q8_hidden -> beta_proj)      [dt_rank output]
///   6.  q5k_gemv(w_alpha, q8_hidden -> alpha_proj)    [dt_rank output]
///   7.  fused_beta_alpha_gate(beta_proj, alpha_proj, dt_bias, ssm_a -> beta_out, gate_exp_out)
///   8.  fused_conv1d_silu_update(conv_state, qkv -> conv_output + new_conv_state)
///   9.  split Q/K/V from conv_output (memcpy_dtod to q_split, k_split; v_copy)
///  10.  l2_norm_groups(q -> q_normed, k -> k_normed)
///  11.  expand_groups(q_normed -> q_expanded, k_normed -> k_expanded)
///  12.  scale(q_expanded -> q_scaled)
///  13.  deltanet_step_raw(state, q_scaled, k_expanded, v_copy, gate_exp, beta -> new_state, ssm_output)
///  14.  fused_rms_norm_silu_gate(ssm_output, ssm_norm, gate_buf -> gated)
///  15.  quantize_q8_1(gated -> q8_gated)
///  16.  q5k_gemv(w_ssm_out, q8_gated -> projected)
///  17.  add_inplace(hidden += projected)   [residual connection]
///  18.  rms_norm(hidden -> normed_ffn)
///  19.  [MoE FFN — caller handles via existing moe_ffn_raw or moe_ffn_fused]
///  20.  add_inplace(hidden += ffn_out)     [residual connection, caller handles]
///
/// Steps 19-20 are NOT included here. The caller is responsible for running
/// the MoE FFN on `bufs.normed_ffn` and adding the result to `hidden`.
/// This separation keeps the GDN and MoE paths independently testable.
///
/// # Arguments
///
/// - `hidden`:     `[hidden_size]` f32 — modified in-place (residual add)
/// - `weights`:    Raw weight pointers for this GDN layer
/// - `gdn_state`:  Double-buffered DeltaNet state `[dt_rank * d_state * d_state]`
/// - `conv_state`: Conv1d sliding window state `[conv_channels * (conv_kernel-1)]` — updated in place
/// - `bufs`:       Pre-allocated scratch buffers (reused across layers)
/// - `config`:     Model architecture config
/// - `dev`:        CUDA device
///
/// # Returns
///
/// `Ok(())` on success. After return:
/// - `hidden` has been updated with the SSM residual (step 17)
/// - `bufs.normed_ffn` contains the FFN input (step 18), ready for MoE
/// - `gdn_state` has been swapped (new state is now `current()`)
/// - `conv_state` has been updated in place
pub fn raw_forward_gdn_layer(
    hidden: &mut CudaSlice<f32>,
    weights: &crate::raw_weights::RawGdnWeights,
    gdn_state: &mut GdnStateDoubleBuffer,
    conv_state: &mut CudaSlice<f32>,
    bufs: &mut RawGdnLayerBuffers,
    config: &crate::config::Qwen35Config,
    dev: &CudaDevice,
) -> Result<()> {
    use crate::kernels::elementwise::{
        raw_rms_norm,
        raw_fused_beta_alpha_gate,
        raw_fused_conv1d_silu_update,
        raw_l2_norm_groups,
        raw_expand_groups,
        raw_scale,
        raw_add_inplace,
        raw_fused_rms_norm_silu_gate,
    };
    use crate::kernels::deltanet_step::deltanet_step_fused_raw;
    use crate::kernels::raw_qmatmul::{raw_quantize_q8_1_to, raw_q5k_gemv_to};

    let hidden_size = config.hidden_size;
    let key_dim = config.ssm_n_group * config.ssm_d_state;
    let value_dim = config.ssm_dt_rank * config.ssm_d_state;
    let conv_channels = key_dim * 2 + value_dim;
    let conv_kernel = config.ssm_conv_kernel;
    let n_group = config.ssm_n_group;
    let d_state = config.ssm_d_state;
    let dt_rank = config.ssm_dt_rank;
    let eps = config.rms_norm_eps as f32;

    // -----------------------------------------------------------------------
    // Step 1: RMSNorm(hidden) -> normed
    // -----------------------------------------------------------------------
    raw_rms_norm(hidden, &weights.attn_norm, &mut bufs.normed, hidden_size, eps, dev)?;

    // -----------------------------------------------------------------------
    // Step 2: Quantize normed to Q8_1 — ONCE, reused for 4 projections
    // -----------------------------------------------------------------------
    {
        let normed_view: CudaView<'_, f32> = bufs.normed.slice(..);
        raw_quantize_q8_1_to(&normed_view, &mut bufs.q8_hidden, hidden_size, dev)?;
    }

    // -----------------------------------------------------------------------
    // Steps 3-6: Q5_K GEMV projections (all share the same Q8_1 input)
    //
    // Each writes directly to its target buffer — no intermediate copy.
    // The Q5_K weight Option<CudaSlice<u8>> was extracted at load time;
    // if None, the raw forward path cannot run (caller must check).
    // -----------------------------------------------------------------------

    // Step 3: QKV projection -> qkv_buf [conv_channels=8192]
    {
        let w = weights.attn_qkv.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("raw_gdn: attn_qkv raw bytes missing".into()))?;
        let w_view: CudaView<'_, u8> = w.slice(..);
        raw_q5k_gemv_to(&w_view, &bufs.q8_hidden, &mut bufs.qkv_buf,
            hidden_size, weights.qkv_rows, dev)?;
    }

    // Step 4: Gate (z) projection -> gate_buf [value_dim=4096]
    {
        let w = weights.attn_gate.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("raw_gdn: attn_gate raw bytes missing".into()))?;
        let w_view: CudaView<'_, u8> = w.slice(..);
        raw_q5k_gemv_to(&w_view, &bufs.q8_hidden, &mut bufs.gate_buf,
            hidden_size, weights.gate_rows, dev)?;
    }

    // Step 5: Beta projection -> beta_proj [dt_rank=32]
    {
        let w = weights.ssm_beta.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("raw_gdn: ssm_beta raw bytes missing".into()))?;
        let w_view: CudaView<'_, u8> = w.slice(..);
        raw_q5k_gemv_to(&w_view, &bufs.q8_hidden, &mut bufs.beta_proj,
            hidden_size, weights.beta_rows, dev)?;
    }

    // Step 6: Alpha projection -> alpha_proj [dt_rank=32]
    {
        let w = weights.ssm_alpha.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("raw_gdn: ssm_alpha raw bytes missing".into()))?;
        let w_view: CudaView<'_, u8> = w.slice(..);
        raw_q5k_gemv_to(&w_view, &bufs.q8_hidden, &mut bufs.alpha_proj,
            hidden_size, weights.alpha_rows, dev)?;
    }

    // -----------------------------------------------------------------------
    // Step 7: Fused beta/alpha/gate computation
    //   beta_out[i] = sigmoid(beta_proj[i])
    //   gate_exp_out[i] = exp(softplus(alpha_proj[i] + dt_bias[i]) * ssm_a[i])
    // -----------------------------------------------------------------------
    raw_fused_beta_alpha_gate(
        &bufs.beta_proj, &bufs.alpha_proj,
        &weights.ssm_dt_bias, &weights.ssm_a,
        &mut bufs.beta_out, &mut bufs.gate_exp_out,
        dt_rank, dev,
    )?;

    // -----------------------------------------------------------------------
    // Step 8: Fused conv1d + SiLU + state update
    //   Reads conv_state [conv_channels * (conv_kernel-1)] and qkv_buf [conv_channels].
    //   Writes conv_output [conv_channels] and new_conv_state [conv_channels * (conv_kernel-1)].
    //   Then copies new_conv_state back to conv_state.
    // -----------------------------------------------------------------------
    raw_fused_conv1d_silu_update(
        &conv_state.slice(..),  // current state (read, zero-copy view)
        &bufs.qkv_buf.slice(..),      // new input
        &weights.ssm_conv1d.slice(..), // depthwise conv weights [conv_channels * conv_kernel]
        &mut bufs.conv_output,
        &mut bufs.new_conv_state,  // new state (separate buffer)
        conv_channels,
        conv_kernel,
        dev,
    )?;
    // Copy new conv state back to the layer's persistent conv_state.
    {
        let src = bufs.new_conv_state.slice(..);
        dev.memcpy_dtod(&src, conv_state)
            .map_err(|e| candle_core::Error::Msg(format!("memcpy new_conv_state: {e}")))?;
    }

    // -----------------------------------------------------------------------
    // Step 9: Split Q/K/V from conv_output
    //   conv_output layout: [Q: key_dim | K: key_dim | V: value_dim]
    //   Q = conv_output[0..key_dim]
    //   K = conv_output[key_dim..key_dim*2]
    //   V = conv_output[key_dim*2..key_dim*2+value_dim]
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Step 9+10: Split Q/K/V from conv_output and L2 normalize Q, K
    //   conv_output layout: [Q: key_dim | K: key_dim | V: value_dim]
    //   Zero-copy: pass conv_output views directly to L2 norm kernels,
    //   eliminating 3 memcpy_dtod (Phase 2.3).
    // -----------------------------------------------------------------------
    let q_view = bufs.conv_output.slice(0..key_dim);
    let k_view = bufs.conv_output.slice(key_dim..key_dim * 2);
    let v_view = bufs.conv_output.slice(key_dim * 2..key_dim * 2 + value_dim);

    raw_l2_norm_groups(&q_view, &mut bufs.q_normed, d_state, n_group, eps, dev)?;
    raw_l2_norm_groups(&k_view, &mut bufs.k_normed, d_state, n_group, eps, dev)?;

    // -----------------------------------------------------------------------
    // Step 11: Expand groups to dt_rank heads
    //   q_normed [n_group * d_state] -> q_expanded [dt_rank * d_state]
    //   k_normed [n_group * d_state] -> k_expanded [dt_rank * d_state]
    //
    //   Each of the n_group groups is repeated (dt_rank / n_group) times.
    //   For Qwen3.5-35B-A3B: n_group=16, dt_rank=32 -> repeats=2.
    // -----------------------------------------------------------------------
    let repeats = dt_rank / n_group;
    raw_expand_groups(&bufs.q_normed.slice(..), &mut bufs.q_expanded, d_state, n_group, repeats, dev)?;
    raw_expand_groups(&bufs.k_normed.slice(..), &mut bufs.k_expanded, d_state, n_group, repeats, dev)?;

    // -----------------------------------------------------------------------
    // Step 12: Scale Q by 1/sqrt(d_state)
    //   q_expanded [dt_rank * d_state] -> q_scaled [dt_rank * d_state]
    // -----------------------------------------------------------------------
    let q_scale = 1.0 / (d_state as f32).sqrt();
    raw_scale(&bufs.q_expanded.slice(..), &mut bufs.q_scaled, value_dim, q_scale, dev)?;

    // -----------------------------------------------------------------------
    // Step 13: DeltaNet recurrent step
    //   Reads current state, q_scaled, k_expanded, v (from conv_output view),
    //   gate_exp_out, beta_out. Writes new state and ssm_output.
    //   V is passed as a zero-copy view into conv_output (Phase 2.3).
    //
    //   Uses double-buffer: current_and_next() returns (&read, &mut write)
    //   without overlapping borrows, then swap() makes the write buffer current.
    // -----------------------------------------------------------------------
    {
        let (state_in, state_out) = gdn_state.current_and_next();
        deltanet_step_fused_raw(
            &state_in.slice(..),
            &bufs.q_scaled.slice(..),
            &bufs.k_expanded.slice(..),
            &v_view,
            &bufs.gate_exp_out.slice(..),
            &bufs.beta_out.slice(..),
            state_out,
            &mut bufs.ssm_output,
            dt_rank,
            d_state,
            dev,
        )?;
    }
    gdn_state.swap();

    // -----------------------------------------------------------------------
    // Step 14: Fused RMSNorm + SiLU gate
    //   For each group g, dimension j:
    //     gated[g*D+j] = rms_norm(ssm_output[g*D+j], ssm_norm[j]) * silu(gate_buf[g*D+j])
    //
    //   ssm_output is [dt_rank * d_state], viewed as [dt_rank groups, d_state each].
    //   gate_buf is [value_dim] = [dt_rank * d_state].
    //   ssm_norm weight is [d_state].
    // -----------------------------------------------------------------------
    raw_fused_rms_norm_silu_gate(
        &bufs.ssm_output,
        &weights.ssm_norm,
        &bufs.gate_buf,
        &mut bufs.gated,
        dt_rank,        // groups
        d_state,        // D per group
        eps,
        dev,
    )?;

    // -----------------------------------------------------------------------
    // Step 15: Quantize gated output to Q8_1 for ssm_out projection
    // -----------------------------------------------------------------------
    {
        let gated_view: CudaView<'_, f32> = bufs.gated.slice(..);
        raw_quantize_q8_1_to(&gated_view, &mut bufs.q8_gated, value_dim, dev)?;
    }

    // -----------------------------------------------------------------------
    // Step 16: SSM output projection -> projected [hidden_size=2048]
    //   w_ssm_out: [hidden_size, value_dim] Q5_K
    //   input: q8_gated (Q8_1 of gated [value_dim])
    //   output: projected [hidden_size]
    // -----------------------------------------------------------------------
    {
        let w = weights.ssm_out.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("raw_gdn: ssm_out raw bytes missing".into()))?;
        let w_view: CudaView<'_, u8> = w.slice(..);
        raw_q5k_gemv_to(&w_view, &bufs.q8_gated, &mut bufs.projected,
            weights.ssm_out_cols, weights.ssm_out_rows, dev)?;
    }

    // -----------------------------------------------------------------------
    // Step 17: Residual connection — hidden += projected
    // -----------------------------------------------------------------------
    raw_add_inplace(hidden, &bufs.projected, hidden_size, dev)?;

    // -----------------------------------------------------------------------
    // Step 18: RMSNorm for FFN input
    //   normed_ffn = RMSNorm(hidden, post_norm)
    //   Caller will use normed_ffn to run MoE FFN.
    // -----------------------------------------------------------------------
    raw_rms_norm(hidden, &weights.post_norm, &mut bufs.normed_ffn, hidden_size, eps, dev)?;

    // Steps 19-20 (MoE FFN + residual) handled by caller.

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor <-> CudaSlice conversion helpers
// ---------------------------------------------------------------------------

/// Extract an owned `CudaSlice<f32>` from a Candle Tensor (device-to-device copy).
///
/// The returned slice is fully owned — no borrow on the source Tensor.
/// The source tensor must be F32 and on a CUDA device.
pub fn cuda_slice_from_tensor(tensor: &Tensor, dev: &CudaDevice) -> Result<CudaSlice<f32>> {
    use candle_core::Storage;
    let t = tensor.contiguous()?;
    let n = t.elem_count();
    let (stor, lay) = t.storage_and_layout();
    let cuda = match &*stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("cuda_slice_from_tensor: tensor not on CUDA"),
    };
    if t.dtype() != candle_core::DType::F32 {
        candle_core::bail!("cuda_slice_from_tensor: expected F32, got {:?}", t.dtype());
    }
    let src = cuda.as_cuda_slice::<f32>()?;
    let off = lay.start_offset();
    let src_view = src.slice(off..off + n);

    let mut owned = dev.alloc_zeros::<f32>(n)
        .map_err(|e| candle_core::Error::Msg(format!("cuda_slice_from_tensor alloc({n}): {e}")))?;
    dev.memcpy_dtod(&src_view, &mut owned)
        .map_err(|e| candle_core::Error::Msg(format!("cuda_slice_from_tensor dtod: {e}")))?;
    Ok(owned)
}

/// Create a Candle Tensor from a `CudaSlice<f32>`, taking ownership.
///
/// The CudaSlice is consumed (moved into Candle's Storage). The returned Tensor
/// has shape `[1, dim]` (batch=1, suitable for QMatMul::forward and layer functions).
pub fn tensor_from_cuda_slice(
    slice: CudaSlice<f32>,
    dim: usize,
    dev: &CudaDevice,
) -> Tensor {
    use candle_core::Storage;
    Tensor::from_storage(
        Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            slice, dev.clone())),
        candle_core::Shape::from_dims(&[1, dim]),
        candle_core::op::BackpropOp::none(), false,
    )
}

// (copy_cuda_slice_into_tensor removed — dead code)

// ---------------------------------------------------------------------------
// AddCudaSliceInplace — zero-alloc add of CudaSlice into Tensor (Fix #3)
// ---------------------------------------------------------------------------

/// Add a `CudaSlice<f32>` into a Candle Tensor's storage in-place.
///
/// Implements `InplaceOp1` so it can be used with `tensor.inplace_op1(&op)`.
/// This avoids the `CudaSlice::try_clone()` (cudaMalloc + cudaMemcpy) that would
/// otherwise be needed to wrap a CudaSlice into a Tensor for addition.
///
/// The source CudaSlice is borrowed for the lifetime of this struct.
/// The kernel launched is the same `add_inplace_kernel` used by `raw_add_inplace`.
pub struct AddCudaSliceInplace<'a> {
    /// Source CudaSlice whose elements will be added to the destination tensor.
    src: &'a CudaSlice<f32>,
    /// Number of f32 elements to add.
    n: usize,
}

impl<'a> AddCudaSliceInplace<'a> {
    /// Create from a `CudaSlice<f32>` reference.
    ///
    /// The caller must ensure the source CudaSlice is not concurrently modified
    /// during the kernel launch (which is guaranteed when called from MoE forward
    /// after the expert kernel has completed).
    pub fn new(src: &'a CudaSlice<f32>, n: usize) -> Self {
        Self { src, n }
    }
}

impl candle_core::InplaceOp1 for AddCudaSliceInplace<'_> {
    fn name(&self) -> &'static str {
        "add_cuda_slice_inplace"
    }

    fn cpu_fwd(
        &self,
        _storage: &mut candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("add_cuda_slice_inplace: CPU not supported")
    }

    fn cuda_fwd(
        &self,
        storage: &mut candle_core::cuda_backend::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;

        let off = layout.start_offset();
        let n = self.n;

        // Clone the device handle before taking the mutable borrow on the slice.
        // CudaDevice is Arc-based, so this clone is cheap (just an Arc bump).
        let dev = storage.device.clone();

        let dst = storage.as_cuda_slice_mut::<f32>()?;

        // Validate: dst must have enough elements past the offset
        if dst.len() < off + n {
            candle_core::bail!(
                "add_cuda_slice_inplace: dst too small ({} < {}+{})",
                dst.len(), off, n
            );
        }

        // Apply offset: if the tensor has a non-zero start offset, we need to
        // add into the correct region. For MoE output tensors (result of Candle
        // elementwise ops), offset is always 0.
        if off == 0 {
            // Fast path: no offset, use raw_add_inplace directly on the full slice.
            crate::kernels::elementwise::raw_add_inplace(
                dst, self.src, n, &dev,
            )?;
        } else {
            // Slow path with offset: slice the destination.
            // raw_add_inplace needs &mut CudaSlice, but we have the full slice
            // and need to operate on a sub-range. Use a view-based kernel launch.
            let ptx = crate::kernels::elementwise::get_add_inplace_ptx();
            let func = dev.get_or_load_custom_func(
                "add_inplace_kernel",
                "chimere_elemwise_v4",
                ptx,
            )?;

            let blocks = ((n as u32) + 255) / 256;
            let n_i32 = n as i32;
            let cfg = candle_core::cuda_backend::cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            // Use slice views to handle the offset correctly.
            let mut dst_view = dst.slice_mut(off..off + n);
            let src_view = self.src.slice(..n);
            let mut builder = func.builder();
            builder.arg(&mut dst_view);
            builder.arg(&src_view);
            builder.arg(&n_i32);
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(
                    format!("add_cuda_slice_inplace launch: {e}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Raw forward token — complete token forward pass using raw CudaSlice path
// ---------------------------------------------------------------------------

/// Complete raw forward pass for one token.
///
/// Toggle: `CHIMERE_RAW_FORWARD=1`
///
/// Uses `raw_forward_gdn_layer` for GDN layers (zero Tensor allocation).
/// Falls back to Candle for attention layers (MRoPE, KV cache, GQA).
/// MoE FFN uses the existing raw path (`moe_ffn_raw` / `moe_ffn_fused`).
///
/// # Arguments
///
/// - `token`:          Input token ID.
/// - `model`:          Loaded Qwen35Model (for embed, attention layers, lm_head).
/// - `state`:          Mutable recurrent state (GDN + KV caches).
/// - `raw_weights`:    Pre-extracted raw weight pointers.
/// - `gdn_bufs`:       Pre-allocated scratch buffers for GDN layers.
/// - `gdn_state_dbufs`: 30 double-buffered DeltaNet states (one per GDN layer).
/// - `conv_state_slices`: 30 conv state CudaSlices (one per GDN layer).
/// - `dev`:            CUDA device handle.
///
/// # Returns
///
/// Logits as `CudaSlice<f32>` of size `[vocab_size]`.
pub fn raw_forward_token(
    token: u32,
    model: &crate::qwen35_model::Qwen35Model,
    state: &mut crate::state::GdnRecurrentState,
    raw_weights: &crate::raw_weights::RawWeights,
    gdn_bufs: &mut RawGdnLayerBuffers,
    gdn_state_dbufs: &mut Vec<GdnStateDoubleBuffer>,
    conv_state_slices: &mut Vec<CudaSlice<f32>>,
    dev: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    use crate::kernels::elementwise::{raw_rms_norm, raw_add_inplace};

    let config = &model.config;
    let hidden_size = config.hidden_size;
    let eps = config.rms_norm_eps as f32;
    let q_layers = model.q_layers.as_ref()
        .ok_or_else(|| candle_core::Error::Msg("raw_forward_token: no q_layers".into()))?;
    let embed_tokens = model.embed_tokens.as_ref()
        .ok_or_else(|| candle_core::Error::Msg("raw_forward_token: no embed_tokens".into()))?;

    // -----------------------------------------------------------------------
    // 1. Embed: index_select on CPU embed table, then memcpy_htod to hidden
    // -----------------------------------------------------------------------
    let tok_tensor = Tensor::new(&[token], &Device::Cpu)?;
    let embd_cpu = embed_tokens.index_select(&tok_tensor, 0)?; // [1, hidden_size] on CPU
    let embd_vec: Vec<f32> = embd_cpu.flatten_all()?.to_vec1()?;
    debug_assert_eq!(embd_vec.len(), hidden_size);

    // Allocate the working hidden buffer
    let mut hidden = dev.alloc_zeros::<f32>(hidden_size)
        .map_err(|e| candle_core::Error::Msg(format!("alloc hidden: {e}")))?;
    dev.memcpy_htod(&embd_vec, &mut hidden)
        .map_err(|e| candle_core::Error::Msg(format!("htod embed: {e}")))?;

    // -----------------------------------------------------------------------
    // 2. Layer loop
    // -----------------------------------------------------------------------
    let mut gdn_idx = 0usize;
    let mut _attn_idx = 0usize;

    for (il, layer) in q_layers.iter().enumerate() {
        match layer {
            crate::qwen35_model::Qwen35LayerQ::GdnMoE(w) => {
                // --- Raw GDN SSM (steps 1-18) ---
                raw_forward_gdn_layer(
                    &mut hidden,
                    &raw_weights.gdn[gdn_idx],
                    &mut gdn_state_dbufs[gdn_idx],
                    &mut conv_state_slices[gdn_idx],
                    gdn_bufs,
                    config,
                    dev,
                )?;

                // --- MoE FFN (steps 19-20) ---
                // normed_ffn is in gdn_bufs.normed_ffn after step 18.
                // Run existing MoE FFN raw path.
                {
                    let normed_view: CudaView<'_, f32> = gdn_bufs.normed_ffn.slice(..);

                    // Extract expert weight views from RawWeights
                    let rw = &raw_weights.gdn[gdn_idx];
                    let g_view: CudaView<'_, u8> = rw.gate_exps.slice(..);
                    let u_view: CudaView<'_, u8> = rw.up_exps.slice(..);
                    let d_view: CudaView<'_, u8> = rw.down_exps.slice(..);

                    // Expert byte strides
                    let (gate_rows_exp, gate_cols_exp, n_experts) = rw.gate_exps_shape;
                    let (up_rows_exp, up_cols_exp, _) = rw.up_exps_shape;
                    let (down_rows_exp, down_cols_exp, _) = rw.down_exps_shape;
                    // IQ3_S bytes per expert = ceil(rows * cols / 256) * 64
                    let iq3s_bytes_per_expert = |rows: usize, cols: usize| -> usize {
                        let n_elements = rows * cols;
                        (n_elements + 255) / 256 * 64
                    };
                    let eb_gate = iq3s_bytes_per_expert(gate_rows_exp, gate_cols_exp);
                    let eb_up = iq3s_bytes_per_expert(up_rows_exp, up_cols_exp);
                    let eb_down = iq3s_bytes_per_expert(down_rows_exp, down_cols_exp);

                    // Router: Candle path for now (correct, simple)
                    // Wrap normed_ffn as Tensor for router
                    let normed_ffn_owned = {
                        let mut buf = dev.alloc_zeros::<f32>(hidden_size)
                            .map_err(|e| candle_core::Error::Msg(format!("alloc normed_ffn clone: {e}")))?;
                        dev.memcpy_dtod(&normed_view, &mut buf)
                            .map_err(|e| candle_core::Error::Msg(format!("copy normed_ffn: {e}")))?;
                        buf
                    };
                    let normed_ffn_tensor = tensor_from_cuda_slice(normed_ffn_owned, hidden_size, dev);

                    // Router logits via F32 GEMV
                    let _router_t = &w.moe.gate_inp_t; // [num_experts, hidden_size] transposed
                    let router_logits = normed_ffn_tensor.matmul(
                        &Tensor::from_storage(
                            candle_core::Storage::Cuda(
                                candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                                    {
                                        let mut buf = dev.alloc_zeros::<f32>(n_experts * hidden_size)
                                            .map_err(|e| candle_core::Error::Msg(format!("alloc router: {e}")))?;
                                        let src = rw.gate_inp_t.slice(..);
                                        dev.memcpy_dtod(&src, &mut buf)
                                            .map_err(|e| candle_core::Error::Msg(format!("copy router: {e}")))?;
                                        buf
                                    },
                                    dev.clone(),
                                ),
                            ),
                            candle_core::Shape::from_dims(&[hidden_size, n_experts]),
                            candle_core::op::BackpropOp::none(),
                            false,
                        ),
                    )?;

                    // Softmax + top-K on CPU
                    let mut logits_cpu: Vec<f32> = router_logits.flatten_all()?.to_vec1()?;
                    softmax_inplace(&mut logits_cpu);
                    let top_k = config.experts_per_token;
                    let selected = topk_cpu(&logits_cpu, top_k);

                    // Upload selected indices + weights to GPU for gpu_topk path
                    {
                        let moe_bufs = state.raw_moe_buffers.as_mut()
                            .ok_or_else(|| candle_core::Error::Msg("no raw_moe_buffers".into()))?;

                        let indices_cpu: Vec<i32> = selected.iter().map(|(i, _)| *i as i32).collect();
                        let weights_cpu: Vec<f32> = selected.iter().map(|(_, w)| *w).collect();
                        dev.memcpy_htod(&indices_cpu, &mut moe_bufs.topk_indices_buf)
                            .map_err(|e| candle_core::Error::Msg(format!("htod indices: {e}")))?;
                        dev.memcpy_htod(&weights_cpu, &mut moe_bufs.topk_weights_buf)
                            .map_err(|e| candle_core::Error::Msg(format!("htod weights: {e}")))?;

                        // Run MoE FFN
                        moe_ffn_raw(
                            &normed_view,
                            &selected,
                            &g_view, &u_view, &d_view,
                            eb_gate, eb_up, eb_down,
                            gate_rows_exp * gate_cols_exp,
                            up_rows_exp * up_cols_exp,
                            down_rows_exp * down_cols_exp,
                            moe_bufs,
                            hidden_size,
                            gate_rows_exp, // expert_ffn = gate output rows
                            n_experts,
                            top_k,
                            dev,
                            &Device::Cuda(dev.clone()),
                        )?;

                        // Residual: hidden += combined_buf (routed experts)
                        raw_add_inplace(&mut hidden, &moe_bufs.combined_buf, hidden_size, dev)?;
                    }

                    // Shared expert: Candle fallback for now
                    // shared_gate, shared_up, shared_down in QMatMul
                    let h_tensor = tensor_from_cuda_slice(
                        {
                            let mut buf = dev.alloc_zeros::<f32>(hidden_size)
                                .map_err(|e| candle_core::Error::Msg(format!("alloc sh_h: {e}")))?;
                            let src = gdn_bufs.normed_ffn.slice(..hidden_size);
                            dev.memcpy_dtod(&src, &mut buf)
                                .map_err(|e| candle_core::Error::Msg(format!("copy sh_h: {e}")))?;
                            buf
                        },
                        hidden_size,
                        dev,
                    );
                    let sh_gate_out = w.moe.gate_shexp.forward(&h_tensor)?;
                    let sh_up_out = w.moe.up_shexp.forward(&h_tensor)?;
                    let sh_inter = (sh_gate_out.silu()? * sh_up_out)?;
                    let sh_down_out = w.moe.down_shexp.forward(&sh_inter)?;

                    // Sigmoid gate on shared expert bias
                    let sh_gate_logit = normed_ffn_tensor.matmul(
                        &w.moe.gate_inp_shexp.unsqueeze(0)?.t()?
                    )?;
                    let sh_gate_val = candle_core::Tensor::ones(sh_gate_logit.shape(), sh_gate_logit.dtype(), sh_gate_logit.device())?
                        .broadcast_div(&(sh_gate_logit.neg()?.exp()? + 1.0)?)?;
                    let sh_final = (sh_down_out * sh_gate_val)?;

                    // Add shared expert to hidden
                    let sh_slice = cuda_slice_from_tensor(&sh_final.flatten_all()?, dev)?;
                    raw_add_inplace(&mut hidden, &sh_slice, hidden_size, dev)?;
                }

                gdn_idx += 1;
            }

            crate::qwen35_model::Qwen35LayerQ::AttentionMoE(w) => {
                // --- Attention: Candle fallback ---
                // Convert CudaSlice hidden -> Tensor, run Candle attention, extract back.
                let hidden_tensor = tensor_from_cuda_slice(
                    {
                        let mut buf = dev.alloc_zeros::<f32>(hidden_size)
                            .map_err(|e| candle_core::Error::Msg(format!("alloc attn_h: {e}")))?;
                        let src = hidden.slice(..hidden_size);
                        dev.memcpy_dtod(&src, &mut buf)
                            .map_err(|e| candle_core::Error::Msg(format!("copy attn_h: {e}")))?;
                        buf
                    },
                    hidden_size,
                    dev,
                );
                let h_out = model.forward_attn_layer_moe(il, w, &hidden_tensor, config.rms_norm_eps, state)?;
                // Extract back to CudaSlice
                let h_out_slice = cuda_slice_from_tensor(&h_out.flatten_all()?, dev)?;
                let src = h_out_slice.slice(..hidden_size);
                dev.memcpy_dtod(&src, &mut hidden)
                    .map_err(|e| candle_core::Error::Msg(format!("copy attn_out: {e}")))?;

                _attn_idx += 1;
            }

            // Dense variants (27B model) — not supported in raw path
            _ => {
                candle_core::bail!("raw_forward_token: unsupported layer type at layer {}", il);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Output norm
    // -----------------------------------------------------------------------
    let mut normed_out = dev.alloc_zeros::<f32>(hidden_size)
        .map_err(|e| candle_core::Error::Msg(format!("alloc normed_out: {e}")))?;
    raw_rms_norm(&hidden, &raw_weights.output_norm, &mut normed_out, hidden_size, eps, dev)?;

    // -----------------------------------------------------------------------
    // 4. LM head: QMatMul fallback via Candle (vocab=248320, too large for raw Q5K path)
    // -----------------------------------------------------------------------
    let normed_tensor = tensor_from_cuda_slice(
        {
            let mut buf = dev.alloc_zeros::<f32>(hidden_size)
                .map_err(|e| candle_core::Error::Msg(format!("alloc lm_normed: {e}")))?;
            let src = normed_out.slice(..hidden_size);
            dev.memcpy_dtod(&src, &mut buf)
                .map_err(|e| candle_core::Error::Msg(format!("copy lm_normed: {e}")))?;
            buf
        },
        hidden_size,
        dev,
    );
    let lm_head = model.lm_head.as_ref()
        .ok_or_else(|| candle_core::Error::Msg("raw_forward_token: no lm_head".into()))?;
    let logits_tensor = lm_head.forward(&normed_tensor)?;
    let logits_slice = cuda_slice_from_tensor(&logits_tensor.flatten_all()?, dev)?;

    // Advance position
    state.advance(1);

    Ok(logits_slice)
}

// (raw_gdn_ssm_benchmark removed — dead code, no callers)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_allocated_sanity() {
        // Verify the byte count matches our expectations without needing a GPU.
        // f32 buffers:
        //   hidden + normed + qkv_buf + gate_buf + ssm_out_buf
        //   + expert_gate_buf + expert_up_buf + expert_inter_buf
        //   + expert_out_buf + combined_buf + logits_buf
        //   + router_probs_buf + topk_weights_buf
        //   + batched_gate_out + batched_up_out + batched_inter_out
        //   + batched_expert_outs
        // i32 buffers: topk_indices_buf
        // u8  buffers: q8_hidden_buf + q8_inter_buf + batched_q8_inter
        let f32_elems = HIDDEN_SIZE + HIDDEN_SIZE + QKV_SIZE + NUM_EXPERTS
            + SSM_D_INNER + EXPERT_FFN_HIDDEN + EXPERT_FFN_HIDDEN + EXPERT_FFN_HIDDEN
            + HIDDEN_SIZE + HIDDEN_SIZE + VOCAB_SIZE + NUM_EXPERTS + EXPERTS_PER_TOKEN
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN   // batched_gate_out
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN   // batched_up_out
            + EXPERTS_PER_TOKEN * EXPERT_FFN_HIDDEN   // batched_inter_out
            + EXPERTS_PER_TOKEN * HIDDEN_SIZE;         // batched_expert_outs
        let computed = f32_elems * std::mem::size_of::<f32>()
            + q8_size(HIDDEN_SIZE)
            + q8_size(EXPERT_FFN_HIDDEN)
            + EXPERTS_PER_TOKEN * q8_size(EXPERT_FFN_HIDDEN)  // batched_q8_inter
            + EXPERTS_PER_TOKEN * std::mem::size_of::<i32>();
        // Should be roughly 1.5 MB (+ 32 KB for batched buffers)
        assert!(computed > 1_000_000, "Expected > 1 MB of buffers, got {}", computed);
        assert!(computed < 5_000_000, "Expected < 5 MB of buffers, got {}", computed);
        println!(
            "RawGpuBuffers total allocation: {} bytes ({:.2} MB)",
            computed,
            computed as f64 / 1_048_576.0
        );
    }

    // (test_forward_token_raw_returns_error removed — tested dead stub)

    // -----------------------------------------------------------------------
    // CPU-only tests for moe_ffn_raw helpers (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1.0, got {:.7}",
            sum
        );
        // Largest logit should have largest probability.
        assert!(v[3] > v[2] && v[2] > v[1] && v[1] > v[0],
            "softmax should preserve order: {:?}", v);
    }

    #[test]
    fn test_softmax_inplace_uniform() {
        let mut v = vec![0.0f32; 8];
        softmax_inplace(&mut v);
        for (i, &x) in v.iter().enumerate() {
            assert!(
                (x - 0.125).abs() < 1e-6,
                "uniform softmax[{}] should be 1/8 = 0.125, got {:.7}",
                i, x
            );
        }
    }

    #[test]
    fn test_softmax_inplace_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v); // Should not panic.
    }

    #[test]
    fn test_topk_cpu_selects_top_k() {
        let probs = vec![0.05f32, 0.60, 0.10, 0.25];
        let result = topk_cpu(&probs, 2);
        assert_eq!(result.len(), 2);
        // Highest prob is index 1 (0.60), second is index 3 (0.25).
        assert_eq!(result[0].0, 1, "top-1 expert should be index 1");
        assert_eq!(result[1].0, 3, "top-2 expert should be index 3");
        // Renormalised weights should sum to 1.0.
        let w_sum: f32 = result.iter().map(|(_, w)| w).sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-5,
            "topk weights should sum to 1.0, got {:.7}",
            w_sum
        );
        // Relative ratio preserved: 0.60 / 0.25 = 2.4.
        let ratio = result[0].1 / result[1].1;
        let expected = 0.60 / 0.25;
        assert!(
            (ratio - expected).abs() < 1e-4,
            "weight ratio should be {:.4}, got {:.4}",
            expected, ratio
        );
        println!("topk_cpu result: {:?}", result);
    }

    #[test]
    fn test_topk_cpu_k_equals_len() {
        // k == n: all experts selected, weights unchanged after renorm.
        let probs = vec![0.1f32, 0.3, 0.4, 0.2];
        let result = topk_cpu(&probs, 4);
        assert_eq!(result.len(), 4);
        let w_sum: f32 = result.iter().map(|(_, w)| w).sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-5,
            "weights should sum to 1.0 when k=n, got {:.7}",
            w_sum
        );
    }

    #[test]
    fn test_topk_cpu_k1() {
        let probs = vec![0.1f32, 0.8, 0.05, 0.05];
        let result = topk_cpu(&probs, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1, "top-1 should be expert 1");
        assert!((result[0].1 - 1.0).abs() < 1e-6, "single expert weight = 1.0");
    }
}
