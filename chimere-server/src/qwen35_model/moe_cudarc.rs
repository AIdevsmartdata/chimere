//! Pure cudarc MoE FFN forward pass — zero Candle Tensor allocations.
//!
//! This is the VALIDATION module for the cudarc rewrite strategy. If 1 layer
//! of MoE FFN runs faster via pure cudarc (this module) than via the existing
//! Candle-based path, we extend the approach to all 40 layers.
//!
//! ## Design
//!
//! The entire MoE FFN (routed experts + shared expert) is computed using only
//! cudarc `CudaSlice` operations and pre-existing CUDA kernels. Zero Candle
//! Tensor ops: the router uses `raw_f32_gemv` on pre-extracted `RawWeights`
//! CudaSlice refs, the routed experts use `fused_moe_iq3s_gpu_resident`,
//! and the shared expert uses Q5_K GEMV kernels.
//!
//! ## Toggle
//!
//! Set `CHIMERE_CUDARC_MOE=1` to enable. The toggle is checked once at
//! startup via `once_cell::sync::Lazy`.
//!
//! ## Buffers
//!
//! `MoeCudarcBufs` holds all pre-allocated scratch buffers for one MoE FFN
//! forward pass. Created once, reused every layer, every token.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Result, Tensor};

use crate::kernels::q5k_mmvq_ggml::GgmlQ5KBuffers;

// ---------------------------------------------------------------------------
// Constants (Qwen3.5-35B-A3B)
// ---------------------------------------------------------------------------

const IQ3S_BLOCK_BYTES: usize = 110;
const IQ3S_BLOCK_ELEMS: usize = 256;

// ---------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------

/// Check (once) whether the cudarc MoE path is enabled.
pub(crate) fn cudarc_moe_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        let enabled = std::env::var("CHIMERE_CUDARC_MOE").is_ok();
        if enabled {
            eprintln!("[MOE_CUDARC] Enabled -- pure cudarc MoE FFN (zero Candle Tensor ops)");
        }
        enabled
    });
    *ENABLED
}

// ---------------------------------------------------------------------------
// Pre-allocated buffers
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for one MoE FFN forward pass.
/// Created once at model init, reused every layer every token.
///
/// Total memory: ~52 KB (trivial vs model's 15 GB).
pub(crate) struct MoeCudarcBufs {
    // --- Routed expert scratch ---
    /// Router logits before top-K: [n_experts=256]
    pub router_logits: CudaSlice<f32>,
    /// Top-K selected expert indices: [top_k=8]
    pub topk_indices: CudaSlice<i32>,
    /// Top-K renormalized expert weights: [top_k=8]
    pub topk_weights: CudaSlice<f32>,
    /// Accumulated routed expert output: [hidden_size=2048]
    pub combined_buf: CudaSlice<f32>,

    // --- Shared expert scratch ---
    /// Shared expert gate GEMV output: [expert_ffn=512]
    pub shexp_gate: CudaSlice<f32>,
    /// Shared expert up GEMV output: [expert_ffn=512]
    pub shexp_up: CudaSlice<f32>,
    /// Shared expert SwiGLU intermediate: [expert_ffn=512]
    pub shexp_inter: CudaSlice<f32>,
    /// Shared expert down GEMV output: [hidden_size=2048]
    pub shexp_out: CudaSlice<f32>,
    /// Sigmoid gate scalar: [1]
    pub shexp_gate_logit: CudaSlice<f32>,
    /// Pre-allocated Q5_K GEMV buffers (Q8_1 input quantization scratch)
    pub shexp_q5k_bufs: GgmlQ5KBuffers,

    // Dimensions (stored for kernel launches)
    hidden_size: usize,
    expert_ffn: usize,
    num_experts: usize,
    top_k: usize,
}

impl MoeCudarcBufs {
    /// Allocate all scratch buffers for the pure cudarc MoE path.
    pub fn new(
        hidden_size: usize,
        expert_ffn: usize,
        num_experts: usize,
        top_k: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        fn alloc_err(name: &str, e: impl std::fmt::Display) -> candle_core::Error {
            candle_core::Error::Msg(format!("MoeCudarcBufs alloc {name}: {e}"))
        }

        let router_logits = dev.alloc_zeros::<f32>(num_experts).map_err(|err| alloc_err("router_logits", err))?;
        let topk_indices = dev.alloc_zeros::<i32>(top_k).map_err(|err| alloc_err("topk_indices", err))?;
        let topk_weights = dev.alloc_zeros::<f32>(top_k).map_err(|err| alloc_err("topk_weights", err))?;
        let combined_buf = dev.alloc_zeros::<f32>(hidden_size).map_err(|err| alloc_err("combined_buf", err))?;
        let shexp_gate = dev.alloc_zeros::<f32>(expert_ffn).map_err(|err| alloc_err("shexp_gate", err))?;
        let shexp_up = dev.alloc_zeros::<f32>(expert_ffn).map_err(|err| alloc_err("shexp_up", err))?;
        let shexp_inter = dev.alloc_zeros::<f32>(expert_ffn).map_err(|err| alloc_err("shexp_inter", err))?;
        let shexp_out = dev.alloc_zeros::<f32>(hidden_size).map_err(|err| alloc_err("shexp_out", err))?;
        let shexp_gate_logit = dev.alloc_zeros::<f32>(1).map_err(|err| alloc_err("shexp_gate_logit", err))?;

        // Q5_K GEMV scratch: max ncols = hidden_size, max nrows = hidden_size
        let shexp_q5k_bufs = GgmlQ5KBuffers::new(hidden_size, hidden_size, dev)?;

        let total_bytes = (num_experts + top_k + hidden_size     // router + topk_wt + combined
            + expert_ffn * 3 + hidden_size + 1)                  // shared expert
            * 4
            + top_k * 4;                                          // topk_indices (i32)
        eprintln!(
            "[MOE_CUDARC] Allocated scratch buffers: {:.1} KB \
             (router={} combined={} shexp={}+{}+{}+{} gate_logit=1 + Q5K bufs)",
            total_bytes as f64 / 1024.0,
            num_experts, hidden_size,
            expert_ffn, expert_ffn, expert_ffn, hidden_size,
        );

        Ok(Self {
            router_logits,
            topk_indices,
            topk_weights,
            combined_buf,
            shexp_gate,
            shexp_up,
            shexp_inter,
            shexp_out,
            shexp_gate_logit,
            shexp_q5k_bufs,
            hidden_size,
            expert_ffn,
            num_experts,
            top_k,
        })
    }
}

// ---------------------------------------------------------------------------
// Pure cudarc MoE FFN forward
// ---------------------------------------------------------------------------

/// Run the full MoE FFN (routed experts + shared expert) using only cudarc ops.
///
/// # Arguments
/// - `hidden_view`: post-norm hidden state [hidden_size] as CudaView
/// - `moe`: weight references (Candle Tensors, but we only extract CudaView from them)
/// - `moe_raw`: pre-extracted CudaSlice refs from RawWeights (if available)
/// - `bufs`: pre-allocated scratch buffers
/// - `dev`: CUDA device handle
///
/// # Returns
/// `CudaSlice<f32>` [hidden_size] containing `routed_output + sigmoid_gate * shared_expert_output`.
/// The caller adds this to the residual stream.
///
/// # Zero Candle Tensor ops
/// All intermediate computations use CudaSlice directly. The only Candle
/// interaction is extracting CudaView from the weight Tensors via
/// `storage_and_layout()` (when RawWeights are not available).
pub(crate) fn moe_ffn_cudarc(
    hidden_view: &CudaView<'_, f32>,
    moe_raw: &crate::raw_weights::MoeRawWeightRefs<'_>,
    bufs: &mut MoeCudarcBufs,
    dev: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    let hidden_size = bufs.hidden_size;
    let expert_ffn = bufs.expert_ffn;
    let num_experts = bufs.num_experts;
    let top_k = bufs.top_k;

    // Expert byte sizes
    let expert_elements_gate = moe_raw.gate_exps_shape.1 * moe_raw.gate_exps_shape.2;
    let expert_elements_up = moe_raw.up_exps_shape.1 * moe_raw.up_exps_shape.2;
    let expert_elements_down = moe_raw.down_exps_shape.1 * moe_raw.down_exps_shape.2;
    let expert_bytes_gate = (expert_elements_gate / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
    let expert_bytes_up = (expert_elements_up / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
    let expert_bytes_down = (expert_elements_down / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;

    // =====================================================================
    // 1. Router: raw_f32_gemv(gate_inp @ hidden) -> router_logits [256]
    // =====================================================================
    {
        let gi_view = moe_raw.gate_inp.slice(..);
        crate::kernels::elementwise::raw_f32_gemv(
            &gi_view,
            hidden_view,
            &mut bufs.router_logits,
            num_experts,
            hidden_size,
            dev,
        )?;
    }

    // =====================================================================
    // 2. GPU top-K softmax -> topk_indices [8], topk_weights [8]
    // =====================================================================
    {
        let rl_view = bufs.router_logits.slice(..);
        crate::kernels::topk_softmax::gpu_topk_softmax(
            &rl_view,
            &mut bufs.topk_indices,
            &mut bufs.topk_weights,
            num_experts,
            top_k,
            dev,
        )?;
    }

    // =====================================================================
    // 3. Routed experts: fused MoE IQ3_S (GPU-resident expert IDs)
    // =====================================================================
    // Zero the output accumulator
    dev.cuda_stream()
        .memset_zeros(&mut bufs.combined_buf)
        .map_err(|e| candle_core::Error::Msg(format!("zero combined_buf: {e}")))?;

    {
        let g_view = moe_raw.gate_exps.slice(..);
        let u_view = moe_raw.up_exps.slice(..);
        let d_view = moe_raw.down_exps.slice(..);

        crate::kernels::fused_moe::fused_moe_iq3s_gpu_resident(
            hidden_view,
            &g_view,
            &u_view,
            &d_view,
            &bufs.topk_indices,
            &bufs.topk_weights,
            &mut bufs.combined_buf,
            hidden_size,
            expert_ffn,
            expert_bytes_gate,
            expert_bytes_up,
            expert_bytes_down,
            top_k,
            dev,
        )?;
    }

    // =====================================================================
    // 4. Shared expert (Q5_K weights, always active)
    // =====================================================================
    // Only available when RawWeights has the shared expert CudaSlice refs.
    let has_shared = moe_raw.shared_gate.is_some()
        && moe_raw.shared_up.is_some()
        && moe_raw.shared_down.is_some()
        && moe_raw.gate_inp_shexp.is_some();

    if has_shared {
        let gate_raw = moe_raw.shared_gate.unwrap();
        let up_raw = moe_raw.shared_up.unwrap();
        let down_raw = moe_raw.shared_down.unwrap();
        let gate_inp_shexp = moe_raw.gate_inp_shexp.unwrap();

        // 4a. Quantize hidden to Q8_1 (reused for gate AND up projections)
        crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
            hidden_view,
            &mut bufs.shexp_q5k_bufs.q8_input,
            hidden_size,
            dev,
        )?;

        // 4b. Gate GEMV: Q5_K [expert_ffn, hidden_size] @ Q8_1(hidden) -> shexp_gate [expert_ffn]
        {
            let gate_view = gate_raw.slice(..);
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                &gate_view,
                &bufs.shexp_q5k_bufs.q8_input,
                &mut bufs.shexp_gate,
                expert_ffn,
                hidden_size,
                dev,
            )?;
        }

        // 4c. Up GEMV: Q5_K [expert_ffn, hidden_size] @ Q8_1(hidden) -> shexp_up [expert_ffn]
        {
            let up_view = up_raw.slice(..);
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                &up_view,
                &bufs.shexp_q5k_bufs.q8_input,
                &mut bufs.shexp_up,
                expert_ffn,
                hidden_size,
                dev,
            )?;
        }

        // 4d. SwiGLU: silu(gate) * up -> inter
        crate::kernels::elementwise::raw_silu_mul(
            &bufs.shexp_gate,
            &bufs.shexp_up,
            &mut bufs.shexp_inter,
            expert_ffn,
            dev,
        )?;

        // 4e. Down GEMV: Q5_K [hidden_size, expert_ffn] @ Q8_1(inter) -> shexp_out [hidden_size]
        // Need to quantize intermediate to Q8_1 (different size: expert_ffn=512)
        crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
            &bufs.shexp_inter.slice(..),
            &mut bufs.shexp_q5k_bufs.q8_input,
            expert_ffn,
            dev,
        )?;

        {
            let down_view = down_raw.slice(..);
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                &down_view,
                &bufs.shexp_q5k_bufs.q8_input,
                &mut bufs.shexp_out,
                hidden_size,
                expert_ffn,
                dev,
            )?;
        }

        // 4f. Sigmoid gate: dot(hidden, gate_inp_shexp) -> scalar
        {
            let shexp_view = gate_inp_shexp.slice(..);
            crate::scratch_pool::raw_dot_product(
                hidden_view,
                &shexp_view,
                &mut bufs.shexp_gate_logit,
                hidden_size,
                dev,
            )?;
        }

        // 4g. Fused: shexp_out = shexp_out * sigmoid(gate_logit) + combined_buf
        crate::scratch_pool::raw_sigmoid_gate_add_inplace(
            &mut bufs.shexp_out,
            &bufs.shexp_gate_logit,
            &bufs.combined_buf,
            hidden_size,
            dev,
        )?;

        // Result is in shexp_out: sigmoid_gate * shared_out + routed_out
        // Clone to return (the buffer itself must remain for reuse next layer)
        let result = bufs.shexp_out.try_clone()
            .map_err(|e| candle_core::Error::Msg(format!("cudarc moe clone result: {e}")))?;
        Ok(result)
    } else {
        // No shared expert raw weights available -- return just the routed output.
        // This shouldn't happen in production (RawWeights always has shared expert).
        let result = bufs.combined_buf.try_clone()
            .map_err(|e| candle_core::Error::Msg(format!("cudarc moe clone combined: {e}")))?;
        Ok(result)
    }
}

/// Wrap a `CudaSlice<f32>` as a Candle `Tensor` [1, n] for residual add compatibility.
///
/// Zero-copy: wraps the existing device memory into a Candle Tensor without any
/// data movement. The CudaSlice ownership is transferred to the Tensor.
pub(crate) fn cuda_slice_to_tensor(
    slice: CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
        slice,
        dev.clone(),
    );
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(&[1, n]),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
