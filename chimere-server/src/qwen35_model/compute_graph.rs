//! Zero-allocation compute graph for cudarc forward pass.
//! All buffers pre-allocated at init. Zero cudaMalloc per token.
//!
//! ## Architecture
//!
//! This is the central data structure for the cudarc rewrite. It defines:
//! - `ComputeGraph`: pre-allocated GPU scratch buffers reused every token.
//! - `ModelWeightsRaw`: raw GGUF weight bytes on GPU (CudaSlice<u8>), no QMatMul.
//! - `forward_token()`: the main loop skeleton calling layer_gdn/layer_attn/moe_ffn.
//!
//! All other agents (layer_gdn, layer_attn, moe_ffn) depend on these types.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;

use crate::config::Qwen35Config;
use crate::gguf_loader::GgufFile;
use crate::kernels::ggml_gpu::GgmlGpuBuffers;
use crate::kernels::q5k_mmvq_ggml::GgmlQ5KBuffers;

// ---------------------------------------------------------------------------
// Helpers: layer type classification
// ---------------------------------------------------------------------------

/// Returns `true` if layer `il` is a GDN (recurrent) layer.
///
/// Qwen3.5-35B-A3B: every 4th layer (il+1) % 4 == 0 is attention, rest are GDN.
#[inline]
pub(crate) fn is_gdn(il: usize, full_attn_interval: usize) -> bool {
    (il + 1) % full_attn_interval != 0
}

/// Returns the GDN state index for layer `il` (0-based among GDN layers only).
///
/// Panics if `il` is not a GDN layer.
#[inline]
pub(crate) fn gdn_index(il: usize, full_attn_interval: usize) -> usize {
    debug_assert!(is_gdn(il, full_attn_interval), "layer {il} is not GDN");
    (0..il).filter(|&l| is_gdn(l, full_attn_interval)).count()
}

// ---------------------------------------------------------------------------
// ComputeGraph — pre-allocated GPU scratch buffers
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for one token forward pass.
///
/// All buffers are allocated once at model init. Zero cudaMalloc per token.
/// Layer intermediates are reused 40 times (once per layer). Attention and GDN
/// buffers are reused only on their respective layer types.
///
/// Total GPU memory: ~2.5 MB (trivial vs 15 GB model weights).
pub(crate) struct ComputeGraph {
    pub dev: CudaDevice,

    // -----------------------------------------------------------------------
    // Layer intermediates (reused 40x)
    // -----------------------------------------------------------------------
    /// Current hidden state: [hidden_size]
    pub hidden: CudaSlice<f32>,
    /// Residual connection buffer: [hidden_size]
    pub residual: CudaSlice<f32>,
    /// Post-RMSNorm output: [hidden_size]
    pub normed: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // MoE FFN scratch (reused 40x)
    // -----------------------------------------------------------------------
    /// Router logits: [n_experts]
    pub router_out: CudaSlice<f32>,
    /// Routed expert gate GEMV output: [expert_ffn]
    pub gate_buf: CudaSlice<f32>,
    /// Routed expert up GEMV output: [expert_ffn]
    pub up_buf: CudaSlice<f32>,
    /// Routed expert down GEMV output: [hidden_size]
    pub down_buf: CudaSlice<f32>,
    /// Accumulated weighted expert outputs: [hidden_size]
    pub expert_accum: CudaSlice<f32>,
    /// Shared expert gate GEMV output: [shexp_ffn]
    pub shexp_gate: CudaSlice<f32>,
    /// Shared expert up GEMV output: [shexp_ffn]
    pub shexp_up: CudaSlice<f32>,
    /// Shared expert down GEMV output: [hidden_size]
    pub shexp_down: CudaSlice<f32>,
    /// Shared expert gate scalar (sigmoid): [1]
    pub shexp_gate_logit: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // GDN scratch (reused 30x — only GDN layers)
    // -----------------------------------------------------------------------
    /// GDN SSM combined QKV+gate projection output: [conv_channels]
    /// conv_channels = key_dim*2 + value_dim = 2048*2 + 4096 = 8192
    pub gdn_proj: CudaSlice<f32>,
    /// GDN gate (z) projection output: [ssm_d_inner]
    pub gdn_gate: CudaSlice<f32>,
    /// GDN ssm_out projection output: [hidden_size]
    pub gdn_out: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // Attention scratch (reused 10x — only attention layers)
    // -----------------------------------------------------------------------
    /// Q projection output: [n_heads * q_head_dim]
    /// For Qwen3.5-35B-A3B: Q head_dim=512 (asymmetric), but the fused Q+gate
    /// projection outputs [n_heads * head_dim * 2] = [16 * 256 * 2] = [8192].
    /// We allocate max(q_proj_size, 8192) to handle the fused case.
    pub q_buf: CudaSlice<f32>,
    /// K projection output: [n_kv_heads * head_dim]
    pub k_buf: CudaSlice<f32>,
    /// V projection output: [n_kv_heads * head_dim]
    pub v_buf: CudaSlice<f32>,
    /// Attention output (after O projection): [hidden_size]
    pub attn_out: CudaSlice<f32>,
    /// Q heads after deinterleave + per-head RMSNorm: [n_heads * head_dim]
    pub q_heads: CudaSlice<f32>,
    /// Gate heads after deinterleave: [n_heads * head_dim]
    pub gate_heads: CudaSlice<f32>,
    /// Q after MRoPE rotation: [n_heads * head_dim]
    pub q_roped: CudaSlice<f32>,
    /// K after MRoPE rotation: [n_kv_heads * head_dim]
    pub k_roped: CudaSlice<f32>,
    /// Attention result before gating: [n_heads * head_dim]
    pub attn_result: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // LM head output
    // -----------------------------------------------------------------------
    /// Final logits: [vocab_size]
    pub logits: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // MoE top-K scratch (reused 40x)
    // -----------------------------------------------------------------------
    /// Top-K selected expert indices: [top_k] on GPU (stays GPU-resident).
    pub topk_indices: CudaSlice<i32>,
    /// Top-K renormalized expert weights: [top_k] on GPU (stays GPU-resident).
    pub topk_weights: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // Q5K GEMV scratch (reused for shared expert)
    // -----------------------------------------------------------------------
    pub q5k_bufs: GgmlQ5KBuffers,

    // -----------------------------------------------------------------------
    // ggml GPU MMVQ scratch (CHIMERE_GGML_GPU=1)
    // -----------------------------------------------------------------------
    /// Pre-allocated buffers for ggml's optimized MMVQ GEMV kernels.
    /// None when CHIMERE_GGML_GPU is not set (falls back to chimere fused kernels).
    pub ggml_gpu_bufs: Option<GgmlGpuBuffers>,

    // -----------------------------------------------------------------------
    // ncmoe CPU staging
    // -----------------------------------------------------------------------
    /// CPU staging buffer for expert weight transfers (~10 MB max).
    pub expert_staging_cpu: Vec<u8>,

    // -----------------------------------------------------------------------
    // Dimensions (for kernel launch configs)
    // -----------------------------------------------------------------------
    pub hidden_size: usize,
    pub expert_ffn: usize,
    pub shexp_ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub full_attn_interval: usize,
    // GDN dimensions
    pub ssm_d_state: usize,
    pub ssm_d_inner: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,
    pub ssm_conv_kernel: usize,

    // -----------------------------------------------------------------------
    // GDN cudarc scratch (lazy-init, for forward_gdn_cudarc)
    // -----------------------------------------------------------------------
    /// Pre-allocated scratch buffers for the pure cudarc GDN forward path.
    /// Lazy-init on first call to `forward_gdn_cudarc`. None until then.
    pub gdn_cudarc: Option<GdnCudarcScratch>,

    // -----------------------------------------------------------------------
    // CUDA Graph cache for GDN layers (Phase 3.1)
    // -----------------------------------------------------------------------
    /// Captured CUDA Graphs for GDN layer forward passes.
    /// Initialized when `CHIMERE_CUDA_GRAPH=1`. None otherwise.
    pub gdn_graph_cache: Option<super::cuda_graph::GdnGraphCache>,

    // -----------------------------------------------------------------------
    // RoPE tables (precomputed, for attention layers)
    // -----------------------------------------------------------------------
    /// Precomputed MRoPE cos/sin tables on GPU. Built at init time.
    /// None until `init_rope_tables()` is called.
    pub rope_tables: Option<crate::kernels::RawMRoPETables>,

    // -----------------------------------------------------------------------
    // Cached weight pointers (raw CUdeviceptr, extracted once at init)
    // -----------------------------------------------------------------------
    /// Pre-extracted raw CUDA device pointers for weight tensors.
    /// Eliminates device_ptr()/Arc/SyncOnDrop overhead per FFI call.
    /// None until `init_cached_ptrs()` is called.
    pub cached_ptrs: Option<CachedWeightPtrs>,

    // -----------------------------------------------------------------------
    // Batch prefill buffers (lazy-init)
    // -----------------------------------------------------------------------
    /// Pre-allocated GPU buffers for batch prefill (N tokens at once).
    /// None until `init_prefill_buffers()` is called.
    pub prefill_bufs: Option<PrefillBuffers>,
}

/// Pre-allocated GPU scratch for the pure cudarc GDN forward path.
///
/// All buffers are allocated once and reused across GDN layers (30x per token).
/// Per-layer persistent state (conv states, DeltaNet states) are stored in
/// the `conv_states` and `state_scratch` fields.
///
/// Total: ~6 MB scratch + 30 * (2 MB state + 96 KB conv) = ~66 MB per-layer state.
pub(crate) struct GdnCudarcScratch {
    // -- SSM projection intermediates --
    /// Beta projection: [dt_rank]
    pub beta_proj: CudaSlice<f32>,
    /// Alpha projection: [dt_rank]
    pub alpha_proj: CudaSlice<f32>,
    /// Beta output (sigmoid): [dt_rank]
    pub beta_out: CudaSlice<f32>,
    /// Gate exp output: [dt_rank]
    pub gate_exp_out: CudaSlice<f32>,

    // -- Conv + split intermediates --
    /// Conv1d + SiLU output: [conv_channels]
    pub conv_output: CudaSlice<f32>,
    /// New conv state: [conv_channels * (conv_kernel - 1)]
    pub new_conv_state: CudaSlice<f32>,

    // -- QKV split + normalize + expand --
    /// Q split: [key_dim]
    pub q_split: CudaSlice<f32>,
    /// K split: [key_dim]
    pub k_split: CudaSlice<f32>,
    /// Q L2-normed: [key_dim]
    pub q_normed: CudaSlice<f32>,
    /// K L2-normed: [key_dim]
    pub k_normed: CudaSlice<f32>,
    /// Q expanded to dt_rank heads: [value_dim]
    pub q_expanded: CudaSlice<f32>,
    /// K expanded to dt_rank heads: [value_dim]
    pub k_expanded: CudaSlice<f32>,
    /// Q scaled by 1/sqrt(d_state): [value_dim]
    pub q_scaled: CudaSlice<f32>,
    /// V copied from conv output: [value_dim]
    pub v_copy: CudaSlice<f32>,

    // -- DeltaNet output --
    /// DeltaNet step output: [value_dim]
    pub ssm_output: CudaSlice<f32>,
    /// DeltaNet new state scratch: [dt_rank * d_state * d_state]
    pub state_scratch: CudaSlice<f32>,

    // -- Post-SSM --
    /// RMSNorm + SiLU gate fused output: [value_dim]
    pub gated: CudaSlice<f32>,

    // -- Q8_1 quantization buffers (for Q5_K GEMV) --
    /// Q8_1 quantized hidden input: pad(hidden_size, 512) * 36 / 32 bytes
    pub q8_hidden: CudaSlice<u8>,
    /// Q8_1 quantized gated output: pad(value_dim, 512) * 36 / 32 bytes
    pub q8_gated: CudaSlice<u8>,

    // -- Per-GDN-layer persistent conv states --
    /// Conv1d sliding window states: one per GDN layer.
    /// Each is [conv_channels * (conv_kernel - 1)] flat.
    pub conv_states: Vec<CudaSlice<f32>>,
}

impl GdnCudarcScratch {
    /// Allocate all GDN cudarc scratch buffers.
    pub fn new(config: &Qwen35Config, dev: &CudaDevice) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let key_dim = config.ssm_n_group * config.ssm_d_state;
        let value_dim = config.ssm_dt_rank * config.ssm_d_state;
        let conv_channels = key_dim * 2 + value_dim;
        let conv_kernel = config.ssm_conv_kernel;
        let dt_rank = config.ssm_dt_rank;
        let d_state = config.ssm_d_state;
        let km1 = conv_kernel - 1;

        let af = |n: usize, name: &str| -> Result<CudaSlice<f32>> {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| candle_core::Error::Msg(format!("GdnCudarcScratch alloc {name}({n}): {e}")))
        };
        let au = |n: usize, name: &str| -> Result<CudaSlice<u8>> {
            dev.alloc_zeros::<u8>(n)
                .map_err(|e| candle_core::Error::Msg(format!("GdnCudarcScratch alloc {name}({n}): {e}")))
        };

        // Q8_1 buffer size: pad to 512, then (n/32)*36
        let q8_size = |n: usize| -> usize {
            let padded = (n + 511) / 512 * 512;
            (padded / 32) * 36
        };

        // Count GDN layers
        let num_gdn = config.num_gdn_layers();
        let conv_state_size = conv_channels * km1;
        let mut conv_states = Vec::with_capacity(num_gdn);
        for i in 0..num_gdn {
            conv_states.push(af(conv_state_size, &format!("conv_state_{i}"))?);
        }

        let state_size = dt_rank * d_state * d_state;
        let total_bytes = (dt_rank * 4 + conv_channels * 2 + conv_state_size
            + key_dim * 4 + value_dim * 6 + state_size + hidden_size) * 4
            + q8_size(hidden_size) + q8_size(value_dim)
            + num_gdn * conv_state_size * 4;
        eprintln!(
            "[GDN_CUDARC] Allocated scratch: {:.1} KB reusable + {:.1} MB per-layer conv states ({} layers)",
            (total_bytes - num_gdn * conv_state_size * 4) as f64 / 1024.0,
            (num_gdn * conv_state_size * 4) as f64 / (1024.0 * 1024.0),
            num_gdn,
        );

        Ok(Self {
            beta_proj: af(dt_rank, "beta_proj")?,
            alpha_proj: af(dt_rank, "alpha_proj")?,
            beta_out: af(dt_rank, "beta_out")?,
            gate_exp_out: af(dt_rank, "gate_exp_out")?,
            conv_output: af(conv_channels, "conv_output")?,
            new_conv_state: af(conv_state_size, "new_conv_state")?,
            q_split: af(key_dim, "q_split")?,
            k_split: af(key_dim, "k_split")?,
            q_normed: af(key_dim, "q_normed")?,
            k_normed: af(key_dim, "k_normed")?,
            q_expanded: af(value_dim, "q_expanded")?,
            k_expanded: af(value_dim, "k_expanded")?,
            q_scaled: af(value_dim, "q_scaled")?,
            v_copy: af(value_dim, "v_copy")?,
            ssm_output: af(value_dim, "ssm_output")?,
            state_scratch: af(state_size, "state_scratch")?,
            gated: af(value_dim, "gated")?,
            q8_hidden: au(q8_size(hidden_size), "q8_hidden")?,
            q8_gated: au(q8_size(value_dim), "q8_gated")?,
            conv_states,
        })
    }

    /// Allocate GDN cudarc scratch from ComputeGraph dimensions (no Qwen35Config needed).
    pub fn from_graph(graph: &ComputeGraph) -> Result<Self> {
        let hidden_size = graph.hidden_size;
        let d_state = graph.ssm_d_state;
        let dt_rank = graph.ssm_dt_rank;
        let n_group = graph.ssm_n_group;
        let conv_kernel = graph.ssm_conv_kernel;
        let key_dim = n_group * d_state;
        let value_dim = dt_rank * d_state;
        let conv_channels = key_dim * 2 + value_dim;
        let km1 = conv_kernel - 1;

        let dev = &graph.dev;
        let af = |n: usize, name: &str| -> Result<CudaSlice<f32>> {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| candle_core::Error::Msg(format!("GdnCudarcScratch alloc {name}({n}): {e}")))
        };
        let au = |n: usize, name: &str| -> Result<CudaSlice<u8>> {
            dev.alloc_zeros::<u8>(n)
                .map_err(|e| candle_core::Error::Msg(format!("GdnCudarcScratch alloc {name}({n}): {e}")))
        };
        let q8_size = |n: usize| -> usize {
            let padded = (n + 511) / 512 * 512;
            (padded / 32) * 36
        };

        // Count GDN layers
        let num_gdn = (0..graph.num_layers)
            .filter(|&l| is_gdn(l, graph.full_attn_interval))
            .count();
        let conv_state_size = conv_channels * km1;
        let mut conv_states = Vec::with_capacity(num_gdn);
        for i in 0..num_gdn {
            conv_states.push(af(conv_state_size, &format!("conv_state_{i}"))?);
        }

        let state_size = dt_rank * d_state * d_state;
        eprintln!(
            "[GDN_CUDARC] from_graph: {} GDN layers, dt_rank={}, d_state={}, conv_channels={}",
            num_gdn, dt_rank, d_state, conv_channels,
        );

        Ok(Self {
            beta_proj: af(dt_rank, "beta_proj")?,
            alpha_proj: af(dt_rank, "alpha_proj")?,
            beta_out: af(dt_rank, "beta_out")?,
            gate_exp_out: af(dt_rank, "gate_exp_out")?,
            conv_output: af(conv_channels, "conv_output")?,
            new_conv_state: af(conv_state_size, "new_conv_state")?,
            q_split: af(key_dim, "q_split")?,
            k_split: af(key_dim, "k_split")?,
            q_normed: af(key_dim, "q_normed")?,
            k_normed: af(key_dim, "k_normed")?,
            q_expanded: af(value_dim, "q_expanded")?,
            k_expanded: af(value_dim, "k_expanded")?,
            q_scaled: af(value_dim, "q_scaled")?,
            v_copy: af(value_dim, "v_copy")?,
            ssm_output: af(value_dim, "ssm_output")?,
            state_scratch: af(state_size, "state_scratch")?,
            gated: af(value_dim, "gated")?,
            q8_hidden: au(q8_size(hidden_size), "q8_hidden")?,
            q8_gated: au(q8_size(value_dim), "q8_gated")?,
            conv_states,
        })
    }

    /// Zero all conv_states for a new request (multi-turn reset).
    ///
    /// Must be called between requests to avoid conv1d state bleeding
    /// from a previous generation into the next one.
    pub fn reset_conv_states(&mut self, dev: &CudaDevice) -> Result<()> {
        for (i, cs) in self.conv_states.iter_mut().enumerate() {
            let zeros = vec![0.0f32; cs.len()];
            dev.memcpy_htod(&zeros, cs)
                .map_err(|e| candle_core::Error::Msg(
                    format!("reset_conv_state[{i}]: {e}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PrefillBuffers — pre-allocated GPU buffers for batch prefill
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for batch prefill.
///
/// Processes N tokens through each layer at once instead of one-at-a-time.
/// Batch projections (Q8_1 quantize + GEMV) and batch elementwise ops are
/// parallelized; GDN recurrence and MoE FFN stay sequential per-token.
///
/// Total GPU memory: ~max_prefill * 200 KB (trivial vs 15 GB model weights).
pub(crate) struct PrefillBuffers {
    /// Batch hidden states: [max_prefill * hidden_size]
    pub hidden_batch: CudaSlice<f32>,
    /// Batch residual accumulator: [max_prefill * hidden_size]
    pub residual_batch: CudaSlice<f32>,
    /// Batch normed output: [max_prefill * hidden_size]
    pub normed_batch: CudaSlice<f32>,

    /// Batch Q8_1 quantized normed: [max_prefill * q8_row_bytes(hidden_size)]
    pub q8_batch: CudaSlice<u8>,

    /// Batch GDN QKV projection output: [max_prefill * conv_channels]
    pub gdn_proj_batch: CudaSlice<f32>,
    /// Batch GDN gate output: [max_prefill * value_dim]
    pub gdn_gate_batch: CudaSlice<f32>,
    /// Batch beta: [max_prefill * dt_rank]
    pub beta_batch: CudaSlice<f32>,
    /// Batch alpha: [max_prefill * dt_rank]
    pub alpha_batch: CudaSlice<f32>,

    /// Batch attention Q+gate output: [max_prefill * attn_q_proj_size]
    pub attn_q_batch: CudaSlice<f32>,
    /// Batch attention K output: [max_prefill * kv_proj_size]
    pub attn_k_batch: CudaSlice<f32>,
    /// Batch attention V output: [max_prefill * kv_proj_size]
    pub attn_v_batch: CudaSlice<f32>,

    /// Batch deinterleaved Q heads: [max_prefill * num_heads * head_dim]
    /// Output of batch deinterleave (Q only, gate-free).
    pub attn_q_heads_batch: CudaSlice<f32>,
    /// Batch deinterleaved gate heads: [max_prefill * num_heads * head_dim]
    /// Output of batch deinterleave (gate only).
    pub attn_gate_heads_batch: CudaSlice<f32>,
    /// Batch Q after per-head RMSNorm: [max_prefill * num_heads * head_dim]
    pub attn_q_normed_batch: CudaSlice<f32>,
    /// Batch K after per-head RMSNorm: [max_prefill * num_kv_heads * head_dim]
    pub attn_k_normed_batch: CudaSlice<f32>,
    /// Batch Q after MRoPE: [max_prefill * num_heads * head_dim]
    pub attn_q_roped_batch: CudaSlice<f32>,
    /// Batch K after MRoPE: [max_prefill * num_kv_heads * head_dim]
    pub attn_k_roped_batch: CudaSlice<f32>,

    /// Batch logits output for the LAST token only: [vocab_size]
    /// (only the last token's logits matter for generation)
    pub last_logits: CudaSlice<f32>,

    /// Maximum number of tokens that can be prefilled at once.
    pub max_prefill: usize,
}

impl PrefillBuffers {
    /// Allocate all prefill batch buffers on GPU.
    ///
    /// `max_prefill`: maximum number of tokens to batch (typically 512..2048).
    pub fn new(config: &Qwen35Config, max_prefill: usize, dev: &CudaDevice) -> Result<Self> {
        let hs = config.hidden_size;
        let dt_rank = config.ssm_dt_rank;
        let d_state = config.ssm_d_state;
        let n_group = config.ssm_n_group;
        let key_dim = n_group * d_state;            // 16*128 = 2048
        let value_dim = dt_rank * d_state;           // 32*128 = 4096
        let conv_channels = key_dim * 2 + value_dim; // 2048*2 + 4096 = 8192

        // Q8_1 row size for hidden_size
        let ncols_padded = crate::kernels::ggml_gpu::pad(hs, crate::kernels::ggml_gpu::MATRIX_ROW_PADDING);
        let q8_row_bytes = (ncols_padded / crate::kernels::ggml_gpu::Q8_1_BLOCK_ELEMS)
                         * crate::kernels::ggml_gpu::Q8_1_BLOCK_BYTES;

        // Attention dimensions
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let attn_q_size = num_heads * head_dim * 2; // Q + gate interleaved (fused projection)
        let attn_kv_size = num_kv_heads * head_dim;

        fn ae(name: &str, e: impl std::fmt::Display) -> candle_core::Error {
            candle_core::Error::Msg(format!("PrefillBuffers alloc {name}: {e}"))
        }

        // V2-2 batch intermediate sizes (deinterleaved Q/gate, normed, roped)
        let attn_q_heads_size = num_heads * head_dim;            // deinterleaved Q (no gate)
        let attn_batch_intermediates = attn_q_heads_size * 2     // q_heads + gate_heads
            + attn_q_heads_size * 2                              // q_normed + q_roped
            + attn_kv_size * 2;                                  // k_normed + k_roped

        let total_f32_elems = max_prefill * (hs * 3              // hidden, residual, normed
            + conv_channels + value_dim + dt_rank * 2            // GDN proj, gate, beta, alpha
            + attn_q_size + attn_kv_size * 2                     // attn Q, K, V
            + attn_batch_intermediates)                           // V2-2 batch intermediates
            + config.vocab_size;                                 // last_logits (1 token only)
        let total_u8_bytes = max_prefill * q8_row_bytes;
        let total_gpu_bytes = total_f32_elems * 4 + total_u8_bytes;
        eprintln!(
            "[PREFILL_BUFS] Allocating batch buffers: max_prefill={}, {:.1} MB GPU",
            max_prefill,
            total_gpu_bytes as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            hidden_batch: dev.alloc_zeros::<f32>(max_prefill * hs)
                .map_err(|e| ae("hidden_batch", e))?,
            residual_batch: dev.alloc_zeros::<f32>(max_prefill * hs)
                .map_err(|e| ae("residual_batch", e))?,
            normed_batch: dev.alloc_zeros::<f32>(max_prefill * hs)
                .map_err(|e| ae("normed_batch", e))?,
            q8_batch: dev.alloc_zeros::<u8>(max_prefill * q8_row_bytes)
                .map_err(|e| ae("q8_batch", e))?,
            gdn_proj_batch: dev.alloc_zeros::<f32>(max_prefill * conv_channels)
                .map_err(|e| ae("gdn_proj_batch", e))?,
            gdn_gate_batch: dev.alloc_zeros::<f32>(max_prefill * value_dim)
                .map_err(|e| ae("gdn_gate_batch", e))?,
            beta_batch: dev.alloc_zeros::<f32>(max_prefill * dt_rank)
                .map_err(|e| ae("beta_batch", e))?,
            alpha_batch: dev.alloc_zeros::<f32>(max_prefill * dt_rank)
                .map_err(|e| ae("alpha_batch", e))?,
            attn_q_batch: dev.alloc_zeros::<f32>(max_prefill * attn_q_size)
                .map_err(|e| ae("attn_q_batch", e))?,
            attn_k_batch: dev.alloc_zeros::<f32>(max_prefill * attn_kv_size)
                .map_err(|e| ae("attn_k_batch", e))?,
            attn_v_batch: dev.alloc_zeros::<f32>(max_prefill * attn_kv_size)
                .map_err(|e| ae("attn_v_batch", e))?,
            // V2-2 batch intermediates
            attn_q_heads_batch: dev.alloc_zeros::<f32>(max_prefill * attn_q_heads_size)
                .map_err(|e| ae("attn_q_heads_batch", e))?,
            attn_gate_heads_batch: dev.alloc_zeros::<f32>(max_prefill * attn_q_heads_size)
                .map_err(|e| ae("attn_gate_heads_batch", e))?,
            attn_q_normed_batch: dev.alloc_zeros::<f32>(max_prefill * attn_q_heads_size)
                .map_err(|e| ae("attn_q_normed_batch", e))?,
            attn_k_normed_batch: dev.alloc_zeros::<f32>(max_prefill * attn_kv_size)
                .map_err(|e| ae("attn_k_normed_batch", e))?,
            attn_q_roped_batch: dev.alloc_zeros::<f32>(max_prefill * attn_q_heads_size)
                .map_err(|e| ae("attn_q_roped_batch", e))?,
            attn_k_roped_batch: dev.alloc_zeros::<f32>(max_prefill * attn_kv_size)
                .map_err(|e| ae("attn_k_roped_batch", e))?,
            last_logits: dev.alloc_zeros::<f32>(config.vocab_size)
                .map_err(|e| ae("last_logits", e))?,
            max_prefill,
        })
    }
}

impl ComputeGraph {
    /// Reset all state for a new request (multi-turn).
    ///
    /// Zeros GDN recurrent states, KV cache position, and the given conv scratch.
    /// Also resets the CUDA Graph cache so graphs are re-captured for the new request.
    pub fn reset_for_new_request(
        &mut self,
        gdn_states: &mut [CudaSlice<f32>],
        gdn_scratch: &mut GdnCudarcScratch,
        kv_cache: &mut KvCacheRaw,
    ) -> Result<()> {
        for state in gdn_states.iter_mut() {
            let zeros = vec![0.0f32; state.len()];
            self.dev.memcpy_htod(&zeros, state)
                .map_err(|e| candle_core::Error::Msg(format!("reset gdn_state: {e}")))?;
        }
        gdn_scratch.reset_conv_states(&self.dev)?;
        kv_cache.pos = 0;
        // Reset CUDA Graph cache so graphs are re-captured for the new request.
        if let Some(ref mut cache) = self.gdn_graph_cache {
            cache.reset();
        }
        Ok(())
    }

    /// Allocate all scratch buffers from config.
    ///
    /// Total allocation: ~2.5 MB GPU + 10 MB CPU staging. Trivial.
    pub fn new(config: &Qwen35Config, dev: &CudaDevice) -> Result<Self> {
        fn ae(name: &str, e: impl std::fmt::Display) -> candle_core::Error {
            candle_core::Error::Msg(format!("ComputeGraph alloc {name}: {e}"))
        }

        let hidden_size = config.hidden_size;
        let expert_ffn = config.expert_ffn_hidden;
        let shexp_ffn = config.shared_expert_ffn_hidden;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let n_experts = config.num_experts;
        let top_k = config.experts_per_token;
        let vocab_size = config.vocab_size;

        // GDN dimensions
        let ssm_d_state = config.ssm_d_state;
        let ssm_d_inner = config.ssm_d_inner;
        let ssm_dt_rank = config.ssm_dt_rank;
        let ssm_n_group = config.ssm_n_group;
        let ssm_conv_kernel = config.ssm_conv_kernel;

        // Derived sizes
        let key_dim = ssm_n_group * ssm_d_state;                  // 16*128 = 2048
        let value_dim = ssm_dt_rank * ssm_d_state;                // 32*128 = 4096
        let conv_channels = key_dim * 2 + value_dim;              // 2048*2 + 4096 = 8192

        // Q projection is fused with gate: output = [n_heads * head_dim * 2]
        // For 35B-A3B: 16 * 256 * 2 = 8192
        let q_proj_size = n_heads * head_dim * 2;
        let kv_proj_size = n_kv_heads * head_dim;

        // --- Layer intermediates ---
        let hidden = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("hidden", e))?;
        let residual = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("residual", e))?;
        let normed = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("normed", e))?;

        // --- MoE FFN ---
        let router_out = dev.alloc_zeros::<f32>(n_experts).map_err(|e| ae("router_out", e))?;
        let gate_buf = dev.alloc_zeros::<f32>(expert_ffn).map_err(|e| ae("gate_buf", e))?;
        let up_buf = dev.alloc_zeros::<f32>(expert_ffn).map_err(|e| ae("up_buf", e))?;
        let down_buf = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("down_buf", e))?;
        let expert_accum = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("expert_accum", e))?;
        let shexp_gate = dev.alloc_zeros::<f32>(shexp_ffn).map_err(|e| ae("shexp_gate", e))?;
        let shexp_up = dev.alloc_zeros::<f32>(shexp_ffn).map_err(|e| ae("shexp_up", e))?;
        let shexp_down = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("shexp_down", e))?;
        let shexp_gate_logit = dev.alloc_zeros::<f32>(1).map_err(|e| ae("shexp_gate_logit", e))?;

        // --- GDN ---
        let gdn_proj = dev.alloc_zeros::<f32>(conv_channels).map_err(|e| ae("gdn_proj", e))?;
        let gdn_gate = dev.alloc_zeros::<f32>(ssm_d_inner).map_err(|e| ae("gdn_gate", e))?;
        let gdn_out = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("gdn_out", e))?;

        // --- Attention ---
        let q_buf = dev.alloc_zeros::<f32>(q_proj_size).map_err(|e| ae("q_buf", e))?;
        let k_buf = dev.alloc_zeros::<f32>(kv_proj_size).map_err(|e| ae("k_buf", e))?;
        let v_buf = dev.alloc_zeros::<f32>(kv_proj_size).map_err(|e| ae("v_buf", e))?;
        let attn_out = dev.alloc_zeros::<f32>(hidden_size).map_err(|e| ae("attn_out", e))?;
        let q_heads_size = n_heads * head_dim;
        let q_heads = dev.alloc_zeros::<f32>(q_heads_size).map_err(|e| ae("q_heads", e))?;
        let gate_heads = dev.alloc_zeros::<f32>(q_heads_size).map_err(|e| ae("gate_heads", e))?;
        let q_roped = dev.alloc_zeros::<f32>(q_heads_size).map_err(|e| ae("q_roped", e))?;
        let k_roped = dev.alloc_zeros::<f32>(kv_proj_size).map_err(|e| ae("k_roped", e))?;
        let attn_result = dev.alloc_zeros::<f32>(q_heads_size).map_err(|e| ae("attn_result", e))?;

        // --- LM head ---
        let logits = dev.alloc_zeros::<f32>(vocab_size).map_err(|e| ae("logits", e))?;

        // --- MoE top-K scratch ---
        let topk_indices = dev.alloc_zeros::<i32>(top_k).map_err(|e| ae("topk_indices", e))?;
        let topk_weights = dev.alloc_zeros::<f32>(top_k).map_err(|e| ae("topk_weights", e))?;

        // --- Q5K GEMV scratch ---
        // Max ncols = hidden_size (2048), max nrows = hidden_size (shared expert down proj)
        let q5k_bufs = GgmlQ5KBuffers::new(hidden_size, hidden_size, dev)?;

        // --- ggml GPU MMVQ scratch (CHIMERE_GGML_GPU=1) ---
        // max_ncols must cover ALL input dimensions including GDN value_dim (dt_rank × d_state)
        // and conv_channels (key_dim*2 + value_dim). These can be larger than hidden_size.
        let ggml_gpu_bufs = if crate::kernels::ggml_gpu::is_enabled() {
            let value_dim = ssm_dt_rank * ssm_d_state;  // 32 × 128 = 4096
            let key_dim = ssm_n_group * ssm_d_state;    // 16 × 128 = 2048
            let conv_channels = key_dim * 2 + value_dim; // 8192
            let max_ncols = hidden_size.max(expert_ffn).max(shexp_ffn).max(value_dim).max(conv_channels);
            let max_nrows = expert_ffn.max(hidden_size).max(vocab_size).max(shexp_ffn).max(conv_channels).max(value_dim);
            eprintln!(
                "[COMPUTE_GRAPH] ggml GPU MMVQ enabled: max_ncols={} max_nrows={}",
                max_ncols, max_nrows,
            );
            let bufs = GgmlGpuBuffers::with_moe(max_ncols, max_nrows, expert_ffn, hidden_size, top_k, dev)?;
            // When using a non-blocking stream (new_with_stream), ALL ggml FFI
            // calls must run on the device stream. Otherwise they execute on the
            // NULL (legacy default) stream, causing race conditions with cudarc.
            let raw_stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
            if !raw_stream.is_null() {
                unsafe { crate::kernels::ggml_gpu::set_global_stream(raw_stream); }
                eprintln!("[COMPUTE_GRAPH] ggml global stream override set (non-blocking)");
            }
            Some(bufs)
        } else {
            None
        };

        // --- CPU staging for ncmoe expert transfers ---
        // 10 MB is generous for batch-copying 8 experts' worth of IQ3_S bytes.
        // gate+up: 8 * ceil(512*2048/256)*110 = 8 * 440*110 = 387200 bytes each
        // down:    8 * ceil(2048*512/256)*110 = same
        // Total per layer: ~1.1 MB. 10 MB handles any realistic scenario.
        let expert_staging_cpu = vec![0u8; 10 * 1024 * 1024];

        let total_gpu_bytes = (hidden_size * 3           // hidden, residual, normed
            + n_experts                                   // router_out
            + expert_ffn * 2 + hidden_size * 2            // gate/up/down/expert_accum
            + shexp_ffn * 2 + hidden_size + 1             // shared expert
            + top_k * 2                                   // topk_indices(i32) + topk_weights(f32)
            + conv_channels + ssm_d_inner + hidden_size   // GDN
            + q_proj_size + kv_proj_size * 2 + hidden_size // attention (q/k/v/attn_out)
            + q_heads_size * 3 + kv_proj_size + q_heads_size // attn cudarc scratch
            + vocab_size)                                  // logits
            * 4;
        eprintln!(
            "[COMPUTE_GRAPH] Allocated scratch buffers: {:.1} KB GPU + 10 MB CPU staging",
            total_gpu_bytes as f64 / 1024.0,
        );

        Ok(Self {
            dev: dev.clone(),
            hidden,
            residual,
            normed,
            router_out,
            gate_buf,
            up_buf,
            down_buf,
            expert_accum,
            shexp_gate,
            shexp_up,
            shexp_down,
            shexp_gate_logit,
            gdn_proj,
            gdn_gate,
            gdn_out,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            q_heads,
            gate_heads,
            q_roped,
            k_roped,
            attn_result,
            logits,
            topk_indices,
            topk_weights,
            q5k_bufs,
            ggml_gpu_bufs,
            expert_staging_cpu,
            hidden_size,
            expert_ffn,
            shexp_ffn,
            n_heads,
            n_kv_heads,
            head_dim,
            n_experts,
            top_k,
            vocab_size,
            num_layers: config.num_main_layers,
            full_attn_interval: config.full_attn_interval,
            ssm_d_state,
            ssm_d_inner,
            ssm_dt_rank,
            ssm_n_group,
            ssm_conv_kernel,
            gdn_cudarc: None,
            gdn_graph_cache: None,
            rope_tables: None,
            cached_ptrs: None,
            prefill_bufs: None,
        })
    }

    /// Initialize RoPE tables from model config. Call once at model load time.
    pub fn init_rope_tables(&mut self, config: &Qwen35Config) -> Result<()> {
        let n_rot = config.rope_sections.iter().sum::<usize>() * 2;
        let tables = crate::kernels::RawMRoPETables::from_config(
            config.head_dim,
            n_rot,
            &config.rope_sections,
            config.rope_theta,
            65536, // max_pos: 64K context
            &self.dev,
        )?;
        self.rope_tables = Some(tables);
        Ok(())
    }

    /// Initialize CUDA Graph cache for GDN layers (Phase 3.1).
    ///
    /// Only activates if `CHIMERE_CUDA_GRAPH=1` environment variable is set.
    /// Call once at model load time, after `new()`.
    pub fn init_gdn_graph_cache(&mut self) -> Result<()> {
        if !super::cuda_graph::cuda_graph_enabled() {
            return Ok(());
        }

        let num_gdn = (0..self.num_layers)
            .filter(|&l| is_gdn(l, self.full_attn_interval))
            .count();
        let num_attn = self.num_layers - num_gdn;

        self.gdn_graph_cache = Some(
            super::cuda_graph::GdnGraphCache::new(num_gdn, num_attn, &self.dev)?
        );
        Ok(())
    }

    /// Initialize cached raw CUDA pointers for all weight tensors.
    ///
    /// Extracts `device_ptr()` once per weight tensor at model load time.
    /// Subsequent forward passes use these raw pointers directly, avoiding
    /// the Arc clone + SyncOnDrop overhead of `device_ptr()` per call.
    ///
    /// Call once at model load time, after `ModelWeightsRaw::from_gguf()`.
    pub fn init_cached_ptrs(&mut self, weights: &ModelWeightsRaw) {
        self.cached_ptrs = Some(CachedWeightPtrs::from_weights(weights));
    }

    /// Initialize batch prefill buffers.
    ///
    /// `max_prefill`: maximum number of tokens to process in one batch
    /// (typically 512..2048). Larger values use more GPU memory but allow
    /// processing longer prompts without chunking.
    ///
    /// Call once at model load time, after `new()`.
    pub fn init_prefill_buffers(&mut self, config: &Qwen35Config, max_prefill: usize) -> Result<()> {
        self.prefill_bufs = Some(PrefillBuffers::new(config, max_prefill, &self.dev)?);
        Ok(())
    }
}

// ===========================================================================
// ModelWeightsRaw — raw GGUF bytes on GPU, no QMatMul wrapper
// ===========================================================================

/// Raw model weights loaded directly from GGUF as CudaSlice<u8>.
///
/// No QMatMul, no Candle Tensor wrappers. The GGUF parser is the ONLY
/// place that uses Candle (for parsing + mmap). After loading, all weight
/// data is in raw CudaSlice form ready for direct kernel consumption.
///
/// The embedding table is kept on CPU as `Vec<f32>` because it is too
/// large for GPU (248320 * 2048 * 4 = ~2 GB). Only 1 row is copied
/// to GPU per token.
pub(crate) struct ModelWeightsRaw {
    /// Embedding table: [vocab_size, hidden_size] F32 on CPU.
    /// Only 1 row (hidden_size floats) is copied to GPU per token.
    pub embed_table: Vec<f32>,
    /// LM head weight: Q8_0 raw bytes on GPU.
    /// Shape: [vocab_size, hidden_size] quantized.
    pub lm_head_raw: CudaSlice<u8>,
    /// Final RMSNorm weight: [hidden_size] F32 on GPU.
    pub final_norm: CudaSlice<f32>,
    /// All layer weights, indexed by layer number.
    pub layers: Vec<LayerWeightsRaw>,
}

/// Layer type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerType {
    Gdn,
    Attention,
}

/// Raw weights for a single transformer layer.
pub(crate) struct LayerWeightsRaw {
    /// Pre-attention RMSNorm weight: [hidden_size] F32
    pub attn_norm: CudaSlice<f32>,
    /// Post-attention / FFN RMSNorm weight: [hidden_size] F32
    pub ffn_norm: CudaSlice<f32>,
    /// Whether this layer is GDN (recurrent) or full attention.
    pub layer_type: LayerType,
    /// GDN-specific weights (only for GDN layers).
    pub gdn: Option<GdnWeightsRaw>,
    /// Attention-specific weights (only for attention layers).
    pub attn: Option<AttnWeightsRaw>,
    /// MoE weights (all layers have MoE in the 35B-A3B model).
    pub moe: MoeWeightsRaw,
}

/// GDN (Gated DeltaNet / recurrent SSM) layer weights.
///
/// Tensor names from GGUF:
/// - `blk.{il}.attn_qkv.weight` — combined QKV projection (IQ3_S or Q5_K)
/// - `blk.{il}.attn_gate.weight` — gate projection (IQ3_S or Q5_K)
/// - `blk.{il}.ssm_out.weight` — output projection (IQ3_S or Q5_K)
/// - `blk.{il}.ssm_conv1d.weight` — conv1d weights (F32, small)
/// - `blk.{il}.ssm_norm.weight` — SSM norm (F32, small)
/// - `blk.{il}.ssm_a` — SSM decay (F32, small)
/// - `blk.{il}.ssm_dt.bias` — dt bias (F32, small)
/// - `blk.{il}.ssm_alpha.weight` — alpha projection (IQ3_S or Q5_K)
/// - `blk.{il}.ssm_beta.weight` — beta projection (IQ3_S or Q5_K)
pub(crate) struct GdnWeightsRaw {
    /// Combined QKV+gate projection: raw quantized bytes.
    /// GGUF name: `blk.{il}.attn_qkv.weight`
    /// Logical shape: [conv_channels, hidden_size] where conv_channels = key_dim*2 + value_dim
    pub ssm_in_raw: CudaSlice<u8>,
    /// Gate (z) projection: raw quantized bytes.
    /// GGUF name: `blk.{il}.attn_gate.weight`
    /// Logical shape: [value_dim, hidden_size]
    pub ssm_gate_raw: CudaSlice<u8>,
    /// Output projection: raw quantized bytes.
    /// GGUF name: `blk.{il}.ssm_out.weight`
    /// Logical shape: [hidden_size, value_dim]
    pub ssm_out_raw: CudaSlice<u8>,
    /// Alpha projection: raw quantized bytes.
    /// GGUF name: `blk.{il}.ssm_alpha.weight`
    /// Logical shape: [ssm_dt_rank, hidden_size]
    pub ssm_alpha_raw: CudaSlice<u8>,
    /// Alpha quant type (Q5_K or Q8_0 in custom-mix).
    pub alpha_quant: crate::gguf_loader::GgmlType,
    /// Beta projection: raw quantized bytes.
    /// GGUF name: `blk.{il}.ssm_beta.weight`
    /// Logical shape: [ssm_dt_rank, hidden_size]
    pub ssm_beta_raw: CudaSlice<u8>,
    /// Beta quant type (Q5_K or Q8_0 in custom-mix).
    pub beta_quant: crate::gguf_loader::GgmlType,
    /// Conv1d weights: F32 on GPU (small).
    /// GGUF name: `blk.{il}.ssm_conv1d.weight`
    /// Shape: [conv_channels, kernel_size]
    pub conv_weight: CudaSlice<f32>,
    /// SSM norm weight: F32 on GPU (small).
    /// GGUF name: `blk.{il}.ssm_norm.weight`
    /// Shape: [ssm_d_state]
    pub ssm_norm: CudaSlice<f32>,
    /// SSM decay parameter: F32 on GPU (small).
    /// GGUF name: `blk.{il}.ssm_a`
    /// Shape: [ssm_dt_rank]
    pub ssm_a: CudaSlice<f32>,
    /// DeltaNet dt bias: F32 on GPU (small).
    /// GGUF name: `blk.{il}.ssm_dt.bias`
    /// Shape: [ssm_dt_rank]
    pub dt_bias: CudaSlice<f32>,
}

/// Full attention layer weights.
///
/// Tensor names from GGUF:
/// - `blk.{il}.attn_q.weight` — Q projection (includes fused gate)
/// - `blk.{il}.attn_k.weight` — K projection
/// - `blk.{il}.attn_v.weight` — V projection
/// - `blk.{il}.attn_output.weight` — O projection
/// - `blk.{il}.attn_q_norm.weight` — Q RMSNorm (F32)
/// - `blk.{il}.attn_k_norm.weight` — K RMSNorm (F32)
pub(crate) struct AttnWeightsRaw {
    /// Q projection (fused with gate): raw quantized bytes.
    /// Logical shape: [n_heads * head_dim * 2, hidden_size]
    pub q_raw: CudaSlice<u8>,
    /// K projection: raw quantized bytes.
    /// Logical shape: [n_kv_heads * head_dim, hidden_size]
    pub k_raw: CudaSlice<u8>,
    /// V projection: raw quantized bytes.
    /// Logical shape: [n_kv_heads * head_dim, hidden_size]
    pub v_raw: CudaSlice<u8>,
    /// O projection: raw quantized bytes.
    /// Logical shape: [hidden_size, n_heads * head_dim]
    pub o_raw: CudaSlice<u8>,
    /// Quantization types for dispatch to correct GEMV kernel
    pub q_quant: crate::gguf_loader::GgmlType,
    pub k_quant: crate::gguf_loader::GgmlType,
    pub v_quant: crate::gguf_loader::GgmlType,
    pub o_quant: crate::gguf_loader::GgmlType,
    /// Q RMSNorm weight: [head_dim] or [n_heads * head_dim] F32
    pub q_norm: CudaSlice<f32>,
    /// K RMSNorm weight: [head_dim] or [n_kv_heads * head_dim] F32
    pub k_norm: CudaSlice<f32>,
}

/// MoE (Mixture-of-Experts) FFN weights for one layer.
///
/// Tensor names from GGUF:
/// - `blk.{il}.ffn_gate_inp.weight` — router gate (F32)
/// - `blk.{il}.ffn_gate_inp_shexp.weight` — shared expert gate (F32)
/// - `blk.{il}.ffn_gate_exps.weight` — routed experts gate (IQ3_S, packed 256)
/// - `blk.{il}.ffn_up_exps.weight` — routed experts up (IQ3_S, packed 256)
/// - `blk.{il}.ffn_down_exps.weight` — routed experts down (IQ3_S, packed 256)
/// - `blk.{il}.ffn_gate_shexp.weight` — shared expert gate (Q5_K)
/// - `blk.{il}.ffn_up_shexp.weight` — shared expert up (Q5_K)
/// - `blk.{il}.ffn_down_shexp.weight` — shared expert down (Q5_K)
pub(crate) struct MoeWeightsRaw {
    // -----------------------------------------------------------------------
    // Routed experts (packed, all 256 experts contiguous)
    // -----------------------------------------------------------------------
    /// Gate projections for all routed experts: IQ3_S packed bytes.
    /// On GPU when not offloaded, otherwise see `gate_exps_cpu`.
    pub gate_exps_raw: CudaSlice<u8>,
    /// Up projections for all routed experts: IQ3_S packed bytes.
    pub up_exps_raw: CudaSlice<u8>,
    /// Down projections for all routed experts: IQ3_S packed bytes.
    pub down_exps_raw: CudaSlice<u8>,
    /// Byte size of one expert's gate projection (IQ3_S).
    pub expert_bytes_gate: usize,
    /// Byte size of one expert's up projection (IQ3_S).
    pub expert_bytes_up: usize,
    /// Byte size of one expert's down projection (IQ3_S).
    pub expert_bytes_down: usize,

    // -----------------------------------------------------------------------
    // Router gate
    // -----------------------------------------------------------------------
    /// Router weight: [n_experts, hidden_size] F32 on GPU.
    /// Pre-transposed for GEMV: output = router_weight @ hidden -> [n_experts]
    pub router_weight: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // Shared expert (Q5_K, always active)
    // -----------------------------------------------------------------------
    /// Shared expert gate projection: Q5_K raw bytes on GPU.
    pub shexp_gate_raw: CudaSlice<u8>,
    /// Shared expert up projection: Q5_K raw bytes on GPU.
    pub shexp_up_raw: CudaSlice<u8>,
    /// Shared expert down projection: Q5_K raw bytes on GPU.
    pub shexp_down_raw: CudaSlice<u8>,
    /// Shared expert gate bias: [hidden_size] F32 on GPU.
    /// GGUF name: `blk.{il}.ffn_gate_inp_shexp.weight`
    pub shexp_gate_proj: CudaSlice<f32>,

    // -----------------------------------------------------------------------
    // ncmoe CPU offloading
    // -----------------------------------------------------------------------
    /// Whether this layer's routed experts are offloaded to CPU.
    pub experts_on_cpu: bool,
    /// CPU-resident gate expert bytes (only when `experts_on_cpu == true`).
    pub gate_exps_cpu: Option<Vec<u8>>,
    /// CPU-resident up expert bytes (only when `experts_on_cpu == true`).
    pub up_exps_cpu: Option<Vec<u8>>,
    /// CPU-resident down expert bytes (only when `experts_on_cpu == true`).
    pub down_exps_cpu: Option<Vec<u8>>,
}

// ===========================================================================
// CachedWeightPtrs — raw CUDA pointers extracted once at model load
// ===========================================================================

/// Pre-extracted raw CUDA device pointers for all weight tensors.
///
/// Avoids `device_ptr()` / Arc / SyncOnDrop overhead on every FFI call.
/// Pointers are extracted ONCE at model load time from the immutable
/// `ModelWeightsRaw` weight slices, then passed directly to FFI.
///
/// # Safety
///
/// - Weights are immutable after loading and owned by `ModelWeightsRaw`.
/// - The `ModelWeightsRaw` must outlive this struct.
/// - Raw pointers are only used for ggml FFI calls on the same CUDA device.
pub(crate) struct CachedWeightPtrs {
    // Per-GDN-layer weight pointers
    /// ssm_in_raw (attn_qkv): Q5K [conv_channels, hidden_size]
    pub gdn_ssm_in: Vec<*const std::ffi::c_void>,
    /// ssm_gate_raw (attn_gate): Q5K [value_dim, hidden_size]
    pub gdn_ssm_gate: Vec<*const std::ffi::c_void>,
    /// ssm_beta_raw: Q5K or Q8_0 [dt_rank, hidden_size]
    pub gdn_ssm_beta: Vec<*const std::ffi::c_void>,
    /// ssm_alpha_raw: Q5K or Q8_0 [dt_rank, hidden_size]
    pub gdn_ssm_alpha: Vec<*const std::ffi::c_void>,
    /// ssm_out_raw: Q5K [hidden_size, value_dim]
    pub gdn_ssm_out: Vec<*const std::ffi::c_void>,

    // Per-MoE-layer weight pointers
    /// All-expert gate projections (IQ3_S packed)
    pub moe_gate_exps: Vec<*const std::ffi::c_void>,
    /// All-expert up projections (IQ3_S packed)
    pub moe_up_exps: Vec<*const std::ffi::c_void>,
    /// All-expert down projections (IQ3_S packed)
    pub moe_down_exps: Vec<*const std::ffi::c_void>,
    /// Router weight: F32 [n_experts, hidden_size]
    pub moe_router: Vec<*const std::ffi::c_void>,

    // Per-MoE-layer byte strides (for batched expert dispatch)
    pub moe_expert_bytes_gate: Vec<usize>,
    pub moe_expert_bytes_up: Vec<usize>,
    pub moe_expert_bytes_down: Vec<usize>,

    // Shared expert pointers (per-layer)
    /// Shared expert gate Q5K raw bytes
    pub shexp_gate: Vec<*const std::ffi::c_void>,
    /// Shared expert up Q5K raw bytes
    pub shexp_up: Vec<*const std::ffi::c_void>,
    /// Shared expert down Q5K raw bytes
    pub shexp_down: Vec<*const std::ffi::c_void>,
    /// Shared expert gate projection F32
    pub shexp_gate_proj: Vec<*const std::ffi::c_void>,
}

// SAFETY: Raw pointers are derived from CudaSlice<u8>/CudaSlice<f32> which
// are GPU-resident and immutable after model loading. The pointers are only
// used for FFI calls on the same CUDA device within a single thread.
unsafe impl Send for CachedWeightPtrs {}
unsafe impl Sync for CachedWeightPtrs {}

/// Extract raw `*const c_void` from a `CudaSlice<T>`, calling `device_ptr()` once.
///
/// This is the key function: it pays the Arc+SyncOnDrop cost exactly once
/// (at init time), and the returned raw pointer can be reused indefinitely
/// as long as the underlying CudaSlice is alive.
#[inline]
fn extract_ptr<T>(slice: &CudaSlice<T>) -> *const std::ffi::c_void {
    let (ptr, _sync) = slice.device_ptr(slice.stream());
    ptr as *const std::ffi::c_void
}

impl CachedWeightPtrs {
    /// Build cached pointers from loaded model weights.
    ///
    /// Must be called AFTER `ModelWeightsRaw::from_gguf()` completes.
    /// The `ModelWeightsRaw` must not be dropped while this struct is in use.
    pub fn from_weights(weights: &ModelWeightsRaw) -> Self {
        let num_layers = weights.layers.len();

        let mut gdn_ssm_in = Vec::new();
        let mut gdn_ssm_gate = Vec::new();
        let mut gdn_ssm_beta = Vec::new();
        let mut gdn_ssm_alpha = Vec::new();
        let mut gdn_ssm_out = Vec::new();

        let mut moe_gate_exps = Vec::with_capacity(num_layers);
        let mut moe_up_exps = Vec::with_capacity(num_layers);
        let mut moe_down_exps = Vec::with_capacity(num_layers);
        let mut moe_router = Vec::with_capacity(num_layers);
        let mut moe_expert_bytes_gate = Vec::with_capacity(num_layers);
        let mut moe_expert_bytes_up = Vec::with_capacity(num_layers);
        let mut moe_expert_bytes_down = Vec::with_capacity(num_layers);
        let mut shexp_gate = Vec::with_capacity(num_layers);
        let mut shexp_up = Vec::with_capacity(num_layers);
        let mut shexp_down = Vec::with_capacity(num_layers);
        let mut shexp_gate_proj = Vec::with_capacity(num_layers);

        for lw in &weights.layers {
            // GDN weights
            if let Some(ref gdn) = lw.gdn {
                gdn_ssm_in.push(extract_ptr(&gdn.ssm_in_raw));
                gdn_ssm_gate.push(extract_ptr(&gdn.ssm_gate_raw));
                gdn_ssm_beta.push(extract_ptr(&gdn.ssm_beta_raw));
                gdn_ssm_alpha.push(extract_ptr(&gdn.ssm_alpha_raw));
                gdn_ssm_out.push(extract_ptr(&gdn.ssm_out_raw));
            }

            // MoE weights (all layers have MoE)
            moe_gate_exps.push(extract_ptr(&lw.moe.gate_exps_raw));
            moe_up_exps.push(extract_ptr(&lw.moe.up_exps_raw));
            moe_down_exps.push(extract_ptr(&lw.moe.down_exps_raw));
            moe_router.push(extract_ptr(&lw.moe.router_weight));
            moe_expert_bytes_gate.push(lw.moe.expert_bytes_gate);
            moe_expert_bytes_up.push(lw.moe.expert_bytes_up);
            moe_expert_bytes_down.push(lw.moe.expert_bytes_down);
            shexp_gate.push(extract_ptr(&lw.moe.shexp_gate_raw));
            shexp_up.push(extract_ptr(&lw.moe.shexp_up_raw));
            shexp_down.push(extract_ptr(&lw.moe.shexp_down_raw));
            shexp_gate_proj.push(extract_ptr(&lw.moe.shexp_gate_proj));
        }

        let num_gdn = gdn_ssm_in.len();
        eprintln!(
            "[CACHED_PTRS] Extracted {} GDN + {} MoE raw CUDA pointers",
            num_gdn * 5, num_layers * 8,
        );

        Self {
            gdn_ssm_in,
            gdn_ssm_gate,
            gdn_ssm_beta,
            gdn_ssm_alpha,
            gdn_ssm_out,
            moe_gate_exps,
            moe_up_exps,
            moe_down_exps,
            moe_router,
            moe_expert_bytes_gate,
            moe_expert_bytes_up,
            moe_expert_bytes_down,
            shexp_gate,
            shexp_up,
            shexp_down,
            shexp_gate_proj,
        }
    }
}

// ===========================================================================
// KV cache placeholder (for attention layers)
// ===========================================================================

/// Raw KV cache for attention layers.
///
/// Stores key and value tensors for all attention layers across all positions.
/// Details TBD by agent 3 (layer_attn).
pub(crate) struct KvCacheRaw {
    /// Per-attention-layer K cache: [max_seq_len, n_kv_heads, head_dim]
    pub k_cache: Vec<CudaSlice<f32>>,
    /// Per-attention-layer V cache: [max_seq_len, n_kv_heads, head_dim]
    pub v_cache: Vec<CudaSlice<f32>>,
    /// Current sequence position (next write index).
    pub pos: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl KvCacheRaw {
    /// Allocate KV cache for all attention layers.
    pub fn new(
        n_attn_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        let kv_size = max_seq_len * n_kv_heads * head_dim;
        let mut k_cache = Vec::with_capacity(n_attn_layers);
        let mut v_cache = Vec::with_capacity(n_attn_layers);
        for _i in 0..n_attn_layers {
            k_cache.push(
                dev.alloc_zeros::<f32>(kv_size)
                    .map_err(|e| candle_core::Error::Msg(format!("alloc k_cache: {e}")))?,
            );
            v_cache.push(
                dev.alloc_zeros::<f32>(kv_size)
                    .map_err(|e| candle_core::Error::Msg(format!("alloc v_cache: {e}")))?,
            );
        }
        Ok(Self {
            k_cache,
            v_cache,
            pos: 0,
            max_seq_len,
        })
    }
}

// ===========================================================================
// ModelWeightsRaw::from_gguf — GGUF loader (raw bytes, no QMatMul)
// ===========================================================================

impl ModelWeightsRaw {
    /// Load raw weight bytes directly from GGUF, bypassing QMatMul.
    ///
    /// This is the ONLY place that uses Candle (for GGUF parsing via mmap).
    /// After loading, all weight data lives as raw `CudaSlice<u8>` on GPU
    /// (or `Vec<u8>` on CPU for ncmoe-offloaded expert weights).
    ///
    /// The embedding table is dequantized to F32 and kept on CPU (~2 GB).
    /// Only 1 row per token is uploaded to GPU at forward time.
    ///
    /// `ncmoe`: number of layers whose routed expert weights are offloaded to CPU.
    pub fn from_gguf(
        gguf: &GgufFile,
        config: &Qwen35Config,
        dev: &CudaDevice,
        ncmoe: usize,
    ) -> Result<Self> {
        use candle_core::Device;

        let candle_dev = Device::Cuda(dev.clone());
        let num_layers = config.num_main_layers;

        eprintln!("[WEIGHTS_RAW] Loading raw model weights from GGUF ({num_layers} layers)...");
        let load_start = std::time::Instant::now();

        // --- Embedding table: dequant to F32, keep on CPU ---
        eprintln!("[WEIGHTS_RAW] Loading embedding table (CPU F32)...");
        let embed_tensor = gguf.load_tensor("token_embd.weight", &Device::Cpu)?;
        let embed_table: Vec<f32> = embed_tensor.flatten_all()?.to_vec1()?;
        eprintln!(
            "[WEIGHTS_RAW]   embed_tokens: {} elements ({:.1} MB CPU)",
            embed_table.len(),
            embed_table.len() as f64 * 4.0 / 1e6,
        );

        // --- LM head: raw quantized bytes on GPU ---
        eprintln!("[WEIGHTS_RAW] Loading LM head (GPU raw)...");
        let (lm_head_tensor, _n_elem, _dims) =
            gguf.load_tensor_u8_any("output.weight", &candle_dev)?;
        let lm_head_raw = tensor_to_cuda_u8(&lm_head_tensor)?;
        eprintln!(
            "[WEIGHTS_RAW]   lm_head: {} bytes on GPU",
            lm_head_raw.len(),
        );

        // --- Final norm: F32 on GPU ---
        let final_norm_tensor = gguf.load_tensor("output_norm.weight", &candle_dev)?;
        let final_norm = tensor_to_cuda_f32(&final_norm_tensor)?;

        // --- All layers ---
        let mut layers = Vec::with_capacity(num_layers);

        for il in 0..num_layers {
            let layer_start = std::time::Instant::now();
            let is_attn = config.is_attention(il);

            // Norms (F32, tiny)
            let attn_norm = load_f32_weight(gguf, &format!("blk.{il}.attn_norm.weight"), dev)?;
            let ffn_norm =
                load_f32_weight(gguf, &format!("blk.{il}.post_attention_norm.weight"), dev)?;

            // GDN or Attention weights
            let (gdn, attn, layer_type) = if is_attn {
                let q_name = format!("blk.{il}.attn_q.weight");
                let k_name = format!("blk.{il}.attn_k.weight");
                let v_name = format!("blk.{il}.attn_v.weight");
                let o_name = format!("blk.{il}.attn_output.weight");
                let attn_w = AttnWeightsRaw {
                    q_quant: gguf.tensor_ggml_type(&q_name).unwrap_or(crate::gguf_loader::GgmlType::Iq3S),
                    k_quant: gguf.tensor_ggml_type(&k_name).unwrap_or(crate::gguf_loader::GgmlType::Iq3S),
                    v_quant: gguf.tensor_ggml_type(&v_name).unwrap_or(crate::gguf_loader::GgmlType::Iq3S),
                    o_quant: gguf.tensor_ggml_type(&o_name).unwrap_or(crate::gguf_loader::GgmlType::Iq3S),
                    q_raw: load_raw_weight(gguf, &q_name, dev)?,
                    k_raw: load_raw_weight(gguf, &k_name, dev)?,
                    v_raw: load_raw_weight(gguf, &v_name, dev)?,
                    o_raw: load_raw_weight(gguf, &o_name, dev)?,
                    q_norm: load_f32_weight(
                        gguf,
                        &format!("blk.{il}.attn_q_norm.weight"),
                        dev,
                    )?,
                    k_norm: load_f32_weight(
                        gguf,
                        &format!("blk.{il}.attn_k_norm.weight"),
                        dev,
                    )?,
                };
                (None, Some(attn_w), LayerType::Attention)
            } else {
                let gdn_w = GdnWeightsRaw {
                    ssm_in_raw: load_raw_weight(
                        gguf,
                        &format!("blk.{il}.attn_qkv.weight"),
                        dev,
                    )?,
                    ssm_gate_raw: load_raw_weight(
                        gguf,
                        &format!("blk.{il}.attn_gate.weight"),
                        dev,
                    )?,
                    ssm_out_raw: load_raw_weight(
                        gguf,
                        &format!("blk.{il}.ssm_out.weight"),
                        dev,
                    )?,
                    ssm_alpha_raw: load_raw_weight(
                        gguf,
                        &format!("blk.{il}.ssm_alpha.weight"),
                        dev,
                    )?,
                    alpha_quant: gguf.tensor_ggml_type(&format!("blk.{il}.ssm_alpha.weight"))
                        .unwrap_or(crate::gguf_loader::GgmlType::Q5K),
                    ssm_beta_raw: load_raw_weight(
                        gguf,
                        &format!("blk.{il}.ssm_beta.weight"),
                        dev,
                    )?,
                    beta_quant: gguf.tensor_ggml_type(&format!("blk.{il}.ssm_beta.weight"))
                        .unwrap_or(crate::gguf_loader::GgmlType::Q5K),
                    conv_weight: load_f32_weight(
                        gguf,
                        &format!("blk.{il}.ssm_conv1d.weight"),
                        dev,
                    )?,
                    ssm_norm: load_f32_weight(
                        gguf,
                        &format!("blk.{il}.ssm_norm.weight"),
                        dev,
                    )?,
                    ssm_a: load_f32_weight(gguf, &format!("blk.{il}.ssm_a"), dev)?,
                    dt_bias: load_f32_weight(gguf, &format!("blk.{il}.ssm_dt.bias"), dev)?,
                };
                (Some(gdn_w), None, LayerType::Gdn)
            };

            // MoE weights (all layers are MoE in 35B-A3B)
            let experts_on_cpu = il < ncmoe;
            let expert_dev = if experts_on_cpu {
                &Device::Cpu
            } else {
                &candle_dev
            };

            // Router gate: F32, always on GPU
            // GGUF shape [hidden_size, n_experts] → Candle loads as [n_experts, hidden_size]
            // row-major. raw_f32_gemv expects W[e * hidden_size + h] = [n_experts, hidden_size]
            // which is exactly what load_tensor gives us. NO transpose needed.
            let router_tensor = gguf.load_tensor(
                &format!("blk.{il}.ffn_gate_inp.weight"),
                &candle_dev,
            )?;
            let router_weight = tensor_to_cuda_f32(&router_tensor)?;

            // Shared expert gate scalar
            let shexp_gate_tensor = gguf.load_tensor(
                &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
                &candle_dev,
            )?;
            let shexp_gate_proj = tensor_to_cuda_f32(&shexp_gate_tensor)?;

            // Routed expert weights: IQ3_S raw bytes
            let (gate_exps_tensor, _ge, _gd) = gguf.load_tensor_u8(
                &format!("blk.{il}.ffn_gate_exps.weight"),
                expert_dev,
            )?;
            let (up_exps_tensor, _ue, _ud) = gguf.load_tensor_u8(
                &format!("blk.{il}.ffn_up_exps.weight"),
                expert_dev,
            )?;
            let (down_exps_tensor, _de, _dd) = gguf.load_tensor_u8(
                &format!("blk.{il}.ffn_down_exps.weight"),
                expert_dev,
            )?;

            // Compute per-expert byte sizes from the total tensor size and num_experts.
            let n_experts = config.num_experts;
            let expert_bytes_gate = gate_exps_tensor.elem_count() / n_experts;
            let expert_bytes_up = up_exps_tensor.elem_count() / n_experts;
            let expert_bytes_down = down_exps_tensor.elem_count() / n_experts;

            // Route to GPU CudaSlice or CPU Vec<u8>
            let (gate_exps_raw, gate_exps_cpu) = if experts_on_cpu {
                let cpu_bytes: Vec<u8> = gate_exps_tensor.flatten_all()?.to_vec1()?;
                // Allocate a dummy 1-byte GPU slice (struct requires CudaSlice)
                let dummy = dev
                    .alloc_zeros::<u8>(1)
                    .map_err(|e| candle_core::Error::Msg(format!("dummy alloc: {e}")))?;
                (dummy, Some(cpu_bytes))
            } else {
                (tensor_to_cuda_u8(&gate_exps_tensor)?, None)
            };
            let (up_exps_raw, up_exps_cpu) = if experts_on_cpu {
                let cpu_bytes: Vec<u8> = up_exps_tensor.flatten_all()?.to_vec1()?;
                let dummy = dev
                    .alloc_zeros::<u8>(1)
                    .map_err(|e| candle_core::Error::Msg(format!("dummy alloc: {e}")))?;
                (dummy, Some(cpu_bytes))
            } else {
                (tensor_to_cuda_u8(&up_exps_tensor)?, None)
            };
            let (down_exps_raw, down_exps_cpu) = if experts_on_cpu {
                let cpu_bytes: Vec<u8> = down_exps_tensor.flatten_all()?.to_vec1()?;
                let dummy = dev
                    .alloc_zeros::<u8>(1)
                    .map_err(|e| candle_core::Error::Msg(format!("dummy alloc: {e}")))?;
                (dummy, Some(cpu_bytes))
            } else {
                (tensor_to_cuda_u8(&down_exps_tensor)?, None)
            };

            // Shared expert weights: Q5_K raw bytes, always on GPU
            let shexp_gate_raw = load_raw_weight(
                gguf,
                &format!("blk.{il}.ffn_gate_shexp.weight"),
                dev,
            )?;
            let shexp_up_raw = load_raw_weight(
                gguf,
                &format!("blk.{il}.ffn_up_shexp.weight"),
                dev,
            )?;
            let shexp_down_raw = load_raw_weight(
                gguf,
                &format!("blk.{il}.ffn_down_shexp.weight"),
                dev,
            )?;

            if experts_on_cpu {
                eprintln!("[WEIGHTS_RAW]   layer {:2}: experts on CPU (ncmoe)", il);
            }

            let moe = MoeWeightsRaw {
                gate_exps_raw,
                up_exps_raw,
                down_exps_raw,
                expert_bytes_gate,
                expert_bytes_up,
                expert_bytes_down,
                router_weight,
                shexp_gate_raw,
                shexp_up_raw,
                shexp_down_raw,
                shexp_gate_proj,
                experts_on_cpu,
                gate_exps_cpu,
                up_exps_cpu,
                down_exps_cpu,
            };

            layers.push(LayerWeightsRaw {
                attn_norm,
                ffn_norm,
                layer_type,
                gdn,
                attn,
                moe,
            });

            let ltype = if is_attn { "attn" } else { "gdn" };
            let cpu_tag = if experts_on_cpu { " CPU-experts" } else { "" };
            eprintln!(
                "[WEIGHTS_RAW] layer {:2} ({ltype}{cpu_tag}): {:.1}s",
                il,
                layer_start.elapsed().as_secs_f64(),
            );
        }

        let load_time = load_start.elapsed();
        eprintln!(
            "[WEIGHTS_RAW] All raw weights loaded in {:.1}s ({num_layers} layers)",
            load_time.as_secs_f64(),
        );

        Ok(Self {
            embed_table,
            lm_head_raw,
            final_norm,
            layers,
        })
    }
}

// ===========================================================================
// Weight loading helpers
// ===========================================================================

/// Load a tensor from GGUF as raw quantized bytes (CudaSlice<u8>) on GPU.
///
/// Works for any quantization type (IQ3_S, Q5_K, Q8_0, etc.).
fn load_raw_weight(gguf: &GgufFile, name: &str, dev: &CudaDevice) -> Result<CudaSlice<u8>> {
    let candle_dev = candle_core::Device::Cuda(dev.clone());
    let (tensor, _n_elem, _dims) = gguf.load_tensor_u8_any(name, &candle_dev)?;
    tensor_to_cuda_u8(&tensor)
}

/// Load a tensor from GGUF as F32 (CudaSlice<f32>) on GPU.
///
/// Used for norm weights, SSM parameters, and other small F32 tensors.
fn load_f32_weight(gguf: &GgufFile, name: &str, dev: &CudaDevice) -> Result<CudaSlice<f32>> {
    let candle_dev = candle_core::Device::Cuda(dev.clone());
    let tensor = gguf.load_tensor(name, &candle_dev)?;
    tensor_to_cuda_f32(&tensor)
}

/// Extract a CudaSlice<u8> from a Candle Tensor on GPU.
///
/// The Tensor must be contiguous and on a CUDA device.
fn tensor_to_cuda_u8(tensor: &candle_core::Tensor) -> Result<CudaSlice<u8>> {
    use candle_core::Storage;
    let tensor = tensor.contiguous()?;
    let (stor, lay) = tensor.storage_and_layout();
    let cuda_stor = match &*stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("tensor_to_cuda_u8: tensor not on CUDA"),
    };
    let slice = cuda_stor.as_cuda_slice::<u8>()?;
    let offset = lay.start_offset();
    let len = tensor.elem_count();
    // Clone the slice region to get an owned CudaSlice (detached from Candle's Storage).
    let view = slice.slice(offset..offset + len);
    let owned = view
        .stream()
        .clone()
        .clone_dtoh(&view)
        .map_err(|e| candle_core::Error::Msg(format!("dtoh for clone: {e}")))?;
    // Re-upload to get an owned CudaSlice not tied to the Tensor's Arc<Storage>.
    let dev = cuda_stor.device().clone();
    let mut dst = dev
        .alloc_zeros::<u8>(len)
        .map_err(|e| candle_core::Error::Msg(format!("alloc for clone: {e}")))?;
    dev.memcpy_htod(&owned, &mut dst)
        .map_err(|e| candle_core::Error::Msg(format!("htod for clone: {e}")))?;
    Ok(dst)
}

/// Extract a CudaSlice<f32> from a Candle Tensor on GPU.
///
/// The Tensor must be contiguous, F32, and on a CUDA device.
fn tensor_to_cuda_f32(tensor: &candle_core::Tensor) -> Result<CudaSlice<f32>> {
    use candle_core::Storage;
    let tensor = tensor.to_dtype(candle_core::DType::F32)?.contiguous()?;
    let (stor, lay) = tensor.storage_and_layout();
    let cuda_stor = match &*stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("tensor_to_cuda_f32: tensor not on CUDA"),
    };
    let slice = cuda_stor.as_cuda_slice::<f32>()?;
    let offset = lay.start_offset();
    let len = tensor.elem_count();
    let view = slice.slice(offset..offset + len);
    let host_data = view
        .stream()
        .clone()
        .clone_dtoh(&view)
        .map_err(|e| candle_core::Error::Msg(format!("dtoh for clone f32: {e}")))?;
    let dev = cuda_stor.device().clone();
    let mut dst = dev
        .alloc_zeros::<f32>(len)
        .map_err(|e| candle_core::Error::Msg(format!("alloc for clone f32: {e}")))?;
    dev.memcpy_htod(&host_data, &mut dst)
        .map_err(|e| candle_core::Error::Msg(format!("htod for clone f32: {e}")))?;
    Ok(dst)
}

// ===========================================================================
// forward_prefill — batch prefill for prompt tokens
// ===========================================================================

impl ComputeGraph {
    /// Batch prefill: process N prompt tokens through all layers at once.
    ///
    /// Much faster than calling `forward_token()` N times because:
    /// - Batch embedding upload (1 PCIe transfer instead of N)
    /// - Batch RMSNorm (1 kernel launch for N tokens per layer)
    /// - Batch Q8_1 quantization (1 kernel for N tokens per layer)
    /// - Batch KV cache fill for attention layers
    ///
    /// GDN recurrence and MoE FFN remain per-token (inherently sequential).
    ///
    /// Currently implemented as a per-token fallback loop that reuses
    /// `forward_token()`-style single-token ops but with batch embedding
    /// upload and batch final norm. Phases 2-6 will add true batch kernels.
    ///
    /// # Arguments
    /// - `tokens`: slice of prompt token IDs to process
    /// - `weights`: raw model weights from GGUF
    /// - `gdn_states`: per-GDN-layer recurrent state
    /// - `kv_cache`: KV cache for attention layers
    ///
    /// # Returns
    /// Logits as `Vec<f32>` on CPU, computed from the LAST token only
    /// (the only one that matters for next-token prediction).
    pub fn forward_prefill(
        &mut self,
        tokens: &[u32],
        weights: &ModelWeightsRaw,
        gdn_states: &mut [CudaSlice<f32>],
        kv_cache: &mut KvCacheRaw,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        assert!(n > 0, "forward_prefill: empty token list");

        // For single token, delegate to forward_token (no batch overhead).
        if n == 1 {
            return self.forward_token(tokens[0], weights, gdn_states, kv_cache);
        }

        let hs = self.hidden_size;
        let eps = 1e-6f32; // Qwen3.5 rms_norm_eps

        // --- Lazy-init PrefillBuffers if not already allocated ---
        if self.prefill_bufs.is_none() {
            // Default max_prefill: cap at 2048 tokens.
            // Callers should call init_prefill_buffers() explicitly for custom sizes.
            let max_pf = n.max(512).min(2048);
            eprintln!(
                "[PREFILL] Lazy-init PrefillBuffers (max_prefill={}, {} tokens requested)",
                max_pf, n,
            );
            let config = self.make_config_for_prefill(eps);
            self.prefill_bufs = Some(PrefillBuffers::new(&config, max_pf, &self.dev)?);
        }

        // Read max_prefill before taking the buffers.
        let max_pf = self.prefill_bufs.as_ref().unwrap().max_prefill;

        // If n exceeds max_prefill, process in chunks.
        if n > max_pf {
            eprintln!(
                "[PREFILL] Prompt ({} tokens) exceeds max_prefill ({}), chunking...",
                n, max_pf,
            );
            let mut offset = 0;
            while offset + max_pf < n {
                let _chunk_logits = self.forward_prefill(
                    &tokens[offset..offset + max_pf],
                    weights, gdn_states, kv_cache,
                )?;
                offset += max_pf;
            }
            return self.forward_prefill(
                &tokens[offset..],
                weights, gdn_states, kv_cache,
            );
        }

        // -----------------------------------------------------------------
        // 1. Batch embedding upload: lookup N rows from CPU, upload once
        // -----------------------------------------------------------------
        // Take PrefillBuffers out of self (take+put pattern to avoid borrow
        // conflicts with &mut self methods like forward_gdn/moe/attn).
        let mut prefill = self.prefill_bufs.take().unwrap();
        let padded_len = prefill.max_prefill * hs;

        let mut embed_cpu = vec![0.0f32; padded_len];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            let row_start = tok * hs;
            let row_end = row_start + hs;
            if row_end > weights.embed_table.len() {
                self.prefill_bufs = Some(prefill);
                candle_core::bail!(
                    "forward_prefill: token {tok} out of range (embed_table has {} rows)",
                    weights.embed_table.len() / hs,
                );
            }
            embed_cpu[i * hs..(i + 1) * hs]
                .copy_from_slice(&weights.embed_table[row_start..row_end]);
        }

        // Single PCIe transfer for all N token embeddings.
        self.dev
            .memcpy_htod(&embed_cpu, &mut prefill.hidden_batch)
            .map_err(|e| candle_core::Error::Msg(format!("prefill embed upload: {e}")))?;

        // Zero residual_batch for the first layer's fused_add_residual_rmsnorm.
        let zeros_residual = vec![0.0f32; padded_len];
        self.dev
            .memcpy_htod(&zeros_residual, &mut prefill.residual_batch)
            .map_err(|e| candle_core::Error::Msg(format!("prefill zero residual: {e}")))?;

        // -----------------------------------------------------------------
        // 2. Layer loop: process all N tokens per layer
        // -----------------------------------------------------------------
        // Take cached_ptrs out of self (take+put pattern, same as forward_token).
        let cached_ptrs = self.cached_ptrs.take();

        let start_pos = kv_cache.pos;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;

        for il in 0..self.num_layers {
            let lw = &weights.layers[il];

            // --- Batch RMSNorm (attention norm) for all N tokens ---
            crate::kernels::fused_ops::fused_add_residual_rmsnorm_batch(
                &prefill.hidden_batch,
                &mut prefill.residual_batch,
                &lw.attn_norm,
                &mut prefill.normed_batch,
                hs, n, eps, &self.dev,
            )?;

            // --- GDN or Attention: batch dispatch ---
            match lw.layer_type {
                LayerType::Gdn => {
                    let gdn_w = lw.gdn.as_ref().expect("GDN layer missing GDN weights");
                    super::layer_gdn::forward_gdn_prefill_cudarc(
                        il, gdn_w, &mut gdn_states[gdn_idx], self,
                        &mut prefill, n, cached_ptrs.as_ref(),
                    )?;
                }
                LayerType::Attention => {
                    let attn_w = lw.attn.as_ref().expect("Attn layer missing weights");
                    let rope_tables = self.rope_tables.take()
                        .expect("RoPE tables not initialized");
                    let result = super::layer_attn::forward_attn_prefill_cudarc(
                        attn_idx, attn_w, self, &mut prefill, kv_cache,
                        start_pos, n, &rope_tables, cached_ptrs.as_ref(),
                    );
                    self.rope_tables = Some(rope_tables);
                    result?;
                }
            }

            // --- Batch RMSNorm (FFN norm) for all N tokens ---
            crate::kernels::fused_ops::fused_add_residual_rmsnorm_batch(
                &prefill.hidden_batch,
                &mut prefill.residual_batch,
                &lw.ffn_norm,
                &mut prefill.normed_batch,
                hs, n, eps, &self.dev,
            )?;

            // --- Per-token MoE FFN (routing is token-dependent) ---
            for t in 0..n {
                let offset = t * hs;
                // Copy normed_batch[t] → self.normed
                {
                    let src = prefill.normed_batch.slice(offset..offset + hs);
                    self.dev.memcpy_dtod(&src, &mut self.normed)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("prefill normed→scratch t={t} il={il}: {e}")))?;
                }
                // Run MoE
                super::moe_ffn::forward_moe_cudarc_with_cache(
                    &lw.moe, self, cached_ptrs.as_ref(), il,
                )?;
                // Copy self.hidden → hidden_batch[t]
                {
                    let mut dst = prefill.hidden_batch.slice_mut(offset..offset + hs);
                    self.dev.memcpy_dtod(&self.hidden, &mut dst)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("prefill moe→batch t={t} il={il}: {e}")))?;
                }
            }

            // Advance sub-layer indices.
            match lw.layer_type {
                LayerType::Gdn => { gdn_idx += 1; }
                LayerType::Attention => { attn_idx += 1; }
            }
        }

        // Put cached_ptrs back (take+put pattern complete).
        self.cached_ptrs = cached_ptrs;

        // Set final KV cache position after all N tokens processed.
        kv_cache.pos = start_pos + n;

        // -----------------------------------------------------------------
        // 3. Final norm on the LAST token only
        // -----------------------------------------------------------------
        // Copy last token's hidden + residual from batch buffers → scratch.
        let last_offset = (n - 1) * hs;
        {
            let src_h = prefill.hidden_batch.slice(last_offset..last_offset + hs);
            self.dev.memcpy_dtod(&src_h, &mut self.hidden)
                .map_err(|e| candle_core::Error::Msg(format!("prefill final hidden dtod: {e}")))?;
        }
        {
            let src_r = prefill.residual_batch.slice(last_offset..last_offset + hs);
            self.dev.memcpy_dtod(&src_r, &mut self.residual)
                .map_err(|e| candle_core::Error::Msg(format!("prefill final residual dtod: {e}")))?;
        }

        // Put PrefillBuffers back (take+put pattern complete).
        self.prefill_bufs = Some(prefill);

        // Fused add + RMSNorm: accumulate last MoE output into residual.
        crate::kernels::fused_ops::fused_add_residual_rmsnorm(
            &self.hidden,
            &mut self.residual,
            &weights.final_norm,
            &mut self.normed,
            hs,
            eps,
            &self.dev,
        )?;

        // -----------------------------------------------------------------
        // 4. LM head on last token: logits = lm_head @ normed
        // -----------------------------------------------------------------
        crate::kernels::gemv_q8_0::gemv_q8_0_f32_slices(
            &weights.lm_head_raw,
            &self.normed,
            &mut self.logits,
            self.vocab_size,
            hs,
            &self.dev,
        )?;

        // -----------------------------------------------------------------
        // 5. Return logits on CPU for sampling
        // -----------------------------------------------------------------
        let mut logits_cpu = vec![0.0f32; self.vocab_size];
        self.dev
            .memcpy_dtoh(&self.logits, &mut logits_cpu)
            .map_err(|e| candle_core::Error::Msg(format!("prefill dtoh logits: {e}")))?;
        Ok(logits_cpu)
    }

    /// Build a minimal `Qwen35Config` from `ComputeGraph` dimensions.
    ///
    /// Used for lazy-init of `PrefillBuffers` when `init_prefill_buffers()`
    /// was not called explicitly. Only the dimension fields used by
    /// `PrefillBuffers::new()` are populated; unused fields are zero.
    fn make_config_for_prefill(&self, eps: f32) -> crate::config::Qwen35Config {
        crate::config::Qwen35Config {
            hidden_size: self.hidden_size,
            num_attention_heads: self.n_heads,
            num_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            num_main_layers: self.num_layers,
            num_total_layers: self.num_layers,
            intermediate_size: 0,
            vocab_size: self.vocab_size,
            rms_norm_eps: eps as f64,
            ssm_d_state: self.ssm_d_state,
            ssm_d_inner: self.ssm_d_inner,
            ssm_dt_rank: self.ssm_dt_rank,
            ssm_n_group: self.ssm_n_group,
            ssm_conv_kernel: self.ssm_conv_kernel,
            full_attn_interval: self.full_attn_interval,
            rope_theta: 0.0,
            rope_sections: [0; 4],
            nextn_predict_layers: 0,
            num_experts: self.n_experts,
            experts_per_token: self.top_k,
            expert_ffn_hidden: self.expert_ffn,
            shared_expert_ffn_hidden: self.shexp_ffn,
        }
    }
}

// ===========================================================================
// forward_token — main inference loop skeleton
// ===========================================================================

impl ComputeGraph {
    /// Run one token through the full model, returning logits on CPU.
    ///
    /// This is the main forward loop. It calls into the layer-specific modules
    /// (layer_gdn, layer_attn, moe_ffn) which are implemented by agents 2-4.
    ///
    /// # Arguments
    /// - `token_id`: input token to process
    /// - `weights`: raw model weights from GGUF
    /// - `gdn_states`: per-GDN-layer recurrent state [n_gdn_layers, n_heads, head_dim, head_dim]
    /// - `kv_cache`: KV cache for attention layers
    ///
    /// # Returns
    /// Logits as `Vec<f32>` on CPU, ready for sampling.
    pub fn forward_token(
        &mut self,
        token_id: u32,
        weights: &ModelWeightsRaw,
        gdn_states: &mut [CudaSlice<f32>],
        kv_cache: &mut KvCacheRaw,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size;
        let eps = 1e-6f32; // Qwen3.5 rms_norm_eps

        // -----------------------------------------------------------------
        // 1. Embed: lookup 1 row from CPU table, copy to GPU hidden buffer
        // -----------------------------------------------------------------
        let tok = token_id as usize;
        let row_start = tok * hidden_size;
        let row_end = row_start + hidden_size;
        if row_end > weights.embed_table.len() {
            candle_core::bail!(
                "token_id {token_id} out of range (embed_table has {} rows)",
                weights.embed_table.len() / hidden_size,
            );
        }
        let row = &weights.embed_table[row_start..row_end];
        self.dev
            .memcpy_htod(row, &mut self.hidden)
            .map_err(|e| candle_core::Error::Msg(format!("htod embed: {e}")))?;

        // Zero residual — each token starts fresh. Without this, the residual
        // from the previous token bleeds into the current one via
        // fused_add_residual_rmsnorm (residual += hidden), causing garbage
        // generation after a few tokens.
        // Uses cuMemsetD8Async (GPU-side, no CPU alloc or PCIe transfer).
        {
            use candle_core::cuda_backend::cudarc::driver::{DeviceSlice, DevicePtrMut};
            let stream = self.dev.cuda_stream();
            let (ptr, _sync) = self.residual.device_ptr_mut(&stream);
            let size = hidden_size * std::mem::size_of::<f32>();
            let err = unsafe {
                candle_core::cuda_backend::cudarc::driver::sys::cuMemsetD8Async(
                    ptr, 0, size, stream.cu_stream(),
                )
            };
            if err != candle_core::cuda_backend::cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                candle_core::bail!("cuMemsetD8Async residual: {:?}", err);
            }
        }

        // -----------------------------------------------------------------
        // 2. Layer loop
        // -----------------------------------------------------------------
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;

        // Take cached weight pointers out of self (take+put pattern to avoid
        // borrow conflicts: forward_gdn_cudarc_with_cache takes &mut self).
        let cached_ptrs = self.cached_ptrs.take();

        // Phase 3.1: Advance CUDA Graph token counter at the start of each token.
        if let Some(ref mut cache) = self.gdn_graph_cache {
            cache.advance_token();
        }

        for il in 0..self.num_layers {
            let lw = &weights.layers[il];

            match lw.layer_type {
                LayerType::Gdn => {
                    let gdn_w = lw.gdn.as_ref().expect("GDN layer missing GDN weights");

                    // Full-layer CUDA Graph: captures RMSNorm + GDN + RMSNorm + MoE.
                    // ncmoe layers (MoE on CPU) cannot be captured (memcpy_dtoh).
                    let mut graph_cache = self.gdn_graph_cache.take();
                    let moe_on_gpu = !lw.moe.experts_on_cpu;
                    let use_graph_replay = moe_on_gpu && graph_cache.as_ref()
                        .map_or(false, |c| c.is_replay_mode());
                    let use_graph_capture = moe_on_gpu && graph_cache.as_ref()
                        .map_or(false, |c| c.is_capture_token());

                    if use_graph_replay {
                        // Replay: 1 graph launch = entire layer
                        let cache = graph_cache.as_ref().unwrap();
                        cache.replay(gdn_idx)?;
                    } else if use_graph_capture {
                        // Capture: RMSNorm + GDN + RMSNorm + MoE
                        let cache = graph_cache.as_mut().unwrap();
                        let raw_stream = cache.raw_stream_ptr();
                        if let Some(ref mut ggml_bufs) = self.ggml_gpu_bufs {
                            unsafe { ggml_bufs.set_stream_override(raw_stream); }
                        }

                        cache.begin_capture()?;
                        self.gdn_graph_cache = graph_cache;

                        let fwd_result = (|| -> candle_core::Result<()> {
                            crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                                &self.hidden, &mut self.residual, &lw.attn_norm,
                                &mut self.normed, hidden_size, eps, &self.dev,
                            )?;
                            super::layer_gdn::forward_gdn_cudarc_with_cache(
                                il, gdn_w, &mut gdn_states[gdn_idx], self,
                                cached_ptrs.as_ref(),
                            )?;
                            crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                                &self.hidden, &mut self.residual, &lw.ffn_norm,
                                &mut self.normed, hidden_size, eps, &self.dev,
                            )?;
                            super::moe_ffn::forward_moe_cudarc_with_cache(
                                &lw.moe, self, cached_ptrs.as_ref(), il,
                            )?;
                            Ok(())
                        })();

                        graph_cache = self.gdn_graph_cache.take();
                        let cache = graph_cache.as_mut().unwrap();
                        let end_result = cache.end_capture_safe(gdn_idx);
                        if let Some(ref mut ggml_bufs) = self.ggml_gpu_bufs {
                            ggml_bufs.clear_stream_override();
                        }

                        if let Err(e) = fwd_result {
                            eprintln!("[CUDA_GRAPH] Full layer {} capture failed: {e}", gdn_idx);
                        } else if let Err(e) = end_result {
                            eprintln!("[CUDA_GRAPH] Full layer {} end_capture failed: {e}", gdn_idx);
                        } else {
                            cache.replay(gdn_idx)?;
                        }
                    } else {
                        // Warmup or ncmoe: run inline
                        self.gdn_graph_cache = graph_cache;
                        crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                            &self.hidden, &mut self.residual, &lw.attn_norm,
                            &mut self.normed, hidden_size, eps, &self.dev,
                        )?;
                        super::layer_gdn::forward_gdn_cudarc_with_cache(
                            il, gdn_w, &mut gdn_states[gdn_idx], self,
                            cached_ptrs.as_ref(),
                        )?;
                        crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                            &self.hidden, &mut self.residual, &lw.ffn_norm,
                            &mut self.normed, hidden_size, eps, &self.dev,
                        )?;
                        super::moe_ffn::forward_moe_cudarc_with_cache(
                            &lw.moe, self, cached_ptrs.as_ref(), il,
                        )?;
                        graph_cache = self.gdn_graph_cache.take();
                    }

                    self.gdn_graph_cache = graph_cache;
                    gdn_idx += 1;
                }
                LayerType::Attention => {
                    // Attention sub-layer: inline (variable KV cache length)
                    crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                        &self.hidden, &mut self.residual, &lw.attn_norm,
                        &mut self.normed, hidden_size, eps, &self.dev,
                    )?;
                    let attn_w = lw.attn.as_ref().expect("Attn layer missing weights");
                    let position = kv_cache.pos;
                    let rope_tables = self.rope_tables.take()
                        .expect("RoPE tables not initialized");
                    let result = self.forward_attn_cudarc(
                        attn_idx, attn_w, kv_cache, position, &rope_tables,
                    );
                    self.rope_tables = Some(rope_tables);
                    result?;

                    // Post-attention: ffn_norm RMSNorm + MoE — CUDA Graph capture
                    let mut graph_cache = self.gdn_graph_cache.take();
                    let moe_on_gpu = !lw.moe.experts_on_cpu;
                    let use_attn_moe_replay = moe_on_gpu && graph_cache.as_ref()
                        .map_or(false, |c| c.is_replay_mode());
                    let use_attn_moe_capture = moe_on_gpu && graph_cache.as_ref()
                        .map_or(false, |c| c.is_capture_token());

                    if use_attn_moe_replay {
                        // Replay: 1 graph launch = ffn_norm + MoE
                        let cache = graph_cache.as_ref().unwrap();
                        cache.replay_attn_moe(attn_idx)?;
                    } else if use_attn_moe_capture {
                        // Capture: ffn_norm RMSNorm + MoE
                        let cache = graph_cache.as_mut().unwrap();
                        let raw_stream = cache.raw_stream_ptr();
                        if let Some(ref mut ggml_bufs) = self.ggml_gpu_bufs {
                            unsafe { ggml_bufs.set_stream_override(raw_stream); }
                        }

                        cache.begin_capture()?;
                        self.gdn_graph_cache = graph_cache;

                        let fwd_result = (|| -> candle_core::Result<()> {
                            crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                                &self.hidden, &mut self.residual, &lw.ffn_norm,
                                &mut self.normed, hidden_size, eps, &self.dev,
                            )?;
                            super::moe_ffn::forward_moe_cudarc_with_cache(
                                &lw.moe, self, cached_ptrs.as_ref(), il,
                            )?;
                            Ok(())
                        })();

                        graph_cache = self.gdn_graph_cache.take();
                        let cache = graph_cache.as_mut().unwrap();
                        let end_result = cache.end_capture_attn_moe_safe(attn_idx);
                        if let Some(ref mut ggml_bufs) = self.ggml_gpu_bufs {
                            ggml_bufs.clear_stream_override();
                        }

                        if let Err(e) = fwd_result {
                            eprintln!("[CUDA_GRAPH] Attn MoE layer {} capture failed: {e}", attn_idx);
                        } else if let Err(e) = end_result {
                            eprintln!("[CUDA_GRAPH] Attn MoE layer {} end_capture failed: {e}", attn_idx);
                        } else {
                            cache.replay_attn_moe(attn_idx)?;
                        }
                    } else {
                        // Warmup or ncmoe: run inline
                        self.gdn_graph_cache = graph_cache;
                        crate::kernels::fused_ops::fused_add_residual_rmsnorm(
                            &self.hidden, &mut self.residual, &lw.ffn_norm,
                            &mut self.normed, hidden_size, eps, &self.dev,
                        )?;
                        super::moe_ffn::forward_moe_cudarc_with_cache(
                            &lw.moe, self, cached_ptrs.as_ref(), il,
                        )?;
                        graph_cache = self.gdn_graph_cache.take();
                    }

                    self.gdn_graph_cache = graph_cache;
                    attn_idx += 1;
                }
            }
        }

        // Put cached weight pointers back (take+put pattern complete)
        self.cached_ptrs = cached_ptrs;

        // Increment KV cache position after all layers processed this token
        kv_cache.pos += 1;

        // -----------------------------------------------------------------
        // 3. Final norm: fused add + RMSNorm (accumulate last MoE output)
        //    residual[i] += hidden[i]; normed = rmsnorm(residual, final_norm)
        // -----------------------------------------------------------------
        crate::kernels::fused_ops::fused_add_residual_rmsnorm(
            &self.hidden,
            &mut self.residual,
            &weights.final_norm,
            &mut self.normed,
            hidden_size,
            eps,
            &self.dev,
        )?;

        // -----------------------------------------------------------------
        // 4. LM head: logits = lm_head @ normed
        // -----------------------------------------------------------------
        // TODO: need gemv_q8_0 or equivalent kernel for lm_head.
        // LM head GEMV: normed[hidden_size] × lm_head[vocab_size, hidden_size] → logits[vocab_size]
        crate::kernels::gemv_q8_0::gemv_q8_0_f32_slices(
            &weights.lm_head_raw,
            &self.normed,
            &mut self.logits,
            self.vocab_size,
            hidden_size,
            &self.dev,
        )?;

        // -----------------------------------------------------------------
        // 5. Return logits on CPU for sampling
        // -----------------------------------------------------------------
        // Copy logits GPU → CPU for sampling
        let mut logits_cpu = vec![0.0f32; self.vocab_size];
        self.dev.memcpy_dtoh(&self.logits, &mut logits_cpu)
            .map_err(|e| candle_core::Error::Msg(format!("dtoh logits: {e}")))?;
        Ok(logits_cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GGUF_PATH: &str = "{HOME}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf";

    /// Compare embedding dequantization AND single-layer output: cudarc vs Candle QMatMul.
    #[test]
    fn test_embed_table_vs_qmatmul() {
        if !std::path::Path::new(GGUF_PATH).exists() { return; }

        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();

        // Path A: cudarc embed_table (load_tensor → flatten → Vec<f32>)
        let embed_tensor = gguf.load_tensor("token_embd.weight", &candle_core::Device::Cpu).unwrap();
        let embed_flat: Vec<f32> = embed_tensor.flatten_all().unwrap().to_vec1().unwrap();
        let hidden_size = 2048usize;
        let tok = 248045usize;
        let cudarc_emb = &embed_flat[tok * hidden_size..(tok + 1) * hidden_size];

        // Path B: Candle QMatMul dequantize
        let qmm = gguf.load_qmatmul("token_embd.weight", &candle_core::Device::Cpu).unwrap();
        let candle_emb: Vec<f32> = match &qmm {
            candle_core::quantized::QMatMul::QTensor(qt) => {
                let deq = qt.dequantize(&candle_core::Device::Cpu).unwrap();
                deq.get(tok).unwrap().to_vec1::<f32>().unwrap()
            }
            candle_core::quantized::QMatMul::Tensor(t) => {
                t.get(tok).unwrap().to_vec1::<f32>().unwrap()
            }
            candle_core::quantized::QMatMul::TensorF16(t) => {
                t.to_dtype(candle_core::DType::F32).unwrap()
                    .get(tok).unwrap().to_vec1::<f32>().unwrap()
            }
        };

        eprintln!("cudarc[0..5]: {:?}", &cudarc_emb[..5]);
        eprintln!("candle[0..5]: {:?}", &candle_emb[..5]);

        let max_diff = cudarc_emb.iter().zip(candle_emb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("max diff: {max_diff}");

        if max_diff > 1e-6 {
            // Find first divergence
            for (i, (a, b)) in cudarc_emb.iter().zip(candle_emb.iter()).enumerate() {
                if (a - b).abs() > 1e-6 {
                    eprintln!("FIRST DIFF at index {i}: cudarc={a}, candle={b}, diff={}", (a-b).abs());
                    break;
                }
            }
            panic!("EMBEDDING MISMATCH! cudarc and Candle dequantize differently. max_diff={max_diff}");
        }
        eprintln!("PASS: embeddings match (max_diff={max_diff})");
    }

    #[test]
    fn test_raw_byte_sanity() {
        if !std::path::Path::new(GGUF_PATH).exists() { return; }

        let dev = candle_core::Device::cuda_if_available(0).unwrap();
        let _cuda_dev = match &dev {
            candle_core::Device::Cuda(_d) => _d,
            _ => { eprintln!("No CUDA"); return; }
        };

        // Load via existing GGUF loader
        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();
        let (raw_tensor, _, _) = gguf.load_tensor_u8(
            "blk.0.ffn_gate_exps.weight", &dev).unwrap();

        // Extract first 4 bytes from the Candle tensor
        let first_bytes: Vec<u8> = raw_tensor.narrow(0, 0, 4).unwrap().to_vec1().unwrap();
        eprintln!("Candle path first 4 bytes: {:?}", first_bytes);

        // Expected from GGUF reader: [201, 6, 178, 230]
        assert_eq!(first_bytes, vec![201, 6, 178, 230],
            "IQ3_S bytes mismatch! Raw loader reads different bytes than GGUF.");
        eprintln!("SANITY CHECK PASSED: IQ3_S bytes match between Candle and GGUF reader");
    }

    /// Full forward pass test: all 40 layers wired (GDN + Attention + MoE FFN).
    ///
    /// Loads the real GGUF model, runs 1 token, checks logits shape and values.
    /// Skipped if the GGUF file is missing or no CUDA is available.
    #[test]
    fn test_forward_token_full() {
        if !std::path::Path::new(GGUF_PATH).exists() {
            eprintln!("GGUF not found at {GGUF_PATH}, skipping test");
            return;
        }
        let dev = candle_core::Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &dev {
            candle_core::Device::Cuda(d) => d.clone(),
            _ => { eprintln!("No CUDA, skipping test"); return; }
        };

        // Load config + weights
        let config = crate::config::Qwen35Config::qwen35_35b_a3b();
        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();
        eprintln!("[TEST] Loading raw weights...");
        let weights = ModelWeightsRaw::from_gguf(&gguf, &config, &cuda_dev, 4).unwrap();
        drop(gguf);

        // Create compute graph + init rope tables + cached weight pointers
        let mut graph = ComputeGraph::new(&config, &cuda_dev).unwrap();
        graph.init_rope_tables(&config).unwrap();
        graph.init_cached_ptrs(&weights);

        // GDN recurrent states: one per GDN layer, each [dt_rank * d_state * d_state] zeros
        let num_gdn = config.num_gdn_layers();
        let state_size = config.ssm_dt_rank * config.ssm_d_state * config.ssm_d_state;
        let mut gdn_states: Vec<_> = (0..num_gdn)
            .map(|_| cuda_dev.alloc_zeros::<f32>(state_size).unwrap())
            .collect();

        // KV cache
        let num_attn = config.num_attn_layers();
        let mut kv_cache = KvCacheRaw::new(
            num_attn,
            config.num_kv_heads,
            config.head_dim,
            512, // max_seq_len for test
            &cuda_dev,
        ).unwrap();

        // Run forward pass with token 248045 (a common Qwen3.5 token)
        eprintln!("[TEST] Running forward_token(248045)...");
        let start = std::time::Instant::now();
        let logits = graph.forward_token(
            248045, &weights, &mut gdn_states, &mut kv_cache,
        ).unwrap();
        let elapsed = start.elapsed();
        eprintln!("[TEST] forward_token completed in {:.1}ms", elapsed.as_secs_f64() * 1000.0);

        // Verify logits
        assert_eq!(logits.len(), config.vocab_size,
            "logits length mismatch: expected {}, got {}", config.vocab_size, logits.len());
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let argmax = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!(max_logit > -1000.0, "logits are all very negative: max={max_logit}");
        assert!(logits.iter().any(|&x| x != logits[0]),
            "all logits are identical (model output is trivial)");
        eprintln!("[TEST] Max logit: {max_logit:.4}, argmax: {argmax}");
        eprintln!("[TEST] KV cache pos after 1 token: {}", kv_cache.pos);
        assert_eq!(kv_cache.pos, 1, "KV cache should be at position 1 after 1 token");
    }

    /// Test: prefill real prompt then generate — compare against ik_llama reference.
    ///
    /// Prefills "What is 2+2? Answer in one word." with proper chat template.
    /// ik_llama reference: first generated token = 26108 ("Four").
    #[test]
    #[ignore]
    fn test_real_prompt_generation() {
        if !std::path::Path::new(GGUF_PATH).exists() {
            eprintln!("GGUF not found, skipping"); return;
        }
        let dev = candle_core::Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &dev {
            candle_core::Device::Cuda(d) => d.clone(),
            _ => { eprintln!("No CUDA"); return; }
        };

        let config = crate::config::Qwen35Config::qwen35_35b_a3b();
        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();
        let weights = ModelWeightsRaw::from_gguf(&gguf, &config, &cuda_dev, 4).unwrap();
        drop(gguf);

        let mut graph = ComputeGraph::new(&config, &cuda_dev).unwrap();
        graph.init_rope_tables(&config).unwrap();
        graph.init_cached_ptrs(&weights);

        let num_gdn = config.num_gdn_layers();
        let state_size = config.ssm_dt_rank * config.ssm_d_state * config.ssm_d_state;
        let mut gdn_states: Vec<_> = (0..num_gdn)
            .map(|_| cuda_dev.alloc_zeros::<f32>(state_size).unwrap())
            .collect();
        let num_attn = config.num_attn_layers();
        let mut kv_cache = KvCacheRaw::new(
            num_attn, config.num_kv_heads, config.head_dim, 512, &cuda_dev,
        ).unwrap();

        // Chat template for "What is 2+2? Answer in one word."
        // <|im_start|>user\nWhat is 2+2? Answer in one word.<|im_end|>\n<|im_start|>assistant\n
        let prompt_tokens: Vec<u32> = vec![
            248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
            21134, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198,
        ];

        // Prefill each token and log argmax after each one to find divergence point
        eprintln!("[PROMPT] Prefilling {} tokens, logging argmax after each...", prompt_tokens.len());
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let logits = graph.forward_token(tok, &weights, &mut gdn_states, &mut kv_cache).unwrap();
            let argmax = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[PREFILL t={i:2}] tok={tok:6} -> argmax={argmax:6} logit={max_logit:.4}");
        }
        eprintln!("[PROMPT] ik_llama reference after full prefill: argmax=26108 (\"Four\")");
    }

    /// Diagnostic: compare logit distribution quality between chimere and ik_llama.
    ///
    /// Prefills the mutex prompt and prints softmax top-10 distribution.
    /// Compare with ik_llama: Here=56.55%, This=42.99% (sharp, confident).
    /// If chimere shows flat distribution (top-1 < 10%), attenuation is the issue.
    #[test]
    #[ignore]
    fn test_logit_distribution_quality() {
        if !std::path::Path::new(GGUF_PATH).exists() {
            eprintln!("GGUF not found, skipping"); return;
        }
        let dev = candle_core::Device::cuda_if_available(0).unwrap();
        let cuda_dev = match &dev {
            candle_core::Device::Cuda(d) => d.clone(),
            _ => { eprintln!("No CUDA"); return; }
        };

        let config = crate::config::Qwen35Config::qwen35_35b_a3b();
        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();
        let weights = ModelWeightsRaw::from_gguf(&gguf, &config, &cuda_dev, 4).unwrap();
        drop(gguf);

        let mut graph = ComputeGraph::new(&config, &cuda_dev).unwrap();
        graph.init_rope_tables(&config).unwrap();
        graph.init_cached_ptrs(&weights);

        let num_gdn = config.num_gdn_layers();
        let state_size = config.ssm_dt_rank * config.ssm_d_state * config.ssm_d_state;
        let mut gdn_states: Vec<_> = (0..num_gdn)
            .map(|_| cuda_dev.alloc_zeros::<f32>(state_size).unwrap())
            .collect();
        let num_attn = config.num_attn_layers();
        let mut kv_cache = KvCacheRaw::new(
            num_attn, config.num_kv_heads, config.head_dim, 512, &cuda_dev,
        ).unwrap();

        // Mutex prompt: <|im_start|>user\nExplain the difference between a mutex and a semaphore<|im_end|>\n<|im_start|>assistant\n<think>\n
        // Tokenize manually or use known IDs
        // For now, use the "2+2" prompt since we have the tokens, then also test with a longer one
        let prompt_tokens: Vec<u32> = vec![
            248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
            21134, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198,
        ];

        eprintln!("\n=== CHIMERE LOGIT DISTRIBUTION DIAGNOSTIC ===\n");
        eprintln!("Prompt: 'What is 2+2? Answer in one word.' ({} tokens)\n", prompt_tokens.len());

        // Prefill all tokens
        let mut last_logits = Vec::new();
        for &tok in &prompt_tokens {
            last_logits = graph.forward_token(tok, &weights, &mut gdn_states, &mut kv_cache).unwrap();
        }

        // Softmax the logits
        let max_logit = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = last_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        // Top-10 by probability
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("Top-10 tokens after prefill (softmax distribution):");
        eprintln!("{:>8} {:>10} {:>10} {:>10}", "rank", "token_id", "prob%", "logit");
        for (rank, &(tok_id, prob)) in indexed.iter().take(10).enumerate() {
            eprintln!("{:>8} {:>10} {:>9.2}% {:>10.4}", rank + 1, tok_id, prob * 100.0, last_logits[tok_id]);
        }

        let top1_prob = indexed[0].1;
        let top2_prob = indexed[1].1;
        let ratio = top1_prob / (top2_prob + 1e-10);
        eprintln!("\nTop-1/Top-2 ratio: {:.2}x", ratio);
        eprintln!("Top-1 + Top-2 mass: {:.2}%", (top1_prob + top2_prob) * 100.0);
        eprintln!("\nik_llama reference: Here=56.55%, This=42.99% (ratio=1.32x, mass=99.54%)");
        eprintln!("NOTE: This test uses '2+2' prompt. For mutex prompt, tokenize separately.\n");

        // The argmax should be the same token
        let chimere_argmax = indexed[0].0;
        eprintln!("Chimere argmax: {} (ik_llama ref: 26108=\"Four\")", chimere_argmax);
        assert_eq!(chimere_argmax, 26108, "Argmax diverges from ik_llama reference!");
    }

    /// Benchmark: generate 100 tokens and measure tok/s.
    ///
    /// This is NOT run by default (too slow for CI). Run manually with:
    /// ```sh
    /// CUDA_COMPUTE_CAP=89 cargo test --release --features server bench_cudarc_forward -- --nocapture --ignored
    /// ```
    #[test]
    #[ignore]
    fn bench_cudarc_forward() {
        if !std::path::Path::new(GGUF_PATH).exists() {
            eprintln!("GGUF not found at {GGUF_PATH}, skipping bench");
            return;
        }
        // Use new_with_stream() to get a non-blocking dedicated stream.
        // The legacy default stream (from cuda_if_available) does NOT support
        // CUDA Graph capture (cuStreamBeginCapture fails with UNSUPPORTED).
        let cuda_dev = candle_core::cuda_backend::CudaDevice::new_with_stream(0).unwrap();
        // Disable event tracking before any allocation (single stream, no sync needed).
        unsafe { cuda_dev.disable_event_tracking(); }
        let dev = candle_core::Device::Cuda(cuda_dev.clone());

        let config = crate::config::Qwen35Config::qwen35_35b_a3b();
        let gguf = crate::gguf_loader::GgufFile::open(GGUF_PATH).unwrap();
        eprintln!("[BENCH] Loading raw weights...");
        let weights = ModelWeightsRaw::from_gguf(&gguf, &config, &cuda_dev, 4).unwrap();
        drop(gguf);

        let mut graph = ComputeGraph::new(&config, &cuda_dev).unwrap();
        graph.init_rope_tables(&config).unwrap();
        graph.init_gdn_graph_cache().unwrap(); // Phase 3.1: enable CUDA Graphs if CHIMERE_CUDA_GRAPH=1
        graph.init_cached_ptrs(&weights);

        let num_gdn = config.num_gdn_layers();
        let state_size = config.ssm_dt_rank * config.ssm_d_state * config.ssm_d_state;
        let mut gdn_states: Vec<_> = (0..num_gdn)
            .map(|_| cuda_dev.alloc_zeros::<f32>(state_size).unwrap())
            .collect();

        let num_attn = config.num_attn_layers();
        let mut kv_cache = KvCacheRaw::new(
            num_attn,
            config.num_kv_heads,
            config.head_dim,
            1024, // enough for 100 tokens
            &cuda_dev,
        ).unwrap();

        // Warmup: 2 tokens (token 0 = ggml init, token 1 = CUDA Graph capture if enabled)
        let mut current_token = 248045u32;
        for _ in 0..2 {
            let logits = graph.forward_token(current_token, &weights, &mut gdn_states, &mut kv_cache).unwrap();
            current_token = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        }

        // Benchmark: 100 tokens
        let n_tokens = 100;
        let start = std::time::Instant::now();
        for i in 0..n_tokens {
            let logits = graph.forward_token(
                current_token, &weights, &mut gdn_states, &mut kv_cache,
            ).unwrap();
            current_token = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            if i % 10 == 0 {
                eprintln!("[BENCH] token {i}: argmax={current_token}");
            }
        }
        let elapsed = start.elapsed();
        let tok_per_sec = n_tokens as f64 / elapsed.as_secs_f64();
        eprintln!(
            "[BENCH] cudarc forward: {:.1} tok/s ({:.1} ms/token) over {n_tokens} tokens",
            tok_per_sec,
            elapsed.as_millis() as f64 / n_tokens as f64,
        );
    }
}
