//! # Qwen3.5 Model — Chimere Engine
//!
//! Full Qwen3.5 model implementation loading directly from GGUF.
//!
//! ## Architecture
//!
//! Qwen3.5 uses a hybrid architecture:
//! - **48 GDN layers** (Gated DeltaNet / recurrent SSM) — layers where `(i+1) % 4 != 0`
//! - **16 attention layers** — every 4th layer (indices 3, 7, 11, ..., 63)
//! - **1 MTP head** (multi-token prediction) at layer 64
//!
//! ## Quantized GPU inference (QMatMul + raw IQ3_S bytes)
//!
//! Dense weight tensors (Q8_0, Q4K, Q6K, Q5K, F32, F16, BF16) are loaded as `QMatMul`
//! — raw quantized bytes uploaded directly to GPU, no CPU dequantization.
//!
//! MoE routed expert tensors are IQ3_S (type 21), which Candle does not support natively.
//! Rather than CPU-dequantizing all 256 experts upfront (~60 GB of F32), we store the
//! raw IQ3_S bytes on GPU as flat `U8` Candle Tensors.  At forward time, a CUDA kernel
//! (`dequant_iq3s_gpu`) dequantizes **only the 8 active experts** per token on demand.
//!
//! Only the tiny F32 norm tensors (~20 KB each) are kept as regular Tensors.
//! All weight matrices are `QMatMul` which does on-the-fly GPU dequant during matmul.
//!
//! ## Forward Pass
//!
//! ```text
//! token → embed_table.index_select →
//!   for each layer: norm → QMatMul.forward → compute → next →
//!   output_norm → lm_head QMatMul.forward → logits
//! ```
//!
//! The GDN layers implement recurrent state updates using SSM parameters
//! (ssm_a, ssm_dt_bias, ssm_alpha, ssm_beta) to accumulate information across tokens.

// Submodules: extracted from this file (pure code movement, zero behavioral changes).
pub(crate) mod compute_graph;
pub(crate) mod cuda_graph;
mod layer_attn;
mod layer_gdn;
mod lm_head;
pub(crate) mod moe_cudarc;
mod moe_ffn;

use std::cell::RefCell;

use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Result, Tensor, D};

use crate::activations::{l2_norm, rms_norm, sigmoid, silu_activation, softplus};
use crate::config::Qwen35Config;
use crate::debug_utils::{
    act_f16_enabled, debug_dump, debug_enabled,
};
use crate::gguf_loader::GgufFile;
use crate::rope::MRoPE;
use crate::state::GdnRecurrentState;

// ---------------------------------------------------------------------------
// Weight structs (kept for synthetic tests only)
// ---------------------------------------------------------------------------

/// Weights for a GDN (recurrent SSM) layer.
pub struct GdnLayerWeights {
    pub attn_norm: Tensor,          // [hidden_size]
    pub post_attention_norm: Tensor, // [hidden_size]
    pub attn_qkv: Tensor,           // combined QKV for GDN
    pub attn_gate: Tensor,           // gate (z) for GDN
    pub ssm_a: Tensor,
    pub ssm_conv1d: Tensor,
    pub ssm_dt_bias: Tensor,
    pub ssm_beta: Tensor,
    pub ssm_alpha: Tensor,
    pub ssm_norm: Tensor,
    pub ssm_out: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
}

/// Weights for a full attention layer.
pub struct AttnLayerWeights {
    pub attn_norm: Tensor,          // [hidden_size]
    pub post_attention_norm: Tensor, // [hidden_size]
    pub wq: Tensor,                  // [n_head * head_dim * 2, hidden] (Q+gate fused)
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,                  // attn_output
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
}

/// MTP (multi-token prediction) head weights.
pub struct MtpHead {
    pub eh_proj: Tensor,             // [2*hidden, hidden]
    pub enorm: Tensor,
    pub hnorm: Tensor,
    pub shared_head_norm: Tensor,
}

/// A single layer — either GDN or full attention.
pub enum Qwen35Layer {
    Gdn(GdnLayerWeights),
    Attention(AttnLayerWeights),
}

// ---------------------------------------------------------------------------
// Quantized weight structs (QMatMul — stays quantized on GPU)
// ---------------------------------------------------------------------------

/// Preloaded quantized weights for a GDN (recurrent SSM) layer.
///
/// Norms are tiny F32 tensors. Weight matrices are `QMatMul` which can be
/// quantized on GPU (for Q8_0, Q4K, etc.) or F32 (for IQ3_S fallback).
pub(crate) struct GdnLayerQ {
    attn_norm: Tensor,          // F32, tiny [hidden_size]
    post_norm: Tensor,          // F32, tiny [hidden_size]
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    // GDN SSM tensors:
    attn_qkv: QMatMul,         // [key_dim*2 + value_dim, hidden_size] = [10240, 5120]
    attn_gate: QMatMul,        // [value_dim, hidden_size] = [6144, 5120]
    ssm_conv1d: Tensor,        // F32, [conv_channels, kernel_size] = [10240, 4]
    ssm_norm: Tensor,          // F32, tiny [ssm_d_state] = [128]
    ssm_out: QMatMul,          // [hidden_size, value_dim] = [5120, 6144]
    // SSM recurrence tensors:
    ssm_a: Tensor,             // F32, tiny [ssm_dt_rank] = [48]
    ssm_dt_bias: Tensor,       // F32, tiny [ssm_dt_rank] = [48]
    ssm_alpha: QMatMul,        // [ssm_dt_rank, hidden_size] = [48, 5120]
    ssm_beta: QMatMul,         // [ssm_dt_rank, hidden_size] = [48, 5120]
}

/// Preloaded quantized weights for a full attention layer.
pub(crate) struct AttnLayerQ {
    attn_norm: Tensor,          // F32, tiny
    post_norm: Tensor,          // F32, tiny
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: Tensor,             // F32, tiny
    k_norm: Tensor,             // F32, tiny
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
}

/// Preloaded quantized MTP head weights.
struct MtpHeadQ {
    eh_proj: QMatMul,
}

// ---------------------------------------------------------------------------
// MoE (Mixture-of-Experts) weight structs — for Qwen3.5-35B-A3B
// ---------------------------------------------------------------------------

/// MoE FFN weights for one layer (Qwen3.5-35B-A3B).
///
/// The router (`gate_inp`) is a small F32 tensor that scores each expert.
///
/// Routed expert tensors (IQ3_S, type 21) are stored as **raw encoded bytes** on GPU
/// as flat `U8` Candle Tensors — no dequantization at load time.  This avoids the ~60 GB
/// F32 materialization that would result from dequantizing all 256 experts upfront.
/// At forward time, a CUDA kernel (`dequant_iq3s_gpu`) dequantizes only the 8 active experts.
///
/// The shared expert runs unconditionally alongside the routed experts.
/// Shared expert weights are Q5_K (Candle-native), kept as `QMatMul`.
///
/// Fields are unused until the MoE forward pass is fully connected.
#[allow(dead_code)]
pub(crate) struct MoeFFN {
    /// Router: linear scores for each expert. Shape: [hidden, num_experts] (F32).
    pub(crate) gate_inp: Tensor,
    /// Router pre-transposed: [num_experts, hidden] for raw GEMV. Created at load time.
    pub(crate) gate_inp_t: Tensor,
    /// Shared expert gate bias: [hidden] (F32).
    pub(crate) gate_inp_shexp: Tensor,

    // ------------------------------------------------------------------
    // Routed expert weights — raw IQ3_S bytes on GPU (U8 tensor, flat).
    // Shapes follow GGUF (innermost-first) reversed to Candle row-major:
    //   gate_exps / up_exps : [hidden_size, expert_ffn, num_experts]
    //   down_exps            : [expert_ffn, hidden_size, num_experts]
    // ------------------------------------------------------------------
    /// Raw IQ3_S encoded bytes for gate projections, stored as flat U8 on GPU.
    pub(crate) gate_exps_raw: Tensor,
    /// Raw IQ3_S encoded bytes for up projections, stored as flat U8 on GPU.
    pub(crate) up_exps_raw: Tensor,
    /// Raw IQ3_S encoded bytes for down projections, stored as flat U8 on GPU.
    pub(crate) down_exps_raw: Tensor,

    /// Total logical elements across all experts: hidden * expert_ffn * num_experts
    /// (or expert_ffn * hidden * num_experts for down).
    pub(crate) gate_exps_elements: usize,
    pub(crate) up_exps_elements:   usize,
    pub(crate) down_exps_elements: usize,

    /// Shape (rows, cols, num_experts) in Candle row-major order.
    pub(crate) gate_exps_shape: (usize, usize, usize),
    pub(crate) up_exps_shape:   (usize, usize, usize),
    pub(crate) down_exps_shape:  (usize, usize, usize),

    // ------------------------------------------------------------------
    // Shared expert weights — Q5_K, Candle-native QMatMul.
    // ------------------------------------------------------------------
    /// Shared expert weights (always active, quantized on GPU as QMatMul).
    pub(crate) gate_shexp: QMatMul,
    pub(crate) up_shexp: QMatMul,
    pub(crate) down_shexp: QMatMul,
    /// Raw Q5_K bytes for ggml MMVQ kernel (shared experts).
    pub(crate) gate_shexp_raw: Option<Tensor>,
    pub(crate) up_shexp_raw: Option<Tensor>,
    pub(crate) down_shexp_raw: Option<Tensor>,

    /// Whether the routed expert weights are on CPU (ncmoe offloading).
    /// When true, the raw IQ3_S bytes (gate_exps_raw, up_exps_raw, down_exps_raw) are on CPU.
    /// The forward path uses CPU dequantization + F32 matmul instead of CUDA kernels.
    /// The shared expert, router, and all non-expert weights remain on GPU.
    pub(crate) experts_on_cpu: bool,
}

/// Preloaded quantized weights for a GDN layer with MoE FFN (35B-A3B).
///
/// Identical to `GdnLayerQ` but replaces the three dense FFN `QMatMul`s with
/// a `MoeFFN` struct that holds all expert and shared-expert weights.
///
/// Fields are unused until the MoE forward pass is implemented.
#[allow(dead_code)]
pub(crate) struct GdnLayerMoE {
    // Norms (F32, tiny)
    pub(crate) attn_norm: Tensor,
    pub(crate) post_norm: Tensor,
    // GDN SSM tensors — QMatMul (Candle native, already well-optimized)
    pub(crate) attn_qkv: QMatMul,
    pub(crate) attn_gate: QMatMul,
    pub(crate) ssm_conv1d: Tensor,
    pub(crate) ssm_norm: Tensor,
    pub(crate) ssm_out: QMatMul,
    pub(crate) ssm_a: Tensor,
    pub(crate) ssm_dt_bias: Tensor,
    pub(crate) ssm_alpha: QMatMul,
    pub(crate) ssm_beta: QMatMul,
    // Raw Q5_K bytes for ALL QMatMul projections — used by the raw forward kernel.
    // Some(_) when the weights are Q5_K; None for other quant types (fallback to QMatMul).
    pub(crate) attn_qkv_raw: Option<Tensor>,   // flat U8, [8192, 2048] logical
    pub(crate) attn_gate_raw: Option<Tensor>,  // flat U8, [4096, 2048] logical
    pub(crate) ssm_out_raw: Option<Tensor>,    // flat U8, [2048, 4096] logical
    pub(crate) ssm_beta_raw: Option<Tensor>,   // flat U8, [32, 2048] logical
    pub(crate) ssm_alpha_raw: Option<Tensor>,  // flat U8, [32, 2048] logical
    // MoE FFN instead of dense gate/up/down
    pub(crate) moe: MoeFFN,
}

/// Preloaded quantized weights for a full attention layer with MoE FFN (35B-A3B).
///
/// Identical to `AttnLayerQ` but replaces the three dense FFN `QMatMul`s with
/// a `MoeFFN` struct.
///
/// Fields are unused until the MoE forward pass is implemented.
#[allow(dead_code)]
pub(crate) struct AttnLayerMoE {
    // Norms (F32, tiny)
    pub(crate) attn_norm: Tensor,
    pub(crate) post_norm: Tensor,
    // Attention projections (QMatMul)
    pub(crate) wq: QMatMul,
    pub(crate) wk: QMatMul,
    pub(crate) wv: QMatMul,
    pub(crate) wo: QMatMul,
    pub(crate) q_norm: Tensor,
    pub(crate) k_norm: Tensor,
    // Raw Q5_K bytes for attention projections — used by the raw forward kernel.
    // Some(_) when the weights are Q5_K; None for other quant types (fallback to QMatMul).
    pub(crate) wq_raw: Option<Tensor>,  // flat U8, [16384, 2048] logical
    pub(crate) wk_raw: Option<Tensor>,  // flat U8, [512, 2048] logical
    pub(crate) wv_raw: Option<Tensor>,  // flat U8, [512, 2048] logical
    pub(crate) wo_raw: Option<Tensor>,  // flat U8, [2048, 8192] logical
    // MoE FFN instead of dense gate/up/down
    pub(crate) moe: MoeFFN,
}

/// A single preloaded quantized layer — dense or MoE variant.
pub(crate) enum Qwen35LayerQ {
    /// Dense GDN layer (27B or MoE model layers with dense FFN).
    Gdn(GdnLayerQ),
    /// Dense attention layer.
    Attention(AttnLayerQ),
    /// GDN layer with MoE FFN (35B-A3B).
    GdnMoE(GdnLayerMoE),
    /// Attention layer with MoE FFN (35B-A3B).
    AttentionMoE(AttnLayerMoE),
}

// ---------------------------------------------------------------------------
// Qwen35Model — preloaded quantized weights (QMatMul)
// ---------------------------------------------------------------------------

/// Full Qwen3.5 model with preloaded quantized weights.
///
/// All weight tensors are loaded at startup as `QMatMul`. For types Candle
/// supports natively (Q8_0, Q4K, Q6K, F32, F16, BF16), the raw quantized
/// bytes are uploaded directly to GPU. For unsupported types (IQ3_S), we
/// CPU-dequant to F32 and store on GPU. Either way, the forward pass is pure
/// GPU computation with no PCIe transfers.
pub struct Qwen35Model {
    /// mmap'd GGUF — kept alive so the mmap stays valid during loading,
    /// but not used during forward. Set to None after loading is complete
    /// to free the memory mapping.
    #[allow(dead_code)]
    gguf: Option<GgufFile>,
    pub config: Qwen35Config,
    pub mrope: MRoPE,
    /// Compute device (CPU or Cuda)
    pub device: Device,

    // --- Preloaded quantized layers ---
    /// Embedding table on device (F32 dequantized, needed for index_select)
    pub(crate) embed_tokens: Option<Tensor>,
    /// All main layers, preloaded as QMatMul
    pub(crate) q_layers: Option<Vec<Qwen35LayerQ>>,
    /// LM head projection
    pub(crate) lm_head: Option<QMatMul>,
    /// LM head raw Q5_K bytes for ggml MMVQ kernel
    pub(crate) lm_head_raw: Option<Tensor>,
    /// LM head raw Q8_0 bytes on CPU for ggml FFI validation (CHIMERE_GGML_LM_HEAD=1).
    /// Loaded at init only when the toggle is set and output.weight is Q8_0.
    pub(crate) lm_head_q8_0_cpu: Option<Vec<u8>>,
    /// MTP head (if present)
    mtp_head: Option<MtpHeadQ>,

    // Only F32 norms (tiny, ~20 KB each):
    pub output_norm: Tensor,          // [hidden_size]
    pub mtp_enorm: Option<Tensor>,    // [hidden_size]
    pub mtp_hnorm: Option<Tensor>,    // [hidden_size]
    pub mtp_shared_head_norm: Option<Tensor>, // [hidden_size]
    /// Whether the model has an MTP head available.
    has_mtp_head: bool,
    /// Hidden state from last forward pass, for deferred MTP computation.
    /// Stored pre-output_norm (the raw layer output).
    last_hidden: RefCell<Option<Tensor>>,

    // --- Synthetic mode: pre-loaded weights for testing ---
    // These are None when loading from GGUF (preloaded mode).
    synthetic_embed: Option<Tensor>,
    synthetic_layers: Option<Vec<Qwen35Layer>>,
    synthetic_lm_head: Option<Tensor>,
    synthetic_mtp: Option<MtpHead>,

    /// Tracing system for internal model behavior analysis.
    /// Zero cost when CHIMERE_TRACE_LEVEL=0 (default).
    tracer: crate::trace::Tracer,

    /// Pre-extracted raw weight CudaSlice pointers (Phase 3: v2 MoE path).
    /// Populated lazily on first forward if CUDA and CHIMERE_NO_RAW_MOE is unset.
    /// None on CPU or when raw weights extraction is not available.
    pub(crate) raw_weights: Option<crate::raw_weights::RawWeights>,

    // --- Cudarc forward path (CHIMERE_CUDARC_FORWARD=1) ---
    // Interior mutability via RefCell since forward_token takes &self.
    // The model is already !Sync (RefCell<Option<Tensor>> above) and wrapped
    // in a Mutex in the server, so this is safe.
    /// Cudarc raw weights (loaded from GGUF, bypasses QMatMul).
    cudarc_weights: RefCell<Option<compute_graph::ModelWeightsRaw>>,
    /// Cudarc compute graph (pre-allocated GPU scratch buffers).
    cudarc_graph: RefCell<Option<compute_graph::ComputeGraph>>,
    /// Per-GDN-layer recurrent state for cudarc path.
    cudarc_gdn_states: RefCell<Option<Vec<CudaSlice<f32>>>>,
    /// KV cache for cudarc attention layers.
    cudarc_kv_cache: RefCell<Option<compute_graph::KvCacheRaw>>,

    // --- libllama FFI forward path (CHIMERE_LLAMA_BACKEND=1) ---
    // Delegates the ENTIRE forward pass to ik_llama's libllama.so for
    // 93 tok/s parity. All state (KV cache, GDN recurrent) is managed
    // internally by libllama. chimere handles only tokenization + sampling.
    llama_forward: RefCell<Option<crate::llama_backend::LlamaForward>>,

    /// Last packed logprobs from forward_token (fast-sampler path).
    /// Format: [token_id, n_top, t0, lp0, t1, lp1, t2, lp2, t3, lp3, t4, lp4]
    /// Set to None when forward_token uses the slow (full-logits) path.
    /// Used by server.rs SSE streaming to emit per-token logprobs.
    pub(crate) last_packed_logprobs: RefCell<Option<Vec<f32>>>,

    // --- GateSkip: per-layer sigmoid gate for adaptive layer skipping ---
    // Based on arXiv:2510.13876. Each layer has a learned scalar gate weight.
    // gate_value = sigmoid(w_gate * mean(|hidden|)). If gate_value < threshold,
    // the layer is skipped (hidden passes through unchanged).
    // Enabled via CHIMERE_GATESKIP=1. Gate weights init to 1.0 (always pass).
    /// Per-layer gate weights (one scalar per layer). Initialized to 1.0.
    gateskip_weights: Vec<f32>,
    /// GateSkip statistics: [layer_idx] -> number of times skipped.
    gateskip_skip_counts: RefCell<Vec<u64>>,
    /// GateSkip statistics: total tokens processed.
    gateskip_total_tokens: RefCell<u64>,
}

impl Qwen35Model {
    /// Load a Qwen3.5 model from a GGUF file with preloaded quantized weights.
    ///
    /// All weight tensors are loaded as `QMatMul` at startup. For types Candle
    /// supports natively (Q8_0, Q4K, Q6K, etc.), raw quantized bytes go directly
    /// to GPU. For unsupported types (IQ3_S), CPU dequant to F32 then GPU upload.
    ///
    /// This one-time loading replaces the old per-layer CPU dequant that took ~350s.
    ///
    /// # Arguments
    /// - `path`: Path to the GGUF file
    /// - `device`: Target device (CPU or Cuda)
    /// - `_max_layers`: Reserved for future use. Kept for API compat.
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        device: &Device,
        _max_layers: Option<usize>,
    ) -> Result<Self> {
        use crate::weight_loader::Qwen35WeightLoader;

        let loader = Qwen35WeightLoader::from_gguf(path.as_ref())
            .map_err(candle_core::Error::Msg)?;
        let config = loader.config().clone();

        // --- Load tiny F32 norms ---
        let output_norm = loader.output_norm(device)?;

        // --- MTP norms (try loading, they might not exist) ---
        let (mtp_enorm, mtp_hnorm, mtp_shared_head_norm, has_mtp_head) =
            if config.nextn_predict_layers > 0 {
                match (
                    loader.mtp_enorm(device),
                    loader.mtp_hnorm(device),
                    loader.mtp_shared_head_norm(device),
                ) {
                    (Ok(en), Ok(hn), Ok(shn)) => {
                        (Some(en), Some(hn), Some(shn), true)
                    }
                    _ => (None, None, None, false),
                }
            } else {
                (None, None, None, false)
            };

        // --- Take ownership of the GgufFile from the loader ---
        let gguf = loader.into_gguf();

        // --- MRoPE ---
        let n_rot = config.rope_sections.iter().sum::<usize>() * 2;
        let mrope = MRoPE::new(
            config.head_dim,
            n_rot,
            &config.rope_sections,
            config.rope_theta,
        );

        // --- ncmoe: CPU expert offloading ---
        // CHIMERE_NCMOE=N offloads routed expert weights of the first N layers to CPU,
        // saving ~330 MB VRAM per offloaded layer. The shared expert, router, and all
        // non-expert weights remain on GPU. Forward uses CPU dequant + F32 matmul.
        let ncmoe: usize = std::env::var("CHIMERE_NCMOE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if ncmoe > 0 {
            eprintln!("[LOAD] ncmoe={}: offloading routed experts of layers 0..{} to CPU (~{:.0} MB VRAM saved)",
                ncmoe, ncmoe, ncmoe as f64 * 330.0);
        }

        // --- Preload ALL weights as QMatMul ---
        let load_start = std::time::Instant::now();

        // Embedding table: dequant on CPU, keep on CPU.
        // Only the single-row result [1, hidden_size] is transferred to GPU
        // during forward. Saves ~4.8 GB VRAM vs full [vocab, hidden] F32 on GPU.
        eprintln!("[LOAD] Loading embedding table (CPU)...");
        // Try direct dequant first; fall back to QMatMul dequant for types like Q5_K
        let embed_tokens = match gguf.load_tensor("token_embd.weight", &Device::Cpu) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("[LOAD]   (using QMatMul fallback for embed dequant)");
                let qmm = gguf.load_qmatmul("token_embd.weight", &Device::Cpu)?;
                // Dequantize by forward-passing an identity-like probe
                // Actually, use Candle's QTensor::dequantize via the QMatMul internals
                match qmm {
                    QMatMul::Tensor(t) => t,
                    QMatMul::TensorF16(t) => t.to_dtype(candle_core::DType::F32)?,
                    QMatMul::QTensor(qt) => qt.dequantize(&Device::Cpu)?,
                }
            }
        };
        eprintln!("[LOAD]   embed_tokens: {:?}, {:.1} MB (CPU)",
            embed_tokens.dims(),
            embed_tokens.elem_count() as f64 * 4.0 / 1e6);

        // Track how many tensors use native QTensor vs F32 fallback
        let mut native_count = 0usize;
        let mut fallback_count = 0usize;
        let mut fallback_f32_bytes = 0u64;

        // Helper: check if a tensor will use native quantized path or F32 fallback
        let classify = |name: &str| -> bool {
            if let Some(info) = gguf.get_tensor_info(name) {
                GgufFile::to_candle_dtype(info.ggml_type).is_some()
            } else {
                true // missing tensor = skip, don't count
            }
        };
        // Helper: get F32 byte size for fallback tensors
        let f32_size = |name: &str| -> u64 {
            gguf.get_tensor_info(name)
                .map(|info| info.n_elements as u64 * 4)
                .unwrap_or(0)
        };

        // LM head — load on GPU (output.weight is Q8_0 = ~515 MB).
        // With ncmoe freeing VRAM, we can keep it on GPU for full-speed inference.
        // If VRAM is too tight, set CHIMERE_LM_HEAD_CPU=1 to offload to CPU.
        let lm_head_cpu = std::env::var("CHIMERE_LM_HEAD_CPU").is_ok();
        let lm_head_device = if lm_head_cpu { &Device::Cpu } else { device };
        eprintln!("[LOAD] Loading LM head ({})...", if lm_head_cpu { "CPU" } else { "GPU" });
        if classify("output.weight") { native_count += 1; } else { fallback_count += 1; fallback_f32_bytes += f32_size("output.weight"); }
        let lm_head = gguf.load_qmatmul("output.weight", lm_head_device)?;
        // Load raw Q5_K bytes for ggml MMVQ kernel
        let lm_head_raw = if gguf.tensor_ggml_type("output.weight")
            == Some(crate::gguf_loader::GgmlType::Q5K)
        {
            let (raw, _, _) = gguf.load_tensor_u8_any("output.weight", device)?;
            eprintln!("[LOAD] lm_head_raw: {} bytes (Q5_K)", raw.elem_count());
            Some(raw)
        } else { None };

        // Load raw Q8_0 bytes on CPU for ggml FFI validation path
        let lm_head_q8_0_cpu = if crate::ggml_backend::is_enabled()
            && gguf.tensor_ggml_type("output.weight")
                == Some(crate::gguf_loader::GgmlType::Q8_0)
        {
            let (raw, n_elements, _dims) = gguf.load_tensor_u8_any("output.weight", &Device::Cpu)?;
            let raw_bytes: Vec<u8> = raw.flatten_all()?.to_vec1()?;
            eprintln!("[LOAD] lm_head_q8_0_cpu: {} bytes ({} elements) for ggml validation",
                raw_bytes.len(), n_elements);
            Some(raw_bytes)
        } else { None };

        // All main layers (or subset if max_layers specified)
        let n_load_layers = _max_layers.unwrap_or(config.num_main_layers).min(config.num_main_layers);
        if n_load_layers < config.num_main_layers {
            eprintln!("[LOAD] max_layers={}: loading only {}/{} layers",
                n_load_layers, n_load_layers, config.num_main_layers);
        }
        let mut q_layers = Vec::with_capacity(n_load_layers);
        for il in 0..n_load_layers {
            let layer_start = std::time::Instant::now();

            // Helper: load a QMatMul and track native vs fallback
            macro_rules! load_q {
                ($name:expr) => {{
                    let n = $name;
                    if classify(&n) { native_count += 1; } else { fallback_count += 1; fallback_f32_bytes += f32_size(&n); }
                    gguf.load_qmatmul(&n, device)?
                }};
            }

            // Detect whether this layer uses MoE FFN or dense FFN.
            // The router gate tensor `ffn_gate_inp.weight` is present only for MoE layers.
            let layer_is_moe = gguf.get_tensor_info(
                &format!("blk.{il}.ffn_gate_inp.weight")
            ).is_some();

            if config.is_attention(il) && layer_is_moe {
                // MoE Attention layer (35B-A3B)
                // Log quant types for first attention-MoE layer (diagnostic).
                if il == config.full_attn_interval - 1 {
                    for name_suffix in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                        let tname = format!("blk.{il}.{name_suffix}.weight");
                        if let Some(info) = gguf.get_tensor_info(&tname) {
                            let native = GgufFile::to_candle_dtype(info.ggml_type).is_some();
                            let dims_str = info.dims.iter().rev()
                                .map(|d| d.to_string()).collect::<Vec<_>>().join("x");
                            eprintln!("[LOAD] {}: {:?} [{}] native={}",
                                tname, info.ggml_type, dims_str, native);
                        }
                    }
                }
                let attn_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_norm.weight"), device)?;
                let post_norm = gguf.load_tensor(
                    &format!("blk.{il}.post_attention_norm.weight"), device)?;
                let wq = load_q!(format!("blk.{il}.attn_q.weight"));
                let wk = load_q!(format!("blk.{il}.attn_k.weight"));
                let wv = load_q!(format!("blk.{il}.attn_v.weight"));
                let wo = load_q!(format!("blk.{il}.attn_output.weight"));
                let q_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_q_norm.weight"), device)?;
                let k_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_k_norm.weight"), device)?;
                // Raw Q5_K bytes for attention projections — only loaded when the tensor is Q5_K.
                let wq_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_q.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_q.weight"), device)?;
                    Some(raw)
                } else { None };
                let wk_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_k.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_k.weight"), device)?;
                    Some(raw)
                } else { None };
                let wv_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_v.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_v.weight"), device)?;
                    Some(raw)
                } else { None };
                let wo_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_output.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_output.weight"), device)?;
                    Some(raw)
                } else { None };
                // MoE FFN tensors — router and shared gate as F32 Tensors.
                let gate_inp = gguf.load_tensor(
                    &format!("blk.{il}.ffn_gate_inp.weight"), device)?;
                let gate_inp_shexp = gguf.load_tensor(
                    &format!("blk.{il}.ffn_gate_inp_shexp.weight"), device)?;
                // Expert weights: IQ3_S raw bytes.
                // ncmoe: offload to CPU for first N layers to save VRAM.
                let experts_cpu = il < ncmoe;
                let expert_device = if experts_cpu { &Device::Cpu } else { device };
                let (gate_exps_raw, gate_exps_elements, gate_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_gate_exps.weight"), expert_device)?;
                let (up_exps_raw, up_exps_elements, up_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_up_exps.weight"), expert_device)?;
                let (down_exps_raw, down_exps_elements, down_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_down_exps.weight"), expert_device)?;
                if experts_cpu {
                    eprintln!("[LOAD]   layer {:2}: experts on CPU (ncmoe)", il);
                }
                // Shared expert weights: Q5_K → Candle-native QMatMul (stays quantized on GPU).
                let gate_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_gate_shexp.weight"), device)?;
                let up_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_up_shexp.weight"), device)?;
                let down_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_down_shexp.weight"), device)?;
                // Raw Q5_K bytes for ggml MMVQ shared expert kernels
                macro_rules! load_shexp_raw { ($name:expr) => {{
                    if gguf.tensor_ggml_type(&format!("blk.{}.{}", il, $name))
                        == Some(crate::gguf_loader::GgmlType::Q5K)
                    { let (r,_,_) = gguf.load_tensor_u8_any(&format!("blk.{}.{}", il, $name), device)?; Some(r) }
                    else { None }
                }}}
                let gate_shexp_raw = load_shexp_raw!("ffn_gate_shexp.weight");
                let up_shexp_raw = load_shexp_raw!("ffn_up_shexp.weight");
                let down_shexp_raw = load_shexp_raw!("ffn_down_shexp.weight");

                q_layers.push(Qwen35LayerQ::AttentionMoE(AttnLayerMoE {
                    attn_norm, post_norm, wq, wk, wv, wo, q_norm, k_norm,
                    wq_raw, wk_raw, wv_raw, wo_raw,
                    moe: MoeFFN {
                        gate_inp_t: gate_inp.t()?.contiguous()?,
                        gate_inp, gate_inp_shexp,
                        gate_exps_raw, up_exps_raw, down_exps_raw,
                        gate_exps_elements, up_exps_elements, down_exps_elements,
                        gate_exps_shape, up_exps_shape, down_exps_shape,
                        gate_shexp, up_shexp, down_shexp,
                        gate_shexp_raw, up_shexp_raw, down_shexp_raw,
                        experts_on_cpu: experts_cpu,
                    },
                }));
                eprintln!("[LOAD] layer {:2} (attn-moe{}): {:.1}s",
                    il, if experts_cpu { " CPU-experts" } else { "" },
                    layer_start.elapsed().as_secs_f64());
            } else if config.is_attention(il) {
                // Dense Attention layer (27B or any layer without MoE tensors)
                let attn_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_norm.weight"), device)?;
                let post_norm = gguf.load_tensor(
                    &format!("blk.{il}.post_attention_norm.weight"), device)?;
                let wq = load_q!(format!("blk.{il}.attn_q.weight"));
                let wk = load_q!(format!("blk.{il}.attn_k.weight"));
                let wv = load_q!(format!("blk.{il}.attn_v.weight"));
                let wo = load_q!(format!("blk.{il}.attn_output.weight"));
                let q_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_q_norm.weight"), device)?;
                let k_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_k_norm.weight"), device)?;
                let ffn_gate = load_q!(format!("blk.{il}.ffn_gate.weight"));
                let ffn_up = load_q!(format!("blk.{il}.ffn_up.weight"));
                let ffn_down = load_q!(format!("blk.{il}.ffn_down.weight"));

                q_layers.push(Qwen35LayerQ::Attention(AttnLayerQ {
                    attn_norm, post_norm, wq, wk, wv, wo,
                    q_norm, k_norm, ffn_gate, ffn_up, ffn_down,
                }));
                eprintln!("[LOAD] layer {:2} (attn): {:.1}s",
                    il, layer_start.elapsed().as_secs_f64());
            } else if layer_is_moe {
                // MoE GDN layer (35B-A3B)
                let attn_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_norm.weight"), device)?;
                let post_norm = gguf.load_tensor(
                    &format!("blk.{il}.post_attention_norm.weight"), device)?;
                // GDN SSM tensors — QMatMul (Candle native quantized, well-optimized)
                let attn_qkv = load_q!(format!("blk.{il}.attn_qkv.weight"));
                let attn_gate = load_q!(format!("blk.{il}.attn_gate.weight"));
                let ssm_out = load_q!(format!("blk.{il}.ssm_out.weight"));
                // Small SSM tensors — keep as F32/QMatMul
                let ssm_conv1d = gguf.load_tensor(
                    &format!("blk.{il}.ssm_conv1d.weight"), device)?;
                let ssm_norm = gguf.load_tensor(
                    &format!("blk.{il}.ssm_norm.weight"), device)?;
                let ssm_a = gguf.load_tensor(
                    &format!("blk.{il}.ssm_a"), device)?;
                let ssm_dt_bias = gguf.load_tensor(
                    &format!("blk.{il}.ssm_dt.bias"), device)?;
                let ssm_alpha = load_q!(format!("blk.{il}.ssm_alpha.weight"));
                let ssm_beta = load_q!(format!("blk.{il}.ssm_beta.weight"));
                // Raw Q5_K bytes for SSM projections — only loaded when the tensor is Q5_K.
                // The custom gemv_q5k_from_tensor kernel uses these at forward time.
                let attn_qkv_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_qkv.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_qkv.weight"), device)?;
                    Some(raw)
                } else { None };
                let attn_gate_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.attn_gate.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.attn_gate.weight"), device)?;
                    Some(raw)
                } else { None };
                let ssm_out_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.ssm_out.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.ssm_out.weight"), device)?;
                    Some(raw)
                } else { None };
                let ssm_beta_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.ssm_beta.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.ssm_beta.weight"), device)?;
                    Some(raw)
                } else { None };
                let ssm_alpha_raw = if gguf.tensor_ggml_type(&format!("blk.{il}.ssm_alpha.weight"))
                    == Some(crate::gguf_loader::GgmlType::Q5K)
                {
                    let (raw, _, _) = gguf.load_tensor_u8_any(
                        &format!("blk.{il}.ssm_alpha.weight"), device)?;
                    Some(raw)
                } else { None };
                // MoE FFN tensors — router and shared gate as F32 Tensors.
                let gate_inp = gguf.load_tensor(
                    &format!("blk.{il}.ffn_gate_inp.weight"), device)?;
                let gate_inp_shexp = gguf.load_tensor(
                    &format!("blk.{il}.ffn_gate_inp_shexp.weight"), device)?;
                // Expert weights: IQ3_S raw bytes.
                // ncmoe: offload to CPU for first N layers to save VRAM.
                let experts_cpu = il < ncmoe;
                let expert_device = if experts_cpu { &Device::Cpu } else { device };
                let (gate_exps_raw, gate_exps_elements, gate_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_gate_exps.weight"), expert_device)?;
                let (up_exps_raw, up_exps_elements, up_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_up_exps.weight"), expert_device)?;
                let (down_exps_raw, down_exps_elements, down_exps_shape) = gguf.load_tensor_u8(
                    &format!("blk.{il}.ffn_down_exps.weight"), expert_device)?;
                if experts_cpu {
                    eprintln!("[LOAD]   layer {:2}: experts on CPU (ncmoe)", il);
                }
                // Shared expert weights: Q5_K → Candle-native QMatMul (stays quantized on GPU).
                let gate_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_gate_shexp.weight"), device)?;
                let up_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_up_shexp.weight"), device)?;
                let down_shexp = gguf.load_qmatmul(
                    &format!("blk.{il}.ffn_down_shexp.weight"), device)?;
                macro_rules! load_shexp_raw { ($name:expr) => {{
                    if gguf.tensor_ggml_type(&format!("blk.{}.{}", il, $name))
                        == Some(crate::gguf_loader::GgmlType::Q5K)
                    { let (r,_,_) = gguf.load_tensor_u8_any(&format!("blk.{}.{}", il, $name), device)?; Some(r) }
                    else { None }
                }}}
                let gate_shexp_raw = load_shexp_raw!("ffn_gate_shexp.weight");
                let up_shexp_raw = load_shexp_raw!("ffn_up_shexp.weight");
                let down_shexp_raw = load_shexp_raw!("ffn_down_shexp.weight");

                q_layers.push(Qwen35LayerQ::GdnMoE(GdnLayerMoE {
                    attn_norm, post_norm,
                    attn_qkv, attn_gate, ssm_conv1d, ssm_norm, ssm_out,
                    ssm_a, ssm_dt_bias, ssm_alpha, ssm_beta,
                    attn_qkv_raw, attn_gate_raw, ssm_out_raw,
                    ssm_beta_raw, ssm_alpha_raw,
                    moe: MoeFFN {
                        gate_inp_t: gate_inp.t()?.contiguous()?,
                        gate_inp, gate_inp_shexp,
                        gate_exps_raw, up_exps_raw, down_exps_raw,
                        gate_exps_elements, up_exps_elements, down_exps_elements,
                        gate_exps_shape, up_exps_shape, down_exps_shape,
                        gate_shexp, up_shexp, down_shexp,
                        gate_shexp_raw, up_shexp_raw, down_shexp_raw,
                        experts_on_cpu: experts_cpu,
                    },
                }));
                eprintln!("[LOAD] layer {:2} (gdn-moe{}): {:.1}s",
                    il, if experts_cpu { " CPU-experts" } else { "" },
                    layer_start.elapsed().as_secs_f64());
            } else {
                // Dense GDN layer (27B or any layer without MoE tensors)
                let attn_norm = gguf.load_tensor(
                    &format!("blk.{il}.attn_norm.weight"), device)?;
                let post_norm = gguf.load_tensor(
                    &format!("blk.{il}.post_attention_norm.weight"), device)?;
                let ffn_gate = load_q!(format!("blk.{il}.ffn_gate.weight"));
                let ffn_up = load_q!(format!("blk.{il}.ffn_up.weight"));
                let ffn_down = load_q!(format!("blk.{il}.ffn_down.weight"));
                // GDN SSM tensors
                let attn_qkv = load_q!(format!("blk.{il}.attn_qkv.weight"));
                let attn_gate = load_q!(format!("blk.{il}.attn_gate.weight"));
                let ssm_conv1d = gguf.load_tensor(
                    &format!("blk.{il}.ssm_conv1d.weight"), device)?;
                let ssm_norm = gguf.load_tensor(
                    &format!("blk.{il}.ssm_norm.weight"), device)?;
                let ssm_out = load_q!(format!("blk.{il}.ssm_out.weight"));
                // SSM recurrence tensors
                let ssm_a = gguf.load_tensor(
                    &format!("blk.{il}.ssm_a"), device)?;
                let ssm_dt_bias = gguf.load_tensor(
                    &format!("blk.{il}.ssm_dt.bias"), device)?;
                let ssm_alpha = load_q!(format!("blk.{il}.ssm_alpha.weight"));
                let ssm_beta = load_q!(format!("blk.{il}.ssm_beta.weight"));

                q_layers.push(Qwen35LayerQ::Gdn(GdnLayerQ {
                    attn_norm, post_norm, ffn_gate, ffn_up, ffn_down,
                    attn_qkv, attn_gate, ssm_conv1d, ssm_norm, ssm_out,
                    ssm_a, ssm_dt_bias, ssm_alpha, ssm_beta,
                }));
                eprintln!("[LOAD] layer {:2} (gdn):  {:.1}s",
                    il, layer_start.elapsed().as_secs_f64());
            }
        }

        // MTP head projection (if present)
        let mtp_head = if has_mtp_head {
            let mtp_layer = config.num_main_layers;
            let mtp_name = format!("blk.{mtp_layer}.nextn.eh_proj.weight");
            if classify(&mtp_name) { native_count += 1; } else { fallback_count += 1; fallback_f32_bytes += f32_size(&mtp_name); }
            let eh_proj = gguf.load_qmatmul(&mtp_name, device)?;
            Some(MtpHeadQ { eh_proj })
        } else {
            None
        };

        let load_time = load_start.elapsed();
        eprintln!("[LOAD] All weights loaded in {:.1}s ({} layers + embed + lm_head)",
            load_time.as_secs_f64(), config.num_main_layers);
        eprintln!("[LOAD] QMatMul stats: {} native (quantized on GPU), {} fallback (F32, {:.1} GB on GPU)",
            native_count, fallback_count, fallback_f32_bytes as f64 / 1e9);
        if fallback_f32_bytes > 12_000_000_000 {
            eprintln!("[LOAD] WARNING: {:.1} GB F32 fallback tensors (IQ3_S -> F32). \
                May exceed GPU VRAM. Consider a GGUF with only Candle-supported types \
                (Q4K, Q5K, Q6K, Q8_0).",
                fallback_f32_bytes as f64 / 1e9);
        }

        let num_layers_for_gateskip = config.num_main_layers;
        let mut model = Self {
            gguf: Some(gguf),
            config,
            mrope,
            device: device.clone(),
            embed_tokens: Some(embed_tokens),
            q_layers: Some(q_layers),
            lm_head: Some(lm_head),
            lm_head_raw,
            lm_head_q8_0_cpu,
            mtp_head,
            output_norm,
            mtp_enorm,
            mtp_hnorm,
            mtp_shared_head_norm,
            has_mtp_head,
            last_hidden: RefCell::new(None),
            synthetic_embed: None,
            synthetic_layers: None,
            synthetic_lm_head: None,
            synthetic_mtp: None,
            tracer: crate::trace::Tracer::new(),
            raw_weights: None,
            cudarc_weights: RefCell::new(None),
            cudarc_graph: RefCell::new(None),
            cudarc_gdn_states: RefCell::new(None),
            cudarc_kv_cache: RefCell::new(None),
            llama_forward: RefCell::new(None),
            last_packed_logprobs: RefCell::new(None),
            gateskip_weights: vec![1.0f32; num_layers_for_gateskip],
            gateskip_skip_counts: RefCell::new(vec![0u64; num_layers_for_gateskip]),
            gateskip_total_tokens: RefCell::new(0),
        };

        // Extract raw weight CudaSlice pointers for the v2 MoE path.
        // This is a one-time cost (~0.5s for 15 GB) that eliminates
        // 5 storage_and_layout calls per MoE layer at runtime.
        if let Device::Cuda(ref cuda_dev) = device {
            // Only extract RawWeights if raw_forward is requested (costs ~110 MB VRAM)
            // Skip when ncmoe > 0: RawWeights assumes all experts are on GPU.
            let use_raw = std::env::var("CHIMERE_RAW_FORWARD").is_ok()
                && std::env::var("CHIMERE_NO_RAW_MOE").is_err()
                && ncmoe == 0;
            if use_raw {
                match crate::raw_weights::RawWeights::from_model(&model, cuda_dev) {
                    Ok(rw) => {
                        eprintln!("[LOAD] RawWeights extracted: {} GDN + {} attn layers",
                            rw.num_gdn_layers(), rw.num_attn_layers());
                        model.raw_weights = Some(rw);
                    }
                    Err(e) => {
                        eprintln!("[LOAD] WARNING: RawWeights extraction failed: {e}");
                        eprintln!("[LOAD] Falling back to v1 MoE path (6 storage_and_layout/layer)");
                    }
                }
            }
        }

        Ok(model)
    }

    /// Construct a lightweight model shell for the cudarc-only forward path.
    ///
    /// This does NOT load any Candle weights to GPU. Only the GGUF metadata
    /// is parsed to build the config and MRoPE tables. All weight fields
    /// (`q_layers`, `embed_tokens`, `lm_head`) are set to `None`.
    ///
    /// After construction, call `init_cudarc_forward()` to load the raw
    /// cudarc weights from GGUF and allocate the compute graph. The resulting
    /// model supports `forward_token` and `forward_prefill` through the
    /// cudarc fast path only.
    ///
    /// # VRAM usage
    ///
    /// This constructor uses ~0 VRAM. The cudarc weights loaded by
    /// `init_cudarc_forward()` use ~14.7 GB — the same as the full model,
    /// but without the Candle duplicate. Total: ~14.7 GB instead of ~28 GB.
    pub fn cudarc_shell(
        path: impl AsRef<std::path::Path>,
        device: &Device,
    ) -> Result<Self> {
        let gguf = crate::gguf_loader::GgufFile::open(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF: {}", e)))?;
        let config = crate::config::Qwen35Config::from_gguf(&gguf)
            .map_err(candle_core::Error::Msg)?;

        // MRoPE tables (CPU, tiny)
        let n_rot = config.rope_sections.iter().sum::<usize>() * 2;
        let mrope = crate::rope::MRoPE::new(
            config.head_dim,
            n_rot,
            &config.rope_sections,
            config.rope_theta,
        );

        // Dummy output_norm: never used in the cudarc path (the compute graph
        // has its own final_norm weight), but the struct field requires a Tensor.
        let output_norm = candle_core::Tensor::ones(
            (config.hidden_size,),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;

        eprintln!(
            "[CUDARC_SHELL] Config loaded: {} layers, hidden={}, vocab={}, experts={}",
            config.num_main_layers, config.hidden_size, config.vocab_size, config.num_experts,
        );

        let nl = config.num_main_layers;
        Ok(Self {
            gguf: None,
            config,
            mrope,
            device: device.clone(),
            embed_tokens: None,
            q_layers: None,
            lm_head: None,
            lm_head_raw: None,
            lm_head_q8_0_cpu: None,
            mtp_head: None,
            output_norm,
            mtp_enorm: None,
            mtp_hnorm: None,
            mtp_shared_head_norm: None,
            has_mtp_head: false,
            last_hidden: RefCell::new(None),
            synthetic_embed: None,
            synthetic_layers: None,
            synthetic_lm_head: None,
            synthetic_mtp: None,
            tracer: crate::trace::Tracer::new(),
            raw_weights: None,
            cudarc_weights: RefCell::new(None),
            cudarc_graph: RefCell::new(None),
            cudarc_gdn_states: RefCell::new(None),
            cudarc_kv_cache: RefCell::new(None),
            llama_forward: RefCell::new(None),
            last_packed_logprobs: RefCell::new(None),
            gateskip_weights: vec![1.0f32; nl],
            gateskip_skip_counts: RefCell::new(vec![0u64; nl]),
            gateskip_total_tokens: RefCell::new(0),
        })
    }

    /// Construct a synthetic model from pre-loaded tensors (for unit tests).
    ///
    /// This bypasses GGUF and stores all weights in memory.  Used by the
    /// synthetic test suite which constructs tiny models with known shapes.
    pub fn synthetic(
        config: Qwen35Config,
        embed_tokens: Tensor,
        layers: Vec<Qwen35Layer>,
        output_norm: Tensor,
        lm_head: Tensor,
        mtp: Option<MtpHead>,
        mrope: MRoPE,
    ) -> Self {
        let has_mtp_head = mtp.is_some();
        let (mtp_enorm, mtp_hnorm, mtp_shared_head_norm) = if let Some(ref m) = mtp {
            (Some(m.enorm.clone()), Some(m.hnorm.clone()), Some(m.shared_head_norm.clone()))
        } else {
            (None, None, None)
        };

        let nl = config.num_main_layers;
        Self {
            gguf: None,
            config,
            mrope,
            device: Device::Cpu,
            embed_tokens: None,
            q_layers: None,
            lm_head: None,
            lm_head_raw: None,
            lm_head_q8_0_cpu: None,
            mtp_head: None,
            output_norm,
            mtp_enorm,
            mtp_hnorm,
            mtp_shared_head_norm,
            has_mtp_head,
            last_hidden: RefCell::new(None),
            synthetic_embed: Some(embed_tokens),
            synthetic_layers: Some(layers),
            synthetic_lm_head: Some(lm_head),
            synthetic_mtp: mtp,
            tracer: crate::trace::Tracer::new(),
            raw_weights: None,
            cudarc_weights: RefCell::new(None),
            cudarc_graph: RefCell::new(None),
            cudarc_gdn_states: RefCell::new(None),
            cudarc_kv_cache: RefCell::new(None),
            llama_forward: RefCell::new(None),
            last_packed_logprobs: RefCell::new(None),
            gateskip_weights: vec![1.0f32; nl],
            gateskip_skip_counts: RefCell::new(vec![0u64; nl]),
            gateskip_total_tokens: RefCell::new(0),
        }
    }

    // -----------------------------------------------------------------------
    // Cudarc forward path (CHIMERE_CUDARC_FORWARD=1)
    // -----------------------------------------------------------------------

    /// Initialize the cudarc forward path by loading raw weights from GGUF
    /// and allocating the compute graph, GDN states, and KV cache.
    ///
    /// This is a no-op if `CHIMERE_CUDARC_FORWARD` is not set or the device
    /// is not CUDA. Safe to call multiple times (idempotent).
    ///
    /// The cudarc path bypasses all Candle tensor machinery and runs the entire
    /// forward pass through pre-allocated `CudaSlice` buffers. It achieves
    /// ~18.5 tok/s on the IQ3_S custom-mix GGUF.
    pub fn init_cudarc_forward(&self) -> Result<()> {
        if std::env::var("CHIMERE_CUDARC_FORWARD").is_err() {
            return Ok(());
        }

        let cuda_dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => {
                eprintln!("[CUDARC] Skipping init: not a CUDA device");
                return Ok(());
            }
        };

        // Already initialized?
        if self.cudarc_weights.borrow().is_some() {
            return Ok(());
        }

        let ncmoe: usize = std::env::var("CHIMERE_NCMOE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        let max_seq_len: usize = std::env::var("CHIMERE_KV_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);

        // Load raw weights from the same GGUF file (double-load — known inefficiency).
        let gguf_path = std::env::var("CHIMERE_MODEL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
            format!(
                "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
                home,
            )
        });

        eprintln!("[CUDARC] Loading raw weights from {} (ncmoe={})...", gguf_path, ncmoe);
        let gguf = crate::gguf_loader::GgufFile::open(&gguf_path)
            .map_err(|e| candle_core::Error::Msg(format!("CUDARC GGUF open failed: {e}")))?;
        let weights = compute_graph::ModelWeightsRaw::from_gguf(
            &gguf, &self.config, &cuda_dev, ncmoe,
        )?;
        drop(gguf);

        // Create compute graph and init RoPE tables
        let mut graph = compute_graph::ComputeGraph::new(&self.config, &cuda_dev)?;
        graph.init_rope_tables(&self.config)?;
        // Phase 3.1: Initialize CUDA Graph cache for GDN layers if enabled.
        graph.init_gdn_graph_cache()?;
        // Cache raw CUDA weight pointers to avoid device_ptr() overhead per FFI call.
        graph.init_cached_ptrs(&weights);

        // GDN recurrent states: one per GDN layer
        let num_gdn = self.config.num_gdn_layers();
        let state_size =
            self.config.ssm_dt_rank * self.config.ssm_d_state * self.config.ssm_d_state;
        let gdn_states: Vec<CudaSlice<f32>> = (0..num_gdn)
            .map(|_| {
                cuda_dev
                    .alloc_zeros::<f32>(state_size)
                    .expect("Failed to alloc GDN state")
            })
            .collect();

        // KV cache for attention layers
        let num_attn = self.config.num_attn_layers();
        let kv_cache = compute_graph::KvCacheRaw::new(
            num_attn,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_seq_len,
            &cuda_dev,
        )?;

        eprintln!(
            "[CUDARC] Forward path initialized: {} GDN + {} attn layers, ncmoe={}, kv_max_seq={}",
            num_gdn, num_attn, ncmoe, max_seq_len,
        );

        *self.cudarc_weights.borrow_mut() = Some(weights);
        *self.cudarc_graph.borrow_mut() = Some(graph);
        *self.cudarc_gdn_states.borrow_mut() = Some(gdn_states);
        *self.cudarc_kv_cache.borrow_mut() = Some(kv_cache);

        Ok(())
    }

    /// Reset the cudarc recurrent state (GDN states + KV cache) for a new
    /// conversation / request. Called at the start of each inference request
    /// so that multi-turn state doesn't leak across requests.
    pub fn reset_cudarc_state(&self) {
        let cuda_dev = match &self.device {
            Device::Cuda(d) => d,
            _ => return,
        };
        // Reset GDN recurrent states (DeltaNet S matrices)
        if let Some(ref mut gdn_states) = *self.cudarc_gdn_states.borrow_mut() {
            let state_size =
                self.config.ssm_dt_rank * self.config.ssm_d_state * self.config.ssm_d_state;
            for st in gdn_states.iter_mut() {
                let zeros = vec![0.0f32; state_size];
                let _ = cuda_dev.memcpy_htod(&zeros, st);
            }
        }
        // Reset conv1d states (CRITICAL: stale conv state corrupts subsequent requests)
        if let Some(ref mut graph) = *self.cudarc_graph.borrow_mut() {
            if let Some(ref mut scratch) = graph.gdn_cudarc {
                if let Err(e) = scratch.reset_conv_states(cuda_dev) {
                    eprintln!("[WARN] reset_cudarc_state: conv reset failed: {e}");
                }
            }
        }
        // Reset KV cache position
        if let Some(ref mut kv) = *self.cudarc_kv_cache.borrow_mut() {
            kv.pos = 0;
        }
    }

    /// Check whether the cudarc forward path is active.
    pub fn cudarc_forward_active(&self) -> bool {
        self.cudarc_weights.borrow().is_some()
    }

    // -----------------------------------------------------------------------
    // libllama FFI forward path (CHIMERE_LLAMA_BACKEND=1)
    // -----------------------------------------------------------------------

    /// Initialize the libllama FFI backend by loading the model via libllama.so.
    ///
    /// This delegates the ENTIRE forward pass to ik_llama's optimized CUDA kernels,
    /// achieving 93 tok/s parity with llama-server. chimere handles only tokenization,
    /// sampling, and the HTTP/SSE layer.
    ///
    /// This is a no-op if `CHIMERE_LLAMA_BACKEND` is not set.
    /// Safe to call multiple times (idempotent).
    ///
    /// **IMPORTANT**: This is mutually exclusive with cudarc_forward. If both are
    /// set, llama_backend takes priority (it subsumes cudarc completely).
    pub fn init_llama_forward(&self) -> Result<()> {
        if !crate::llama_backend::is_enabled() {
            return Ok(());
        }

        // Already initialized?
        if self.llama_forward.borrow().is_some() {
            return Ok(());
        }

        eprintln!("[LLAMA_BACKEND] Initializing libllama FFI forward path...");

        let llama = crate::llama_backend::from_env()
            .map_err(|e| candle_core::Error::Msg(format!("llama_backend init failed: {}", e)))?;

        // Verify vocab size matches our config
        if llama.n_vocab() != self.config.vocab_size {
            eprintln!(
                "[LLAMA_BACKEND] WARNING: vocab size mismatch: llama={} vs config={}. \
                 Using llama's vocab size for logits.",
                llama.n_vocab(), self.config.vocab_size,
            );
        }

        *self.llama_forward.borrow_mut() = Some(llama);

        eprintln!("[LLAMA_BACKEND] Ready. Forward pass delegated to libllama.so (ik_llama sm120).");
        Ok(())
    }

    /// Reset the libllama backend state (KV cache + recurrent state) for a new request.
    pub fn reset_llama_state(&self) {
        if let Some(ref mut llama) = *self.llama_forward.borrow_mut() {
            llama.reset();
        }
    }

    /// Check whether the libllama FFI forward path is active.
    pub fn llama_forward_active(&self) -> bool {
        self.llama_forward.borrow().is_some()
    }

    /// Get mutable access to LlamaForward (for agent context switching).
    pub fn llama_forward_mut(&self) -> std::cell::RefMut<'_, Option<crate::llama_backend::LlamaForward>> {
        self.llama_forward.borrow_mut()
    }

    /// Take the last packed logprobs from the fast-sampler path.
    /// Returns None if the last forward_token used the slow path.
    /// Format: [token_id, n_top, t0, lp0, t1, lp1, t2, lp2, t3, lp3, t4, lp4]
    pub fn take_last_packed_logprobs(&self) -> Option<Vec<f32>> {
        self.last_packed_logprobs.borrow_mut().take()
    }

    /// Set logit bias on the C++ sampler (e.g., suppress </think>).
    pub fn llama_set_logit_bias(&self, token_id: u32, bias: f32) {
        if let Some(ref mut llama) = *self.llama_forward.borrow_mut() {
            llama.set_logit_bias(token_id, bias);
        }
    }

    /// Set Engram logit biases (n-gram predictions). Merges with existing biases.
    pub fn llama_set_engram_bias(&self, predictions: &[(u32, f32)]) {
        if let Some(ref llama) = *self.llama_forward.borrow() {
            llama.set_engram_bias(predictions);
        }
    }

    /// Clear Engram biases only (keep manual biases).
    pub fn llama_clear_engram_bias(&self) {
        if let Some(ref llama) = *self.llama_forward.borrow() {
            llama.clear_engram_bias();
        }
    }

    /// Forward pass for a single token.
    ///
    /// # Arguments
    /// - `token`: Token ID
    /// - `state`: Mutable GDN recurrent state (updated in-place)
    ///
    /// # Returns
    /// - `logits` [1, vocab_size]: unnormalised log-probabilities
    /// - `mtp_logits`: optional MTP logits [1, vocab_size] (if MTP head present)
    ///
    /// # Note
    /// GDN layers implement the DeltaNet recurrence:
    ///   S = decay * S + outer(delta, k) where delta = beta * (v - S @ k).
    /// The state accumulates across tokens, allowing the model to remember context.
    pub fn forward_token(
        &self,
        token: u32,
        state: &mut GdnRecurrentState,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Clear stale logprobs (set in fast-sampler path only).
        *self.last_packed_logprobs.borrow_mut() = None;

        // ---- libllama FFI fast path (CHIMERE_LLAMA_BACKEND=1) ----
        // Delegates the entire forward pass to ik_llama's libllama.so.
        // All state (KV cache, GDN recurrent) managed by libllama internally.
        // This is the fastest path: 93 tok/s parity with llama-server.
        {
            let is_llama = self.llama_forward.borrow().is_some();
            if is_llama {
                let mut llama_ref = self.llama_forward.borrow_mut();
                let llama = llama_ref.as_mut().unwrap();

                // Fast path: forward + sample with logprobs, zero-copy.
                // Returns a fake tensor (1, 11): [token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4]
                // The caller detects shape (1, 11) as "pre-sampled with logprobs".
                if llama.has_fast_sampler() {
                    llama.forward_token_no_logits(token)
                        .map_err(|e| candle_core::Error::Msg(e))?;

                    let (sampled, logprobs) = llama.sample_token_fast_with_logprobs()
                        .map_err(|e| candle_core::Error::Msg(e))?;

                    state.position = llama.pos() as usize;

                    // Pack into [token_id, n_top, t0, lp0, t1, lp1, t2, lp2, t3, lp3, t4, lp4]
                    let mut packed = vec![sampled as f32, logprobs.len() as f32];
                    for lp in &logprobs {
                        packed.push(lp.token as f32);
                        packed.push(lp.logprob);
                    }
                    while packed.len() < 12 { packed.push(0.0); }

                    // Store packed logprobs for SSE streaming to read.
                    *self.last_packed_logprobs.borrow_mut() = Some(packed.clone());

                    let logits_tensor = Tensor::from_vec(
                        packed,
                        (1, 12),
                        &Device::Cpu,
                    )?;
                    return Ok((logits_tensor, None));
                }

                // Slow path: copy all logits to Rust (993KB per token)
                let logits_cpu = llama.forward_token(token)
                    .map_err(|e| candle_core::Error::Msg(e))?;

                let n_vocab = logits_cpu.len();
                let logits_tensor = Tensor::from_vec(
                    logits_cpu,
                    (1, n_vocab),
                    &Device::Cpu,
                )?;

                // Sync position counter for the server's multi-turn tracking.
                state.position = llama.pos() as usize;

                return Ok((logits_tensor, None));
            }
        }

        // ---- Cudarc fast path (CHIMERE_CUDARC_FORWARD=1) ----
        // Uses pre-allocated CudaSlice buffers — zero Candle overhead.
        // The cudarc state (GDN + KV cache) is maintained in self via RefCell.
        // The Candle GdnRecurrentState is NOT used in this path (position is
        // tracked separately in cudarc_kv_cache.pos).
        {
            let is_cudarc = self.cudarc_weights.borrow().is_some();
            if is_cudarc {
                // Borrow all four RefCells. These are independent cells, so
                // simultaneous borrows are safe (no aliasing).
                let weights_ref = self.cudarc_weights.borrow();
                let mut graph_ref = self.cudarc_graph.borrow_mut();
                let mut gdn_ref = self.cudarc_gdn_states.borrow_mut();
                let mut kv_ref = self.cudarc_kv_cache.borrow_mut();

                let weights = weights_ref.as_ref().unwrap();
                let graph = graph_ref.as_mut().unwrap();
                let gdn_states = gdn_ref.as_mut().unwrap();
                let kv_cache = kv_ref.as_mut().unwrap();

                let logits_cpu = graph.forward_token(
                    token, weights, gdn_states, kv_cache,
                )?;

                // Wrap as Candle Tensor for compatibility with the sampling code.
                // Sampling runs on CPU anyway, so this is zero-copy on the GPU side.
                let logits_tensor = Tensor::from_vec(
                    logits_cpu,
                    (1, self.config.vocab_size),
                    &Device::Cpu,
                )?;

                // Sync the position counter so the Candle state stays coherent
                // (needed for multi-turn: the server reads state.position).
                state.position = kv_cache.pos;

                return Ok((logits_tensor, None));
            }
        }

        if self.q_layers.is_some() {
            // Raw forward v4: bypass Candle Tensor overhead for GDN layers.
            // Toggle: CHIMERE_RAW_FORWARD=1
            use once_cell::sync::Lazy;
            static USE_RAW: Lazy<bool> = Lazy::new(|| {
                let v = std::env::var("CHIMERE_RAW_FORWARD").is_ok();
                if v { eprintln!("[RAW_FORWARD] enabled — bypassing Candle for GDN layers"); }
                v
            });
            if *USE_RAW {
                if let Some(ref rw) = self.raw_weights {
                    let dev = match &self.device {
                        Device::Cuda(d) => d,
                        _ => candle_core::bail!("raw_forward requires CUDA"),
                    };
                    // Take raw_forward_bufs out of state to avoid borrow conflict:
                    // raw_forward_token borrows &mut state AND we need &mut raw_bufs.
                    let mut raw_bufs = state.raw_forward_bufs.take().unwrap_or_else(|| {
                        crate::raw_forward::RawForwardBufs::new(&self.config, dev)
                            .expect("Failed to allocate raw forward buffers")
                    });
                    let logits_slice = crate::raw_forward::raw_forward_token(
                        token, self, state, rw,
                        &mut raw_bufs.gdn_bufs,
                        &mut raw_bufs.gdn_state_dbufs,
                        &mut raw_bufs.conv_state_slices,
                        dev,
                    )?;
                    // Put the bufs back
                    state.raw_forward_bufs = Some(raw_bufs);
                    let logits_tensor = crate::raw_forward::tensor_from_cuda_slice(
                        logits_slice, self.config.vocab_size, dev,
                    );
                    return Ok((logits_tensor, None));
                }
            }
            self.forward_token_preloaded(token, state)
        } else {
            self.forward_token_synthetic(token, state)
        }
    }


    /// Forward pass using preloaded quantized weights (QMatMul).
    ///
    /// All weights are already on the compute device. The forward pass is pure
    /// GPU computation: no mmap reads, no CPU dequant, no PCIe transfers.
    fn forward_token_preloaded(
        &self,
        token: u32,
        state: &mut GdnRecurrentState,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let eps = self.config.rms_norm_eps;
        let q_layers = self.q_layers.as_ref().unwrap();
        let embed_tokens = self.embed_tokens.as_ref().unwrap();
        let lm_head = self.lm_head.as_ref().unwrap();

        let dbg = debug_enabled();

        // Per-layer skip: skip individual layers where hidden state barely changes.
        // Set CHIMERE_SKIP_LAYERS=8,12,16 to skip those layer indices entirely
        // (hidden passes through unchanged). Determine candidates by running with
        // CHIMERE_TRACE_LEVEL=1 and looking for cos_sim ≈ 1.0 layers.
        let skip_layers: &std::collections::HashSet<usize> = {
            use once_cell::sync::Lazy;
            static SKIP: Lazy<std::collections::HashSet<usize>> = Lazy::new(|| {
                match std::env::var("CHIMERE_SKIP_LAYERS") {
                    Ok(val) if !val.is_empty() => {
                        let set: std::collections::HashSet<usize> = val
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if !set.is_empty() {
                            let mut sorted: Vec<usize> = set.iter().copied().collect();
                            sorted.sort();
                            eprintln!("[SKIP_LAYERS] skipping {} layers: {:?}", set.len(), sorted);
                        }
                        set
                    }
                    _ => std::collections::HashSet::new(),
                }
            });
            &SKIP
        };

        // Early exit: skip remaining GDN layers when hidden state has converged.
        // Enable with CHIMERE_EARLY_EXIT=1. Disabled by default.
        let early_exit_enabled = {
            use once_cell::sync::Lazy;
            static EARLY_EXIT: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_EARLY_EXIT").is_ok());
            *EARLY_EXIT
        };
        let early_exit_threshold: f32 = {
            use once_cell::sync::Lazy;
            static THRESH: Lazy<f32> = Lazy::new(|| {
                std::env::var("CHIMERE_EXIT_THRESH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.05) // default: cos_sim > 0.95
            });
            *THRESH
        };
        // GateSkip: per-layer adaptive skipping (arXiv:2510.13876).
        // gate_value = sigmoid(w_gate * mean(|hidden|)). Skip layer if gate < threshold.
        // Enabled with CHIMERE_GATESKIP=1. Default off.
        let gateskip_enabled = {
            use once_cell::sync::Lazy;
            static GATESKIP: Lazy<bool> = Lazy::new(|| {
                let v = std::env::var("CHIMERE_GATESKIP").is_ok();
                if v { eprintln!("[GATESKIP] enabled — adaptive layer skipping active"); }
                v
            });
            *GATESKIP
        };
        let gateskip_threshold: f32 = {
            use once_cell::sync::Lazy;
            static THRESH: Lazy<f32> = Lazy::new(|| {
                std::env::var("CHIMERE_GATESKIP_THRESH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.1) // default: skip if gate < 0.1
            });
            *THRESH
        };
        let mut gateskip_skipped_this_token: usize = 0;

        // Candle ops counter: reset before each token
        crate::candle_counter::reset();

        let mut prev_hidden: Option<Tensor> = None;
        let min_layers: usize = {
            use once_cell::sync::Lazy;
            static MIN_LAYERS: Lazy<usize> = Lazy::new(|| {
                std::env::var("CHIMERE_EXIT_MIN_LAYERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(32) // default: half the model (64 layers)
            });
            *MIN_LAYERS
        };

        // 1. Embed: index into preloaded embedding table (on CPU),
        //    then transfer the single row to compute device.
        let token_tensor = Tensor::new(&[token], &Device::Cpu)?;
        let hidden_cpu = embed_tokens.index_select(&token_tensor, 0)?; // [1, hidden_size] on CPU
        let mut hidden = hidden_cpu.to_device(&self.device)?;
        if dbg {
            debug_dump(&format!("embed(token={})", token), &hidden);
        }

        // 2. Layer loop: use preloaded QMatMul weights
        let act_f16 = act_f16_enabled();
        let mut exited_early = false;
        for (il, layer) in q_layers.iter().enumerate() {
            // Per-layer skip: if this layer is in the skip set, pass hidden through unchanged
            if skip_layers.contains(&il) {
                continue;
            }
            // GateSkip: compute scalar gate and skip layer if gate value is below threshold.
            // gate = sigmoid(w_gate * mean(|hidden|)). Weights init to 1.0 → sigmoid(mean_abs)
            // is always close to 1.0 for typical hidden magnitudes, so no layers are skipped
            // until gate weights are tuned (lowered) for specific layers.
            if gateskip_enabled && il < self.gateskip_weights.len() {
                // Compute mean absolute value of hidden state (cheap: 1 abs + 1 mean on [1, 2048])
                let mean_abs: f32 = hidden.abs()?.mean_all()?.to_scalar()?;
                let gate_input = self.gateskip_weights[il] * mean_abs;
                let gate_value = 1.0 / (1.0 + (-gate_input).exp()); // sigmoid
                if gate_value < gateskip_threshold {
                    // Skip this layer: hidden passes through unchanged
                    gateskip_skipped_this_token += 1;
                    self.gateskip_skip_counts.borrow_mut()[il] += 1;
                    continue;
                }
            }
            // F16 activation: upcast hidden to F32 before layer computation.
            // All layer-internal ops (rms_norm, QMatMul, residuals) require F32.
            if act_f16 && hidden.dtype() == candle_core::DType::F16 {
                hidden = hidden.to_dtype(candle_core::DType::F32)?;
            }
            match layer {
                Qwen35LayerQ::Gdn(w) => {
                    hidden = self.forward_gdn_layer_q(il, w, &hidden, eps, state)?;
                }
                Qwen35LayerQ::Attention(w) => {
                    hidden = self.forward_attn_layer_q(il, w, &hidden, eps, state)?;
                }
                Qwen35LayerQ::GdnMoE(w) => {
                    // GDN (recurrent SSM) — same as dense but with MoE FFN
                    hidden = self.forward_gdn_layer_moe(il, w, &hidden, eps, state)?;
                }
                Qwen35LayerQ::AttentionMoE(w) => {
                    // Full attention — same as dense but with MoE FFN
                    hidden = self.forward_attn_layer_moe(il, w, &hidden, eps, state)?;
                }
            }
            // F16 activation: cast layer output to F16 to halve inter-layer memory.
            // The F32 intermediate is dropped, freeing GPU memory immediately.
            if act_f16 {
                hidden = hidden.to_dtype(candle_core::DType::F16)?;
            }
            if dbg && (il < 4 || il == 7 || il == 63) {
                let ltype = if self.config.is_recurrent(il) { "GDN" } else { "ATN" };
                debug_dump(&format!("layer_{:02}_{}", il, ltype), &hidden);
            }

            // Periodic GPU sync to force cudaFree of intermediate tensors.
            // Without this, Candle's async cudaFree accumulates hundreds of MB
            // of unreleased fragments over 40 layers, causing OOM.
            // Sync every 10 layers = 4 syncs/token — minimal perf impact (~0.1ms each).
            if il % 10 == 9 {
                if let Device::Cuda(ref cd) = self.device {
                    use candle_core::backend::BackendDevice;
                    let _ = cd.synchronize();
                }
            }

            // Early exit: check convergence every 4 layers after min_layers
            if early_exit_enabled && il >= min_layers && il % 4 == 3 {
                if let Some(ref prev) = prev_hidden {
                    let dot: f32 = (&hidden * prev)?.sum_all()?.to_scalar()?;
                    let h_norm: f32 = hidden.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    let p_norm: f32 = prev.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    let cos_sim = dot / (h_norm * p_norm + 1e-8);

                    if cos_sim > (1.0 - early_exit_threshold) {
                        eprintln!("[EARLY_EXIT] layer {}/{}: cos_sim={:.6}, exiting",
                            il + 1, q_layers.len(), cos_sim);
                        exited_early = true;
                        break;
                    }
                }
                prev_hidden = Some(hidden.clone());
            }
        }

        // Early exit statistics (reported every 50 tokens)
        if early_exit_enabled {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static TOTAL: AtomicUsize = AtomicUsize::new(0);
            static EARLY: AtomicUsize = AtomicUsize::new(0);
            TOTAL.fetch_add(1, Ordering::Relaxed);
            if exited_early {
                EARLY.fetch_add(1, Ordering::Relaxed);
            }
            let t = TOTAL.load(Ordering::Relaxed);
            if t % 50 == 0 && t > 0 {
                let e = EARLY.load(Ordering::Relaxed);
                eprintln!("[EARLY_EXIT] {}/{} tokens exited early ({:.1}%)",
                    e, t, 100.0 * e as f64 / t as f64);
            }
        }

        // GateSkip statistics (reported every 100 tokens)
        if gateskip_enabled {
            let mut total = self.gateskip_total_tokens.borrow_mut();
            *total += 1;
            let t = *total;
            if t % 100 == 0 {
                let counts = self.gateskip_skip_counts.borrow();
                let total_skips: u64 = counts.iter().sum();
                let total_possible = t * q_layers.len() as u64;
                eprintln!(
                    "[GATESKIP] after {} tokens: {}/{} layer-evals skipped ({:.1}%), this token skipped {}/{}",
                    t, total_skips, total_possible,
                    100.0 * total_skips as f64 / total_possible as f64,
                    gateskip_skipped_this_token, q_layers.len(),
                );
                // Per-layer breakdown (only layers with >0 skips)
                let skipped_layers: Vec<String> = counts.iter().enumerate()
                    .filter(|(_, &c)| c > 0)
                    .map(|(i, c)| format!("L{}:{}", i, c))
                    .collect();
                if !skipped_layers.is_empty() {
                    eprintln!("[GATESKIP] per-layer: {}", skipped_layers.join(" "));
                }
            }
        }

        // Advance position after processing this token
        state.advance(1);

        // F16 activation: upcast hidden back to F32 for output_norm, lm_head, and MTP
        if act_f16 && hidden.dtype() == candle_core::DType::F16 {
            hidden = hidden.to_dtype(candle_core::DType::F32)?;
        }

        // Store hidden for deferred MTP computation
        *self.last_hidden.borrow_mut() = Some(hidden.clone());

        // 3. Output norm
        let h_last = rms_norm(&hidden, &self.output_norm, eps)?;

        // 4. LM head — dispatch to ggml FFI, ggml Q5_K, or Candle QMatMul

        // Path A: ggml FFI Q8_0 validation (CHIMERE_GGML_LM_HEAD=1)
        // CPU round-trip: GPU->CPU->ggml_Q8_0_gemv->CPU->GPU. Slow but correct.
        let use_ggml_q8 = crate::ggml_backend::is_enabled();
        let use_ggml_q5k = {
            use once_cell::sync::Lazy;
            static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *G
        };
        let logits = if use_ggml_q8 && self.lm_head_q8_0_cpu.is_some() {
            let w_raw = self.lm_head_q8_0_cpu.as_ref().unwrap();
            let ggml_logits = crate::ggml_backend::ggml_lm_head_forward(
                &h_last,
                w_raw,
                self.config.vocab_size,
                self.config.hidden_size,
                &self.device,
            )?;

            // When debug is on, also compute Candle logits and compare top-5
            if dbg {
                let candle_logits = Self::lm_head_forward(lm_head, &h_last)?;
                crate::ggml_backend::compare_top5(
                    "ggml_Q8_0", &ggml_logits,
                    "candle_QMatMul", &candle_logits,
                )?;
            }

            ggml_logits
        } else if use_ggml_q8 && self.lm_head_q8_0_cpu.is_none() {
            // Toggle is set but output.weight is not Q8_0 -- warn once and fall through
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!("[GGML_LM_HEAD] WARNING: CHIMERE_GGML_LM_HEAD=1 but output.weight \
                        is not Q8_0. Falling back to Candle QMatMul.");
                }
            }
            Self::lm_head_forward(lm_head, &h_last)?
        }
        // Path B: ggml Q5_K GPU kernel (CHIMERE_GGML_Q5K=1)
        else if use_ggml_q5k && self.lm_head_raw.is_some() {
            use candle_core::Storage;
            let h_flat = h_last.flatten_all()?.contiguous()?;
            let raw_t = self.lm_head_raw.as_ref().unwrap();
            let raw_c = raw_t.contiguous()?;

            let (w_stor, w_lay) = raw_c.storage_and_layout();
            let w_cuda = match &*w_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("lm_head not CUDA") };
            let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(
                w_lay.start_offset()..w_lay.start_offset() + raw_c.elem_count());

            let (i_stor, i_lay) = h_flat.storage_and_layout();
            let i_cuda = match &*i_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("h_last not CUDA") };
            let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(
                i_lay.start_offset()..i_lay.start_offset() + h_flat.elem_count());

            let Device::Cuda(cuda_dev) = h_flat.device() else {
                candle_core::bail!("ggml lm_head requires CUDA");
            };

            // Allocate temp buffers for Q8_1 + output
            let ncols = hidden.dim(1)?;
            let nrows = self.config.vocab_size;
            let q8_size = ((ncols + 31) / 32) * 36;
            let mut q8_buf = cuda_dev.alloc_zeros::<u8>(q8_size)
                .map_err(|e| candle_core::Error::Msg(format!("lm_head q8 alloc: {e}")))?;
            let mut out_buf = cuda_dev.alloc_zeros::<f32>(nrows)
                .map_err(|e| candle_core::Error::Msg(format!("lm_head out alloc: {e}")))?;

            crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv_f32(
                &w_view, &i_view, &mut q8_buf, &mut out_buf, nrows, ncols, cuda_dev)?;

            // Wrap as Tensor [1, vocab_size]
            let out_tensor = Tensor::from_storage(
                Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    out_buf, cuda_dev.clone())),
                candle_core::Shape::from_dims(&[1, nrows]),
                candle_core::op::BackpropOp::none(), false);
            drop(w_stor); drop(i_stor);
            out_tensor
        }
        // Path C: default Candle QMatMul (lm_head may be on CPU)
        else {
            Self::lm_head_forward(lm_head, &h_last)?
        };

        if dbg {
            debug_dump("output_norm", &h_last);
            // Print top-5 logits
            let logits_cpu: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = logits_cpu.iter().enumerate()
                .map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("[DEBUG] logits top-5:");
            for &(idx, val) in indexed.iter().take(5) {
                eprintln!("[DEBUG]   token={} logit={:.4}", idx, val);
            }
        }

        // 5. MTP is computed via compute_mtp() after the caller samples the main token.
        // This enables the correct architecture: MTP receives embed(predicted_token),
        // not embed(input_token).
        let mtp_logits: Option<Tensor> = None;

        // Candle ops counter: print total after forward pass
        if crate::candle_counter::enabled() {
            // Count embed (1 index_select + 1 to_device) + output_norm (1) + lm_head (1)
            crate::candle_counter::tick_n(4);
            eprintln!("[CANDLE_OPS] {} total ops this token", crate::candle_counter::get());
        }

        Ok((logits, mtp_logits))
    }




    /// Forward pass using pre-loaded synthetic weights (for tests).
    fn forward_token_synthetic(
        &self,
        token: u32,
        state: &mut GdnRecurrentState,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let embed = self.synthetic_embed.as_ref()
            .expect("synthetic model requires embed_tokens");
        let layers = self.synthetic_layers.as_ref()
            .expect("synthetic model requires layers");
        let lm_head = self.synthetic_lm_head.as_ref()
            .expect("synthetic model requires lm_head");

        let device = embed.device();
        let eps = self.config.rms_norm_eps;

        // 1. Embed
        let token_tensor = Tensor::new(&[token], device)?;
        let mut hidden = embed.index_select(&token_tensor, 0)?;

        // 2. Layer loop (track global layer index for attn_index mapping)
        for (il, layer) in layers.iter().enumerate() {
            match layer {
                Qwen35Layer::Gdn(w) => {
                    hidden = self.forward_gdn_layer_synthetic(w, &hidden, eps)?;
                }
                Qwen35Layer::Attention(w) => {
                    hidden = self.forward_attn_layer_synthetic(w, &hidden, eps, state, il)?;
                }
            }
        }

        // Advance position after processing this token
        state.advance(1);

        // 3. Output norm
        let h_last = rms_norm(&hidden, &self.output_norm, eps)?;

        // 4. LM head
        let logits = h_last.matmul(&lm_head.t()?)?;

        // 5. MTP (optional)
        let mtp_logits = if let Some(mtp) = &self.synthetic_mtp {
            let e_norm = rms_norm(&hidden, &mtp.enorm, eps)?;
            let h_norm = rms_norm(&h_last, &mtp.hnorm, eps)?;
            let concat = Tensor::cat(&[&e_norm, &h_norm], 1)?;
            let projected = concat.matmul(&mtp.eh_proj.t()?)?;
            let projected = rms_norm(&projected, &mtp.shared_head_norm, eps)?;
            let mtp_logits = projected.matmul(&lm_head.t()?)?;
            Some(mtp_logits)
        } else {
            None
        };

        Ok((logits, mtp_logits))
    }

    /// Simplified GDN layer forward (Phase 6 stub) — synthetic weights path.
    fn forward_gdn_layer_synthetic(
        &self,
        w: &GdnLayerWeights,
        hidden: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let _normed = rms_norm(hidden, &w.attn_norm, eps)?;
        let h_mid = hidden.clone();
        let normed_ffn = rms_norm(&h_mid, &w.post_attention_norm, eps)?;
        let ffn_out = self.swiglu_ffn(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;
        let h_out = (&h_mid + &ffn_out)?;
        Ok(h_out)
    }

    /// Full attention layer forward with KV cache — synthetic weights path.
    fn forward_attn_layer_synthetic(
        &self,
        w: &AttnLayerWeights,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let attn_idx = self.config.attn_index(layer_idx).unwrap();

        let normed = rms_norm(hidden, &w.attn_norm, eps)?;
        let q_full = normed.matmul(&w.wq.t()?)?;
        let k_proj = normed.matmul(&w.wk.t()?)?;
        let v_proj = normed.matmul(&w.wv.t()?)?;

        let q_dim = num_heads * head_dim;
        let q = q_full.narrow(1, 0, q_dim)?;
        let q_gate_raw = q_full.narrow(1, q_dim, q_dim)?;

        // QK norm
        let q = q.reshape((1, num_heads, head_dim))?;
        let q = rms_norm(&q, &w.q_norm, eps)?;              // [1, num_heads, head_dim]
        let k = k_proj.reshape((1, num_kv_heads, head_dim))?;
        let k = rms_norm(&k, &w.k_norm, eps)?;              // [1, num_kv_heads, head_dim]

        // Apply MRoPE to Q and K
        // MRoPE.apply expects [seq_len, num_heads, head_dim]
        // q is [batch=1, num_heads, head_dim], reinterpret batch as seq_len (both are 1)
        let positions = MRoPE::text_positions(1, state.position);
        let q_rotated = self.mrope.apply(&q, &positions)?;   // [1, num_heads, head_dim]
        let k_rotated = self.mrope.apply(&k, &positions)?;   // [1, num_kv_heads, head_dim]

        // Reshape to cache format [1, heads, 1, head_dim]
        let q_attn = q_rotated.unsqueeze(2)?;                // [1, num_heads, 1, head_dim]
        let k_new = k_rotated.unsqueeze(2)?;                 // [1, num_kv_heads, 1, head_dim]
        let v_new = v_proj.reshape((1, num_kv_heads, 1, head_dim))?;

        // Append to KV cache
        let (k_cache, v_cache) = state.kv_append(attn_idx, &k_new, &v_new)?;

        // GQA expansion
        let group_size = num_heads / num_kv_heads;
        let cached_seq_len = k_cache.dim(2)?;
        let k_expanded = k_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;
        let v_expanded = v_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;

        // Scaled dot-product attention
        let scores = q_attn.matmul(&k_expanded.transpose(2, 3)?)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_out = attn_weights.matmul(&v_expanded)?;
        let attn_out = attn_out.squeeze(2)?;                 // [1, num_heads, head_dim]

        // Apply gate (sigmoid)
        let q_gate_raw = q_gate_raw.reshape((1, num_heads, head_dim))?;
        let gate = sigmoid(&q_gate_raw)?;
        let gated_out = (&attn_out * &gate)?;
        let gated_out = gated_out.reshape((1, num_heads * head_dim))?;

        // Output projection
        let attn_projected = gated_out.matmul(&w.wo.t()?)?;
        let h_mid = (hidden + &attn_projected)?;

        let normed_ffn = rms_norm(&h_mid, &w.post_attention_norm, eps)?;
        let ffn_out = self.swiglu_ffn(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;
        let h_out = (&h_mid + &ffn_out)?;

        Ok(h_out)
    }

    /// SwiGLU FFN: down(silu(gate(x)) * up(x)) — for synthetic F32 tensors.
    fn swiglu_ffn(
        &self,
        x: &Tensor,
        ffn_gate: &Tensor,
        ffn_up: &Tensor,
        ffn_down: &Tensor,
    ) -> Result<Tensor> {
        let gate_out = x.matmul(&ffn_gate.t()?)?;
        let up_out = x.matmul(&ffn_up.t()?)?;
        let activated = silu_activation(&gate_out)?;
        let intermediate = (&activated * &up_out)?;
        intermediate.matmul(&ffn_down.t()?)
    }

    /// SwiGLU FFN: down(silu(gate(x)) * up(x)) — for QMatMul weights.
    ///
    /// `QMatMul::forward` handles the transpose internally (it stores the
    /// weight as [out_features, in_features] and applies x @ W^T).
    fn swiglu_ffn_q(
        &self,
        x: &Tensor,
        ffn_gate: &QMatMul,
        ffn_up: &QMatMul,
        ffn_down: &QMatMul,
    ) -> Result<Tensor> {
        let gate_out = ffn_gate.forward(x)?;
        let up_out = ffn_up.forward(x)?;
        let activated = silu_activation(&gate_out)?;
        let intermediate = (&activated * &up_out)?;
        ffn_down.forward(&intermediate)
    }

    /// Number of layers (all main layers for preloaded mode).
    pub fn num_layers(&self) -> usize {
        if let Some(ref layers) = self.synthetic_layers {
            layers.len()
        } else if let Some(ref layers) = self.q_layers {
            layers.len()
        } else {
            self.config.num_main_layers
        }
    }

    /// Check whether MTP head is available.
    pub fn has_mtp(&self) -> bool {
        self.has_mtp_head
    }

    // -----------------------------------------------------------------------
    // Batch prefill (forward_prefill + helpers)
    // -----------------------------------------------------------------------

    /// Batch prefill: process all prompt tokens in a single call.
    ///
    /// Instead of calling `forward_token` N times (N forward passes through all
    /// 64 layers), this method:
    ///
    /// - Embeds all N tokens at once via `index_select`.
    /// - For each **GDN layer**: batches all linear projections into one QMatMul
    ///   call, then runs the cheap sequential DeltaNet recurrence.
    /// - For each **attention layer**: runs full causal self-attention over all N
    ///   positions with a lower-triangular mask.
    ///
    /// After prefill the KV cache and GDN state contain the full prompt context,
    /// and `state.position` is advanced by N, ready for single-token generation.
    ///
    /// # Returns
    ///
    /// Logits `[1, vocab_size]` for the **last** prompt token only, so the
    /// caller can immediately sample the first generated token.
    ///
    /// # Panics / Errors
    ///
    /// Returns `Err` if the model was not loaded from GGUF (synthetic mode has
    /// no preloaded weights).  Will also error if `tokens` is empty.
    pub fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        assert!(
            !tokens.is_empty(),
            "forward_prefill: tokens slice must not be empty"
        );

        // ---- libllama FFI fast path: batch prefill ----
        if self.llama_forward.borrow().is_some() {
            let n = tokens.len();
            let prefill_start = std::time::Instant::now();

            let mut llama_ref = self.llama_forward.borrow_mut();
            let llama = llama_ref.as_mut().unwrap();

            // Prefill always copies logits (one-time cost, ~1ms).
            // The real gain is in the generation loop (sample_token_fast).
            let logits_cpu = llama.forward_prefill(tokens)
                .map_err(|e| candle_core::Error::Msg(e))?;

            let elapsed = prefill_start.elapsed();
            let tps = n as f64 / elapsed.as_secs_f64();
            eprintln!(
                "[LLAMA_BACKEND] Prefill done: {} tokens in {:.2}s ({:.1} tok/s)",
                n, elapsed.as_secs_f64(), tps,
            );

            let n_vocab = logits_cpu.len();
            let logits_tensor = Tensor::from_vec(
                logits_cpu,
                (1, n_vocab),
                &Device::Cpu,
            )?;

            state.position = llama.pos() as usize;
            return Ok(logits_tensor);
        }

        // ---- Cudarc fast path: batch prefill ----
        // Uses ComputeGraph::forward_prefill() which batches RMSNorm,
        // Q8_1 quantization, and GDN projections across all N tokens.
        // The residual zero fix (da3ce91) ensures clean state transition
        // from forward_prefill() to subsequent forward_token() calls.
        if self.cudarc_weights.borrow().is_some() {
            let n = tokens.len();
            eprintln!("[CUDARC] Batch prefilling {} tokens...", n);
            let prefill_start = std::time::Instant::now();

            let weights_ref = self.cudarc_weights.borrow();
            let weights = weights_ref.as_ref().unwrap();
            let mut graph_ref = self.cudarc_graph.borrow_mut();
            let graph = graph_ref.as_mut().unwrap();
            let mut gdn_states_ref = self.cudarc_gdn_states.borrow_mut();
            let gdn_states = gdn_states_ref.as_mut().unwrap();
            let mut kv_cache_ref = self.cudarc_kv_cache.borrow_mut();
            let kv_cache = kv_cache_ref.as_mut().unwrap();

            let logits_cpu = graph.forward_prefill(
                tokens, weights, gdn_states, kv_cache,
            )?;

            let elapsed = prefill_start.elapsed();
            let tps = n as f64 / elapsed.as_secs_f64();
            eprintln!(
                "[CUDARC] Batch prefill done: {} tokens in {:.2}s ({:.1} tok/s)",
                n, elapsed.as_secs_f64(), tps,
            );

            state.position = kv_cache.pos;

            let logits_tensor = Tensor::from_vec(
                logits_cpu,
                (1, self.config.vocab_size),
                &candle_core::Device::Cpu,
            )?;
            return Ok(logits_tensor);
        }

        let eps = self.config.rms_norm_eps;
        let q_layers = self.q_layers.as_ref().expect(
            "forward_prefill requires a model loaded from GGUF (q_layers must be Some)"
        );
        let embed_tokens = self.embed_tokens.as_ref().expect(
            "forward_prefill requires embed_tokens to be preloaded"
        );
        let lm_head = self.lm_head.as_ref().expect(
            "forward_prefill requires lm_head to be preloaded"
        );

        let n_tokens = tokens.len();

        // 1. Embed all N tokens at once.
        //    embed_tokens lives on CPU; index_select returns [N, hidden_size] on CPU.
        //    We transfer the whole block to the compute device in one PCIe transfer.
        let token_tensor = Tensor::from_vec(
            tokens.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
            n_tokens,
            &Device::Cpu,
        )?;
        let hidden_cpu = embed_tokens.index_select(&token_tensor, 0)?; // [N, hidden_size] CPU
        let mut hidden = hidden_cpu.to_device(&self.device)?;           // [N, hidden_size] GPU

        // 2. Layer loop — each layer processes the full [N, hidden_size] tensor.
        for (il, layer) in q_layers.iter().enumerate() {
            match layer {
                Qwen35LayerQ::Gdn(w) => {
                    hidden = self.prefill_gdn_layer_q(il, w, &hidden, eps, state)?;
                }
                Qwen35LayerQ::Attention(w) => {
                    hidden = self.prefill_attn_layer_q(il, w, &hidden, eps, state)?;
                }
                Qwen35LayerQ::GdnMoE(_w) => {
                    // MoE prefill: sequential per-token (MoE routing is per-token)
                    let mut outputs = Vec::with_capacity(n_tokens);
                    for t in 0..n_tokens {
                        let h_t = hidden.narrow(0, t, 1)?;
                        let out = self.forward_gdn_layer_moe(il, _w, &h_t, eps, state)?;
                        outputs.push(out);
                    }
                    hidden = Tensor::cat(&outputs, 0)?;
                }
                Qwen35LayerQ::AttentionMoE(_w) => {
                    let mut outputs = Vec::with_capacity(n_tokens);
                    for t in 0..n_tokens {
                        let h_t = hidden.narrow(0, t, 1)?;
                        outputs.push(self.forward_attn_layer_moe(il, _w, &h_t, eps, state)?);
                    }
                    hidden = Tensor::cat(&outputs, 0)?;
                }
            }

            // Periodic GPU sync in prefill too (more allocs per layer due to N tokens)
            if il % 5 == 4 {
                if let Device::Cuda(ref cd) = self.device {
                    use candle_core::backend::BackendDevice;
                    let _ = cd.synchronize();
                }
            }

        }

        // 3. Advance position by N (state now reflects full prompt).
        state.advance(n_tokens);

        // 4. Store last hidden for deferred MTP computation.
        let last_h = hidden.narrow(0, n_tokens - 1, 1)?; // [1, hidden_size]
        *self.last_hidden.borrow_mut() = Some(last_h.clone());

        // 5. Output norm on the last token only.
        let h_last = rms_norm(&last_h, &self.output_norm, eps)?;

        // 6. LM head → logits [1, vocab_size] (lm_head may be on CPU).
        let logits = Self::lm_head_forward(lm_head, &h_last)?;

        Ok(logits)
    }

    /// Batch GDN layer prefill: all projections in one QMatMul, sequential recurrence.
    ///
    /// # Arguments
    /// - `il`     — global layer index (used to look up gdn_idx and debug labels)
    /// - `w`      — preloaded GDN layer weights
    /// - `hidden` — `[N, hidden_size]` — all prompt token hidden states
    /// - `eps`    — RMSNorm epsilon
    /// - `state`  — mutable recurrent state (conv + GDN matrix, updated in place)
    ///
    /// # Returns
    /// `[N, hidden_size]` — updated hidden states for all N tokens.
    fn prefill_gdn_layer_q(
        &self,
        il: usize,
        w: &GdnLayerQ,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        let n_group   = self.config.ssm_n_group;         // 16
        let d_state   = self.config.ssm_d_state;         // 128
        let dt_rank   = self.config.ssm_dt_rank;         // 48
        let key_dim   = n_group * d_state;               // 2048
        let value_dim = dt_rank * d_state;               // 6144
        let conv_channels = key_dim * 2 + value_dim;     // 10240
        let conv_kernel   = self.config.ssm_conv_kernel; // 4
        let gdn_idx   = self.config.gdn_index(il).unwrap();
        let n_tokens  = hidden.dim(0)?;

        // ------------------------------------------------------------------
        // Step 1: Batch pre-attention norm + all linear projections.
        //   Each QMatMul::forward accepts [N, in] and produces [N, out].
        // ------------------------------------------------------------------
        let normed = rms_norm(hidden, &w.attn_norm, eps)?; // [N, 5120]

        // QKV mixed: [N, conv_channels=10240]
        let qkv_all = w.attn_qkv.forward(&normed)?;
        // Gate (z):   [N, value_dim=6144]
        let z_all   = w.attn_gate.forward(&normed)?;
        // Beta:  sigmoid(W_beta @ normed) → [N, dt_rank=48]
        let beta_all = sigmoid(&w.ssm_beta.forward(&normed)?)?;
        // Alpha → gate:  softplus(W_alpha @ normed + dt_bias) * ssm_a → [N, dt_rank=48]
        let alpha_proj = w.ssm_alpha.forward(&normed)?;                             // [N, 48]
        let alpha_biased = alpha_proj.broadcast_add(&w.ssm_dt_bias.unsqueeze(0)?)?;   // [N, 48]
        let alpha_sp     = softplus(&alpha_biased)?;                                // [N, 48]
        let gate_all     = alpha_sp.broadcast_mul(&w.ssm_a.unsqueeze(0)?)?;        // [N, 48]

        // ------------------------------------------------------------------
        // Step 2: Batch causal conv1d.
        //   Prepend the saved conv state (last conv_kernel-1 columns of the
        //   previous window) as left padding, then slide the kernel over all
        //   N+3 positions.  We process each output position t in a tight loop —
        //   cheap because it's just element-wise multiply + sum, no QMatMul.
        // ------------------------------------------------------------------

        // conv_state: [1, conv_channels, conv_kernel-1]
        let conv_state_f32 = &state.conv_states[gdn_idx]; // [1, 10240, 3]

        // Reshape qkv_all → [1, conv_channels, N] for concatenation
        let qkv_seq = qkv_all
            .reshape((n_tokens, conv_channels))?  // ensure contiguous layout
            .transpose(0, 1)?                      // [conv_channels, N]
            .unsqueeze(0)?;                        // [1, conv_channels, N]

        // padded: [1, conv_channels, N + conv_kernel - 1]
        let padded = Tensor::cat(&[conv_state_f32, &qkv_seq], 2)?;

        // Slide the depthwise kernel across N output positions.
        // conv_weight (w.ssm_conv1d): [conv_channels, conv_kernel]
        // For position t: window = padded[0, :, t .. t+conv_kernel]  [conv_channels, conv_kernel]
        //   output_t = sum_k(window * kernel)                         [conv_channels]
        let mut conv_outputs: Vec<Tensor> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let window = padded.narrow(2, t, conv_kernel)?;  // [1, conv_channels, conv_kernel]
            let window_2d = window.squeeze(0)?;               // [conv_channels, conv_kernel]
            let out_t = (&window_2d * &w.ssm_conv1d)?         // [conv_channels, conv_kernel]
                .sum(1)?                                       // [conv_channels]
                .unsqueeze(0)?;                                // [1, conv_channels]
            conv_outputs.push(out_t);
        }
        let conv_out = Tensor::cat(&conv_outputs, 0)?; // [N, conv_channels]

        // Save the last (conv_kernel-1) columns of padded as the new conv state.
        let new_conv = padded.narrow(2, n_tokens, conv_kernel - 1)?;  // [1, conv_channels, 3]
        state.conv_states[gdn_idx] = new_conv.contiguous()?;

        // ------------------------------------------------------------------
        // Step 3: SiLU activation.
        // ------------------------------------------------------------------
        let conv_activated = silu_activation(&conv_out)?; // [N, conv_channels=10240]

        // ------------------------------------------------------------------
        // Step 4: Split Q/K/V, L2-norm, tile groups — all on [N, ...] tensors.
        // ------------------------------------------------------------------
        // QKV layout: [Q (key_dim) | K (key_dim) | V (value_dim)]
        let q_raw = conv_activated.narrow(1, 0, key_dim)?;
        let k_raw = conv_activated.narrow(1, key_dim, key_dim)?;
        let v_raw = conv_activated.narrow(1, key_dim * 2, value_dim)?;

        // Reshape to [N, n_group, d_state]
        let q_3d = q_raw.reshape((n_tokens, n_group, d_state))?;
        let k_3d = k_raw.reshape((n_tokens, n_group, d_state))?;

        // L2-normalise along last dim (same as ggml_l2_norm, per head)
        let q_normed = l2_norm(&q_3d, eps)?;  // [N, 16, 128]
        let k_normed = l2_norm(&k_3d, eps)?;  // [N, 16, 128]

        // Tile from n_group (16) to dt_rank (48) using ggml_repeat_4d convention
        let repeats = dt_rank / n_group; // 3
        let q_expanded = crate::prefill::tile_groups(&q_normed, repeats)?; // [N, 48, 128]
        let k_expanded = crate::prefill::tile_groups(&k_normed, repeats)?; // [N, 48, 128]

        // V: reshape to [N, dt_rank, d_state]
        let v_3d = v_raw.reshape((n_tokens, dt_rank, d_state))?; // [N, 48, 128]

        // Scale q by 1/sqrt(S_k)
        let scale = 1.0 / (d_state as f64).sqrt();
        let q_scaled = (&q_expanded * scale)?; // [N, 48, 128]

        // ------------------------------------------------------------------
        // Step 5: Sequential DeltaNet recurrence.
        //   Cheap: no QMatMul, only small tensor ops (matmul on 128×128 slices).
        //   We extract per-token slices from the batched tensors, then run the
        //   same recurrence as forward_gdn_layer_q.
        // ------------------------------------------------------------------

        // Load GDN state (S^T convention)
        let mut s_t = state.gdn_states[gdn_idx].copy()?; // [1, 48, 128, 128]

        let mut outputs: Vec<Tensor> = Vec::with_capacity(n_tokens);

        for t in 0..n_tokens {
            // Extract per-token slices: [1, 48, 128]
            let q_t     = q_scaled.narrow(0, t, 1)?;    // [1, 48, 128]
            let k_t     = k_expanded.narrow(0, t, 1)?;  // [1, 48, 128]
            let v_t     = v_3d.narrow(0, t, 1)?;        // [1, 48, 128]
            let beta_t  = beta_all.narrow(0, t, 1)?;    // [1, 48]
            let gate_t  = gate_all.narrow(0, t, 1)?;    // [1, 48]

            // Step 1: decay — s = s * exp(gate)
            let gate_exp = gate_t.exp()?;                        // [1, 48]
            let gate_4d  = gate_exp.unsqueeze(2)?.unsqueeze(3)?; // [1, 48, 1, 1]
            let s_decayed = s_t.broadcast_mul(&gate_4d)?;        // [1, 48, 128, 128]

            // Step 2: sk = S^T @ k  (S^T is already in memory)
            let k_col = k_t.unsqueeze(3)?;                       // [1, 48, 128, 1]
            let sk    = s_decayed.transpose(2, 3)?.matmul(&k_col)?.squeeze(3)?; // [1, 48, 128]

            // Step 3: delta = (v - sk) * beta
            let delta_raw = (&v_t - &sk)?;                       // [1, 48, 128]
            let beta_3d   = beta_t.unsqueeze(2)?;                // [1, 48, 1]
            let delta     = delta_raw.broadcast_mul(&beta_3d)?;  // [1, 48, 128]

            // Step 4: outer product kd = k ⊗ delta, add to state
            let k_col_outer = k_t.unsqueeze(3)?;                 // [1, 48, 128, 1]
            let d_row       = delta.unsqueeze(2)?;               // [1, 48, 1, 128]
            let kd          = k_col_outer.broadcast_mul(&d_row)?;// [1, 48, 128, 128]
            let s_t_new     = (&s_decayed + &kd)?;               // [1, 48, 128, 128]

            // Step 5: readout o = S_new^T @ q
            let q_col = q_t.unsqueeze(3)?;                       // [1, 48, 128, 1]
            let output_t = s_t_new.transpose(2, 3)?.matmul(&q_col)?.squeeze(3)?; // [1, 48, 128]

            s_t = s_t_new;
            outputs.push(output_t);
        }

        // Save updated GDN state (F32 → F16 for storage)
        state.gdn_states[gdn_idx] = s_t;

        // Concatenate per-token recurrence outputs: [N, 48, 128]
        let output_seq = Tensor::cat(&outputs, 0)?; // [N, 48, 128]

        // ------------------------------------------------------------------
        // Step 6: Batch output gating + ssm_out projection.
        // ------------------------------------------------------------------
        // RMSNorm on [N, 48, 128] — mean_keepdim along last dim works correctly.
        let normed_out  = rms_norm(&output_seq, &w.ssm_norm, eps)?; // [N, 48, 128]
        let normed_flat = normed_out.reshape((n_tokens, value_dim))?; // [N, 6144]
        let gated       = (&normed_flat * &silu_activation(&z_all)?)?; // [N, 6144]

        // ssm_out projection: QMatMul::forward accepts [N, 6144] → [N, 5120]
        let projected = w.ssm_out.forward(&gated)?;   // [N, 5120]
        let h_mid     = (hidden + &projected)?;        // [N, 5120] residual

        // ------------------------------------------------------------------
        // Step 7: Batch post-attention norm + FFN.
        // ------------------------------------------------------------------
        let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;
        let ffn_out    = self.swiglu_ffn_q(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;
        let h_out      = (&h_mid + &ffn_out)?;         // [N, 5120]

        Ok(h_out)
    }

    /// Batch attention layer prefill: full causal self-attention over all N positions.
    ///
    /// # Arguments
    /// - `il`     — global layer index
    /// - `w`      — preloaded attention layer weights
    /// - `hidden` — `[N, hidden_size]`
    /// - `eps`    — RMSNorm epsilon
    /// - `state`  — mutable state (KV cache populated, position used for MRoPE)
    ///
    /// # Returns
    /// `[N, hidden_size]` — updated hidden states.
    fn prefill_attn_layer_q(
        &self,
        il: usize,
        w: &AttnLayerQ,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        let num_heads    = self.config.num_attention_heads; // 24
        let num_kv_heads = self.config.num_kv_heads;        // 4
        let head_dim     = self.config.head_dim;            // 256
        let attn_idx     = self.config.attn_index(il).unwrap();
        let n_tokens     = hidden.dim(0)?;

        // ------------------------------------------------------------------
        // Step 1: Batch projections — QMatMul accepts [N, hidden_size].
        // ------------------------------------------------------------------
        let normed = rms_norm(hidden, &w.attn_norm, eps)?;

        // Q+gate interleaved: [N, num_heads * 2 * head_dim]
        let q_full  = w.wq.forward(&normed)?;
        // K: [N, num_kv_heads * head_dim]
        let k_proj  = w.wk.forward(&normed)?;
        // V: [N, num_kv_heads * head_dim]
        let v_proj  = w.wv.forward(&normed)?;

        // Unpack Q and Q-gate (interleaved per head):
        //   [N, num_heads * 2 * head_dim] → [N, num_heads, 2 * head_dim]
        let q_full_3d = q_full.reshape((n_tokens, num_heads, 2 * head_dim))?;
        let q_raw     = q_full_3d.narrow(2, 0, head_dim)?;           // [N, num_heads, head_dim]
        let q_gate_raw = q_full_3d.narrow(2, head_dim, head_dim)?;   // [N, num_heads, head_dim]

        // ------------------------------------------------------------------
        // Step 2: QK-norm per head — rms_norm works on [N, heads, head_dim].
        // ------------------------------------------------------------------
        let q = rms_norm(&q_raw, &w.q_norm, eps)?;                    // [N, num_heads, head_dim]
        let k = k_proj.reshape((n_tokens, num_kv_heads, head_dim))?;
        let k = rms_norm(&k, &w.k_norm, eps)?;                        // [N, num_kv_heads, head_dim]
        let v = v_proj.reshape((n_tokens, num_kv_heads, head_dim))?;  // [N, num_kv_heads, head_dim]

        // ------------------------------------------------------------------
        // Step 3: Apply MRoPE to all N positions at once.
        //   MRoPE::apply expects [seq_len, num_heads, head_dim].
        //   Here seq_len = N, reusing the same tensor layout.
        // ------------------------------------------------------------------
        let positions = crate::rope::MRoPE::text_positions(n_tokens, state.position);
        let q_rotated = self.mrope.apply(&q, &positions)?;            // [N, num_heads, head_dim]
        let k_rotated = self.mrope.apply(&k, &positions)?;            // [N, num_kv_heads, head_dim]

        // ------------------------------------------------------------------
        // Step 4: Build KV cache from prefill keys/values.
        //   Reshape to cache format [1, heads, N, head_dim], then cat with any
        //   existing cache entries (empty at start, non-empty on subsequent calls).
        // ------------------------------------------------------------------
        // [N, num_kv_heads, head_dim] → [1, num_kv_heads, N, head_dim]
        // k_rotated is [N, num_kv_heads, head_dim]
        // Transpose to [num_kv_heads, N, head_dim] then unsqueeze batch:
        let k_for_cache = k_rotated
            .transpose(0, 1)?                              // [num_kv_heads, N, head_dim]
            .unsqueeze(0)?;                                // [1, num_kv_heads, N, head_dim]
        let v_for_cache = v
            .transpose(0, 1)?                              // [num_kv_heads, N, head_dim]
            .unsqueeze(0)?;                                // [1, num_kv_heads, N, head_dim]

        // Append to existing KV cache (may be empty if this is the first prefill)
        let (k_cache, v_cache) = state.kv_append(attn_idx, &k_for_cache, &v_for_cache)?;
        let cached_seq_len = k_cache.dim(2)?; // total sequence length including prior context

        // ------------------------------------------------------------------
        // Step 5: GQA expansion — repeat KV heads to match query heads.
        //   k_cache: [1, num_kv_heads, cached_seq_len, head_dim]
        //   → expand → [1, num_heads, cached_seq_len, head_dim]
        // ------------------------------------------------------------------
        let group_size = num_heads / num_kv_heads;
        let k_exp = k_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;
        let v_exp = v_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;

        // ------------------------------------------------------------------
        // Step 6: Causal scaled dot-product attention.
        //   q_rotated: [N, num_heads, head_dim]
        //   Reshape to [1, num_heads, N, head_dim] for batched matmul.
        // ------------------------------------------------------------------
        // q_for_attn: [1, num_heads, N, head_dim]
        let q_for_attn = q_rotated
            .transpose(0, 1)?           // [num_heads, N, head_dim]
            .unsqueeze(0)?              // [1, num_heads, N, head_dim]
            .contiguous()?;

        // scores: [1, num_heads, N, cached_seq_len]
        let attn_scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k_exp.transpose(2, 3)?.contiguous()?;
        let scores = q_for_attn.matmul(&k_t)?;
        let scores = (scores * attn_scale)?;

        // Apply causal mask over the NEW N tokens attending to the full cached context.
        // If there is prior context (len = cached_seq_len - n_tokens), we only need to
        // mask the last N rows (current tokens) against the last N columns (current tokens).
        // Tokens in the prior context don't need masking (they don't appear in Q).
        // We build a [N, cached_seq_len] mask:
        //   - all prior context columns are visible (0.0)
        //   - among new token columns, upper triangle is masked (-inf)
        let prior_len = cached_seq_len - n_tokens;
        let device = hidden.device();
        let mask_new = crate::prefill::create_causal_mask(n_tokens, device)?; // [N, N]
        let full_mask = if prior_len > 0 {
            // Prepend a [N, prior_len] block of zeros (all prior positions visible)
            let prior_zeros = Tensor::zeros((n_tokens, prior_len), candle_core::DType::F32, device)?;
            Tensor::cat(&[&prior_zeros, &mask_new], 1)? // [N, cached_seq_len]
        } else {
            mask_new // [N, N] == [N, cached_seq_len] when no prior context
        };
        // Reshape for broadcast: [1, 1, N, cached_seq_len]
        let full_mask = full_mask.unsqueeze(0)?.unsqueeze(0)?;

        let scores = scores.broadcast_add(&full_mask)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?; // [1, num_heads, N, cached_seq_len]

        // Attention output: [1, num_heads, N, head_dim]
        let attn_out = attn_weights.matmul(&v_exp)?;

        // Reshape back to [N, num_heads, head_dim]
        let attn_out = attn_out.squeeze(0)?                // [num_heads, N, head_dim]
            .transpose(0, 1)?;                             // [N, num_heads, head_dim]

        // ------------------------------------------------------------------
        // Step 7: Apply gate (sigmoid) + output projection.
        // ------------------------------------------------------------------
        let gate  = sigmoid(&q_gate_raw)?;                 // [N, num_heads, head_dim]
        let gated = (&attn_out * &gate)?;                  // [N, num_heads, head_dim]
        let gated_flat = gated.reshape((n_tokens, num_heads * head_dim))?;

        // Output projection via QMatMul
        let attn_projected = w.wo.forward(&gated_flat)?;   // [N, hidden_size]

        // Residual
        let h_mid = (hidden + &attn_projected)?;

        // ------------------------------------------------------------------
        // Step 8: Batch post-attention norm + FFN.
        // ------------------------------------------------------------------
        let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;
        let ffn_out    = self.swiglu_ffn_q(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;
        let h_out      = (&h_mid + &ffn_out)?;

        Ok(h_out)
    }
}

// ---------------------------------------------------------------------------
// ChimereModel trait impl (Step 1 of multi-arch refactor)
// ---------------------------------------------------------------------------
//
// This block makes Qwen35Model satisfy the model-agnostic `ChimereModel` trait
// by delegating every method to the existing inherent implementations above.
// **Zero behavioral change**: callers that hold a `&Qwen35Model` are unaffected,
// and at this step nothing in the codebase invokes the trait yet — it just has
// to compile cleanly. Steps 2-5 of the refactor migrate `generate.rs`,
// `mtp_scheduler.rs`, `block_generate.rs` and `server.rs` to call through the
// trait. Capability flags here advertise everything Qwen3.5 supports today
// (MTP, block diffusion, DART, entropy routing).

impl crate::chimere_model::ChimereModel for Qwen35Model {
    fn arch(&self) -> crate::chimere_model::ModelArch {
        crate::chimere_model::ModelArch::Qwen35A3B
    }

    fn num_layers(&self) -> usize {
        // Forward to the existing inherent method (defined above in this impl).
        Qwen35Model::num_layers(self)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn supports_mtp(&self) -> bool {
        // Mirrors the existing `has_mtp()` accessor.
        self.has_mtp()
    }

    fn supports_block_diffusion(&self) -> bool {
        true
    }

    fn supports_dart(&self) -> bool {
        // DART verification only works when the libllama FFI backend owns the
        // forward pass (it provides the multi-token decode primitive).
        self.has_mtp()
    }

    fn supports_entropy_routing(&self) -> bool {
        true
    }

    fn forward_token(
        &self,
        token: u32,
        state: &mut crate::chimere_model::InferenceState<'_>,
    ) -> candle_core::Result<crate::chimere_model::ForwardOutput> {
        let gdn = state.as_gdn_mut()?;
        let (logits, mtp_logits) = Qwen35Model::forward_token(self, token, gdn)?;
        Ok(crate::chimere_model::ForwardOutput { logits, mtp_logits })
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &mut crate::chimere_model::InferenceState<'_>,
    ) -> candle_core::Result<crate::chimere_model::ForwardOutput> {
        let gdn = state.as_gdn_mut()?;
        let logits = Qwen35Model::forward_prefill(self, tokens, gdn)?;
        Ok(crate::chimere_model::ForwardOutput {
            logits,
            mtp_logits: None,
        })
    }

    fn reset_for_new_request(&self) {
        // Mirrors the order used today by `server.rs::run_inference`: libllama
        // first (no-op if FFI backend not active), then cudarc (no-op if cudarc
        // backend not active). Only one of the two is ever live in production.
        self.reset_llama_state();
        self.reset_cudarc_state();
    }

    // -- libllama FFI hooks: forward to existing inherent methods ---------

    fn llama_forward_active(&self) -> bool {
        Qwen35Model::llama_forward_active(self)
    }

    fn llama_forward_mut(
        &self,
    ) -> Option<std::cell::RefMut<'_, Option<crate::llama_backend::LlamaForward>>> {
        Some(Qwen35Model::llama_forward_mut(self))
    }

    fn llama_set_logit_bias(&self, token_id: u32, bias: f32) {
        Qwen35Model::llama_set_logit_bias(self, token_id, bias);
    }

    fn llama_set_engram_bias(&self, predictions: &[(u32, f32)]) {
        Qwen35Model::llama_set_engram_bias(self, predictions);
    }

    fn llama_clear_engram_bias(&self) {
        Qwen35Model::llama_clear_engram_bias(self);
    }

    fn take_last_packed_logprobs(&self) -> Option<Vec<f32>> {
        Qwen35Model::take_last_packed_logprobs(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_gguf_path() -> String {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home
        )
    }

    fn skip_if_missing(path: &str) -> bool {
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: GGUF file not found at {}", path);
            true
        } else {
            false
        }
    }

    /// Test that we can open the GGUF, parse config, and verify layer structure.
    ///
    /// This does NOT load tensor data (which would require IQ3_S dequant),
    /// it only validates the config and tensor existence checks.
    #[test]
    fn test_qwen35_model_config() {
        use crate::weight_loader::Qwen35WeightLoader;

        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }

        let loader = Qwen35WeightLoader::from_gguf(&path)
            .expect("Failed to open GGUF for config");
        let cfg = loader.config();

        // Verify layer counts
        assert_eq!(cfg.num_main_layers, 64, "Expected 64 main layers");
        assert_eq!(cfg.num_total_layers, 65, "Expected 65 total layers (64 + 1 MTP)");
        assert_eq!(cfg.num_gdn_layers(), 48, "Expected 48 GDN layers");
        assert_eq!(cfg.num_attn_layers(), 16, "Expected 16 attention layers");
        assert_eq!(cfg.nextn_predict_layers, 1, "Expected 1 MTP layer");

        // Verify layer classification for first 8 layers
        for i in 0..8 {
            let expected_attn = (i + 1) % 4 == 0;
            assert_eq!(
                cfg.is_attention(i), expected_attn,
                "Layer {} attention classification mismatch", i
            );
        }

        // Verify MTP tensor existence
        assert!(loader.has_tensor("blk.64.nextn.eh_proj.weight"));
        assert!(loader.has_tensor("blk.64.nextn.enorm.weight"));
        assert!(loader.has_tensor("blk.64.nextn.hnorm.weight"));
        assert!(loader.has_tensor("blk.64.nextn.shared_head_norm.weight"));

        println!(
            "Qwen35Model config: {} main layers ({} GDN + {} attn), {} MTP, vocab={}",
            cfg.num_main_layers, cfg.num_gdn_layers(), cfg.num_attn_layers(),
            cfg.nextn_predict_layers, cfg.vocab_size
        );
    }

    /// Test loading the model from GGUF with preloaded quantized weights.
    ///
    /// Verifies that `from_gguf` loads all weights as QMatMul and stores them.
    #[test]
    fn test_qwen35_model_load() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }

        let model = Qwen35Model::from_gguf(&path, &Device::Cpu, None)
            .expect("Failed to load model");

        // All 64 layers should be preloaded
        assert_eq!(model.num_layers(), 64, "Expected 64 layers preloaded");

        // MTP should be detected
        assert!(model.has_mtp(), "Expected MTP head to be detected");

        // Verify norms are loaded
        assert_eq!(model.output_norm.dims(), &[5120], "output_norm shape");
        assert!(model.mtp_enorm.is_some(), "mtp_enorm should be loaded");
        assert!(model.mtp_hnorm.is_some(), "mtp_hnorm should be loaded");
        assert!(model.mtp_shared_head_norm.is_some(), "mtp_shared_head_norm should be loaded");

        // Verify preloaded weights exist
        assert!(model.embed_tokens.is_some(), "embed_tokens should be preloaded");
        assert!(model.q_layers.is_some(), "q_layers should be preloaded");
        assert!(model.lm_head.is_some(), "lm_head should be preloaded");
        assert!(model.mtp_head.is_some(), "mtp_head should be preloaded");

        println!("Qwen35Model loaded (preloaded): {} layers, mtp={}", model.num_layers(), model.has_mtp());
    }

    /// Test forward pass shape with preloaded quantized weights.
    ///
    /// Forward one token through all 64 layers, verify logits shape [1, 248320].
    /// With preloaded weights, this should be FAST (pure GPU, no CPU dequant).
    #[test]
    fn test_qwen35_forward_shape_gpu() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }

        // Use GPU if available, fallback to CPU
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        eprintln!("[TEST] Using device: {:?}", device);

        let model = Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");

        let mut state = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        let fwd_start = std::time::Instant::now();
        let (logits, mtp) = model.forward_token(1, &mut state)
            .expect("Forward pass failed");
        let fwd_time = fwd_start.elapsed();

        assert_eq!(
            logits.dims(), &[1, 248320],
            "Logits shape mismatch: expected [1, 248320], got {:?}", logits.dims()
        );

        // MTP should also produce logits
        assert!(mtp.is_some(), "MTP logits should be produced");
        if let Some(mtp_logits) = &mtp {
            assert_eq!(
                mtp_logits.dims(), &[1, 248320],
                "MTP logits shape mismatch: expected [1, 248320], got {:?}", mtp_logits.dims()
            );
        }

        println!("Forward pass: logits shape = {:?}, mtp = {:?}, time = {:.2}s",
            logits.dims(), mtp.as_ref().map(|t| t.dims().to_vec()), fwd_time.as_secs_f64());
    }

    /// Test that the forward pass is deterministic (no randomness).
    ///
    /// Same token twice = same logits.
    #[test]
    fn test_qwen35_forward_deterministic() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let model = Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");

        // Run forward twice with fresh state
        let mut state1 = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");
        let (logits1, _) = model.forward_token(42, &mut state1)
            .expect("Forward pass 1 failed");

        let mut state2 = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");
        let (logits2, _) = model.forward_token(42, &mut state2)
            .expect("Forward pass 2 failed");

        let diff = (&logits1 - &logits2).unwrap().abs().unwrap()
            .sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            diff < 1e-4,
            "Forward pass should be deterministic, but diff = {}", diff
        );

        println!("Determinism verified: diff = {:.2e}", diff);
    }

    /// Test that the model structure can be built with synthetic F32 tensors.
    ///
    /// This bypasses GGUF loading entirely and constructs a tiny model with
    /// known shapes to verify the forward pass logic without needing
    /// IQ3_S dequantization.
    #[test]
    fn test_qwen35_forward_synthetic() -> Result<()> {
        let device = Device::Cpu;

        // Tiny synthetic config for testing
        let hidden = 32usize;
        let intermediate = 64usize;
        let vocab = 100usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = hidden / num_heads; // 8
        let eps = 1e-6f64;

        // Embedding [vocab, hidden]
        let embed_tokens = Tensor::randn(0f32, 0.02, (vocab, hidden), &device)?;
        let output_norm = Tensor::ones((hidden,), candle_core::DType::F32, &device)?;
        let lm_head = Tensor::randn(0f32, 0.02, (vocab, hidden), &device)?;

        // GDN layer 0
        let gdn_weights = GdnLayerWeights {
            attn_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            post_attention_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            attn_qkv: Tensor::randn(0f32, 0.02, (hidden * 2, hidden), &device)?,
            attn_gate: Tensor::randn(0f32, 0.02, (hidden, hidden), &device)?,
            ssm_a: Tensor::ones((num_heads,), candle_core::DType::F32, &device)?,
            ssm_conv1d: Tensor::ones((4, hidden), candle_core::DType::F32, &device)?,
            ssm_dt_bias: Tensor::zeros((num_heads,), candle_core::DType::F32, &device)?,
            ssm_beta: Tensor::randn(0f32, 0.02, (hidden, num_heads), &device)?,
            ssm_alpha: Tensor::randn(0f32, 0.02, (hidden, num_heads), &device)?,
            ssm_norm: Tensor::ones((head_dim,), candle_core::DType::F32, &device)?,
            ssm_out: Tensor::randn(0f32, 0.02, (hidden, hidden), &device)?,
            ffn_gate: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_up: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_down: Tensor::randn(0f32, 0.02, (hidden, intermediate), &device)?,
        };

        // Attention layer 1
        let q_out = num_heads * head_dim * 2; // Q + gate fused
        let kv_out = num_kv_heads * head_dim;
        let attn_weights = AttnLayerWeights {
            attn_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            post_attention_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            wq: Tensor::randn(0f32, 0.02, (q_out, hidden), &device)?,
            wk: Tensor::randn(0f32, 0.02, (kv_out, hidden), &device)?,
            wv: Tensor::randn(0f32, 0.02, (kv_out, hidden), &device)?,
            wo: Tensor::randn(0f32, 0.02, (hidden, num_heads * head_dim), &device)?,
            q_norm: Tensor::ones((head_dim,), candle_core::DType::F32, &device)?,
            k_norm: Tensor::ones((head_dim,), candle_core::DType::F32, &device)?,
            ffn_gate: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_up: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_down: Tensor::randn(0f32, 0.02, (hidden, intermediate), &device)?,
        };

        let layers = vec![
            Qwen35Layer::Gdn(gdn_weights),
            Qwen35Layer::Attention(attn_weights),
        ];

        let config = Qwen35Config {
            hidden_size: hidden,
            num_attention_heads: num_heads,
            num_kv_heads: num_kv_heads,
            head_dim: head_dim,
            num_main_layers: 2,
            num_total_layers: 2,
            intermediate_size: intermediate,
            vocab_size: vocab,
            rms_norm_eps: eps,
            ssm_d_state: 16,
            ssm_d_inner: hidden,
            ssm_dt_rank: num_heads,
            ssm_n_group: 2,
            ssm_conv_kernel: 4,
            full_attn_interval: 2, // layer 1 is attention ((1+1)%2==0)
            rope_theta: 10_000_000.0,
            rope_sections: [2, 2, 0, 0],
            nextn_predict_layers: 0,
            // Dense test model — no MoE
            num_experts: 0,
            experts_per_token: 0,
            expert_ffn_hidden: 0,
            shared_expert_ffn_hidden: 0,
        };

        let n_rot = config.rope_sections.iter().sum::<usize>() * 2;
        let mrope = MRoPE::new(head_dim, n_rot, &config.rope_sections, config.rope_theta);

        let model = Qwen35Model::synthetic(
            config,
            embed_tokens,
            layers,
            output_norm,
            lm_head,
            None,
            mrope,
        );

        // Create a minimal state
        let mut state = GdnRecurrentState::new(&model.config, &device)?;

        // Forward a token
        let (logits, mtp_logits) = model.forward_token(5, &mut state)?;

        // Verify shapes
        assert_eq!(
            logits.dims(), &[1, vocab],
            "Logits shape: expected [1, {}], got {:?}", vocab, logits.dims()
        );
        assert!(mtp_logits.is_none(), "No MTP head, should be None");

        // Verify determinism: same token -> same logits
        let mut state2 = GdnRecurrentState::new(&model.config, &device)?;
        let (logits2, _) = model.forward_token(5, &mut state2)?;
        let diff = (&logits - &logits2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "Forward should be deterministic, diff={}", diff);

        // Verify no NaN
        let sum: f32 = logits.sum_all()?.to_scalar()?;
        assert!(sum.is_finite(), "Logits contain NaN or Inf");

        println!(
            "Synthetic forward pass: logits {:?}, sum={:.4}, deterministic (diff={:.2e})",
            logits.dims(), sum, diff
        );
        Ok(())
    }

    /// Test that the synthetic model with MTP produces MTP logits.
    #[test]
    fn test_qwen35_forward_synthetic_mtp() -> Result<()> {
        let device = Device::Cpu;

        let hidden = 32usize;
        let intermediate = 64usize;
        let vocab = 100usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = hidden / num_heads;
        let eps = 1e-6f64;

        let embed_tokens = Tensor::randn(0f32, 0.02, (vocab, hidden), &device)?;
        let output_norm = Tensor::ones((hidden,), candle_core::DType::F32, &device)?;
        let lm_head = Tensor::randn(0f32, 0.02, (vocab, hidden), &device)?;

        // Just one GDN layer
        let gdn_weights = GdnLayerWeights {
            attn_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            post_attention_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            attn_qkv: Tensor::randn(0f32, 0.02, (hidden * 2, hidden), &device)?,
            attn_gate: Tensor::randn(0f32, 0.02, (hidden, hidden), &device)?,
            ssm_a: Tensor::ones((num_heads,), candle_core::DType::F32, &device)?,
            ssm_conv1d: Tensor::ones((4, hidden), candle_core::DType::F32, &device)?,
            ssm_dt_bias: Tensor::zeros((num_heads,), candle_core::DType::F32, &device)?,
            ssm_beta: Tensor::randn(0f32, 0.02, (hidden, num_heads), &device)?,
            ssm_alpha: Tensor::randn(0f32, 0.02, (hidden, num_heads), &device)?,
            ssm_norm: Tensor::ones((head_dim,), candle_core::DType::F32, &device)?,
            ssm_out: Tensor::randn(0f32, 0.02, (hidden, hidden), &device)?,
            ffn_gate: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_up: Tensor::randn(0f32, 0.02, (intermediate, hidden), &device)?,
            ffn_down: Tensor::randn(0f32, 0.02, (hidden, intermediate), &device)?,
        };

        // MTP head
        let mtp = MtpHead {
            eh_proj: Tensor::randn(0f32, 0.02, (hidden, 2 * hidden), &device)?,
            enorm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            hnorm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
            shared_head_norm: Tensor::ones((hidden,), candle_core::DType::F32, &device)?,
        };

        let config = Qwen35Config {
            hidden_size: hidden,
            num_attention_heads: num_heads,
            num_kv_heads: num_kv_heads,
            head_dim: head_dim,
            num_main_layers: 1,
            num_total_layers: 2,
            intermediate_size: intermediate,
            vocab_size: vocab,
            rms_norm_eps: eps,
            ssm_d_state: 16,
            ssm_d_inner: hidden,
            ssm_dt_rank: num_heads,
            ssm_n_group: 2,
            ssm_conv_kernel: 4,
            full_attn_interval: 4,
            rope_theta: 10_000_000.0,
            rope_sections: [2, 2, 0, 0],
            nextn_predict_layers: 1,
            // Dense test model — no MoE
            num_experts: 0,
            experts_per_token: 0,
            expert_ffn_hidden: 0,
            shared_expert_ffn_hidden: 0,
        };

        let n_rot = config.rope_sections.iter().sum::<usize>() * 2;
        let mrope = MRoPE::new(head_dim, n_rot, &config.rope_sections, config.rope_theta);

        let model = Qwen35Model::synthetic(
            config,
            embed_tokens,
            vec![Qwen35Layer::Gdn(gdn_weights)],
            output_norm,
            lm_head,
            Some(mtp),
            mrope,
        );

        let mut state = GdnRecurrentState::new(&model.config, &device)?;
        let (logits, mtp_logits) = model.forward_token(10, &mut state)?;

        assert_eq!(logits.dims(), &[1, vocab]);
        assert!(mtp_logits.is_some(), "MTP head present, should produce logits");
        let mtp_logits = mtp_logits.unwrap();
        assert_eq!(mtp_logits.dims(), &[1, vocab], "MTP logits shape mismatch");

        // Both should be finite
        let sum1: f32 = logits.sum_all()?.to_scalar()?;
        let sum2: f32 = mtp_logits.sum_all()?.to_scalar()?;
        assert!(sum1.is_finite(), "Main logits contain NaN/Inf");
        assert!(sum2.is_finite(), "MTP logits contain NaN/Inf");

        println!(
            "Synthetic MTP forward: main={:?} (sum={:.4}), mtp={:?} (sum={:.4})",
            logits.dims(), sum1, mtp_logits.dims(), sum2
        );
        Ok(())
    }

    /// Test that loading F32 tensors from the real GGUF works correctly.
    /// These are the norm/bias tensors that don't require IQ3_S dequant.
    #[test]
    fn test_qwen35_load_f32_tensors() -> Result<()> {
        let device = Device::Cpu;
        use crate::weight_loader::Qwen35WeightLoader;

        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return Ok(());
        }

        let loader = Qwen35WeightLoader::from_gguf(&path)
            .map_err(candle_core::Error::Msg)?;

        // All F32 tensors from layer 0 (GDN)
        let attn_norm = loader.attn_norm(0, &device)?;
        assert_eq!(attn_norm.dims(), &[5120], "attn_norm shape");

        let post_norm = loader.attn_post_norm(0, &device)?;
        assert_eq!(post_norm.dims(), &[5120], "post_attention_norm shape");

        let ssm_a = loader.ssm_a(0, &device)?;
        assert_eq!(ssm_a.dims(), &[48], "ssm_a shape");

        let ssm_conv1d = loader.ssm_conv1d(0, &device)?;
        // GGUF dims [4, 10240] reversed -> [10240, 4]
        assert_eq!(ssm_conv1d.dims(), &[10240, 4], "ssm_conv1d shape");

        let ssm_dt_bias = loader.ssm_dt_bias(0, &device)?;
        assert_eq!(ssm_dt_bias.dims(), &[48], "ssm_dt_bias shape");

        let ssm_norm = loader.ssm_norm(0, &device)?;
        assert_eq!(ssm_norm.dims(), &[128], "ssm_norm shape");

        // F32 tensors from layer 3 (attention)
        let q_norm = loader.attn_q_norm(3, &device)?;
        assert_eq!(q_norm.dims(), &[256], "q_norm shape");

        let k_norm = loader.attn_k_norm(3, &device)?;
        assert_eq!(k_norm.dims(), &[256], "k_norm shape");

        // Global F32 tensors
        let output_norm = loader.output_norm(&device)?;
        assert_eq!(output_norm.dims(), &[5120], "output_norm shape");

        // MTP F32 tensors
        let mtp_enorm = loader.mtp_enorm(&device)?;
        assert_eq!(mtp_enorm.dims(), &[5120], "mtp_enorm shape");

        let mtp_hnorm = loader.mtp_hnorm(&device)?;
        assert_eq!(mtp_hnorm.dims(), &[5120], "mtp_hnorm shape");

        let mtp_shared_head_norm = loader.mtp_shared_head_norm(&device)?;
        assert_eq!(mtp_shared_head_norm.dims(), &[5120], "mtp_shared_head_norm shape");

        // All loaded values should be finite
        for (name, tensor) in &[
            ("attn_norm", &attn_norm),
            ("post_norm", &post_norm),
            ("ssm_a", &ssm_a),
            ("ssm_dt_bias", &ssm_dt_bias),
            ("ssm_norm", &ssm_norm),
            ("q_norm", &q_norm),
            ("k_norm", &k_norm),
            ("output_norm", &output_norm),
        ] {
            let sum: f32 = tensor.sum_all()?.to_scalar()?;
            assert!(sum.is_finite(), "{} contains NaN/Inf", name);
        }

        println!("All F32 tensors loaded and verified from real GGUF");
        Ok(())
    }

    /// Test: compare QMatMul output vs manual F32 matmul on same weights.
    /// This verifies candle's quantized matmul produces correct results.
    ///
    /// Run with: CHIMERE_DEBUG=1 cargo test test_qmatmul_vs_f32 -- --nocapture
    #[test]
    fn test_qmatmul_vs_f32() -> Result<()> {
        if std::env::var("CHIMERE_DEBUG").is_err() {
            eprintln!("[QMATMUL_TEST] Set CHIMERE_DEBUG=1 to run");
            return Ok(());
        }

        let path = test_gguf_path();
        if skip_if_missing(&path) { return Ok(()); }

        let device = candle_core::Device::Cpu;
        let gguf = crate::gguf_loader::GgufFile::open(&path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Load attn_qkv for layer 0 as BOTH QMatMul and F32 tensor
        let qmatmul = gguf.load_qmatmul("blk.0.attn_qkv.weight", &device)?;
        let f32_tensor = gguf.load_tensor("blk.0.attn_qkv.weight", &device)?;

        eprintln!("[QMATMUL_TEST] QMatMul loaded for blk.0.attn_qkv.weight");
        eprintln!("[QMATMUL_TEST] F32 tensor shape: {:?}", f32_tensor.dims());

        // Load embedding for token 248045
        let embed = gguf.load_tensor("token_embd.weight", &device)?;
        let token_tensor = Tensor::new(&[248045u32], &device)?;
        let hidden = embed.index_select(&token_tensor, 0)?; // [1, 5120]
        eprintln!("[QMATMUL_TEST] hidden shape: {:?}", hidden.dims());

        // Method 1: QMatMul
        let qmatmul_result = qmatmul.forward(&hidden)?;
        eprintln!("[QMATMUL_TEST] QMatMul result shape: {:?}", qmatmul_result.dims());

        // Method 2: Manual F32 matmul
        // f32_tensor has candle shape [out_dim, in_dim] after dim reversal
        // x @ f32_tensor.t() = [1, in_dim] @ [in_dim, out_dim] = [1, out_dim]
        let f32_result = hidden.matmul(&f32_tensor.t()?)?;
        eprintln!("[QMATMUL_TEST] F32 matmul result shape: {:?}", f32_result.dims());

        // Compare
        let q_vec: Vec<f32> = qmatmul_result.flatten_all()?.to_vec1()?;
        let f_vec: Vec<f32> = f32_result.flatten_all()?.to_vec1()?;
        assert_eq!(q_vec.len(), f_vec.len(), "output sizes differ");

        let mut max_diff: f32 = 0.0;
        let mut sum_diff: f64 = 0.0;
        for i in 0..q_vec.len() {
            let d = (q_vec[i] - f_vec[i]).abs();
            if d > max_diff { max_diff = d; }
            sum_diff += d as f64;
        }
        let avg_diff = sum_diff / q_vec.len() as f64;

        eprintln!("[QMATMUL_TEST] Comparison ({} elements):", q_vec.len());
        eprintln!("[QMATMUL_TEST]   max_diff = {:.6}", max_diff);
        eprintln!("[QMATMUL_TEST]   avg_diff = {:.6}", avg_diff);
        eprintln!("[QMATMUL_TEST]   QMatMul first-10: {:?}", &q_vec[..10]);
        eprintln!("[QMATMUL_TEST]   F32     first-10: {:?}", &f_vec[..10]);

        // Also compare element 5000 and 10000
        for idx in [0, 100, 1000, 5000, 10000].iter() {
            if *idx < q_vec.len() {
                eprintln!("[QMATMUL_TEST]   [{:5}] Q={:.6} F={:.6} diff={:.6}",
                    idx, q_vec[*idx], f_vec[*idx], (q_vec[*idx] - f_vec[*idx]).abs());
            }
        }

        // The diff should be small (Q8_0 quantization of input introduces ~0.01 error)
        assert!(max_diff < 1.0, "QMatMul vs F32 max diff too large: {}", max_diff);

        Ok(())
    }

    /// Debug test: process a single token and dump layer-by-layer hidden states.
    ///
    /// Run with: CHIMERE_DEBUG=1 cargo test test_debug_layer_dump -- --nocapture
    #[test]
    fn test_debug_layer_dump() -> Result<()> {
        // Only run if explicitly requested
        if std::env::var("CHIMERE_DEBUG").is_err() {
            eprintln!("[DEBUG_DUMP] Set CHIMERE_DEBUG=1 to run this test");
            return Ok(());
        }

        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return Ok(());
        }

        // Use GPU — VRAM is free (no cudarc loaded in this test)
        let device = candle_core::Device::cuda_if_available(0)?;
        eprintln!("[DEBUG_DUMP] Device: {:?}", device);

        // Load only 4 layers to fit in VRAM (~4 GB vs 14.7 GB full)
        let model = Qwen35Model::from_gguf(&path, &device, Some(4))?;
        let mut state = crate::state::GdnRecurrentState::new(
            &model.config, &device)?;

        // Process token 248045 (<|im_start|>) — first token of any chat prompt
        let token: u32 = 248045;
        eprintln!("\n=== Processing token {} (<|im_start|>) ===\n", token);

        // Run 4-layer forward manually (no lm_head - only compare hidden states)
        let eps = model.config.rms_norm_eps;
        let q_layers = model.q_layers.as_ref().unwrap();
        let embed_tokens = model.embed_tokens.as_ref().unwrap();

        let token_tensor = Tensor::new(&[token], &Device::Cpu)?;
        let mut hidden = embed_tokens.index_select(&token_tensor, 0)?.to_device(&device)?;

        let h_vec: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
        let l2: f64 = h_vec.iter().map(|&x| (x as f64)*(x as f64)).sum::<f64>().sqrt();
        eprintln!("[CANDLE EMBED] L2={l2:.4} [{:.6},{:.6},{:.6},{:.6},{:.6}]",
            h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4]);

        for (il, layer) in q_layers.iter().enumerate() {
            match layer {
                Qwen35LayerQ::GdnMoE(w) => {
                    hidden = model.forward_gdn_layer_moe(il, w, &hidden, eps, &mut state)?;
                }
                Qwen35LayerQ::AttentionMoE(w) => {
                    hidden = model.forward_attn_layer_moe(il, w, &hidden, eps, &mut state)?;
                }
                Qwen35LayerQ::Gdn(w) => {
                    hidden = model.forward_gdn_layer_q(il, w, &hidden, eps, &mut state)?;
                }
                Qwen35LayerQ::Attention(w) => {
                    hidden = model.forward_attn_layer_q(il, w, &hidden, eps, &mut state)?;
                }
            }
            let h_vec: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
            let l2: f64 = h_vec.iter().map(|&x| (x as f64)*(x as f64)).sum::<f64>().sqrt();
            let ltype = if model.config.is_recurrent(il) { "GDN" } else { "ATN" };
            eprintln!("[CANDLE L{il:02} {ltype} FINAL] L2={l2:.4} [{:.6},{:.6},{:.6},{:.6},{:.6}]",
                h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4]);
        }

        // Compare with cudarc dump:
        // [T0 L00 FINAL] L2=23.1403 [-0.460503,0.827288,-0.640330,-0.741382,0.519498]
        // [T0 L01 FINAL] L2=37.3568 [-1.173556,1.101801,-1.628113,-0.976550,0.430583]
        // [T0 L02 FINAL] L2=41.6393 [-1.059093,1.201728,-1.972191,-0.997158,0.886830]
        // [T0 L03 FINAL] L2=44.3979 [-0.399740,0.573190,-1.697665,-1.117165,1.198967]
        eprintln!("\n=== CUDARC REFERENCE (from earlier dump) ===");
        eprintln!("[cudarc L00 FINAL] L2=23.1403 [-0.460503,0.827288,-0.640330,-0.741382,0.519498]");
        eprintln!("[cudarc L01 FINAL] L2=37.3568 [-1.173556,1.101801,-1.628113,-0.976550,0.430583]");
        eprintln!("[cudarc L02 FINAL] L2=41.6393 [-1.059093,1.201728,-1.972191,-0.997158,0.886830]");
        eprintln!("[cudarc L03 FINAL] L2=44.3979 [-0.399740,0.573190,-1.697665,-1.117165,1.198967]");

        Ok(())
    }

    /// Test loading and running the 35B-A3B MoE model.
    #[test]
    fn test_moe_35b_forward() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("[MOE] Skipping: MoE GGUF not found at {}", path);
            return;
        }

        let device = candle_core::Device::cuda_if_available(0)
            .unwrap_or(candle_core::Device::Cpu);
        eprintln!("[MOE] Device: {:?}", device);
        eprintln!("[MOE] Loading 35B-A3B MoE from {}...", path);

        let model = Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load MoE model");

        eprintln!("[MOE] Loaded: {} layers, moe={}, experts={}",
            model.num_layers(), model.config.is_moe(), model.config.num_experts);

        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        eprintln!("[MOE] Running forward_token(1)...");
        let start = std::time::Instant::now();
        let (logits, _mtp) = model.forward_token(1, &mut state)
            .expect("MoE forward_token failed");
        let elapsed = start.elapsed();

        let logits_cpu: Vec<f32> = logits.squeeze(0).unwrap().to_vec1().unwrap();
        let mut indexed: Vec<(usize, f32)> = logits_cpu.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!("[MOE] First token forward: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        eprintln!("[MOE] Logits shape: {:?}", logits.dims());
        eprintln!("[MOE] Top-5:");
        for (i, &(tok, val)) in indexed.iter().take(5).enumerate() {
            eprintln!("[MOE]   {}: token {} = {:.4}", i + 1, tok, val);
        }

        // Generate tokens — GPU argmax for realistic tok/s measurement
        eprintln!("[MOE] Generating 20 tokens (GPU argmax)...");
        let gen_start = std::time::Instant::now();
        let mut next_tok = indexed[0].0 as u32;
        let mut token_times = Vec::new();
        for i in 0..20 {
            let t0 = std::time::Instant::now();
            let (logits, _) = model.forward_token(next_tok, &mut state)
                .expect("Forward failed");
            // Force GPU sync to measure REAL compute time (not just dispatch)
            if let candle_core::Device::Cuda(d) = &device {
                use candle_core::backend::BackendDevice;
                let _ = d.synchronize();
            }
            let t_forward = t0.elapsed();
            let t1 = std::time::Instant::now();
            // Raw GPU argmax: 1 kernel + 4 bytes dtoh (no pipeline flush)
            let idx: u32 = if matches!(device, candle_core::Device::Cuda(_)) {
                use candle_core::Storage;
                let l_c = logits.contiguous().unwrap();
                let (l_stor, l_lay) = l_c.storage_and_layout();
                let l_cuda = match &*l_stor { Storage::Cuda(c) => c, _ => panic!("") };
                let l_slice = l_cuda.as_cuda_slice::<f32>().unwrap();
                let l_n = logits.elem_count();
                let cuda_dev = match &device { candle_core::Device::Cuda(d) => d, _ => panic!("") };
                let mut argmax_buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
                    cuda_dev.alloc_zeros::<i32>(1).unwrap();
                crate::kernels::elementwise::raw_argmax(l_slice, &mut argmax_buf, l_n, cuda_dev).unwrap();
                drop(l_stor);
                let result: Vec<i32> = argmax_buf.stream().clone().clone_dtoh(&argmax_buf).unwrap();
                result[0] as u32
            } else {
                logits.argmax(candle_core::D::Minus1).unwrap()
                    .flatten_all().unwrap().to_vec1::<u32>().unwrap()[0]
            };
            let t_sample = t1.elapsed();
            next_tok = idx;
            token_times.push((t_forward, t_sample));
            if i < 3 {
                eprintln!("[MOE] token {}: top-1 = {} (forward={:.2}ms sample={:.2}ms)",
                    i + 1, next_tok,
                    t_forward.as_secs_f64() * 1000.0,
                    t_sample.as_secs_f64() * 1000.0);
            }
        }
        let gen_elapsed = gen_start.elapsed();
        // Compute averages (skip token 0 = warmup)
        let avg_forward: f64 = token_times[1..].iter()
            .map(|(f, _)| f.as_secs_f64() * 1000.0).sum::<f64>() / 19.0;
        let avg_sample: f64 = token_times[1..].iter()
            .map(|(_, s)| s.as_secs_f64() * 1000.0).sum::<f64>() / 19.0;
        eprintln!("[MOE] 20 tokens in {:.2}s = {:.1} tok/s (GPU argmax)",
            gen_elapsed.as_secs_f64(),
            20.0 / gen_elapsed.as_secs_f64());
        eprintln!("[MOE] Avg per-token: forward={:.2}ms sample={:.2}ms total={:.2}ms",
            avg_forward, avg_sample, avg_forward + avg_sample);
    }
}
