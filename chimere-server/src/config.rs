//! # Model Configuration — Chimère Engine
//!
//! Unified configuration that bridges HuggingFace model configs to Chimère's
//! internal module configs (GatedDeltaNet, GQA, MoE, etc.).

use once_cell::sync::Lazy;
use serde::Deserialize;
use std::path::Path;

use crate::gguf_loader::GgufFile;
use crate::hybrid_attention::{HybridAttentionConfig, HybridStackConfig, RoutingMode};
use crate::moe_router::MoeRouterConfig;
use crate::GatedDeltaNetConfig;

/// Unified model configuration for Chimère.
#[derive(Debug, Clone)]
pub struct ChimereModelConfig {
    /// Model hidden dimension (2560 for Nanbeige, 2048 for Qwen3-CN)
    pub hidden_size: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA grouping)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// FFN intermediate size
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE base frequency
    pub rope_theta: f64,
    /// Which layers use full GQA attention (others use DeltaNet)
    pub attention_layers: Vec<usize>,
    /// Number of routed MoE experts (0 = dense FFN, no routing)
    pub num_experts: usize,
    /// Number of shared experts (always active, bypasses router)
    pub num_shared_experts: usize,
    /// Top-K for MoE routing
    pub moe_top_k: usize,
}

/// HuggingFace config.json format (LlamaForCausalLM and compatible).
#[derive(Debug, Deserialize)]
struct HfConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_shared_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
}

fn default_head_dim() -> usize {
    128
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl ChimereModelConfig {
    /// Parse a HuggingFace config.json into ChimereModelConfig.
    ///
    /// `attention_layers` defaults to every 8th layer (1:7 ratio) unless overridden.
    pub fn from_hf_config(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let hf: HfConfig = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let num_experts = hf.num_experts.unwrap_or(0);
        let num_shared_experts = hf.num_shared_experts.unwrap_or(if num_experts > 0 { 1 } else { 0 });
        let moe_top_k = hf.num_experts_per_tok.unwrap_or(if num_experts > 0 { 2 } else { 0 });

        // Default attention layers: every 8th layer (Chimère 1:7 ratio)
        let attention_layers: Vec<usize> = (0..hf.num_hidden_layers)
            .filter(|i| (i + 1) % 8 == 0)
            .collect();

        Ok(Self {
            hidden_size: hf.hidden_size,
            num_attention_heads: hf.num_attention_heads,
            num_kv_heads: hf.num_key_value_heads,
            head_dim: hf.head_dim,
            num_layers: hf.num_hidden_layers,
            intermediate_size: hf.intermediate_size,
            vocab_size: hf.vocab_size,
            rms_norm_eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            attention_layers,
            num_experts,
            num_shared_experts,
            moe_top_k,
        })
    }

    /// Preset for Nanbeige4.1-3B (LlamaForCausalLM, 32 layers, GQA 20Q/4KV).
    pub fn nanbeige_3b() -> Self {
        Self {
            hidden_size: 2560,
            num_attention_heads: 20,
            num_kv_heads: 4,
            head_dim: 128,
            num_layers: 32,
            intermediate_size: 10496,
            vocab_size: 166144,
            rms_norm_eps: 1e-5,
            rope_theta: 7e7,
            attention_layers: vec![7, 15, 23, 31],
            num_experts: 0,
            num_shared_experts: 0,
            moe_top_k: 0,
        }
    }

    /// Is a given layer a GQA attention layer?
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        self.attention_layers.contains(&layer_idx)
    }

    /// Number of DeltaNet layers.
    pub fn num_deltanet_layers(&self) -> usize {
        self.num_layers - self.attention_layers.len()
    }

    /// Convert to GatedDeltaNetConfig for DeltaNet layers.
    pub fn to_deltanet_config(&self) -> GatedDeltaNetConfig {
        GatedDeltaNetConfig {
            hidden_dim: self.hidden_size,
            num_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            conv_kernel: 0, // no conv when loading from pretrained MHA
            gate_mode: crate::GateMode::Scalar,
        }
    }

    /// Convert to HybridAttentionConfig for GQA layers.
    pub fn to_gqa_config(&self) -> HybridAttentionConfig {
        HybridAttentionConfig {
            hidden_dim: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            routing_mode: RoutingMode::FullOnly,
            delta_threshold: 0.5,
            full_attention_capacity: 1.0,
        }
    }

    /// Convert to HybridStackConfig.
    pub fn to_stack_config(&self) -> HybridStackConfig {
        HybridStackConfig {
            num_layers: self.num_layers,
            attention_layers: self.attention_layers.clone(),
            dynamic_routing: false,
        }
    }

    /// Convert to MoeRouterConfig for MoE layers.
    pub fn to_moe_config(&self) -> MoeRouterConfig {
        MoeRouterConfig {
            hidden_dim: self.hidden_size,
            num_experts: self.num_experts,
            min_k: 1,
            max_k: self.moe_top_k.max(1),
            use_shared_expert: self.num_shared_experts > 0,
            tsallis_q: 2.0,
            temperature: 1.0,
            routed_scaling_factor: 1.0,
            sinkhorn_iterations: 3,
            entropy_threshold_low: 0.3,
            entropy_threshold_high: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 Architecture Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Qwen3.5 architecture (GDN-recurrent + sparse attention).
///
/// Qwen3.5 uses a hybrid architecture where most layers are GDN (Gated DeltaNet /
/// recurrent SSM) and every Nth layer is full attention.  This config captures
/// all architectural parameters needed to instantiate the model, including the
/// SSM state dimensions, RoPE sections, and MTP (multi-token prediction) head.
///
/// For the 35B-A3B MoE variant, `num_experts > 0` and each FFN is a mixture-of-experts
/// with a shared expert plus `num_experts` routed experts (top-`experts_per_token` selected).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_main_layers: usize,
    pub num_total_layers: usize,
    /// FFN intermediate size (used by dense models; ignored for MoE models).
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    // GDN
    pub ssm_d_state: usize,
    pub ssm_d_inner: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,
    pub ssm_conv_kernel: usize,
    // Attention pattern: every Nth layer is full attention
    pub full_attn_interval: usize,
    // RoPE
    pub rope_theta: f64,
    pub rope_sections: [usize; 4],
    // MTP
    pub nextn_predict_layers: usize,
    // MoE (Mixture of Experts) — 0 = dense FFN, >0 = MoE (35B-A3B)
    /// Number of routed experts. 0 means dense FFN (not a MoE model).
    pub num_experts: usize,
    /// Number of experts selected per token (top-k routing).
    pub experts_per_token: usize,
    /// Hidden dimension of each routed expert's FFN.
    pub expert_ffn_hidden: usize,
    /// Hidden dimension of the shared expert's FFN (always active, bypasses router).
    pub shared_expert_ffn_hidden: usize,
}

impl Qwen35Config {
    /// Preset for Qwen3.5-35B-A3B (MoE) with all known architectural values.
    ///
    /// 35B-A3B uses 256 routed experts with top-8 selection, an expert FFN hidden
    /// dimension of 512, and a shared expert that is always active.  The MTP layer
    /// (layer 40) has been pruned, so `nextn_predict_layers` is 0 and
    /// `num_total_layers` equals `num_main_layers`.
    pub fn qwen35_35b_a3b() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_kv_heads: 2,
            // KV head dim (Q head dim is 512 but managed in attention code)
            head_dim: 256,
            num_main_layers: 40,
            // MTP pruned — no extra layer beyond the 40 main layers
            num_total_layers: 40,
            // intermediate_size is unused for MoE layers; set to expert_ffn_hidden for compat
            intermediate_size: 512,
            vocab_size: 248320,
            rms_norm_eps: 1e-6,
            // SSM / GDN parameters for 35B-A3B
            ssm_d_state: 128,
            ssm_d_inner: 4096,
            ssm_dt_rank: 32,
            ssm_n_group: 16,
            ssm_conv_kernel: 4,
            full_attn_interval: 4,
            rope_theta: 10_000_000.0,
            // From GGUF metadata
            rope_sections: [11, 11, 10, 0],
            // MTP pruned
            nextn_predict_layers: 0,
            // MoE parameters
            num_experts: 256,
            experts_per_token: 8,
            expert_ffn_hidden: 512,
            shared_expert_ffn_hidden: 512,
        }
    }

    /// Returns `true` if this model uses Mixture-of-Experts FFN layers.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Convenience: open a GGUF file and build configuration from its metadata.
    ///
    /// This parses only the GGUF header (metadata + tensor index) but does NOT
    /// load any weight data to GPU. The `GgufFile` is dropped after extracting
    /// the config.
    pub fn from_gguf_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let gguf = GgufFile::open(path.as_ref())
            .map_err(|e| format!("Failed to open GGUF: {}", e))?;
        Self::from_gguf(&gguf)
    }

    /// Build configuration from GGUF metadata.
    ///
    /// Reads architecture-prefixed keys (e.g. `qwen35.block_count`,
    /// `qwen35.embedding_length`) from the parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .get_metadata("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing general.architecture in GGUF".to_string())?
            .to_string();

        let get_u32 = |key: &str| -> Result<u32, String> {
            gguf.get_metadata(key)
                .and_then(|v| v.as_u32())
                .ok_or_else(|| format!("Missing or invalid GGUF key: {}", key))
        };

        let get_f32 = |key: &str| -> Result<f32, String> {
            gguf.get_metadata(key)
                .and_then(|v| v.as_f32())
                .ok_or_else(|| format!("Missing or invalid GGUF key: {}", key))
        };

        let get_u32_or = |key: &str, default: u32| -> u32 {
            gguf.get_metadata(key)
                .and_then(|v| v.as_u32())
                .unwrap_or(default)
        };

        let hidden_size = get_u32(&format!("{}.embedding_length", arch))? as usize;
        let num_attention_heads = get_u32(&format!("{}.attention.head_count", arch))? as usize;
        let num_kv_heads = get_u32(&format!("{}.attention.head_count_kv", arch))? as usize;
        let head_dim = get_u32_or(
            &format!("{}.attention.key_length", arch),
            (hidden_size / num_attention_heads) as u32,
        ) as usize;
        let num_total_layers = get_u32(&format!("{}.block_count", arch))? as usize;
        let nextn_predict_layers = get_u32_or(
            &format!("{}.nextn_predict_layers", arch),
            0,
        ) as usize;
        // If the GGUF was pruned (--prune-layers), MTP tensors are gone but
        // nextn_predict_layers metadata is still 1.  Detect this by checking
        // whether blk.{num_total_layers - 1}.nextn.eh_proj.weight exists.
        // If it does NOT exist, the MTP layer was pruned → use all blocks as main.
        let mtp_actually_present = if nextn_predict_layers > 0 {
            let mtp_layer = num_total_layers - 1; // would-be MTP layer index
            let probe_name = format!("blk.{mtp_layer}.nextn.eh_proj.weight");
            gguf.get_tensor_info(&probe_name).is_some()
        } else {
            false
        };
        let effective_nextn = if mtp_actually_present { nextn_predict_layers } else { 0 };
        if nextn_predict_layers > 0 && !mtp_actually_present {
            eprintln!(
                "[CONFIG] nextn_predict_layers={} but MTP tensors not found (pruned GGUF). Using all {} layers as main.",
                nextn_predict_layers, num_total_layers
            );
        }
        let num_main_layers = num_total_layers - effective_nextn;
        // feed_forward_length is optional for MoE models (they use expert_feed_forward_length)
        let intermediate_size = get_u32_or(
            &format!("{}.feed_forward_length", arch),
            0,
        ) as usize;

        // Vocab size: try the GGUF token count from tokenizer metadata first,
        // then fall back to the token_embd tensor shape.
        let vocab_size = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .or_else(|| {
                gguf.get_tensor_info("token_embd.weight")
                    .map(|t| t.dims[0] as usize)
            })
            .ok_or_else(|| "Cannot determine vocab_size from GGUF".to_string())?;

        let rms_norm_eps = get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            .unwrap_or(1e-6) as f64;

        // SSM parameters (GDN)
        let ssm_d_state = get_u32_or(&format!("{}.ssm.state_size", arch), 128) as usize;
        let ssm_d_inner = get_u32_or(&format!("{}.ssm.inner_size", arch), 6144) as usize;
        let ssm_dt_rank = get_u32_or(&format!("{}.ssm.time_step_rank", arch), 48) as usize;
        let ssm_n_group = get_u32_or(&format!("{}.ssm.group_count", arch), 16) as usize;
        let ssm_conv_kernel = get_u32_or(&format!("{}.ssm.conv_kernel", arch), 4) as usize;

        // Attention pattern interval
        let full_attn_interval = get_u32_or(
            &format!("{}.attention.full_attn_interval", arch),
            4,
        ) as usize;

        // RoPE
        let rope_theta = get_f32(&format!("{}.rope.freq_base", arch))
            .unwrap_or(10_000_000.0) as f64;

        // RoPE sections: stored as array of int32 in the GGUF (not uint32 as one might expect).
        // as_u32() now handles Int32 values with v >= 0, so this parse is correct.
        let rope_sections = gguf
            .get_metadata(&format!("{}.rope.dimension_sections", arch))
            .and_then(|v| v.as_array())
            .map(|arr| {
                let mut s = [0usize; 4];
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u32().unwrap_or(0) as usize;
                }
                s
            })
            .unwrap_or([11, 11, 10, 0]);

        // MoE parameters — present in 35B-A3B (defaults to 0 = dense for non-MoE models)
        let num_experts = get_u32_or(
            &format!("{}.expert_count", arch),
            0,
        ) as usize;
        let experts_per_token = get_u32_or(
            &format!("{}.expert_used_count", arch),
            0,
        ) as usize;
        let expert_ffn_hidden = get_u32_or(
            &format!("{}.expert_feed_forward_length", arch),
            0,
        ) as usize;
        // Shared expert FFN hidden: try dedicated key first, fall back to expert_ffn_hidden
        let shared_expert_ffn_hidden = get_u32_or(
            &format!("{}.expert_shared_feed_forward_length", arch),
            expert_ffn_hidden as u32,
        ) as usize;

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            num_main_layers,
            num_total_layers,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            ssm_d_state,
            ssm_d_inner,
            ssm_dt_rank,
            ssm_n_group,
            ssm_conv_kernel,
            full_attn_interval,
            rope_theta,
            rope_sections,
            nextn_predict_layers: effective_nextn,
            num_experts,
            experts_per_token,
            expert_ffn_hidden,
            shared_expert_ffn_hidden,
        })
    }

    /// Returns `true` for GDN (recurrent) layers.
    ///
    /// A layer is recurrent when it is a main layer and NOT a full-attention layer.
    pub fn is_recurrent(&self, layer: usize) -> bool {
        layer < self.num_main_layers && !self.is_attention(layer)
    }

    /// Returns `true` for full attention layers.
    ///
    /// In Qwen3.5, every `full_attn_interval`-th layer (1-indexed) uses full
    /// multi-head attention instead of GDN recurrence.
    pub fn is_attention(&self, layer: usize) -> bool {
        layer < self.num_main_layers && (layer + 1) % self.full_attn_interval == 0
    }

    /// Count of GDN (recurrent) layers among the main layers.
    pub fn num_gdn_layers(&self) -> usize {
        (0..self.num_main_layers)
            .filter(|&l| self.is_recurrent(l))
            .count()
    }

    /// Count of full attention layers among the main layers.
    pub fn num_attn_layers(&self) -> usize {
        (0..self.num_main_layers)
            .filter(|&l| self.is_attention(l))
            .count()
    }

    /// Map a global layer index to the GDN-layer index (0-based among GDN layers).
    ///
    /// Returns `None` if the layer is not a GDN layer.
    pub fn gdn_index(&self, layer: usize) -> Option<usize> {
        if !self.is_recurrent(layer) {
            return None;
        }
        Some(
            (0..layer)
                .filter(|&l| self.is_recurrent(l))
                .count(),
        )
    }

    /// Map a global layer index to the attention index (0-based among attention layers).
    ///
    /// Returns `None` if the layer is not an attention layer.
    pub fn attn_index(&self, layer: usize) -> Option<usize> {
        if !self.is_attention(layer) {
            return None;
        }
        Some(
            (0..layer)
                .filter(|&l| self.is_attention(l))
                .count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Runtime Environment Configuration — ChimereConfig
// ---------------------------------------------------------------------------
//
// Centralizes the ~10 essential CHIMERE_* environment variables into a single
// struct that is read once at startup and accessible globally via
// `chimere_config()`.
//
// ## Debug/profiling env vars (candidates for removal from callers)
//
// The following env vars are scattered across the codebase for debug/profiling
// purposes.  They are NOT included in ChimereConfig and should be removed from
// callers in a future cleanup pass:
//
//   CHIMERE_TOKEN_DUMP      — dump token IDs during generation
//   CHIMERE_LAYER_DUMP      — dump per-layer hidden states
//   CHIMERE_L0_DUMP         — dump layer-0 activations
//   CHIMERE_GPU_PROF        — GPU profiling markers
//   CHIMERE_PROFILE         — general profiling toggle
//   CHIMERE_DISPATCH_PROF   — dispatch-level profiling
//   CHIMERE_GRAN_PROF       — granular profiling
//   CHIMERE_DEBUG           — verbose debug output
//   CHIMERE_VRAM_LOG        — VRAM allocation logging
//   CHIMERE_GDN_PROFILE     — GDN layer profiling
//   CHIMERE_MOE_PROFILE     — MoE routing profiling
//   CHIMERE_COUNT_OPS       — operation counting
//   CHIMERE_DUMP_LOGITS     — dump raw logits
//   CHIMERE_FUSED_MOE_V2    — fused MoE v2 toggle (experimental)
//   CHIMERE_FUSED_ELEM      — fused element-wise ops toggle
//   CHIMERE_FUSED_SSM_PROJ  — fused SSM projection toggle
//   CHIMERE_TRACE           — execution tracing
//   CHIMERE_EARLY_EXIT      — early exit from layers
//   CHIMERE_SKIP_LAYERS     — skip specific layers

/// Runtime environment configuration for the Chimere engine.
///
/// Consolidates the essential `CHIMERE_*` environment variables into a single
/// struct.  Read once at process startup via [`chimere_config()`] and cached
/// for the lifetime of the process.
#[derive(Debug, Clone)]
pub struct ChimereConfig {
    /// Enable ggml GPU kernels for quantized GEMV (CHIMERE_GGML_GPU).
    pub ggml_gpu: bool,
    /// Enable the cudarc forward path (CHIMERE_CUDARC_FORWARD).
    pub cudarc_forward: bool,
    /// HTTP server listen port (CHIMERE_PORT, default "8090").
    pub port: String,
    /// Path to the GGUF model file (CHIMERE_MODEL).
    pub model: String,
    /// Path to the tokenizer (CHIMERE_TOKENIZER, optional — empty if unset).
    pub tokenizer: String,
    /// Model name reported by the API (CHIMERE_NAME, default "chimere-deltanet").
    pub name: String,
    /// Number of MoE layers whose experts are offloaded to CPU (CHIMERE_NCMOE, default 0).
    pub ncmoe: usize,
    /// Maximum sequence length for KV cache (CHIMERE_KV_MAX_SEQ, default 65536).
    pub kv_max_seq: usize,
    /// Token budget for thinking/reasoning (CHIMERE_THINK_BUDGET, default 16384).
    pub think_budget: usize,
    /// Run lm_head on CPU to save VRAM (CHIMERE_LM_HEAD_CPU).
    pub lm_head_cpu: bool,
}

impl Default for ChimereConfig {
    fn default() -> Self {
        Self {
            ggml_gpu: false,
            cudarc_forward: false,
            port: "8090".to_string(),
            model: String::new(),
            tokenizer: String::new(),
            name: "chimere-deltanet".to_string(),
            ncmoe: 0,
            kv_max_seq: 65536,
            think_budget: 16384,
            lm_head_cpu: false,
        }
    }
}

impl ChimereConfig {
    /// Build a `ChimereConfig` by reading all `CHIMERE_*` environment variables.
    ///
    /// Boolean env vars are considered `true` if set to any non-empty value
    /// that is not `"0"` or `"false"` (case-insensitive).
    ///
    /// Numeric env vars that fail to parse fall back to their defaults with a
    /// warning on stderr.
    pub fn from_env() -> Self {
        fn env_bool(key: &str) -> bool {
            match std::env::var(key) {
                Ok(v) => {
                    let v = v.trim().to_lowercase();
                    !v.is_empty() && v != "0" && v != "false"
                }
                Err(_) => false,
            }
        }

        fn env_string(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }

        fn env_usize(key: &str, default: usize) -> usize {
            match std::env::var(key) {
                Ok(v) => v.trim().parse::<usize>().unwrap_or_else(|e| {
                    eprintln!(
                        "[ChimereConfig] WARNING: {}={:?} is not a valid usize ({}), using default {}",
                        key, v, e, default
                    );
                    default
                }),
                Err(_) => default,
            }
        }

        Self {
            ggml_gpu: env_bool("CHIMERE_GGML_GPU"),
            cudarc_forward: env_bool("CHIMERE_CUDARC_FORWARD"),
            port: env_string("CHIMERE_PORT", "8090"),
            model: env_string("CHIMERE_MODEL", ""),
            tokenizer: env_string("CHIMERE_TOKENIZER", ""),
            name: env_string("CHIMERE_NAME", "chimere-deltanet"),
            ncmoe: env_usize("CHIMERE_NCMOE", 0),
            kv_max_seq: env_usize("CHIMERE_KV_MAX_SEQ", 65536),
            think_budget: env_usize("CHIMERE_THINK_BUDGET", 16384),
            lm_head_cpu: env_bool("CHIMERE_LM_HEAD_CPU"),
        }
    }
}

/// Global singleton accessor for the runtime configuration.
///
/// Reads all `CHIMERE_*` environment variables on first access and caches the
/// result for the lifetime of the process.
///
/// # Example
///
/// ```rust,no_run
/// use chimere_deltanet::config::chimere_config;
///
/// let cfg = chimere_config();
/// if cfg.ggml_gpu {
///     println!("Using ggml GPU kernels");
/// }
/// println!("Listening on port {}", cfg.port);
/// ```
pub fn chimere_config() -> &'static ChimereConfig {
    static CONFIG: Lazy<ChimereConfig> = Lazy::new(|| {
        let cfg = ChimereConfig::from_env();
        eprintln!("[ChimereConfig] Loaded from environment: {:?}", cfg);
        cfg
    });
    &CONFIG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nanbeige_preset() {
        let cfg = ChimereModelConfig::nanbeige_3b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_attention_heads, 20);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.vocab_size, 166144);
        assert_eq!(cfg.attention_layers, vec![7, 15, 23, 31]);
        assert_eq!(cfg.num_deltanet_layers(), 28);
        assert_eq!(cfg.num_experts, 0);
        println!("Nanbeige preset: {} layers ({} DeltaNet + {} GQA)",
            cfg.num_layers, cfg.num_deltanet_layers(), cfg.attention_layers.len());
    }

    #[test]
    fn test_parse_nanbeige_config_json() {
        let path = std::path::Path::new(
            &std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into())
        ).join(".chimere/models/Nanbeige4.1-3B/config.json");

        if !path.exists() {
            println!("Skipping: Nanbeige model not found at {:?}", path);
            return;
        }

        let cfg = ChimereModelConfig::from_hf_config(&path).expect("Failed to parse config.json");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_attention_heads, 20);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.intermediate_size, 10496);
        assert_eq!(cfg.vocab_size, 166144);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((cfg.rope_theta - 7e7).abs() < 1.0);
        println!("Parsed Nanbeige config.json: hidden={}, layers={}, vocab={}",
            cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
    }

    #[test]
    fn test_layer_type_classification() {
        let cfg = ChimereModelConfig::nanbeige_3b();
        assert!(!cfg.is_attention_layer(0));
        assert!(!cfg.is_attention_layer(6));
        assert!(cfg.is_attention_layer(7));
        assert!(cfg.is_attention_layer(15));
        assert!(cfg.is_attention_layer(23));
        assert!(cfg.is_attention_layer(31));
        assert!(!cfg.is_attention_layer(8));
    }

    // -----------------------------------------------------------------------
    // Qwen35Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_qwen35_35b_a3b_preset() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.num_main_layers, 40);
        // MTP pruned: num_total_layers == num_main_layers
        assert_eq!(cfg.num_total_layers, 40);
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.ssm_d_state, 128);
        assert_eq!(cfg.ssm_d_inner, 4096);
        assert_eq!(cfg.ssm_dt_rank, 32);
        assert_eq!(cfg.ssm_n_group, 16);
        assert_eq!(cfg.ssm_conv_kernel, 4);
        assert_eq!(cfg.full_attn_interval, 4);
        assert_eq!(cfg.rope_sections, [11, 11, 10, 0]);
        // MTP pruned: no extra prediction layers
        assert_eq!(cfg.nextn_predict_layers, 0);
        // MoE fields
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.experts_per_token, 8);
        assert_eq!(cfg.expert_ffn_hidden, 512);
        assert_eq!(cfg.shared_expert_ffn_hidden, 512);
        assert!(cfg.is_moe());
        // Sanity: num_total_layers == num_main_layers + nextn_predict_layers
        assert_eq!(cfg.num_total_layers, cfg.num_main_layers + cfg.nextn_predict_layers);
        println!(
            "Qwen3.5-35B-A3B preset: hidden={}, experts={}, top_k={}, expert_ffn={}, \
             main_layers={}, total_layers={}, vocab={}",
            cfg.hidden_size, cfg.num_experts, cfg.experts_per_token, cfg.expert_ffn_hidden,
            cfg.num_main_layers, cfg.num_total_layers, cfg.vocab_size
        );
    }

    #[test]
    fn test_layer_classification() {
        let cfg = Qwen35Config::qwen35_35b_a3b();

        // Layers 0,1,2 = GDN (recurrent), layer 3 = attention (since (3+1)%4 == 0)
        assert!(cfg.is_recurrent(0), "Layer 0 should be GDN");
        assert!(cfg.is_recurrent(1), "Layer 1 should be GDN");
        assert!(cfg.is_recurrent(2), "Layer 2 should be GDN");
        assert!(cfg.is_attention(3), "Layer 3 should be attention");

        // Layers 4,5,6 = GDN, layer 7 = attention
        assert!(cfg.is_recurrent(4), "Layer 4 should be GDN");
        assert!(cfg.is_recurrent(5), "Layer 5 should be GDN");
        assert!(cfg.is_recurrent(6), "Layer 6 should be GDN");
        assert!(cfg.is_attention(7), "Layer 7 should be attention");

        // Last attention layer: 39 => (39+1)%4 == 0
        assert!(cfg.is_attention(39), "Layer 39 should be attention");

        // Out-of-range layer (layer 40 is beyond main layers)
        assert!(!cfg.is_attention(40), "Layer 40 should not be attention");
        assert!(!cfg.is_recurrent(40), "Layer 40 should not be recurrent");

        println!("Layer classification verified for all edge cases");
    }

    #[test]
    fn test_gdn_attn_counts() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let gdn = cfg.num_gdn_layers();
        let attn = cfg.num_attn_layers();
        assert_eq!(gdn, 30, "Expected 30 GDN layers, got {}", gdn);
        assert_eq!(attn, 10, "Expected 10 attention layers, got {}", attn);
        assert_eq!(gdn + attn, cfg.num_main_layers, "GDN + attn should equal main layers");
        println!("Layer counts: {} GDN + {} attention = {} main", gdn, attn, cfg.num_main_layers);
    }

    #[test]
    fn test_gdn_attn_indices() {
        let cfg = Qwen35Config::qwen35_35b_a3b();

        // Layer 0 is GDN index 0
        assert_eq!(cfg.gdn_index(0), Some(0));
        assert_eq!(cfg.attn_index(0), None);

        // Layer 3 is attention index 0
        assert_eq!(cfg.gdn_index(3), None);
        assert_eq!(cfg.attn_index(3), Some(0));

        // Layer 4 is GDN index 3 (layers 0,1,2 are GDN before it)
        assert_eq!(cfg.gdn_index(4), Some(3));

        // Layer 7 is attention index 1
        assert_eq!(cfg.attn_index(7), Some(1));

        // Last attention layer (39) should be attn index 9
        assert_eq!(cfg.attn_index(39), Some(9));

        // Last GDN layer before 39 is 38, its GDN index should be 29
        assert_eq!(cfg.gdn_index(38), Some(29));

        println!("GDN/attention index mapping verified");
    }

    #[test]
    fn test_from_gguf() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home
        );

        if !std::path::Path::new(&path).exists() {
            println!("Skipping: GGUF file not found at {}", path);
            return;
        }

        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("Failed to build Qwen35Config from GGUF");

        // Verify against known Qwen3.5-35B-A3B values
        assert_eq!(cfg.hidden_size, 2048, "hidden_size mismatch");
        assert_eq!(cfg.num_attention_heads, 16, "num_attention_heads mismatch");
        assert_eq!(cfg.num_kv_heads, 2, "num_kv_heads mismatch");
        assert_eq!(cfg.head_dim, 256, "head_dim mismatch");
        assert_eq!(cfg.num_total_layers, 40, "num_total_layers mismatch");
        assert_eq!(cfg.num_main_layers, 40, "num_main_layers mismatch");
        assert!(cfg.vocab_size > 100000, "vocab_size too small: {}", cfg.vocab_size);
        assert_eq!(cfg.num_experts, 256, "num_experts mismatch");
        assert_eq!(cfg.experts_per_token, 8, "experts_per_token mismatch");

        println!(
            "Qwen35Config from GGUF: hidden={}, heads={}, kv={}, layers={}/{}, vocab={}, \
             ssm_d_state={}, ssm_d_inner={}, full_attn_interval={}, rope_theta={}, \
             experts={}, top_k={}",
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads,
            cfg.num_main_layers, cfg.num_total_layers, cfg.vocab_size,
            cfg.ssm_d_state, cfg.ssm_d_inner, cfg.full_attn_interval, cfg.rope_theta,
            cfg.num_experts, cfg.experts_per_token
        );
    }

    // -----------------------------------------------------------------------
    // ChimereConfig (runtime env) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chimere_config_defaults() {
        let cfg = ChimereConfig::default();
        assert!(!cfg.ggml_gpu);
        assert!(!cfg.cudarc_forward);
        assert_eq!(cfg.port, "8090");
        assert!(cfg.model.is_empty());
        assert!(cfg.tokenizer.is_empty());
        assert_eq!(cfg.name, "chimere-deltanet");
        assert_eq!(cfg.ncmoe, 0);
        assert_eq!(cfg.kv_max_seq, 65536);
        assert_eq!(cfg.think_budget, 16384);
        assert!(!cfg.lm_head_cpu);
    }

    #[test]
    fn test_chimere_config_from_env() {
        // from_env() should not panic even with no CHIMERE_* vars set.
        // It will pick up whatever is in the real environment, so we just
        // verify the struct is valid and Debug-printable.
        let cfg = ChimereConfig::from_env();
        println!("ChimereConfig from env: {:?}", cfg);
        // port should be non-empty (either env or default)
        assert!(!cfg.port.is_empty());
    }

    #[test]
    fn test_chimere_config_global_accessor() {
        // chimere_config() returns a &'static reference; calling it twice
        // must return the same pointer (same Lazy instance).
        let a = chimere_config() as *const ChimereConfig;
        let b = chimere_config() as *const ChimereConfig;
        assert_eq!(a, b, "chimere_config() should return the same static reference");
    }
}
