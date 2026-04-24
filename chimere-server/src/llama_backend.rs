//! libllama FFI backend — uses ik_llama's optimized forward pass for 93 tok/s parity.
//!
//! Toggle: `CHIMERE_LLAMA_BACKEND=1`
//!
//! This backend delegates the ENTIRE forward pass (embedding, all layers, lm_head)
//! to libllama.so from ik_llama.cpp. chimere's own weight loading (Candle, cudarc)
//! is completely bypassed — libllama loads weights from GGUF once, using its own
//! optimized CUDA kernels (MMVQ, flash attention, fused MoE).
//!
//! ## Why not just use llama-server directly?
//!
//! chimere adds value on top of libllama's forward pass:
//! - Custom sampling (DRY penalty, Engram-aware temperature)
//! - Entropy-adaptive routing (future: switch attention heads)
//! - Block diffusion / multi-draft generation
//! - SSE streaming with OpenAI-compatible API
//!
//! ## Architecture
//!
//! ```text
//! chimere-server
//!   ├── tokenizer (tokenizers crate)
//!   ├── LlamaForward (this module)
//!   │     ├── llama_model (loaded via libllama FFI)
//!   │     ├── llama_context (KV cache, GDN state managed by libllama)
//!   │     └── llama_decode() → raw logits
//!   ├── sampling (chimere's own: temperature, top_p, DRY, etc.)
//!   └── SSE streaming server (axum)
//! ```
//!
//! ## Thread safety
//!
//! `LlamaForward` is `!Send` and `!Sync` (raw pointers). The server wraps it
//! in `tokio::sync::Mutex` for serialized access — same pattern as the cudarc path.
//!
//! ## ncmoe
//!
//! The `-ncmoe N` mechanism is replicated via `tensor_buft_overrides`: for each
//! layer 0..N, MoE expert weights are pinned to CPU buffer type. This matches
//! ik_llama-server's `--n-cpu-moe` behavior exactly.

use std::ffi::{c_char, c_float, c_int, c_void, CString};

// ---------------------------------------------------------------------------
// Opaque types — llama.h forward-declares these as opaque structs
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaVocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct GgmlBackendBufferType {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Struct reproductions from llama.h (must match C layout EXACTLY)
// ---------------------------------------------------------------------------

/// Mirrors `struct llama_batch` from llama.h (line 314-332).
///
/// When using `llama_batch_get_one()`, only `n_tokens`, `token`, `all_pos_0`,
/// `all_pos_1`, and `all_seq_id` are set. The pointer fields (pos, n_seq_id,
/// seq_id, logits) are NULL.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut i32,
    pub embd: *mut c_float,
    pub pos: *mut i32,        // llama_pos = i32
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32, // llama_seq_id** = i32**
    pub logits: *mut i8,
    // Helper fields for simple batches:
    pub all_pos_0: i32,  // llama_pos
    pub all_pos_1: i32,  // llama_pos
    pub all_seq_id: i32, // llama_seq_id
}

/// Mirrors `struct llama_model_tensor_buft_override` from llama.h (line 354-357).
#[repr(C)]
pub struct LlamaModelTensorBuftOverride {
    pub pattern: *const c_char,
    pub buft: *mut GgmlBackendBufferType,
}

/// Mirrors `struct llama_chat_message` from llama.h (line 523-526).
///
/// Used by `llama_chat_apply_template()` to format conversation history with
/// the model's built-in chat template (jinja-equivalent C++ engine). Both
/// pointers must outlive the call — pass slices into stable `CString` storage.
#[repr(C)]
pub struct LlamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

/// Progress callback type.
type LlamaProgressCallback = Option<extern "C" fn(f32, *mut c_void) -> bool>;

/// Mirrors `struct llama_model_params` from llama.h (line 359-403).
/// Field order and types must match EXACTLY for ABI compatibility.
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *const c_char,
    pub n_gpu_layers: i32,
    pub mla: i32,
    pub split_mode: i32, // enum llama_split_mode
    pub main_gpu: i32,
    pub max_gpu: i32,
    pub tensor_split: *const c_float,
    pub rpc_servers: *const c_char,
    pub progress_callback: LlamaProgressCallback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void, // llama_model_kv_override*
    pub tensor_buft_overrides: *const LlamaModelTensorBuftOverride,
    // Booleans (packed together per llama.h comment)
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub repack_tensors: bool,
    pub use_thp: bool,
    pub validate_quants: bool,
    pub merge_qkv: bool,
    pub merge_up_gate_exps: bool,
    pub mtp: bool,
}

/// Eval callback type (ggml_backend_sched_eval_callback).
type GgmlBackendSchedEvalCallback =
    Option<extern "C" fn(*mut c_void, bool, *mut c_void) -> bool>;

/// Abort callback type (ggml_abort_callback).
type GgmlAbortCallback = Option<extern "C" fn(*mut c_void) -> bool>;

/// Mirrors `struct llama_context_params` from llama.h (line 408-468).
/// Field order and types must match EXACTLY.
#[repr(C)]
pub struct LlamaContextParams {
    pub seed: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub max_extra_alloc: i32,
    pub rope_scaling_type: i32, // enum llama_rope_scaling_type
    pub pooling_type: i32,      // enum llama_pooling_type
    pub attention_type: i32,    // enum llama_attention_type
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: GgmlBackendSchedEvalCallback,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32, // enum ggml_type
    pub type_v: i32, // enum ggml_type
    pub type_reduce: i32, // enum ggml_type
    // Booleans
    pub logits_all: bool,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub flash_attn: bool,
    pub mla_attn: c_int,
    pub attn_max_batch: c_int,
    pub fused_moe_up_gate: bool,
    pub grouped_expert_routing: bool,
    pub fused_up_gate: bool,
    pub fused_mmad: bool,
    pub rope_cache: bool,
    pub graph_reuse: bool,
    pub min_experts: c_int,
    pub thresh_experts: c_float,
    pub only_active_experts: bool,
    pub k_cache_hadamard: bool,
    pub split_mode_graph_scheduling: bool,
    pub scheduler_async: bool,
    pub mtp: bool,
    pub mtp_op_type: i32, // enum llama_mtp_op_type
    pub abort_callback: GgmlAbortCallback,
    pub abort_callback_data: *mut c_void,
    pub offload_policy: *mut c_void,
    pub cuda_params: *mut c_void,
}

/// Performance timing information from llama.h.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LlamaTimings {
    pub t_start_ms: f64,
    pub t_end_ms: f64,
    pub t_load_ms: f64,
    pub t_sample_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_sample: i32,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

/// FFI struct for logprob results from C++ sampler
#[repr(C)]
pub struct ChimereLogprobResult {
    pub token_id: i32,
    pub n_top: i32,
    pub top_tokens: [i32; 5],
    pub top_logprobs: [f32; 5],
}

/// Top-k logprob entry (token ID + log probability)
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
}

/// Per-sequence entry for [`LlamaForward::forward_multi_seq`] (M1 J3).
///
/// Each entry represents one token to feed into the batch, tagged with the
/// `seq_id` it belongs to and the position within that sequence. Set
/// `request_logits = true` on the entries whose output the caller needs
/// (typically the last token of a prefill chunk, or every generate-step
/// token).
#[derive(Debug, Clone, Copy)]
pub struct MultiSeqEntry {
    pub token: u32,
    pub pos: i32,
    pub seq_id: i32,
    pub request_logits: bool,
}

// ggml_type constants we need for KV cache type specification
#[allow(dead_code)]
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_Q8_0: i32 = 8;
const GGML_TYPE_Q4_0: i32 = 2;

/// MTP operation types — mirrors `enum llama_mtp_op_type` from llama.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MtpOp {
    None = 0,
    Warmup = 1,
    UpdateAccepted = 2,
    DraftGen = 3,
}

// ---------------------------------------------------------------------------
// FFI function declarations
// ---------------------------------------------------------------------------

extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();

    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_context_default_params() -> LlamaContextParams;

    fn llama_model_load_from_file(
        path: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;
    fn llama_free_model(model: *mut LlamaModel);

    fn llama_init_from_model(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);

    fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> i32;

    fn llama_get_logits_ith(ctx: *mut LlamaContext, i: i32) -> *mut c_float;

    fn llama_batch_get_one(
        tokens: *mut i32,  // llama_token*
        n_tokens: i32,
        pos_0: i32,        // llama_pos
        seq_id: i32,       // llama_seq_id
    ) -> LlamaBatch;

    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    fn llama_batch_free(batch: LlamaBatch);

    fn llama_kv_cache_clear(ctx: *mut LlamaContext);

    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;
    fn llama_n_vocab(model: *const LlamaModel) -> i32;
    fn llama_n_layer(model: *const LlamaModel) -> i32;

    // -- Tokenizer / chat template (Step 6 of multi-arch refactor) --------
    //
    // These are needed by `GenericModel` (Step 7) which has no Rust-side
    // tokenizer.json file: it loads the tokenizer + chat template directly
    // from the GGUF metadata via libllama. Qwen35Model still uses
    // tokenizers::Tokenizer with a separate JSON, so these declarations are
    // additive — no existing code path consumes them yet.

    fn llama_tokenize(
        model: *const LlamaModel,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,           // llama_token*
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    fn llama_token_to_piece(
        model: *const LlamaModel,
        token: i32,                 // llama_token
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    fn llama_model_chat_template(
        model: *const LlamaModel,
        name: *const c_char,        // null = default template
    ) -> *const c_char;

    fn llama_chat_apply_template(
        tmpl: *const c_char,        // null = use model default
        chat: *const LlamaChatMessage,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: i32,
    ) -> i32;

    fn llama_get_timings(ctx: *mut LlamaContext) -> LlamaTimings;
    fn llama_print_timings(ctx: *mut LlamaContext);
    fn llama_reset_timings(ctx: *mut LlamaContext);

    fn llama_model_is_hybrid(model: *const LlamaModel) -> bool;
    fn llama_model_is_recurrent(model: *const LlamaModel) -> bool;
    fn llama_model_has_recurrent(model: *const LlamaModel) -> bool;

    // ggml function for CPU buffer type (needed for ncmoe overrides)
    fn ggml_backend_cpu_buffer_type() -> *mut GgmlBackendBufferType;

    // State serialization (for multi-agent context switching)
    fn llama_state_seq_get_size(ctx: *mut LlamaContext, seq_id: i32, flags: u32) -> usize;
    fn llama_state_seq_get_data(ctx: *mut LlamaContext, dst: *mut u8, size: usize, seq_id: i32, flags: u32) -> usize;
    fn llama_state_seq_set_data(ctx: *mut LlamaContext, src: *const u8, size: usize, seq_id: i32, flags: u32) -> usize;

    // MTP (Multi-Token Prediction)
    fn llama_model_n_nextn_layer(model: *const LlamaModel) -> i32;
    fn llama_set_mtp_op_type(ctx: *mut LlamaContext, mtp_op_type: i32);
    fn llama_set_draft_input_hidden_state(ctx: *mut LlamaContext, hidden_state: *const c_float);
    fn llama_get_embeddings(ctx: *mut LlamaContext) -> *mut c_float;
    fn llama_model_n_embd(model: *const LlamaModel) -> i32;

    // KV cache manipulation (for draft rejection)
    fn llama_kv_cache_seq_rm(ctx: *mut LlamaContext, seq_id: i32, p0: i32, p1: i32) -> bool;

    // chimere_sampler — C++ wrapper for fast sampling (avoids 993KB logits copy)
    fn chimere_sampler_init(
        model: *const LlamaModel,
        temperature: f32, top_p: f32, top_k: c_int,
        min_p: f32, presence_penalty: f32,
        dry_multiplier: f32, dry_base: f32,
        dry_min_length: c_int, dry_penalty_last_n: c_int,
    ) -> *mut c_void;
    fn chimere_sampler_sample(smpl: *mut c_void, ctx: *mut LlamaContext, idx: c_int) -> i32;
    fn chimere_sampler_sample_with_logprobs(
        smpl: *mut c_void, ctx: *mut LlamaContext, idx: c_int,
        result: *mut ChimereLogprobResult);
    fn chimere_sampler_accept(smpl: *mut c_void, ctx: *mut LlamaContext, token: i32);
    fn chimere_sampler_set_logit_bias(smpl: *mut c_void, token_id: i32, bias: f32);
    fn chimere_sampler_clear_logit_bias(smpl: *mut c_void);
    fn chimere_sampler_set_engram_bias(
        smpl: *mut c_void, token_ids: *const i32, biases: *const f32, n_entries: c_int);
    fn chimere_sampler_clear_engram_bias(smpl: *mut c_void);
    fn chimere_sampler_reset(smpl: *mut c_void);
    fn chimere_sampler_free(smpl: *mut c_void);
}

// ---------------------------------------------------------------------------
// M1 J3 — per-slot sampler allocation helpers (thin safe wrappers)
// ---------------------------------------------------------------------------

/// Allocate a fresh C++ sampler handle bound to `model`, using the same
/// Qwen3.5 tuned defaults as `LlamaForward::new`'s built-in sampler.
///
/// Used by `crate::multi_seq::MultiSeqDriver` at boot to allocate one
/// independent sampler per slot so that each concurrent request has its
/// own logit-bias / DRY history / repetition state.
///
/// # Safety
/// `model` must be a valid pointer obtained from `LlamaForward::model_raw`
/// and must remain valid for the lifetime of the returned handle. The
/// caller is responsible for calling [`chimere_sampler_free_handle`] on
/// every returned non-null pointer to avoid leaking C++ state.
pub unsafe fn chimere_sampler_alloc(
    model: *const LlamaModel,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    presence_penalty: f32,
) -> *mut c_void {
    chimere_sampler_init(
        model,
        temperature, top_p, top_k,
        min_p, presence_penalty,
        0.8,   // dry_multiplier — same defaults as `LlamaForward::new`
        1.75,  // dry_base
        2,     // dry_min_length
        -1,    // dry_penalty_last_n
    )
}

/// Like [`chimere_sampler_alloc`] but lets the caller override DRY
/// parameters. Use `dry_multiplier = 0.0` to disable DRY entirely —
/// useful for smoke tests that don't need repetition penalties, and for
/// smaller models whose tokenizer cannot resolve the default DRY
/// sequence-breakers (`\n`, `:`, `"`, `*`) without running into
/// `llama_sampler_init_dry`'s tokenisation path.
///
/// # Safety
/// Same as `chimere_sampler_alloc` — `model` must be a valid pointer
/// from `LlamaForward::model_raw()`.
pub unsafe fn chimere_sampler_alloc_with_dry(
    model: *const LlamaModel,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    presence_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_min_length: i32,
    dry_penalty_last_n: i32,
) -> *mut c_void {
    chimere_sampler_init(
        model,
        temperature, top_p, top_k,
        min_p, presence_penalty,
        dry_multiplier, dry_base, dry_min_length, dry_penalty_last_n,
    )
}

/// Free a sampler handle allocated by [`chimere_sampler_alloc`] or by the
/// built-in constructor in `LlamaForward::new`.
///
/// # Safety
/// `sampler` must be a valid pointer returned by
/// `chimere_sampler_init / chimere_sampler_alloc`. Double-free is UB.
pub unsafe fn chimere_sampler_free_handle(sampler: *mut c_void) {
    if !sampler.is_null() {
        chimere_sampler_free(sampler);
    }
}

/// Reset a per-slot sampler (clear DRY/repetition history).
///
/// # Safety
/// `sampler` must be a valid handle.
pub unsafe fn chimere_sampler_reset_handle(sampler: *mut c_void) {
    if !sampler.is_null() {
        chimere_sampler_reset(sampler);
    }
}

/// Clear the Engram biases attached to a per-slot sampler, preserving
/// manual `-inf` biases (e.g. `</think>` suppression).
///
/// # Safety
/// `sampler` must be a valid handle.
pub unsafe fn chimere_sampler_clear_engram_bias_handle(sampler: *mut c_void) {
    if !sampler.is_null() {
        chimere_sampler_clear_engram_bias(sampler);
    }
}

/// Push Engram biases into a per-slot sampler. `token_ids` and `biases`
/// must be parallel arrays of length `n_entries`. Merges with existing
/// biases — manual suppressions (e.g. `</think>` at `-inf`) are preserved;
/// previous Engram entries are overwritten. See `chimere_sampler.cpp` for
/// the exact merge semantics.
///
/// # Safety
/// `sampler` must be a valid handle. `token_ids` / `biases` must point to
/// arrays of at least `n_entries` elements, readable for the duration of
/// the call. Null sampler is a silent no-op.
pub unsafe fn chimere_sampler_set_engram_bias_handle(
    sampler: *mut c_void,
    token_ids: *const i32,
    biases: *const f32,
    n_entries: c_int,
) {
    if !sampler.is_null() && n_entries > 0 {
        chimere_sampler_set_engram_bias(sampler, token_ids, biases, n_entries);
    }
}

/// Set a single logit bias (e.g. `-inf` for `</think>` suppression) on a
/// per-slot sampler.
///
/// # Safety
/// `sampler` must be a valid handle.
pub unsafe fn chimere_sampler_set_logit_bias_handle(
    sampler: *mut c_void,
    token_id: i32,
    bias: f32,
) {
    if !sampler.is_null() {
        chimere_sampler_set_logit_bias(sampler, token_id, bias);
    }
}

// ---------------------------------------------------------------------------
// Safe wrapper
// ---------------------------------------------------------------------------

/// Safe wrapper around libllama's model + context for full forward pass delegation.
///
/// Owns the llama_model and llama_context pointers, freeing them on drop.
/// All state (KV cache, GDN recurrent state) is managed internally by libllama.
pub struct LlamaForward {
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    n_vocab: usize,
    n_embd: usize,
    pos: i32,
    mtp_available: bool,
    /// C++ sampler for fast token sampling (avoids 993KB logits copy)
    sampler: *mut c_void,
    /// Stored ncmoe CString patterns — must outlive the model load.
    _ncmoe_patterns: Vec<CString>,
    /// Stored override structs — must outlive the model load.
    _ncmoe_overrides: Vec<LlamaModelTensorBuftOverride>,
}

// Raw pointers are not Send/Sync by default, which is correct —
// we rely on the server's Mutex for thread safety.
// Explicitly mark as Send so it can be held inside tokio::sync::Mutex.
// Safety: single-threaded access enforced by the Mutex in server.rs.
unsafe impl Send for LlamaForward {}

impl LlamaForward {
    /// Load a model from GGUF and create a context for inference.
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF file
    /// * `n_gpu_layers` - Number of layers to offload to GPU (99 = all)
    /// * `n_ctx` - Context size (e.g. 65536 for 64K)
    /// * `ncmoe` - Number of layers whose MoE experts stay on CPU (0 = all on GPU)
    /// * `type_k` - KV cache key type (ggml_type, e.g. GGML_TYPE_Q8_0 = 8)
    /// * `type_v` - KV cache value type (ggml_type, e.g. GGML_TYPE_Q4_0 = 2)
    /// * `flash_attn` - Enable flash attention
    ///
    /// Single-sequence convenience wrapper over [`new_multi_seq`] with
    /// `n_seq_max = 1`. All existing call sites that do not care about
    /// multi-slot serving use this.
    pub fn new(
        model_path: &str,
        n_gpu_layers: i32,
        n_ctx: u32,
        ncmoe: u32,
        type_k: Option<i32>,
        type_v: Option<i32>,
        flash_attn: bool,
    ) -> Result<Self, String> {
        Self::new_multi_seq(
            model_path,
            n_gpu_layers,
            n_ctx,
            ncmoe,
            type_k,
            type_v,
            flash_attn,
            1,
        )
    }

    /// Identical to [`new`] but also sets `cparams.n_seq_max` to the
    /// requested value, making the libllama context ready to drive up to
    /// `n_seq_max` concurrent sequences (M1 multi-slot, J3+).
    ///
    /// `n_seq_max = 1` is the production default and behaves exactly like
    /// the pre-J3 [`new`] call. `n_seq_max >= 2` enables the multi-seq
    /// driver path (see `crate::multi_seq`).
    ///
    /// # Arguments
    /// Same as [`new`] plus:
    /// * `n_seq_max` - Maximum number of distinct sequence IDs the context
    ///   will serve concurrently. Recurrent/GDN models allocate one state
    ///   per seq; transformer KV pages are tagged per seq.
    #[allow(clippy::too_many_arguments)]
    pub fn new_multi_seq(
        model_path: &str,
        n_gpu_layers: i32,
        n_ctx: u32,
        ncmoe: u32,
        type_k: Option<i32>,
        type_v: Option<i32>,
        flash_attn: bool,
        n_seq_max: u32,
    ) -> Result<Self, String> {
        // Initialize the llama backend (idempotent, safe to call multiple times)
        unsafe {
            llama_backend_init();
        }

        let c_path = CString::new(model_path).map_err(|e| e.to_string())?;

        // --- Build ncmoe tensor_buft_overrides ---
        // For each layer 0..ncmoe, create a pattern that pins MoE expert
        // weights to CPU buffer, exactly like ik_llama's --n-cpu-moe.
        let cpu_buft = unsafe { ggml_backend_cpu_buffer_type() };
        let mut ncmoe_patterns: Vec<CString> = Vec::new();
        let mut ncmoe_overrides: Vec<LlamaModelTensorBuftOverride> = Vec::new();

        for layer in 0..ncmoe {
            let pattern_str = format!(
                "blk\\.{}\\.ffn_(up|down|gate)_exps\\.weight",
                layer
            );
            let c_pattern = CString::new(pattern_str).map_err(|e| e.to_string())?;
            ncmoe_patterns.push(c_pattern);
        }

        // Build override array — patterns must point into ncmoe_patterns
        for pattern in &ncmoe_patterns {
            ncmoe_overrides.push(LlamaModelTensorBuftOverride {
                pattern: pattern.as_ptr(),
                buft: cpu_buft,
            });
        }
        // Sentinel: NULL-terminated array
        ncmoe_overrides.push(LlamaModelTensorBuftOverride {
            pattern: std::ptr::null(),
            buft: std::ptr::null_mut(),
        });

        // --- Model params ---
        let mut mparams = unsafe { llama_model_default_params() };
        mparams.n_gpu_layers = n_gpu_layers;
        mparams.tensor_buft_overrides = ncmoe_overrides.as_ptr();
        // Enable MTP if the GGUF has MTP layers
        mparams.mtp = true;

        eprintln!(
            "[LLAMA_BACKEND] Loading model from {} (ngl={}, ncmoe={}, ctx={})...",
            model_path, n_gpu_layers, ncmoe, n_ctx,
        );

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), mparams) };
        if model.is_null() {
            return Err(format!("llama_model_load_from_file failed for {}", model_path));
        }

        // Report model type
        let is_hybrid = unsafe { llama_model_is_hybrid(model) };
        let is_recurrent = unsafe { llama_model_is_recurrent(model) };
        let has_recurrent = unsafe { llama_model_has_recurrent(model) };
        eprintln!(
            "[LLAMA_BACKEND] Model loaded: hybrid={}, recurrent={}, has_recurrent={}",
            is_hybrid, is_recurrent, has_recurrent,
        );

        // --- Context params ---
        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = n_ctx;
        cparams.n_batch = std::env::var("CHIMERE_BATCH")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        cparams.n_ubatch = std::env::var("CHIMERE_UBATCH")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
        // M1 J3: expose concurrent-sequence count to libllama. Recurrent/GDN
        // models allocate one SSM state per seq; transformer KV pages are
        // tagged per seq. Clamped to >=1 so legacy callers (n_seq_max=1) are
        // unchanged bit-for-bit.
        cparams.n_seq_max = n_seq_max.max(1);
        cparams.n_threads = std::env::var("CHIMERE_THREADS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(14);
        cparams.n_threads_batch = std::env::var("CHIMERE_THREADS_BATCH")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(14);
        cparams.flash_attn = flash_attn;
        cparams.offload_kqv = true;
        // KV cache types: default to production settings (q8_0 keys, q4_0 values)
        cparams.type_k = type_k.unwrap_or(GGML_TYPE_Q8_0);
        cparams.type_v = type_v.unwrap_or(GGML_TYPE_Q4_0);
        // Hadamard rotation on K cache (improves quantized KV quality)
        let k_cache_hadamard = std::env::var("CHIMERE_KV_HADAMARD")
            .map(|v| v != "0")
            .unwrap_or(true);
        cparams.k_cache_hadamard = k_cache_hadamard;
        // Enable MTP in context params (needed for embeddings + logits buffer allocation)
        cparams.mtp = true;

        // Save fields we want to log after the ctx init (cparams is moved
        // into the FFI call so reading it after would borrow-after-move).
        let logged_n_seq_max = cparams.n_seq_max;

        let ctx = unsafe { llama_init_from_model(model, cparams) };
        if ctx.is_null() {
            unsafe {
                llama_free_model(model);
            }
            return Err("llama_init_from_model failed".into());
        }

        // Get vocab size and embedding dimension
        let n_vocab = unsafe { llama_n_vocab(model) } as usize;
        let n_embd = unsafe { llama_model_n_embd(model) } as usize;
        let n_mtp_layers = unsafe { llama_model_n_nextn_layer(model) };
        let mtp_available = n_mtp_layers > 0;

        if mtp_available {
            // Enable MTP in context params — needed for has_mtp checks + embedding extraction
            unsafe { llama_set_mtp_op_type(ctx, MtpOp::None as i32); }
            eprintln!("[LLAMA_BACKEND] MTP available: {} nextn layers, n_embd={}", n_mtp_layers, n_embd);
        }

        eprintln!(
            "[LLAMA_BACKEND] Context created: n_ctx={}, n_seq_max={}, n_batch=4096, \
             n_ubatch=512, flash_attn={}, type_k={}, type_v={}, k_cache_hadamard={}, \
             n_vocab={}, mtp={}",
            n_ctx, logged_n_seq_max, flash_attn,
            type_k.unwrap_or(GGML_TYPE_Q8_0),
            type_v.unwrap_or(GGML_TYPE_Q4_0),
            k_cache_hadamard,
            n_vocab, mtp_available,
        );

        // Reset timings for clean measurement
        unsafe {
            llama_reset_timings(ctx);
        }

        // Create C++ sampler with Qwen3.5 optimal params (thinking mode)
        // Ref: Unsloth docs + Qwen3.5 official sampling recommendations
        //
        // J3 smoke path: set CHIMERE_SKIP_SAMPLER_INIT=1 to skip common_sampler
        // construction. Useful when exercising forward_multi_seq alone (j3-smoke
        // does argmax externally and doesn't need the chimere sampler). Also
        // avoids the libcommon grammar-path symbol that ik_llama doesn't expose.
        //
        // Value "0" or "false" means DO NOT skip — allocate normally. This
        // matters for j5-smoke which sets the var to 0 to force sampler
        // allocation even when the invoking shell had it set to 1.
        let skip_sampler = std::env::var("CHIMERE_SKIP_SAMPLER_INIT")
            .map(|v| {
                let v = v.trim();
                !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(false);
        let sampler = if skip_sampler {
            eprintln!("[LLAMA_BACKEND] CHIMERE_SKIP_SAMPLER_INIT=1 → skipping C++ sampler");
            std::ptr::null_mut()
        } else {
            unsafe {
                chimere_sampler_init(
                    model as *const LlamaModel,
                    0.6,   // temperature — Qwen3.5 thinking mode (0.7 for chat)
                    0.95,  // top_p
                    20,    // top_k
                    0.05,  // min_p — better than top_k alone (2026 consensus)
                    0.0,   // presence_penalty (0 for thinking, 1.5 if loops)
                    0.8,   // dry_multiplier — prevents thinking loops
                    1.75,  // dry_base
                    2,     // dry_min_length
                    -1,    // dry_penalty_last_n (-1 = whole sequence)
                )
            }
        };
        if sampler.is_null() {
            eprintln!("[LLAMA_BACKEND] Warning: C++ sampler init skipped/failed, falling back to Rust sampling");
        } else {
            eprintln!("[LLAMA_BACKEND] C++ sampler initialized (avoids 993KB logits copy per token)");
        }

        Ok(Self {
            model,
            ctx,
            n_vocab,
            n_embd,
            pos: 0,
            mtp_available,
            sampler,
            _ncmoe_patterns: ncmoe_patterns,
            _ncmoe_overrides: ncmoe_overrides,
        })
    }

    /// Forward pass for a single token. Returns raw logits (pre-softmax).
    ///
    /// The position is auto-incremented. libllama manages KV cache and
    /// recurrent state internally.
    ///
    /// # Returns
    /// Vec<f32> of length n_vocab containing unnormalized log-probabilities.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let mut tok = token as i32;
        let batch = unsafe { llama_batch_get_one(&mut tok, 1, self.pos, 0) };

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            return Err(format!(
                "llama_decode failed with code {} at pos={}",
                ret, self.pos
            ));
        }

        // llama_batch_get_one sets logits for the last token only.
        // With n_tokens=1, the logits are at index 0 in the output buffer.
        // Use index -1 (last token) which is the recommended approach.
        let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, -1) };
        if logits_ptr.is_null() {
            return Err("llama_get_logits_ith returned null".into());
        }

        let logits = unsafe { std::slice::from_raw_parts(logits_ptr, self.n_vocab) };
        self.pos += 1;

        Ok(logits.to_vec())
    }

    /// Forward pass for a single token WITHOUT copying logits.
    /// Use with sample_token_fast() for zero-copy sampling.
    pub fn forward_token_no_logits(&mut self, token: u32) -> Result<(), String> {
        let mut tok = token as i32;
        let batch = unsafe { llama_batch_get_one(&mut tok, 1, self.pos, 0) };

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            return Err(format!("llama_decode failed with code {} at pos={}", ret, self.pos));
        }
        self.pos += 1;
        Ok(())
    }

    /// Prefill without copying logits (for use with sample_token_fast).
    pub fn forward_prefill_no_logits(&mut self, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("forward_prefill_no_logits: empty token slice".into());
        }
        let mut toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let batch = unsafe { llama_batch_get_one(toks.as_mut_ptr(), toks.len() as i32, self.pos, 0) };
        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            return Err(format!("llama_decode failed with code {} at pos={}", ret, self.pos));
        }
        self.pos += toks.len() as i32;
        Ok(())
    }

    /// Batch-verify draft tokens for speculative decoding (DART).
    ///
    /// Decodes N draft tokens in a single llama_decode call and returns
    /// the logits at each position. The caller can then check if each
    /// draft token matches the argmax of the logits at that position.
    ///
    /// Uses llama_batch_init with per-position logits enabled.
    /// Returns Vec of (position, logits_vec) for each draft token.
    ///
    /// On success, self.pos is advanced by the number of ACCEPTED tokens + 1.
    pub fn forward_batch_verify(&mut self, draft_tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        if draft_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let n = draft_tokens.len() as i32;
        let mut batch = unsafe { llama_batch_init(n, 0, 1) };

        // Fill batch: each draft token at its position, all requesting logits
        unsafe {
            batch.n_tokens = n;
            for (i, &tok) in draft_tokens.iter().enumerate() {
                *batch.token.add(i) = tok as i32;
                *batch.pos.add(i) = self.pos + i as i32;
                *batch.n_seq_id.add(i) = 1;
                let seq_ptr = *batch.seq_id.add(i);
                *seq_ptr = 0; // seq_id = 0
                *batch.logits.add(i) = 1; // request logits for ALL positions
            }
        }

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            unsafe { llama_batch_free(batch); }
            return Err(format!("llama_decode batch verify failed: code {}", ret));
        }

        // Extract logits at each position
        let mut all_logits = Vec::with_capacity(draft_tokens.len());
        for i in 0..draft_tokens.len() {
            let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, i as i32) };
            if logits_ptr.is_null() {
                unsafe { llama_batch_free(batch); }
                return Err(format!("null logits at position {}", i));
            }
            let logits = unsafe { std::slice::from_raw_parts(logits_ptr, self.n_vocab) };
            all_logits.push(logits.to_vec());
        }

        unsafe { llama_batch_free(batch); }

        // Don't advance pos here — caller decides how many tokens to accept
        Ok(all_logits)
    }

    /// Accept N tokens after successful batch verification.
    /// Advances pos by count.
    pub fn accept_draft_tokens(&mut self, count: usize) {
        self.pos += count as i32;
    }

    // ---------- M1 J3 — true multi-seq forward pass ----------
    //
    // Unlike [`forward_prefill`] / [`forward_token`], these entry points do
    // NOT touch `self.pos`. Each slot/seq_id tracks its own position and
    // passes it explicitly. A single `llama_decode` call handles N slots,
    // each tagged with its own `seq_id` — KV pages are segregated by seq.

    /// Decode one batch that mixes tokens from **multiple** sequences.
    ///
    /// This is the core primitive for M1 multi-slot serving: the scheduler
    /// collects one token per active slot, tags each with the slot's
    /// `seq_id`, and calls this once per scheduler step. The libllama
    /// KV cache automatically routes K/V writes to per-seq pages
    /// (transformer path) or to per-seq SSM states (GDN/Mamba path).
    ///
    /// # Preconditions
    /// - Context was built with `new_multi_seq(n_seq_max)` where
    ///   `n_seq_max > max(entry.seq_id)`. `new(...)` allocates n_seq_max=1
    ///   and will reject entries with seq_id > 0.
    /// - Positions per `seq_id` are monotonically increasing across calls
    ///   (libllama does not reorder within a seq).
    /// - `entries.len()` must be ≤ `cparams.n_batch`.
    ///
    /// # Returns
    /// Logits (size `n_vocab`) for every entry where `request_logits=true`,
    /// tagged with the corresponding `seq_id`. Order matches batch index,
    /// which matches the order of `request_logits=true` entries in input.
    ///
    /// Entries with `request_logits=false` (e.g. all-but-last tokens of a
    /// prefill chunk) are processed by the model but produce no output.
    pub fn forward_multi_seq(
        &mut self,
        entries: &[MultiSeqEntry],
    ) -> Result<Vec<(i32, Vec<f32>)>, String> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let n = entries.len() as i32;
        // llama_batch_init's 3rd arg is `n_seq_max` per token, not total.
        // Each entry carries exactly one seq_id, so 1 is correct here.
        let mut batch = unsafe { llama_batch_init(n, 0, 1) };

        unsafe {
            batch.n_tokens = n;
            for (i, e) in entries.iter().enumerate() {
                *batch.token.add(i) = e.token as i32;
                *batch.pos.add(i) = e.pos;
                *batch.n_seq_id.add(i) = 1;
                let seq_ptr = *batch.seq_id.add(i);
                *seq_ptr = e.seq_id;
                *batch.logits.add(i) = if e.request_logits { 1 } else { 0 };
            }
        }

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            unsafe { llama_batch_free(batch); }
            return Err(format!("llama_decode multi_seq failed: code {}", ret));
        }

        // Collect logits in batch order, only for the requested entries.
        let mut result: Vec<(i32, Vec<f32>)> = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            if !e.request_logits { continue; }
            let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, i as i32) };
            if logits_ptr.is_null() {
                unsafe { llama_batch_free(batch); }
                return Err(format!(
                    "null logits at batch idx {} seq_id {}",
                    i, e.seq_id,
                ));
            }
            let logits = unsafe { std::slice::from_raw_parts(logits_ptr, self.n_vocab) };
            result.push((e.seq_id, logits.to_vec()));
        }

        unsafe { llama_batch_free(batch); }
        Ok(result)
    }

    /// FINDING 2 (audit 2026-04-24): no-copy variant of
    /// [`forward_multi_seq`]. Decodes the multi-seq batch identically but
    /// returns only `(seq_id, batch_idx)` pairs instead of copying the
    /// per-slot 993 KB (n_vocab=248320) logits buffer out of libllama's
    /// internal tensor.
    ///
    /// The native scheduler only needs the presence check; actual logits
    /// are read inside the C++ sampler via `llama_get_logits_ith` when
    /// `sample_slot_with_logprobs(batch_idx)` runs. Keeping a borrowed
    /// pointer alive between `forward_multi_seq_borrow` and the sampler
    /// call is safe because:
    ///   - `self` (owning the `ctx`) remains borrowed `&mut` for the
    ///     entire scheduler tick, so no other FFI call can mutate the
    ///     internal logits buffer mid-flight;
    ///   - the sampler reads from the same `ctx` via `llama_get_logits_ith`
    ///     which returns a pointer into the same buffer the non-copy
    ///     variant would have dereferenced.
    ///
    /// `batch_idx` here is the index into the input `entries` slice (the
    /// same value the copying variant would have used internally before
    /// `to_vec()`).
    ///
    /// Legacy callers that need owned `Vec<f32>` logits should keep using
    /// [`forward_multi_seq`]. This method is purely additive.
    pub fn forward_multi_seq_borrow(
        &mut self,
        entries: &[MultiSeqEntry],
    ) -> Result<Vec<(i32, usize)>, String> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let n = entries.len() as i32;
        let mut batch = unsafe { llama_batch_init(n, 0, 1) };

        unsafe {
            batch.n_tokens = n;
            for (i, e) in entries.iter().enumerate() {
                *batch.token.add(i) = e.token as i32;
                *batch.pos.add(i) = e.pos;
                *batch.n_seq_id.add(i) = 1;
                let seq_ptr = *batch.seq_id.add(i);
                *seq_ptr = e.seq_id;
                *batch.logits.add(i) = if e.request_logits { 1 } else { 0 };
            }
        }

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            unsafe { llama_batch_free(batch); }
            return Err(format!("llama_decode multi_seq_borrow failed: code {}", ret));
        }

        // Presence check: each requested entry must have non-null logits.
        // We do NOT copy the 248 K-float buffer — the sampler reads it
        // directly from libllama's internal ctx tensor via
        // llama_get_logits_ith(batch_idx).
        let mut result: Vec<(i32, usize)> = Vec::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            if !e.request_logits { continue; }
            let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, i as i32) };
            if logits_ptr.is_null() {
                unsafe { llama_batch_free(batch); }
                return Err(format!(
                    "null logits at batch idx {} seq_id {}",
                    i, e.seq_id,
                ));
            }
            result.push((e.seq_id, i));
        }

        unsafe { llama_batch_free(batch); }
        Ok(result)
    }

    /// Release all KV cache pages (and SSM state, if recurrent) owned by
    /// `seq_id`. Call when a slot finishes or the client disconnects.
    ///
    /// Safe to call on a seq_id with no allocated pages (no-op, returns
    /// false).
    pub fn kv_cache_seq_rm_for(&mut self, seq_id: i32) -> bool {
        unsafe { llama_kv_cache_seq_rm(self.ctx, seq_id, -1, -1) }
    }

    /// Public `n_vocab` for callers that receive raw logits from
    /// [`forward_multi_seq`] and need to slice them.
    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    /// Accept a token into the sampler's history (for DRY/repetition tracking).
    pub fn sampler_accept(&self, token: u32) {
        if !self.sampler.is_null() {
            unsafe { chimere_sampler_accept(self.sampler, self.ctx, token as i32); }
        }
    }

    /// Prefill a sequence of tokens. Returns logits for the LAST token only.
    ///
    /// Uses llama_batch_get_one which auto-sets positions starting at self.pos.
    /// For large prompts (> n_batch), this should be split into chunks —
    /// but for typical use (< 4096 tokens), a single call suffices.
    ///
    /// # Returns
    /// Vec<f32> of length n_vocab (logits for the last prompt token).
    pub fn forward_prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("forward_prefill: empty token slice".into());
        }

        let n = tokens.len();
        let mut toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

        // Process in chunks of n_batch to respect the batch size limit.
        let n_batch = std::env::var("CHIMERE_BATCH")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(4096usize);
        let mut last_logits: Option<Vec<f32>> = None;

        let mut offset = 0;
        while offset < n {
            let chunk_size = (n - offset).min(n_batch);
            let chunk = &mut toks[offset..offset + chunk_size];

            let batch = unsafe {
                llama_batch_get_one(
                    chunk.as_mut_ptr(),
                    chunk_size as i32,
                    self.pos,
                    0,
                )
            };

            let ret = unsafe { llama_decode(self.ctx, batch) };
            if ret != 0 {
                return Err(format!(
                    "llama_decode failed with code {} during prefill (pos={}, chunk={})",
                    ret, self.pos, chunk_size,
                ));
            }

            self.pos += chunk_size as i32;
            offset += chunk_size;

            // Extract logits from the last chunk
            if offset == n {
                let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, -1) };
                if logits_ptr.is_null() {
                    return Err("llama_get_logits_ith returned null during prefill".into());
                }
                let logits =
                    unsafe { std::slice::from_raw_parts(logits_ptr, self.n_vocab) };
                last_logits = Some(logits.to_vec());
            }
        }

        last_logits.ok_or_else(|| "forward_prefill: no logits produced".into())
    }

    /// Reset all state (KV cache + recurrent state + sampler) for a new conversation.
    /// Save sequence state (KV + GDN recurrent) for multi-agent context switching.
    pub fn state_seq_save(&self, seq_id: i32) -> Result<Vec<u8>, String> {
        let size = unsafe { llama_state_seq_get_size(self.ctx, seq_id, 0) };
        if size == 0 { return Ok(vec![]); }
        let mut buf = vec![0u8; size];
        let written = unsafe { llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, seq_id, 0) };
        if written == 0 { return Err("state_seq_get_data failed".into()); }
        buf.truncate(written);
        Ok(buf)
    }

    /// Restore a previously saved sequence state.
    pub fn state_seq_restore(&mut self, seq_id: i32, data: &[u8]) -> Result<(), String> {
        if data.is_empty() { return Ok(()); }
        let loaded = unsafe { llama_state_seq_set_data(self.ctx, data.as_ptr(), data.len(), seq_id, 0) };
        if loaded == 0 { return Err("state_seq_set_data failed".into()); }
        Ok(())
    }

    /// Set position counter (for agent context restore).
    pub fn set_pos(&mut self, pos: i32) { self.pos = pos; }

    // ---------- M2-J2b — prefix-cache FFI aliases ----------
    //
    // The underlying `state_seq_save` / `state_seq_restore` already exist
    // for multi-agent switching (see `agent_scheduler.rs`). M2 reuses them
    // for prompt-prefix caching, exposed here under scheduler-friendly
    // names so call sites in `slot_scheduler::NativeDriver` read as
    // "cache save/restore" rather than "agent save/restore". Behaviour is
    // bit-identical; the distinct names keep a rename-free path to the
    // existing `agent_scheduler.rs` consumers.

    /// Save the KV cache + GDN recurrent state for `seq_id` as an opaque
    /// blob. Returns `Err` on FFI failure. A zero-byte seq (no tokens
    /// decoded) returns `Ok(vec![])`.
    ///
    /// Used by `NativeDriver::reap_draining` (M2-J2c) to snapshot a slot's
    /// cache before freeing it, so the trie can retain the prefix for the
    /// next request that shares it.
    ///
    /// # Blob format
    ///
    /// The returned bytes are ik_llama-internal: per-layer KV pages,
    /// GDN recurrent matrices, and sampler position markers. The blob
    /// is **seq_id-independent** — a blob saved from `seq_id=0` restores
    /// cleanly into any other slot's `seq_id` (verified by the agent
    /// switcher since Mar 2026).
    pub fn save_seq_state(&self, seq_id: i32) -> Result<Vec<u8>, String> {
        self.state_seq_save(seq_id)
    }

    /// Restore a previously saved blob into `seq_id`. Caller MUST also
    /// call [`set_pos`](Self::set_pos) with the token count covered by
    /// the blob (i.e. `KVBlock::token_count`) so the Rust-side position
    /// counter agrees with the restored KV extent.
    ///
    /// Canonical restore pattern (copied from `agent_scheduler.rs:189`):
    ///
    /// ```ignore
    /// llama.restore_seq_state(slot_seq_id, &block.seq_bytes)?;
    /// llama.set_pos(block.token_count as i32);
    /// ```
    ///
    /// The scheduler MUST also replay `tokens[..block.token_count]`
    /// through `Slot::push_context` to rebuild the engram n-gram
    /// history — see plan-M2-prefix-cache.md § 5.
    pub fn restore_seq_state(&mut self, seq_id: i32, blob: &[u8]) -> Result<(), String> {
        self.state_seq_restore(seq_id, blob)
    }

    /// Read the Rust-side position counter. `NativeDriver` does NOT use
    /// this accessor (it tracks per-slot positions via `Slot.pos`), but
    /// the single-slot path (`AppStateModel`) occasionally introspects it
    /// for diagnostics, and M2 consumers may find it useful for asserts.
    pub fn current_pos(&self) -> i32 { self.pos }

    pub fn reset(&mut self) {
        unsafe {
            llama_kv_cache_clear(self.ctx);
            if !self.sampler.is_null() {
                chimere_sampler_reset(self.sampler);
            }
        }
        self.pos = 0;
    }

    /// Get the vocab size (for logits array sizing).
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Fast token sampling via C++ (no 993KB logits copy).
    /// Returns sampled token ID. Uses the C++ common_sampler.
    pub fn sample_token_fast(&mut self) -> Result<u32, String> {
        if self.sampler.is_null() {
            return Err("C++ sampler not initialized".into());
        }
        let token = unsafe {
            chimere_sampler_sample(self.sampler, self.ctx, -1)
        };
        unsafe {
            chimere_sampler_accept(self.sampler, self.ctx, token);
        }
        Ok(token as u32)
    }

    /// Fast sampling with top-5 logprobs (for ABF entropy/confidence).
    pub fn sample_token_fast_with_logprobs(&mut self) -> Result<(u32, Vec<TokenLogprob>), String> {
        if self.sampler.is_null() {
            return Err("C++ sampler not initialized".into());
        }
        let mut result = ChimereLogprobResult {
            token_id: 0, n_top: 0,
            top_tokens: [0; 5], top_logprobs: [0.0; 5],
        };
        unsafe {
            chimere_sampler_sample_with_logprobs(self.sampler, self.ctx, -1, &mut result);
            chimere_sampler_accept(self.sampler, self.ctx, result.token_id);
        }
        let mut logprobs = Vec::with_capacity(result.n_top as usize);
        for i in 0..result.n_top as usize {
            logprobs.push(TokenLogprob {
                token: result.top_tokens[i] as u32,
                logprob: result.top_logprobs[i],
            });
        }
        Ok((result.token_id as u32, logprobs))
    }

    /// Suppress a token (e.g., </think> in response mode).
    pub fn set_logit_bias(&self, token_id: u32, bias: f32) {
        if !self.sampler.is_null() {
            unsafe { chimere_sampler_set_logit_bias(self.sampler, token_id as i32, bias); }
        }
    }

    /// Clear all logit biases.
    pub fn clear_logit_bias(&self) {
        if !self.sampler.is_null() {
            unsafe { chimere_sampler_clear_logit_bias(self.sampler); }
        }
    }

    /// Set Engram logit biases from n-gram predictions.
    /// Merges with existing biases (preserves </think> suppression).
    pub fn set_engram_bias(&self, predictions: &[(u32, f32)]) {
        if self.sampler.is_null() || predictions.is_empty() { return; }
        let token_ids: Vec<i32> = predictions.iter().map(|(t, _)| *t as i32).collect();
        let biases: Vec<f32> = predictions.iter().map(|(_, b)| *b).collect();
        unsafe {
            chimere_sampler_set_engram_bias(
                self.sampler,
                token_ids.as_ptr(),
                biases.as_ptr(),
                predictions.len() as c_int,
            );
        }
    }

    /// Clear only Engram biases (keep manual biases like </think>).
    pub fn clear_engram_bias(&self) {
        if !self.sampler.is_null() {
            unsafe { chimere_sampler_clear_engram_bias(self.sampler); }
        }
    }

    /// Check if sampler is available (for fallback logic).
    pub fn has_fast_sampler(&self) -> bool {
        !self.sampler.is_null()
    }

    /// Get the current position counter.
    pub fn pos(&self) -> i32 {
        self.pos
    }

    /// Print performance timings to stderr (llama.cpp format).
    pub fn print_timings(&self) {
        unsafe {
            llama_print_timings(self.ctx);
        }
    }

    /// Get performance timings.
    pub fn get_timings(&self) -> LlamaTimings {
        unsafe { llama_get_timings(self.ctx) }
    }

    /// Reset timings for a fresh measurement period.
    pub fn reset_timings(&self) {
        unsafe {
            llama_reset_timings(self.ctx);
        }
    }

    // -----------------------------------------------------------------
    // MTP (Multi-Token Prediction) methods
    // -----------------------------------------------------------------

    /// Whether the loaded model has MTP layers.
    pub fn has_mtp(&self) -> bool {
        self.mtp_available
    }

    /// Get embedding dimension.
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Set MTP operation mode for the next decode call.
    pub fn set_mtp_op(&mut self, op: MtpOp) {
        unsafe { llama_set_mtp_op_type(self.ctx, op as i32); }
    }

    /// Get hidden state embeddings from the last decode (n_embd floats).
    /// Returns None if embeddings are not available.
    pub fn get_embeddings(&self) -> Option<Vec<f32>> {
        let ptr = unsafe { llama_get_embeddings(self.ctx) };
        if ptr.is_null() {
            return None;
        }
        let embd = unsafe { std::slice::from_raw_parts(ptr, self.n_embd) };
        Some(embd.to_vec())
    }

    /// Get raw embedding pointer (zero-copy, for passing to set_draft_hidden_state).
    pub fn get_embeddings_ptr(&self) -> *const c_float {
        unsafe { llama_get_embeddings(self.ctx) }
    }

    /// Set hidden state for MTP DRAFT_GEN operation.
    pub fn set_draft_hidden_state(&mut self, hidden_state: *const c_float) {
        unsafe { llama_set_draft_input_hidden_state(self.ctx, hidden_state); }
    }

    /// Forward pass with MTP: main decode → get embeddings → MTP draft decode → MTP logits.
    ///
    /// Returns (main_logits, mtp_logits_option).
    /// MTP logits are only returned if the model has MTP and the main decode produced embeddings.
    pub fn forward_token_with_mtp(&mut self, token: u32) -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        // 1. Main decode (MTP_OP_NONE)
        self.set_mtp_op(MtpOp::None);
        let main_logits = self.forward_token(token)?;

        if !self.mtp_available {
            return Ok((main_logits, None));
        }

        // 2. Get embeddings from the main decode
        let embd_ptr = self.get_embeddings_ptr();
        if embd_ptr.is_null() {
            return Ok((main_logits, None));
        }

        // 3. MTP decode — uses WARMUP which reads from lctx.embd automatically.
        //    WARMUP also initializes/updates MTP's own KV cache.
        //    (DRAFT_GEN requires separate hidden state + pre-initialized KV — more complex)
        self.set_mtp_op(MtpOp::Warmup);

        // MTP decode at same position as main decode (pos was already incremented)
        let mut tok = token as i32;
        let batch = unsafe { llama_batch_get_one(&mut tok, 1, self.pos - 1, 0) };
        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            // MTP decode failed — return main logits only
            eprintln!("[MTP] Draft gen decode failed: code {}", ret);
            self.set_mtp_op(MtpOp::None);
            return Ok((main_logits, None));
        }

        // 4. Extract MTP logits
        let mtp_ptr = unsafe { llama_get_logits_ith(self.ctx, -1) };
        let mtp_logits = if mtp_ptr.is_null() {
            None
        } else {
            let mtp = unsafe { std::slice::from_raw_parts(mtp_ptr, self.n_vocab) };
            Some(mtp.to_vec())
        };

        // Reset to normal mode
        self.set_mtp_op(MtpOp::None);

        Ok((main_logits, mtp_logits))
    }

    /// Remove KV cache entries for a position range [p0, p1).
    /// Used to undo rejected draft tokens from the attention KV cache.
    ///
    /// NOTE: hardcodes `seq_id=0` for the single-slot path. Multi-slot
    /// callers must use [`kv_cache_seq_rm_seq`] to target a specific
    /// sequence ID.
    pub fn kv_cache_seq_rm(&mut self, p0: i32, p1: i32) -> bool {
        unsafe { llama_kv_cache_seq_rm(self.ctx, 0, p0, p1) }
    }

    /// Multi-seq variant of [`kv_cache_seq_rm`]. Removes KV cache entries
    /// for a specific sequence ID over a position range.
    ///
    /// Conventions (match ik_llama's `llama_kv_cache_seq_rm` C API):
    /// - `seq_id < 0`: match any sequence
    /// - `p0 < 0`: from position 0
    /// - `p1 < 0`: to +inf
    ///
    /// For slot cleanup on request completion: call with `(slot.seq_id, 0, -1)`.
    pub fn kv_cache_seq_rm_seq(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        unsafe { llama_kv_cache_seq_rm(self.ctx, seq_id, p0, p1) }
    }

    /// M1 J3 multi-seq batch decode. Composes a single `llama_batch` from
    /// an arbitrary per-token (seq_id, pos, want_logits) layout and invokes
    /// `llama_decode` once.
    ///
    /// The returned `Vec` has one entry per input token, in input order,
    /// giving the `batch_idx` that should be fed to [`sample_slot`] or
    /// [`get_logits_at`] to retrieve that token's logits (or `None` if the
    /// token did not request logits).
    ///
    /// # Arguments
    /// * `tokens` - Slice of `(token_id, pos, seq_id, want_logits)` tuples.
    ///   Must not exceed `cparams.n_batch`.
    ///
    /// # Returns
    /// * `Ok(Vec<Option<usize>>)` - Per-token batch index for logits lookup,
    ///   or `None` if the token did not request logits.
    /// * `Err(String)` on libllama failure.
    ///
    /// # Safety notes
    /// The caller is responsible for tracking each sequence's position
    /// counter. This method does NOT mutate `self.pos` because in multi-seq
    /// mode `self.pos` has no meaning (see `multi_seq::MultiSeqSlot.pos`).
    pub fn forward_batch_multiseq(
        &mut self,
        tokens: &[(u32, i32, i32, bool)],
    ) -> Result<Vec<Option<usize>>, String> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let n = tokens.len() as i32;
        // `n_seq_max=1`: each token only tagged with 1 seq ID. We do not
        // currently use broadcast-to-many-seqs (that's a future speculative
        // decoding feature).
        let mut batch = unsafe { llama_batch_init(n, 0, 1) };

        let mut logits_indices: Vec<Option<usize>> = Vec::with_capacity(tokens.len());

        unsafe {
            batch.n_tokens = n;
            for (i, &(tok, pos, seq_id, want_logits)) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok as i32;
                *batch.pos.add(i) = pos;
                *batch.n_seq_id.add(i) = 1;
                let seq_ptr = *batch.seq_id.add(i);
                *seq_ptr = seq_id;
                *batch.logits.add(i) = if want_logits { 1 } else { 0 };
                logits_indices.push(if want_logits { Some(i) } else { None });
            }
        }

        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            unsafe { llama_batch_free(batch); }
            return Err(format!(
                "llama_decode (multi-seq) failed with code {} for n_tokens={}",
                ret, n
            ));
        }

        unsafe { llama_batch_free(batch); }
        Ok(logits_indices)
    }

    /// Retrieve logits for the token at batch index `i` from the most
    /// recent `forward_batch_multiseq` call.
    ///
    /// Returns a borrowed slice into libllama's internal buffer — valid
    /// only until the next `llama_decode` call. Caller must clone if
    /// longer-lived storage is needed.
    pub fn get_logits_at(&self, batch_idx: usize) -> Option<&[f32]> {
        let ptr = unsafe { llama_get_logits_ith(self.ctx, batch_idx as i32) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) })
    }

    /// Raw context pointer — used by multi-seq callers that need to allocate
    /// per-slot C++ samplers bound to this context. All usage sites must
    /// document the unsafe access pattern.
    ///
    /// # Safety
    /// The returned pointer is only valid while `self` is live. Do not
    /// retain it across drops of `LlamaForward`.
    pub unsafe fn ctx_raw(&self) -> *mut LlamaContext {
        self.ctx
    }

    /// Raw model pointer — needed by multi-seq callers to allocate per-slot
    /// samplers with `chimere_sampler_init(model, ...)`.
    ///
    /// # Safety
    /// The returned pointer is only valid while `self` is live.
    pub unsafe fn model_raw(&self) -> *const LlamaModel {
        self.model as *const LlamaModel
    }

    /// Sample with a caller-provided per-slot sampler handle at a specific
    /// batch index. Used by the multi-seq driver to invoke per-slot samplers
    /// on the shared context.
    ///
    /// # Safety
    /// `sampler` must be a valid pointer returned by
    /// `chimere_sampler_init(...)` and bound to this model.
    pub unsafe fn sample_slot(&self, sampler: *mut c_void, batch_idx: usize) -> u32 {
        let tok = chimere_sampler_sample(sampler, self.ctx, batch_idx as c_int);
        chimere_sampler_accept(sampler, self.ctx, tok);
        tok as u32
    }

    /// Sample at `batch_idx` using a per-slot sampler AND return top-5
    /// logprobs reflecting the sampler's current logit-bias state. The
    /// J5 smoke uses this to observe that per-slot engram biases yield
    /// distinct distributions across slots (not just distinct argmaxes).
    ///
    /// # Safety
    /// `sampler` must be a valid pointer returned by
    /// `chimere_sampler_init(...)` and bound to this model's context.
    pub unsafe fn sample_slot_with_logprobs(
        &self,
        sampler: *mut c_void,
        batch_idx: usize,
    ) -> (u32, Vec<TokenLogprob>) {
        let mut result = ChimereLogprobResult {
            token_id: 0,
            n_top: 0,
            top_tokens: [0; 5],
            top_logprobs: [0.0; 5],
        };
        chimere_sampler_sample_with_logprobs(sampler, self.ctx, batch_idx as c_int, &mut result);
        chimere_sampler_accept(sampler, self.ctx, result.token_id);
        let mut logprobs = Vec::with_capacity(result.n_top as usize);
        for i in 0..result.n_top as usize {
            logprobs.push(TokenLogprob {
                token: result.top_tokens[i] as u32,
                logprob: result.top_logprobs[i],
            });
        }
        (result.token_id as u32, logprobs)
    }

    /// Rollback position counter by n tokens.
    pub fn rollback_pos(&mut self, n: i32) {
        self.pos -= n;
    }

    /// Number of main transformer/SSM layers in the loaded model. Equivalent
    /// to the GGUF metadata field `<arch>.block_count`. Used by `GenericModel`
    /// to satisfy `ChimereModel::num_layers`.
    pub fn n_layer(&self) -> usize {
        unsafe { llama_n_layer(self.model) as usize }
    }

    // ----------------------------------------------------------------------
    // Tokenizer / chat template helpers (Step 6 of multi-arch refactor)
    // ----------------------------------------------------------------------
    //
    // These wrappers expose libllama's GGUF-embedded tokenizer + chat template
    // engine to the Rust side. They are needed by `GenericModel` (Step 7) for
    // architectures that don't ship with a separate `tokenizer.json` file
    // (Mamba, Nemotron-h, etc.). `Qwen35Model` keeps using `tokenizers::Tokenizer`
    // with its own JSON for backward compatibility — these helpers do not
    // alter any existing code path.

    /// Tokenize a UTF-8 string into a vector of token IDs using the model's
    /// own GGUF-embedded tokenizer.
    ///
    /// * `add_special` — prepend BOS / append EOS if the model is configured
    ///   to do so (typically `true` at the start of a request).
    /// * `parse_special` — interpret control tokens like `<|im_start|>` rather
    ///   than treating them as plain text (typically `true` for chat).
    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, String> {
        // First call: pass a zero-sized buffer to ask libllama how many tokens
        // we need. Negative return = -required_count.
        let text_bytes = text.as_bytes();
        let text_len = text_bytes.len() as i32;
        let probe = unsafe {
            llama_tokenize(
                self.model,
                text_bytes.as_ptr() as *const c_char,
                text_len,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        if probe == 0 {
            return Ok(Vec::new());
        }
        let needed = probe.unsigned_abs() as usize;
        let mut buf = vec![0i32; needed];
        let written = unsafe {
            llama_tokenize(
                self.model,
                text_bytes.as_ptr() as *const c_char,
                text_len,
                buf.as_mut_ptr(),
                needed as i32,
                add_special,
                parse_special,
            )
        };
        if written < 0 {
            return Err(format!(
                "llama_tokenize failed: returned {written}, needed {needed}"
            ));
        }
        buf.truncate(written as usize);
        Ok(buf)
    }

    /// Detokenize a single token ID to its UTF-8 piece.
    ///
    /// * `special` — render control tokens (`<|im_start|>` etc.) verbatim if
    ///   `true`, otherwise hide them.
    pub fn token_to_piece(&self, token: i32, special: bool) -> Result<String, String> {
        // 64 bytes covers >99% of pieces; for the rare big ones we re-call.
        let mut buf = vec![0u8; 64];
        let written = unsafe {
            llama_token_to_piece(
                self.model,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                special,
            )
        };
        let n = if written < 0 {
            // Buffer too small; libllama returned -required_size.
            let needed = written.unsigned_abs() as usize;
            buf = vec![0u8; needed];
            let written2 = unsafe {
                llama_token_to_piece(
                    self.model,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    special,
                )
            };
            if written2 < 0 {
                return Err(format!(
                    "llama_token_to_piece failed twice for token {token}: {written2}"
                ));
            }
            written2 as usize
        } else {
            written as usize
        };
        buf.truncate(n);
        String::from_utf8(buf).map_err(|e| format!("token_to_piece: invalid UTF-8: {e}"))
    }

    /// Detokenize a slice of token IDs by concatenating their pieces.
    ///
    /// Note: this is a per-token loop (cheap for short outputs). For very
    /// large detokenize batches, prefer the streaming token-by-token decode.
    pub fn detokenize(&self, tokens: &[i32], special: bool) -> Result<String, String> {
        let mut out = String::with_capacity(tokens.len() * 4);
        for &t in tokens {
            out.push_str(&self.token_to_piece(t, special)?);
        }
        Ok(out)
    }

    /// Borrow the GGUF-embedded chat template (Jinja-equivalent string), or
    /// `None` if the model does not ship one. Pass `name = None` to get the
    /// default template.
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let cname: Option<CString> = name.map(|n| CString::new(n).ok()).flatten();
        let cname_ptr: *const c_char = cname
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());
        let raw = unsafe { llama_model_chat_template(self.model, cname_ptr) };
        if raw.is_null() {
            return None;
        }
        // Safety: libllama returns a pointer to a static string inside the
        // model's metadata; valid for the lifetime of the model.
        let cstr = unsafe { std::ffi::CStr::from_ptr(raw) };
        cstr.to_str().ok().map(|s| s.to_owned())
    }

    /// Apply the chat template (built-in C++ engine, NOT jinja) to a list of
    /// `(role, content)` pairs and return the formatted prompt string.
    ///
    /// `tmpl` is the template name to use, or `None` to use the model's
    /// default. `add_assistant_prefix = true` appends the assistant prefix
    /// token sequence (typically `<|im_start|>assistant\n`).
    ///
    /// Note: this engine supports a fixed list of templates (chatml, llama3,
    /// gemma, mistral, etc.). For arbitrary jinja, use a Rust-side renderer.
    pub fn apply_chat_template(
        &self,
        tmpl: Option<&str>,
        messages: &[(&str, &str)],
        add_assistant_prefix: bool,
    ) -> Result<String, String> {
        // Hold backing CStrings for the lifetime of the FFI call.
        let mut backing: Vec<(CString, CString)> = Vec::with_capacity(messages.len());
        for (role, content) in messages {
            let r = CString::new(*role)
                .map_err(|_| "chat role contains interior NUL".to_string())?;
            let c = CString::new(*content)
                .map_err(|_| "chat content contains interior NUL".to_string())?;
            backing.push((r, c));
        }
        let chat: Vec<LlamaChatMessage> = backing
            .iter()
            .map(|(r, c)| LlamaChatMessage {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();
        let tmpl_c: Option<CString> = tmpl.map(|t| CString::new(t).ok()).flatten();
        let tmpl_ptr: *const c_char = tmpl_c
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null());

        // Estimate size: 2× total message length, minimum 256 bytes.
        let estimate: usize = backing
            .iter()
            .map(|(r, c)| r.as_bytes().len() + c.as_bytes().len())
            .sum::<usize>()
            .saturating_mul(2)
            .max(256);
        let mut buf = vec![0u8; estimate];
        let written = unsafe {
            llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_assistant_prefix,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        let n = if written < 0 {
            return Err(format!(
                "llama_chat_apply_template failed: {written} (likely unsupported template)"
            ));
        } else if (written as usize) > buf.len() {
            // Buffer too small — re-call with the exact size.
            buf = vec![0u8; written as usize];
            let written2 = unsafe {
                llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_assistant_prefix,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if written2 < 0 || (written2 as usize) > buf.len() {
                return Err(format!(
                    "llama_chat_apply_template failed on retry: {written2}"
                ));
            }
            written2 as usize
        } else {
            written as usize
        };
        buf.truncate(n);
        String::from_utf8(buf).map_err(|e| format!("chat template: invalid UTF-8: {e}"))
    }
}

impl Drop for LlamaForward {
    fn drop(&mut self) {
        eprintln!("[LLAMA_BACKEND] Shutting down...");
        // Print final timings before cleanup
        self.print_timings();
        unsafe {
            llama_free(self.ctx);
            llama_free_model(self.model);
            llama_backend_free();
        }
        eprintln!("[LLAMA_BACKEND] Cleanup complete.");
    }
}

// ---------------------------------------------------------------------------
// Environment-driven constructor (reads CHIMERE_* env vars)
// ---------------------------------------------------------------------------

/// Check if the llama backend is enabled.
pub fn is_enabled() -> bool {
    std::env::var("CHIMERE_LLAMA_BACKEND").is_ok()
}

/// Create a LlamaForward from environment variables.
///
/// Reads:
/// - `CHIMERE_MODEL` — GGUF path (default: Qwen3.5 IQ3_S custom-mix)
/// - `CHIMERE_NCMOE` — number of ncmoe layers (default: 4)
/// - `CHIMERE_KV_MAX_SEQ` — context size (default: 65536)
/// - `CHIMERE_KV_TYPE_K` — KV cache key type (default: 8 = Q8_0)
/// - `CHIMERE_KV_TYPE_V` — KV cache value type (default: 2 = Q4_0)
/// - `CHIMERE_FLASH_ATTN` — enable flash attention (default: 1)
pub fn from_env() -> Result<LlamaForward, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());

    let model_path = std::env::var("CHIMERE_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home,
        )
    });

    let ncmoe: u32 = std::env::var("CHIMERE_NCMOE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let n_ctx: u32 = std::env::var("CHIMERE_KV_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);

    let type_k: i32 = std::env::var("CHIMERE_KV_TYPE_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GGML_TYPE_Q8_0);

    let type_v: i32 = std::env::var("CHIMERE_KV_TYPE_V")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GGML_TYPE_Q4_0);

    let flash_attn = std::env::var("CHIMERE_FLASH_ATTN")
        .map(|v| v != "0")
        .unwrap_or(true);

    // M1 J4-rewrite: when the NativeScheduler is armed (`CHIMERE_MULTISLOT>=2`
    // AND `CHIMERE_MULTISLOT_NATIVE=1`), the context must allocate per-slot
    // KV pages / SSM states. Read the slot count here and pass via
    // `new_multi_seq`. Legacy path (MULTISLOT_NATIVE unset) keeps n_seq_max=1
    // for bit-identical production behaviour.
    let native_active = std::env::var("CHIMERE_MULTISLOT_NATIVE")
        .ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) >= 1;
    let n_seq_max: u32 = if native_active {
        std::env::var("CHIMERE_MULTISLOT")
            .ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(1).max(1).min(8)
    } else {
        1
    };
    if n_seq_max > 1 {
        eprintln!(
            "[llama_backend] NativeScheduler active: allocating ctx with n_seq_max={}",
            n_seq_max,
        );
    }

    LlamaForward::new_multi_seq(
        &model_path,
        99, // -ngl 99 = offload all layers
        n_ctx,
        ncmoe,
        Some(type_k),
        Some(type_v),
        flash_attn,
        n_seq_max,
    )
}
