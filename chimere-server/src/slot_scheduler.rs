//! # Multi-slot scheduler (M1, Apr 2026)
//!
//! Type-safe plumbing for moving from `Mutex<AppStateModel>` (single-slot)
//! to continuous-batching multi-slot serving.
//!
//! ## J1 (delivered) — scaffolding types
//!
//! - `SlotState`, `Slot`, `SlotPool`, `BatchBuilder`, `SchedulerConfig`.
//! - 3 unit tests (pool bookkeeping, batch layout, config default).
//! - Legacy behaviour unchanged; scheduler is only active when
//!   `CHIMERE_MULTISLOT >= 2`.
//!
//! ## J2 — admission queue + worker loop (closure-based)
//!
//! - `Scheduler::new()` takes the `SchedulerConfig`, allocates an `mpsc`
//!   admission channel, and holds the receiver end until `spawn_worker` is
//!   called.
//! - `Scheduler::admission_tx()` returns a cheap-clone sender for HTTP
//!   handlers to enqueue requests.
//! - `ScheduledRequest` carries a `Box<dyn FnOnce()>` closure so the scheduler
//!   remains decoupled from `chimere_model::*` types. The HTTP handler builds
//!   the closure (captures `Arc<AppState>`, its own `tx`, the prompt, etc.)
//!   and submits. The worker thread drains admissions and, for each request,
//!   runs the closure to completion on a dedicated slot.
//! - The worker loop supports N slots via a simple round-robin execution
//!   strategy (N worker threads, one per slot). Honest FFI scope: inference
//!   is still serialised by the `AppState.model` mutex inside each closure;
//!   the **observable** interleaving comes from OS-thread scheduling across
//!   the `Mutex<AppStateModel>` re-acquisitions between `generate_with_mtp`
//!   bursts (see `chat_completions_stream` in `server.rs`).
//!
//! ## J4-rewrite — NativeScheduler (this commit)
//!
//! - `NativeScheduler` owns the `LlamaForward` directly (transferred from
//!   the main model loader at boot). A single OS thread drives
//!   `forward_multi_seq` with per-slot sequence IDs.
//! - `NativeScheduledRequest` carries raw prompt tokens + sampling params;
//!   no closure indirection. The scheduler calls `forward_multi_seq`,
//!   routes per-slot logits to per-slot samplers (J5a), applies per-slot
//!   engram biases (J5b), emits tokens through per-slot channels (J6 try_emit).
//! - Opt-in via `CHIMERE_MULTISLOT_NATIVE=1` (ignored unless
//!   `CHIMERE_MULTISLOT>=2`). Legacy J2 closure path is unchanged.
//! - Legacy `CHIMERE_MULTISLOT<2` path is bit-identical.
//!
//! ## J5+ achieved
//!
//! - Per-slot sampler via `SamplerHandle` (J5a, `Drop` frees C++ state).
//! - Per-slot engram bias via `apply_engram_bias_to_sampler` (J5b).
//! - `</think>` flip + stop tokens + try_emit on client disconnect (J6).
//!
//! ## M2+ (future)
//!
//! - Per-slot engram hot-swap (today shared via `attach_engram`).
//! - Mixed prefill+gen in the same batch for non-qwen3next arches.
//! - MTP re-enabled on a slot-exclusive basis.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tokio::sync::{mpsc, Notify};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Default slot count when `CHIMERE_MULTISLOT` is unset or set to `"1"`.
/// Kept at 1 so `from_env()` is a no-op for production.
pub const NUM_SLOTS_DEFAULT: usize = 1;

/// Upper bound on concurrent slots. Anything above this is capped and warned,
/// because per-slot KV pages grow linearly and VRAM on a 16 GB card is tight.
/// This value is *not* load-bearing — change it once we have J7 stress data.
pub const NUM_SLOTS_MAX: usize = 8;

/// Upper bound on the admission queue. Back-pressure reads this.
pub const ADMISSION_QUEUE_CAP: usize = 64;

/// Runtime configuration for the scheduler. Cheap to clone.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_slots: usize,
    pub queue_cap: usize,
    /// `false` → legacy `Mutex<AppStateModel>` path (production today).
    /// `true` → route through the admission channel (J2+).
    pub enabled: bool,
    /// J4-rewrite: `true` iff `CHIMERE_MULTISLOT_NATIVE=1` AND `enabled`.
    /// When true, HTTP handlers route through `NativeScheduler` instead of
    /// the closure-based `Scheduler`. Both schedulers can coexist in
    /// `AppState` — see `server.rs::chat_completions_handler`.
    pub native: bool,
}

impl SchedulerConfig {
    /// Read `CHIMERE_MULTISLOT` and friends. Legacy default: 1 slot, disabled.
    pub fn from_env() -> Self {
        let num_slots: usize = std::env::var("CHIMERE_MULTISLOT")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(NUM_SLOTS_DEFAULT);

        let clamped = num_slots.min(NUM_SLOTS_MAX).max(1);
        if num_slots > NUM_SLOTS_MAX {
            eprintln!(
                "[slot_scheduler] CHIMERE_MULTISLOT={} capped to {}",
                num_slots, NUM_SLOTS_MAX
            );
        }

        let queue_cap: usize = std::env::var("CHIMERE_ADMISSION_QUEUE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(ADMISSION_QUEUE_CAP);

        let enabled = clamped >= 2;
        let native = enabled
            && std::env::var("CHIMERE_MULTISLOT_NATIVE")
                .map(|v| {
                    let t = v.trim();
                    !(t.is_empty() || t == "0" || t.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(false);

        Self {
            num_slots: clamped,
            queue_cap,
            enabled,
            native,
        }
    }

    /// `true` when the scheduler should actually be used. On `false`, callers
    /// must keep using the legacy `Mutex<AppStateModel>` code path.
    pub fn is_active(&self) -> bool {
        self.enabled
    }

    /// `true` when the J4-rewrite native `forward_multi_seq` path should be
    /// used. Requires `CHIMERE_MULTISLOT>=2 && CHIMERE_MULTISLOT_NATIVE=1`.
    pub fn is_native(&self) -> bool {
        self.native
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_slots: NUM_SLOTS_DEFAULT,
            queue_cap: ADMISSION_QUEUE_CAP,
            enabled: false,
            native: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Slot
// ---------------------------------------------------------------------------

/// State machine for a single serving slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Not assigned to any request. Admission will look here first.
    Free,
    /// Prefilling the prompt in one or more chunks. `chunks_done` counts how
    /// many 512-token (or ubatch-sized) pieces have been pushed into the batch.
    Prefilling { chunks_done: usize },
    /// Generating one token per `step()`. The hot path.
    Generating,
    /// Emitting the final `Done` marker then freeing. One step transition.
    Draining,
}

/// Minimal per-request sampling knobs. The real sampler config lives in
/// `chimere_sampler.cpp`; this struct only carries request-scoped overrides
/// (temperature, top-p, max_tokens, stop tokens, and thinking toggle).
///
/// We intentionally keep this separate from `SamplerHandle` so that the
/// FFI handle can be lazily allocated (or, on J5, pre-allocated per slot).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub max_tokens: u32,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Mirror server.rs defaults. DO NOT drift silently — if these change
        // in server.rs we must update here too, or slots will sample
        // differently from the legacy path.
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.05,
            presence_penalty: 0.0,
            max_tokens: 512,
            stop_tokens: Vec::new(),
            enable_thinking: true,
        }
    }
}

/// Per-slot streaming channel message. The HTTP handler on the receiving end
/// converts these to SSE `data:` frames (J2 for handler integration).
#[derive(Debug, Clone)]
pub enum StreamMsg {
    /// Regular generated token. The scheduler decoded it already.
    Token { text: String, logprob: Option<f32> },
    /// Token inside a `<think>` block — kept separate so the handler can emit
    /// `reasoning_content` rather than `content`, just like the legacy path.
    Thinking { text: String },
    /// Tool-call fragment (Qwen3.5 `<tool_call>` syntax). Emitted once fully
    /// parsed — NOT streamed character-by-character.
    ToolCall { json: String },
    /// Terminal marker. After this the slot can be freed.
    Done { finish_reason: String },
    /// Client disconnected or internal error. Slot is freed anyway.
    Error { message: String },
}

/// Per-slot stats, reset on `free()`. Useful for `/v1/status` later.
#[derive(Debug, Default, Clone)]
pub struct SlotStats {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_ms: u32,
    pub gen_ms: u32,
}

/// A single serving slot. Owns its `seq_id` across the libllama context.
///
/// Everything raw-pointer-ish (sampler handle, engram Arc) lives here so
/// the scheduler can work with `&mut Slot` and not juggle C state on the side.
///
/// NOTE: `SamplerHandle` and `EngramHandle` are placeholders in J1. J5 wires
/// them to the C++ sampler and the real engram lookup. We use `Option<...>`
/// so the struct is usable with `None` today — the FFI call sites in J3/J5
/// know to skip when the option is empty.
pub struct Slot {
    pub id: u32,                          // libllama seq_id (0..num_slots)
    pub state: SlotState,
    /// Position in the (conceptual) per-seq KV cache.
    pub pos: i32,
    pub prompt_tokens: Vec<u32>,
    pub generated: Vec<u32>,
    pub params: SamplingParams,
    /// Optional per-slot sampler. Allocated at J5. Holds a raw C pointer.
    pub sampler: Option<SamplerHandle>,
    /// Optional per-request engram table ref. J5.
    pub engram: Option<EngramHandle>,
    pub engram_alpha: f32,
    /// Sliding context window kept for engram lookups + DRY penalty.
    /// Bounded to avoid unbounded growth on long generations.
    pub recent_context: Vec<u32>,
    /// One-way channel back to the HTTP handler that admitted this request.
    pub tx: Option<mpsc::Sender<StreamMsg>>,
    pub want_logprobs: bool,
    pub request_id: String,
    pub stats: SlotStats,
    /// Set by the HTTP handler if the client drops the connection.
    /// `step()` checks this at the top of each iteration.
    pub cancelled: Arc<AtomicBool>,
    /// J6 — reason captured when `mark_draining()` is called. Consumed by
    /// the dispatcher when it sends the terminal `StreamMsg::Done`.
    pub finish_reason: Option<String>,
    /// J6 — `true` while the current token stream is inside a
    /// `<think>...</think>` block. Flipped off in `on_token_sampled` when
    /// the just-sampled token equals the `</think>` closer. Mirrors the
    /// `thinking` bool in `mtp_scheduler::generate_with_mtp_streaming`
    /// so SSE frames can choose between `reasoning_content` (Thinking)
    /// and `content` (Token).
    pub thinking: bool,
}

impl Slot {
    /// Empty, unassigned slot. `id` is set once (on pool build) and never
    /// changes so the libllama seq_id can be remembered in kv_cache pages.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            state: SlotState::Free,
            pos: 0,
            prompt_tokens: Vec::new(),
            generated: Vec::new(),
            params: SamplingParams::default(),
            sampler: None,
            engram: None,
            engram_alpha: 0.0,
            recent_context: Vec::with_capacity(256),
            tx: None,
            want_logprobs: false,
            request_id: String::new(),
            stats: SlotStats::default(),
            cancelled: Arc::new(AtomicBool::new(false)),
            finish_reason: None,
            thinking: false,
        }
    }

    /// `true` when the slot is not currently holding any work. Used by
    /// `SlotPool::alloc_free()`.
    pub fn is_free(&self) -> bool {
        matches!(self.state, SlotState::Free)
    }

    /// Reset state to `Free`. Does *not* release the libllama KV pages —
    /// that's the caller's job (J5 will wire `llama_kv_cache_seq_rm`).
    ///
    /// The per-slot sampler's DRY / repetition history is reset here so the
    /// slot is ready for a fresh conversation. Engram biases are cleared
    /// separately (they accumulate per-token anyway).
    pub fn mark_free(&mut self) {
        self.state = SlotState::Free;
        self.pos = 0;
        self.prompt_tokens.clear();
        self.generated.clear();
        self.recent_context.clear();
        self.tx = None;
        self.request_id.clear();
        self.stats = SlotStats::default();
        self.cancelled.store(false, Ordering::SeqCst);
        self.finish_reason = None;
        self.thinking = false;
        // J5a: reset per-slot sampler state between conversations.
        if let Some(s) = self.sampler.as_ref() {
            s.reset();
            s.clear_engram_bias();
        }
    }

    // ------------------------------------------------------------------
    // J6 — rollout safety: stop tokens, </think> toggle
    // ------------------------------------------------------------------

    /// Canonical Qwen3.5 closing thinking token. Mirrors the `THINK_END`
    /// constant in `mtp_scheduler.rs` so the two code paths agree on the
    /// exact id. Hard-coded on purpose — if the tokenizer changes this we
    /// want the test suite to break loudly rather than silently infer.
    pub const THINK_END_TOKEN: u32 = 248069;
    /// Canonical Qwen3.5 opening thinking token (tracked for symmetry;
    /// the opener is in the prompt template, not normally re-sampled).
    pub const THINK_START_TOKEN: u32 = 248068;

    /// Transition into the terminal `Draining` state. The dispatcher is
    /// expected to observe this on the next scheduler tick, emit the final
    /// `StreamMsg::Done { finish_reason }` frame, then call `mark_free`.
    ///
    /// Idempotent: calling `mark_draining` on an already-draining slot
    /// only updates `finish_reason` if one was not set yet.
    ///
    /// `reason` conventions (kept aligned with OpenAI's chat API):
    /// - `"stop"`   → stop token hit, natural end.
    /// - `"length"` → `max_tokens` / budget exhausted.
    /// - `"cancel"` → client disconnected or aborted.
    /// - `"error"` → internal failure; caller should also send an Error frame.
    pub fn mark_draining(&mut self, reason: &str) {
        self.state = SlotState::Draining;
        if self.finish_reason.is_none() {
            self.finish_reason = Some(reason.to_string());
        }
    }

    /// `true` once `mark_draining` has fired. Dispatcher uses this to avoid
    /// sampling further tokens on dead slots (saves one llama_decode step
    /// between the disconnect and the slot being reclaimed).
    pub fn is_draining(&self) -> bool {
        matches!(self.state, SlotState::Draining)
    }

    /// Called right after a token has been sampled by the scheduler's
    /// per-step logic. Returns `true` if generation should continue on
    /// this slot, `false` if the slot must stop (stop-token hit). The
    /// slot's `thinking` flag is updated **before** we evaluate stop
    /// tokens so that the `</think>` closer (which may also be present
    /// in a client-supplied stop list for non-thinking clients) still
    /// exits reasoning cleanly on the first occurrence — the flag flip
    /// happens either way.
    ///
    /// When this returns `false`, the caller MUST:
    ///   1. call `mark_draining("stop")` on this slot, and
    ///   2. emit the sampled token AS-IS to the stream (so the client
    ///      sees the `</think>` closer on screen rather than a truncated
    ///      reasoning block), then
    ///   3. emit the terminal `StreamMsg::Done` frame.
    pub fn on_token_sampled(&mut self, tok: u32) -> bool {
        // 1) Toggle thinking on encountering `</think>`. This MUST happen
        //    before stop-token evaluation, because a client may well add
        //    `</think>` to `stop` even when thinking is active (seen in
        //    OpenClaw ODO routing). Semantics we pick: the closer token
        //    itself always flips thinking off; *then* stop-token logic
        //    decides whether to also terminate the stream.
        if self.thinking && tok == Self::THINK_END_TOKEN {
            self.thinking = false;
        }
        // 2) Per-request stop tokens defined on SamplingParams.
        if self.params.stop_tokens.contains(&tok) {
            return false;
        }
        true
    }

    /// Try to emit a `StreamMsg` to the HTTP handler. Returns `true` on
    /// success, `false` if the channel is closed (client disconnect) or
    /// the slot has no channel attached.
    ///
    /// When this returns `false`, the dispatcher MUST:
    ///   1. call `mark_draining("cancel")` immediately,
    ///   2. skip the rest of the current decode step for this slot, and
    ///   3. let the next scheduler tick reclaim the slot via `mark_free`.
    ///
    /// A dropped receiver is the only terminal failure mode of
    /// `mpsc::Sender::try_send` once the channel has been established.
    /// `try_send` is used on the sampler-driver path (blocking OS thread,
    /// not a tokio task) so we don't have to `.await` inside the hot
    /// decode loop. Back-pressure for slow-but-alive clients is handled
    /// by treating `Full` as a transient success: we keep the slot alive
    /// and let the next step try again — the channel buffer drains as
    /// soon as the receiver side polls.
    ///
    /// Rationale for `Full` being non-fatal: the buffered size is 64 and
    /// an SSE client that lags by more than 64 frames is almost certainly
    /// about to disconnect. At that point the OS will close the TCP
    /// connection and the receiver will be dropped, turning the next
    /// emit into `Closed` — at which point we do kill the slot. This
    /// keeps a brief network stutter from terminating a healthy stream.
    pub fn try_emit(&mut self, msg: StreamMsg) -> bool {
        let tx = match self.tx.as_ref() {
            Some(tx) => tx,
            None => return false,
        };
        match tx.try_send(msg) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => true,
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }

    // ------------------------------------------------------------------
    // J5b — per-slot engram bias application
    // ------------------------------------------------------------------

    /// Append `token` to the bounded `recent_context` window used for
    /// engram lookups and DRY penalty. The window is capped at 256 tokens
    /// to keep lookup cost O(order=5..8) regardless of generation length.
    pub fn push_context(&mut self, token: u32) {
        const MAX_CONTEXT: usize = 256;
        self.recent_context.push(token);
        if self.recent_context.len() > MAX_CONTEXT {
            let drop = self.recent_context.len() - MAX_CONTEXT;
            self.recent_context.drain(0..drop);
        }
    }

    /// Look up the slot's engram table using its `recent_context` and push
    /// the resulting biases into the slot's sampler. No-op if either the
    /// engram or the sampler is unset. Biases already installed by other
    /// code paths (e.g. `</think>` suppression at `-inf`) are preserved;
    /// see `chimere_sampler_set_engram_bias` in `ffi/chimere_sampler.cpp`
    /// for the exact merge semantics.
    ///
    /// Biases are computed as `alpha * ln(prob + 1e-10)` to match the
    /// existing production pipeline (see `mtp_scheduler.rs`).
    ///
    /// This is the method the multi-slot scheduler will call once per
    /// generate step, right before invoking `LlamaForward::sample_slot`.
    pub fn apply_engram_bias_to_sampler(&self) {
        let _g = crate::prof!("engram.lookup_and_bias");
        let (engram, sampler) = match (&self.engram, &self.sampler) {
            (Some(e), Some(s)) if s.is_active() => (e, s),
            _ => return,
        };
        // `engram_alpha` overrides the handle's default when set non-zero
        // by the admission path (per-request). Otherwise fall back to the
        // per-handle alpha that was attached via `SlotPool::attach_engram`.
        let alpha = if self.engram_alpha > 0.0 {
            self.engram_alpha
        } else {
            engram.alpha
        };
        if alpha <= 0.0 {
            return;
        }
        let preds = engram.lookup.lookup(&self.recent_context);
        if preds.is_empty() {
            // No n-gram hit for this context window. Clear any stale Engram
            // bias from the previous token so we don't keep dragging the
            // sampler toward outdated predictions.
            sampler.clear_engram_bias();
            return;
        }

        // Convert (token, prob) → (token, alpha * ln(prob + eps)). This
        // matches mtp_scheduler.rs exactly so the two paths stay
        // numerically equivalent.
        let token_ids: Vec<i32> = preds.iter().map(|&(t, _)| t as i32).collect();
        let biases: Vec<f32> = preds
            .iter()
            .map(|&(_, p)| alpha * (p + 1e-10).ln())
            .collect();
        unsafe {
            crate::llama_backend::chimere_sampler_set_engram_bias_handle(
                sampler.as_raw(),
                token_ids.as_ptr(),
                biases.as_ptr(),
                preds.len() as std::ffi::c_int,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// FFI handles — J5a wires the sampler, J5b wires the engram
// ---------------------------------------------------------------------------

/// Owning handle to a C++ `chimere_sampler` allocated by
/// [`llama_backend::chimere_sampler_alloc`]. Each active slot owns one so
/// that logit biases, DRY history, and repetition counters are **per-slot**.
///
/// The pointer is freed on drop via `chimere_sampler_free_handle`, so the
/// handle must NEVER be `Clone`. It is `Send` because the FFI side has no
/// thread-local state of its own — the scheduler worker thread uses the
/// handle concurrently with the admitting HTTP thread only through the
/// mpsc channel (ownership transfer, not shared mutation).
///
/// NOTE on safety: the handle is only usable while the `LlamaForward` that
/// allocated it is alive. In practice the slot pool is owned by the same
/// `AppState` that owns the model, so the two lifetimes are coupled.
pub struct SamplerHandle {
    raw: *mut std::ffi::c_void,
}

impl SamplerHandle {
    /// Wrap a raw pointer returned by `chimere_sampler_alloc`. `null` is
    /// accepted and treated as "no-op sampler"; all operations below then
    /// degrade silently so callers do not need to special-case it.
    ///
    /// # Safety
    /// `raw` must either be null or a valid pointer returned by
    /// `chimere_sampler_alloc`. The returned handle takes ownership and
    /// will free the pointer on drop.
    pub unsafe fn from_raw(raw: *mut std::ffi::c_void) -> Self {
        Self { raw }
    }

    /// `true` when a real C++ sampler is attached.
    pub fn is_active(&self) -> bool {
        !self.raw.is_null()
    }

    /// Raw pointer for FFI calls (e.g. `LlamaForward::sample_slot`).
    ///
    /// # Safety
    /// Caller must not free the returned pointer — ownership remains with
    /// this `SamplerHandle`.
    pub unsafe fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw
    }

    /// Reset DRY / repetition history on this slot's sampler (e.g. between
    /// conversations served on the same slot).
    pub fn reset(&self) {
        if self.raw.is_null() {
            return;
        }
        unsafe { crate::llama_backend::chimere_sampler_reset_handle(self.raw); }
    }

    /// Clear only Engram biases (preserve manual `-inf` biases like
    /// `</think>` suppression).
    pub fn clear_engram_bias(&self) {
        if self.raw.is_null() {
            return;
        }
        unsafe { crate::llama_backend::chimere_sampler_clear_engram_bias_handle(self.raw); }
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { crate::llama_backend::chimere_sampler_free_handle(self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

// Safety: chimere_sampler has no thread-local state of its own; the only
// constraint is that the underlying `llama_context` is single-threaded,
// which is enforced by the scheduler's one-OS-thread-drives-decode design.
unsafe impl Send for SamplerHandle {}

impl std::fmt::Debug for SamplerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerHandle")
            .field("active", &self.is_active())
            .finish()
    }
}

/// Per-slot reference to the globally loaded `MultiEngramLookup` tree plus
/// an engram blending strength `alpha`. Concretely, we wrap an `Arc` around
/// the lookup so all slots share the mmap'd `.engr` tables (no duplication)
/// but each slot can run with its own alpha.
///
/// The engram tables themselves are read-only after boot; only the biases
/// applied to the sampler are per-slot — that isolation lives in
/// `Slot::apply_engram_bias_to_sampler()` (J5b).
#[derive(Clone)]
pub struct EngramHandle {
    pub lookup: Arc<crate::engram_lookup::MultiEngramLookup>,
    pub alpha: f32,
}

impl EngramHandle {
    pub fn new(lookup: Arc<crate::engram_lookup::MultiEngramLookup>, alpha: f32) -> Self {
        Self { lookup, alpha }
    }
}

impl std::fmt::Debug for EngramHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramHandle")
            .field("n_tables", &self.lookup.len())
            .field("order", &self.lookup.order())
            .field("alpha", &self.alpha)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SlotPool
// ---------------------------------------------------------------------------

/// Fixed-size pool of slots. Index == seq_id.
pub struct SlotPool {
    slots: Vec<Slot>,
    /// Signalled whenever a slot goes Free, so a parked request can retry.
    pub free_notify: Arc<Notify>,
}

impl SlotPool {
    pub fn new(num_slots: usize) -> Self {
        let slots: Vec<Slot> = (0..num_slots as u32).map(Slot::new).collect();
        Self {
            slots,
            free_notify: Arc::new(Notify::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Find the first Free slot, or None if all busy.
    pub fn alloc_free(&mut self) -> Option<&mut Slot> {
        self.slots.iter_mut().find(|s| s.is_free())
    }

    /// Active (non-Free) slots for the build_decode_batch loop (J3).
    pub fn active_mut(&mut self) -> impl Iterator<Item = &mut Slot> {
        self.slots.iter_mut().filter(|s| !s.is_free())
    }

    /// By seq_id == index.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Slot> {
        self.slots.get_mut(id as usize)
    }

    pub fn num_active(&self) -> usize {
        self.slots.iter().filter(|s| !s.is_free()).count()
    }

    /// J4-rewrite — iterate all slots (free and active). Used by the
    /// native dispatcher to reap `Draining` slots before the next tick.
    pub fn all_mut(&mut self) -> impl Iterator<Item = &mut Slot> {
        self.slots.iter_mut()
    }

    /// J5a — allocate one independent C++ sampler per slot.
    ///
    /// Each slot gets its own `chimere_sampler` with the production
    /// Qwen3.5 thinking-mode defaults (same values as the single-slot path
    /// in `LlamaForward::new`). The sampler handle takes ownership and
    /// frees itself on slot drop, so pool lifetime drives sampler
    /// lifetime — no explicit cleanup needed outside of `mark_free()`
    /// which only resets state, not memory.
    ///
    /// # Parameters
    ///
    /// - `model_ptr` is the `*const LlamaModel` obtained via
    ///   `LlamaForward::model_raw()`. Must be valid for the lifetime of
    ///   this pool.
    /// - `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` are
    ///   the initial defaults; per-request overrides still happen via
    ///   `Slot.params` at admission time (J6 backlog — today only the
    ///   defaults are used).
    ///
    /// Returns `Ok(n_allocated)` on success, `Err(msg)` if any slot's
    /// allocation fails (in which case *all* previously allocated handles
    /// are dropped to keep the pool in a clean "no sampler" state).
    ///
    /// # Safety
    /// `model_ptr` must be a valid pointer from `LlamaForward::model_raw()`.
    /// The caller is responsible for ensuring the `LlamaForward` outlives
    /// the `SlotPool`.
    pub unsafe fn alloc_samplers(
        &mut self,
        model_ptr: *const crate::llama_backend::LlamaModel,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        presence_penalty: f32,
    ) -> Result<usize, String> {
        // Defer to the DRY-configurable variant with production defaults
        // (dry_multiplier=0.8, matching LlamaForward::new's built-in sampler).
        self.alloc_samplers_with_dry(
            model_ptr,
            temperature, top_p, top_k, min_p, presence_penalty,
            0.8,   // dry_multiplier
            1.75,  // dry_base
            2,     // dry_min_length
            -1,    // dry_penalty_last_n
        )
    }

    /// Like [`SlotPool::alloc_samplers`] but with DRY parameters exposed.
    /// Set `dry_multiplier = 0.0` to allocate a sampler with DRY
    /// effectively disabled — required for smoke tests on models whose
    /// vocab cannot tokenise the default DRY sequence-breakers.
    ///
    /// # Safety
    /// `model_ptr` must be a valid pointer from `LlamaForward::model_raw()`.
    pub unsafe fn alloc_samplers_with_dry(
        &mut self,
        model_ptr: *const crate::llama_backend::LlamaModel,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        presence_penalty: f32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_min_length: i32,
        dry_penalty_last_n: i32,
    ) -> Result<usize, String> {
        if model_ptr.is_null() {
            return Err("alloc_samplers: model_ptr is null".into());
        }
        let mut allocated = 0usize;
        let mut failed_slot_id: Option<u32> = None;
        for slot in self.slots.iter_mut() {
            let raw = crate::llama_backend::chimere_sampler_alloc_with_dry(
                model_ptr,
                temperature, top_p, top_k,
                min_p, presence_penalty,
                dry_multiplier, dry_base, dry_min_length, dry_penalty_last_n,
            );
            if raw.is_null() {
                failed_slot_id = Some(slot.id);
                break;
            }
            slot.sampler = Some(SamplerHandle::from_raw(raw));
            allocated += 1;
        }
        if let Some(id) = failed_slot_id {
            // Drop all previously allocated handles to avoid leaking C++ state.
            for s in self.slots.iter_mut() {
                s.sampler = None;
            }
            return Err(format!(
                "chimere_sampler_alloc returned null for slot {} (allocated {} before failure)",
                id, allocated,
            ));
        }
        Ok(allocated)
    }

    /// J5b — attach a shared engram lookup to every slot with the given
    /// blending alpha. Slots start with `None` engram; this is the opt-in
    /// that lights up per-slot engram biasing in `Slot::apply_engram_bias_to_sampler`.
    ///
    /// The same `Arc` is cloned into each slot — tables are memory-mapped
    /// and read-only, so sharing is zero-copy. Per-slot `engram_alpha`
    /// (set by admission) overrides the handle's default.
    pub fn attach_engram(
        &mut self,
        lookup: Arc<crate::engram_lookup::MultiEngramLookup>,
        alpha: f32,
    ) {
        for slot in self.slots.iter_mut() {
            slot.engram = Some(EngramHandle::new(Arc::clone(&lookup), alpha));
        }
    }

    /// J5b — attach a distinct engram lookup per slot. Useful for tests
    /// and for future multi-tenant setups where each slot represents a
    /// different domain (kine / cyber / research). `lookups.len()` must
    /// equal `self.len()` or the function returns Err.
    pub fn attach_engram_per_slot(
        &mut self,
        lookups: Vec<Arc<crate::engram_lookup::MultiEngramLookup>>,
        alpha: f32,
    ) -> Result<(), String> {
        if lookups.len() != self.slots.len() {
            return Err(format!(
                "attach_engram_per_slot: got {} lookups for {} slots",
                lookups.len(),
                self.slots.len(),
            ));
        }
        for (slot, lookup) in self.slots.iter_mut().zip(lookups.into_iter()) {
            slot.engram = Some(EngramHandle::new(lookup, alpha));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BatchBuilder — pure Rust, zero FFI
// ---------------------------------------------------------------------------

/// Accumulator for one `llama_decode` call's batch. The layout mirrors
/// `LlamaBatch` (tokens / pos / n_seq_id / seq_id**) but stays pure-Rust
/// until `as_llama_batch()` (J3) materialises the C pointers.
#[derive(Debug, Default)]
pub struct BatchBuilder {
    pub toks: Vec<i32>,
    pub pos: Vec<i32>,
    pub n_seq_id: Vec<i32>,
    /// Per-token seq_id lists. For a single-seq token, `seq_ids[i] == vec![seq_id]`.
    pub seq_ids: Vec<Vec<i32>>,
    /// Per-token "compute logits" flag (1 = final token of prefill or a
    /// generate step, 0 otherwise).
    pub logits: Vec<i8>,
    /// Per-slot index into `toks`, for sample-after-decode. One entry per
    /// slot that emitted a token on this step.
    pub slot_emit_indices: Vec<(u32, usize)>,
}

impl BatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.toks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Push one token belonging to `seq_id` at position `pos`. Sets the
    /// `logits` flag according to `want_logits`.
    pub fn push(&mut self, tok: i32, pos: i32, seq_id: i32, want_logits: bool) {
        self.toks.push(tok);
        self.pos.push(pos);
        self.n_seq_id.push(1);
        self.seq_ids.push(vec![seq_id]);
        self.logits.push(if want_logits { 1 } else { 0 });
    }

    /// Record that slot `slot_id` will consume the sample from batch index
    /// `batch_idx` (set to the `i` at which `push` was called with logits=1).
    pub fn mark_slot_emit(&mut self, slot_id: u32, batch_idx: usize) {
        self.slot_emit_indices.push((slot_id, batch_idx));
    }
}

// ---------------------------------------------------------------------------
// ScheduledRequest — HTTP handler → scheduler (J2 shape)
// ---------------------------------------------------------------------------

/// Work item travelling over the admission channel. The HTTP handler packages
/// the full inference call as a closure and drops it on the queue — the
/// scheduler does not need to know about `Qwen35Model`, `SamplingParams`, or
/// any FFI surface.
///
/// This keeps the scheduler module decoupled from the model types (and keeps
/// the existing `chat_completions_stream` logic reusable inside the closure).
///
/// Closure contract:
/// - It is `FnOnce + Send + 'static`, runs to completion synchronously.
/// - It is responsible for sending its own `Done` marker on the per-request
///   channel it captured; the scheduler does not send anything.
/// - It must honour the `cancelled` flag passed in `metadata` (polled before
///   each `generate_with_mtp_streaming` callback).
pub struct ScheduledRequest {
    pub metadata: ScheduledRequestMeta,
    /// The actual inference function. Owns everything it needs via captures.
    pub run: Box<dyn FnOnce(ScheduledRequestMeta) + Send + 'static>,
}

/// Small, cheap-clone envelope for scheduler telemetry — kept separate from
/// the closure so the worker can log / gate on it before invoking `run`.
#[derive(Debug, Clone)]
pub struct ScheduledRequestMeta {
    pub request_id: String,
    pub prompt_token_count: usize,
    pub max_tokens: u32,
    pub cancelled: Arc<AtomicBool>,
    /// Enqueue timestamp (from `Instant::now()` at submission). Used for
    /// queue-wait metrics.
    pub enqueued_at: std::time::Instant,
}

// ---------------------------------------------------------------------------
// Scheduler (J2 closure-based)
// ---------------------------------------------------------------------------

/// Scheduler handle. Owns the admission channel and (once `spawn_workers` is
/// called) the pool of worker OS threads.
///
/// The admission side (`admission_tx()`) is `Clone`-ed into every HTTP
/// handler via `AppState`. The worker side (`admission_rx`) is consumed
/// exactly once by `spawn_workers`.
pub struct Scheduler {
    pub config: SchedulerConfig,
    /// Sender half of the admission queue. Cheap to clone.
    admission_tx: mpsc::Sender<ScheduledRequest>,
    /// Consumed by `spawn_workers`. `None` after the workers are running.
    admission_rx: Option<mpsc::Receiver<ScheduledRequest>>,
    /// Slot pool state. Used for telemetry at J2; J3+ will consult it from
    /// the worker loop to drive multi-seq batches.
    pub pool: std::sync::Mutex<SlotPool>,
    /// Flipped by `shutdown()`. Workers check before blocking on `recv()`.
    pub shutdown: Arc<AtomicBool>,
}

impl Scheduler {
    /// Prometheus/observability accessors (polish 2026-04-24).
    pub fn slot_pool_size_or_default(&self) -> usize {
        self.config.num_slots
    }
    pub fn slot_active_count_or_default(&self) -> usize {
        self.slot_active_count()
    }
    pub fn queue_depth_or_default(&self) -> usize { 0 }

    /// Build a scheduler. The admission channel is created here; the worker
    /// threads are started by `spawn_workers`.
    pub fn new(config: SchedulerConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_cap);
        let pool = SlotPool::new(config.num_slots);
        Self {
            config,
            admission_tx: tx,
            admission_rx: Some(rx),
            pool: std::sync::Mutex::new(pool),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Clone the admission sender. HTTP handlers hold this to enqueue work.
    pub fn admission_tx(&self) -> mpsc::Sender<ScheduledRequest> {
        self.admission_tx.clone()
    }

    /// Signal workers to exit on the next recv.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Is the scheduler armed (N>=2 slots, multi-slot enabled)? Callers use
    /// this to decide whether to route through the admission channel or fall
    /// back to the legacy `Mutex<AppStateModel>` path.
    pub fn is_active(&self) -> bool {
        self.config.is_active()
    }

    /// Spawn the J2 worker. Returns the JoinHandle(s) — kept alive by
    /// `AppState` for the lifetime of the process.
    ///
    /// J2 worker topology:
    /// - **One** dedicated OS thread drains the admission queue.
    /// - For each `ScheduledRequest`, the worker calls `req.run(meta)` on
    ///   the dispatcher thread. The closure itself spawns a short-lived
    ///   compute thread if needed (that's how `chat_completions_stream`
    ///   works today), so the dispatcher returns quickly and can accept
    ///   the next request.
    /// - Back-pressure: the admission channel capacity is `config.queue_cap`
    ///   (default 64). Once full, `send().await` from the HTTP handler
    ///   suspends until the worker pulls one off.
    ///
    /// J3+ will replace this with a **multi-seq `llama_decode` driver** —
    /// one OS thread, N logical slots, round-robin batch assembly, single
    /// FFI call per step producing one token per active slot.
    pub fn spawn_workers(&mut self) -> Vec<std::thread::JoinHandle<()>> {
        let mut rx: mpsc::Receiver<ScheduledRequest> = match self.admission_rx.take() {
            Some(rx) => rx,
            None => {
                eprintln!("[slot_scheduler] spawn_workers called twice; ignoring.");
                return Vec::new();
            }
        };
        let shutdown = Arc::clone(&self.shutdown);
        let num_slots = self.config.num_slots;

        let handle = std::thread::Builder::new()
            .name("chimere-sched-dispatch".into())
            .spawn(move || {
                eprintln!(
                    "[slot_scheduler] dispatcher running, num_slots={}, queue_cap tracked by mpsc",
                    num_slots
                );
                let mut request_counter: u64 = 0;
                // Blocking loop: block on recv; dispatch; repeat.
                while !shutdown.load(Ordering::SeqCst) {
                    let req = match rx.blocking_recv() {
                        Some(r) => r,
                        None => {
                            eprintln!(
                                "[slot_scheduler] admission channel closed (all senders dropped); exiting dispatcher."
                            );
                            return;
                        }
                    };
                    request_counter = request_counter.wrapping_add(1);
                    let meta = req.metadata.clone();
                    let wait_ms = meta.enqueued_at.elapsed().as_millis();
                    eprintln!(
                        "[slot_scheduler] dispatch req={} (global #{}), prompt_toks={}, max_toks={}, queue_wait_ms={}",
                        meta.request_id,
                        request_counter,
                        meta.prompt_token_count,
                        meta.max_tokens,
                        wait_ms,
                    );
                    // Run the closure. It is responsible for spawning its own
                    // compute thread (as `chat_completions_stream` does) so
                    // the dispatcher is not blocked by long generations.
                    (req.run)(meta);
                }
                eprintln!("[slot_scheduler] dispatcher exit (shutdown flag set).");
            })
            .expect("failed to spawn chimere-sched-dispatch");

        vec![handle]
    }

    /// Per-slot quick stat summary. Used by `/v1/status` in the future.
    pub fn slot_active_count(&self) -> usize {
        self.pool
            .lock()
            .map(|p| p.num_active())
            .unwrap_or(0)
    }
}

// ===========================================================================
// J4-rewrite — NativeScheduler
//
// A second scheduler implementation that OWNS the LlamaForward directly
// and drives `forward_multi_seq` from a dedicated OS thread. Opt-in via
// CHIMERE_MULTISLOT_NATIVE=1 ; legacy Scheduler above is untouched.
// ===========================================================================

/// Native multi-slot work item. Carries raw prompt tokens + sampling params;
/// no closure indirection. The HTTP handler builds one of these, sends it
/// through the admission queue, and reads from its own per-request mpsc rx
/// to stream SSE tokens.
///
/// ## Ownership
///
/// The scheduler dispatcher takes ownership of the request (`mpsc::Receiver`
/// yields owned values). From that moment the HTTP handler's only reference
/// to the request is the `rx` end of the per-request channel. Cancellation
/// is observed by the dispatcher via `tx.try_send` returning Closed when
/// the HTTP handler drops its axum SSE stream (which drops `rx`).
pub struct NativeScheduledRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub engram_alpha: f32,
    /// Reserved for M2+ multi-tenant domain routing. Today ignored; the
    /// scheduler uses the globally attached engram lookup for all slots.
    pub engram_hint: Option<String>,
    pub tx: mpsc::Sender<StreamMsg>,
    pub want_logprobs: bool,
    pub top_logprobs_n: usize,
    pub enable_thinking: bool,
    pub cancelled: Arc<AtomicBool>,
    pub enqueued_at: std::time::Instant,
}

/// Native scheduler. Owns the `LlamaForward` that drives `forward_multi_seq`.
///
/// The main loader (`bin/chimere-server.rs`) extracts `LlamaForward` from
/// the original `Qwen35Model` / `GenericModel` and hands it to
/// `NativeScheduler::new(...)`. The scheduler then owns the FFI context for
/// the process lifetime. Legacy non-native HTTP requests (`stream=false`)
/// stay in `AppState.model`, but that path can no longer serve Qwen35 when
/// native mode is active — see `APPLY.md` red flag for details.
pub struct NativeScheduler {
    pub config: SchedulerConfig,
    /// Sender half — HTTP handlers clone this into `admission_tx()`.
    admission_tx: mpsc::Sender<NativeScheduledRequest>,
    /// Consumed exactly once by `spawn_native_driver`.
    admission_rx: Option<mpsc::Receiver<NativeScheduledRequest>>,
    /// Global engram lookup, attached to every slot at driver startup.
    /// `None` if `MultiEngramLookup::from_env()` returned None at boot.
    engram_global: Option<Arc<crate::engram_lookup::MultiEngramLookup>>,
    /// Default engram alpha used when the HTTP body does not override.
    /// 0.0 means "engram disabled by default" even if tables are loaded.
    pub default_engram_alpha: f32,
    /// Flipped by `shutdown()`. Driver checks before each tick.
    pub shutdown: Arc<AtomicBool>,
    /// Shared view of the driver's current active-slot count. Published by
    /// the driver at the top of each tick (and on admit/reap), read by the
    /// `/metrics` scrape and `/v1/status` handlers. `Relaxed` is sufficient
    /// — observability gauge, eventual consistency is fine.
    active_count: Arc<AtomicUsize>,
}

impl NativeScheduler {
    /// Prometheus/observability accessors (polish 2026-04-24).
    pub fn slot_pool_size_or_default(&self) -> usize {
        self.config.num_slots
    }
    /// Snapshot of the driver's current occupancy. Published on every tick
    /// (see `NativeDriver::run`). Reads the shared `AtomicUsize`; never
    /// blocks. Returns 0 before the driver has run its first tick.
    pub fn slot_active_count_or_default(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }
    pub fn queue_depth_or_default(&self) -> usize { 0 }

    /// Build a native scheduler. The `LlamaForward` is NOT stored here —
    /// it is passed to `spawn_native_driver` which moves it into the
    /// dedicated driver thread.
    ///
    /// # Parameters
    /// - `config` — must have `config.native == true`, else returns Err.
    /// - `engram_global` — optional `MultiEngramLookup` to attach to every
    ///   slot at driver startup. Pass `None` to disable engram biasing.
    /// - `default_engram_alpha` — alpha used when `NativeScheduledRequest`
    ///   does not override (typically 0.8 matching mtp_scheduler defaults).
    pub fn new(
        config: SchedulerConfig,
        engram_global: Option<Arc<crate::engram_lookup::MultiEngramLookup>>,
        default_engram_alpha: f32,
    ) -> Result<Self, String> {
        if !config.is_native() {
            return Err(format!(
                "NativeScheduler::new called with non-native config: \
                 enabled={}, native={}, num_slots={}. \
                 Set CHIMERE_MULTISLOT>=2 AND CHIMERE_MULTISLOT_NATIVE=1.",
                config.enabled, config.native, config.num_slots,
            ));
        }
        let (tx, rx) = mpsc::channel(config.queue_cap);
        Ok(Self {
            config,
            admission_tx: tx,
            admission_rx: Some(rx),
            engram_global,
            default_engram_alpha,
            shutdown: Arc::new(AtomicBool::new(false)),
            active_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Clone the admission sender. HTTP handlers use this to enqueue
    /// `NativeScheduledRequest`.
    pub fn admission_tx(&self) -> mpsc::Sender<NativeScheduledRequest> {
        self.admission_tx.clone()
    }

    /// `true` when the scheduler is armed (num_slots>=2, native=true).
    pub fn is_active(&self) -> bool {
        self.config.is_native()
    }

    /// Request shutdown — the driver will exit after its next admission
    /// recv timeout (or immediately if currently idle).
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Spawn the single driver OS thread. Consumes `self.admission_rx` —
    /// can only be called once. Returns the `JoinHandle` for the driver.
    ///
    /// # Parameters
    /// - `llama` — the `LlamaForward` context, transferred in from the
    ///   main loader. The driver owns it for the process lifetime.
    ///
    /// # Safety / Ownership
    /// `llama` must have been constructed via `LlamaForward::new_multi_seq`
    /// with `n_seq_max >= config.num_slots`. If this constraint is not
    /// respected, `forward_multi_seq` will reject seq_ids `>= n_seq_max`
    /// and the driver will log and drain the slot with `Error` messages.
    pub fn spawn_native_driver(
        &mut self,
        llama: crate::llama_backend::LlamaForward,
    ) -> Result<std::thread::JoinHandle<()>, String> {
        let rx = self.admission_rx.take().ok_or_else(|| {
            "NativeScheduler::spawn_native_driver called twice".to_string()
        })?;
        let shutdown = Arc::clone(&self.shutdown);
        let active_count = Arc::clone(&self.active_count);
        let num_slots = self.config.num_slots;
        let engram_global = self.engram_global.clone();
        let default_alpha = self.default_engram_alpha;

        // Build the slot pool on the main thread so we can alloc_samplers
        // here (model_ptr is valid while `llama` is alive). Then MOVE the
        // pool into the driver thread alongside `llama`.
        let mut pool = SlotPool::new(num_slots);
        unsafe {
            pool.alloc_samplers_with_dry(
                llama.model_raw(),
                0.6, 0.95, 20, 0.05, 0.0, // Qwen3.5 thinking defaults
                0.8, 1.75, 2, -1, // DRY enabled (prod-matching)
            )
            .map_err(|e| format!("NativeScheduler: sampler alloc failed: {}", e))?;
        }
        if let Some(eg) = &engram_global {
            pool.attach_engram(Arc::clone(eg), default_alpha);
            eprintln!(
                "[slot_scheduler:native] Attached global engram ({} tables) with alpha={}",
                eg.len(), default_alpha,
            );
        }

        let tick_us: u64 = std::env::var("CHIMERE_NATIVE_TICK_US")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // Operator-facing alias. `CHIMERE_MAX_PREFILL_CHUNK` is the preferred
        // name (shorter, listed in bin/chimere-server.rs env-var table). The
        // legacy `CHIMERE_NATIVE_MAX_PREFILL_CHUNK` is kept as a fallback so
        // existing systemd units and tests stay bit-identical.
        //
        // Precedence: CHIMERE_MAX_PREFILL_CHUNK > CHIMERE_NATIVE_MAX_PREFILL_CHUNK > 256.
        let max_prefill_chunk: usize = std::env::var("CHIMERE_MAX_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .or_else(|| {
                std::env::var("CHIMERE_NATIVE_MAX_PREFILL_CHUNK")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
            })
            .unwrap_or(256);

        eprintln!(
            "[slot_scheduler:native] driver spawning: num_slots={}, tick_us={}, \
             max_prefill_chunk={}, engram_attached={}",
            num_slots, tick_us, max_prefill_chunk, engram_global.is_some(),
        );

        let handle = std::thread::Builder::new()
            .name("chimere-native-driver".into())
            .spawn(move || {
                let mut driver = NativeDriver {
                    llama,
                    pool,
                    rx,
                    shutdown,
                    max_prefill_chunk,
                    tick_us,
                    active_count,
                };
                driver.run();
            })
            .map_err(|e| format!("failed to spawn chimere-native-driver: {}", e))?;

        Ok(handle)
    }
}

/// Private driver state. Lives entirely on the driver OS thread; never
/// shared across threads after `spawn_native_driver` moves it in.
struct NativeDriver {
    llama: crate::llama_backend::LlamaForward,
    pool: SlotPool,
    rx: mpsc::Receiver<NativeScheduledRequest>,
    shutdown: Arc<AtomicBool>,
    max_prefill_chunk: usize,
    tick_us: u64,
    /// Shared with `NativeScheduler::active_count`. Published each tick
    /// so the `/metrics` scrape sees a live occupancy gauge rather than
    /// a hardcoded zero.
    active_count: Arc<AtomicUsize>,
}

impl NativeDriver {
    /// Main driver loop. Alternates between admission drain, batch build,
    /// forward_multi_seq, per-slot sample+emit, and reap.
    fn run(&mut self) {
        eprintln!("[slot_scheduler:native] driver main loop entered");
        let mut tick_counter: u64 = 0;

        while !self.shutdown.load(Ordering::SeqCst) {
            tick_counter = tick_counter.wrapping_add(1);

            // 1) Drain admission (non-blocking): pull as many new requests
            //    as there are free slots.
            self.admit_new();

            // 2) Reap any slots already in Draining — emit final Done,
            //    release KV pages, return slot to Free.
            self.reap_draining();

            // Publish occupancy after admit/reap so `/metrics` sees the
            // current number of slots serving a request. Relaxed is fine
            // — this is a gauge sampled at scrape time. See also below,
            // after seat_request, to cover the idle-wake path.
            self.active_count
                .store(self.pool.num_active(), Ordering::Relaxed);

            // 3) If no active slots, wait for one admission or shutdown.
            if self.pool.num_active() == 0 {
                // Block on the next admission (up to 100 ms so shutdown can
                // flip). Using try_recv + short sleep here so we don't need
                // a tokio runtime on the driver thread.
                match self.rx.try_recv() {
                    Ok(req) => {
                        self.seat_request(req);
                        // Re-publish after seating so the first tick's
                        // pre-forward phase already shows occupancy > 0.
                        self.active_count
                            .store(self.pool.num_active(), Ordering::Relaxed);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => {
                        // No work. Short sleep to avoid 100% CPU spin.
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        eprintln!("[slot_scheduler:native] admission closed; exiting driver");
                        return;
                    }
                }
                continue;
            }

            // 4) Build and execute one batch.
            if let Err(e) = self.run_one_tick() {
                eprintln!(
                    "[slot_scheduler:native] tick {} failed: {}. Draining all active slots.",
                    tick_counter, e,
                );
                self.drain_all_on_error(&e);
            }

            // 5) Optional idle throttle.
            if self.tick_us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(self.tick_us));
            }
        }
        // Clear the gauge so post-shutdown scrapes don't report stale occupancy.
        self.active_count.store(0, Ordering::Relaxed);
        eprintln!("[slot_scheduler:native] driver exited (shutdown flag set)");
    }

    /// Drain the admission queue into free slots (non-blocking).
    fn admit_new(&mut self) {
        loop {
            // Peek capacity — if no free slot, break.
            if self.pool.alloc_free().is_none() {
                break;
            }
            // Try to pull one req. Non-blocking.
            let req = match self.rx.try_recv() {
                Ok(r) => r,
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // All senders dropped → shutdown triggered by server stop.
                    self.shutdown.store(true, Ordering::SeqCst);
                    return;
                }
            };
            self.seat_request(req);
        }
    }

    /// Seat a new request in the first free slot. Caller MUST have
    /// verified a free slot is available via `alloc_free().is_some()`.
    fn seat_request(&mut self, req: NativeScheduledRequest) {
        // Defense: an empty prompt would cause `end-start-1` to underflow
        // in `tick_prefill_one`. Reject loudly, don't silently hang.
        if req.prompt_tokens.is_empty() {
            let _ = req.tx.try_send(StreamMsg::Error {
                message: "empty prompt tokens — rejecting".to_string(),
            });
            let _ = req.tx.try_send(StreamMsg::Done {
                finish_reason: "error".to_string(),
            });
            return;
        }
        let free = match self.pool.alloc_free() {
            Some(s) => s,
            None => {
                // Shouldn't happen given the caller's precondition, but
                // defend by sending an Error and moving on.
                let _ = req.tx.try_send(StreamMsg::Error {
                    message: "No free slot after admission — race condition".to_string(),
                });
                return;
            }
        };
        eprintln!(
            "[slot_scheduler:native] seat req={} on slot {} (prompt={} toks, max={}, wait_ms={})",
            req.request_id,
            free.id,
            req.prompt_tokens.len(),
            req.params.max_tokens,
            req.enqueued_at.elapsed().as_millis(),
        );
        // Reset slot to a clean state (defensive — mark_free was called
        // when the previous tenant vacated).
        free.mark_free();
        free.state = SlotState::Prefilling { chunks_done: 0 };
        free.pos = 0;
        free.prompt_tokens = req.prompt_tokens;
        free.params = req.params;
        free.engram_alpha = req.engram_alpha;
        free.tx = Some(req.tx);
        free.want_logprobs = req.want_logprobs;
        free.request_id = req.request_id;
        free.cancelled = req.cancelled;
        free.thinking = free.params.enable_thinking;
        free.stats.prompt_tokens = free.prompt_tokens.len() as u32;
        // Seed the recent_context with the prompt tail for engram lookups.
        // We push the whole prompt so the first gen step's engram query
        // has full context; Slot::push_context bounds the window to 256.
        let prompt_tokens_clone = free.prompt_tokens.clone();
        for t in prompt_tokens_clone {
            free.push_context(t);
        }
    }

    /// One scheduler tick: build + execute + sample.
    fn run_one_tick(&mut self) -> Result<(), String> {
        let _tick_g = crate::prof!("sched.tick_build_and_exec");
        // Collect per-slot actions first. We need to borrow slots mutably
        // below when applying sampling results, so we avoid a longer-lived
        // &mut iter by snapshotting action decisions into Vec<(u32, Action)>.
        //
        // Two tick flavours, mutually exclusive (qwen3next constraint,
        // see j4-smoke module doc):
        //   A) Any slot in Prefilling → push one prefill chunk for ONE slot
        //      (prefer the slot with fewest chunks_done for FIFO-ish).
        //   B) All active slots in Generating → push one gen token per slot.
        let has_prefill = self.pool.all_mut().any(|s| {
            matches!(s.state, SlotState::Prefilling { .. })
        });

        if has_prefill {
            self.tick_prefill_one()
        } else {
            self.tick_generate_all()
        }
    }

    /// Tick type A: push one prefill chunk for one Prefilling slot.
    ///
    /// Selects the slot whose `chunks_done` is the smallest (tie-breaker:
    /// lowest `id`). Pushes up to `max_prefill_chunk` tokens in a single
    /// `forward_multi_seq` call. On the final chunk, also samples the
    /// first generate token and transitions the slot to `Generating`.
    fn tick_prefill_one(&mut self) -> Result<(), String> {
        // Find the target slot.
        let slot_id = {
            let mut best: Option<(u32, usize)> = None; // (id, chunks_done)
            for s in self.pool.all_mut() {
                if let SlotState::Prefilling { chunks_done } = s.state {
                    let cand = (s.id, chunks_done);
                    match best {
                        None => best = Some(cand),
                        Some((_, bcd)) if chunks_done < bcd => best = Some(cand),
                        _ => {}
                    }
                }
            }
            best.map(|(id, _)| id).ok_or_else(|| {
                "tick_prefill_one: no Prefilling slot — caller bug".to_string()
            })?
        };

        // Snapshot the prefill chunk we'll push.
        let (chunk_entries, is_last_chunk, starting_pos): (
            Vec<crate::llama_backend::MultiSeqEntry>,
            bool,
            i32,
        ) = {
            let slot = self.pool.get_mut(slot_id).unwrap();
            let chunks_done = match slot.state {
                SlotState::Prefilling { chunks_done } => chunks_done,
                _ => unreachable!(),
            };
            let start = chunks_done * self.max_prefill_chunk;
            let end = (start + self.max_prefill_chunk).min(slot.prompt_tokens.len());
            let is_last = end == slot.prompt_tokens.len();
            let abs_pos_start = start as i32;

            let mut entries: Vec<crate::llama_backend::MultiSeqEntry> =
                Vec::with_capacity(end - start);
            for (i, &tok) in slot.prompt_tokens[start..end].iter().enumerate() {
                let abs_pos = abs_pos_start + i as i32;
                let want_logits = is_last && (i == (end - start - 1));
                entries.push(crate::llama_backend::MultiSeqEntry {
                    token: tok,
                    pos: abs_pos,
                    seq_id: slot.id as i32,
                    request_logits: want_logits,
                });
            }
            (entries, is_last, abs_pos_start)
        };

        let n_entries = chunk_entries.len();
        let out = crate::prof!("ffi.forward_multi_seq", { self.llama.forward_multi_seq(&chunk_entries)? });

        // Update slot state.
        let slot = self.pool.get_mut(slot_id).unwrap();
        let (new_chunks_done, transition_to_gen) = if let SlotState::Prefilling { chunks_done } = slot.state {
            (chunks_done + 1, is_last_chunk)
        } else {
            return Err(format!("slot {} left Prefilling mid-tick", slot_id));
        };
        slot.pos = starting_pos + n_entries as i32;

        if !transition_to_gen {
            slot.state = SlotState::Prefilling { chunks_done: new_chunks_done };
            return Ok(());
        }

        // Last chunk — sample first gen token and transition.
        if !out.iter().any(|(s, _)| *s == slot.id as i32) {
            return Err(format!(
                "tick_prefill_one: no logits returned for seq_id {}",
                slot.id
            ));
        }
        // The logits position in the returned Vec matches the batch idx.
        // Since we only requested logits for ONE entry (last token of
        // last chunk), the sampler must use batch_idx = n_entries - 1.
        let batch_idx = n_entries - 1;

        // Apply per-slot engram bias before sampling. Then release the
        // mutable borrow by extracting the raw sampler pointer (still
        // owned by the slot, not freed) before calling self.llama.
        slot.apply_engram_bias_to_sampler();
        let sampler_raw: Option<*mut std::ffi::c_void> = slot
            .sampler
            .as_ref()
            .filter(|s| s.is_active())
            .map(|s| unsafe { s.as_raw() });
        // end the `slot` borrow; re-fetch after FFI.
        let _ = slot;

        // Sample with per-slot sampler + logprobs.
        let (tok, _logprobs) = match sampler_raw {
            Some(raw) => unsafe {
                self.llama.sample_slot_with_logprobs(raw, batch_idx)
            },
            None => {
                // No sampler → argmax fallback. Shouldn't happen in prod.
                let logits = self.llama.get_logits_at(batch_idx).ok_or_else(|| {
                    format!("null logits at batch_idx {} in prefill fallthrough", batch_idx)
                })?;
                let tok = argmax_u32(logits);
                (tok, Vec::new())
            }
        };

        self.emit_sampled_token(slot_id, tok);

        let slot = self.pool.get_mut(slot_id).unwrap();
        if !matches!(slot.state, SlotState::Draining) {
            slot.state = SlotState::Generating;
        }
        Ok(())
    }

    /// Tick type B: one gen token per Generating slot, all in the same
    /// `forward_multi_seq` batch (distinct seq_ids).
    fn tick_generate_all(&mut self) -> Result<(), String> {
        // Collect (slot_id, last_token, pos) for every Generating slot.
        let gen_inputs: Vec<(u32, u32, i32)> = self
            .pool
            .all_mut()
            .filter_map(|s| {
                if !matches!(s.state, SlotState::Generating) {
                    return None;
                }
                // The last-token is either last in .generated or the last
                // prompt token if no tokens have been generated yet (should
                // not happen because prefill already produced the first
                // gen token during tick_prefill_one).
                let last_tok = *s.generated.last().unwrap_or(s.prompt_tokens.last().unwrap_or(&0));
                Some((s.id, last_tok, s.pos))
            })
            .collect();

        if gen_inputs.is_empty() {
            return Ok(());
        }

        let entries: Vec<crate::llama_backend::MultiSeqEntry> = gen_inputs
            .iter()
            .map(|&(slot_id, tok, pos)| crate::llama_backend::MultiSeqEntry {
                token: tok,
                pos,
                seq_id: slot_id as i32,
                request_logits: true,
            })
            .collect();

        let _out = crate::prof!("ffi.forward_multi_seq", { self.llama.forward_multi_seq(&entries)? });

        // Per-slot apply_bias → sample → emit. Batch index matches input order.
        for (batch_idx, &(slot_id, _tok, _pos)) in gen_inputs.iter().enumerate() {
            // Fetch slot fresh each iteration (can't hold &mut across loop
            // with other slots touching the same pool).
            if let Some(slot) = self.pool.get_mut(slot_id) {
                slot.apply_engram_bias_to_sampler();
            }

            let sampler_raw = self
                .pool
                .get_mut(slot_id)
                .and_then(|s| s.sampler.as_ref().map(|h| unsafe { h.as_raw() }));
            let (tok, _logprobs) = match sampler_raw {
                Some(raw) if !raw.is_null() => unsafe {
                    self.llama.sample_slot_with_logprobs(raw, batch_idx)
                },
                _ => {
                    // Argmax fallback.
                    let raw = self.llama.get_logits_at(batch_idx).ok_or_else(|| {
                        format!("null logits at batch_idx {}", batch_idx)
                    })?;
                    let tok = argmax_u32(raw);
                    (tok, Vec::new())
                }
            };

            self.emit_sampled_token(slot_id, tok);

            // Post-sample: advance pos, check stop/length.
            if let Some(slot) = self.pool.get_mut(slot_id) {
                if matches!(slot.state, SlotState::Draining) {
                    continue;
                }
                slot.pos += 1;
                if slot.stats.generated_tokens >= slot.params.max_tokens {
                    slot.mark_draining("length");
                }
            }
        }

        Ok(())
    }

    /// Push `tok` into the slot (bookkeeping), decode text, emit via try_emit.
    /// Handles stop_tokens, `</think>` flip, client-disconnect in one place.
    fn emit_sampled_token(&mut self, slot_id: u32, tok: u32) {
        let slot = match self.pool.get_mut(slot_id) {
            Some(s) => s,
            None => return,
        };
        slot.generated.push(tok);
        slot.push_context(tok);
        slot.stats.generated_tokens = slot.stats.generated_tokens.saturating_add(1);

        // On-token housekeeping: thinking flip, stop tokens.
        let was_thinking = slot.thinking;
        let cont = slot.on_token_sampled(tok);

        // Decode token to text. We use the simplest possible bytes→utf8
        // approach by delegating to libllama's `token_to_piece`. The
        // driver thread owns `llama` so this is safe.
        let text = self.llama.token_to_piece(tok as i32, false).unwrap_or_default();

        // Build the StreamMsg. Note the semantics from J6:
        //   - When was_thinking=true AND tok==THINK_END (so cont=true unless
        //     stop list also contained it): emit as Thinking (user-visible
        //     </think> stays in reasoning block).
        //   - Otherwise: emit as Token or Thinking based on current flag.
        let msg = if was_thinking {
            StreamMsg::Thinking { text }
        } else {
            StreamMsg::Token { text, logprob: None }
        };

        // Try to emit. On client disconnect (Closed), mark_draining("cancel").
        let emit_ok = slot.try_emit(msg);
        if !emit_ok {
            slot.mark_draining("cancel");
            return;
        }

        // Honour the cancellation flag set by the HTTP handler (e.g. if
        // axum observed the client TCP-dropping before our try_send).
        if slot.cancelled.load(Ordering::SeqCst) {
            slot.mark_draining("cancel");
            return;
        }

        if !cont {
            slot.mark_draining("stop");
        }
    }

    /// Reap every Draining slot: emit exactly one Done frame, release KV
    /// pages for the seq_id, mark the slot Free.
    fn reap_draining(&mut self) {
        // Collect ids first to avoid borrow conflicts with llama kv_cache call.
        let draining_ids: Vec<(u32, String)> = self
            .pool
            .all_mut()
            .filter_map(|s| {
                if matches!(s.state, SlotState::Draining) {
                    let reason = s.finish_reason.clone().unwrap_or_else(|| "stop".to_string());
                    Some((s.id, reason))
                } else {
                    None
                }
            })
            .collect();

        for (slot_id, reason) in draining_ids {
            // Emit Done (best-effort — client may have dropped already).
            if let Some(slot) = self.pool.get_mut(slot_id) {
                let _ = slot.try_emit(StreamMsg::Done {
                    finish_reason: reason.clone(),
                });
            }
            // Free KV/SSM state for this seq_id.
            let _ = self.llama.kv_cache_seq_rm_for(slot_id as i32);
            // Mark slot free (also clears sampler state + recent_context).
            if let Some(slot) = self.pool.get_mut(slot_id) {
                slot.mark_free();
            }
        }
    }

    /// Propagate a fatal tick error to every active slot: emit Error +
    /// Done("error"), release KV pages.
    fn drain_all_on_error(&mut self, err: &str) {
        let active_ids: Vec<u32> = self
            .pool
            .all_mut()
            .filter(|s| !s.is_free())
            .map(|s| s.id)
            .collect();
        for slot_id in active_ids {
            if let Some(slot) = self.pool.get_mut(slot_id) {
                let _ = slot.try_emit(StreamMsg::Error {
                    message: err.to_string(),
                });
                slot.mark_draining("error");
            }
            let _ = self.llama.kv_cache_seq_rm_for(slot_id as i32);
            if let Some(slot) = self.pool.get_mut(slot_id) {
                let _ = slot.try_emit(StreamMsg::Done {
                    finish_reason: "error".to_string(),
                });
                slot.mark_free();
            }
        }
    }
}

/// Internal greedy argmax over a logit slice. Fallback path for when the
/// per-slot sampler is not allocated.
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best_idx: u32 = 0;
    let mut best_val = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests — J1 only verifies types compile and the pool bookkeeping works.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_alloc_and_free() {
        let mut pool = SlotPool::new(2);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.num_active(), 0);

        {
            let s = pool.alloc_free().expect("a free slot");
            assert_eq!(s.id, 0);
            s.state = SlotState::Generating;
        }
        assert_eq!(pool.num_active(), 1);

        {
            let s = pool.alloc_free().expect("second free slot");
            assert_eq!(s.id, 1);
            s.state = SlotState::Generating;
        }
        assert_eq!(pool.num_active(), 2);
        assert!(pool.alloc_free().is_none());

        pool.get_mut(0).unwrap().mark_free();
        assert_eq!(pool.num_active(), 1);
    }

    #[test]
    fn batch_builder_layout() {
        let mut b = BatchBuilder::new();
        b.push(10, 0, 0, false);
        b.push(20, 1, 0, true);
        b.push(30, 0, 1, true);
        b.mark_slot_emit(0, 1);
        b.mark_slot_emit(1, 2);

        assert_eq!(b.toks, vec![10, 20, 30]);
        assert_eq!(b.pos, vec![0, 1, 0]);
        assert_eq!(b.n_seq_id, vec![1, 1, 1]);
        assert_eq!(b.seq_ids, vec![vec![0], vec![0], vec![1]]);
        assert_eq!(b.logits, vec![0, 1, 1]);
        assert_eq!(b.slot_emit_indices, vec![(0u32, 1usize), (1u32, 2usize)]);
    }

    #[test]
    fn config_from_env_default_is_legacy() {
        // Ensure the env var does not pollute this test. We can't reliably
        // set env vars without a global mutex, so just check the hard-coded
        // default value.
        assert_eq!(NUM_SLOTS_DEFAULT, 1);
        let cfg = SchedulerConfig::default();
        assert!(!cfg.is_active());
        assert!(!cfg.is_native());
        assert_eq!(cfg.num_slots, 1);
    }

    #[test]
    fn scheduler_new_disabled_is_noop() {
        // num_slots=1 (default): enabled=false, admission channel exists but
        // handlers will never route to it.
        let cfg = SchedulerConfig::default();
        let sched = Scheduler::new(cfg);
        assert!(!sched.is_active());
        assert_eq!(sched.slot_active_count(), 0);
    }

    #[test]
    fn scheduler_admission_tx_clone_ok() {
        let cfg = SchedulerConfig { num_slots: 2, queue_cap: 4, enabled: true, native: false };
        let sched = Scheduler::new(cfg);
        assert!(sched.is_active());
        let tx1 = sched.admission_tx();
        let tx2 = sched.admission_tx();
        // Capacity is the bound of the mpsc; both senders share it.
        assert_eq!(tx1.capacity(), 4);
        assert_eq!(tx2.capacity(), 4);
    }

    // ----------------------------------------------------------------
    // J6 — rollout safety state machine tests
    // ----------------------------------------------------------------

    #[test]
    fn j6_stop_tokens_empty_never_stops() {
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        assert!(s.params.stop_tokens.is_empty());
        for tok in [0u32, 1, 42, 248069, 1_000_000] {
            assert!(s.on_token_sampled(tok), "empty stop list must never terminate on tok={}", tok);
        }
        assert!(!s.is_draining());
    }

    #[test]
    fn j6_stop_tokens_terminate() {
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.params.stop_tokens = vec![99, 100];
        assert!(s.on_token_sampled(1));
        assert!(s.on_token_sampled(2));
        assert!(!s.on_token_sampled(99));
        s.mark_draining("stop");
        assert!(s.is_draining());
        assert_eq!(s.finish_reason.as_deref(), Some("stop"));
        // mark_draining idempotent on finish_reason.
        s.mark_draining("length");
        assert_eq!(s.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn j6_thinking_flag_flips_on_close() {
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.thinking = true;
        assert!(s.on_token_sampled(123));
        assert!(s.thinking);
        assert!(s.on_token_sampled(456));
        assert!(s.thinking);
        // </think> flips off but does NOT terminate (no stop token set).
        assert!(s.on_token_sampled(Slot::THINK_END_TOKEN));
        assert!(!s.thinking);
        assert!(s.on_token_sampled(789));
        assert!(!s.thinking);
    }

    #[test]
    fn j6_thinking_close_and_stop_coexist() {
        // Client added </think> to stop list (non-reasoning client).
        // Flag must still flip AND generation must terminate.
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.thinking = true;
        s.params.stop_tokens = vec![Slot::THINK_END_TOKEN];
        let cont = s.on_token_sampled(Slot::THINK_END_TOKEN);
        assert!(!s.thinking, "flag must flip before we decide to stop");
        assert!(!cont, "stop_tokens must still terminate on </think>");
    }

    #[test]
    fn j6_thinking_close_when_not_thinking_is_noop() {
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.thinking = false;
        assert!(s.on_token_sampled(Slot::THINK_END_TOKEN));
        assert!(!s.thinking);
    }

    #[test]
    fn j6_try_emit_no_channel_is_false() {
        let mut s = Slot::new(0);
        assert!(s.tx.is_none());
        let ok = s.try_emit(StreamMsg::Token { text: "hi".into(), logprob: None });
        assert!(!ok, "slot without tx must refuse emit");
    }

    #[test]
    fn j6_try_emit_live_channel_succeeds() {
        let (tx, mut rx) = mpsc::channel::<StreamMsg>(4);
        let mut s = Slot::new(0);
        s.tx = Some(tx);
        let ok = s.try_emit(StreamMsg::Token { text: "hi".into(), logprob: None });
        assert!(ok);
        match rx.try_recv().expect("message must be enqueued") {
            StreamMsg::Token { text, .. } => assert_eq!(text, "hi"),
            other => panic!("unexpected msg: {:?}", other),
        }
    }

    #[test]
    fn j6_try_emit_closed_channel_returns_false() {
        // Client drop simulated by dropping the receiver half.
        let (tx, rx) = mpsc::channel::<StreamMsg>(4);
        let mut s = Slot::new(0);
        s.tx = Some(tx);
        drop(rx);
        let ok = s.try_emit(StreamMsg::Token { text: "x".into(), logprob: None });
        assert!(!ok, "closed channel must be detected");
        // Dispatcher contract: mark_draining immediately with "cancel".
        s.mark_draining("cancel");
        assert!(s.is_draining());
        assert_eq!(s.finish_reason.as_deref(), Some("cancel"));
    }

    #[test]
    fn j6_try_emit_full_channel_keeps_slot_alive() {
        // Slow receiver must NOT be confused with a disconnected one.
        let (tx, _rx) = mpsc::channel::<StreamMsg>(1);
        let mut s = Slot::new(0);
        s.tx = Some(tx.clone());
        tx.try_send(StreamMsg::Token { text: "fill".into(), logprob: None }).unwrap();
        // Second send is `Full` — helper must still report success.
        let ok = s.try_emit(StreamMsg::Token { text: "overflow".into(), logprob: None });
        assert!(ok, "Full must not be treated as disconnect");
        assert!(!s.is_draining(), "slow receiver must not kill the slot");
    }

    #[test]
    fn j6_state_machine_free_to_draining_to_free() {
        // Full cycle: Free → Prefilling → Generating → Draining → Free.
        let mut s = Slot::new(0);
        assert!(s.is_free());
        s.state = SlotState::Prefilling { chunks_done: 0 };
        assert!(!s.is_free());
        s.state = SlotState::Generating;
        assert!(!s.is_free());
        s.mark_draining("stop");
        assert!(s.is_draining());
        assert!(!s.is_free());
        s.mark_free();
        assert!(s.is_free());
        assert!(s.finish_reason.is_none());
        assert!(!s.thinking);
    }

    #[test]
    fn j6_slot_reuse_resets_runtime_flags() {
        // Slot reused for a new request — stale state must NOT leak.
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.thinking = true;
        s.params.stop_tokens = vec![1, 2, 3];
        s.finish_reason = Some("stop".into());
        s.mark_free();
        assert!(!s.thinking);
        assert!(s.finish_reason.is_none());
    }

    #[test]
    fn j6_mark_draining_idempotent_state_transition() {
        // Calling mark_draining twice must leave the slot in Draining,
        // preserving the FIRST reason.
        let mut s = Slot::new(0);
        s.state = SlotState::Generating;
        s.mark_draining("stop");
        assert!(s.is_draining());
        s.mark_draining("cancel");
        assert!(s.is_draining());
        assert_eq!(s.finish_reason.as_deref(), Some("stop"));
    }

    // ----------------------------------------------------------------
    // J4-rewrite — NativeScheduler invariants (unit, no FFI)
    // ----------------------------------------------------------------

    #[test]
    fn j4rw_native_config_requires_multislot_env() {
        // Without CHIMERE_MULTISLOT>=2 the native flag is silently ignored.
        // We assert on the struct directly since env-setting in tests is
        // global mutation and racy.
        let cfg_legacy = SchedulerConfig {
            num_slots: 1, queue_cap: 4, enabled: false, native: false,
        };
        let err = NativeScheduler::new(cfg_legacy, None, 0.8)
            .err().expect("expected Err for legacy config");
        assert!(err.contains("CHIMERE_MULTISLOT"));

        let cfg_j2_only = SchedulerConfig {
            num_slots: 2, queue_cap: 4, enabled: true, native: false,
        };
        let err = NativeScheduler::new(cfg_j2_only, None, 0.8)
            .err().expect("expected Err for j2-only config");
        assert!(err.contains("native="));
    }

    #[test]
    fn j4rw_native_active_when_both_flags_set() {
        let cfg = SchedulerConfig {
            num_slots: 2, queue_cap: 8, enabled: true, native: true,
        };
        let sched = NativeScheduler::new(cfg, None, 0.0).unwrap();
        assert!(sched.is_active());
        let tx = sched.admission_tx();
        assert_eq!(tx.capacity(), 8);
    }

    #[test]
    fn j4rw_argmax_fallback_on_empty_is_zero() {
        // edge case: argmax over an empty logits slice returns the default 0
        // (would NaN-crash only if we dereferenced; we don't).
        assert_eq!(argmax_u32(&[]), 0);
        assert_eq!(argmax_u32(&[0.1, 0.2, 0.9, 0.3]), 2);
    }
}
