//! # Multi-slot scheduler scaffolding (M1, Apr 2026)
//!
//! This file is **J1 of the M1 plan** — the types, traits, and state shapes
//! required to move from single-`Mutex<AppStateModel>` to continuous-batching
//! multi-slot serving. It *does not* replace the existing legacy path yet; the
//! scheduler is constructed only when `CHIMERE_MULTISLOT >= 2` and the
//! HTTP handlers still route to the Mutex path by default.
//!
//! See `~/Bureau/plan-M1-multislot-2026-04-24.md` for the full 7-8 day plan.
//! This file covers **J1 only** (types compile, no behavioural change).
//!
//! ## Non-goals in this file
//!
//! - No actual `llama_decode` multi-seq driver yet (that's J3).
//! - No FFI to `chimere_sampler` per-slot allocation (that's J5).
//! - No Engram per-slot bias application (that's J5).
//! - No MTP slot-exclusive gating (that's J5).
//! - No HTTP handler refactor (that's J2).
//!
//! ## What's here
//!
//! - `SlotState`, `Slot`, `SlotPool`
//! - `BatchBuilder` (pure Vec layout, never touches FFI here)
//! - `ScheduledRequest`, `StreamMsg`
//! - `SchedulerConfig::from_env()` — reads `CHIMERE_MULTISLOT`
//! - `NUM_SLOTS_DEFAULT = 1` → path unchanged when the env var is unset
//!
//! Anything that dereferences a raw C pointer is **explicitly TODO**, to keep
//! the J1 commit non-functional and easy to revert.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

        Self {
            num_slots: clamped,
            queue_cap,
            enabled: clamped >= 2,
        }
    }

    /// `true` when the scheduler should actually be used. On `false`, callers
    /// must keep using the legacy `Mutex<AppStateModel>` code path.
    pub fn is_active(&self) -> bool {
        self.enabled
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { num_slots: NUM_SLOTS_DEFAULT, queue_cap: ADMISSION_QUEUE_CAP, enabled: false }
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
        }
    }

    /// `true` when the slot is not currently holding any work. Used by
    /// `SlotPool::alloc_free()`.
    pub fn is_free(&self) -> bool {
        matches!(self.state, SlotState::Free)
    }

    /// Reset state to `Free`. Does *not* release the libllama KV pages —
    /// that's the caller's job (J5 will wire `llama_kv_cache_seq_rm`).
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
    }
}

// ---------------------------------------------------------------------------
// FFI placeholders — J1 keeps them opaque. J5 wires them.
// ---------------------------------------------------------------------------

/// Opaque handle to a `chimere_sampler` C++ struct. Real definition arrives
/// in J5. Until then we use `Arc<()>` so the type exists and `Drop` is a
/// no-op.
#[derive(Debug, Clone)]
pub struct SamplerHandle(pub Arc<()>);

/// Opaque handle to a `MultiEngramLookup` tree. Real definition arrives in J5.
#[derive(Debug, Clone)]
pub struct EngramHandle(pub Arc<()>);

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
// ScheduledRequest — HTTP handler → scheduler
// ---------------------------------------------------------------------------

/// Minimal request descriptor carried through the admission channel. All
/// heavy fields (serialised messages, full sampler config) live in this
/// struct so the scheduler doesn't need to re-parse anything.
pub struct ScheduledRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub want_logprobs: bool,
    pub engram_hint: Option<String>, // route/table name, looked up in J5
    /// How the handler wants to receive tokens.
    pub tx: mpsc::Sender<StreamMsg>,
    /// Set by the handler if the client disconnects. Scheduler polls it.
    pub cancelled: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Scheduler skeleton — J1 stops here
// ---------------------------------------------------------------------------

/// High-level scheduler handle. In J3 this will own the libllama context and
/// spawn a blocking OS thread with a `step()` loop. Today it only holds
/// a config + a channel and answers `is_active()`.
pub struct Scheduler {
    pub config: SchedulerConfig,
    pub admission_tx: mpsc::Sender<ScheduledRequest>,
    #[allow(dead_code)]
    admission_rx: Option<mpsc::Receiver<ScheduledRequest>>,
    pub shutdown: Arc<AtomicBool>,
}

impl Scheduler {
    /// Build a scheduler in the *armed-but-not-running* state. The OS thread
    /// is started by `spawn()` in J3.
    pub fn new(config: SchedulerConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_cap);
        Self {
            config,
            admission_tx: tx,
            admission_rx: Some(rx),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal the (future) scheduler thread to exit on the next step.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
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
        assert_eq!(cfg.num_slots, 1);
    }
}
