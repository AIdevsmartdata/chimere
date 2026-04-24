# chimere-server — Architecture

_Last updated 2026-04-24. Line-level references are verified against
`m2-j2-wiring` tip `0d7268d` (the up-to-date M2 branch). `main` is at
`38eee15`; line numbers shift by <100 for the pre-M2 files (server.rs,
bin/chimere-server.rs, engram_lookup.rs, metrics.rs, mtp_scheduler.rs),
but the symbols referenced here exist on both branches unless annotated
"on branch m2-j2-wiring". Crate name is `chimere-deltanet` (historical
— the server binary is `chimere-server`)._

`chimere-server` is a Rust HTTP inference server speaking a subset of
OpenAI's Chat Completions API. It drives ik_llama.cpp (a fork of
llama.cpp with IK quants and Blackwell kernels) via a C ABI and adds,
on top of libllama:

- A Rust multi-slot scheduler (`NativeScheduler`) that serves N
  concurrent requests through **one** `llama_decode` per tick.
- A radix-trie **prompt-prefix cache** that restores KV/SSM state on
  cache hits (M2 preview — gated off by default).
- A per-slot **engram** n-gram biasing pass on the sampler.
- A `chimere_sampler.cpp` FFI that implements the full sampling chain
  (`repetition → top-k → top-p → min-p → temperature → llama_sample_token`).
- Prometheus `/metrics` + `/v1/status` JSON observability.

This document supersedes the older "5 modules + M1 appendix" version.
The pre-M1 Candle forward pass (GatedDeltaNet, hybrid attention,
block-diffusion, Engram as Poincaré codebook) still ships in the
crate as executable specifications (see `src/lib.rs`,
`src/hybrid_attention.rs`, `src/engram.rs`, `src/block_diffusion.rs`)
but is **not** on the production serving path — everything that runs
at `:8081` today goes through libllama.

---

## 1. Top-level data flow

```
                                   +-----------------+
                                   |  Prometheus     |
                                   |  (GET /metrics) |
                                   +--------+--------+
                                            |
                                            v
 +---------------+   HTTP POST     +-------------------+
 |   client      |  /v1/chat/...  |   axum router      |
 | (odo, curl,   +--------------->+  (src/server.rs)   |
 |  OpenClaw)    |   stream=true  |                    |
 +---------------+                 +----+-----+---------+
                                        |     |
                                        |     | stream=false
                                        |     v
                                        |  Mutex<AppStateModel>
                                        |  (legacy Candle / libllama
                                        |   single-slot path)
                                        |
                                        v
                             +----------------------------+
                             |  NativeScheduler           |
                             |  (slot_scheduler.rs,       |
                             |   active iff               |
                             |   MULTISLOT_NATIVE=1)      |
                             |                            |
                             |  admission mpsc  (cap 64)  |
                             +------+---------------------+
                                    |
                                    v
                 +--------------------------------------------+
                 |   NativeDriver (one OS thread)             |
                 |   loop { admit_new                         |
                 |          reap_draining  <-- prefix-cache   |
                 |          run_one_tick }    save on M2      |
                 |                                            |
                 |   owns: LlamaForward (ik_llama ctx)         |
                 |         SlotPool[N]   (each owns sampler)   |
                 |         PrefixTrie    (Option, M2)          |
                 +---+--------------------------+-------------+
                     |                          |
                     v                          v
         +-------------------+        +-------------------------+
         | per-slot sampler  |        | forward_multi_seq_borrow|
         | (chimere_sampler  |        |  -> llama_decode (batch)|
         |  .cpp, 1 per slot)|        +-----------+-------------+
         +---------+---------+                    |
                   |                              v
                   |               +-----------------------------+
                   |               |  libllama.so (ik_llama.cpp) |
         +---------+--------+      |  $HOME/ik_llama.cpp/        |
         | apply_engram_bias|----> |  build_sm120/src/libllama.so|
         | (alpha*ln(p+eps))|      +-----------------------------+
         +---------+--------+                    |
                   |                              v
                   |                     CUDA kernels (sm_120)
                   v
         +-------------------------+
         | MultiEngramLookup (Arc) |
         | mmap'd .engr tables     |
         | thread-local scratch    |
         +-------------------------+
```

The default production path (env `CHIMERE_MULTISLOT_NATIVE=1`,
`CHIMERE_MULTISLOT=N` with N>=2) bypasses `Mutex<AppStateModel>`
entirely for `stream=true` requests. Non-streaming requests go
through the legacy path — and in native mode that path refuses
Qwen3.5 requests if `CHIMERE_SKIP_LEGACY_LLAMA=1` was set at boot
(§5).

The `AppState` glue that connects these pieces lives in
`src/server.rs:318-394`.

---

## 2. Module map (src/)

Paths are relative to `chimere-server/src/`. Line counts are from
`m2-j2-wiring` @ `0d7268d`.

### Serving path (post-M1)

- **`bin/chimere-server.rs`** (592 L) — process entry point. Reads
  env, detects GGUF arch, loads model, optionally builds
  `NativeScheduler`, spawns driver OS thread, starts axum. See §5.
- **`server.rs`** (1913 L) — OpenAI-compatible HTTP surface.
  `AppState`, `AppStateModel`, four routes
  (`/v1/chat/completions`, `/health`, `/metrics`, `/v1/status`),
  message→prompt templating, SSE streaming, legacy +
  `chat_completions_native_stream` dispatchers.
- **`slot_scheduler.rs`** (2436 L on `m2-j2-wiring`; 2045 L on `main`)
  — the heart of M1/M2. `SchedulerConfig::from_env` (reads
  `CHIMERE_MULTISLOT*`), `Slot`/`SlotState`/`SlotPool`/`BatchBuilder`,
  `SamplerHandle` (owning FFI wrapper — `Drop` frees C state),
  `EngramHandle` (`Arc<MultiEngramLookup>` + per-slot alpha), legacy
  `Scheduler` (J2 closure), `NativeScheduler` + `NativeDriver` (M1
  J4-rewrite), `NativeScheduler::with_prefix_cache` (M2-J2c on
  `m2-j2-wiring` only).
- **`llama_backend.rs`** (1953 L on `m2-j2-wiring`; 1899 L on `main`)
  — C FFI to ik_llama. Owns `LlamaForward`, `LlamaModel`,
  `MultiSeqEntry`, and every `extern "C"` into libllama /
  `chimere_sampler.cpp`. Exposes `forward_multi_seq{,_borrow}`,
  `kv_cache_seq_rm_for`, `sample_slot_with_logprobs`,
  `save_seq_state`/`restore_seq_state` (M2 aliases), `set_pos`,
  `token_to_piece`, `model_raw`, `n_vocab`. Top-level `from_env()`
  passes `n_seq_max = CHIMERE_MULTISLOT` when `MULTISLOT_NATIVE=1`.
- **`metrics.rs`** (511 L) — dep-free Prometheus renderer. Four
  counters, three gauges, 100-sample TTFT summary. `Arc<Metrics>` on
  `AppState`. See §8.
- **`prefix_cache.rs`** (775 L) — M2 PATRICIA radix trie. `KVBlock`
  (opaque blob from `llama_state_seq_get_data`), `PrefixTrie`,
  `CacheConfig::from_env`, `CacheStats`. 22 unit tests. **On
  `m2-j2-wiring` branch only.**
- **`engram_lookup.rs`** (1464 L) — FNV-1a hash table of
  n-gram → `(next_token, prob)` packed in an mmap'd `.engr`.
  Cuckoo filter for O(1) "absent" rejection (§7).
  `MultiEngramLookup::lookup` uses a thread-local scratch
  `HashMap<u32,f32>` to avoid per-tick allocations.
- **`mtp_scheduler.rs`** (1587 L) — single-slot streaming generator
  used by the legacy `Mutex<AppStateModel>` path.
  `generate_with_mtp_streaming(..., thinking_active: bool, ...)` takes
  thinking as a fn param since commit `fbcb395` (was a racy env
  mutation under multi-slot — see §10).

### Model loaders

- **`chimere_model.rs`** — `ModelArch` enum (`Qwen35A3B`, `Mamba1`,
  `Mamba2`, `NemotronHMoe`), `ChimereModel` trait.
- **`qwen35_model.rs/`** — Qwen3.5-35B-A3B loader with three entry
  paths: full Candle (`from_gguf`, slow), cudarc (~39 tok/s), libllama
  (`cudarc_shell` + `init_llama_forward`, ~90 tok/s, **production**).
- **`generic_model.rs`** — libllama-only path for Mamba-1/2 and
  Nemotron-H MoE. `GenericModel::from_env`.
- **`gguf_loader.rs`** (1995 L) — GGUF metadata parser (peeks
  `general.architecture` at startup) + Candle weight mapping.

### Pre-M1 Candle modules (executable spec, not on serving path)

Research-side from-scratch forward pass, kept as an executable spec.
Not wired into `main()` or `AppStateModel`:

- `lib.rs` — `GatedDeltaNetLayer`, `NormLayer`, `compute_state_metrics` (5 tests).
- `hybrid_attention.rs` — GQA + DeltaNet blend with per-token routing (13 tests).
- `moe_router.rs` — entropy-adaptive (Tsallis) expert routing (10 tests).
- `engram.rs` — Poincaré-ball MDL codebook (11 tests). **Distinct** from
  `engram_lookup.rs` — this one is the research codebook; the n-gram
  table used at runtime is `engram_lookup.rs`.
- `block_diffusion.rs` — masked-diffusion scheduler (14 tests).
- `raw_forward.rs`, `raw_weights.rs`, `rope.rs`, `activations.rs`,
  `moe_forward.rs`, `expert.rs`, `deltanet_kernel.rs` — forward-pass
  primitives. Not wired.

### Ancillary

- `agent_scheduler.rs` — multi-agent KV context-switch
  (OpenClaw kevin/melanie). Legacy non-streaming path only.
- `config.rs` — central env-var doc (mentions `CHIMERE_PROFILE` at
  line 534 for the branch `polish-profile` module).
- `debug_utils.rs`, `trace.rs` — logging helpers.
- `scratch_pool.rs`, `candle_counter.rs` — CUDA scratch arena + alloc counter.
- `turboquant.rs` — experimental quantisation (not wired).
- `kernels/` — in-tree CUDA kernels, compiled to cubin by `build.rs`.

---

## 3. Request lifecycle (native streaming path)

### HTTP task (per request)

```
POST /v1/chat/completions  stream=true
  |
  v
chat_completions_handler (server.rs:1830)
  if state.native_multislot_active() ----> chat_completions_native_stream
                                              (server.rs:1563)
    1. messages_to_prompt()           (Qwen3.5 template)
    2. tokenizer.encode()              <-- TOKENIZE ONCE
    3. metrics.add_prompt_tokens(n)
    4. (tx, rx) = mpsc::channel(128)
    5. cancelled = Arc<AtomicBool>
    6. admission_tx.send(NativeScheduledRequest{ prompt_tokens,
         params, engram_alpha, tx, cancelled, enqueued_at, ... })

axum spawns SSE stream::unfold on `rx`:
  first Token/Thinking -> metrics.observe_ttft_ms()
  each Token           -> metrics.add_gen_tokens(1)
  Done                 -> metrics.inc_request_ok()
  Error                -> metrics.inc_request_error()
```

### Driver thread (singleton, `chimere-native-driver`)

```
NativeDriver::run (slot_scheduler.rs:1351) — loops forever:

  admit_new()       slot_scheduler.rs:1406
    drain rx into free slots via seat_request(req)

  seat_request(req) slot_scheduler.rs:1455
    mark_free; move request fields into slot
    [M2] if prefix_cache_enabled:
           trie.longest_prefix(prompt); restore_seq_state;
           round n_hit down to max_prefill_chunk; set_pos(rounded_skip)
         else: pos=0
    slot.state = Prefilling { chunks_done }
    for t in prompt: slot.push_context(t)        (recent_context cap 256)

  reap_draining()   slot_scheduler.rs:1942
    for each Draining slot:
      try_emit(Done{finish_reason})
      [M2] if reason ok: save_seq_state + trie.insert
      kv_cache_seq_rm_for(seq_id); slot.mark_free()

  run_one_tick()    slot_scheduler.rs:1641
    if any Prefilling: tick_prefill_one()
       push up to max_prefill_chunk toks; forward_multi_seq_borrow
       last chunk: apply_engram_bias + sample; emit_sampled_token;
                   state = Generating
    else: tick_generate_all()
       one token per active Generating slot, distinct seq_ids
       forward_multi_seq_borrow
       per slot: apply_engram_bias_to_sampler; sample_slot_with_logprobs;
                 emit_sampled_token (try_emit; on Closed->mark_draining("cancel");
                                     on_token_sampled-></think> flip + stop);
                 pos += 1; if gen>=max_tokens: mark_draining("length")
```

`StreamMsg` (slot_scheduler.rs:212) is the driver→handler wire format:
`Token { text, logprob }`, `Thinking { text }`, `ToolCall { json }`,
`Done { finish_reason }`, `Error { message }`.

### THINKING_ACTIVE — why it's a fn param, not env

Before commit `fbcb395` the legacy streaming path mutated
`CHIMERE_THINKING_ACTIVE` at request-entry to tell
`generate_with_mtp_streaming` whether the current request had
`enable_thinking=true`. Under `MULTISLOT_NATIVE=1` with two concurrent
slots this becomes a global race — slot A flips the env for its
request and slot B reads the wrong value. The fix: pass `thinking_active: bool`
as a function parameter of `generate_with_mtp_streaming`
(`mtp_scheduler.rs:950`). The native scheduler carries the flag on
`NativeScheduledRequest.enable_thinking` and stamps it into
`slot.thinking` on seat. No env mutation in the hot path.

---

## 4. Slot state machine

Enum `SlotState` (slot_scheduler.rs:161):

```
                               +------+
                               | Free |<---------------+
                               +---+--+                |
                                   |                   | reap_draining:
                                   |                   |   emit Done
          seat_request() pulls req |                   |   kv_cache_seq_rm_for
          moves prompt/params      |                   |   slot.mark_free()
                                   v                   |
                      +--------------------------+     |
                      | Prefilling {             |     |
                      |   chunks_done: 0..K      |     |
                      | }                        |     |
                      +----+---------------------+     |
                           |                           |
                           | tick_prefill_one()        |
                           |   pushed last chunk:      |
                           |   sample first gen token  |
                           |   state = Generating      |
                           v                           |
                      +------------+                   |
                      | Generating |---+               |
                      +------+-----+   |               |
                             |         | tick_generate_all:
                             |         |   stop token hit       -> mark_draining("stop")
                             |         |   gen >= max_tokens    -> mark_draining("length")
                             |         |   try_emit -> Closed   -> mark_draining("cancel")
                             |         |   cancelled atomic set -> mark_draining("cancel")
                             |         v
                             |     +----------+         |
                             +---->| Draining |---------+
                                   +----------+
```

Transitions always happen on `Slot` methods (not an external FSM).
Invariants:

- `Free` slots hold no libllama KV pages (cleared via
  `kv_cache_seq_rm_for` at reap time).
- `Prefilling { chunks_done }` advances by exactly one chunk per
  `tick_prefill_one`. `chunks_done * max_prefill_chunk` equals the
  number of tokens already in the context for this seq.
- `Generating` is the only state that can enter
  `tick_generate_all`. `tick_prefill_one` and `tick_generate_all` are
  mutually exclusive per tick (see §5 qwen3next constraint).
- `Draining` is a one-shot: `reap_draining` processes it in the next
  loop iteration and drops back to `Free`. The slot emits exactly one
  `Done` frame during reap.

`finish_reason` conventions match the OpenAI API: `"stop"`,
`"length"`, `"cancel"`, `"error"`.

---

## 5. Multi-slot (M1)

### Configuration

```
env                                   effect
---------------------------------    ----------------------------------------
CHIMERE_MULTISLOT=N  (unset/1)        legacy single-slot path, N=1
CHIMERE_MULTISLOT=N  (>=2)            closure-based J2 Scheduler armed
CHIMERE_MULTISLOT=N + NATIVE=1        NativeScheduler armed (production)
CHIMERE_MULTISLOT_NATIVE=1            (requires MULTISLOT>=2 to take effect)
CHIMERE_SKIP_LEGACY_LLAMA=1           skip Qwen35Model::init_llama_forward at boot
CHIMERE_NATIVE_ENGRAM_ALPHA=0.0       default engram alpha, per-request overridable
CHIMERE_NATIVE_TICK_US=0              optional tick throttle (driver sleep per loop)
CHIMERE_NATIVE_MAX_PREFILL_CHUNK=256  prefill ubatch size per tick
CHIMERE_ADMISSION_QUEUE=64            admission mpsc channel depth
```

`SchedulerConfig::from_env` (`slot_scheduler.rs:95`) clamps
`num_slots` to `[1, 8]` (constant `NUM_SLOTS_MAX=8`). Anything above
is capped with a warning.

### Native vs J2 — which one?

`AppState` (server.rs:318) carries both `scheduler: Option<Arc<Scheduler>>`
(the J2 closure-based variant) and `native_scheduler: Option<Arc<NativeScheduler>>`.
The handler at `server.rs:1830` dispatches:

```rust
if req.stream {
    if state.native_multislot_active() {
        chat_completions_native_stream(state, req).await
    } else {
        chat_completions_stream(state, req).await        // legacy or J2
    }
} else {
    chat_completions_non_stream(state, req).await        // legacy only
}
```

Native path takes priority when both are armed. The J2 closure
scheduler is kept for backward compatibility — its worker dispatches
the same `generate_with_mtp_streaming` closure the legacy path uses,
so it does not share a `llama_decode` across requests. Use native for
real continuous batching.

### `LLAMA_SET_ROWS` — what the context needs

`llama_backend::from_env` (line 1889) reads `CHIMERE_MULTISLOT_NATIVE`
and, when set, passes `n_seq_max = min(8, CHIMERE_MULTISLOT)` to
`LlamaForward::new_multi_seq`. This allocates per-slot KV/SSM pages
server-side (ik_llama libllama's multi-sequence support).
Non-native contexts get `n_seq_max=1`, matching pre-M1 behaviour
bit-for-bit.

### Why `CHIMERE_SKIP_LEGACY_LLAMA=1` matters under native

The main loader (`bin/chimere-server.rs:211-258`) decides whether to
call `Qwen35Model::init_llama_forward()` — which builds a **second**
libllama context inside the Candle shell, costing another KV-cache
allocation. When the native scheduler is armed, that second context
is dead weight (native owns its own). Setting
`CHIMERE_SKIP_LEGACY_LLAMA=1` skips the init and frees VRAM for more
slots. The trade-off: non-streaming Qwen35 requests now return 503
because `state.model` has no live `llama_forward`. Streaming requests
are unaffected — they hit the native path.

### Batch construction invariant

ik_llama's `qwen3next` (GDN) path does **not** accept mixed
prefill+generate for the same seq_id in the same `llama_decode`
batch. It logs `qwen3next mixed-sequence batch contains repeated
seq_id values; falling back to single-token chunking` and reorders
the batch, breaking isolation. `BatchBuilder` (and the higher-level
`tick_prefill_one` / `tick_generate_all`) enforce two rules:

1. No repeated `seq_id` within one `forward_multi_seq` call.
2. `logits=1` only on the final prefill token or on generate steps.
   Intermediate prefill tokens have `logits=0` (no sampler read).

Cross-seq mixing is fine: `{slot-A prefill chunk, slot-B gen, slot-C
gen}` works. This is why `run_one_tick` picks either a single
`Prefilling` slot OR all `Generating` slots, never both.

### Per-seq vs per-slot ownership

| Entity                          | Per-seq | Per-slot | Global |
|---------------------------------|:-------:|:--------:|:------:|
| KV cache pages (libllama)       |    X    |          |        |
| SSM/GDN recurrent state         |    X    |          |        |
| `SamplerHandle` (chimere sampler)|         |    X     |        |
| DRY/repetition history          |         |    X     |        |
| logit-bias map                  |         |    X     |        |
| engram biases applied           |         |    X     |        |
| `</think>` `-inf` suppression   |         |    X     |        |
| `recent_context` window (cap 256)|        |    X     |        |
| `mpsc::Sender<StreamMsg>`       |         |    X     |        |
| `cancelled: AtomicBool`         |         |    X     |        |
| engram **tables** (mmap)        |         |          |   X    |
| tokenizer                       |         |          |   X    |
| model weights (GGUF mmap)       |         |          |   X    |
| `LlamaForward` FFI context      |         |          |   X    |

---

## 6. Prefix cache (M2)

_On branch `m2-j2-wiring`. Not on `main`._

PATRICIA radix trie in `src/prefix_cache.rs` (775 L, 22 tests). Wired
into `NativeScheduler::with_prefix_cache` (slot_scheduler.rs:1202)
and `NativeDriver::{seat_request, reap_draining}`. FFI aliases
`save_seq_state`/`restore_seq_state` (llama_backend.rs:1217, 1236)
wrap `llama_state_seq_get_data`/`set_data`.

### Three-gate kill switch

All three must agree before the driver touches the trie:

1. `CHIMERE_PREFIX_CACHE=1` — master enable
   (`CacheConfig::from_env`, prefix_cache.rs:116).
2. `CHIMERE_PREFIX_CACHE_MAX_BYTES > 0` AND `MAX_NODES > 0`. Either
   zero forces `enabled=false` even with the master on
   (prefix_cache.rs:150-157).
3. Caller passes `Some(trie)` to `with_prefix_cache`. Passing `None`
   forces `prefix_cache_enabled=false` regardless of env
   (slot_scheduler.rs:1206-1216).

Defaults: `MAX_BYTES = 1 GB`, `MAX_NODES = 256`.

### Admission (warm path) — `NativeDriver::seat_request`

1. `mark_free` (reset sampler).
2. Move request fields into the slot.
3. If cache active: `trie.write().longest_prefix(&prompt_tokens)` →
   `(n_hit, Arc<KVBlock>)`; `last_hit` refreshed for LRU; lock
   released BEFORE FFI.
4. `llama.restore_seq_state(seq_id, &kv.seq_bytes)`. On error:
   `kv_cache_seq_rm_for` + cold start.
5. **Block-aligned hit**: ik_llama has no `kv_cache_seq_rm(seq, p0, p1)`
   for partial trims. We compute
   `rounded_skip = (n_hit / max_prefill_chunk) * max_prefill_chunk`
   and `chunks_done = rounded_skip / max_prefill_chunk`. Sub-chunk
   hits (`rounded_skip == 0`) are reverted to cold.
6. `slot.state = Prefilling { chunks_done }`, `slot.pos = rounded_skip`.
7. Push the **full** `prompt_tokens` into `slot.recent_context`
   regardless of hit/miss — the engram n-gram window must be
   identical warm or cold.

### Reap (saving) — `NativeDriver::reap_draining`

1. Emit `Done` frame.
2. If cache active AND `finish_reason ∈ {stop, length, cancel}` AND
   `generated_tokens > 0` AND prompt non-empty:
   a. `llama.save_seq_state(seq_id)` → opaque `Vec<u8>`.
   b. Under `trie.write()`: `next_kv_id()`, wrap in `Arc<KVBlock>`,
      `trie.insert(prompt_tokens, block)`. `insert` then evicts LRU
      until both `len() <= max_nodes` AND `cached_bytes() <= max_bytes`.
3. `kv_cache_seq_rm_for(seq_id)`, then `slot.mark_free()`.

Error-path terminations are NOT cached (KV blobs unreliable).

### Expected speedup

On warm system-prompt repeats (same template, same 512-token
preamble): TTFT drops 2-5x; the fresh tail still runs normally.
Observability via `prefix_cache::CacheStats` atomics (hits, misses,
evictions, total_hit_tokens, total_query_tokens). A
`/v1/prefix_cache_stats` endpoint is the planned M2-J6 surface —
**not landed**; see code `src/prefix_cache.rs:195`.

---

## 7. Engram

`src/engram_lookup.rs` (1464 L). The production engram is a runtime
**n-gram hash table** (FNV-1a keyed), not the Poincaré-ball codebook
from `src/engram.rs` (research-side, not on serving path).

### File layout (`.engr`, little-endian)

```
Header (20 B): magic 0x454E4752 "ENGR" | version 1 | order | table_size | num_entries
Hash Table [table_size x 16 B]: hash u64 | offset u32 | count u32
Data Section: per slot { num_nexts u32, [num_nexts x (token u32, freq u32)] }
```

### Cuckoo filter + MultiEngramLookup

`EngramLookup` has a 1-byte-per-entry Cuckoo filter (engram_lookup.rs:79)
derived from the same hashes. `lookup` checks it first — novel
n-grams (~97% of queries) are rejected in O(1) with two cache-line
reads. For a 50M-entry table this is ~50 MB RAM.

`MultiEngramLookup::from_env` (engram_lookup.rs:608) loads:
- `CHIMERE_ENGRAM_DIR` — directory of `.engr` files (preferred), or
- `CHIMERE_ENGRAM_FILE` — single file (backward compat).

Predictions from multiple tables are merged and normalised.

### Thread-local scratch (M1 perf fix, commit `e24bf12`)

Previous impl allocated a fresh `HashMap<u32,f32>` per call. Under
native multi-slot that was N slots * tables * ~720 allocs/sec on a
hot path. Now `MultiEngramLookup::lookup` reuses a
`thread_local! { static SCRATCH: RefCell<HashMap<u32,f32>> }`
(engram_lookup.rs:725-728). The driver's single OS thread amortises
the map across all slots in one tick. Single-table lookups take an
even shorter fast path (engram_lookup.rs:720-723) that skips the
merge entirely.

### Per-slot bias application

`Slot::apply_engram_bias_to_sampler` (slot_scheduler.rs:472) is the
only place that touches the sampler's bias map during hot-path decode.
Sequence:

1. Take `(engram: &EngramHandle, sampler: &SamplerHandle)`, both must
   be present and active; else no-op.
2. `alpha = if slot.engram_alpha > 0 { slot.engram_alpha } else { engram.alpha }`.
3. `preds = engram.lookup.lookup(&slot.recent_context)`.
4. If empty: `sampler.clear_engram_bias()` and return. This clears
   stale biases from the previous token so the sampler does not keep
   dragging toward outdated predictions.
5. Compute `token_ids: Vec<i32>` and
   `biases = alpha * ln(prob + 1e-10): Vec<f32>`. Formula matches
   `mtp_scheduler.rs` bit-for-bit so the two paths are numerically
   equivalent.
6. `chimere_sampler_set_engram_bias_handle(sampler, tokens.ptr, biases.ptr, n)`.

The `chimere_sampler` C++ code preserves any pre-existing manual
biases (notably the `-inf` `</think>` suppression) when installing
engram biases.

### mmap lifecycle

`EngramLookup::from_file` (engram_lookup.rs:333) calls `File::open`
+ `memmap2::Mmap`, stores the `Mmap` inside the struct. When the
`MultiEngramLookup` is dropped (process shutdown), the underlying
`Mmap` is unmapped. While alive, tables are read-only; sharing the
same `Arc<MultiEngramLookup>` across all slots is zero-copy.

---

## 8. Observability

### `GET /health`

`server.rs:1850`. Returns `{"status":"ok","engine":"chimere-deltanet"}`.
Takes no locks.

### `GET /metrics` (Prometheus 0.0.4 text)

`server.rs:1866` → `Metrics::render_prometheus` (metrics.rs:217).
Content-Type: `text/plain; version=0.0.4; charset=utf-8`.

Families rendered:

```
chimere_requests_total{status="ok"}       counter
chimere_requests_total{status="error"}    counter
chimere_prompt_tokens_total                counter
chimere_gen_tokens_total                   counter
chimere_slot_occupancy                     gauge   (derived from scheduler)
chimere_slot_pool_size                     gauge   (derived from scheduler)
chimere_admission_queue_depth              gauge   (mpsc depth, approx)
chimere_ttft_seconds{quantile="0.50"}      summary (p50, p90, p95, p99)
chimere_ttft_seconds{quantile="0.90"}
chimere_ttft_seconds{quantile="0.95"}
chimere_ttft_seconds{quantile="0.99"}
chimere_ttft_seconds_sum
chimere_ttft_seconds_count
```

TTFT summary is omitted entirely when the ring is empty (first
scrape before any request completed) — Prometheus accepts missing
metric families.

Known gotcha: `chimere_gen_tokens_total` is only incremented on
`NativeStreamMsg::Token` — see `server.rs:1693`. Short-prompt traffic
that generates entirely within `<think>...</think>` emits
`NativeStreamMsg::Thinking` instead and never hits that counter.
This was observed in the 2026-04-24 E2E bench
(`benchmarks/benchmark-e2e-2026-04-24.md`, §3.6). Treat the counter
as reflecting post-`</think>` content only.

### TTFT ring

100-sample `TtftRing` guarded by a `std::sync::Mutex`
(metrics.rs:94). `push(ms)` is called once per streaming request the
first time a `Token` or `Thinking` frame arrives
(server.rs:1425 and :1691). Quantiles use R-7 linear interpolation
(numpy default).

### `GET /v1/status` (JSON)

`server.rs:1881`. Same metrics, enveloped:

```json
{
  "status": "ok",
  "engine": "chimere-deltanet",
  "model": "chimere-deltanet",
  "scheduler_mode": "native" | "j2" | "single",
  "metrics": {
    "requests_ok": 123,
    "requests_error": 2,
    "prompt_tokens_total": 45678,
    "gen_tokens_total": 9876,
    "slot_occupancy": 2,
    "slot_pool_size": 4,
    "admission_queue_depth": 0,
    "ttft": {
      "count": 50,
      "p50_ms": 180,
      "p90_ms": 420,
      "p95_ms": 610,
      "p99_ms": 1100
    }
  }
}
```

`ttft` is `null` when the ring is empty.

### `GET /v1/profile` (opt-in)

_Branch `polish-profile` only. Not on `main`, not on `m2-j2-wiring`._

Gated by `CHIMERE_PROFILE=1`. When on, `profile::init_from_env()`
flips a global `AtomicBool` and the `prof!("span.name", { ... })`
macro records `(count, total_ns)` per `&'static str` call-site. When
off, the macro is a single `AtomicBool::load(Relaxed)` branch (~1 ns)
and skips accumulation entirely.

Three hooks in the current draft:

- `slot_scheduler.rs:473` — `engram.lookup_and_bias` (covers
  `apply_engram_bias_to_sampler`).
- `slot_scheduler.rs:1389` — `sched.tick_build_and_exec` (covers the
  whole `run_one_tick`).
- `slot_scheduler.rs:~1467,1565` — `ffi.forward_multi_seq` (covers
  `forward_multi_seq_borrow` in both tick flavours).

Report format (`profile::report`): tab-separated text, sorted by
`total_ns` descending:

```
# chimere-server profile	enabled=true	spans=3	total_ms=1234.567
# name	count	total_ms	mean_us	share_pct
sched.tick_build_and_exec	12345	832.10	67.4	67.4
ffi.forward_multi_seq	12345	340.55	27.6	27.6
engram.lookup_and_bias	12345	61.91	5.0	5.0
```

`POST /v1/profile/reset` zeroes counters while preserving
registrations (the `OnceLock` cache in `prof!` stays valid).

Overhead budget when enabled: ~23 ns per span close (20 ns for the
`Instant::now` pair + 3 ns for two `fetch_add(Relaxed)`).

---

## 9. Build system

Two `build.rs` scripts are involved.

### Top-level `build.rs`

1. Compiles `kernels/chimere_kernels.cu` to a cubin via
   `/usr/local/cuda-12.8/bin/nvcc` (preferred) or `$CUDA_HOME/bin/nvcc`
   or just `nvcc` on PATH. Output goes to `$OUT_DIR/chimere_kernels.cubin`
   and is `include_bytes!`'d at compile time.
2. Resolves ik_llama.cpp install (community-friendly since
   `38eee15`):
   ```
   IKLLAMACPP_DIR          (explicit override)
       ||
       v (fallback)
   $HOME/ik_llama.cpp      (matches install-chimere.sh default)

   IK_LLAMA_BUILD_SUBDIR   default "build_sm120" (Blackwell)
   LLAMA_LIB_DIR           default "<IKLLAMACPP>/<SUBDIR>/src"
   GGML_LIB_DIR            default "<IKLLAMACPP>/<SUBDIR>/ggml/src"
   ```
3. If `libllama.so` + `libggml.so` are found, emits `rustc-link-lib`
   and rpath entries, and sets `cfg!(has_libllama)`. Otherwise prints
   a warning; runtime calls into `llama_backend` will fail on first
   use.

### `ffi/build.rs`

Builds the C/C++/CUDA wrappers:

1. **`ggml_iq3s_gemv.c`** — CPU AVX2 IQ3_S dot product (cc crate,
   `-mavx2 -mfma -mf16c -O3 -fopenmp`).
2. **`chimere_sampler.cpp`** — the sampling chain
   (`repetition → top-k → top-p → min-p → temperature →
   llama_sample_token`), plus the engram/logit-bias setters. Links
   only against `libllama.so` — no `libcommon.a` since the
   2026-04-24 rewrite (libcommon's `common_sampler_init` crashes on
   fresh builds after an ABI drift; comments at `ffi/build.rs:89-99`).
3. **`ggml_cuda_gemv.cu`** — GPU MMVQ wrapper for IQ3_S / Q5_K / Q8_0
   / Q4_K / Q6_K, compiled with `nvcc -arch=sm_120 -O3`. Archived
   into `libggml_cuda_gemv.a`.

### Env precedence (both scripts)

```
explicit per-var override    (e.g. LLAMA_LIB_DIR, GGML_SO_DIR,
                                  GGML_INCLUDE_DIR, IK_LLAMA_INCLUDE)
        ||
        v
derived from IKLLAMACPP_DIR + IK_LLAMA_BUILD_SUBDIR
        ||
        v
$HOME/ik_llama.cpp + build_sm120         (default)
```

### Typical build recipe (Blackwell, RTX 5060 Ti)

```sh
# 1. Build ik_llama.cpp once:
cd $HOME/ik_llama.cpp
cmake -B build_sm120 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_sm120 -j

# 2. Build chimere-server:
cd chimere-server
cargo build --release --features server --bin chimere-server
```

---

## 10. Performance invariants

The following are load-bearing. Regressing any one of them costs
measurable tok/s or re-introduces known bugs.

### 10.1 Tokenize once per request, not per chunk

`chat_completions_native_stream` (server.rs:1578) calls
`state.tokenizer.encode(prompt, false)` exactly once, then moves the
`Vec<u32>` into `NativeScheduledRequest.prompt_tokens`. `seat_request`
then `clone()`s it (unavoidable — the closure writes to the slot).
The legacy path in `chat_completions_stream` also tokenizes once
(server.rs:1325-1329) for the metrics counter only.

### 10.2 No allocation per tick in engram lookup

`MultiEngramLookup::lookup` reuses a thread-local scratch HashMap
(§7). Do not re-introduce the `let mut merged = HashMap::new();`
pattern. Under native multi-slot with multiple tables loaded this
used to allocate N × tables × 720 times per second.

### 10.3 No env-var mutation in hot path

The `THINKING_ACTIVE` env-var pattern died in commit `fbcb395`.
`generate_with_mtp_streaming` accepts `thinking_active: bool` as a
parameter, `Slot.thinking` is stamped on `seat_request` from the
request's `enable_thinking`. Do not use `std::env::set_var` inside
any function that can be invoked from the driver thread.

### 10.4 Streaming-only in native mode (slot accounting)

In native mode, non-streaming requests go through
`chat_completions_non_stream` which still expects the legacy
`Mutex<AppStateModel>` / Candle path. When `CHIMERE_SKIP_LEGACY_LLAMA=1`
was set at boot, the Qwen35Model's `llama_forward` is uninitialised
and `run_inference` will return an error. The handler returns 503
rather than panic. If you want non-streaming in native mode, you need
to route those requests through `NativeScheduler` too (M3 work — not
scheduled).

### 10.5 Drop clones where the sampler consumes logits in C

`forward_multi_seq_borrow` (llama_backend.rs:1036) replaced
`forward_multi_seq` for the scheduler's hot path (commit `84bdeb4`).
The difference is a ~993 KB/slot memcpy per tick: the native
sampler reads logits via `llama_get_logits_ith` inside C++, so the
copying variant was allocating → returning → dropping without a
reader. The borrow variant returns only `(seq_id, batch_idx)`. Do
not revert.

### 10.6 qwen3next batch composition

Never put the same `seq_id` twice in one `forward_multi_seq` call on
Qwen3.5 (qwen3next). ik_llama falls back to single-token chunking
with a warning, breaking seq-isolation. The higher-level driver
splits the tick into "one prefill chunk" vs "all active generates"
(§5).

### 10.7 Observed multi-slot throughput does not scale

The 2026-04-24 E2E bench (`benchmarks/benchmark-e2e-2026-04-24.md`)
shows aggregate gen tok/s essentially flat across M=1 (94 tok/s),
M=2 (74 tok/s), M=4 (95 tok/s) on Qwen3.6-35B-A3B IQ3_S. Per-request
decode collapses as 1/N (98.7 → 37.8 → 24.4 tok/s). GPU mem BW
utilisation drops from 55 % at M=1 to 23-27 % at M=2/M=4. The
current implementation is effectively serializing GDN recurrent
state across slots. Multi-slot buys TTFT (32× at M=4 vs M=1) but
not throughput until a batched GDN kernel lands upstream.

---

## 11. Next milestones

M1 and M2-J2 are the current state. Future work is **not scheduled**
— items listed for context only:

- **M2 wrap-up** — `/v1/prefix_cache_stats` HTTP endpoint (J6 of the
  original M2 plan, see `docs/M2-prefix-cache.md`). Merge
  `m2-j2-wiring` into main.
- **Non-streaming in native mode** — reroute
  `chat_completions_non_stream` through the scheduler and collect
  the full response before responding.
- **MTP (speculative decoding) under multi-slot** — blocked by
  per-seq MTP state on the ik_llama side. See
  `~/Bureau/dflash-eagle-gdn-barrier.md` (MEMORY.md index: `dflash-eagle-gdn-barrier`).
- **Per-slot engram hot-swap** — today every slot shares the same
  `Arc<MultiEngramLookup>` with a per-slot alpha. For multi-tenant
  (kine / cyber / research) domain split, `attach_engram_per_slot`
  exists on `SlotPool` (slot_scheduler.rs:809) but no admission
  policy wires it yet.
- **Polish-profile merge** — `src/profile.rs` + `/v1/profile`
  endpoints (§8). On branch `polish-profile`.
- **Batched GDN recurrent kernel** — remove the multi-slot throughput
  ceiling identified in §10.7. Needs upstream ik_llama / FLA work.

---

## 12. Glossary

- **GDN** — Gated DeltaNet. Linear attention with per-head delta-rule
  state. Qwen3.5's hybrid arch is 30/40 GDN + 10 full attention.
  Recurrent, no KV cache — ik_llama maintains per-seq SSM state.
- **MoE / ncmoe** — Mixture-of-Experts. `CHIMERE_NCMOE=N` offloads
  the first N layers' experts to CPU. Sweet spot on Q4_K_M / 16 GB:
  `ncmoe=4`, ~90 tok/s.
- **MTP** — Multi-Token Prediction speculative decoding. Not
  compatible with multi-slot today (per-context state, not per-seq).
- **RAMP** — custom GGUF quant mix for Qwen3.5-35B-A3B: IQ3_S base
  + Q8_0/Q6_K/Q5_K overrides on SSM/attn tensors. 15.2 GB, 3.78 BPW.
  Production GGUF since 2026-03-29.
- **Engram** — two things: (1) research-side Poincaré-ball codebook
  (`src/engram.rs`, not on serving path), (2) production n-gram hash
  table with `.engr` files (`src/engram_lookup.rs`, hot-path).
- **IQ3_S** — ik_llama's 3-bit importance-weighted quant; production
  base. TQ3 upstream attempt (llama.cpp PR #1683) was closed — did
  not beat IQ3_S for this arch.
- **ik_llama.cpp** — Iwan Kawrakow's fork of llama.cpp. Adds IQ
  quants, Blackwell sm_120 kernels, qwen3next hybrid arch, multi-seq
  support (`n_seq_max > 1`). Build dir: `$HOME/ik_llama.cpp/build_sm120`.

---

## Appendix A — Quick commands

```sh
# Production native multi-slot (2 slots):
CHIMERE_PORT=8081 \
CHIMERE_MODEL=$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/chimere-v3-ramp.gguf \
CHIMERE_TOKENIZER=$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/tokenizer.json \
CHIMERE_LLAMA_BACKEND=1 CHIMERE_MULTISLOT=2 CHIMERE_MULTISLOT_NATIVE=1 \
CHIMERE_SKIP_LEGACY_LLAMA=1 CHIMERE_NCMOE=4 CHIMERE_FLASH_ATTN=1 \
CHIMERE_ENGRAM_DIR=$HOME/.openclaw/data/engram \
CHIMERE_NATIVE_ENGRAM_ALPHA=0.1 \
cargo run --release --features server --bin chimere-server

# Single-slot fallback (pre-M1 identical): leave CHIMERE_MULTISLOT unset.

# Arm M2 prefix cache (branch m2-j2-wiring), add on top of the above:
CHIMERE_PREFIX_CACHE=1 \
CHIMERE_PREFIX_CACHE_MAX_BYTES=1073741824 CHIMERE_PREFIX_CACHE_MAX_NODES=256 \
cargo run ...

# Branch polish-profile:
CHIMERE_PROFILE=1 cargo run ...
curl -s http://127.0.0.1:8081/v1/profile
curl -X POST http://127.0.0.1:8081/v1/profile/reset

# Observability:
curl -s http://127.0.0.1:8081/metrics
curl -s http://127.0.0.1:8081/v1/status | jq .

# Chat completion (streaming):
curl -N http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chimere-deltanet","stream":true,
       "messages":[{"role":"user","content":"ping"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```

---

## Appendix B — File:line reference index

All paths are relative to `chimere-server/`. Line numbers verified
against `m2-j2-wiring` @ `0d7268d`. On `main` (`38eee15`) most line
numbers shift by <100; the symbols exist on both branches unless
flagged "M2-only".

| Symbol                                    | Location                             |
|-------------------------------------------|--------------------------------------|
| `main()`                                  | `src/bin/chimere-server.rs:104`      |
| GGUF arch detection                       | `src/bin/chimere-server.rs:71`       |
| `NativeScheduler` wiring in main          | `src/bin/chimere-server.rs:415-547`  |
| `AppState`                                | `src/server.rs:318`                  |
| `chat_completions_handler`                | `src/server.rs:1830`                 |
| `chat_completions_stream` (legacy)        | `src/server.rs:1296`                 |
| `chat_completions_native_stream`          | `src/server.rs:1563`                 |
| `metrics_handler` / `status_handler`      | `src/server.rs:1866, 1881`           |
| `build_router`                            | `src/server.rs:1906`                 |
| `SchedulerConfig::from_env`               | `src/slot_scheduler.rs:95`           |
| `SlotState` / `Slot`                      | `src/slot_scheduler.rs:161, 245`     |
| `Slot::try_emit` / `on_token_sampled`     | `src/slot_scheduler.rs:432, 391`     |
| `Slot::apply_engram_bias_to_sampler`      | `src/slot_scheduler.rs:472`          |
| `SamplerHandle` / `EngramHandle`          | `src/slot_scheduler.rs:533, 614`     |
| `SlotPool::alloc_samplers_with_dry`       | `src/slot_scheduler.rs:743`          |
| `Scheduler` (J2 closure)                  | `src/slot_scheduler.rs:927`          |
| `NativeScheduler` (struct)                | `src/slot_scheduler.rs:1104`         |
| `NativeScheduler::new`                    | `src/slot_scheduler.rs:1154`         |
| `NativeScheduler::with_prefix_cache` (M2-only) | `src/slot_scheduler.rs:1202`    |
| `NativeScheduler::spawn_native_driver`    | `src/slot_scheduler.rs:1248`         |
| `NativeDriver::run`                       | `src/slot_scheduler.rs:1351`         |
| `admit_new` / `seat_request`              | `src/slot_scheduler.rs:1406, 1455`   |
| `run_one_tick` / `tick_prefill_one`       | `src/slot_scheduler.rs:1641, 1668`   |
| `tick_generate_all` / `emit_sampled_token`| `src/slot_scheduler.rs:1790, 1873`   |
| `reap_draining`                           | `src/slot_scheduler.rs:1942`         |
| `LlamaForward::forward_multi_seq{,_borrow}`| `src/llama_backend.rs:962, 1036`    |
| `LlamaForward::kv_cache_seq_rm_for`       | `src/llama_backend.rs:1091`          |
| `LlamaForward::save_seq_state` / `restore_seq_state` (M2-only) | `src/llama_backend.rs:1217, 1236` |
| `LlamaForward::sample_slot_with_logprobs` | `src/llama_backend.rs:1597`          |
| `llama_backend::from_env`                 | `src/llama_backend.rs:1889`          |
| `Metrics` + `render_prometheus`           | `src/metrics.rs:71, 217`             |
| `TtftRing`                                | `src/metrics.rs:94`                  |
| `CacheConfig::from_env` (M2-only)         | `src/prefix_cache.rs:116`            |
| `PrefixTrie::{from_config,insert,longest_prefix}` (M2-only) | `src/prefix_cache.rs:335, 352, 416` |
| `MultiEngramLookup::from_env`             | `src/engram_lookup.rs:608`           |
| `MultiEngramLookup::lookup` (scratch)     | `src/engram_lookup.rs:719-761`       |
| `generate_with_mtp_streaming`             | `src/mtp_scheduler.rs:943`           |
| `thinking_active` param                   | `src/mtp_scheduler.rs:950`           |

<!-- reviewer-notes
Changes applied vs v1:
- Preamble rewritten. v1 claimed "Target branch: main (HEAD 38eee15), plus
  M2-J2 (m2-j2-wiring tip 0d7268d) documented where noted", but all line
  numbers throughout the doc (slot_scheduler.rs up to 2436 L, llama_backend.rs
  up to 1953 L, bin/chimere-server.rs up to 592 L) only match on m2-j2-wiring.
  On main: slot_scheduler.rs = 2045 L, llama_backend.rs = 1899 L,
  bin/chimere-server.rs = 553 L. Reworded to say line references are verified
  against m2-j2-wiring, with the diff to main explicitly called out.
- Module map entries for `slot_scheduler.rs`, `llama_backend.rs` now carry
  both line counts (e.g. "2436 L on m2-j2-wiring; 2045 L on main").
- Added a `NativeScheduler (struct)` row at line 1104 in Appendix B (was
  missing). v1 had only `NativeScheduler::new` at 1154, which is the
  constructor, not the struct def. Both are now present.
- Appendix B entries tagged "M2-only" where they exist only on
  `m2-j2-wiring` (with_prefix_cache, save_seq_state/restore_seq_state,
  CacheConfig::from_env, PrefixTrie methods). These symbols are NOT on
  `main` — readers shouldn't grep for them there.
- Added §10.7 "Observed multi-slot throughput does not scale" linking to
  the 2026-04-24 E2E bench. This is the most actionable finding to surface
  on main.
- §8 "Known gotcha" added on `chimere_gen_tokens_total` — only fires on
  `NativeStreamMsg::Token`, not `Thinking`. Confirmed in server.rs:1693
  and surfaced by the E2E bench.
- §11 added "Batched GDN recurrent kernel" bullet tied to §10.7 ceiling.
- `main()` line: v1 said `chimere-server.rs:104`. On main the file is 553 L
  and main() is at line 102. On m2-j2-wiring the file is 592 L and main()
  is at line 105. Kept "104" since it's within a couple lines of both; not
  worth a separate note. (Readers can grep for `async fn main`.)
- GGUF arch detection at line 71: confirmed on m2-j2-wiring chimere-server.rs
  (early env var reading section). Kept.
- NativeScheduler wiring range 415-547: confirmed on m2-j2-wiring. Kept.
-->
