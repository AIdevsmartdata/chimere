# Changelog

All notable changes to `chimere-server` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project loosely
follows semantic versioning at the binary level.

## [Unreleased]

### M1 Multi-slot (2026-04-24)

Ships the scaffolding, FFI primitives and documentation required to
serve N concurrent `/v1/chat/completions` requests through a single
`chimere-server` process. The legacy `Mutex<AppStateModel>` path is
byte-for-byte unchanged — the new serving path activates only when
`CHIMERE_MULTISLOT` is explicitly set to `>= 2`.

See `chimere-server/ARCHITECTURE.md` §"M1 Multi-slot + Continuous
Batching (Apr 2026)" for the end-to-end design, per-seq / per-slot
data ownership, batch-construction invariants, and the MTP gating
policy. See `~/Bureau/plan-M1-multislot-2026-04-24.md` for the 7-8
day roadmap this work implements.

#### Added — scaffolding (J1)

- **`slot_scheduler.rs`** (`chimere-server/src/slot_scheduler.rs`,
  ~960 lines across all M1 commits). Contains:
  - `SchedulerConfig` + `from_env()` — reads `CHIMERE_MULTISLOT`,
    clamps to `[1, 8]`, and sets `enabled = num_slots >= 2`.
  - `Scheduler` — owns the admission mpsc channel
    (`ADMISSION_QUEUE_CAP = 64` default, overridable via
    `CHIMERE_ADMISSION_QUEUE`), a `SlotPool` behind a `std::Mutex`,
    and a `shutdown` atomic.
  - `SlotPool` + `Slot` + `SlotState` finite-state machine
    (`Free | Prefilling { chunks_done } | Generating | Draining`).
  - `BatchBuilder` — pure-Rust accumulator matching the `LlamaBatch`
    layout (`toks`, `pos`, `n_seq_id`, `seq_ids`, `logits`,
    `slot_emit_indices`).
  - `ScheduledRequest` / `ScheduledRequestMeta` — cheap-clone work
    envelope plus a `Box<dyn FnOnce + Send>` closure so the
    scheduler stays decoupled from `chimere_model::*`.
  - 5 unit tests: pool bookkeeping, batch layout, config default,
    scheduler-new is a no-op at N=1, `admission_tx` is cheap-clone.
- **`AppState.scheduler: Option<Arc<Scheduler>>`** field in
  `server.rs:329`, plus `AppState::multislot_active()` helper.

#### Added — admission queue + dispatcher (J2)

- `Scheduler::admission_tx()` cheap-clone sender and
  `Scheduler::spawn_workers()` OS-thread dispatcher
  (`chimere-sched-dispatch`) that drains the admission channel and
  runs each `ScheduledRequest.run` closure. Per-request queue-wait
  ms is logged to stderr on dispatch.
- `bin/chimere-server.rs:312-342` builds the scheduler iff
  `SchedulerConfig::is_active()`, spawns the dispatcher, and
  detaches the JoinHandle (process-lifetime worker).
- **`bin/j2_smoke.rs`** — 2 concurrent fake-inference closures,
  asserts the dispatcher accepts both and interleaves their outputs.

#### Added — multi-seq FFI decoder driver (J3)

- **`LlamaForward::forward_multi_seq(&mut self, entries)
  -> Result<Vec<(i32, Vec<f32>)>, String>`** —
  `chimere-server/src/llama_backend.rs`. Composes N seq_ids into a
  single `llama_batch`, calls `llama_decode` once, returns per-entry
  logits for entries with `request_logits = true`. libllama routes
  K/V writes to per-seq pages (transformer) or per-seq SSM states
  (Mamba / Nemotron-H / qwen3next GDN).
- **`MultiSeqEntry { token, pos, seq_id, request_logits }`** — input
  shape for the above.
- **`LlamaForward::kv_cache_seq_rm_for(seq_id) -> bool`** —
  frees KV pages owned by a finished sequence. The legacy seq_id=0
  hard-code at line 1015 is kept for the single-slot path.
- **`LlamaForward::vocab_size() -> usize`** — public accessor so
  callers can slice the `Vec<f32>` of logits returned by
  `forward_multi_seq`.
- **`bin/j3_smoke.rs`** — loads a model with `n_seq_max = 2`,
  prefills 2 distinct prompts on seq_id 0 and 1 in one multi-seq
  batch, then 10 generate steps alternated via multi-seq batches,
  asserts the two token streams diverge.

#### Added — chunked prefill + mixed-seq generate (J4)

- **`bin/j4_smoke.rs`** — chunked prefill of one seq interleaved with
  concurrent generate of another seq. Uses `forward_multi_seq` as the
  scheduler will once the HTTP dispatcher rewrite lands. The smoke
  proves seq-1's token stream is bit-for-bit identical whether it
  runs in isolation or interleaved with a 512-token prefill of
  seq-0, and documents the ik_llama qwen3next constraint "no
  repeated seq_id within one `llama_decode` batch".
- **`llama_grammar_apply` FFI shim** flipped from `abort` to no-op
  (`chimere_sampler.cpp`) — the original `abort()` crashed every
  fresh build on top of the 2026-04-24 libcommon rebuild.
- **`CHIMERE_SKIP_SAMPLER_INIT`** environment variable wired for the
  smokes (default `0`, honours `0`/`false`). Lets targeted repros
  bypass the C++ sampler init path.

#### Added — per-slot sampler + per-slot engram (J5)

- **`SamplerHandle`** (owned, `Send`, `!Sync`, `!Clone`) —
  `slot_scheduler.rs`. Wraps a `*mut c_void` returned by
  `chimere_sampler_alloc_with_dry`; `Drop` calls
  `chimere_sampler_free_handle`. One handle per active slot →
  logit_bias maps, DRY histories and repetition counters are
  **per-slot**, no cross-slot leakage by construction.
- **`EngramHandle { lookup: Arc<MultiEngramLookup>, alpha: f32 }`** —
  cheap-clone. Engram tables (mmap'd `.engr` files) stay global;
  only the alpha is per-slot.
- **`Slot::apply_engram_bias_to_sampler()`** — implements the
  production formula `alpha * ln(prob + 1e-10)` identically to
  `mtp_scheduler.rs`, so the multi-slot path is numerically
  equivalent to the single-slot path on the same prompt.
- **`SlotPool::alloc_samplers_with_dry()`** — allocates N
  independent samplers at scheduler boot; rolls back to "no sampler"
  if any slot's alloc fails.
- **`SlotPool::attach_engram{,_per_slot}()`** — attaches a single
  shared lookup Arc (homogeneous deployment) or one lookup per slot
  (multi-tenant, useful for tests and kine / cyber / research domain
  split).
- **FFI helpers** on `chimere_sampler.cpp`:
  `chimere_sampler_alloc_with_dry`,
  `chimere_sampler_set_engram_bias_handle`,
  `chimere_sampler_set_logit_bias_handle`,
  `chimere_sampler_clear_engram_bias_handle`,
  `chimere_sampler_reset_handle`,
  `chimere_sampler_free_handle`.
- **`LlamaForward::sample_slot_with_logprobs(sampler, idx)`** —
  per-slot sample from a given batch index (used by the
  scheduler's per-slot sample step).
- **`bin/j5_smoke.rs`** — two slots with DIFFERENT in-memory engram
  tables, asserts target_a never appears in slot 1's top-5 and
  target_b never appears in slot 0's top-5 (no cross-slot
  logit_bias leakage).

#### Fixed — libcommon ABI drift (J5 unblock)

Previously the chimere FFI `chimere_sampler.cpp` depended on
`common_sampler_init` from `libcommon.a`. On 2026-04-24, ik_llama's
`sampling.cpp` was rebuilt with upstream's autoparser refactor
(`rbudget` field + `reasoning_budget_*` params) while the
`sampling.h` checked out on the chimere branch did not carry those
changes. The linker resolved init anyway, init wrote past the end of
the smaller `new common_sampler()` allocation, and a C++ exception
propagated through `extern "C"` into Rust — aborting with
`fatal runtime error: Rust cannot catch foreign exceptions`.

Fix: `chimere_sampler.cpp` no longer calls anything from libcommon.
A minimal in-file sampler is built entirely on libllama.so's stable
`llama_sample_*` C API (repetition → top-k → top-p → min-p →
temperature → `llama_sample_token`, greedy when temp ≤ 0).
`libcommon.a` is no longer linked (`ffi/build.rs`). The
`llama_grammar_apply` shim is kept as a defensive no-op.

Full write-up: `~/Bureau/chimere-sampler-unblock-2026-04-24.md`.

#### Added — bench harness + docs (J8)

- **`bin/bench_m1.rs`** — async reqwest load generator. Reads
  `BENCH_URL` (default `:8082`, hard-refuse on `:8081` so production
  is never targeted by accident), `BENCH_CONC`, `BENCH_N`,
  `BENCH_MAX_TOKENS`, `BENCH_PASS_LABEL`, `BENCH_BASELINE_TPS`.
  Emits aggregate tok/s, p50/p95/p99 latency, VRAM delta via
  `nvidia-smi`, and a coarse isolation assertion (no two distinct
  prompts returning byte-identical bodies). Exit 5 = ratio below
  target; exit 6 = isolation broken.
- **`scripts/bench_m1.sh`** — one-shot sweep wrapper. Launches three
  fresh `chimere-server` processes on `:8082` with
  `CHIMERE_MULTISLOT = 1 / 2 / 4` and runs `bench-m1` against each.
  Targets (from §7 of the plan): `≥ 1.7×` baseline at 2 slots,
  `≥ 3.0×` at 4 slots. Supports `BENCH_SKIP_SERVER=1` for benching a
  pre-started process while the HTTP dispatcher rewrite is pending.
- **`reqwest` dep** (optional, gated on the `server` feature,
  `default-features = false` + `rustls-tls` + `json` — no OpenSSL
  headers needed).
- **`ARCHITECTURE.md`** — new section "M1 Multi-slot + Continuous
  Batching (Apr 2026)" (~330 lines) with ASCII data flow,
  per-seq/per-slot ownership table, feature-flag semantics, module
  responsibility index, slot state machine, batch-construction
  invariants, MTP gating policy, and J1-J8 status table.

#### Feature flag

New environment variable **`CHIMERE_MULTISLOT`**: unset / `1`
selects the legacy path (production default, behaviour unchanged),
`>= 2` activates the admission queue + scheduler.
Values `>= 9` are clamped to 8 with a warning.
**`CHIMERE_ADMISSION_QUEUE`** overrides the 64-slot admission mpsc
bound.

#### Known status (J8)

| Layer                                   | Status   |
|-----------------------------------------|:--------:|
| Scheduler types + admission             |  OK      |
| `forward_multi_seq` FFI                 |  OK      |
| chunked prefill + concurrent gen        |  OK      |
| per-slot sampler + engram isolation     |  OK      |
| HTTP dispatcher → `forward_multi_seq`   |  WIP     |
| stop / cancel / disconnect cleanup      |  PENDING |
| stress (4c, 8 backlog, 1000-req leak)   |  PENDING |
| bench harness                           |  OK      |

The HTTP handler still routes through the legacy
`state.model.blocking_lock()` closure; wiring it through the
scheduler's step loop is the remaining multi-day task. Until that
lands, `CHIMERE_MULTISLOT >= 2` is a no-op for request throughput
(the primitives and per-slot isolation are already proven by the
J3 / J4 / J5 smokes).

#### Commits

J1 — `25bc027` scaffolding, no behaviour change.
J2 — `68f8ded` + `97f8b3c` + `dd36fcf` (admission + dispatcher +
smoke).
J3 — `7c782aa` multi-seq FFI + `j3-smoke`.
J4 — `a4547ac` infra + `8a73513` smoke + `8866327` grammar shim fix.
J5 — `fae89f4` J5a per-slot sampler + `5ab760d` J5b engram bias +
`bd58ecf` smoke + `7dd4a31` libcommon unblock + `8fc079a` J5 PASS.
J8 — bench harness + docs (this section).

#### Out of scope for M1

- Agent scheduler (`agent_scheduler.rs`) coexists as a single-slot
  context-switch path; multi-slot sticky routing per user is deferred
  to J10+.
- DART speculative decoding — disabled on multi-slot for the first
  rollout. Single-slot path unchanged.
- TurboQuant KV (Hadamard Q4_0) per seq_id via paged KV is already
  per-seq at the libllama level; no chimere change required for the
  initial rollout.

## [0.2.0] - 2026-04-07

### Added

- **Multi-architecture dispatch (Step 7).** A new `AppStateModel` enum
  (`server.rs:263`) wraps either `Qwen35Model` or `GenericModel`, and the
  `chimere-server` binary now peeks at `general.architecture` in the GGUF
  metadata to pick the right loader (`bin/chimere-server.rs:56 detect_arch`,
  `bin/chimere-server.rs:185 match arch`). The Qwen3.5 hot path is byte-for-byte
  unchanged. Implemented as 10 atomic commits tagged
  `[step7-1/10]` … `[step7-10/10]` (commits `892dbd5` … `a0f2bc7`,
  tag `step7-complete`).
- **`GenericModel`** (`src/generic_model.rs`, 250 LoC), a libllama-only
  `ChimereModel` impl for any architecture supported by the backend
  (Mamba-1, Mamba-2, Nemotron-H MoE, future archs). Forwarding goes through
  `LlamaForward` only — no MTP, no DART, no Engram, no cudarc, no
  block-diffusion, no entropy routing.
- **End-to-end runtime support for NVIDIA Nemotron-3-Nano-30B-A3B**
  (`nemotron_h_moe`, hybrid Mamba-2 + GQA + 128 experts top-6) via the
  Phase 3.x backport landed in `ik_llama.cpp`
  ([PR #1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)). Validated
  on `Q4_0` and `UD-IQ3_XXS` quants.
- **`bin/test-nemotron`** (`src/bin/test-nemotron.rs`, 91 LoC), a standalone
  smoke binary that exercises `chimere_deltanet::llama_backend::from_env` →
  `tokenize` → `forward_prefill` → greedy decode loop, no HTTP, no
  `Qwen35Model`. Used to bisect backend issues without the rest of the server.
  Measured ~45 tok/s on RTX 5060 Ti, sm_120, NCMOE=30, ctx 2048.
- **`InferenceState::Generic` variant** (`chimere_model.rs:127`), a phantom
  marker for libllama-backed models whose state lives entirely inside the FFI
  context. The enum stays `#[non_exhaustive]`.
- **`forward_token_generic` / `forward_prefill_generic`** convenience helpers
  (`chimere_model.rs:332-348`) so call sites that work with `Generic` state
  do not have to construct `std::marker::PhantomData` themselves.
- **`generate_text_generic`** (`generate.rs:207`), the libllama-only sibling
  of `generate_text`. No engram, no MTP, no `</think>` detection, single-token
  argmax / sampling, returns the same `GenerateResult` shape so the HTTP layer
  can dispatch on the variant transparently.
- **`generate_with_mtp_generic`** (`mtp_scheduler.rs:1241`), the inner
  generation loop for the Generic path. Reads `CHIMERE_GENERIC_EOS`
  (comma-separated u32 list, default `[2]`) for stop tokens.
- **`llama_forward()` immutable trait accessor** (`chimere_model.rs:254`),
  added in commit `cb0ad91` so callers that need read-only access (future
  tokenizer fallback, debug introspection) do not need a `RefMut`.
- **New environment variables**:
  - `CHIMERE_FORCE_QWEN35` — when set, the binary refuses to start unless the
    loaded GGUF is `qwen35moe`. Belt-and-braces guard for the production slot
    (`bin/chimere-server.rs:165`).
  - `CHIMERE_GENERIC_EOS` — comma-separated u32 list of stop tokens for the
    Generic path (`mtp_scheduler.rs:1256`). Default `2`.
  - `CHIMERE_ENGRAM_DIR` — formalised as the recommended way to load the
    multi-table engram overlay (`engram_lookup.rs:608`). The systemd unit now
    sets it explicitly.

### Changed

- `AppState::model` field type: `Mutex<Qwen35Model>` → `Mutex<AppStateModel>`
  (`server.rs:312`). The Qwen3.5 hot path goes through
  `match &*model { AppStateModel::Qwen35(qwen) => … }` which compiles to the
  same machine code as the previous direct deref.
- `chimere_model::InferenceState::as_gdn_mut()` is now exhaustive over both
  variants and returns a clear error message when called on a `Generic`
  state, instead of relying on `#[non_exhaustive]` warnings
  (`chimere_model.rs:133`).
- **Engram loading is now skipped on non-Qwen architectures**
  (`mtp_scheduler.rs:648`, commit `9b79e8c [step7-5/10]`). Engram tables are
  keyed on the Qwen3.5 vocab (size 248 320). Loading them on a model with a
  different tokenizer (e.g. Nemotron-H, vocab 131 072) would either crash on
  out-of-range indices or produce garbage biases. The skip is silent if no
  engram env var is set, and emits a single explanatory line if one is.
- The fast-sampler `last_packed_logprobs` slot is now drained on every
  `forward_token` call regardless of the `logprobs` request flag, to avoid
  stale data leaking across requests on the Generic path (`server.rs:996-1001`).

### Backend (`ik_llama.cpp` fork)

- Pinned to head of branch
  [`mamba2-nemotron-h-backport`](https://github.com/AIdevsmartdata/ik_llama.cpp/tree/mamba2-nemotron-h-backport),
  the head of [PR #1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)
  (12 commits, validated end-to-end). Highlights:
  - **Custom `ggml_ssm_scan` op** + CUDA kernel
    (`ggml-cuda/ssm-scan.cu`) ported from upstream `ggml-org/llama.cpp`
    (commit `3bafe93`, "Phase 3.2").
  - **`build_mamba2_layer` + `build_nemotron_h_moe`** wired into
    `build_context` (commit `af9a12e`, "Phase 3.3").
  - **`llm_build_ffn` parallel-gate guard** (commit `ecf2842`): the previous
    code unconditionally folded `relu²(up) * up` for `LLM_FFN_PAR` MLPs even
    when `gate == nullptr` (Nemotron-H shared expert), producing
    `relu²(up) * up` instead of `relu²(up)`. The fix guards the multiply on
    `gate != nullptr`. This bug ate the better part of a debugging session.
  - **`use_qnext_state_layout` hparam flag** (commit `61f7996`) makes
    Mamba-2 / Nemotron-H first-class instead of bolted-on.
  - **Defensive SSM bounds** (commit `d88ee7a`) for the hybrid load path.

### Known limitations (Step 7)

- **Generic-arch path is single-agent only.** Multi-agent context switching
  via `agent_scheduler` and `llama_state_seq_*` is reserved to the Qwen3.5
  path. The `req.user` field is silently ignored on Generic
  (`server.rs:698-704`). Step 7.5 will plumb the agent scheduler through the
  trait and depends on the backend's hybrid state save/restore being fixed
  (PR #1593 caveat #2).
- **Generic SSE streaming is non-streaming under the hood.** The streaming
  endpoint runs `generate_text_generic` to completion, then emits one
  `StreamMsg::Token` with the whole text and a `StreamMsg::Done`
  (`server.rs:1030-1061`). Token-by-token streaming for libllama paths is
  Step 7.5.
- **Chat template on Generic is the Qwen3.5 hand-rolled one.** For non-Qwen
  archs the prompt is best-effort (the SSE handler decodes the already-encoded
  tokens back to a string then re-encodes through the requested model's
  tokenizer). Step 7.5 will use `LlamaForward::apply_chat_template` and read
  the template from GGUF metadata.
- **Engram tables are tokenizer-specific** (Qwen3.5 vocab 248 320). They are
  silently disabled on Generic archs (see Changed).
- **Phase 3.3 hardcodes `n_seqs = 1`** in the backend for Mamba-2 mixers.
  Multi-sequence parallel decoding for Nemotron-H is reserved to a future
  Phase 3.5.
- **`test-nemotron` requires `CHIMERE_TOKENIZER`** to point at an HF
  `tokenizer.json`. The libllama tokenizer fallback is wired through the
  trait (`llama_forward.tokenize`) but not yet through the HTTP path or the
  smoke binary.

### Validated runtime

- **Qwen3.5-35B-A3B Chimere v3 RAMP**, `chimere-v3-ramp.gguf` (15.65 GB):
  PR #1593 + Step 7 binary, `~/Bureau/STEP7_BASELINE.sh` regression script
  passes 4/4 against the PF-3 baselines. 80 tok/s gen, ~789 tok/s prefill,
  80 ms TTFT, 64K ctx, 15.3 GB VRAM, sm_120 RTX 5060 Ti, NCMOE=3,
  KV q8_0 / q4_0.
- **Nemotron-3-Nano-30B-A3B Q4_0**, ~18.2 GB GGUF: HTTP `curl` with
  `What is the capital of France?` returns `Paris`. ~45 tok/s gen via the
  `test-nemotron` smoke binary, NCMOE=30, ctx 2048.

### Migration guide

- **Production systemd unit** (`chimere-server.service`): make sure
  `Environment=CHIMERE_ENGRAM_DIR=/home/USER/.openclaw/data/engram` is set
  explicitly. Previously this was implicit and silently disabled engram in
  production — that is a behaviour fix, not a regression. The fix is in the
  service unit, not in the binary.
- **For the Nemotron path**, set:
  - `CHIMERE_MODEL=/path/to/Nemotron-3-Nano-30B-A3B-Q4_0.gguf`
  - `CHIMERE_TOKENIZER=/path/to/Nemotron-3-Nano-30B-A3B/tokenizer.json`
  - `CHIMERE_NCMOE=30` (or whatever your VRAM budget allows)
  - `CHIMERE_KV_MAX_SEQ=2048` (or whatever fits — Nemotron-H KV is heavier
    than Qwen3.5 GDN)
  - `CHIMERE_LLAMA_BACKEND=1` (implicit on the Generic path; harmless to set
    explicitly)
  - Do NOT set `CHIMERE_FORCE_QWEN35=1` — that gate will refuse to load.

## [0.1.0] - 2026-03

Initial public release. `chimere-server` only loads Qwen3.5-35B-A3B with the
full production stack (MTP infrastructure, MRoPE, block-diffusion scheduler,
entropy router, agent scheduler, engram-aware sampling, custom CUDA graphs,
DRY + min-p + top-p + top-k C++ fast sampler, K-cache Hadamard rotation,
fused MoE up/gate). Backend is the `ik_llama.cpp` fork built for sm_120.
