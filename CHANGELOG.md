# Changelog

All notable changes to `chimere-server` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project loosely
follows semantic versioning at the binary level.

## [Unreleased]

Nothing yet.

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
