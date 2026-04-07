# Step 7 — Multi-architecture dispatch in `chimere-server`

This document describes the architecture introduced by the Step 7 commits
(`892dbd5` … `a0f2bc7`, tag `step7-complete`, 10 atomic commits), which let
`chimere-server` load any GGUF whose `general.architecture` is supported by
the `ik_llama.cpp` backend, in addition to the production Qwen3.5-35B-A3B
path.

It is the design rationale and the operational reference. The user-facing
view is in [`README.md`](../../README.md) and
[`CHANGELOG.md`](../../CHANGELOG.md). The change log per file lives in
`~/Bureau/STEP7_DIFFS_PER_FILE.md` (not committed) and the regression script in
`~/Bureau/STEP7_BASELINE.sh`.

---

## 1. Why

Before Step 7, `chimere-server` was hard-wired to `Qwen35Model`:

```rust
// pre-Step-7
pub struct AppState {
    pub model: Mutex<Qwen35Model>,
    ...
}
```

`Qwen35Model::cudarc_shell` and the rest of `qwen35_model/mod.rs` (3460 LoC)
encode the Qwen3.5-35B-A3B GDN config schema directly: 48 GDN layers + 16
attention layers, MoE 256 experts top-8, MRoPE multi-section RoPE, an MTP head
with `n_nextn_layer = 1`, the engram tokenizer (Qwen3.5 vocab 248 320), the
hand-rolled chat template, the cudarc raw weights path, the entropy router,
the block diffusion scheduler. Loading any other GGUF through that struct
either crashes immediately on the schema mismatch or — worse — silently
produces garbage on the engram-biased fast sampler.

The Mamba-2 + Nemotron-H backport landed in
[ikawrakow/ik_llama.cpp#1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)
unblocks an entire class of architectures at the backend level (Mamba-1,
Mamba-2 pure, Nemotron-H MoE, and by extension Granite 4.0 H-Tiny / H-Small,
Falcon-H1, Bamba-9B, Codestral-Mamba, AI21-Jamba-Reasoning-3B, …) — but every
one of those calls into `libllama.so` directly, NOT into the Qwen3.5-specific
shell. We needed a code path in `chimere-server` that does not depend on
`Qwen35Model::cudarc_shell` and does not assume a Qwen3.5 vocab.

Step 7 is that code path.

---

## 2. Approach: closed enum, not trait object

The dispatch type is a closed enum, not `Box<dyn ChimereModel>`:

```rust
// chimere-server/src/server.rs:263
pub enum AppStateModel {
    /// Qwen3.5-35B-A3B — full production stack (MTP, cudarc, block diffusion,
    /// entropy routing, Candle path, Engram-aware sampling).
    Qwen35(Qwen35Model),
    /// libllama-native architectures (Mamba-1 / Mamba-2 / Nemotron-H MoE / ...).
    /// No MTP, no DART, no block diffusion — forward via LlamaForward only.
    Generic(GenericModel),
}

impl AppStateModel {
    pub fn as_trait(&self) -> &dyn ChimereModel {
        match self {
            AppStateModel::Qwen35(m) => m,
            AppStateModel::Generic(m) => m,
        }
    }

    pub fn arch(&self) -> ModelArch {
        match self {
            AppStateModel::Qwen35(m) => ChimereModel::arch(m),
            AppStateModel::Generic(m) => ChimereModel::arch(m),
        }
    }
}
```

### Why an enum and not `Box<dyn ChimereModel>`

1. **Inherent methods preserved.** `Qwen35Model` exposes a long tail of
   inherent methods that the prod path needs (`reset_llama_state`,
   `reset_cudarc_state`, `init_llama_forward`, `init_cudarc_forward`,
   `cudarc_shell`, `from_gguf`, …). These are NOT on the `ChimereModel` trait
   on purpose: most of them are Qwen3.5-specific and would pollute the trait
   surface. An enum lets the call site `match` and access them on the
   `Qwen35` arm without going through the trait at all. With
   `Box<dyn ChimereModel>` we would have had to either move every Qwen3.5
   method onto the trait (bad) or downcast at every call site (worse).

2. **Capability flags become compile-time exhaustive.** With the enum, any
   future variant added to `AppStateModel` triggers a missing-arm error in
   every `match` site, which is exactly what we want for an inference engine
   where the wrong code path silently produces wrong tokens.

3. **No vtable cost on the hot path.** The Qwen3.5 inference loop on a
   100-tokens-per-second budget calls back into the model per token. The
   enum-then-match compiles down to the same machine code as a direct deref;
   the trait object would add an indirect call per forward.

4. **Adding a new arch is one variant + one constructor**, not "implement
   N trait methods + register in a global map".

The trait `ChimereModel` (`chimere_model.rs:164`) still exists and is used by
generation helpers in `generate.rs` and `mtp_scheduler.rs` that take
`&dyn ChimereModel` — that is the only code path where a trait object is
useful, because those helpers do not care which concrete type they hold.

---

## 3. Flow

```
main() in src/bin/chimere-server.rs
  │
  │  std::env::var("CHIMERE_MODEL") → model_path
  │
  ▼
detect_arch(&model_path)              // bin/chimere-server.rs:56
  │
  │  GgufFile::open(model_path)
  │     .get_metadata("general.architecture")
  │  match arch_str.to_ascii_lowercase() { ... }
  │
  ▼
Result<ModelArch, String>             // chimere_model.rs:54
  │
  │  Qwen35A3B  → "qwen35moe", "qwen35", "qwen3.5", "qwen3.5moe", "qwen3next"
  │  Mamba1     → "mamba", "mamba1"
  │  Mamba2     → "mamba2", "mamba_2"
  │  NemotronHMoe → "nemotron_h_moe", "nemotronh", "nemotron-h", "nemotron_h"
  │
  ▼
Optional belt-and-braces guard       // bin/chimere-server.rs:165
  │
  │  if CHIMERE_FORCE_QWEN35 set and arch != Qwen35A3B → exit(1)
  │
  ▼
match arch                            // bin/chimere-server.rs:185
  │
  ├── Qwen35A3B
  │     ├── if CHIMERE_LLAMA_BACKEND  → Qwen35Model::cudarc_shell + init_llama_forward
  │     ├── if CHIMERE_CUDARC_FORWARD → Qwen35Model::cudarc_shell + init_cudarc_forward
  │     └── else                       → Qwen35Model::from_gguf (Candle path)
  │     Wraps in AppStateModel::Qwen35
  │
  └── Mamba1 | Mamba2 | NemotronHMoe
        │
        │  Require CHIMERE_TOKENIZER (HF tokenizer.json)   ← Step 7.5 will lift
        │  Warn if CHIMERE_LLAMA_BACKEND set (implicit)
        │  Warn if CHIMERE_CUDARC_FORWARD set (ignored)
        │
        └── GenericModel::from_env(arch)
              ├── llama_backend::from_env() → LlamaForward
              ├── cache n_vocab, n_layer, chat_template
              └── wrap in RefCell<Option<LlamaForward>>
              Wraps in AppStateModel::Generic
  │
  ▼
AppState { model: Mutex<AppStateModel>, ... }
  │
  ▼
build_router(state) → axum
  │
  ▼  POST /v1/chat/completions
chat_completions_handler  (server.rs:1208)
  │
  ▼
{ chat_completions_non_stream | chat_completions_stream }
  │
  │  builds prompt via messages_to_prompt (Qwen3.5 hand-rolled template)
  │  builds SamplingParams (req fields + hardcoded min_p, dry_*)
  │
  ▼
run_inference (server.rs:594) for non-stream
or inline match in chat_completions_stream (server.rs:938) for SSE
  │
  ▼
match &*model_guard            // both call sites
  │
  ├── AppStateModel::Qwen35(qwen)  → existing path:
  │     reset_llama_state | reset_cudarc_state | agent context switch
  │     GdnRecurrentState::new
  │     generate_text(qwen, ...)            // generate.rs
  │     OR
  │     generate_with_mtp_streaming(qwen as &dyn ChimereModel, ...)   // SSE
  │
  └── AppStateModel::Generic(gm) → new path:
        if user.is_some(): warn (Step 7 ignores agent on Generic)
        gm.reset_for_new_request()
        generate_text_generic(gm, ...)         // generate.rs:207
        For SSE: re-decode prompt_ids → call generate_text_generic
        → emit one StreamMsg::Token + StreamMsg::Done
```

Send across `await`: see §5.

---

## 4. Schema

### `ChimereModel` trait (closed surface)

```text
ChimereModel  (chimere_model.rs:164)
├── arch() -> ModelArch                                      [required]
├── num_layers() -> usize                                    [required]
├── vocab_size() -> usize                                    [required]
│
├── supports_mtp() -> bool                                   [default false]
├── supports_block_diffusion() -> bool                       [default false]
├── supports_dart() -> bool                                  [default false]
├── supports_entropy_routing() -> bool                       [default false]
│
├── forward_token(token, &mut InferenceState) -> ForwardOutput   [required]
├── forward_prefill(tokens, &mut InferenceState) -> ForwardOutput [required]
│
├── reset_for_new_request()                                  [default no-op]
│
├── llama_forward_active() -> bool                           [default false]
├── llama_forward()    -> Option<Ref<...>>                   [default None]
├── llama_forward_mut() -> Option<RefMut<...>>               [default None]
├── llama_set_logit_bias(token_id, bias)                     [default no-op]
├── llama_set_engram_bias(predictions)                       [default no-op]
├── llama_clear_engram_bias()                                [default no-op]
└── take_last_packed_logprobs() -> Option<Vec<f32>>          [default None]
```

### Implementations

```text
impl ChimereModel for Qwen35Model           // qwen35_model/mod.rs:2607
  ├── arch = ModelArch::Qwen35A3B
  ├── supports_mtp / supports_block_diffusion /
  │   supports_dart / supports_entropy_routing = true
  ├── forward_token / forward_prefill: full Qwen3.5 stack
  │     (Candle, cudarc, libllama backends, MTP head, engram-aware sampling)
  ├── llama_forward / llama_forward_mut: yes (libllama backend mode)
  └── delegates llama_set_logit_bias / set_engram_bias / take_last_packed_logprobs

impl ChimereModel for GenericModel          // generic_model.rs:126
  ├── arch = ModelArch::{Mamba1 | Mamba2 | NemotronHMoe}
  ├── ALL supports_* return false
  ├── forward_token / forward_prefill: LlamaForward FFI only (state ignored)
  ├── reset_for_new_request: llama.reset()
  ├── llama_forward / llama_forward_mut: yes (single LlamaForward in RefCell)
  └── delegates llama_set_logit_bias / set_engram_bias  / take_last_packed_logprobs
```

### `InferenceState` enum

```text
InferenceState<'a>           // chimere_model.rs:117 (#[non_exhaustive])
├── Gdn(&'a mut GdnRecurrentState)            // Qwen3.5
└── Generic(PhantomData<&'a ()>)              // libllama-owned state
```

The `'a` lifetime on `Generic` is purely structural (the variant carries no
data) — keeping it lets the rest of the call chain be generic over `'a`
without writing two trait method signatures.

The two convenience helpers:

```rust
// chimere_model.rs:332
pub fn forward_token_generic(model: &dyn ChimereModel, token: u32)
    -> Result<ForwardOutput>

// chimere_model.rs:342
pub fn forward_prefill_generic(model: &dyn ChimereModel, tokens: &[u32])
    -> Result<ForwardOutput>
```

are mirrors of the existing `*_gdn` helpers, used by `generate_text_generic`.

---

## 5. The `Send` constraint and the agent-id pattern

`Qwen35Model` contains `RefCell` fields (it has to: `LlamaForward` mutates the
libllama context pointer, and the Candle path mutates internal scratch
buffers). Therefore:

```text
Qwen35Model: !Sync           ← because of RefCell
&Qwen35Model: Send  ⇔  Qwen35Model: Sync   ← false, so &Qwen35Model: !Send
```

In `axum`, request handlers must return `Future<Output = Response>` where the
future is `Send` (it has to cross thread boundaries on the runtime's worker
pool). If we held a `&Qwen35Model` borrowed from the `Mutex` guard while
awaiting *anything*, the future would capture the `&Qwen35Model` across the
await point and the compiler would refuse to compile (`future is not Send`).

Pre-Step-7 this was managed by carefully ordering awaits inside
`run_inference`. Step 7 makes it explicit and documents it
(`server.rs:608-637`):

```rust
// STEP 7: resolve agent_id BEFORE locking the model.
// The match on `&*model` later gives us `&Qwen35Model` which is NOT
// Send. To keep the inference future Send (required by axum's Handler
// bound), no `.await` may run while we hold the typed variant ref. So
// we do all the multi-agent bookkeeping awaits first, then lock the
// model and dispatch sync.
let agent_id_opt: Option<usize> = if let Some(user_name) = user {
    let id = {
        let mut map = state.user_agent_map.lock().await;       // <-- AWAIT
        if let Some(&id) = map.get(user_name) {
            id
        } else {
            let mut sched = state.agent_scheduler.lock().await; // <-- AWAIT
            let id = sched.register_agent(user_name);
            map.insert(user_name.to_string(), id);
            id
        }
    };
    Some(id)
} else {
    None
};

// Lock the model for the duration of inference. NO MORE AWAITS BEYOND
// THIS POINT (until the lock is dropped at the end of the function).
let model = state.model.lock().await;
let gen = match &*model {
    AppStateModel::Qwen35(qwen) => { ... try_lock the scheduler with the
                                         pre-resolved agent_id ... }
    AppStateModel::Generic(gm) => { ... ignore agent_id, single-agent ... }
};
```

The inner agent context switch (`switch_to`) uses `try_lock` rather than
`lock().await`, which is safe here because the only contention point is the
initial registration that already happened above.

`GenericModel` is `Send` by virtue of an explicit `unsafe impl Send for
GenericModel {}` in `generic_model.rs:80`, mirroring the same pattern that
`Qwen35Model` uses. The `RefCell` is still `!Sync`, but the outer `Mutex` in
`AppState` serialises access, so `Send` (cross-thread move) is fine.

---

## 6. Tradeoffs (Step 7)

These are the bills we accepted to ship a working multi-arch path in one
shot. They are individually small and they each have a known fix.

| Tradeoff | Where | Why | Step 7.5 fix |
|---|---|---|---|
| Single-agent only on Generic. `req.user` is silently ignored. | `server.rs:698-704` | Backend `llama_state_seq_*` walks legacy K-cache layout for hybrid Nemotron-H (PR #1593 caveat #2). | Lift the backend caveat, then thread `agent_scheduler` through the trait. |
| Generic SSE streaming is "one Token + Done" rather than per-token. | `server.rs:1030-1061` | `generate_text_generic` is a sync function that returns the complete `GenerateResult`. The streaming wrapper would need to be inverted into a callback API. | Build `generate_with_mtp_streaming_generic` mirroring the Qwen3.5 callback path. |
| Generic chat template = the Qwen3.5 hand-rolled one. | `server.rs:327` `messages_to_prompt` is shared. | The SSE handler decodes already-encoded prompt_ids back to a string then re-encodes via the model's tokenizer. Best-effort but lossy. | Use `LlamaForward::apply_chat_template` and read the template from GGUF metadata via `LlamaForward::chat_template`. |
| Engram tables silently disabled on Generic. | `mtp_scheduler.rs:648` `model.arch() == Qwen35A3B`. | Engram tables are keyed on the Qwen3.5 vocab (248 320). Loading them on Nemotron-H (vocab 131 072) would either crash on out-of-range indices or produce garbage biases. | Per-architecture engram namespacing (one set of tables per tokenizer), or a tokenizer-aware lookup. |
| Tokenizer is HF-only. Generic path requires `CHIMERE_TOKENIZER`. | `bin/chimere-server.rs:241` | The Qwen3.5 path uses the `tokenizers` crate; we did not want to plumb the FFI tokenizer fallback through `messages_to_prompt` and the SSE decoder in the same Step. | `LlamaForward::tokenize` / `detokenize` are already exposed; wire them as a fallback when `CHIMERE_TOKENIZER` is unset. |
| `n_seqs == 1` hardcoded in the backend for Mamba-2. | ik_llama backport `build_mamba2_layer`. | Phase 3.3 of PR #1593 was scoped to "make it run". | Phase 3.5: lift the constant. |

---

## 7. Step 7.5 roadmap

In rough priority order:

1. **FFI tokenizer fallback for Generic.** Make `CHIMERE_TOKENIZER` optional
   when `arch != Qwen35A3B`. Plumb `LlamaForward::tokenize` into
   `generate_text_generic` and into the SSE re-encode path.
2. **Per-token Generic SSE streaming.** Lift `generate_text_generic` into a
   callback-driven generation loop and reuse the existing
   `tx.blocking_send(StreamMsg::Token)` machinery.
3. **GGUF chat template for Generic.** Use `LlamaForward::chat_template(None)`
   (already cached on `GenericModel.chat_template`) instead of
   `messages_to_prompt`. Falls back to Qwen3.5 template if absent.
4. **Multi-agent on Generic.** Depends on the backend hybrid state
   save/restore being fixed (PR #1593 caveat #2). Needs the
   `agent_scheduler.switch_to(agent_id, llama)` call to work for
   `nemotron_h_moe` GGUFs.
5. **Engram per-tokenizer namespacing.** A new directory layout under
   `CHIMERE_ENGRAM_DIR/<arch>/...` plus a tokenizer hash check on load. Stops
   silent disablement on non-Qwen archs.

---

## 8. Code pointers

All paths are relative to `chimere-server/src/` unless noted.

| File | Lines (HEAD as of 2026-04-07) | Step 7 surface |
|---|---|---|
| `chimere_model.rs` | 348 | `InferenceState::Generic` (l. 127), exhaustive `as_gdn_mut` (l. 133), `llama_forward()` immutable accessor (l. 254), `forward_token_generic` / `forward_prefill_generic` helpers (l. 332-348). |
| `generic_model.rs` | 250 | Whole file is new. `GenericModel::from_env` (l. 89), `impl ChimereModel` (l. 126), `unsafe impl Send` (l. 80). |
| `qwen35_model/mod.rs` | 3460 | `impl crate::chimere_model::ChimereModel for Qwen35Model` (l. 2607). Unchanged behavior. |
| `generate.rs` | 946 | `generate_text_generic` (l. 207). |
| `mtp_scheduler.rs` | 1586 | Engram skip for non-Qwen (l. 648, l. 966), `generate_with_mtp_generic` (l. 1241), `CHIMERE_GENERIC_EOS` (l. 1256). |
| `server.rs` | 1240 | `AppStateModel` enum (l. 263), `AppStateModel::as_trait` / `arch` (l. 272-288), `AppState.model: Mutex<AppStateModel>` (l. 312), `run_inference` agent-id-before-lock pattern (l. 594-715), SSE dispatch (l. 938, Generic arm at l. 1030-1061). |
| `bin/chimere-server.rs` | 332 | `detect_arch` (l. 56), `CHIMERE_FORCE_QWEN35` guard (l. 165), `match arch` loader dispatch (l. 185-289). |
| `bin/test-nemotron.rs` | 91 | Whole file is new. Standalone smoke binary, no `Qwen35Model`, no HTTP. |

Tag `step7-complete` on branch `main` of `AIdevsmartdata/chimere`. Commits
`892dbd5` (`feat(model): add Generic InferenceState variant [step7-1/10]`) …
`a0f2bc7` (`chore: build verification post-Step-7 [step7-10/10]`).

---

## Notes for main agent

- The trait doc-comment in `chimere_model.rs:28-37` still says **"No external
  caller uses the trait yet"** and refers to the `Generic` variant as
  "intentionally absent — it lands in Step 7". Both statements are now
  obsolete: `generate_text_generic`, `generate_with_mtp_generic`, and
  `AppStateModel::as_trait` all use the trait, and the `Generic` variant
  exists as of `step7-1/10`. Worth updating in a follow-up commit.
- `bin/chimere-server.rs` still has a stale doc-comment header (lines 5-31)
  citing the old `qwopus-27b-bf16/tokenizer.json` default and not mentioning
  `CHIMERE_FORCE_QWEN35` / Generic-arch behaviour. Worth updating.
- `engram_trained_eval.json` is the only saved engram eval and it is on
  GPT-2 + wikitext-2 with a −13.39 % PPL regression. Until a Qwen3.5-specific
  eval lands, the README and the model card should not claim "+quality" for
  engram. We have softened both in the v0.2.0 deliverables.
- The old README mentioned MTP "+49.5 % acceptance rate" without a matching
  bench. `bench_mtp.rs:104-167` has Benchmarks 2 and 5 hard-coded as
  `SKIPPED — crash in ik_llama MTP graph, KV cache issue for layer 41`. The
  v0.2.0 README reframes MTP as "infrastructure present, gated, fix planned".
