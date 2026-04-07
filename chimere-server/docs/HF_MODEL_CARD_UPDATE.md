# Updated model card sections for `Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF`

This is a draft of the README sections to add or replace on the Hugging Face
model card at
<https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF>.

The main agent will apply this with `hf` CLI (`hf model upload-readme` or
manual edit), not from this repo. **Nothing in this file ships in the
chimere-server binary.**

## Summary of changes

1. Add a "Compatible runtimes" section listing chimere-server (the official
   Rust runtime) alongside stock `llama-server` / `ik_llama-server`.
2. Replace the existing Quick Start (which uses stock `llama-server`) with a
   chimere-server example that exercises the prod path (Engram +
   `CHIMERE_LLAMA_BACKEND` + `CHIMERE_NCMOE`). The stock `llama-server`
   example moves into a "Generic GGUF runtimes" subsection so users with
   other tooling are not left out.
3. Add a "Backend" section pointing at `AIdevsmartdata/chimere` and at PR
   `ikawrakow/ik_llama.cpp#1593` (Mamba-2 + Nemotron-H backport), so
   downstream users understand that the same chimere backend now also runs
   Mamba-2 / Nemotron-H MoE models.
4. Add a "Multi-architecture support" subsection mentioning the Nemotron-H
   path and the broader Mamba-2 ecosystem.
5. Soften / correct three narratives that the inventory pass flagged as
   either unverified or out-of-date:
   - **MTP "+49.5% acceptance"** — gate the number behind the known
     `bench_mtp.rs` skipped benchmarks.
   - **Engram "+quality"** — frame as a domain-knowledge overlay, not as a
     measured quality boost. The only saved eval is on GPT-2 + wikitext-2
     and shows a regression on that out-of-distribution setup.
   - **Performance numbers** — keep the 80 tok/s figure (which is
     reproducible against the prod chimere-server stack) but pin the
     hardware and the env vars used.

---

## Sections to ADD or REPLACE

### Section: Compatible runtimes (NEW, place after Description)

```markdown
## Compatible runtimes

This GGUF can be loaded by any runtime that supports the Qwen3.5-35B-A3B
(`qwen35moe`) architecture. The reference runtime — and the one that exercises
all chimere-specific features (Engram n-gram bias, multi-agent context
switching, the C++ fast sampler with DRY + min-p, K-cache Hadamard rotation,
fused MoE up/gate) — is **chimere-server**.

| Runtime | Engram | Multi-agent | DRY sampler | K-cache Hadamard | Notes |
|---|---|---|---|---|---|
| [chimere-server](https://github.com/AIdevsmartdata/chimere) (Rust, official) | yes | yes | yes (C++ fast path) | yes | Production target. Also runs Mamba-2 / Nemotron-H MoE through the same backend (see PR [ikawrakow/ik_llama.cpp#1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)). |
| [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) `llama-server` | no | no | optional | optional | Same backend that chimere-server links against, just without the Rust HTTP/sampling layer. |
| [`llama.cpp`](https://github.com/ggml-org/llama.cpp) stock `llama-server` | no | no | no | no | Works, but ~20-30 % slower on Qwen3.5 MoE on our hardware (no `iqk` matmul, no fused MoE up/gate). |
```

### Section: Quick Start (REPLACE existing)

The current model card shows:

```sh
llama-server -m chimere-v3-ramp.gguf -ngl 99 --n-cpu-moe 4 -c 32768 \
    --flash-attn on --jinja --port 8081
```

That works on stock `llama-server` but does not exercise any chimere-specific
feature and does not match the production deployment. Replace with:

```markdown
## Quick start (chimere-server, recommended)

```sh
# 1. Backend (one-time): build the ik_llama.cpp fork with sm_120 CUDA + Mamba-2 backport
git clone https://github.com/AIdevsmartdata/ik_llama.cpp.git ~/ik_llama.cpp
cd ~/ik_llama.cpp
git checkout mamba2-nemotron-h-backport
cmake -B build_sm120 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF
cmake --build build_sm120 -j

# 2. Server
git clone https://github.com/AIdevsmartdata/chimere.git
cd chimere/chimere-server
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
  cargo build --release --features server --bin chimere-server

# 3. Model + tokenizer
cd ~/models
hf download Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF chimere-v3-ramp.gguf
hf download Qwen/Qwen3.5-35B-A3B tokenizer.json --local-dir tokenizers/qwen35

# 4. Run
CHIMERE_MODEL=$PWD/chimere-v3-ramp.gguf \
CHIMERE_TOKENIZER=$PWD/tokenizers/qwen35/tokenizer.json \
CHIMERE_LLAMA_BACKEND=1 \
CHIMERE_NCMOE=3 \
CHIMERE_KV_MAX_SEQ=65536 \
CHIMERE_PORT=8081 \
CHIMERE_FORCE_QWEN35=1 \
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
~/chimere/chimere-server/target/release/chimere-server

# 5. Hello world
curl -s http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

### Engram (optional, prod-only)

Chimere ships an n-gram logit bias overlay loaded from binary `.engr` tables.
To enable it, set:

```sh
CHIMERE_ENGRAM_DIR=/path/to/engram_tables   # directory of *.engr files
CHIMERE_ENGRAM_ALPHA=0.5                     # logit bias strength
CHIMERE_ENGRAM_NEST=1                        # NEST adaptive alpha (default on)
```

The engram tables are tokenizer-specific (Qwen3.5 vocab) and used as a
per-domain overlay (kine, code, cyber, general). They are intended as a
domain-knowledge injector, not a quality booster — see the
[chimere repo README](https://github.com/AIdevsmartdata/chimere#performance)
for the honest status of the path.

## Quick start (generic GGUF runtimes)

If you do not need the chimere stack, the GGUF works with any
Qwen3.5-compatible runtime:

```sh
llama-server -m chimere-v3-ramp.gguf -ngl 99 --n-cpu-moe 4 -c 32768 \
    --flash-attn on --jinja --port 8081
```
```

### Section: Backend (NEW, before License)

```markdown
## Backend

The official `chimere-server` runtime links against a customized
[`ik_llama.cpp`](https://github.com/AIdevsmartdata/ik_llama.cpp) fork
(branch `mamba2-nemotron-h-backport`, head of upstream PR
[ikawrakow/ik_llama.cpp#1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)).

Highlights of the chimere-specific layer on top of ik_llama:

- Custom C++ fast sampler exporting `sample_token_fast`, `set_logit_bias`,
  `set_engram_bias`, `clear_engram_bias` and `take_packed_logprobs` —
  avoids a ~993 KB logits copy per token, packs OpenAI-format top-5
  logprobs.
- K-cache Hadamard rotation, fused MoE up/gate, grouped expert routing —
  all enabled by default via `cparams`.
- Multi-agent KV / SSM state save & restore via `llama_state_seq_*`, keyed
  on the OpenAI `user` field. Up to `CHIMERE_MAX_AGENTS` (default 4)
  concurrent personas with their own conversation state.
- An OpenAI-compatible HTTP layer in Rust (axum 0.8), supporting non-
  streaming and SSE streaming, tool calls, `<think>` reasoning extraction
  and `chat_template_kwargs.enable_thinking`.
```

### Section: Multi-architecture support (NEW, after Backend)

```markdown
## Multi-architecture support

The same `chimere-server` runtime is **not** Qwen-only any more. As of
[Step 7](https://github.com/AIdevsmartdata/chimere/blob/main/chimere-server/docs/STEP7_MULTI_ARCH.md)
(April 2026), it dispatches between two code paths based on the GGUF's
`general.architecture` metadata:

- **Qwen3.5-35B-A3B** (`qwen35moe`) — full production stack: MTP, MRoPE,
  Engram, agent scheduler, custom Candle / cudarc / libllama paths. **This
  GGUF.**
- **Mamba-2 / Nemotron-H MoE / Mamba-1 / Mamba-2 hybrids** — libllama-only
  path via `GenericModel`. No MTP, no Engram, single-agent only at Step 7.
  Validated end-to-end on
  `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` (Q4_0 and UD-IQ3_XXS) at
  ~45 tok/s on RTX 5060 Ti, NCMOE=30, ctx 2048, via the bundled
  `test-nemotron` smoke binary.

Models that **should** run via the same Generic path (untested at the
chimere level — your mileage may vary): Granite 4.0 H-Tiny / H-Small /
H-Micro, Falcon-H1 0.5B – 34B, Bamba-9B v1 / v2, `state-spaces/mamba2-*`,
`mistralai/Mamba-Codestral-7B-v0.1`, AI21-Jamba-Reasoning-3B.
```

### Section: Performance (REPLACE the existing benchmark table)

The current model card shows a one-line `~80 tok/s on RTX 5060 Ti` plus an
IFEval / GSM8K / HumanEval / BFCL table. Keep the IFEval / GSM8K / HumanEval
/ BFCL table as-is (those numbers are from the distillation eval, not the
runtime), but reframe the runtime numbers:

```markdown
## Performance (runtime)

Measured against `chimere-server` linked to the chimere fork of `ik_llama.cpp`,
on the following hardware:

- NVIDIA RTX 5060 Ti, 16 GB VRAM, sm_120 (Blackwell)
- Intel i5-14600KF, 32 GB DDR5
- CUDA 12.8, NVIDIA driver 590.48
- Build: `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF`

| NCMOE | Ctx | Gen tok/s | Prefill tok/s | VRAM used | VRAM free | Notes |
|---|---|---|---|---|---|---|
| 3 | 64K | **80** | 789 | 15.3 GB | 560 MB | Production. TTFT 80 ms. KV q8_0/q4_0. |
| 4 | 64K | 77 | — | — | 702 MB | Safe headroom. |
| 4 | 32K | 80 | — | — | 1.2 GB | Maximum VRAM headroom. |
| 2 | 64K | OOM | — | — | — | Not viable on 16 GB. |

Numbers are from `~/Bureau/STEP7_BASELINE.sh` regression run, 4/4 baselines
PASS against pre-Step-7 reference.

### About MTP

This GGUF carries an MTP (multi-token prediction) head — chimere-server
detects it via `n_nextn_layer = 1` and exposes the speculative-decoding
infrastructure (`mtp_scheduler.rs`, `MtpOp` FFI). An early March bench on a
previous build measured **+49.5 % token acceptance rate** for the MTP draft
path; that figure is **not currently reproducible** because
`bench_mtp.rs:104-167` has Benchmarks 2 and 5 hard-coded as `SKIPPED` with
the comment `crash in ik_llama MTP graph, KV cache issue for layer 41`.
Until that fix lands the 80 tok/s figure above is the non-MTP path. We will
re-publish the MTP gain once the bench passes.
```

### Section: Limitations (NEW or REPLACE)

```markdown
## Limitations

- **MTP infrastructure present, gated.** See "About MTP" above. Treat as
  forward-looking, not as a current speed-up.
- **Engram is a domain-knowledge overlay, not a measured quality boost.**
  The only saved engram eval in the chimere repo
  (`benchmarks/engram_trained_eval.json`) was run on GPT-2 + wikitext-2
  and shows a −13.39 % PPL regression on that out-of-distribution setup.
  No Qwen3.5-specific perplexity eval has been published yet. Engram is
  shipped as an optional per-domain n-gram bias (kine, code, cyber,
  general); qualitative use shows specialized vocabulary in responses
  (`drainage bronchique postural`, `EMII`, …) on the kiné domain, but
  there is no quantitative claim attached to it today.
- **Multi-slot concurrent decoding via `ik_llama.cpp` is broken** under
  heavy load (`ik_llama` multi-slot bug, slot 0 contamination of system
  prompts under contention). The `chimere-server` production deployment is
  single-slot. Stock `llama-server` does NOT have this bug if you need
  parallel slots.
- **Tool-calling sampler defaults**: see
  [chimere/chimere-server/src/server.rs](https://github.com/AIdevsmartdata/chimere/blob/main/chimere-server/src/server.rs)
  l. 738-744. `presence_penalty` defaults to `0.0` — a previous default of
  `1.5` killed code generation and long reasoning blocks.
```

---

## Sections to KEEP unchanged

- The distillation eval table (IFEval 100 %, GSM8K 84 %, HumanEval 83 %,
  BFCL 75 %). These are model-quality numbers, not runtime numbers, and the
  inventory did not flag them.
- The license block.
- The "Built on top of …" / acknowledgements block, if it exists.

---

## Notes for main agent

- Double-check the existing model card before applying. The inventory pass
  describes the current Quick Start as "stock `llama-server` …" — verify
  that has not been edited in the meantime (the HF page has 1557 downloads
  and may have been updated).
- The "Compatible runtimes" table claims chimere-server is ~20-30 % faster
  than stock `llama-server` on Qwen3.5 MoE. The MEMORY.md numbers are
  +18 % / +32 % / +19 % depending on quant; "20-30 %" is a defensible
  rounded summary but it is the chimere-aligned ik_llama backend doing the
  work, not the chimere Rust layer per se. Phrase as "the same backend
  that chimere-server links against, just without the Rust HTTP/sampling
  layer" (already done above).
- Do not bump the main HF model name or the file name. This card is for
  the existing `chimere-v3-ramp.gguf` artifact.
- The Nemotron-H mention in "Multi-architecture support" cross-references
  Nemotron-3-Nano on a different HF org (`unsloth/...`). That is correct —
  we did not re-upload the Nemotron GGUFs under `Kevletesteur`. Linking
  out is fine.
- The 45 tok/s Nemotron number is from the `test-nemotron` smoke binary
  (`bin/test-nemotron.rs`). It is reproducible. The README and CHANGELOG
  cite it under the same conditions (NCMOE=30, ctx 2048).
