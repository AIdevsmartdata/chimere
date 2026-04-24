# Chimère

**The Rust inference runtime for local-first hybrid SSM + MoE models.**
**35B parameters. 94 tokens/s. 16 GB consumer GPU. Single binary.**

A Rust inference server for hybrid State-Space + MoE language models, built on a
customised `ik_llama.cpp` fork. Production target: Qwen3.5-35B-A3B (Gated DeltaNet
+ MoE) at ~94 tok/s on a single 16 GB consumer GPU. Also runs Mamba-2 and
Nemotron-H MoE architectures end-to-end via a backend backport landed in our
open upstream PR ([ikawrakow/ik_llama.cpp#1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/Rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![Backend: ik_llama.cpp](https://img.shields.io/badge/Backend-ik__llama.cpp-green.svg)](https://github.com/AIdevsmartdata/ik_llama.cpp)
[![CUDA sm_120](https://img.shields.io/badge/CUDA-sm__120-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Why Chimère

- **One 16 GB consumer GPU.** No H100, no cloud, no multi-GPU tricks. RTX 5060 Ti target.
- **Rust end-to-end.** Single `chimere-server` binary, axum 0.8, OpenAI-compatible API.
- **Engram n-gram logit bias.** A per-token personalisation overlay we believe is unique — see [Features](#features).
- **Multi-architecture dispatch.** One trait, two impls: Qwen3.5 full stack *and* the Mamba-2 / Nemotron-H family.
- **TurboQuant-flavoured K-cache.** Hadamard-rotated keys, Q8_0/Q4_0 KV, free ~8 % tok/s.

## The Chimère family

| Repo | Role | Link |
|---|---|---|
| **chimere** (this repo) | Rust inference runtime | you are here |
| **chimere-odo** | Python orchestrator: intent routing, deep-search, quality gate | <https://github.com/AIdevsmartdata/chimere-odo> |
| **chimere-studio** | Tauri 2 desktop / mobile UI | <https://github.com/AIdevsmartdata/chimere-studio> |
| **ramp-quant** | RAMP / TQ3 mixed-precision quant pipeline | <https://github.com/AIdevsmartdata/ramp-quant> |
| **ik_llama.cpp** fork | Backend C++/CUDA kernels, Mamba-2 + Nemotron-H backport | <https://github.com/AIdevsmartdata/ik_llama.cpp> |
| **GGUF model weights** | Distilled, RAMP, IQ3_S custom mixes | <https://huggingface.co/Kevletesteur> |

> TL;DR benchmark: **94 tok/s** on Qwen3.5-35B-A3B (TQ3 custom mix, 32 K ctx, 13 GB VRAM used) — see [benchmarks/](benchmarks/) and the [public benchmark note](https://github.com/AIdevsmartdata/chimere/blob/main/benchmarks/benchmark-qwen35-2026-03-07.md).

---

## Highlights

- **Multi-architecture dispatch (Step 7, Apr 2026).** A closed `AppStateModel`
  enum routes incoming requests to either the full Qwen3.5 production stack or a
  generic libllama path. Adding a new architecture is one new enum variant plus
  one loader.
- **Qwen3.5-35B-A3B Chimere v3 RAMP** as the prod target: 48 GDN + 16 attention
  layers, 256 experts top-8, 1 MTP head, custom RAMP IQK quantization mix,
  ~80 tok/s gen on RTX 5060 Ti, 64K context, 80 ms TTFT.
- **Mamba-2 / Nemotron-H MoE** runtime support via libllama FFI, on top of our
  Phase 3.x backport to `ik_llama.cpp`
  ([PR #1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)). Validated
  on `Nemotron-3-Nano-30B-A3B` Q4_0 and UD-IQ3_XXS.
- **Engram n-gram logit bias.** Four prebuilt domain tables (kine 19.7 MB, code,
  cyber, general), FNV-1a hashed with a tier-0 Cuckoo filter, mmap zero-copy,
  loaded as a per-domain overlay. Active on the Qwen3.5 path; tokenizer-bound,
  intentionally disabled on non-Qwen architectures.
- **Native sm_120 / Blackwell CUDA** through our `ik_llama.cpp` fork built with
  `-DCMAKE_CUDA_ARCHITECTURES=120` and CUDA 12.8.
- **OpenAI-compatible HTTP API**: `POST /v1/chat/completions` (non-streaming +
  SSE) and `GET /health`. Tool calls (Qwen3.5 `<tool_call>` syntax),
  `<think>` reasoning extraction, OpenAI top-5 logprobs, multi-agent context
  switching keyed on the `user` field.
- **Custom C++ fast sampler** (DRY + min-p + top-p + top-k + presence penalty),
  Hadamard-rotated K-cache, fused MoE up/gate, grouped expert routing.

---

## Performance at a glance

Twelve-cell `(M, PCH)` grid on Qwen3.6-35B-A3B IQ3_S, RTX 5060 Ti 16 GB,
12 streaming requests per cell, `max_tokens=128` — aggregate tokens per
second across concurrent slots:

| M \ PCH | 256 | 512 | 1024 | 2048 |
|---:|:---:|:---:|:---:|:---:|
| **1** | 84.4 | 83.3 | 64.9 [outlier] | 83.2 |
| **4** | 79.9 | **83.5** | 83.0 | 82.3 |
| **8** | 81.3 | 81.9 | 91.8 | **92.9** |

Headlines (all numbers read directly from
[`chimere-server/benchmarks/2026-04-24-multislot-study.md`](chimere-server/benchmarks/2026-04-24-multislot-study.md)
§8, source CSV `/tmp/chimere-sweep-wide/sweep-merged.csv`, chimere-server
SHA `e722ff0`):

- **Single-user peak decode:** 98.7 tok/s per slot at M=1.
- **Production config target (2 to 4 concurrent users):** M=4 / PCH=512,
  aggregate 83.5 tok/s, TTFT p50 **422 ms** (down from 747 ms at PCH=256),
  per-slot decode 22.5 tok/s.
- **Batch workload ceiling:** M=8 / PCH=2048, aggregate 92.9 tok/s —
  unexpected +11 % over M=4, reported honestly as an empirical observation
  pending a higher-N rerun.

Multi-slot does not lift aggregate throughput: per-slot decode scales
as 1/M (98.7 → 22.3 → 14.2 tok/s for M = 1, 4, 8), a signature of the
GDN serialisation barrier rooted in `ik_llama.cpp/src/llama-delta-net.cpp`
and analysed in detail in
[`docs/scheduling-gap-analysis-2026-04-24.md`](docs/scheduling-gap-analysis-2026-04-24.md).
What multi-slot *does* buy is TTFT fairness under concurrent load.

- Operator tuning guide with full decision tree:
  [`docs/perf-tuning.md`](docs/perf-tuning.md).
- Full multi-slot study with methodology, caveats, and honest-limitations
  discussion:
  [`chimere-server/benchmarks/2026-04-24-multislot-study.md`](chimere-server/benchmarks/2026-04-24-multislot-study.md).
- Per-study raw CSV + the reproducible sweep-bench harness:
  [`chimere-server/benchmarks/`](chimere-server/benchmarks/).

---

## Supported architectures

| Arch | GGUF `general.architecture` | Code path | Status | Measured perf | Notes |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (GDN + GQA + MoE) | `qwen35moe` | `Qwen35Model` (full stack) | **PRODUCTION** | 80 tok/s gen, 789 tok/s prefill, 64K ctx, 15.3 GB VRAM | RTX 5060 Ti, ncmoe=3, KV q8_0/q4_0, see [Performance](#performance) |
| Nemotron-3-Nano-30B-A3B (Mamba-2 + GQA + MoE 128top6) | `nemotron_h_moe` | `GenericModel` | Validated end-to-end | ~45 tok/s gen via `test-nemotron`, ctx 2048, ncmoe=30 | Q4_0 and UD-IQ3_XXS, single agent only at Step 7 |
| Mamba-2 (pure SSM) | `mamba2` | `GenericModel` | Backend supported, untested in `chimere-server` | n/a | `state-spaces/mamba2-*`, `mistralai/Mamba-Codestral-7B-v0.1` |
| Mamba-1 | `mamba` | enum present, backend stub | Not loadable today | n/a | Legacy `build_mamba()` body still stubbed in PR #1593 |
| Future Mamba-2 hybrids (Granite 4.0 H-Tiny / H-Small, Falcon-H1, Bamba-9B) | various | `GenericModel` | Untested but expected to work via the same path | n/a | See [Roadmap](#roadmap) |

---

## Quick start

### One-shot install (Linux, NVIDIA)

```sh
git clone https://github.com/AIdevsmartdata/chimere.git
cd chimere
./install-chimere.sh                   # auto-detects SM (sm_120 / sm_89 / sm_86)
./install-chimere.sh --with-model      # also pulls Chimere v3 RAMP GGUF (~15.2 GB)
```

`install-chimere.sh` checks prerequisites (`cmake`, `rustc`, `nvcc`), clones and
builds the `ik_llama.cpp` fork, then builds `chimere-server` in release mode.
It never runs `sudo` — missing packages are reported, not installed.

### Manual build

```sh
# Backend (one-time)
git clone https://github.com/AIdevsmartdata/ik_llama.cpp.git ~/ik_llama.cpp
cd ~/ik_llama.cpp
git checkout mamba2-nemotron-h-backport      # or main once PR #1593 merges
cmake -B build_sm120 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF
cmake --build build_sm120 -j

# Server
git clone https://github.com/AIdevsmartdata/chimere.git
cd chimere/chimere-server
cargo build --release --features server --bin chimere-server
```

`build.rs` and `ffi/build.rs` resolve `IKLLAMACPP_DIR` (default
`$HOME/ik_llama.cpp`) and `IK_LLAMA_BUILD_SUBDIR` (default `build_sm120`),
so the build finds `libllama.so` / `libggml.so` without manual exports.
Override either when your backend lives elsewhere, e.g.:

```sh
IKLLAMACPP_DIR=/opt/ik_llama IK_LLAMA_BUILD_SUBDIR=build_sm89 \
  cargo build --release --features server --bin chimere-server
```

At **runtime**, the binary still needs `LD_LIBRARY_PATH` to find the
shared libraries unless you rely on the embedded rpath (set automatically
when libraries are discovered at build time):

```sh
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
  ./target/release/chimere-server
```

Requirements: CUDA 12.8 toolkit, Rust 1.80+, an NVIDIA GPU with at least 16 GB
of VRAM (Ada `sm_89` works too, replace `120` with `89` in `CMAKE_CUDA_ARCHITECTURES`).

### Run on Qwen3.5-35B-A3B Chimere v3 RAMP (production target)

```sh
hf download Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF chimere-v3-ramp.gguf
hf download Qwen/Qwen3.5-35B-A3B tokenizer.json --local-dir tokenizers/qwen35

CHIMERE_MODEL=$PWD/chimere-v3-ramp.gguf \
CHIMERE_TOKENIZER=$PWD/tokenizers/qwen35/tokenizer.json \
CHIMERE_LLAMA_BACKEND=1 \
CHIMERE_NCMOE=3 \
CHIMERE_KV_MAX_SEQ=65536 \
CHIMERE_PORT=8081 \
CHIMERE_ENGRAM_DIR=$HOME/.openclaw/data/engram \
CHIMERE_FORCE_QWEN35=1 \
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
./target/release/chimere-server
```

### Run on Nemotron-3-Nano-30B-A3B (Mamba-2 + MoE)

```sh
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-Q4_0.gguf
hf download unsloth/Nemotron-3-Nano-30B-A3B tokenizer.json --local-dir tokenizers/nemo

CHIMERE_MODEL=$PWD/Nemotron-3-Nano-30B-A3B-Q4_0.gguf \
CHIMERE_TOKENIZER=$PWD/tokenizers/nemo/tokenizer.json \
CHIMERE_LLAMA_BACKEND=1 \
CHIMERE_NCMOE=30 \
CHIMERE_KV_MAX_SEQ=2048 \
CHIMERE_PORT=8081 \
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
./target/release/chimere-server
```

`chimere-server` peeks at the GGUF metadata, sees `general.architecture =
nemotron_h_moe`, and dispatches to `GenericModel` automatically. The Qwen3.5
hot path is byte-for-byte unchanged.

### Smoke-test the libllama-only path

A bundled binary exercises `LlamaForward` directly, no HTTP, no `Qwen35Model`,
no Engram — useful for bisecting backend issues:

```sh
CHIMERE_MODEL=.../Nemotron-3-Nano-30B-A3B-Q4_0.gguf \
CHIMERE_TOKENIZER=.../Nemotron-3-Nano-30B-A3B/tokenizer.json \
CHIMERE_NCMOE=30 \
CHIMERE_KV_MAX_SEQ=2048 \
LD_LIBRARY_PATH=$HOME/ik_llama.cpp/build_sm120/ggml/src:$HOME/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64 \
cargo run --release --bin test-nemotron
```

### Hello, world

```sh
curl -s http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":32}'
```

---

## Architecture

```
HTTP request (axum 0.8)
       │
       ▼  /v1/chat/completions
chat_completions_handler (server.rs:1208)
       │
       ▼
AppState (server.rs:309)
   ├── model           : Mutex<AppStateModel>
   ├── tokenizer       : Arc<tokenizers::Tokenizer>
   ├── agent_scheduler : Mutex<AgentScheduler>
   ├── user_agent_map  : Mutex<HashMap<user, agent_id>>
   └── model_name, max_agents
       │
       ▼  run_inference() / chat_completions_stream()
match &*AppStateModel               (server.rs:640 / :938)
   │
   ├── Qwen35(Qwen35Model)          → generate_text + generate_with_mtp_streaming
   │      └── full Qwen3.5 stack: MTP, MRoPE, cudarc, block diffusion,
   │          entropy routing, engram-aware sampling, agent context switch
   │
   └── Generic(GenericModel)        → generate_text_generic
          └── libllama FFI only: forward via LlamaForward, no engram,
              no MTP, no DART, no agent switch (Step 7 limitations)
       │
       ▼
LlamaForward (llama_backend.rs)
       │
       ▼
libllama.so  (ik_llama.cpp + Mamba-2 + Nemotron-H Phase 3.x backport)
       │
       ▼
CUDA kernels  (sm_120 native, MoE fused, K-cache Hadamard, ggml_ssm_scan)
```

Both `Qwen35Model` and `GenericModel` implement the `ChimereModel` trait
(`chimere_model.rs:164`). The trait surface is intentionally minimal: identity
(`arch`, `num_layers`, `vocab_size`), capability flags (`supports_mtp`,
`supports_block_diffusion`, `supports_dart`, `supports_entropy_routing`),
forward methods (`forward_token`, `forward_prefill`), and a few libllama
hooks (`llama_set_logit_bias`, `llama_set_engram_bias`). Hoisting `generate()`
onto the trait was deliberately avoided so MTP, NEST and engram interleaving
stays in one place on the Qwen3.5 path.

---

## Features

| Feature | Module | Active on Qwen3.5 | Active on Generic | Description |
|---|---|---|---|---|
| OpenAI `/v1/chat/completions` (non-streaming) | `server.rs` | yes | yes | `messages`, `tools`, `logprobs`, `top_logprobs`, `chat_template_kwargs.enable_thinking` |
| OpenAI SSE streaming | `server.rs` | yes (token-by-token) | yes (single Token + Done, see Limitations) | |
| Qwen3.5 hand-rolled chat template | `server.rs:327 messages_to_prompt` | yes | shared (best-effort) | `<\|im_start\|>` formatter mirroring the Jinja template |
| Tool-call extraction | `server.rs:381` | yes | yes (template-shared) | Parses Qwen3.5 `<tool_call><function=…>` into OpenAI tool_call JSON |
| `<think>` reasoning extraction | `server.rs:454` | yes | n/a | Splits response into `reasoning_content` + `content` |
| Engram multi-table n-gram bias | `engram_lookup.rs`, `mtp_scheduler.rs:648` | yes | **no** (tokenizer-locked) | mmap, FNV-1a, tier-0 Cuckoo filter, per-domain overlay |
| NEST adaptive alpha | `mtp_scheduler.rs:54` | yes (default on) | no | `α_eff = base × engram_conf × (1 − model_conf)` |
| MTP speculative decoding | `mtp_scheduler.rs`, `llama_backend.rs MtpOp` | infrastructure present (gated, see Performance) | no | Sequential verify, n_nextn_layer = 1 for chimere-v3-ramp |
| DART (engram-drafted speculation) | `mtp_scheduler.rs::dart_enabled` | opt-in via `CHIMERE_ENGRAM_DART=1` | no | Uses engram n-grams as a free drafter |
| C++ fast sampler (DRY + min-p + top-p + top-k) | `chimere_sampler_*` FFI in `llama_backend.rs` | yes | yes | Avoids ~993 KB logits copy/token, exports OpenAI-format top-5 logprobs |
| K-cache Hadamard rotation | `llama_backend.rs:513` | yes | yes | Default on, `CHIMERE_KV_HADAMARD=0` to disable |
| Fused MoE up/gate, grouped expert routing | libllama context params | yes | yes | ik_llama defaults |
| Agent context switching | `agent_scheduler.rs` + `llama_state_seq_*` | yes (`max_agents=4`) | **no** (Step 7) | Saves/restores KV + GDN per `req.user` field |
| Block diffusion (MDLM/BD3-LM) | `block_diffusion.rs` | infrastructure present, not wired to HTTP | no | Cosine schedule, confidence-based unmasking |
| Entropy routing (AR ↔ diffusion) | `entropy_router.rs` | infrastructure present | no | 6 signals, 3D decision space |
| Multi-section RoPE (Qwen3.5) | `rope.rs` | yes | n/a | |
| Quality-gated nightly Engram write | `~/.openclaw/bin/engram_write_nightly.py` + systemd timer | external pipeline | n/a | Score ≥ 4 → ingest → decay |

---

## API

### `POST /v1/chat/completions`

Standard OpenAI request, plus a few chimere-specific knobs.

```jsonc
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "What is the capital of France?" }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 0.0,
  "stream": false,
  "logprobs": false,
  "top_logprobs": 5,
  "tools": null,
  "user": "kevin",
  "chat_template_kwargs": { "enable_thinking": true }
}
```

Defaults are defined in `server.rs:151-166` and the upper cap on `max_tokens`
is `MAX_TOKENS_LIMIT = 32768`. Sampling defaults that are NOT request fields
(hardcoded in `server.rs:738-744`):

```
min_p              0.05
dry_multiplier     0.8
dry_base           1.75
dry_min_length     2
dry_penalty_last_n -1   // scan whole sequence
```

The `presence_penalty` default is `0.0` on purpose: a previous default of `1.5`
killed code generation and long reasoning blocks (see comment in `server.rs:165`).

#### Chimere-specific request fields

| Field | Default | Notes |
|---|---|---|
| `user` | none | Routes to a per-user agent ID via `agent_scheduler` (Qwen3.5 only). |
| `chat_template_kwargs.enable_thinking` | `true` | When `false`, the server avoids opening a `<think>` block. |
| `engram_table` | none | Field is parsed; per-request routing to the backend is not wired today. Engram tables are loaded once at server start from `CHIMERE_ENGRAM_DIR`. |
| `engram_alpha` | none | Same — parsed for forward compatibility. The active α is `CHIMERE_ENGRAM_ALPHA` (default `0.5`). |

### `GET /health`

```json
{ "status": "ok", "engine": "chimere-deltanet" }
```

### Endpoints not implemented

`/v1/models`, `/v1/completions`, `/v1/embeddings` are not provided.

---

## Environment variables

The full list (≈55 vars) lives in the source. The ones the production
service unit actually sets, plus the new Step 7 vars:

### Model and dispatch

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_MODEL` | `$HOME/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf` | GGUF path. Used to detect the architecture from `general.architecture`. |
| `CHIMERE_TOKENIZER` | auto-detect | HF `tokenizer.json`. Required for the Generic path until Step 7.5 wires the FFI tokenizer fallback. |
| `CHIMERE_NAME` | `chimere-deltanet` | `model` field echoed in responses. |
| `CHIMERE_LLAMA_BACKEND` | unset | Set to any value to enable the libllama FFI path. **Implicit on the Generic path.** |
| `CHIMERE_CUDARC_FORWARD` | unset | Cudarc raw-weights path. Qwen3.5 only. Ignored on Generic. |
| `CHIMERE_FORCE_QWEN35` | unset (Step 7) | When set, the binary refuses to start unless the loaded GGUF is `qwen35moe`. Belt-and-braces guard for the production slot. |
| `CHIMERE_PORT` | `8090` standalone, `8081` in the systemd unit | Listen port. |
| `CHIMERE_MAX_AGENTS` | `4` | `agent_scheduler` capacity (Qwen3.5 only). |

### Generic / libllama path (Step 7)

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_GENERIC_EOS` | `[2]` | Comma-separated list of stop tokens for `generate_with_mtp_generic` (`mtp_scheduler.rs:1256`). |

### Context and KV cache

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_KV_MAX_SEQ` | `65536` | Context length. |
| `CHIMERE_KV_TYPE_K` | `8` (Q8_0) | Key cache type. |
| `CHIMERE_KV_TYPE_V` | `2` (Q4_0) | Value cache type. |
| `CHIMERE_KV_HADAMARD` | `1` | Hadamard rotation on keys. |
| `CHIMERE_FLASH_ATTN` | `1` | |
| `CHIMERE_BATCH` | `4096` | |
| `CHIMERE_UBATCH` | `512` | |
| `CHIMERE_THREADS` | `14` | |
| `CHIMERE_NCMOE` | `4` (default) / `3` (prod service) / `30` (Nemotron-H smoke test) | First N layers' MoE experts offloaded to CPU. |

### Engram

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_ENGRAM_DIR` | unset | Directory of `.engr` tables. The production unit sets this to `~/.openclaw/data/engram`. |
| `CHIMERE_ENGRAM_FILE` | unset | Single-file backward-compat path. |
| `CHIMERE_ENGRAM_ALPHA` | `0.5` (`generate.rs`) / `0.1` (`mtp_scheduler.rs`, attenuated for response phase) | Logit bias strength `logits[t] += α × ln(p_engram[t])`. |
| `CHIMERE_ENGRAM_NEST` | `1` | NEST adaptive α (Qwen3.5 path). |
| `CHIMERE_ENGRAM_DART` | unset | DART speculative drafter using engram n-grams. |
| `CHIMERE_DART_STEPS` | `5` | DART look-ahead. |

### Debug / experimental

`CHIMERE_DEBUG`, `CHIMERE_VRAM_LOG`, `CHIMERE_TRACE`, `CHIMERE_TRACE_LEVEL`,
`CHIMERE_DISPATCH_PROF`, `CHIMERE_COUNT_OPS`, `CHIMERE_MOE_PROFILE`,
`CHIMERE_CUDA_GRAPH`, `CHIMERE_LM_HEAD_CPU`, `CHIMERE_FLASH_PREFILL`,
`CHIMERE_GQA_FUSED`, `CHIMERE_RAW_FORWARD`, `CHIMERE_NO_FUSED_MOE`,
`CHIMERE_EARLY_EXIT`, … (~40 more, see grep `CHIMERE_` in `chimere-server/src`).

---

## Performance

All numbers measured on the same hardware:

- GPU: NVIDIA RTX 5060 Ti, 16 GB VRAM, sm_120 (Blackwell)
- CPU: Intel i5-14600KF
- RAM: 32 GB DDR5
- Driver: NVIDIA 590.48 (CUDA 12.8)
- Build: `ik_llama.cpp` fork @ branch `mamba2-nemotron-h-backport`,
  `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF`

### Qwen3.5-35B-A3B Chimere v3 RAMP (production)

| Setup | NCMOE | Ctx | Gen tok/s | Prefill tok/s | VRAM used | VRAM free | Notes |
|---|---|---|---|---|---|---|---|
| chimere-server FFI, prod service | 3 | 64K | **80** | 789 | 15.3 GB | 560 MB | TTFT 80 ms, KV q8_0/q4_0 |
| chimere-server FFI, headroom | 4 | 64K | 77 | — | — | 702 MB | "safe" margin |
| chimere-server FFI, max headroom | 4 | 32K | 80 | — | — | 1.2 GB | |
| chimere-server FFI | 2 | 64K | OOM | — | — | — | Not viable on 16 GB |

### Nemotron-3-Nano-30B-A3B (Mamba-2 + MoE)

| Setup | NCMOE | Ctx | Gen tok/s | Notes |
|---|---|---|---|---|
| `cargo run --release --bin test-nemotron` on Q4_0 | 30 | 2048 | ~45 | First end-to-end on the chimere FFI path, sm_120, single agent. Reproduce: `test-nemotron` smoke binary. |

To reproduce: see the [Smoke-test](#smoke-test-the-libllama-only-path) section.
`test-nemotron` is a 91-line binary that loads the GGUF via `llama_backend::from_env`,
prefills the prompt, then greedy-samples N tokens.

### MTP infrastructure status (honest)

The MTP scheduler (`mtp_scheduler.rs`, ~1500 LoC) and the `MtpOp` FFI surface
(`llama_backend.rs`) are both wired up, the Qwen3.5 RAMP build advertises a
single `nextn` head, and an early-March benchmark on a previous build measured
**+49.5 % token acceptance rate** for the MTP draft path.

The current `bench_mtp.rs` binary, however, has Benchmark 2 (MTP decode) and
Benchmark 5 (MTP acceptance rate) hard-coded as `SKIPPED` with the comment
`crash in ik_llama MTP graph, KV cache issue for layer 41` — so the +49.5 %
figure is **not reproducible against the present `ik_llama` head**. Treat MTP
as "infrastructure present, gated, fix planned" rather than as a marketing
number. The non-MTP path is what powers the 80 tok/s figure above.

### Engram quality status (honest)

The engram path is real and useful as a domain-knowledge overlay: the `kine`
table is 19.7 MB of corpus-derived n-grams, and qualitative use on the
production stack shows specialized vocabulary appearing in responses
(`drainage bronchique postural`, `EMII`, etc., on the kiné domain).

A **quantitative** perplexity gain on Qwen3.5 has not been measured yet. The
only saved engram eval in the repo
(`~/.openclaw/workspaces/chimere/benchmarks/engram_trained_eval.json`) was
**run on GPT-2 + wikitext-2** (a different tokenizer and a different model
class) and shows **−13.39 % PPL regression** on that out-of-distribution
setup, which is not representative of the prod path. We are not citing it as
a quality claim and we will publish a Qwen3.5-specific eval before doing so.
Engram is shipped as an opt-in domain overlay, not as a "quality boost"
button.

### ik_llama vs stock llama.cpp (Qwen3.5-35B-A3B MoE, our hardware)

Same model, same context, same KV cache config:

| Quant | Gen tok/s gain | Prefill tok/s gain |
|---|---|---|
| Q4_K_M | +18 % | — |
| IQ3_S | +32 % | — |
| Q5_K_XL | +19 % | +165 % |

(Numbers from `benchmarks/benchmark-qwen35-2026-03-07.md`. ik_llama also has a
known multi-slot concurrency bug — the chimere prod path is single-slot.)

---

## Backend: `ik_llama.cpp` fork + Phase 3.x backport

Chimere does not ship its own GGUF reader or its own CUDA kernels; both come
from a customized `ik_llama.cpp` fork:

- Upstream: <https://github.com/ikawrakow/ik_llama.cpp>
- Our fork: <https://github.com/AIdevsmartdata/ik_llama.cpp>
- Open PR: [ikawrakow/ik_llama.cpp#1593 — Mamba-2 + Nemotron-H MoE backport](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)

### What the backport adds (12 commits, branch `mamba2-nemotron-h-backport`)

| Commit | Purpose |
|---|---|
| `edbd64f` | Phase 1: stub `mamba2` + `nemotron_h_moe` metadata |
| `0c578cb` | Phase 2: hparams loading + tensor allocation |
| `b7f9209` | Rename `n_embd_k_s/v_s` → `n_embd_r/s`, move to `.cpp` |
| `61f7996` | First-class `use_qnext_state_layout` flag |
| `b9c58a0` | Stub before `ggml_ssm_scan` signature change |
| `3bafe93` | **Phase 3.2**: port upstream `ggml_ssm_scan` op + CUDA backend |
| `d88ee7a` | Defensive SSM bounds for Mamba-2 + Nemotron-H load |
| `af9a12e` | **Phase 3.3**: `build_mamba` (Mamba-2) + `build_nemotron_h_moe` |
| `fcdbfc2` | `eval-callback`: sum over full tensor, not printed slice |
| `807bf7b` | `inp_ssm_ids` reads recurrent slot 0, not `kv_self.head` |
| `ecf2842` | `llm_build_ffn`: guard parallel-gate fold on `gate != nullptr` (gateless RELU² FFNs, Nemotron-H shared expert) |
| `8c33d29` | API drift catch-up vs current upstream |

### Validation quoted in the PR body

`llama-cli -p "The capital of France is" -n 20`:

- `Nemotron-3-Nano-30B-A3B-Q4_0.gguf`
  → "The capital of France is Paris." ...
- `Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf`
  → "The capital of France is Paris, and the capital of Italy is Rome, ..."
- `Qwen3.5-35B-A3B-Chimere-v3-RAMP.gguf` (regression check)
  → "The capital of France is Paris, a city with..."

### Known backend limitations (from PR #1593)

1. **`n_seqs == 1`** is hardcoded in `build_mamba2_layer` for the Mamba-2 /
   Nemotron-H path. Qwen3.5 GDN multi-seq decoding is unaffected. Multi-sequence
   decoding for Mamba-2 is reserved to a future Phase 3.5.
2. **State save / restore** (`llama_state_seq_*`) walks the legacy K-cache
   layout for hybrid Nemotron-H. `--cache-reuse` is therefore broken for that
   architecture, and chimere's `agent_scheduler` does not work on the Generic
   path. Fresh prompts are fine.
3. **Mamba-1 legacy `build_mamba()`** is still stubbed and aborts. Use
   `mamba2`-class GGUFs instead.
4. **Phase 3.3 reuses the old 4-arg `ggml_ssm_conv`** rather than upstream's new
   2-arg rewrite. Numerically identical for `n_seqs=1`, 23 ops saved per SSM
   layer that we are not yet capturing.

---

## Models tested

- **`Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF`** —
  `chimere-v3-ramp.gguf` (15.65 GB, custom RAMP IQK mix from
  [ramp-quant](https://github.com/AIdevsmartdata/ramp-quant)). The production
  target.
- **`unsloth/Nemotron-3-Nano-30B-A3B-GGUF`** — `Q4_0` and `UD-IQ3_XXS` quants.
  Validated end-to-end on the Generic path.

The same Generic code path **should** also load the broader Mamba-2 hybrid
ecosystem listed below (we have not run them yet — your mileage may vary, see
[Roadmap](#roadmap)):

- IBM Granite 4.0 H-Tiny / H-Small / H-Micro (`granitemoehybrid`,
  `nemotron_h`-class hybrids)
- Falcon-H1 0.5B – 34B (`tiiuae`, parallel attention + Mamba-2)
- Bamba-9B v1 / v2 (`ibm-ai-platform`, dense Mamba-2 hybrid)
- `state-spaces/mamba2-*` (pure Mamba-2)
- `mistralai/Mamba-Codestral-7B-v0.1` (pure Mamba-2 code)
- AI21-Jamba-Reasoning-3B (Mamba + attention + MoE triple)

A complete inventory of architectures and quant formats reachable from
chimere's backend lives in
[`paper/`](paper/) and in the formats survey on the author's desk.

---

## Roadmap

### Step 7.5 (next)

- FFI tokenizer fallback for `GenericModel` so the Generic path no longer
  requires an external `tokenizer.json` (use `LlamaForward::tokenize` /
  `LlamaForward::detokenize`).
- Multi-agent context switching on the Generic path (depends on PR #1593
  caveat #2 being lifted).
- True token-by-token SSE streaming on the Generic path (today: one Token +
  Done envelope, see `server.rs:1030-1061`).
- Wire `LlamaForward::apply_chat_template` so non-Qwen archs use the
  GGUF-embedded template instead of the Qwen3.5 hand-rolled one.

### Backend (`ik_llama.cpp`)

- Phase 3.5: lift `n_seqs == 1` for Mamba-2 mixers.
- Port upstream's new 2-arg `ggml_ssm_conv` rewrite (23 ops/layer saved).
- Hybrid state save/restore for `llama_state_seq_*` so `agent_scheduler`
  works on Nemotron-H.
- MXFP4 cherry-pick from upstream `ggml-org/llama.cpp` (gpt-oss support).
- NVFP4 once upstream stabilises (RTX 5060 Ti is `sm_120`-native FP4).

### Quantization (`ramp-quant`)

- Trellis IQ_KT bench on Qwen3.5-35B-A3B (ik_llama-only, ~3 bpw QTIP-derived).

### Architecture exploration

- Validate Granite 4.0 H-Tiny and Bamba-9B end-to-end on the Generic path.
- Explore Mamba-3 (arXiv:2603.15569) once upstream support lands.

---

## Citations

### Architectures

- Mamba (Gu & Dao, 2023). <https://arxiv.org/abs/2312.00752>
- Mamba-2 / SSD (Dao & Gu, ICML 2024). <https://arxiv.org/abs/2405.21060>
- Nemotron-H (NVIDIA, 2024). <https://arxiv.org/abs/2411.15241>
- Qwen3-Next / Gated DeltaNet (Qwen, 2025).
- Hyperbolic embeddings, Poincaré ball (Nickel & Kiela, NeurIPS 2017).

### Algorithms

- Multi-Token Prediction (Gloeckle et al., Meta 2024).
  <https://arxiv.org/abs/2404.19737>
- NEST n-gram retrieval bias (referenced in `mtp_scheduler.rs`).
- DRY sampling (community, 2024).
- Cuckoo filter (Fan, Andersen, Kaminsky, Mitzenmacher, 2014).
- BD3-LM / MDLM discrete masked diffusion (referenced in `block_diffusion.rs`).
- Entropix (2024) — entropy × varentropy 2D routing.

### Repos

- ik_llama.cpp upstream: <https://github.com/ikawrakow/ik_llama.cpp>
- ik_llama.cpp fork: <https://github.com/AIdevsmartdata/ik_llama.cpp>
- chimere PR backport: <https://github.com/ikawrakow/ik_llama.cpp/pull/1593>
- llama.cpp upstream: <https://github.com/ggml-org/llama.cpp>
- ramp-quant: <https://github.com/AIdevsmartdata/ramp-quant>
- chimere-odo: <https://github.com/AIdevsmartdata/chimere-odo>

---

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright Kevin Rémondière.

---

## Acknowledgments

- **Iwan Kawrakow** for `ik_llama.cpp`, the K-quants, the IQ-quants, and the
  trellis IQ_KT family. Chimere lives on top of his fork.
- **Tri Dao** and **Albert Gu** for Mamba and Mamba-2 (SSD).
- **NVIDIA** for the Nemotron-H paper and the open-weight 30B-A3B release that
  drove the Phase 3.x backport.
- **Unsloth** for the dynamic quants and the corrected GGUFs (chimere-v3 RAMP
  is built on top of an Unsloth BF16 imatrix).
- **Anthropic Claude** for development assistance throughout the multi-arch
  refactor.
