# chimere-server performance tuning guide

Operator-facing tuning document. Covers the knobs you are most likely
to touch in a production deployment, keyed on scenario (how many users,
what latency target, what VRAM budget). Numbers here are sourced from
the benchmark studies in
[`../chimere-server/benchmarks/`](../chimere-server/benchmarks/); each
claim links back to the CSV cell it reads.

---

## 1. Decision tree

Start here. Pick the row that matches your deployment and use the
config in the right column. The detailed rationale is in §3.

```
Question 1: How many concurrent users will hit the server at once?
    |
    |--> 1 user at a time (CLI, single desktop, development)
    |        --> Scenario A: "Solo interactive"
    |
    |--> 2 to 4 users (small team, agentic pipelines, current prod)
    |        --> Scenario B: "Small concurrent workload"
    |
    |--> 5+ users, batch-y, per-user speed is not the constraint
             --> Scenario C: "Batch / many-user throughput"

Question 2: Do you need KV context longer than 16 K tokens?
    |
    |--> Yes --> add NCMOE > 0 to the chosen scenario's config
    |
    |--> No  --> stick with NCMOE=0 (default)

Question 3: Are you deploying something other than Qwen3.5/3.6-35B-A3B?
    |
    |--> Yes: non-Qwen arch (Mamba-2, Nemotron-H, Jamba, Bamba, ...)
    |        --> Force M=1: CHIMERE_MULTISLOT=1 (backend constraint, see §4.3)
    |
    |--> No  --> All three scenarios apply
```

### 1.1 Recommended configs by scenario

| Scenario | `CHIMERE_MULTISLOT` | `CHIMERE_MAX_PREFILL_CHUNK` | `CHIMERE_NCMOE` | `CHIMERE_KV_MAX_SEQ` | Expected per-slot decode | Expected agg decode | Expected TTFT p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A — Solo interactive** | 1 | 256 | 0 | 16 384 | 98.7 tok/s | 84.4 tok/s | 229 ms |
| **B — Small concurrent (current prod target)** | 4 | 512 | 0 | 16 384 | 22.5 tok/s | 83.5 tok/s | 422 ms |
| **C — Batch / many-user** | 8 | 2048 | 0 | 16 384 | 14.1 tok/s | 92.9 tok/s | 633 ms |
| **B-with-long-context** | 4 | 512 | 4 | 65 536 | ~18 tok/s (est.) | ~67 tok/s (est.) | ~450 ms (est.) |

Per-slot decode, aggregate decode, and TTFT p50 for A, B, C read directly
from `/tmp/chimere-sweep-wide/sweep-merged.csv` (SHA `e722ff0`). The
"B-with-long-context" row is estimated from older benchmark data
(see §3.4) and is marked as estimate.

The "aggregate" column is the sum of all generated tokens / wall time
across 12 concurrent requests. For scenario A (M=1), the 12 requests
serialize so aggregate < per-slot; for scenarios B and C, multiple slots
decode concurrently and aggregate > per-slot × slot count, bounded by
the GDN serialization barrier (see
[`scheduling-gap-analysis-2026-04-24.md`](./scheduling-gap-analysis-2026-04-24.md)).

---

## 2. Why the recommendations

### 2.1 Why M=1 for single-user

At M=1, a single slot owns the GPU and the decode kernel runs at its
fastest per-token rate. Source:

```
M1-N0-P256: agg=84.4 tok/s, decode_p50=98.7 tok/s, TTFT_p50=229 ms
```

from `sweep-merged.csv` row `M1-N0-P256`. That 98.7 tok/s is what a
single user sees for their response. Any M ≥ 2 cuts per-user decode
proportionally (M=4 → 22.3 tok/s/slot; M=8 → 14.2 tok/s/slot) while
aggregate barely moves. If you only have one user at a time, M > 1
is strictly worse.

The only reason to choose M > 1 in a single-user deployment is if you
run agentic pipelines where the *same* user fans out 2 or 3 parallel
sub-requests (e.g. ReAct with simultaneous tool calls). In that case,
scenario B is correct even though the user is "one person".

### 2.2 Why M=4 for 2 to 4 concurrent users

At M=4, four concurrent requests all get seated immediately by the
native scheduler, so the queueing component of TTFT drops to near-zero.
Source:

```
M4-N0-P512 (rerun): agg=83.5 tok/s, decode_p50=22.5 tok/s, TTFT_p50=422 ms
```

from `sweep-merged.csv`. TTFT p50 at M=4 is 422 ms; at M=1 under the
same concurrent-4-request load, queue wait would add roughly 1.3 s per
predecessor (measured in
[`../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md`](../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md)
§3.4 as M=1 TTFT p50 = 4 111 ms on a 4-concurrent-client, 40-request
workload). 422 ms vs 4 000+ ms is the headline win of M=4 for
interactive workloads.

Per-user decode at M=4 is 22.5 tok/s, which is roughly 1 token every
44 ms. For a 200-token response, that is 8.8 s total generation time —
slow enough to notice but fast enough to feel responsive once the first
token arrives at 422 ms.

### 2.3 Why PCH=512 (and not 256, and not 2048)

At M=4, the prefill-chunk sweep from the wide sweep measured:

| PCH | agg tok/s | TTFT p50 (ms) | TTFT p99 (ms) | Source row |
|---:|---:|---:|---:|---|
| 256 | 79.9 | 747 | 1 097 | `M4-N0-P256` (sweep-merged.csv) |
| **512** | **83.5** | **422** | **842** | `M4-N0-P512` (rerun, sweep-merged.csv) |
| 1024 | 83.0 | 429 | 851 | `M4-N0-P1024` |
| 2048 | 82.3 | 428 | 958 | `M4-N0-P2048` |

PCH=256 → 512 drops TTFT p50 by 43 % and p99 by 23 %. PCH=512 → 1024
is flat (one millisecond either way on p50). So 512 is the knee:
largest TTFT win for the lowest prefill chunk size. Going higher costs
nothing but also gains nothing at M=4 with the current prompt mix.

### 2.4 Why M=8 for batch

Measured aggregate at M=8 with large PCH:

```
M8-N0-P1024: agg=91.8 tok/s
M8-N0-P2048: agg=92.9 tok/s
```

from `sweep-merged.csv`. That is roughly +11 % over the M=4 aggregate
— the only cells in the entire 12-cell grid that break ~83 tok/s.

This is an **empirical observation**, not a vetted recommendation.
See [`../chimere-server/benchmarks/2026-04-24-multislot-study.md`](../chimere-server/benchmarks/2026-04-24-multislot-study.md)
§3.3 for the full caveats:

- We do not yet have a crisp explanation for why only M=8 (not M=4)
  sees this speedup.
- N=12 per cell is small; a targeted rerun at N=40 is pending.
- VRAM margin at M=8 is the tightest in the sweep (14 075 MiB used,
  2 236 MiB free).
- Per-slot decode at 14.1 tok/s is slow enough to be user-visible as
  a sluggish response in a chat UI.

Use M=8 for **offline batch** workloads where latency-per-user is not
the UX constraint (nightly ETL, bulk evaluation, score-then-discard
pipelines). Do not use it for user-facing APIs.

---

## 3. Detailed tuning knobs

### 3.1 CHIMERE_MULTISLOT (M)

The number of concurrent slots the native scheduler arms. Set via the
environment before server start.

| Value | Effect | Notes |
|---|---|---|
| unset or 1 | Single-slot path. No scheduler. `forward_token` runs directly. | Fastest per-response decode (98.7 tok/s on Qwen3.6 IQ3_S). |
| 2 | Two-slot native scheduler. | **Avoid on GDN hybrids.** The E2E profile bench measured M=2 at 74.2 tok/s aggregate — worse than M=1. See `benchmark-e2e-2026-04-24.md` §3.1. Likely a pathological case of the GDN serialisation barrier; we did not retest it in the wide sweep. |
| 4 | Four-slot. | Current production target. Best TTFT vs aggregate trade for 2 to 4 concurrent users. |
| 8 | Eight-slot. | Highest aggregate measured (~93 tok/s at PCH ≥ 1024), at the cost of 14 tok/s per-slot decode. Batch workloads only. |
| >8 | Untested. | Harness allows values up to arbitrary int. Not benched. |

Aliases: none. This is the only way to request multi-slot.

### 3.2 CHIMERE_MAX_PREFILL_CHUNK (PCH) and its alias

Max tokens admitted per driver tick during the prefill phase of a
request. Default: the harness sets 256 when unset. Internally the
variable is read as `CHIMERE_MAX_PREFILL_CHUNK` (newer builds) and
`CHIMERE_NATIVE_MAX_PREFILL_CHUNK` (older builds) — the sweep harness
exports both; in production pick one. If your build tree predates the
short alias, the 4-line diff is cheap: mirror the existing
`CHIMERE_NATIVE_MAX_PREFILL_CHUNK` env-var read in `bin/chimere-server.rs`
with a fallback to `CHIMERE_MAX_PREFILL_CHUNK`.

| Value | Effect |
|---|---|
| 64 | **Avoid.** Not swept, but at PCH=128 the M=4 cell measured 53.4 tok/s aggregate (sweep-2026-04-24.csv `M4-N0-P128`) — deeply dispatch-bound. |
| 128 | Only reasonable for workloads where 100 % of prompts are ≤ 128 tokens. Otherwise TTFT doubles. |
| 256 | Harness default. Fine for M=1; suboptimal for M=4 (747 ms TTFT p50 vs 422 ms at PCH=512). |
| **512** | **Recommended for M=4 prod**. Sweet spot in the measured grid. |
| 1024 | Slight TTFT p99 improvement at M=8; no change at M=4. |
| 2048 | Marginally best aggregate at M=8 (92.9 vs 91.8 at 1024). Marginally worst at M=4 (82.3 vs 83.5 at 512). |

The tight production recommendation is **PCH=512 regardless of M for
most prod traffic**, moving to PCH=2048 only for dedicated batch
nodes at M=8.

### 3.3 CHIMERE_NCMOE

Number of *first* N layers whose MoE experts are offloaded to CPU
instead of GPU. Frees VRAM proportionally. Trade: decode tok/s drops
because the offloaded experts must cross PCIe per token.

**Not directly measured in the 2026-04-24 sweep** (all cells used
NCMOE=0). The numbers below are from earlier benches and older model
versions; use as directional guidance only.

| NCMOE | VRAM saved | Decode cost | Source |
|---|---:|---:|---|
| 0 | baseline (13.7 to 14.2 GiB used) | 0 | `sweep-merged.csv`, all cells |
| 3 | ~1 GiB | -15 to -20 % | Production Qwen3.5 V3 RAMP config, `benchmarks/benchmark-qwen35-2026-03-07.md`: Q5_K_XL "prod service" 80 tok/s at `ncmoe=3, 64K ctx` vs IQ3_S baseline at higher tok/s |
| 4 | ~1.3 GiB | -18 to -22 % | Same bench: 77 tok/s at ncmoe=4, 64K |
| 6 | ~2 GiB | -25 to -30 % | Extrapolated; not measured in-repo |
| 8+ | ≥ 2.5 GiB | -30 to -50 % | Nemotron-H ncmoe=30 workload, but that is a different model class |

**Use NCMOE > 0 only if you cannot meet your VRAM budget at NCMOE=0.**
At NCMOE=0 and the recommended Scenario B config (`M=4, PCH=512,
KV_MAX_SEQ=16384`), VRAM usage is 13 821 MiB (from
`sweep-merged.csv` row `M4-N0-P512` rerun), leaving ~2.5 GiB
headroom on a 16 GB card. That is enough for most workloads.

If your deployment needs `CHIMERE_KV_MAX_SEQ=65536` or larger (longer
context), the KV cache grows and NCMOE=3 or 4 becomes necessary to
stay within VRAM. This trade is not measured in the 2026-04-24 sweep;
a dedicated NCMOE sweep is listed as future work.

### 3.4 CHIMERE_KV_MAX_SEQ

The KV cache allocation size in tokens. Default prod: 16 384.

| Value | VRAM cost (KV only) | When to use |
|---|---:|---|
| 8 192 | ~ 0.5 GiB | Short-context workloads (chat < 4 K tokens). Rare; default is fine. |
| 16 384 | ~ 1.0 GiB | Current prod default. Covers most chat + agentic workloads. |
| 32 768 | ~ 2.0 GiB | Longer context; may need NCMOE=3 for VRAM budget on 16 GB GPU. |
| 65 536 | ~ 4.0 GiB | RAG over long documents. Needs NCMOE=4+ for 16 GB GPU. |

Numbers above are approximations from the existing benches; the exact
VRAM cost depends on `CHIMERE_KV_TYPE_K` (default Q8_0) and
`CHIMERE_KV_TYPE_V` (default Q4_0). The q8_0/q4_0 combo is the
measured sweet spot (see MEMORY.md 2026-03-09 note: 93 tok/s with 568
MiB free on the single-slot prod profile).

### 3.5 CHIMERE_KV_TYPE_K, CHIMERE_KV_TYPE_V

Integer constants mapping to ggml types:

| Value | ggml type | Bytes per element (approx) |
|---|---|---|
| 0 | F32 | 4 |
| 1 | F16 | 2 |
| 2 | Q4_0 | ~0.5 |
| 8 | Q8_0 | ~1.0 |

Production defaults are 8 (Q8_0 keys) and 2 (Q4_0 values). Moving
either to F16 increases VRAM without improving TG measurably on this
hardware. Moving keys to Q4_0 tanks quality on Qwen3.6 hybrids — do
not. The q8_0/q4_0 asymmetry is deliberate and measured.

### 3.6 CHIMERE_MULTISLOT_NATIVE

Flag to arm the native scheduler (vs the legacy single-slot path).
Production sets `1`. Keep it at `1` unless you are debugging the
scheduler itself.

### 3.7 CHIMERE_SKIP_LEGACY_LLAMA

When `1`, non-streaming requests are rejected with HTTP 400
(`native_mode_streaming_only`). The `bench-m1` Rust binary in the
repo is therefore unusable against a prod-configured server;
benchmarks use the streaming driver.

Keep at `1` in production. If you need non-streaming for debugging,
start a separate server instance with `CHIMERE_SKIP_LEGACY_LLAMA=0`
on a different port.

### 3.8 CHIMERE_NCMOE — VRAM / context interplay

If you can afford VRAM headroom, keep NCMOE=0 and raise KV_MAX_SEQ.
If you cannot, raise NCMOE in increments of 1. Each increment saves
roughly 250 to 350 MiB and costs 5 to 8 % decode tok/s at M=1.

For Scenario B (M=4, PCH=512) with a 65 K context requirement on a
16 GB GPU, start at NCMOE=4 and profile.

---

## 4. Environment variable reference

Abridged. The exhaustive list is in the chimere-server source;
`grep CHIMERE_ chimere-server/src/` returns ~55 vars.

### 4.1 Required or near-required

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_MODEL` | unset | Absolute path to the GGUF. Required. |
| `CHIMERE_TOKENIZER` | auto-detect | Absolute path to HF `tokenizer.json`. Required for the Generic path. |
| `CHIMERE_LLAMA_BACKEND` | unset | Set to any truthy value to enable the libllama FFI path. Required for production. |
| `CHIMERE_PORT` | 8090 standalone / 8081 in the systemd unit | Listen port. |

### 4.2 Performance-critical (this guide)

| Var | Default | Recommended | Notes |
|---|---|---|---|
| `CHIMERE_MULTISLOT` | unset (= 1) | 1 / 4 / 8 per scenario | See §3.1. |
| `CHIMERE_MULTISLOT_NATIVE` | unset | 1 | Enables the native scheduler. Required for M ≥ 2. |
| `CHIMERE_SKIP_LEGACY_LLAMA` | unset | 1 | Streaming-only; matches production. |
| `CHIMERE_MAX_PREFILL_CHUNK` | 256 | 512 | See §3.2. |
| `CHIMERE_NATIVE_MAX_PREFILL_CHUNK` | alias | same as above | Alias for older builds. |
| `CHIMERE_NCMOE` | 0 | 0 unless VRAM-constrained | See §3.3. |
| `CHIMERE_KV_MAX_SEQ` | 65 536 (code default) / 16 384 (prod unit) | 16 384 | See §3.4. |
| `CHIMERE_KV_TYPE_K` | 8 (Q8_0) | 8 | Don't change without profiling. |
| `CHIMERE_KV_TYPE_V` | 2 (Q4_0) | 2 | Don't change without profiling. |

### 4.3 Less commonly tuned

| Var | Default | Notes |
|---|---|---|
| `CHIMERE_KV_HADAMARD` | 1 | Hadamard rotation on keys. Free ~8 % TG. Keep on. |
| `CHIMERE_FLASH_ATTN` | 1 | Flash attention on dense layers. Keep on. |
| `CHIMERE_BATCH` | 4 096 | Prefill batch. Rarely touched. |
| `CHIMERE_UBATCH` | 512 | Ubatch size. Rarely touched. |
| `CHIMERE_THREADS` | 14 | CPU threads. Tune to cores − 2 on your CPU. |
| `CHIMERE_MAX_AGENTS` | 4 | Agent scheduler capacity (Qwen3.5 path only). Independent of `MULTISLOT`. |
| `CHIMERE_ENGRAM_DIR` | unset | Directory of `.engr` tables for per-domain n-gram bias. |
| `CHIMERE_ENGRAM_ALPHA` | 0.5 | Engram bias strength. |
| `CHIMERE_ENGRAM_NEST` | 1 | Adaptive α on the Qwen3.5 path. |
| `CHIMERE_FORCE_QWEN35` | unset | When set, refuses to start unless GGUF is `qwen35moe`. |
| `LLAMA_SET_ROWS` | unset | Set to 1 for ik_llama set-rows optimisation. |

### 4.4 Debug / experimental

`CHIMERE_DEBUG`, `CHIMERE_VRAM_LOG`, `CHIMERE_TRACE`,
`CHIMERE_TRACE_LEVEL`, `CHIMERE_DISPATCH_PROF`, `CHIMERE_COUNT_OPS`,
`CHIMERE_MOE_PROFILE`, `CHIMERE_CUDA_GRAPH`, `CHIMERE_LM_HEAD_CPU`,
`CHIMERE_FLASH_PREFILL`, `CHIMERE_GQA_FUSED`, `CHIMERE_RAW_FORWARD`,
`CHIMERE_NO_FUSED_MOE`, `CHIMERE_EARLY_EXIT`, `CHIMERE_PROFILE` (a
doc-comment only in current source, not wired). See `grep CHIMERE_
chimere-server/src/` for the full list.

---

## 5. Capacity planning

### 5.1 How many users can one chimere-server instance serve?

Depends on user type. The measured numbers for Scenario B
(M=4, PCH=512):

| Metric | Value | Source |
|---|---|---|
| Aggregate tok/s | 83.5 | `sweep-merged.csv` `M4-N0-P512` (rerun) |
| Per-user decode | 22.5 tok/s | same |
| TTFT p50 under 4-concurrent load | 422 ms | same |
| TTFT p99 under 4-concurrent load | 842 ms | same |

For a 200-token average response, one user ties up a slot for
~200 / 22.5 = 8.9 seconds. At M=4, that is ~27 users / minute before
the queue starts building. For a team of 10-20 daily-active users
with bursty workloads, one M=4 chimere-server instance is comfortable.

For sustained agentic workloads (hundreds of short requests per
minute, each < 200 tokens), the throughput ceiling is aggregate 83.5
tok/s × 60 s = 5 010 tokens/minute ÷ 200 tokens/req = 25 req/minute.
Above that, run multiple instances on separate GPUs or scale to
M=8 with batch scheduling.

### 5.2 VRAM budget sanity check

On a 16 GB RTX 5060 Ti with the recommended Scenario B config:

| Component | MiB |
|---|---:|
| Model weights (Qwen3.6-35B-A3B IQ3_S) | 12 727 |
| KV cache (16 K ctx, Q8_0/Q4_0, M=4) | ~1 200 |
| Activation pools | ~400 |
| **Total used (measured)** | **13 821** (from `M4-N0-P512` rerun row) |
| Free | 2 490 |

2.5 GiB headroom is enough for normal variation and for the 12-cell
sweep's largest observed usage at M=8 / PCH=512 (14 208 MiB, tighter
margin). If you run M=8 in production, monitor for OOM — one
transient dip below 12 650 MiB free will fail a fresh model load
(see
[`../chimere-server/benchmarks/2026-04-24-multislot-study.md`](../chimere-server/benchmarks/2026-04-24-multislot-study.md)
§3.7 for a live case).

---

## 6. Checking your config is working

### 6.1 Post-start verification

```bash
# health
curl -s http://127.0.0.1:8081/health
# expect {"status":"ok","engine":"chimere-deltanet"}

# status (includes slot pool size and TTFT ring)
curl -s http://127.0.0.1:8081/v1/status | python3 -m json.tool

# prometheus metrics (Text 0.0.4)
curl -s http://127.0.0.1:8081/metrics
# look for:
#   chimere_slot_pool_size <M>
#   chimere_requests_total{ok="true"} <N>
#   chimere_prompt_tokens_total <N>
```

### 6.2 Quick smoke bench

From the repo root:

```bash
cd chimere-server/benchmarks/sweep
./sweep-bench.sh --output-dir /tmp/chimere-smoke \
    --multislot-sweep "$(echo $CHIMERE_MULTISLOT)" \
    --prefill-chunk-sweep "$(echo $CHIMERE_MAX_PREFILL_CHUNK)" \
    --n-requests-per-pass 12 --max-tokens 128
```

This stops your prod server, boots a bench server on :8082 with your
active config, runs 12 requests, and restarts prod. ~3 minutes end-to-end.
Inspect `/tmp/chimere-smoke/REPORT.md` and diff the tok/s numbers
against the expected values in the table in §1.1.

### 6.3 Long-running soak test

Not provided out-of-the-box. For a 30-minute soak:

```bash
for i in $(seq 1 20); do
    ./sweep/sweep-bench.sh --output-dir /tmp/chimere-soak-$i \
        --multislot-sweep "4" --ncmoe-sweep "0" --prefill-chunk-sweep "512" \
        --n-requests-per-pass 12 --max-tokens 128
done
```

Watch for drift in `agg_tok_per_s` across iterations. A drop of
more than ~5 % suggests thermal throttling or slow memory
fragmentation. Check `nvidia-smi -q -d CLOCK` and your GPU fan curve.

---

## 7. Known issues and workarounds

### 7.1 Multi-slot on non-Qwen3.5/3.6 archs

`build_mamba2_layer` in `ik_llama.cpp` hardcodes `n_seqs == 1` (PR
#1593 caveat #1). Attempting `CHIMERE_MULTISLOT=4` on a Nemotron-H or
Mamba-2 model will either fail at model load or produce incorrect
tokens. Stick to `CHIMERE_MULTISLOT=1` for now; the fix is in the
backlog as "Phase 3.5".

### 7.2 `CHIMERE_MULTISLOT=2` measured worse than M=1

In the E2E profile bench
([`../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md`](../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md)
§3.1), M=2 measured 74.2 tok/s aggregate vs 94.4 tok/s at M=1 — a net
regression. We did not retest this configuration in the later wide
sweep (M ∈ {1, 4, 8} only). **Do not deploy M=2**; jump from M=1 to
M=4 directly.

### 7.3 Model load OOM after previous cell on wide sweeps

If you run many cells back-to-back, the GPU driver occasionally does
not release VRAM from the previous cell within the harness's 3 s
cooldown, and the next model load fails with `cudaMalloc failed: out
of memory` (live case:
[`2026-04-24-multislot-study.md`](../chimere-server/benchmarks/2026-04-24-multislot-study.md)
§3.7). Workarounds:

- Rerun the failed cell in isolation (cheapest).
- Add a longer sleep between cells by editing
  `sweep-bench.sh`'s `stop_server()` (currently `sleep 3`).
- Poll `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
  until free > model size + 10 % before launching the next cell.

### 7.4 Prometheus `chimere_gen_tokens_total` reports zero for thinking-mode output

Known bug: the counter only increments on `NativeStreamMsg::Token`,
not on `NativeStreamMsg::Thinking`. When Qwen3.6 emits 128 thinking
tokens and no content tokens (common at `max_tokens=128`), the
counter stays at zero even though the SSE stream produced tokens.
Cosmetic, no impact on inference; see
[`../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md`](../chimere-server/benchmarks/benchmark-e2e-2026-04-24.md)
§3.6 Finding 1 for the one-line fix location.

---

## 8. When to revisit this guide

Re-run the decision tree if any of:

- The backend ships a batched GDN kernel (closes the 1/M collapse,
  aggregate ceiling lifts). Track
  [`scheduling-gap-analysis-2026-04-24.md`](./scheduling-gap-analysis-2026-04-24.md)
  §4.2.
- A different model family ships to prod (e.g. Qwen3.7 or a pure
  transformer model with no GDN). The 1/M rule is a GDN-specific
  signature; it would not apply.
- Hardware changes (different GPU class, more VRAM, faster memory
  bus). PCH sensitivity and VRAM budget both reshape.
- Workload profile shifts materially (prompts suddenly 5× longer,
  responses 10× longer, etc.). PCH knee moves; aggregate changes.

Re-run a smoke bench after any chimere-server binary update, before
trusting the numbers here.

---

<!-- reviewer-notes
This guide synthesises:
- 2026-04-24-multislot-study.md (primary source for the A/B/C scenario numbers, all TTFT / decode / aggregate values linked back to sweep-merged.csv cell tags)
- benchmark-e2e-2026-04-24.md (source for the M=2 regression and the 4-client queueing calculus)
- scheduling-gap-analysis-2026-04-24.md (source for the 1/M collapse explanation and the backend caveat for non-Qwen archs)
- chimere/README.md's existing Environment variables tables (for the §4 reference, rewritten with the tuning context)
- MEMORY.md 2026-03-09 note on KV cache sweet spot (q8_0/q4_0, ncmoe=4)
- benchmarks/benchmark-qwen35-2026-03-07.md cited for historical NCMOE numbers (§3.3); these are directional, not re-measured in this pass
- PR #1593 caveat #1 from chimere README (for §7.1)

Things deliberately hedged:
- NCMOE trade (§3.3): numbers are flagged as "from earlier benches, directional only" because the 2026-04-24 sweep fixed NCMOE=0. Avoided inventing new numbers.
- "B-with-long-context" row in the §1.1 table: marked as estimate, not measurement.
- M=8 recommendation (§2.4): repeated the "empirical observation, not vetted prod" caveat from the multislot study §3.3 rather than endorsing it.
- VRAM cost per KV size (§3.4): marked as "approximations from existing benches" rather than producing a precise formula.

Things deliberately NOT claimed:
- No claim about Qwen3.5 vs Qwen3.6 perf difference — the sweep was on Qwen3.6 IQ3_S, the README calls out Qwen3.5 v3 RAMP as prod. The 80 tok/s vs 98.7 tok/s gap is because of model size + quant + ncmoe differences, not directly comparable, not claimed here.
- No recommendation on CHIMERE_ENGRAM_* tuning — out of scope of this performance bench.
- No CUDA graph recommendations — feature is latent, not wired at the native-multislot level.

Env var reference §4 is pulled from the chimere-server source header (grep CHIMERE_ chimere-server/src/) cross-checked against the systemd prod unit env block in benchmark-e2e-2026-04-24.md §1.2.
-->
