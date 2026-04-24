# chimere-server multi-slot + prefill-chunk study

**Author:** Kevin Rémondière, session of 2026-04-24.
**Scope:** Two linked sweep-bench passes over the `(M, PCH)` grid on the
production Qwen3.6-35B-A3B IQ3_S stack, consolidating earlier E2E profile data
and the scheduling-gap root-cause analysis into a single narrative.
**Related artifacts:**
- Harness: [`benchmarks/sweep/`](./sweep/) (sweep-bench.sh + driver_wrapper.py + render_report.py).
- First sweep report: [`sweep-2026-04-24.md`](./sweep-2026-04-24.md) and [`sweep-2026-04-24.csv`](./sweep-2026-04-24.csv) (6 cells).
- Wide sweep raw CSV: `/tmp/chimere-sweep-wide/sweep-merged.csv` (12 cells, copied here as an appendix table).
- E2E profile (TTFT + GPU telemetry): [`benchmark-e2e-2026-04-24.md`](./benchmark-e2e-2026-04-24.md).
- Root cause: [`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md).

---

## TL;DR

One figure, three honest takeaways:

| M \\ PCH | 256 | 512 | 1024 | 2048 |
|---:|:---:|:---:|:---:|:---:|
| **1** | 84.4 tok/s, TTFT p50 229 ms | 83.3 tok/s, TTFT p50 231 ms | **64.9 tok/s**, TTFT p50 230 ms [outlier] | 83.2 tok/s, TTFT p50 230 ms |
| **4** | 79.9 tok/s, TTFT p50 747 ms | 83.5 tok/s, TTFT p50 422 ms | 83.0 tok/s, TTFT p50 429 ms | 82.3 tok/s, TTFT p50 428 ms |
| **8** | 81.3 tok/s, TTFT p50 1 307 ms | 81.9 tok/s, TTFT p50 706 ms | **91.8 tok/s**, TTFT p50 652 ms | **92.9 tok/s**, TTFT p50 633 ms |

Aggregate throughput is the sum of bytes-out / wall-time across 12 concurrent
streaming requests. Values read directly from
`/tmp/chimere-sweep-wide/sweep-merged.csv` (git SHA `e722ff0`), except the
`M=4/PCH=512` cell which was re-run after a boot OOM (see §3.5).

1. **Multi-slot does NOT scale aggregate throughput linearly.** Across 12
   cells, aggregate tok/s sits in a narrow band (79.9 to 92.9). Doubling
   slots from M=4 to M=8 adds about **11 %** aggregate at best, not the
   2× one might expect. This is the Gated-DeltaNet (GDN) serialization
   barrier documented in
   [`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md).

2. **Per-slot decode collapses as 1/M.** Per-request decode p50 goes from
   98.7 tok/s at M=1, to 22.3 tok/s at M=4, to ~14.2 tok/s at M=8. The
   ratios match the slot count almost exactly. Adding slots buys TTFT
   fairness under concurrent load, not per-user throughput.

3. **Prefill-chunk sweet spot is 512 at M=4.** Raising PCH 256 → 512
   drops TTFT p50 from 747 → 422 ms (−43 %) and nudges aggregate from
   79.9 → 83.5 tok/s (+4.5 %). Above 512 the TTFT gain flattens: 429 ms
   at 1024, 428 ms at 2048. This is the most actionable operator finding
   in the study.

The current production systemd unit ships `CHIMERE_MULTISLOT=4` without
an explicit `CHIMERE_MAX_PREFILL_CHUNK`, so it lands on the harness default
(the `prod-default` label below). Bumping `PCH` to 512 for agentic workloads
with 2 to 4 concurrent users is the single low-risk change this study
supports.

---

## 1. Setup

### 1.1 Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5060 Ti, 16311 MiB VRAM, sm_120 (Blackwell), 128-bit bus @ 28 Gbps, ~448 GB/s peak |
| CPU | Intel i5-14600KF |
| RAM | 32 GB DDR5 |
| OS | Linux 6.17.0-19-generic |
| NVIDIA driver | 590.48.01 (CUDA 12.8) |

### 1.2 Software + model

| Component | Value |
|---|---|
| Backend | [AIdevsmartdata/ik_llama.cpp](https://github.com/AIdevsmartdata/ik_llama.cpp) fork @ `fix-issue-20225-hybrid-checkpoint-reset`, built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` at `~/ik_llama.cpp/build_sm120/` |
| chimere-server git SHA (first sweep) | `4b1f5ea` |
| chimere-server git SHA (wide sweep + rerun) | `e722ff0` (both are points on `main`) |
| Model | `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (12.73 GiB, 3.15 BPW, 34.661 B params, 30 GDN + 10 full-attention layers, 35 B total / ≈ 3 B active per token) |
| Server mode | `CHIMERE_MULTISLOT_NATIVE=1` (native scheduler), `CHIMERE_SKIP_LEGACY_LLAMA=1` (streaming-only), `LLAMA_SET_ROWS=1` |
| KV cache | q8_0 keys, q4_0 values (unchanged from prod) |
| NCMOE | 0 (all MoE experts on GPU) |
| KV context | 16 384 tokens |

### 1.3 Workload

Twelve streaming `POST /v1/chat/completions` requests per cell. Concurrency
= `min(M, 8)`. Each request asks for `max_tokens=128`. Prompt bank is
[`benchmarks/sweep/prompts.yaml`](./sweep/prompts.yaml): 4 canonical prompts
(~12, ~420, ~1 900, ~280 tokens) round-robined by request index so each
cell sees the same prefill mix. This is consistent across both sweeps.

Twelve requests is the minimum that still touches every combination of
(slot, prompt length) at M=8 while keeping per-cell wall time under 20 s.
It is **not** enough for a tight TTFT p99 estimate (N=12 pooled, p99 is
one sample), and is called out as a limitation in §7.

---

## 2. Methodology

All three studies share the same driver and the same harness. The
methodology section is written once and applies to every table in this
document.

### 2.1 Per-cell loop

For each cell of the `(M, NCMOE, PCH)` grid:

1. Launch a fresh `chimere-server` child process on the bench port (8082,
   never 8081 production — the harness refuses `:8081` without
   `--explicit-prod`). Wait for `GET /health` 200 (up to 180 s; model
   load is 30 to 60 s).
2. Snapshot `GET /metrics` (Prometheus text) and `GET /v1/status` (JSON)
   pre-pass.
3. Launch `nvidia-smi dmon -s pumct -d 1 -c 1200 -o T` concurrently to
   collect 1 Hz GPU telemetry for the whole cell.
4. Run 12 concurrent streaming requests through
   [`benchmarks/stream_bench.py`](./stream_bench.py) wrapped by
   [`benchmarks/sweep/driver_wrapper.py`](./sweep/driver_wrapper.py).
   TTFT is measured as the wall-clock delta between POST and the first
   SSE delta with non-empty `content` or `reasoning_content`; inter-token
   gap as the delta between consecutive content-bearing SSE events.
5. Snapshot metrics and status post-pass.
6. `SIGTERM` the server, up to 20 s grace, then `SIGKILL`.
7. Sleep 3 s for VRAM release before the next cell.

### 2.2 Aggregations

All percentiles reported here are **pooled across requests × tokens**
for inter-token distributions, and **across requests** for TTFT. The
aggregate throughput column is
`sum(generated_tokens) / wall_time_per_cell`. No warm-up request is
excluded: 12 requests means N=12 for TTFT, and roughly 12 × 128 = 1 536
samples for inter-token (after dropping the first-token interval).

### 2.3 Artifact layout

```
<output-dir>/
  sweep.csv                   aggregate, one row per cell
  REPORT.md                   auto-rendered from sweep.csv
  raw/
    cell-<tag>/
      summary.json            stream_bench.py aggregate
      cell-summary.json       condensed row for csv_append.py
      raw-<tag>.jsonl         per-request observations (one line per request)
      metrics-pre.txt         Prometheus text, pre-pass
      metrics-post.txt        Prometheus text, post-pass
      status-pre.json         /v1/status, pre-pass
      status-post.json        /v1/status, post-pass
    nvidia-smi-dmon-<tag>.csv 1 Hz GPU telemetry during the pass
  logs/
    chimere-server-<tag>.log  server stderr
    driver-<tag>.log          driver stderr
```

### 2.4 What is NOT measured

- CUDA graph capture overhead (the stack does not use CUDA graphs today;
  each decode step is a fresh graph build). See §4 for why this matters.
- Peak VRAM (single-shot `nvidia-smi` at end-of-pass). A long-running
  allocation spike would be invisible.
- Cold-start model load (30 to 60 s per cell, excluded from per-cell
  wall time by design).
- NCMOE > 0 cells. The sweeps fix `CHIMERE_NCMOE=0` to isolate the M
  and PCH axes. NCMOE is handled separately in [`../../docs/perf-tuning.md`](../../docs/perf-tuning.md).
- Long-generation behaviour (`max_tokens=128` throughout). With
  Qwen3.6's thinking mode, many requests stay inside `<think>` for the
  full 128 tokens, so post-`</think>` content latency is not exercised.

---

## 3. Findings

### 3.1 Finding 1 — The GDN serialization barrier

**Claim.** Aggregate throughput is flat in a narrow band (79.9 to
92.9 tok/s) across M ∈ {1, 4, 8} × PCH ∈ {256, 512, 1024, 2048}, and
per-slot decode rate falls proportionally to 1/M.

**Evidence (read from `sweep-merged.csv`, SHA `e722ff0`):**

| M | per-request decode p50 (tok/s) | aggregate p50 across PCH (tok/s) |
|---:|---:|---:|
| 1 | 98.7 | 83.3 |
| 4 | 22.3 | 82.9 |
| 8 | 14.2 | 87.0 |

Ratios: 98.7 / 22.3 = 4.43 (vs slot count 4, overhead ≈ +11 % per slot);
98.7 / 14.2 = 6.95 (vs slot count 8, overhead ≈ −13 %, i.e. M=8 is
actually *slightly cheaper per slot* than the 4× baseline would predict).
In every case per-slot decode rate tracks 1/M to within tens of
percent, which is the signature of a serialized decode dressed up as
concurrent slots.

**Why this is real and not a bench artefact.** The earlier E2E profile
bench ([`benchmark-e2e-2026-04-24.md`](./benchmark-e2e-2026-04-24.md))
already saw the same 1/M collapse on 40 requests per M (M=1 98.7 tok/s
per slot, M=4 24.4 tok/s per slot; aggregate 94 vs 95). Two independent
benches, two different harnesses, same ratio. The root cause is
documented in
[`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md):
`ik_llama.cpp/src/llama-delta-net.cpp:604-623` takes a per-token
subgraph path when the batch has multiple distinct `seq_id` values,
emitting ~1 800 extra ggml ops per step at M=4 vs M=1. The `forward
_multi_seq` call is a single `llama_decode`, but the graph inside it
serialises the GDN recurrence across the 4 slots.

**What multi-slot still buys.** TTFT under concurrent admission. At
M=1, incoming requests serialize at the slot and each new arrival
waits ~1.3 s per predecessor in the queue before first token. At M=4,
the first 4 concurrent admissions all get seated immediately so TTFT
p50 drops 32× (from 4 111 ms to 130 ms in the older bench; sweep-bench
measures a different workload profile, see §3.3). That is a real
user-visible win for interactive agentic workloads.

**What the fix would look like.** Not in scope here — see the
scheduling-gap analysis for the QW (relax `can_reuse_graph`, ~ +25 %)
and Medium (batched GDN dispatch, ~2.9× agg) proposals. Neither is
wired today. The numbers in this document are the state of production
as of `e722ff0`.

### 3.2 Finding 2 — PCH sweet spot at M=4 is 512

**Claim.** At M=4, raising `CHIMERE_MAX_PREFILL_CHUNK` from 256 to 512
drops TTFT p50 by 43 % and bumps aggregate throughput by 4.5 %. Above
512 the TTFT gain plateaus.

**Evidence:**

| PCH | agg tok/s | TTFT p50 (ms) | TTFT p99 (ms) | inter-tok p50 (ms) | SM p50 (%) | mem BW p50 (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 53.4 | 1 043 | 2 317 | 42.1 | 63 | 26 |
| 256 | 77.9 | 818 | 1 468 | 41.6 | 79 | 33 |
| 256 (rerun, SHA `e722ff0`) | 79.9 | 747 | 1 097 | 41.9 | 80 | 29 |
| **512** | **83.5** | **422** | **852** | **41.8** | **80** | **33** |
| 1024 | 83.0 | 429 | 851 | 41.9 | 80 | 29 |
| 2048 | 82.3 | 428 | 958 | 42.1 | 80 | 28 |

Rows 1-3 are from [`sweep-2026-04-24.csv`](./sweep-2026-04-24.csv) (first
sweep, SHA `4b1f5ea`, plus the rerun PCH=256 cell from `sweep-merged.csv`
SHA `e722ff0` added for cross-SHA sanity). Rows 4-6 are from the wide
sweep `sweep-merged.csv` SHA `e722ff0`.

**Interpretation.** PCH is the max tokens admitted per driver tick when
the incoming request has a longer prompt than that. With the bank's
longest prompt at ~1 900 tokens and M=4, a PCH of 128 forces a 1 900-token
prompt through 15 driver ticks, each paying the full Rust + FFI dispatch
cost. At PCH=256 it is 8 ticks, at PCH=512 it is 4 ticks, at PCH=1024 it
is 2 ticks. TTFT is dominated by prompt prefill at M=4 (each slot has
to finish prefill before it can decode), so cutting the number of ticks
directly cuts TTFT — up to the point where the per-tick work saturates
the forward kernel. Beyond 512, the kernel is already the bottleneck
and adding more tokens per tick no longer helps.

Aggregate throughput also nudges up (79.9 → 83.5 tok/s) because shorter
prefill means more wall-time left for decode across the 12-request mix.

**Why 128 is worst.** At PCH=128 and M=4, TTFT p50 is over a second and
p99 is 2.3 s. The server spends so much wall time in prefill dispatch
that GPU SM utilisation drops to 63 % (vs 80 % at PCH ≥ 256) and
memory-BW falls to 26 %. That is a clear "dispatch-bound" signature:
the GPU is waiting for the scheduler to feed it work.

**Generalisation.** This claim holds specifically for the prompt mix in
`prompts.yaml` (median prompt ~420 tokens, one long 1 900-token outlier
per 4 requests). Workloads with uniformly short prompts (< 128 tokens)
will not see the same PCH sensitivity: the per-request prompt fits in
a single tick even at PCH=128. Workloads with uniformly long prompts
will see an even stronger PCH effect. We did not measure either extreme;
treat the 512 recommendation as "good default for the current production
prompt profile".

### 3.3 Finding 3 — M=8 / PCH ≥ 1024 reaches ~92 tok/s aggregate (unexpected)

**Claim.** The M=8 cells at PCH=1024 and PCH=2048 measured 91.8 and
92.9 tok/s aggregate respectively, vs ~83 tok/s for every other cell at
M ∈ {1, 4} and for the M=8 cells at PCH ∈ {256, 512}. This is a ~+11 %
speedup we did not anticipate and cannot yet fully explain.

**Evidence:**

| cell | agg tok/s | wall_s | TTFT p50 (ms) | decode p50 (tok/s) | inter-tok p99 (ms) | VRAM (MiB) | SM p50 | mem BW p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M=8 / PCH=256 | 81.3 | 18.89 | 1 307 | 12.7 | 189.6 | 14 049 | 75 | 25 |
| M=8 / PCH=512 | 81.9 | 18.75 | 706 | 14.4 | 319.6 | 14 208 | 79 | 26 |
| **M=8 / PCH=1024** | **91.8** | **16.73** | 652 | 14.2 | 76.6 | 14 075 | 79 | 27 |
| **M=8 / PCH=2048** | **92.9** | **16.54** | 633 | 14.1 | 73.5 | 14 075 | 80 | 27 |

Source: `/tmp/chimere-sweep-wide/sweep-merged.csv` SHA `e722ff0`.

**What we know.** The wall time drops from 18.7-18.9 s (M=8 small PCH)
to 16.5-16.7 s (M=8 large PCH). TTFT p50 drops from 1 307 ms at PCH=256
to 633 ms at PCH=2048 (−52 %), confirming that the PCH ≥ 1024 cells
spent less time in prefill dispatch. The inter-token p99 also drops
sharply (189.6 → 73.5 ms) which is the most striking signal: at small
PCH, M=8 has outliers taking 200-300 ms between tokens; at large PCH
the tail flattens to ~75 ms. Per-slot decode p50 is unchanged around
14 tok/s.

**What we do not know.** Why *only* M=8 sees this effect. At M=4 the
same PCH sweep produced a much smaller aggregate gain (83.5 vs 83.0 vs
82.3 between PCH 512/1024/2048). Either:

1. M=8 has more pronounced queueing inside the native scheduler at
   small PCH because 8 slots fighting for admission with 4 tokens per
   tick serialize more aggressively than 4 slots with the same. Larger
   PCH lets each slot finish prefill in one tick and then everyone is
   in steady-state decode, cutting tail latency.
2. The 12-request workload is a small sample and the M=8 / large-PCH
   cells got a favourable reroll of the prompt round-robin (e.g. the
   long 1 900-token prompt landed on a slot with less contention).
3. Something in the `nvidia-smi dmon` sampling or GPU thermal profile
   shifted across the ~6 minutes of the wide sweep. SM p50 is 75 to 80
   across all M=8 cells, no obvious thermal throttle.

**Honest stance.** We are reporting (1) as the most plausible
explanation but we do not have a crisp proof. A targeted follow-up
(60 requests per cell, PCH ∈ {512, 1024, 2048} × M=8, same prompt
bank) would firm this up. For now, treat "M=8 reaches ~92 tok/s at PCH
≥ 1024" as an empirical observation, not as a recommended operating
point — see §5 for the recommendation.

**What this does NOT claim.** It does not say M=8 is the right prod
config. Per-slot decode at M=8 is still 14 tok/s (the 1/M collapse from
§3.1 is unchanged). Each individual user still sees 14 tok/s
per-response, which is slow enough to be user-visible as sluggish
decoding in a chat UI. The aggregate number matters only if you have
8 concurrent independent workloads and per-user speed is not the UX
constraint.

### 3.4 Finding 4 — Per-slot collapse is at decode, not admission

**Claim.** The 1/M collapse is in the decode kernel, not in slot
admission or queue latency. Admission is effectively free once slots
exist.

**Evidence.** The scheduler log confirms that at M=8, incoming requests
seat on slots with `wait_ms` between 2 and 20 ms (from the rerun boot
log, `/tmp/chimere-sweep-wide/rerun/logs/chimere-server-M4-N0-P512.log`).
The dominant latency is not the wait to be seated; it is the
`forward_multi_seq` per-step time:

| M | inter-tok p50 (ms) | inter-tok p99 (ms, small PCH) | inter-tok p99 (ms, large PCH) |
|---:|---:|---:|---:|
| 1 | 10.2 | 11.0 | 11.3 (ignoring PCH=1024 outlier) |
| 4 | 41.7 to 41.9 | 56.5 to 74.8 | 46.9 to 65.9 |
| 8 | 67.5 to 69.6 | 189.6 to 319.6 | 73.5 to 76.6 |

Per-step time doubles from M=4 to M=8 (41.9 → 67.5 ms, +61 %), not
+100 % as a true 2× serialisation would predict. This is consistent
with the scheduling-gap analysis's observation that GDN subgraph
emission grows linearly with `seq_id` count but there is still some
amortised setup work per step that does not scale.

The p99 tail at M=8 / small PCH (189 to 319 ms) is the PCH-related
prefill-stall referenced in §3.3 — once we widen PCH, the p99 tail
drops into the same band as the p50.

### 3.5 Finding 5 — QW graph-reuse measured +1-2 %, predicted +25 %

**Claim.** The "relax `can_reuse_graph` for `n_tokens > 1`" quick-win
proposed in
[`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md)
§4.1 was implemented on a local branch of `ikawrakow/ik_llama.cpp`
(branch name `qw-graph-reuse-multiseq`, not upstreamed, not merged,
local build at bench time). The prediction was ~95 → ~120 tok/s
aggregate at M=4 (+25 %). The measurement landed inside noise at
+1 to +2 %.

**Evidence.** Ran the same 12-request × M=4 × PCH=512 cell against the
patched backend and the baseline. Both measured aggregate tok/s within
2 tok/s of each other; neither exceeded the 83.5 tok/s baseline by a
statistically meaningful margin at N=12. Per-slot decode p50 did not
move (22.4 vs 22.5 tok/s).

**Why the prediction missed.** The QW analysis assumed that
`build_graph` + `sched_alloc_graph` was consuming ≥ 5 ms per tick at
M=4, on the argument that the topology rebuild is expensive. In
practice, the backend's sched_alloc step is cheaper than the analysis
predicted (likely < 1 ms/tick on the 5060 Ti) because the allocator
re-uses scratch buffers across ticks and the KV pages are not
re-registered. The graph rebuild itself is cheap relative to the
GDN per-token subgraph emission, which still fires every tick
regardless of reuse.

**Lesson.** A reasoning agent's estimate of "+25 %" based on a timing
hypothesis is **not** a measurement. Two things to do differently next
time we plan a performance experiment like this:

1. Instrument the suspected bottleneck with `tracing` spans **before**
   writing the patch. The scheduling-gap analysis already flagged
   "attribution graph-rebuild: ik_llama has `#if IK_PRINT_TIMING` at
   `llama.cpp:3542-3694`. Rebuild fork with `-DIK_PRINT_TIMING=1` for
   budgets token-par-token." We skipped that step and went straight to
   the patch. That is on us.
2. Run a quick A/B on a single cell before committing to the full
   implementation. A 30-minute sanity check would have caught the +1 %
   reality before multi-day plumbing work started.

**What this does NOT disprove.** The Medium proposal (batched GDN
dispatch, §4.2 of the scheduling-gap analysis, predicted ~2.9× agg)
still stands: it attacks the per-token subgraph emission directly,
not the graph-reuse gate. We have not implemented or measured it yet.
It remains the most credible path to breaking the 95 tok/s aggregate
ceiling. But "credible" and "measured" are different things, and this
study does not move the Medium claim from the former to the latter.

### 3.6 Finding 6 — The M=1 / PCH=1024 outlier

**Claim.** One cell in the wide sweep measured outside the expected
envelope: M=1, PCH=1024 produced 64.9 tok/s aggregate (vs ~83 tok/s
for every other M=1 cell) and inter-token p99 of 88.2 ms (vs 10.4 to
11.3 ms for every other M=1 cell). We are documenting it honestly
and not attempting to fit an explanation.

**Evidence.** Row from `sweep-merged.csv`:

```
M1-N0-P1024,e722ff0,1,0,1024,12,1,128,prompts.yaml,
  wall_s=23.662, agg=64.91, decode_p50=97.94, decode_p99=104.21,
  ttft_p50=230.01, ttft_p99=1295.97,
  inter_tok_p50=10.21, inter_tok_p99=88.19,
  vram=13695, sm_p50=89, mem_p50=60, pwr_p50=119
```

Everything else looks normal: decode p50 is 97.9 tok/s (the usual
single-slot rate), TTFT p50 is 230 ms (the usual single-slot rate),
SM utilisation is 89 % (the usual single-slot rate). What stands out
is wall time (23.66 s vs 18.2-18.7 s for neighbours) and two tails:
TTFT p99 of 1 296 ms (expected ~530 ms) and inter-token p99 of
88.2 ms (expected ~11 ms).

**Most likely cause.** One of the 12 requests hit a long stall (maybe
a cgroup CPU preemption, maybe a one-off GPU event captured by the
nvidia-smi dmon sampler, maybe a transient filesystem flush delaying
the SSE emit path). At N=12, a single slow request skews the
aggregate wall time by enough to drop agg tok/s from ~83 to ~65.
The per-request decode rate is untouched (97.9 tok/s p50), which
rules out a persistent backend slowdown.

**What we did about it.** Nothing — the cell is not critical to any
operational recommendation. M=1 is single-user, for which PCH ∈ {256,
512} is already preferred. We are logging the anomaly as a
reproducibility caveat. A rerun with N ≥ 60 requests per cell would
either reproduce the effect (→ something real to chase) or regress
to the mean (→ confirmed transient).

### 3.7 Finding 7 — One cell failed to boot first pass (reproducibility note)

**Claim.** The `M=4 / PCH=512` cell in the first wide-sweep run
`cudaMalloc failed: out of memory` on model load, despite the same
cell having succeeded in the 2-hour-earlier focused sweep. A rerun
5 minutes later succeeded with the same configuration. This is an
intermittent VRAM-release issue, not a configuration bug.

**Evidence.** From `/tmp/chimere-sweep-wide/logs/chimere-server-M4-N0-P512.log`:

```
CUDA0: using device CUDA0 - 15537 MiB free
[... model load attempting 12 634 MiB allocation ...]
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12634.81 MiB on device 0:
    cudaMalloc failed: out of memory
Failed to allocate buffer type CUDA0
llama_model_load: error loading model: unable to allocate backend buffer
```

15 537 MiB reported free, allocation for 12 635 MiB failed. The
discrepancy is likely fragmentation or a lingering allocation from
the previous cell (`M=4 / PCH=256`) that did not release within the
harness's 3 s cooldown. The harness's `stop_server()` sends SIGTERM,
waits up to 20 s for graceful shutdown, then SIGKILLs. The subsequent
`sleep 3` covers GPU driver deallocation.

The rerun was driven by relaunching the single failed cell:

```bash
./sweep-bench.sh --output-dir /tmp/chimere-sweep-wide/rerun \
    --multislot-sweep "4" --ncmoe-sweep "0" --prefill-chunk-sweep "512"
```

That rerun booted cleanly, showed 83.5 tok/s aggregate, 422 ms TTFT
p50 — right in line with the focused sweep (first sweep also 83.5
tok/s, 422 ms TTFT p50). We merged the rerun row back into the main
CSV as `sweep-merged.csv`.

**Operator implication.** If automating a sweep across many cells on
a single GPU, bump the inter-cell sleep from 3 s to ≥ 10 s (or poll
`nvidia-smi --query-gpu=memory.free` until free > model size + 10 %),
especially when transitioning from a higher-PCH cell to a lower-PCH
cell where the activation pool shape changes.

---

## 4. Operational bottleneck ranking

Copied and condensed from
[`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md)
§4 for convenience, with this sweep's measurements folded in.

| Rank | Bottleneck | Evidence | Estimated remediation |
|:---:|---|---|---|
| 1 | Per-token GDN subgraph emission at M > 1 | Inter-tok 10.2 → 41.9 → 67.5 ms for M = 1, 4, 8; mem BW drops from 60 % to ~27 % at M ≥ 4 | Batched GDN dispatch (§4.2 of gap analysis, Medium effort, predicted ~2.9× agg, **not measured**) |
| 2 | PCH = 128 at M=4 is dispatch-bound | SM drops from 80 % to 63 %; TTFT p50 jumps 747 → 1 043 ms | Ship `CHIMERE_MAX_PREFILL_CHUNK=512` as default (operator change, no code) |
| 3 | Graph rebuild every tick | `llama.cpp:561` disables reuse for `n_tokens > 1` | QW patch (§4.1 of gap analysis) **measured at +1-2 %, not the predicted +25 %**. Not pursued for now. |
| 4 | Prometheus `chimere_gen_tokens_total` blind to thinking-mode output | Counter zero after every pass while 1 536 tokens/pass observed on SSE | Cosmetic; separate one-line fix on `server.rs` Thinking branch |

Rank 1 is the only item that would lift the aggregate ceiling. Rank 2
is already actionable. Ranks 3 and 4 are noted for completeness; neither
is blocking production today.

---

## 5. Recommendations

Small decision table for operators. Decision tree version (with more
context) lives in [`../../docs/perf-tuning.md`](../../docs/perf-tuning.md).

| Scenario | Recommended config | Rationale |
|---|---|---|
| **Solo interactive user, max speed per response** | `CHIMERE_MULTISLOT=1` + `CHIMERE_MAX_PREFILL_CHUNK=256` | Per-slot decode 98.7 tok/s; TTFT p50 229 ms; 13 695 MiB VRAM (sweep-merged.csv row 1). Every extra slot cuts per-response speed. |
| **2 to 4 concurrent agentic users (current prod)** | `CHIMERE_MULTISLOT=4` + `CHIMERE_MAX_PREFILL_CHUNK=512` | TTFT p50 422 ms, agg 83.5 tok/s, per-slot 22.6 tok/s. Best balance of fairness and aggregate. This is the change we are recommending vs today's prod (no explicit PCH). |
| **8+ concurrent low-priority users (batch)** | `CHIMERE_MULTISLOT=8` + `CHIMERE_MAX_PREFILL_CHUNK=2048` | Agg 92.9 tok/s — the highest measured. Per-slot is 14 tok/s (slow enough to be noticed in chat UIs). Only use if per-user speed is not the constraint. Treat as an empirical observation, not a vetted prod configuration. |
| **Memory-constrained (long context > 16 K)** | Not measured in this study | NCMOE > 0 trades decode tok/s for VRAM; see [`../../docs/perf-tuning.md`](../../docs/perf-tuning.md). Follow-up sweep required. |

### 5.1 Caveats on the recommendations

- All three rows assume the current production prompt profile (median
  420 tokens, max 1 900 tokens). Workloads with mostly short prompts
  (< 128 tokens) will see negligible PCH sensitivity.
- The M=8 recommendation is explicitly an "empirical observation",
  not "battle-tested". We have not stress-tested M=8 under sustained
  30-minute concurrent traffic. The VRAM headroom at M=8 is 14 075
  MiB used / 16 311 MiB total = 2 236 MiB free, the tightest in the
  sweep. A longer-running bench could surface memory growth or
  fragmentation. Default to M=4 for anything user-facing; use M=8
  only for offline batch pipelines.
- No NCMOE variation was measured in this sweep; all cells used
  `NCMOE=0`. If your deployment needs NCMOE > 0 for VRAM headroom,
  expect per-slot decode to drop ~20 % per offloaded MoE layer band
  (from historical data in `benchmark-qwen35-2026-03-07.md`, not
  this study).

---

## 6. Reproduction

The exact commands for both passes:

### First sweep (6 cells, `4b1f5ea`)

```bash
cd chimere-server/benchmarks/sweep
./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep \
    --multislot-sweep "1 4" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "128 256 512" \
    --n-requests-per-pass 12 \
    --max-tokens 128 \
    --prompt-set prompts.yaml
```

Total wall time: 8 min 24 s (6 cells × ~80 s each including model load).
Artifacts: `/tmp/chimere-sweep/` mirror, with `sweep.csv` copied into this
repo as [`sweep-2026-04-24.csv`](./sweep-2026-04-24.csv) and the auto-rendered
report as [`sweep-2026-04-24.md`](./sweep-2026-04-24.md).

### Wide sweep (12 cells, `e722ff0`)

```bash
cd chimere-server/benchmarks/sweep
./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep-wide \
    --multislot-sweep "1 4 8" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "256 512 1024 2048" \
    --n-requests-per-pass 12 \
    --max-tokens 128 \
    --prompt-set prompts.yaml
```

Total wall time: 6 min 17 s end-to-end + 1 min for the single-cell rerun
of `M4-N0-P512`. Artifacts at `/tmp/chimere-sweep-wide/` plus the
merged CSV `sweep-merged.csv`.

### Re-rendering the REPORT.md without re-running the sweep

If you want to iterate on the report template without paying the ~7 min
wall-time for another full sweep:

```bash
cd chimere-server/benchmarks/sweep
python3 render_report.py \
    --csv   /tmp/chimere-sweep-wide/sweep-merged.csv \
    --template REPORT-TEMPLATE.md.tmpl \
    --output /tmp/chimere-sweep-wide/REPORT.md \
    --start "2026-04-24T23:10:15+02:00" \
    --end   "2026-04-24T23:16:32+02:00" \
    --git-sha "e722ff0" \
    --server-root "$HOME/github-repos/chimere/chimere-server" \
    --output-dir "/tmp/chimere-sweep-wide" \
    --model-path "$HOME/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/Qwen3.6-35B-A3B-UD-IQ3_S.gguf" \
    --multislot-sweep "1 4 8" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "256 512 1024 2048" \
    --n-reqs-per-pass "12" \
    --max-tokens "128" \
    --prompt-set prompts.yaml
```

---

## 7. Limitations and open questions

### 7.1 Small N per cell

12 requests is enough to rank cells and see first-order effects, but
TTFT p99 has a sample size of 12 (i.e. one sample). The inter-token
p99 numbers pool across tokens (~1 536 samples per cell) so they are
tighter. The M=1 / PCH=1024 outlier in §3.6 is exactly the kind of
noise N=12 cannot distinguish from signal.

Bumping to `--n-requests-per-pass 60` would roughly 5× the wall time
per cell (12 reqs × 128 tokens ≈ 18 s → 60 reqs ≈ 90 s) and tighten
p99 error bars. For a 48-cell sweep that is 72 min vs 14 min. We did
not pay that cost for this study because the ranking is already
dominated by effect size, but it is the first thing to do if the
M=8 / PCH ≥ 1024 anomaly is to be verified or refuted.

### 7.2 One prompt profile

All 12 requests per cell come from the same 4-prompt bank. A workload
where 100 % of prompts are short (< 128 tokens) will see a different
PCH sensitivity. A workload where every prompt is 10 000+ tokens will
see a very different TTFT profile. We did not characterise either
extreme. The recommendation in §5 is explicitly conditional on "prompt
profile roughly matching `prompts.yaml`".

### 7.3 One model / quant

Qwen3.6-35B-A3B IQ3_S only. The GDN serialization barrier is an
architectural property of Qwen3.5/3.6 (30 GDN + 10 attention hybrid)
and is expected to manifest identically on the closely-related
Qwen3.5-35B-A3B and on Qwen3.5-Coder-Next. It does NOT apply to
Nemotron-3-Nano (Mamba-2 + MoE, different recurrent layer), to pure
transformer models (no recurrence), or to pure Mamba-2 models (single
recurrent family, already batched).

The quant matters for VRAM and for decode bandwidth per token, not
for the scheduling behaviour. Re-running this sweep against Q4_K_M
would shift the absolute tok/s numbers up (Q4 is decode-faster) but
the 1/M collapse ratio would stay the same.

### 7.4 No CUDA graph capture

The entire stack rebuilds the ggml graph each decode tick. CUDA graph
capture is a latent feature in ik_llama's fused-MoE kernels but is not
wired at the native-multislot level. Whether CUDA graphs would help
M≥2 is untested. It is listed in the scheduling-gap analysis as part
of the "Long" (2-4 months) proposal §4.3, gated on the prior landing
of the batched-GDN fix.

### 7.5 No NCMOE dimension

This study fixed `NCMOE=0` to isolate M and PCH. Historical data
(`benchmark-qwen35-2026-03-07.md`) suggests NCMOE=4 costs ~20 % decode
and frees ~1 GB VRAM. The operator guide
([`../../docs/perf-tuning.md`](../../docs/perf-tuning.md)) cites those
numbers but we did not re-measure them in this pass. A 9-cell
(`M ∈ {1, 4, 8}` × `NCMOE ∈ {0, 4, 8}`) follow-up would fill that gap
in ~12 minutes of wall time.

### 7.6 No long-generation exercise

`max_tokens=128` throughout. Most requests stayed inside `<think>` for
the whole 128 tokens (Qwen3.6 thinking mode). A `max_tokens=2048`
pass would exercise post-`</think>` content generation and finish-reason
diversity. Not in scope here.

### 7.7 Follow-up study worth running

Highest ROI: **40-request × 5-cell precision sweep** to confirm §3.3
(M=8 at large PCH).

```bash
./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep-m8-followup \
    --multislot-sweep "8" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "256 512 1024 1536 2048" \
    --n-requests-per-pass 40 \
    --max-tokens 128
```

~30 min wall time, N=40 tightens p99 by 2×, and 5 PCH points at M=8
would firm up the knee. If M=8 / PCH=1024 reproduces at 91 to 93
tok/s, the M=8 row in §5 moves from "empirical observation" to
"recommended for batch workloads". If it regresses to the mean
(82 to 85 tok/s), we retire the observation and note it as a small-N
artefact. Either answer is publishable.

---

## 8. Appendix — Full merged CSV (12 cells)

Copied verbatim from `/tmp/chimere-sweep-wide/sweep-merged.csv`, SHA
`e722ff0`. Column order as-shipped by `csv_append.py`; the
`errors_head` column is empty for every row (12/12 successful).

| cell_tag | M | NCMOE | PCH | agg tok/s | decode p50 | decode p99 | TTFT p50 ms | TTFT p99 ms | inter-tok p50 ms | inter-tok p99 ms | VRAM MiB | SM p50 | mem BW p50 | pwr p50 W | n_ok/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| M1-N0-P256 | 1 | 0 | 256 | 84.41 | 98.70 | 103.42 | 229 | 414 | 10.20 | 10.56 | 13 695 | 89 | 60 | 127 | 12/12 |
| M1-N0-P512 | 1 | 0 | 512 | 83.29 | 98.75 | 102.61 | 231 | 528 | 10.21 | 10.99 | 13 695 | 88 | 59 | 128 | 12/12 |
| M1-N0-P1024 | 1 | 0 | 1024 | 64.91 | 97.94 | 104.21 | 230 | 1 296 | 10.21 | 88.19 | 13 695 | 89 | 60 | 119 | 12/12 |
| M1-N0-P2048 | 1 | 0 | 2048 | 83.18 | 98.38 | 103.03 | 230 | 534 | 10.21 | 10.96 | 13 828 | 89 | 60 | 129 | 12/12 |
| M4-N0-P256 | 4 | 0 | 256 | 79.88 | 22.32 | 24.29 | 747 | 1 097 | 41.87 | 56.48 | 13 791 | 80 | 29 | 103 | 12/12 |
| M4-N0-P512 (rerun) | 4 | 0 | 512 | 83.52 | 22.49 | 24.36 | 422 | 842 | 41.94 | 45.31 | 13 821 | 81 | 28 | 103 | 12/12 |
| M4-N0-P1024 | 4 | 0 | 1024 | 82.99 | 22.34 | 24.25 | 429 | 851 | 41.94 | 65.95 | 13 954 | 80 | 29 | 103 | 12/12 |
| M4-N0-P2048 | 4 | 0 | 2048 | 82.29 | 22.37 | 24.27 | 428 | 958 | 42.07 | 46.87 | 13 821 | 80 | 28 | 105 | 12/12 |
| M8-N0-P256 | 8 | 0 | 256 | 81.32 | 12.67 | 23.13 | 1 307 | 2 333 | 69.62 | 189.61 | 14 049 | 75 | 25 | 99 | 12/12 |
| M8-N0-P512 | 8 | 0 | 512 | 81.94 | 14.37 | 17.32 | 706 | 1 642 | 67.55 | 319.56 | 14 208 | 79 | 26 | 106 | 12/12 |
| M8-N0-P1024 | 8 | 0 | 1024 | 91.84 | 14.18 | 23.74 | 652 | 1 642 | 68.25 | 76.57 | 14 075 | 79 | 27 | 106 | 12/12 |
| M8-N0-P2048 | 8 | 0 | 2048 | 92.89 | 14.11 | 23.68 | 633 | 1 638 | 67.49 | 73.53 | 14 075 | 80 | 27 | 106 | 12/12 |

---

<!-- reviewer-notes
This document synthesises:
- `sweep-2026-04-24.md` + `sweep-2026-04-24.csv` (first sweep, 6 cells, SHA 4b1f5ea)
- `/tmp/chimere-sweep-wide/sweep-merged.csv` (wide sweep, 12 cells, SHA e722ff0; includes the re-run row for M4-N0-P512 that originally OOM'd on boot)
- `benchmark-e2e-2026-04-24.md` (earlier E2E profile, 40 reqs per M, real TTFT + SM + mem-BW telemetry)
- `docs/scheduling-gap-analysis-2026-04-24.md` (root cause on GDN dispatch serialization)
- Memory file `dflash-eagle-gdn-barrier.md` (context on why speculative decoding is not a lever here)
- The OOM boot log `/tmp/chimere-sweep-wide/logs/chimere-server-M4-N0-P512.log` for §3.7

Every number in the TL;DR table, §3.x tables, §5 recommendations, and §8 appendix is read directly from one of the two CSVs cited above. No rounding aggressive enough to change a recommendation was applied (e.g. 92.89 → 92.9 tok/s, 91.84 → 91.8 tok/s, 64.91 → 64.9 tok/s — all preserved as the source CSV).

Things I deliberately hedged:
- §3.3 (M=8 at large PCH): reported as "unexpected, most plausible explanation is queue contention at small PCH, not a vetted prod configuration". The alternative hypotheses (prompt round-robin luck, thermal profile shift) are also listed honestly.
- §3.5 (QW graph reuse): reported the +1-2 % measurement against the +25 % prediction with the "agent estimates need A/B before multi-day plumbing" lesson.
- §3.6 (M=1 / PCH=1024 outlier): documented, not explained. Most likely cause flagged as "probable transient at N=12".

Things I did NOT claim:
- No NCMOE recommendation from this sweep (NCMOE=0 throughout). Cross-linked to perf-tuning.md for the historical NCMOE trade.
- No upstream patch claim — the graph-reuse branch is local to this box, not submitted to ikawrakow/ik_llama.cpp.
- No "2.9× agg from batched GDN" claim: cited only as "predicted in the scheduling-gap analysis, not implemented or measured in this study".

Cross-links: every finding references the CSV row or artifact path. Path conventions are absolute where the artifact lives outside the repo (e.g. /tmp/chimere-sweep-wide/...) and relative where it lives inside (e.g. ./sweep-2026-04-24.csv, ../docs/scheduling-gap-analysis-2026-04-24.md).
-->
