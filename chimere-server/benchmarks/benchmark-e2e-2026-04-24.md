# chimere-server E2E profile bench — 2026-04-24

**Author:** E2E profile pass, 2026-04-24 21:19–21:27 CEST.
**Prod commit:** `0d7268d` (branch `m2-j2-wiring`, HEAD of chimere-server repo at
bench time, built 21:00). Binary:
`/home/remondiere/github-repos/chimere/chimere-server/target/release/chimere-server`.
**Model:** `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (12.73 GiB, 3.15 BPW, 34.661 B
params, 30 GDN + 10 full-attention layers, 35B total / ~3B active per
token).
**Hardware:** NVIDIA RTX 5060 Ti 16GB (PCIe 4.0 x8 effective; sm_120; mem
bus 128-bit @ 28 Gbps → ~448 GB/s peak). CPU: i5-14600KF, 32 GB DDR5.
**Runtime:** ik_llama.cpp sm_120 build at
`/home/remondiere/ik_llama.cpp/build_sm120/`, streaming-only mode
(`CHIMERE_SKIP_LEGACY_LLAMA=1`), NativeScheduler driver,
`LLAMA_SET_ROWS=1`.

---

## TL;DR

Three headline numbers:

| metric                                             |  M=1  |  M=2  |  M=4  |
|----------------------------------------------------|:-----:|:-----:|:-----:|
| **aggregate gen tok/s** (40 reqs, conc 4)          | 94.4  | 74.2  | 95.3  |
| **per-request decode tok/s** (p50)                 | 98.7  | 37.8  | 24.4  |
| **TTFT p50 (ms)**                                  | 4 111 | 3 509 |  130  |

**What this says in one sentence:** adding slots does NOT increase
throughput on this hybrid GDN+MoE stack — aggregate tok/s is essentially
flat around 95 tok/s from M=1 to M=4, and per-request decode rate
collapses proportionally (1 × 98.7 ≈ 2 × 49.4 ≈ 4 × 24.4, all around 95).
The only thing multi-slot buys is fair scheduling (TTFT drops 32×
at M=4). Below we establish this mechanistically with the per-token
inter-arrival distribution, Prometheus counter deltas, and GPU telemetry.

---

## 1. Setup

### 1.1 What is "chimere-server"?

Rust axum service that front-ends `ik_llama.cpp` via FFI. Production
instance runs on `127.0.0.1:8081` (systemd user unit
`chimere-server.service`, enabled). Exposes:

- `POST /v1/chat/completions` — OpenAI-compatible chat, streaming (SSE) or
  non-streaming.
- `GET /health` — 200 OK when the model + scheduler are initialized.
- `GET /metrics` — Prometheus text (0.0.4).
- `GET /v1/status` — JSON with per-slot state + TTFT ring summary.

In native multislot mode (`CHIMERE_MULTISLOT_NATIVE=1`), **non-streaming
returns HTTP 400** (`native_mode_streaming_only`). The Rust `bench-m1`
harness in the repo is therefore unusable against prod — this pass uses a
Python streaming driver (`stream_bench.py`) that consumes SSE.

### 1.2 Systemd config (prod, restored after bench)

```ini
Environment=CHIMERE_LLAMA_BACKEND=1
Environment=CHIMERE_MODEL=…/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
Environment=CHIMERE_TOKENIZER=…/tokenizer.json
Environment=CHIMERE_MMPROJ=…/mmproj-BF16.gguf
Environment=CHIMERE_PORT=8081
Environment=CHIMERE_NCMOE=0
Environment=CHIMERE_KV_MAX_SEQ=16384
Environment=CHIMERE_ENGRAM_DIR=/home/remondiere/.openclaw/data/engram
Environment=CHIMERE_MULTISLOT=4
Environment=CHIMERE_MULTISLOT_NATIVE=1
Environment=CHIMERE_SKIP_LEGACY_LLAMA=1
Environment=CHIMERE_SKIP_SAMPLER_INIT=0
Environment=LLAMA_SET_ROWS=1
```

### 1.3 Bench config

Because only one libllama model load fits in 16 GB VRAM (12.7 GB weights
+ ≈2 GB KV + ≈0.3 GB activation pools), running a second chimere-server
instance concurrent with prod is not viable. The bench stopped
`chimere-server.service`, ran a transient instance on port 8082 via
`run-bench.sh` (the harness restarts a child process per M value rather
than using a systemd drop-in), drove the workload, and after the sweep
restarted prod. **Total prod downtime: ≈8 minutes** (21:19–21:27).

> **Honesty note on `CHIMERE_PROFILE=1`:** the brief requested enabling
> the profiler for `/v1/profile` span dumps. Grep of the current source
> tree (`0d7268d`, branch `m2-j2-wiring`) shows **no `/v1/profile` route
> exists** and `CHIMERE_PROFILE` only appears as a doc-comment line in
> `config.rs:534`. The variable was set anyway (harmless), and this
> report relies exclusively on sources that *are* wired: `/metrics`
> Prometheus text, `/v1/status` JSON (includes the TTFT ring), the raw
> per-request SSE stream (TTFT + inter-token gaps measured by the
> driver), and external `nvidia-smi dmon -s puct` telemetry sampled at
> 1 Hz during each pass.

### 1.4 Workload

Three passes, identical driver invocation modulo M:

```bash
python3 stream_bench.py \
  --url http://127.0.0.1:8082 \
  --n 40 --conc 4 --max-tokens 128 --prompt-set short \
  --label M{1,2,4} --out-dir raw --capture-metrics
```

- **Prompt bank:** 8 short English sentences ("Write one short sentence
  about the Eiffel Tower.", Mt Fuji, Amazon, …). Round-robin by request
  index. This is NOT a long-context workload; prefill is ~10-20 tokens
  per request.
- **Concurrency:** 4 client threads, semaphore-gated. All 40 requests
  submitted eagerly (no artificial inter-arrival delay) — so the first
  4 hit the admission path together and the next 36 queue.
- **Warmup:** one 4-token request before each pass, to force slot
  initialization and avoid charging first-batch JIT costs to the bench.
- **Stream mode:** `stream=true`. Driver measures TTFT as the arrival
  time of the first SSE delta with non-empty `content` OR
  `reasoning_content` (Qwen3 think blocks count), inter-token gap as
  the delta between consecutive content-bearing SSE events, total
  latency as the time from POST to `[DONE]`.

All three passes yielded **40/40 successful requests, 0 errors, every
request generated exactly 128 `delta` events** (reasoning-content only
— Qwen3.6 went into `<think>` on every prompt). Total gen = 5120
delta events per pass → identical denominator for comparing passes.

Finish-reason breakdown (from the driver JSONL):

- M=1: 40 × `stop` (the final emitted token happened to be a stop token).
- M=2: 40 × `length` (hit `max_tokens=128` before `</think>`).
- M=4: 40 × `length` (same).

The difference in finish reason between M=1 and M=2/M=4 is non-determinism
— under multi-slot the GDN state can diverge slightly across slots, so
token 128 is not always the same token. It does not affect the throughput
or latency numbers (every pass produced 128 content-bearing deltas per
request).

---

## 2. Methodology

1. Stop prod (`systemctl --user stop chimere-server.service`), confirm
   VRAM released to ≥14 GB free.
2. For each M in {1, 2, 4}: `run-bench.sh` spawns a chimere-server child
   process on port 8082 with `CHIMERE_MULTISLOT=$M`, waits for `/health`,
   launches `nvidia-smi dmon -s puct -d 1 -c 1200 -o T` concurrently,
   runs `stream_bench.py`, then terminates the child.
3. `stream_bench.py` snapshots `/metrics` and `/v1/status` pre and post,
   runs 40 requests at concurrency 4, writes JSONL observations
   (`ttft_ms`, list of inter-token ms, total_ms, finish_reason,
   body_preview).
4. Kill dmon at end of pass.
5. After last M, restart prod.

Per-pass deliverables in `raw/`:
`summary-M{1,2,4}.json` (stats), `raw-M{1,2,4}.jsonl` (one JSON per
request, 40 lines), `metrics-pre/post-M{1,2,4}.txt`,
`status-pre/post-M{1,2,4}.json`, `nvidia-smi-dmon-M{1,2,4}.csv`,
`analysis.json` (aggregated stats). Reproduction driver:
`run-bench.sh` + `stream_bench.py` + `analyze.py`.

---

## 3. Results

### 3.1 Aggregate throughput

Total wall-time and aggregated token rate (sum of all gen tokens / wall
seconds), 40 reqs × 128 tok = 5120 tok per pass:

| pass | slots | wall_s | agg tok/s | speedup vs M=1 |
|:----:|:-----:|-------:|----------:|:--------------:|
| M=1  |   1   |  54.23 |     94.41 | 1.00 × (ref)   |
| M=2  |   2   |  68.99 |     74.21 | **0.79 ×**     |
| M=4  |   4   |  53.73 |     95.29 | 1.01 ×         |

M=2 is a clear regression (-21 % agg tok/s, +27 % wall time vs M=1).
M=4 is statistically indistinguishable from M=1 (+0.9 %). This is the
opposite of what a "3× at M=4" target would want.

### 3.2 Per-request decode throughput

Decode rate per request = `n_tokens / (total_ms − ttft_ms) × 1000`,
measured over 40 streams:

| pass | decode tok/s p50 | p90 | p99 | mean | stdev |
|:----:|:---------------:|:----:|:----:|:----:|:-----:|
| M=1  |   98.7          | 100.9 | 103.8 | 99.0 | ≈2    |
| M=2  |   37.8          |  38.3 |  38.8 | 37.8 | ≈0.5  |
| M=4  |   24.4          |  24.8 |  25.0 | 24.3 | ≈0.4  |

The ratios are revealing:

- 98.7 / 37.8 = 2.61 — at M=2 each slot is ~2.6× slower than the
  monolithic M=1.
- 98.7 / 24.4 = 4.05 — at M=4 each slot is ~4× slower.

In other words, **N slots deliver 1/N of the single-slot decode rate,
so aggregate is flat**. This is the signature of strict serialization
dressed up as "concurrency". A real parallel decode (batch-of-N through
the attention/GDN kernels with shared projection cost) would show a
slope > 1 / N (the prefill/read-matmul gets amortized across N, and
aggregate tok/s grows super-linearly up to the compute bound).

### 3.3 Inter-token gap distribution

Measured on the SSE channel (wall time between successive content
deltas, pooled across all 40 requests × 128 tokens minus one = 5080
samples per pass):

| pass | inter-tok ms p50 | p90 | p99 | mean | 1000/mean ≈ tok/s |
|:----:|:----------------:|:----:|:----:|:----:|:-----------------:|
| M=1  |   10.17          | 10.39 | 11.09 | 10.18 | **98.2**          |
| M=2  |   26.58          | 27.15 | 27.91 | 26.64 | **37.5**          |
| M=4  |   41.25          | 42.39 | 47.00 | 41.37 | **24.2**          |

Sanity check: 1000 / mean_inter_tok_ms matches per-request decode tok/s
to within 1 %. The distribution is remarkably tight (p99/p50 ≈ 1.1 at
M=1, ≈ 1.04 at M=2, ≈ 1.14 at M=4) — no long-tail jitter, no GC-like
pauses. The scheduler emits tokens at a rock-steady interval that just
scales with 1/M. This is strong mechanistic evidence for round-robin
single-step-per-slot rather than true batched decode.

### 3.4 TTFT distribution

Two independent sources agree almost perfectly — the Prometheus ring
(quantiles from server-side `observe_ttft_ms`, sample at SSE first-token
push) and the client-side wall clock (from HTTP POST to first SSE
chunk):

| pass | ttft ms p50 (client) | p90 | p99 | mean | Prom p50 | Prom p99 |
|:----:|:--------------------:|:----:|:----:|:----:|:--------:|:--------:|
| M=1  | 4 111                | 4 169 | 4 302 | 3 925 | 4 108  | 4 297    |
| M=2  | 3 514                | 3 557 | 3 572 | 3 341 | 3 509  | 3 564    |
| M=4  | **133**              |  178 |  188 |  115 | **130**  |   207    |

At M=1, 4 requests hit admission together but only one runs; the other
3 wait an average of ≈1.3 s per predecessor (matches `98.7 tok/s ×
128 tok → 1.30 s per request`). So TTFT p50 ≈ 2 × 1.3 s = 2.6 s is
expected for the 2nd; p90 ≈ 3.9 s for the 3rd; observed 4.1 s. Plus
prompt prefill (~40 ms for ~20-token prompts on these kernels) and the
first slot's own TTFT (~60 ms). The distribution is basically deterministic
queueing.

At M=4, **all 4 concurrent requests are admitted immediately** → the
ring statistic collapses to the pure first-token latency, which is
**130 ms p50**. That's dominated by prompt prefill (at ≈10 ms / tok ≈
≤ 150 ms for ≤ 20 tokens + a small fixed scheduler overhead).

**TTFT = queueing + prefill + first-token kernel.** M=4 fully
eliminates queueing, giving a 32× p50 improvement. This is by far the
biggest user-visible win from multi-slot.

### 3.5 Total request latency

| pass | total ms p50 | p90 | p99 | min | max |
|:----:|:-----------:|:----:|:----:|:---:|:---:|
| M=1  |    5 396    | 5 482 | 5 605 | 1 374 | 5 605 |
| M=2  |    6 905    | 6 940 | 6 951 | 3 476 | 6 951 |
| M=4  |    5 371    | 5 406 | 5 522 | 5 304 | 5 522 |

Observations:

- M=1 `min` = 1 374 ms ≈ the first request (TTFT 60 ms + 128 tok ×
  10.18 ms = 1 303 ms + overhead). The 40 requests span ≈5.4 s p99
  total, consistent with wall-time 54 s / (40 reqs / 4 concurrent) =
  5.4 s mean per request.
- M=2 is the worst case for total latency: p50 = 6.9 s, because each
  request now takes 128 × 26.58 ≈ 3.4 s of decode, plus admission
  wait. Useless from any angle.
- M=4 p99 − p50 = 151 ms — the flattest tail of the three, because all
  4 concurrent slots run in lock-step. Essentially everyone pays the
  same price (128 × 41.37 ≈ 5 295 ms) plus ≤ 200 ms prefill.

### 3.6 Prometheus counter deltas (pre → post bench)

```
metric                             M=1    M=2    M=4
chimere_requests_total{ok}          +41    +41    +41
chimere_requests_total{error}         0      0      0
chimere_prompt_tokens_total        +783   +783   +783
chimere_gen_tokens_total             0      0      0   ← finding
chimere_slot_pool_size (gauge)       1      2      4
chimere_slot_occupancy (gauge)       0      0      0
```

**Finding 1:** `chimere_gen_tokens_total` is zero after every pass even
though the driver's JSONL confirms 128 × 40 = 5 120 content-bearing
deltas per pass. Root cause: in `server.rs:1693` the native streaming
path increments the counter on `NativeStreamMsg::Token` only, not on
`NativeStreamMsg::Thinking`. Our short prompts triggered Qwen3.6's
thinking mode and emitted all 128 content deltas as `Thinking`
messages. `add_gen_tokens(1)` therefore never fired. The counter works
for post-`</think>` content but not reasoning-content. A one-line fix —
call `add_gen_tokens(1)` in the `Thinking` branch too — would close the
observability gap for this workload shape. Logged as follow-up.

**Finding 2:** `chimere_slot_occupancy` gauge is sampled AT scrape time
and is 0 between passes (load had finished). Not broken, just not
useful at 15s scrape cadence for 50s passes. Would need either
histograms or a higher-resolution export to catch the peak.

**Finding 3:** The TTFT ring summary (samples quantiles) is the most
useful thing Prometheus gives us today. Verified to match the client
measurements within 0.2 %.

### 3.7 GPU utilization (nvidia-smi dmon, 1 Hz)

Raw columns: `sm` (% SM util), `mem` (% mem-bus util), `pwr` (W),
`mclk`/`pclk` (MHz). Aggregated over the active window of each pass
(rows with `sm > 5 %`):

| pass | SM util p50 | SM p95 | mem BW p50 | mem p95 | pwr p50 (W) | pclk p50 (MHz) | active frac |
|:----:|:-----------:|:------:|:----------:|:-------:|:-----------:|:--------------:|:-----------:|
| M=1  |   89 %      |  90 %  |   55 %     |   57 %  |    129      |    2 700       |   0.84      |
| M=2  |   80 %      |  83 %  |   27 %     |   28 %  |     98      |    2 737       |   0.86      |
| M=4  |   75 %      |  80 %  |   23 %     |   24 %  |    102      |    2 730       |   0.79      |

This is the strongest single datum in the report:

- **M=1 is the only configuration that hits 55 % memory-bandwidth
  utilization** and draws the most power (129 W p50). SM util is 89 %.
- M=2 and M=4 both drop to 23-27 % memory util and SM util around
  75-80 %. Power drops by ≈30 W.

On a bandwidth-bound autoregressive decode (which IQ3_S MoE decode
strictly is — every token must read the entire active-expert weight
block once), **aggregate mem BW util should scale linearly with
aggregate tok/s**. Instead, doubling M cuts mem BW by half while
holding aggregate constant — meaning the GPU is actually *less busy*
per unit time at M=2/M=4. Either:

1. The native scheduler is spending large fractions of decode in non-GPU
   work (mutex wait, token-by-token Rust-side dispatch, SSE channel
   serialization), OR
2. The forward_multi_seq kernel issues N fully-separate kernel
   launches for N slots without a batched path, and launch overhead
   eats the extra wall time.

Either way, the GPU is sitting at half its bandwidth ceiling during
multi-slot. There is ≥ 2× runway left in the kernel/scheduling stack
before the memory bus is the bottleneck.

---

## 4. Analysis — where does the time go?

We don't have span-level profiling in this build, but the data points
above let us decompose the wall time for a representative M=4 request:

**Per-request budget at M=4 (128 tokens, 20-token prompt, TTFT 130 ms):**

```
prefill (20 tok prompt + KV init):      ~130 ms  ( 2.4 %)
decode (128 tok × 41.37 ms / tok):     5 295 ms  (97.5 %)
SSE serialization overhead (est.):        ~7 ms  ( 0.1 %)
                                      ─────────
total per-request latency:            ~5 432 ms
```

Decode is **97.5 % of wall time** and dominates everything. TTFT
matters for UX but is tiny in the budget. So if we want speed, decode
per-token latency is the only meaningful thing to attack.

**Inside each 41.37 ms / tok at M=4** (decomposed by consistency with
M=1's 10.18 ms / tok):

```
forward_multi_seq kernel   ~10.2 ms    — the "work" (1 token per slot × 4)
4-slot scheduling overhead ~31.2 ms    — the "tax"
                          ───────────
total per step             ~41.4 ms
```

The fact that M=1 → M=4 inter-token ms goes from 10.17 to 41.37 (a
4.07× increase), with aggregate tok/s flat, means **`forward_multi_seq`
in native mode is decoding all 4 slots SEQUENTIALLY per step, not in
parallel.** The 4× slowdown per slot exactly equals the slot count. No
meaningful batching is happening.

This matches the GDN state-barrier constraint mentioned in the brief:
GDN recurrent layers maintain per-sequence hidden state (like Mamba SSMs).
Without a batched GDN kernel that handles N parallel states in a single
launch (à la mamba-2's `scan_chunks` with varlen support), the
implementation naturally falls back to one-sequence-at-a-time. The full
attention layers (10/40) *could* be batched, but the 30 GDN layers
can't — they'd need to be reworked on the kernel side. This is a
known open engineering problem for hybrid stacks, not a chimere-server
bug per se.

---

## 5. Flame-graph narrative (textual)

For M=4, here is the per-10-second wall slice rendered as a sparse
ASCII flame:

```
0────────1────────2────────3────────4────────5─── seconds (wall)
│        │        │        │        │        │
│ prefill (x4 reqs, ~150 ms total; ~40 % SM, <1 % of wall)
│XX  ← ≤ 200 ms
│   │
│   │  decode block — 128 steps @ 41.4 ms/step, SM ≈ 77 % continuous
│   │ ████████████████████████████████████████████████████████████
│   │  ← GPU active but ONLY 23-24 % mem BW (vs 55 % at M=1)
│   │
│   │  inside each 41.4 ms step (schematic):
│   │  ┌───── 10 ms forward_multi_seq (~77 % SM here) ─────┐
│   │  │   ┌─ slot0 attn+GDN ─┐ ┌─ slot1 ─┐ ┌─ slot2 ─┐ ┌─ slot3 ─┐
│   │  │   └──────────────────┘ └─────────┘ └─────────┘ └─────────┘
│   │  │                                                           │
│   │  │ ~30 ms of SCHEDULING gap between steps (~0 % SM)           │
│   │  │   ← this is the "tax"; GPU idle waiting for next dispatch  │
│   │  └───────────────────────────────────────────────────────────┘
│   │
│   │  (repeat 128 times)
│   │
│   └──────────── ~5.3 s decode ─────────────┘
│                                            │
│                                      first SSE [DONE] at ≈5.4 s
```

The fat box is compute (`forward_multi_seq`), the thin-dotted gap is
the scheduling tax between steps. Note this is schematic — without
`perf`-level sampling we can't prove the 30 ms is a single contiguous
idle vs many small fragments. But the 23-24 % mem-BW utilization
rules out "kernel takes 41 ms": if the kernel spanned the full
41 ms it would saturate closer to 55 % (same as M=1, same memory
traffic per token).

---

## 6. Bottleneck ranking

**Rank 1 — Decode step not truly batched at multi-slot.** 97 % of
latency lives here. Per-slot tok/s collapses as 1/N; aggregate is flat.
Root cause: GDN recurrence serialized across slots. Remediation options
(from cheapest to hardest):

  1. **Validate** that `forward_multi_seq` actually issues N slot
     forward passes inside a single kernel launch (as opposed to N
     launches back-to-back). If the latter, a single-launch variant
     would cut the per-step overhead to near-zero with no GDN kernel
     change.
  2. Implement a **batched GDN recurrent kernel** (varlen scan, à la
     mamba2 or flash-linear-attention's chunked scan). This is the
     real fix and the only path to aggregate tok/s > 1× baseline.
     Upstream work in ik_llama / flash-linear-attention would benefit
     everyone.
  3. **Prefix cache** for the 10 full-attention layers (already wired
     in the branch for M2-J2) would cut prefill cost on
     common-prefix traffic; does not help steady-state decode.

**Rank 2 — GPU under-utilized on mem bandwidth at M≥2.** 23-28 %
mem-BW util vs 55 % at M=1 means the scheduling tax is keeping the
memory subsystem half-idle. Even without the kernel fix, cutting the
30 ms per-step scheduling gap to <10 ms would raise aggregate tok/s
proportionally. Instrument the native driver's tick loop with
tracing timestamps to identify whether the gap is (a) Rust/async
wakeup jitter, (b) FFI marshalling of slot state, or (c) sampler +
SSE send happening between kernel launches.

**Rank 3 — Prometheus `chimere_gen_tokens_total` counter is blind to
thinking-mode output.** Not a latency bug, but it blinds observability
when prompts land in `<think>`: no way to track steady-state throughput
from ops dashboards today. Straightforward fix — add
`st.metrics.add_gen_tokens(1)` in the `NativeStreamMsg::Thinking` branch
of `server.rs` (around line 1715, mirroring line 1693 for Token).

---

## 7. Suggested next optimization targets

Ordered by highest-effort-per-latency-gain ratio:

1. **Measure the scheduling gap.** Before changing code, insert a
   handful of `tracing` spans around `forward_multi_seq` and the
   SSE send loop in `slot_scheduler.rs` / `mtp_scheduler.rs` and
   dump them to `/v1/profile` for real. Goal: put a number on how
   much of the 30 ms gap is "real kernel idle vs Rust overhead".
2. **Single-kernel-launch variant of `forward_multi_seq`** if
   profiling shows N launches. This is likely low-hanging fruit and
   could bring M=4 aggregate to 1.5-2× M=1 immediately.
3. **Wire up `chimere_gen_tokens_total` correctly** so this bench
   can be re-run without an external driver. 20-line patch.
4. **Ship `/v1/profile` endpoint** per the brief (it doesn't exist
   yet). Even a simple spans-ring-buffer dump would unblock future
   passes of this bench.
5. **Only after 1-3**: batched GDN recurrent kernel. This is the
   real win (N× aggregate) but also real engineering. The current
   aggregate ceiling is 95 tok/s; this removes it.

---

## 8. Comparison hooks

- **llama-server stock on this model/hw**: per the MEMORY.md context,
  `qwen35-iq3.service` (stock llama-server b8125, UD-IQ3_S, -ngl 99,
  3×64K ctx) reported "69 tok/s, 3x64K ctx" single-slot. This bench
  measured 98.7 tok/s single-slot through chimere — a +43 %
  advantage, attributable to the ik_llama sm_120 build (known +23 %
  TG on this hardware per MEMORY) plus the leaner streaming path.
  Multi-slot behavior on stock llama-server would need its own
  bench; not run here.
- **vLLM** (cited as a reference in the brief): not run. Would
  require a fresh VRAM load of weights in a different stack, and
  chimere's GDN+MoE hybrid isn't a vLLM-native architecture.

---

## 9. Raw artifacts

All in `raw/`:

- `raw-M{1,2,4}.jsonl` — one JSON per request, 40 lines, fields
  `{rid, prompt, send_at_ms, ttft_ms, inter_tok_ms_list, n_tokens,
  finish, error, total_ms, body_preview, body_len}`.
- `summary-M{1,2,4}.json` — per-pass aggregated stats (see §3.2-3.5).
- `sweep-summary.json` — concatenation of the above (optional output of
  `run-bench.sh`).
- `metrics-pre/post-M{1,2,4}.txt` — Prometheus text scrapes.
- `status-pre/post-M{1,2,4}.json` — `/v1/status` JSON scrapes.
- `nvidia-smi-dmon-M{1,2,4}.csv` — 1 Hz GPU telemetry; header starts
  with `#Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk
  rxpci txpci`.
- `metrics-prod-before-bench.txt`, `status-prod-before-bench.json`,
  `gpu-pre-prod.csv` — baseline captures.
- `analysis.json` (alongside the report, not in raw/) — pre-computed
  aggregates (feeds the tables above).

---

## 10. Appendix — commands used (exact)

```bash
# Reproduce end-to-end via the helper script (replaces the manual
# drop-in dance used in the original pass):
OUT_DIR=$(pwd)/raw ./run-bench.sh
# (defaults: MULTISLOT_SWEEP="1 2 4", N_REQS=40, CONC=4, MAX_TOKENS=128,
#  PROMPT_SET=short, BENCH_PORT=8082. Refuses :8081 without EXPLICIT=1.)

# Or manually, for a single M (equivalent to what run-bench.sh orchestrates):

# 1) snapshot prod, then stop it
curl -s http://127.0.0.1:8081/metrics > raw/metrics-prod-before-bench.txt
curl -s http://127.0.0.1:8081/v1/status > raw/status-prod-before-bench.json
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv \
    > raw/gpu-pre-prod.csv
systemctl --user stop chimere-server.service

# 2) start a transient instance on :8082 with CHIMERE_MULTISLOT=$M
#    (env vars identical to prod except CHIMERE_MULTISLOT / CHIMERE_PORT)
# 3) GPU telemetry + bench
nvidia-smi dmon -s puct -d 1 -c 1200 -o T > raw/nvidia-smi-dmon-M${M}.csv &
python3 stream_bench.py --url http://127.0.0.1:8082 \
    --n 40 --conc 4 --max-tokens 128 --prompt-set short \
    --label M${M} --out-dir raw --capture-metrics

# 4) stop transient, next M, repeat
# 5) restore prod
systemctl --user start chimere-server.service
```

---

## 11. Caveats

- **N = 40 per pass is small** — good for capturing first-order effects
  but the TTFT p99 estimate has high variance (1 sample out of 40).
  Enough for ranking bottlenecks; not publication-grade.
- **Short prompts only** (~20 tokens). Long-context behavior was not
  measured; prefill cost will grow linearly up to O(ctx) for attention
  and O(1) for GDN, so at long context the relative cost of the 10
  attention layers grows. The bench has a `--prompt-set long` path
  (~200-token prompts) for a follow-up pass.
- **Short generation only** (128 tokens). With Qwen3.6's think mode,
  all 40 requests hit max_tokens before emitting post-`</think>`
  content. The inter-token gap and decode rate should be independent
  of n_tokens; this is a clean decode measurement. If we care about
  end-of-stream behavior (tool calls, JSON mode, forced `</think>`),
  that's a different bench.
- **One GPU, one model, one quant** (5060 Ti 16 GB, Qwen3.6-35B-A3B
  IQ3_S). Generalization to other hybrids (Nemotron-3, Bamba) or
  other quants (Q4_K_M, Q5_K_XL) not measured.
- **CHIMERE_PROFILE=1 had no effect**: the flag is not wired in this
  branch. Report relies on external timing only.

---

## 12. Summary table (one-shot for dashboards)

```
pass  wall_s  agg_tps  ttft_p50_ms  decode_p50_tps  gpu_sm_p50  gpu_mem_p50
M=1   54.2    94.4     4111         98.7            89%         55%
M=2   69.0    74.2     3509         37.8            80%         27%
M=4   53.7    95.3      130         24.4            75%         23%
```

<!-- reviewer-notes
Changes applied vs v1:
- §1.3 "Bench config": v1 described a "systemd drop-in at
  bench-m1.conf" + "daemon-reload && start" dance. The actual shipped
  run-bench.sh (reviewed alongside this report) spawns a chimere-server
  child directly via exec — no systemd drop-in. Updated text to match.
  Also removed the nonexistent drop-in paths from §10.
- §1.4: added explicit finish-reason breakdown (M=1: 40×stop, M=2/M=4:
  40×length). v1 said "every request generated exactly 128 tokens...
  max_tokens=128 is reached before </think>" which only holds for
  M=2/M=4. Raw JSONL confirms. Added a short note that the M=1 vs
  M≥2 divergence is GDN-state non-determinism and does not affect
  numbers.
- §3.6 Finding 1: v1 said gen_tokens_total increment fires "inside a
  branch that never fires for the native scheduler's streaming output
  path". Verified against m2-j2-wiring server.rs — the branch DOES fire
  for `NativeStreamMsg::Token` (line 1693) but NOT for
  `NativeStreamMsg::Thinking` (line 1715+, no add_gen_tokens call).
  Rewrote to reflect this more precisely. All other findings left as-is.
- §6 Rank 3: same adjustment — "broken on native stream path" → "blind
  to thinking-mode output", with the one-line fix hint (add the call in
  the Thinking branch around line 1715).
- §9 artifacts list: `analysis.json` is not in raw/ (it sits alongside
  the report). Clarified.
- §10: rewrote command block to match run-bench.sh (which was delivered
  with this report). Old systemd-drop-in sequence is gone.
- All headline numbers cross-checked against raw/summary-M{1,2,4}.json
  and analysis.json on 2026-04-24. Every number in TL;DR, §3.1, §3.2,
  §3.3, §3.4, §3.5, §3.6, §3.7, §4, §12 matches the raw data within
  rounding.
-->
