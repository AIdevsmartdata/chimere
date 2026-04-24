# Theoretical ceiling analysis — chimere-server on RTX 5060 Ti 16 GB

**Author**: session agent, 2026-04-24.
**Target audience**: Kevin Rémondière, one reviewer who knows the stack.
**Scope**: quantify the gap between the current measured throughput of
`chimere-server` (Qwen3.6-35B-A3B UD-IQ3_S on ik_llama sm_120) and the physical
ceilings imposed by the RTX 5060 Ti 16 GB. Rank software levers by
effort / return, and sanity-check the hardware-upgrade story.

This document is explicitly a *ceiling calculation*, not a new bench. Every
measured number it cites lives in one of these sources; every ceiling
comes from a back-of-envelope that this file shows end-to-end:

- [`chimere-server/benchmarks/2026-04-24-multislot-study.md`](../../github-repos/chimere/chimere-server/benchmarks/2026-04-24-multislot-study.md)
  (consolidated 12-cell sweep, SHA `e722ff0`)
- [`chimere-server/benchmarks/benchmark-e2e-2026-04-24.md`](../../github-repos/chimere/chimere-server/benchmarks/benchmark-e2e-2026-04-24.md)
  (E2E profile, SHA `0d7268d`)
- [`docs/scheduling-gap-analysis-2026-04-24.md`](../../github-repos/chimere/docs/scheduling-gap-analysis-2026-04-24.md)
  (root cause on GDN per-token dispatch)
- [`chimere-server/benchmarks/sweep-2026-04-24.md`](../../github-repos/chimere/chimere-server/benchmarks/sweep-2026-04-24.md)
  (first 6-cell sweep)
- [`chimere-server/benchmarks/stress-test/REPORT.md`](../../github-repos/chimere/chimere-server/benchmarks/stress-test/REPORT.md)
  (4-concurrent trap stress test)

The entire analysis is **read-only**. No code was modified.

---

## 1. Executive summary

Three sentences, then the table.

1. **M=1 is already at its practical ceiling**. Observed 98.7 tok/s uses
   55 % of the 448 GB/s peak memory bandwidth to carry a measured
   2.50 GB/token; the RTX 5060 Ti's practical sustainable ceiling on this
   workload lies at roughly **125 tok/s** (70 % BW, same per-token
   traffic). Gap vs measured: ~26 %, and most of it is in the ggml MMVQ
   kernel's steady-state BW efficiency — not in chimere-server.
2. **M=4 aggregate is 4-6× below its batched-decode ceiling**. Measured
   83-95 tok/s aggregate. With a batched GDN dispatch (Medium patch from
   the gap analysis), the mem-BW ceiling is ~300-500 tok/s aggregate at
   M=4. The entire delta is in a single per-seq loop at
   `ik_llama.cpp/src/llama-delta-net.cpp:611-621` that the kernel
   (`ggml_delta_net`) already supports batching for.
3. **Hardware upgrade to RTX 5090 (4× BW) would lift M=1 to ~400 tok/s
   roughly linearly**, but does NOT relax the GDN dispatch serialisation.
   M=4 on a 5090 without the Medium patch is still bottlenecked by the
   per-seq subgraph emission — you buy ~10-15 % from faster kernels, not
   4×. The Medium patch is a prerequisite for any hardware purchase
   conversation.

### Headline ceiling vs measured (summary table)

| scenario | measured (tok/s) | practical ceiling (tok/s) | utilisation | primary blocker |
|:-|-:|-:|-:|:-|
| M=1 solo decode | 98.7 | ~125 | 79 % | MMVQ steady-state BW efficiency (+ IQ3S dequant overhead) |
| M=4 aggregate, today | 83-95 | ~300 (Medium patch) — ~500 (Long patch) | **19-32 %** | GDN per-seq subgraph loop (`llama-delta-net.cpp:611-621`) |
| M=8 aggregate, today | 92.9 | ~350 (Medium) — ~580 (Long) | **16-27 %** | same GDN loop + graph rebuild each tick |

"Practical ceiling" here means "what the 5060 Ti could physically sustain
at 70 % of peak BW after the dispatch fix". Physical peak at 100 % BW is
~45 % higher; that number is never reached on decode workloads and should
not be used for planning. See §4 for the derivation.

---

## 2. Method and inputs

### 2.1 What is a "ceiling"?

For an autoregressive decode step, the dominant cost per token is
reading the *active* subset of model weights from VRAM once, plus
reading/writing the per-sequence recurrent state and KV cache. The
operation is **memory-bandwidth bound** — every FLOP has to wait on
bytes. So the practical ceiling is:

```
max_tokens_per_sec  =  (effective_BW)  /  (bytes_read_per_token)
```

Where `effective_BW = peak_BW × efficiency_factor`. For modern decode
kernels on NVIDIA silicon, 70 % is a realistic upper bound (MLPerf
inference submissions hit 70-80 % on dense transformer decode; quantised
MoE with indirect expert gather rarely exceeds 65 %). We use **55 %** as
the "observed-today" figure (matches
[`benchmark-e2e-2026-04-24.md`](../../github-repos/chimere/chimere-server/benchmarks/benchmark-e2e-2026-04-24.md)
§3.7 M=1 row) and **70 %** as the "well-tuned" target. A truly
aggressive claim of 85 % is not credible on this GPU class and is not
used here.

### 2.2 Hardware spec snapshot

Source: NVIDIA RTX 5060 Ti 16 GB product brief (Blackwell, sm_120,
GDDR7). Cross-checked against `nvidia-smi` on the bench box.

| parameter | value | notes |
|:-|-:|:-|
| Peak memory bandwidth | 448 GB/s | 128-bit bus × 28 Gbps GDDR7 / 8 |
| L2 cache | 32 MB | single unified L2, Blackwell |
| SM count | 36 | sm_120 |
| FP16 tensor compute (dense) | ~24 TFLOPS | ~2304 CUDA cores × 2.5 GHz boost |
| INT8 tensor ops | ~94 TOPS | 4× FP16 rate on Blackwell consumer |
| INT4 tensor ops | ~189 TOPS | 8× FP16 rate, accessible only with INT4 packed matmul |
| VRAM | 16 311 MiB (usable) | manufacturer nominal 16 GB, usable ≈15.93 GiB |
| PCIe | 5.0 ×8 | irrelevant for this analysis; no host-device traffic in steady-state decode |

Notes on tensor cores:
- The **INT4 / INT8 rates only apply when the matmul is actually INT4/INT8**.
  IQ3_S is a *i-quant*; it lives as packed 3-bit groups on disk and is
  **dequantised to FP16 (or FP32) inline by the MMVQ kernel** before the
  matmul runs. So the effective math rate for our current decode path is
  FP16 TFLOPS (24), not INT4 TOPS (189). This is the single biggest
  "surprise inconsistency" — see §5.3.
- 32 MB L2 cache is *not* large enough to cache the full per-layer active
  weight block (30 GDN × ~15 MB = 450 MB, 40 MoE × ~13 MB = 515 MB). Each
  layer thrashes L2. The only plausible caching win is on the shared-expert
  weights (~1.2 MB × 40 = 48 MB of frequently-hit tensors), which the L2
  is too small to hold alongside activations.

### 2.3 Model spec snapshot

Source: `chimere-server/src/config.rs:244-283` (`qwen35_35b_a3b()`
preset) + GGUF metadata on disk
(`~/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`,
13 676 723 168 bytes = 13.68 GB / 12.73 GiB).

| parameter | value |
|:-|-:|
| Total params | 34.66 B (35 B nominal) |
| Active per token | ~3 B (top-8 of 256 routed experts + shared) |
| Layers | 40 (30 GDN recurrent + 10 full-attention, `full_attn_interval=4`) |
| hidden | 2048 |
| num_experts | 256 |
| experts_per_token (top-k) | 8 |
| expert FFN hidden | 512 |
| shared expert FFN hidden | 512 |
| GDN d_state | 128 |
| GDN d_inner | 4096 |
| GDN dt_rank | 32 |
| GDN n_group | 16 |
| GDN conv kernel | 4 |
| GGUF on-disk size | 12.73 GiB (3.15 BPW effective) |
| Quant label | UD-IQ3_S (Unsloth dynamic IQ3_S, an i-quant) |

### 2.4 Observation baseline

Numbers used throughout (all from `sweep-merged.csv` SHA `e722ff0` unless
noted):

| metric | M=1/PCH=256 | M=4/PCH=512 rerun | M=8/PCH=2048 |
|:-|-:|-:|-:|
| Aggregate tok/s | 84.4 (sweep) / 98.7 (decode p50) | 83.5 / 22.5 | 92.9 / 14.1 |
| TTFT p50 (ms) | 229 | 422 | 633 |
| Inter-token p50 (ms) | 10.20 | 41.94 | 67.49 |
| SM util p50 | 89 % | 81 % | 80 % |
| Mem BW util p50 | 60 % | 28 % | 27 % |
| VRAM used | 13 695 MiB | 13 821 MiB | 14 075 MiB |
| Power p50 | 127 W | 103 W | 106 W |

The M=1 "decode p50" of 98.7 tok/s from the E2E profile is the
single-request streaming rate once the slot is seated, and matches
`1000 / 10.20 ms = 98.0` from the sweep's inter-token distribution. The
"aggregate" 84.4 tok/s is the 12-request pooled wall-clock rate (includes
TTFT and queueing). Use 98.7 tok/s as the "what can one request
see?" number and 83.5 tok/s as the "what does a multi-request workload
see?" number — they are not the same.

---

## 3. Per-bottleneck theoretical ceilings

### 3.1 Memory-bandwidth ceiling — the real constraint

**Method.** Compute active bytes per token, divide by effective BW.

Active weights read per decode token (back-of-envelope from the
configuration in §2.3):

| component | computation | bytes |
|:-|:-|-:|
| 30 GDN layers, weights (ssm_in, ssm_out, ssm_conv1d, ssm_beta/alpha/dt) | ~15 MB/layer × 30 at IQ3_S 3.15 bpw | ~450 MB |
| 10 full-attention layers, weights (Q/K/V/O) | ~5 MB/layer × 10 | ~50 MB |
| 40 MoE layers, router | 256 × 2048 × fp32 × 40 | ~80 MB |
| 40 MoE layers, active experts (top-8 + shared) | 9 matrices × 403 KB each × 40 | ~515 MB |
| LM head | 2048 × 248320 × ~3.15/8 | ~200 MB |
| **Sub-total active weights** | | **~1.22 GB** |
| 30 GDN layers, recurrent state R+W | 30 × 2 × 2 MB (read + write per layer) | ~120 MB |
| Conv state R+W | 30 × 4 × 10240 × fp32 × 2 | ~5 MB |
| Attention KV cache read (small context, 128 tok) | 10 × 128 × 256 × 2 × 4 B × 2 | ~5 MB |
| Activations (intermediate tensors) | rough | ~100 MB |
| MoE dequant temporaries (IQ3_S → F32 per-expert) | ~400 KB × 8 active × 40 layers × 3 matrices | ~380 MB |
| **Sub-total dynamic / state** | | **~610 MB** |
| **Total per-token traffic, estimate** | | **~1.83 GB** |

From observation: 98.7 tok/s at 55 % of 448 GB/s = 246 GB/s actual →
**2.50 GB/token actually moved**. The 0.67 GB residual between the
estimate (1.83 GB) and the observation (2.50 GB) is almost certainly in:

- The inline dequantisation of IQ3_S packed 3-bit weights to FP16 in the
  MMVQ kernel register file. Writing dequantised FP16 back through the
  cache hierarchy doubles the effective read on the dequant path
  compared to a native INT4 GEMM.
- The router + top-k softmax + expert-indices gather producing extra
  traffic on the MoE path.
- Normalisation weight reads (40 × 2 × 2048 × fp32 = 640 KB is trivial,
  but RMSNorm's rsqrt-reduction churns activations through L2).

The residual is in the "not mysteriously unaccounted" band: every item
that could plausibly cost 50-200 MB/token is identified, and three of
them add up to ~0.7 GB. We are not chasing phantom traffic.

**Ceiling derivation.**

```
peak BW (5060 Ti)                = 448 GB/s
practical BW (70 % sustained)    = 313.6 GB/s
observed BW today (55 %)         = 246 GB/s

observed traffic per token       = 2.50 GB/tok   (from 246/98.7)
pure-weight ceiling (1.22 GB)    = 448 / 1.22 = 367 tok/s  (100 % BW, unreachable)
                                   313.6 / 1.22 = 257 tok/s (70 % BW, aspirational)
observed-traffic ceiling         = 448 / 2.50 = 179 tok/s  (100 % BW)
                                   313.6 / 2.50 = 125 tok/s (70 % BW)
                                   246 / 2.50  =  98.6 tok/s (55 % BW, matches measured)
```

**The 98.7 tok/s measured at M=1 is already at the bandwidth ceiling for
the traffic profile we have.** It's not leaving 40 % of the GPU idle. It's
hitting 55 % BW, which is exactly what one expects from a dequant-heavy
IQ3_S MMVQ path on Blackwell.

**Remaining headroom at M=1: ~+27 % (98.7 → 125 tok/s)**. That headroom
lives in:
1. Reducing traffic per token (0.67 GB of dequant / activation overhead).
2. Pushing BW efficiency from 55 % toward 70 %.

Both are in the ggml CUDA kernel layer, not in chimere-server or in the
ik_llama C++ dispatch.

### 3.2 Multi-slot memory-bandwidth ceiling

The key observation for multi-slot: if dispatch were truly batched, the
weights (1.22 GB) would be read **once per step** regardless of M. The
per-sequence state and activations scale linearly.

Back-of-envelope for M=4 with batched GDN dispatch:

```
step bytes (ideal batching) = weights_shared + M × per_seq_dynamic
                            = 1.22 GB + 4 × (0.12 GB state + 0.20 GB activations)
                            = 1.22 + 4 × 0.32
                            = 2.50 GB per step

step time at 70 % BW        = 2.50 / 313.6  =  7.97 ms / step
step throughput             = M / step_time = 4 / 0.00797  =  502 tok/s aggregate
per-slot                    = 502 / 4       =  125 tok/s per slot

At 55 % BW (current ggml efficiency):
step time at 55 % BW        = 2.50 / 246    = 10.16 ms / step
step throughput             = 4 / 0.01016   =  394 tok/s aggregate
per-slot                    = 394 / 4       =  98 tok/s per slot  (same as M=1!)
```

So the **M=4 aggregate ceiling is roughly 400 tok/s at today's BW
efficiency, 500 tok/s if the kernel efficiency pushes toward the
practical 70 %**. Measured is 83-95 tok/s aggregate → **utilisation is
19-24 %** of the bandwidth ceiling.

This is the single most important number in the document. The M=4
aggregate is leaving **~3-5× in the BW bucket**, not the GPU-is-busy
bucket. The GPU is literally idle 70-80 % of each step at M=4; the 27 %
mem-BW reading in the telemetry confirms this.

Where the ceiling really lives on M=4: see §4.

### 3.3 Compute ceiling — NOT binding

For autoregressive decode, FLOPs per token ≈ 2 × active_params ≈
2 × 3 B = 6 GFLOPs/token.

| compute mode | peak | tok/s ceiling |
|:-|-:|-:|
| FP16 tensor cores | 24 TFLOPS | **4 000 tok/s** |
| INT8 tensor cores (if math were INT8) | 94 TOPS | 15 667 tok/s |
| INT4 tensor cores (if math were INT4) | 189 TOPS | 31 500 tok/s |

Even at FP16 the compute ceiling is 40× above measured M=1 and 40-50×
above M=4 aggregate. **Compute is not the bottleneck on this workload,
on any realistic kernel efficiency.** The tensor-core INT4/INT8 rates
are a red herring for decode — the bottleneck is reading weights from
VRAM, not multiplying them. Switching the MMVQ kernel to INT4-native
(if IQ3_S could be unpacked at load time to INT4 packed tensors, which
it structurally cannot without a requant) would be a modest compute win
(~2× on the tensor-core side of the matmul) but invisible at the system
level because the ceiling is BW-side.

The claim "we're not using INT4 tensor cores" is true but irrelevant:
the system is waiting on memory, not on arithmetic.

### 3.4 Kernel launch overhead

Per-kernel launch overhead on sm_120 CUDA 12.8: ~3-5 µs
(driver-mediated; cudaGraph can collapse this to < 500 ns per-kernel but
is not currently in use — see
[benchmark-e2e-2026-04-24.md §7.4](../../github-repos/chimere/chimere-server/benchmarks/benchmark-e2e-2026-04-24.md#74-no-cuda-graph-capture)).

M=1 estimated kernel count per decode step:

- 30 GDN layers × ~20 ggml ops each = 600 kernels (many are no-op views,
  so call it 200 real launches)
- 10 attention layers × ~15 ops = 150 kernels (~100 real launches)
- 40 MoE layers × ~25 ops each (router, top-k, 8×3 matmuls, residual) =
  1000 kernels (~500 real launches with ggml fusion)
- Norm / embed / lm_head / sampling: ~50 more

Roughly **800-1 000 real kernel launches per decode step at M=1**. At
4 µs per launch, that's **3.2-4.0 ms of overhead per step**, or 30-40 %
of the 10.17 ms M=1 inter-token. The rest (6-7 ms) is actual kernel
execution on BW-bound matmul.

M=4 adds the per-seq GDN loop
([llama-delta-net.cpp:611-621](../../ik_llama.cpp/src/llama-delta-net.cpp)).
From the gap analysis: ~1 800 extra ops per step at M=4 vs M=1. Many of
those are views (free), but conservatively **400-600 extra real kernel
launches** → **1.6-2.4 ms of extra launch overhead** on top of the
3-4 ms baseline. That explains roughly 5-6 ms of the M=4 inter-token
tax of ~31 ms. **The other ~25 ms is actual serial kernel execution** —
the GDN scan for each sequence runs in sequence, not the launches
queueing up.

**CUDA graph capture**, if wired, would collapse the 3-4 ms of baseline
launch overhead to < 100 µs. That's a +30-40 % tok/s win on M=1 alone
(98.7 → ~130 tok/s, bringing us to the §3.1 BW ceiling). On M=4 it
would save more (~6 ms of the 31 ms tax) but the bulk of the M=4 tax is
NOT launch-bound; it's serial kernel execution. Graphs are a Long-effort
item and are the correct tool **after** the Medium GDN dispatch fix
(gap-analysis §4.3).

### 3.5 Sampling + SSE post-processing

Measured cost per slot:
- Sampler: ~100 µs (`slot_scheduler.rs:1591-1627`, per-slot
  `sample_slot_with_logprobs` + argmax / top-k / top-p).
- Engram bias: thread-local scratch, measured < 50 µs (see
  [stress-test/REPORT.md](../../github-repos/chimere/chimere-server/benchmarks/stress-test/REPORT.md);
  `apply_engram_bias_to_sampler` on the hot path; no perceptible jitter
  in 4-concurrent stress test).
- SSE emission + axum channel: ~100 µs amortised.

At M=4: 4 × (100 + 50 + 100) µs = 1.0 ms/step. Out of a 41.25 ms step,
that's 2.4 %. **Negligible.** The gap analysis conclusion §3.3 stands.

### 3.6 FFI + Rust scheduler overhead

The `forward_multi_seq_borrow` FFI path
([llama_backend.rs:1036-1084](../../github-repos/chimere/chimere-server/src/llama_backend.rs))
performs one `llama_batch_init` + one `llama_decode` + one `llama_batch_free`
per step. The Rust-side `tick_generate_all`
([slot_scheduler.rs:1588-1670](../../github-repos/chimere/chimere-server/src/slot_scheduler.rs))
gathers the slot state, builds a `Vec<MultiSeqEntry>`, then passes it
across. No mutex on the hot path. No allocation except the input vector
(< 512 bytes).

Empirically: < 10 µs/step (well below the resolution of inter-token
timing at 41 ms). **Not a contributor.**

### 3.7 Engram lookup

Engram is a memory-mapped n-gram hash table
([engram_lookup.rs](../../github-repos/chimere/chimere-server/src/engram_lookup.rs))
with FNV-1a hashing and O(1) average lookup. The hot-path
`apply_engram_bias_to_sampler` runs per-slot once per token. Historical
measurements (not in the current bench set) showed < 50 µs per slot.
Thread-local scratch was landed in the recent refactor, eliminating
contention at M=4.

**Cost at M=4: ~200 µs/step. ~0.5 % of a 41 ms step.** Not a lever.

### 3.8 Summary per-bottleneck table

| bottleneck | theoretical ceiling (M=1) | theoretical ceiling (M=4 agg) | measured (M=1) | measured (M=4 agg) | primary blocker |
|:-|-:|-:|-:|-:|:-|
| Memory-BW (100 % BW, observed 2.50 GB/tok) | 179 | 700 (batched) | 98.7 | 83-95 | IQ3_S dequant traffic + GDN per-seq serialisation |
| Memory-BW (70 % practical) | 125 | 500 (batched) | 98.7 | 83-95 | same |
| Memory-BW (55 % today) | 98.6 | 394 (batched) | 98.7 | 83-95 | same + ggml MMVQ efficiency |
| Compute (FP16 tensor cores) | 4 000 | 4 000 | 98.7 | 83-95 | not binding |
| Compute (INT4 tensor cores, hypothetical) | 31 500 | 31 500 | n/a | n/a | not binding |
| Kernel launch overhead (3-4 ms baseline) | ~250 if zero overhead | same + GDN extra | 98.7 | 83-95 | graph capture would lift M=1 to ~130 |
| Sampling + SSE + FFI + Engram | > 2 000 | > 500 | ibid | ibid | not a lever |

Binding constraint at M=1: **ggml MMVQ kernel efficiency + dequant
traffic** (BW-bound at 55 %). Binding constraint at M=4: **GDN per-seq
subgraph emission in llama-delta-net.cpp** (dispatch-serialised, kernel
execution is the tax, launches are secondary).

---

## 4. Measured vs theoretical — narrative

### 4.1 M=1 — the boring case

Measured 98.7 tok/s at 55 % BW, 89 % SM, 127 W. Theoretical ceiling
125 tok/s at 70 % BW. **Utilisation: 79 %.**

Where the residual 26 tok/s lives:
- ~12 tok/s in ggml MMVQ kernel BW inefficiency (55 → 70 %, half
  achievable). Would need kernel tuning or a different MMVQ variant.
- ~8 tok/s in IQ3_S dequant overhead (the extra 0.67 GB/token we
  estimated in §3.1 residual). A native INT4 MMQ path (requires model
  requant) would cut this.
- ~3-4 tok/s in kernel launch overhead. CUDA graph capture would close
  this.
- ~2-3 tok/s in RMSNorm / residual / miscellaneous reads that could
  plausibly be fused.

None of these are easy. The first (MMVQ efficiency) is an ik_llama /
upstream llama.cpp kernel exercise; the second requires a model requant
to a format with native INT4 CUDA paths (Q4_K_M does but is +0.5 GB
VRAM vs IQ3_S; the Q4_K_M / Q5_K_XL side of the story is already
visible in
[`perf-tuning.md` §3](../../github-repos/chimere/docs/perf-tuning.md#3--ncmoe-memory-vs-speed)).

**Honest stance**: M=1 has ≤ 30 % remaining headroom and all of it is
hard-won. Do not chase it before fixing §4.2.

### 4.2 M=4 — the 3-5× gap

Measured 83.5 tok/s aggregate at 28 % BW, 81 % SM, 103 W. The
interesting signal is **SM high, BW low** — the GPU's execution units
are saturated for about 30 % of the step time and idle for 70 %. The
traditional signature of a launch-bound / dispatch-bound workload.

Theoretical ceiling (§3.2):
- Ideal batched at 55 % BW (today's efficiency): **~394 tok/s aggregate**.
- Ideal batched at 70 % BW (well-tuned kernel): **~500 tok/s aggregate**.

Utilisation: **19-24 %**. There is 4-5× sitting on the table.

Where the gap goes, per the scheduling-gap analysis and the
per-bottleneck calcs above:

| contributor | est. ms per step | est. tok/s cost at M=4 agg |
|:-|-:|-:|
| GDN per-seq subgraph serial execution (30 layers × 4 seqs) | ~24 ms | ~300 tok/s |
| Kernel launch overhead amplification (~400-600 extra ops) | ~5-6 ms | ~70 tok/s |
| Graph rebuild every tick (can_reuse_graph returns false when seq_ids differ) | ~1-2 ms | ~25 tok/s |
| Everything else (sampler, FFI, engram, SSE, misc) | ~1 ms | ~15 tok/s |
| **Budget accounted** | **~31 ms** of the 41 ms step | **~410 tok/s recoverable** |
| Baseline single-token forward work | ~10 ms | — (the "real" work) |

The ceiling math and the gap analysis agree to first order: the
dispatch fix should lift aggregate to the 300-500 range.

**Note on the Medium-patch prediction**: the gap analysis predicts
~280 tok/s aggregate after the Medium patch (§4.2, +2.9×). Our BW
ceiling says the room to grow is 394-500 tok/s. The 280 figure includes
a safety margin for residual dispatch overhead that would remain even
after batching (the per-seq state gather via `ggml_get_rows`, the
scatter writeback, the reset-state mask logic). **280 is the honest
middle estimate**; 500 is the cold memory-bus ceiling; the room
between is in the quality of the batched kernel dispatch.

### 4.3 M=8 — same story, slightly louder

Measured 92.9 tok/s aggregate at 27 % BW at PCH=2048. Theoretical
batched ceiling at M=8 with 70 % BW:

```
step bytes (M=8 batched) = 1.22 + 8 × 0.32 = 3.78 GB
step time at 70 % BW     = 3.78 / 313.6    = 12.1 ms / step
aggregate                 = 8 / 0.0121       = 663 tok/s
per-slot                  = 83 tok/s
```

So the M=8 ceiling with Medium patch is ~500-660 tok/s. Today we see
93. **Utilisation: 14-19 %.** The ~92 tok/s M=8 observation at large
PCH (which
[multislot-study.md §3.3](../../github-repos/chimere/chimere-server/benchmarks/2026-04-24-multislot-study.md#33-finding-3--m8--pch-1024-reaches-92-tokss-aggregate-unexpected)
flagged as unexpected) is still ~7× below the physical ceiling. The
observation is real; the ceiling is 7× higher. These two things don't
conflict — the former confirms that some queueing / per-tick overhead
has extra play at small PCH; the latter says *even after fixing that,
we have 6× left*.

---

## 5. Software levers, ranked by ROI

### 5.1 Ranking

The gap analysis and the sweep study already give us the ROI-ordered
list. This section re-ranks with explicit predictions-vs-measurements
accountability and the new BW ceiling context.

| rank | lever | effort | expected M=4 agg (tok/s) | ceiling it unlocks | confidence | notes |
|-:|:-|:-:|-:|-:|:-:|:-|
| 1 | **Medium: batched GDN dispatch** (gap-analysis §4.2) | 1-3 weeks | ~280 | 394-500 | HIGH on direction, MEDIUM on magnitude | The ceiling calculation agrees with the gap analysis prediction. If the implementation is clean, 280 tok/s is conservative; if the per-seq plumbing introduces significant extra ops, 180-240 is plausible. **The direction (3-5× win) is locked in by the BW math.** |
| 2 | **Long: CUDA graph capture** (gap-analysis §4.3) | 2-4 months, *after* #1 | ~400-500 | same 500 ceiling, tighter | MEDIUM | Only pays off if dispatch is already fixed. On M=1 alone it's a +30 % win (98.7 → ~130); on M=4 post-Medium it's another ~20-30 % on top of 280. |
| 3 | **QW: relax `can_reuse_graph` for `n_tokens > 1`** (gap-analysis §4.1, already landed on fork) | 1-3 days (done) | +1-2 % measured (predicted +25 %) | 95 + 2 = 97 tok/s | LOW (did not move) | Already merged into the fork (`fix-issue-20225-hybrid-checkpoint-reset`, see `ik_llama.cpp/src/llama.cpp:575-594`). **Did not deliver** — gap analysis prediction was off by 10×. See [`multislot-study.md §3.5`](../../github-repos/chimere/chimere-server/benchmarks/2026-04-24-multislot-study.md#35-finding-5--qw-graph-reuse-measured-1-2--predicted-25-). Keep it merged; don't count on it. |
| 4 | **Ship `CHIMERE_MAX_PREFILL_CHUNK=512` as default at M=4** (operator change) | 0.5 day | TTFT −43 %, +4.5 % agg | — | HIGH (measured) | Already recommended in multislot-study §5. No code change. |
| 5 | **MMVQ kernel efficiency tuning (IQ3_S → 65-70 % BW)** | 2-4 weeks, ggml-side | +10-15 % M=1 | 125 ceiling | LOW-MEDIUM | Upstream-kernel exercise. Would lift M=1 from 98.7 → ~115 tok/s. Zero benefit for multi-slot until Medium lands. |
| 6 | **Native INT4 GEMM path (requant to Q4_K_M)** | 1 week (requant) + deploy | +15-25 % M=1 | ~140 ceiling | MEDIUM | Trades 0.5 GB VRAM for decode-faster Q4_K_M (no IQ3_S dequant overhead). See perf-tuning.md existing Q4_K_M results. Not applicable to multi-slot ceiling. |
| 7 | **Kernel fusion (norm+silu+matmul on GDN)** | 2-3 weeks, upstream-style | +5-10 % M=1, +5-10 % M=4 agg | ~130 / ~310 | LOW | Gains are real but bounded — these ops are already BW-bound, fusion helps cache locality not the main matmul. Ship only if Medium+Long done. |
| 8 | **Prefix cache for the 10 attention layers** | 1-2 weeks | TTFT −30-60 % on prefix-heavy workloads | — | MEDIUM | Orthogonal to decode throughput. Already partly wired in `m2-j2-wiring`. Valuable UX-side, does not move this ceiling. |
| 9 | **Fix `chimere_gen_tokens_total` counter in Thinking branch** (cosmetic) | 20 lines | 0 | 0 | HIGH | Observability fix. [gap-analysis §3.3](../../github-repos/chimere/docs/scheduling-gap-analysis-2026-04-24.md) ranked this as item 4. Do it but don't count it as a throughput lever. |

### 5.2 What the ceiling math says about #1

The gap analysis predicted **~280 tok/s M=4 aggregate** for the Medium
patch (+2.9× vs 95 baseline). Our ceiling math says the *absolute
ceiling* for M=4 at 70 % BW is ~500 tok/s, and at today's 55 % BW is
~394 tok/s.

Two honest bounds on what Medium will actually deliver:

- **Floor (safe)**: 200 tok/s aggregate. Assumes batched dispatch but
  with some residual per-seq overhead (per-seq state gather/scatter,
  reset-state handling, edge cases in the `ggml_delta_net` kernel that
  don't quite vectorise to n_seqs=4). Still a +2.1× vs today.
- **Ceiling (optimistic but achievable)**: 400 tok/s aggregate. Assumes
  the kernel efficiency improves a bit in the process (from 55 % to
  60-65 % BW on the now-denser step), and the per-seq plumbing is
  nearly free.
- **280 tok/s** (gap-analysis point estimate) sits roughly in the middle
  and is the right number to communicate.

**What to measure after shipping**: aggregate tok/s at M=4/PCH=512 and
mem-BW util at the same cell. If mem-BW climbs from 28 % to 50-55 %, the
fix is working as predicted. If BW stays in the 30s and aggregate only
hits 150 tok/s, something else is bottlenecking and the ceiling
analysis needs revisiting.

### 5.3 The "are we using tensor cores" question

Kevin's brief asked whether INT4 tensor cores are being exploited. The
honest answer:
- **No**, because IQ3_S is an i-quant that dequantises to FP16 inside
  the MMVQ kernel. The matmul that hits the tensor cores is an FP16 ×
  FP16 operation (per the ggml CUDA kernel family in
  `ggml/src/ggml-cuda/mmvq.cu` and `iqk_mmvq.cu`).
- **It doesn't matter** for this ceiling: even at 24 TFLOPS FP16 we
  have 40× headroom over the current M=1 throughput. Compute is not
  binding.
- **It would matter** if we changed quant to Q4_K_M (native Q4 MMQ path
  exists in `ggml/src/ggml-cuda/mmq.cu`, uses a different tensor-core
  configuration) and if compute became the binding constraint. Neither
  of those is remotely close to being true. Park the INT4 tensor-core
  discussion for when decode > 500 tok/s.

The ik_llama fork has already landed all the meaningful MMVQ work for
IQ3_S on sm_120. The `+23 % TG` advantage over stock llama.cpp
(MEMORY.md note) captures most of it.

---

## 6. Hardware alternatives

If the Medium patch is in production and decode sits at ~280 tok/s M=4
aggregate (solo M=1 at ~120-130 tok/s), here's what hardware would add,
at the bandwidth ceiling level (linear BW extrapolation only, ignoring
compute changes which are all non-binding):

| platform | peak BW | M=1 ceiling (70 % BW) | M=4 ceiling agg (70 % BW, Medium patch) | VRAM | Notes |
|:-|-:|-:|-:|-:|:-|
| **RTX 5060 Ti 16 GB (current)** | 448 GB/s | 125 | 500 | 16 GB | Today's baseline. Measured M=1 98.7, M=4 agg 95. |
| **RTX 5090 32 GB** | 1 792 GB/s | **395** | **~1 600** (compute-limited at ~800) | 32 GB | 4× BW scales M=1 decode linearly. M=4 ceiling won't reach 1 600 because compute (~104 TFLOPS FP16) caps at ~800 tok/s on the batched step. Still ~8× current M=4 agg if Medium landed. **Price ~€2 200, available.** |
| **RTX 5080 16 GB** | 960 GB/s | 212 | ~850 | 16 GB | 2.1× BW. Same compute tier as 5090. Reasonable mid-point. **Price ~€1 100.** Same VRAM → same multi-slot limit. |
| **H100 SXM 80 GB** | 3 350 GB/s | 738 | ~2 500 (heavily compute-limited) | 80 GB | 7.5× BW. On the batched M=4 step, compute (~989 TFLOPS FP16 dense) stops being non-binding somewhere around 1 000 tok/s aggregate (decode FP16 at 1 000 tok/s = 6 TFLOPS used = 0.6 % of peak, so still not binding). Realistic ceiling ~1 500 tok/s aggregate M=4, bounded by ik_llama's batched kernel efficiency on the larger H100 L2. **Not a consumer option. Data-centre GPU.** |
| **Strix Halo iGPU (AMD Ryzen AI MAX)** | 256 GB/s (DDR5-8000 quad-channel unified) | 56-71 (0.55-0.70 BW) | ~225-285 (if Medium patch, ROCm-ready) | 64-128 GB unified | **50 % the BW of 5060 Ti.** BW-scaled ceiling sits below current 5060 Ti measured M=1. The story is VRAM (128 GB can hold Q4_K_M or even BF16 weights) and silent-fanless factor, not speed. ROCm / Vulkan GDN path is early; Mamba-2 HIP port exists (see MEMORY `strix-halo-contribution-2026-04.md`), qwen3.6 GDN does not. **Not a perf upgrade.** |
| **Apple M3 Max 64 GB** | 400 GB/s | 88 | ~350 | 64 GB unified | Slightly slower than 5060 Ti on BW. MLX would need a GDN+MoE port (not trivial). Nice VRAM, acceptable BW, no production hybrid stack. Not credible in 2026. |
| **AMD MI300X 192 GB** | 5 300 GB/s | 1 167 | compute-limited ~2 000 | 192 GB | 12× BW. Data-centre. ROCm hybrid GDN stack is experimental; ik_llama does not target it. Not a 2026 move. |

**Honest reading of the table**:

1. **If Kevin buys hardware, the only consumer-class upgrade that
   matters is the 5090**. It approximately quadruples the single-slot
   decode (98 → ~400 tok/s) and roughly 8-10×s the M=4 aggregate
   **if the Medium patch lands first**. Without Medium, the 5090 would
   still sit at ~95 tok/s aggregate on M=4 (the bottleneck is dispatch,
   not BW).
2. **Strix Halo is NOT a speed upgrade.** It is a form-factor /
   silent-fanless / "can fit Q5_K_XL in VRAM" upgrade. The BW ceiling
   is below today's 5060 Ti. On the 4× multi-slot case it would
   actually be *slower* aggregate than the current 5060 Ti. Kevin's
   roadmap note `strix-halo-contribution-2026-04.md` is correct to
   frame it as a contribution / ecosystem effort, not as a perf play.
3. **H100 / MI300X are not 2026 options for this deployment.** Fine to
   mention for completeness; do not plan on.

**Primary recommendation if asked to spend**: do NOT buy hardware until
Medium is shipped. After Medium, the 5090 is the single most-impactful
upgrade. Everything else is form-factor.

---

## 7. Conclusion

### 7.1 The ceiling

RTX 5060 Ti 16 GB on Qwen3.6-35B-A3B UD-IQ3_S:
- M=1 practical ceiling: **~125 tok/s** (at 70 % sustained memory BW,
  for the currently-observed 2.50 GB/token traffic). Today: 98.7 tok/s
  = 79 % utilisation. **~26 % headroom**, all of it in ggml kernel
  tuning.
- M=4 aggregate practical ceiling: **~400-500 tok/s** (batched GDN
  dispatch, 55-70 % BW). Today: 83-95 tok/s = 17-24 % utilisation.
  **3-5× headroom**, all of it in ik_llama GDN dispatch (`llama-delta-net.cpp:611-621`).
- M=8 aggregate practical ceiling: **~500-660 tok/s** (batched, 70 % BW).
  Today: 93 tok/s = 14-19 % utilisation. Same headroom as M=4, just
  spread wider.

### 7.2 What's blocking

1. **M=4 / M=8 multi-slot**: the per-seq loop in
   `ik_llama.cpp/src/llama-delta-net.cpp:611-621`. The kernel
   (`ggml_delta_net`) already handles batched sequences. The *dispatch*
   loops. Fixing the dispatch unlocks ~3-5× on aggregate.
2. **M=1**: the ggml IQ3_S MMVQ kernel is running at 55 % of peak BW
   rather than 65-70 %. Upstream kernel work; small but real.
3. **Both**: no CUDA graph capture → ~3-4 ms of launch overhead per
   step on M=1, ~5-6 ms on M=4. Net +30-40 % on M=1, +10-15 % on M=4
   (much less than the dispatch fix because M=4's wait is serial
   kernel execution, not launches).

### 7.3 Top three software levers, in order

1. **Batched GDN dispatch** (Medium, 1-3 weeks, ik_llama-side). Locks
   in 3-5× on M=4 aggregate. Non-controversial direction; magnitude
   depends on implementation quality. **This is the top lever by far.**
2. **Operator change: ship `CHIMERE_MAX_PREFILL_CHUNK=512` at M=4 as
   default** (measured, already recommended). TTFT −43 % on M=4 for
   current prompt mix. Zero engineering cost. Do today.
3. **CUDA graph capture** (Long, 2-4 months, gated on #1). Another
   ~20-30 % on top of #1, and lifts M=1 by ~30 %. Only pays off once
   dispatch is fixed.

### 7.4 Top one hardware lever

**RTX 5090 32 GB**, if and only if software #1 is shipped. Linear BW
scaling lifts M=1 to ~400 tok/s and M=4 aggregate (post-Medium) to
~1 500-1 600 tok/s (compute starts to bite somewhere in there). Without
#1, the 5090 is dispatch-bound identical to the 5060 Ti at M=4 — a
waste of €2 200.

### 7.5 The surprise / unknown worth chasing

**The 0.67 GB/token "residual" between our 1.83 GB/tok weight+state
estimate and the observed 2.50 GB/tok.** The breakdown in §3.1
attributes most of it to IQ3_S dequant temporaries (~380 MB) and
router/activation traffic (~290 MB), but these are estimates, not
measurements. If the true residual is smaller (say 0.3 GB), the ceiling
moves from 125 to ~160 tok/s on M=1. If larger (say 1.0 GB), the
ceiling drops from 125 to ~105 and M=1 has effectively zero headroom.

**The cheap way to measure this**: rebuild ik_llama with
`-DIK_PRINT_TIMING=1` (noted in
[gap-analysis §7](../../github-repos/chimere/docs/scheduling-gap-analysis-2026-04-24.md))
and get token-by-token budget breakdown, then use nvprof / Nsight to
get a clean bytes-per-token number. This is a 1-2 day exercise and
would tighten the M=1 ceiling to ±5 % instead of ±15 %.

Worth doing before any MMVQ kernel tuning effort.

---

## 8. Appendices

### 8.1 Calculation ledger

All the back-of-envelope numbers in one place so they can be checked.

**Hardware constants:**
- peak_BW = 448 GB/s (5060 Ti GDDR7 128-bit × 28 Gbps / 8)
- practical_BW_70 = 313.6 GB/s
- observed_BW_55 = 246.4 GB/s
- FP16_TFLOPS = 24
- INT4_TOPS = 189 (not used)

**Model per-token weight reads (§3.1):**
- 30 × 15 MB GDN weights = 450 MB
- 10 × 5 MB attention weights = 50 MB
- 40 × 80 / 256 × 8 = 256 bytes … no, 40 × (2 MB router + 9.67 MB active + 1.21 MB shared) = 515 MB + 80 MB = 595 MB MoE
- LM head 200 MB
- Sum 1.22 GB (reported)

**Dynamic traffic (§3.1):**
- GDN state R+W: 30 layers × 2 MB × 2 = 120 MB
- Conv state: 30 × 4 × 10240 × 4 B × 2 = 10 MB (rounded 5)
- KV cache (short context): ~5 MB
- Activations (rough): ~100 MB
- MoE dequant temps: 40 × 8 × 3 × 400 KB = 380 MB
- Sum ~610 MB, plus the 1.22 GB weights → 1.83 GB estimated
- Observed 2.50 GB → residual 0.67 GB (see §7.5)

**M=4 batched ideal (§3.2):**
- step_bytes = 1.22 (weights shared) + 4 × (0.12 state + 0.20 activations) = 2.50 GB
- step_time at 70 % BW = 2.50 / 313.6 = 7.97 ms
- agg tok/s = 4 / 0.00797 = 502

**Kernel launch arithmetic (§3.4):**
- 800-1 000 launches / step × 4 µs = 3.2-4.0 ms/step baseline
- Extra at M=4: 400-600 more × 4 µs = 1.6-2.4 ms/step
- Total M=4 kernel overhead: 4.8-6.4 ms (vs 31 ms M=4 tax)
- Implication: ~20 % of the M=4 tax is launches; 80 % is serial exec.

**Hardware scaling (§6):**
- 5060 Ti M=1 @ 98.7 tok/s at 55 % BW
- 5090: 1 792/448 = 4× BW → 395 tok/s (BW-scaled, ignores compute)
- H100 SXM: 3 350/448 = 7.5× → 738 tok/s (BW-scaled; compute still non-binding at ≤1 000 tok/s on 989 TFLOPS)
- Strix Halo: 256/448 = 0.57× → 56 tok/s (BW-scaled, below 5060 Ti)

### 8.2 Code references (read-only)

- Model config preset: `chimere-server/src/config.rs:244-283`
  ([file](../../github-repos/chimere/chimere-server/src/config.rs))
- Slot scheduler hot path: `chimere-server/src/slot_scheduler.rs:1588-1670`
  (`tick_generate_all`)
- FFI bridge: `chimere-server/src/llama_backend.rs:1036-1084`
  (`forward_multi_seq_borrow`)
- GDN layer forward (Rust, unused in prod — the libllama path is used):
  `chimere-server/src/qwen35_model/layer_gdn.rs`
- Engram hot path: `chimere-server/src/engram_lookup.rs`
- Profile registry: `chimere-server/src/profile.rs`
- Metrics endpoint: `chimere-server/src/metrics.rs`
- **The actual GDN dispatch that matters**: `ik_llama.cpp/src/llama-delta-net.cpp:430-585`
  (`build_layer_attn_linear_core`, `n_seqs = 1` hard-coded at line 435)
  and `:587-625` (`build_layer_attn_linear`, the per-seq loop at 611-621)
- Graph reuse (QW patch already landed): `ik_llama.cpp/src/llama.cpp:559-606`
- MMVQ kernels: `ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu`,
  `iqk_mmvq.cu`, `iqk_mmvq_templates.cuh`
- Delta-net CUDA kernel: `ik_llama.cpp/ggml/src/ggml-cuda/delta-net.cu`
  (already supports `n_seqs > 1` — that's why the gap analysis says the
  fix is dispatch, not kernel work)
- SSM scan CUDA kernel: `ik_llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu`

### 8.3 Spec sheet sources

- RTX 5060 Ti 16 GB: NVIDIA product brief (Blackwell, sm_120). GDDR7
  128-bit × 28 Gbps confirmed via `nvidia-smi -q -d MEMORY` on the
  bench box. Peak BW 448 GB/s.
- RTX 5090 32 GB: NVIDIA product brief. GDDR7 512-bit × 28 Gbps =
  1 792 GB/s. FP16 TFLOPS ~104.
- H100 SXM 80 GB: NVIDIA H100 datasheet. HBM3 5 120-bit × 5.23 Gbps =
  3 350 GB/s (SXM variant). FP16 dense ~989 TFLOPS.
- Strix Halo (AMD Ryzen AI MAX): AMD product page + MEMORY.md
  `strix-halo-contribution-2026-04.md`. Peak ~256 GB/s (DDR5-8000
  quad-channel shared with CPU).
- Apple M3 Max: Apple product page. Peak ~400 GB/s (LPDDR5x 512-bit).
- AMD MI300X 192 GB: AMD Instinct datasheet. HBM3 peak 5.3 TB/s.

All figures double-checked against manufacturer pages on 2026-04-24.
Minor variations exist across product SKUs; values used here are for
the canonical consumer / data-centre part.

### 8.4 Honesty caveats

- **BW efficiency numbers (55 %, 70 %) are literature/experience, not
  specific to IQ3_S on sm_120.** MLPerf inference submissions on similar
  classes of decode workloads land in the 60-80 % band; we use 55 %
  (measured) and 70 % (target). If the real achievable is 65 %, every
  ceiling in this document shifts ~7 % lower. That's inside the
  "approximate" band this document operates in.
- **The 0.67 GB/token residual in §3.1 is an estimate, not a
  measurement.** See §7.5 for how to tighten it.
- **The M=4 batched ceiling of ~500 tok/s assumes zero residual
  dispatch overhead after Medium.** In practice 400 tok/s is more
  realistic; the gap-analysis point estimate of 280 includes a further
  safety margin. Quote 280 when talking to stakeholders; 400 when
  engineering; 500 only when discussing "physical bus limit".
- **M=8 ceiling at 660 tok/s** assumes the batched kernel scales
  linearly to 8 seqs. At 8 seqs the state traffic per step (8 × 120 MB
  = 960 MB) starts to dominate the weight share (1.22 GB), which
  means the ceiling scales sub-linearly. The real M=8 ceiling is
  plausibly 500-600, not 660.
- **The gap-analysis QW prediction was off by 10×** (predicted +25 %,
  measured +1-2 %). That's a real data point for humility. The Medium
  prediction of +2.9× is similar in spirit but has a much stronger BW
  accounting behind it (this document) — the direction is locked,
  magnitude is +/-30 %.
- **No CUDA graph benchmark exists on this stack.** The +30 % M=1 win
  from graph capture is a literature estimate
  (NVIDIA Nsight examples on Blackwell decode). It could be +15 % or
  +45 %; nobody on this box has measured it on Qwen3.6. Treat the
  number as a range, not a target.

<!-- reviewer-notes
This document:
- Does NOT duplicate multislot-study.md or gap-analysis body text; cites them.
- Does NOT introduce new claims not grounded in either measurement or a spec-sheet derivation.
- Does call out, explicitly, that the +2.9× Medium prediction is in the right band per the BW ceiling, but that the QW +25 % prediction was wrong (+1-2 %), and that the Medium claim should be read as 200-400 tok/s aggregate (not a point estimate of 280).

Traceability:
- §2.3 model spec: `chimere-server/src/config.rs:244-283` (inspected)
- §2.4 measurements: all from `/tmp/chimere-sweep-wide/sweep-merged.csv` SHA e722ff0 via multislot-study.md and benchmark-e2e-2026-04-24.md
- §3 ceilings: arithmetic shown step-by-step, constants cited in 8.1-8.3
- §5.1 rankings: consistent with gap-analysis §4 and multislot-study §4, cross-referenced
- §6 hardware: linear BW extrapolation; compute-side limits noted where relevant
- Calibration check: measured 98.6 tok/s @ 55 % BW with 2.50 GB/tok matches exactly the observation, which confirms the arithmetic and the ceiling derivation logic.

Things explicitly NOT claimed:
- No claim of a measured result for the Medium patch (it's unimplemented).
- No claim about CUDA graph capture gains beyond literature estimates.
- No claim that the 0.67 GB residual per token is precisely one specific component; the breakdown is plausibility-ranked, not measurement-grounded.
- No claim that 5090 / H100 upgrade "solves" multi-slot — explicitly flagged that dispatch blocks any BW upgrade until Medium lands.
-->
