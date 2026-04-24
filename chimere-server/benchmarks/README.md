# chimere-server benchmarks

This directory holds the benchmark studies, the reproducible harness, the
CSV+Markdown artifacts, and the raw per-request observations for
`chimere-server`'s performance work. Everything here is designed to be
re-run on any machine with the same GPU class (sm_120 on a 16 GB card),
the same quant family (IQ3_S Qwen3.6-35B-A3B or a sibling), and the
chimere-server + ik_llama.cpp sm_120 build tree as documented in the
top-level [README.md](../../README.md).

Performance work in this repo is organised around three tiers of
investigation and a single shared harness. Read the tier descriptions in
§1, then pick the study in §2 that matches your question.

---

## 1. Methodology tiers

We split bench work into three tiers because the cost of a run grows
with tier and the precision grows in the same direction. Do the cheap
tier first, always.

| Tier | Wall time | Requests per cell | Cell count | Purpose |
|---|---:|---:|---:|---|
| **Smoke** | 5 to 10 min | 12 to 20 | 3 to 6 | "Does the server come up and serve traffic at expected orders of magnitude?" Used to validate a fresh build or a new kernel before paying for a real bench. |
| **Focused sweep** | 15 to 30 min | 12 to 40 | 6 to 12 | Targeted at one axis (e.g. PCH alone, or M alone with NCMOE fixed). Enough cells to trace a knee; enough samples to rank. Error bars still wide on p99. |
| **Wide sweep** | 60 to 120 min | 12 to 40 | 24 to 48 | Multi-axis (M × PCH × NCMOE). Publishable shape, but not publication-grade p99. Reserve for a quiet GPU window. |

Each tier uses the same harness [`sweep/sweep-bench.sh`](./sweep/sweep-bench.sh);
only the CLI flags change.

A second class of bench lives alongside: **deep-profile benches**
([`run-bench.sh`](./run-bench.sh) + [`stream_bench.py`](./stream_bench.py)) —
40 requests per M at fixed PCH, with `nvidia-smi dmon` and Prometheus
scrapes. Used when you need GPU telemetry and tight per-token distributions,
not grid shape. These are the benches that produced
[`benchmark-e2e-2026-04-24.md`](./benchmark-e2e-2026-04-24.md).

---

## 2. Studies in this directory

Files are dated from the day the data was captured.

### 2.1 Current reports

| File | Date | Tier | Git SHA | Headline |
|---|---|---|---|---|
| [`2026-04-24-multislot-study.md`](./2026-04-24-multislot-study.md) | 2026-04-24 | Consolidated (two focused sweeps + E2E profile) | `4b1f5ea` and `e722ff0` | 12-cell `M × PCH` grid on Qwen3.6-35B-A3B IQ3_S. Prefill-chunk sweet spot at 512 for M=4 (agg 83.5 tok/s, TTFT p50 422 ms). |
| [`sweep-2026-04-24.md`](./sweep-2026-04-24.md) | 2026-04-24 | Focused sweep (6 cells) | `4b1f5ea` | Auto-rendered from the first sweep-bench pass, `M ∈ {1, 4} × PCH ∈ {128, 256, 512}`. Superseded in narrative by `2026-04-24-multislot-study.md` but kept as the raw first-pass artifact. |
| [`sweep-2026-04-24.csv`](./sweep-2026-04-24.csv) | 2026-04-24 | Focused sweep (6 cells) | `4b1f5ea` | Source CSV for the first sweep. Machine-readable. |
| [`benchmark-e2e-2026-04-24.md`](./benchmark-e2e-2026-04-24.md) | 2026-04-24 | Deep profile | `0d7268d` | 40-request passes at M ∈ {1, 2, 4}, PCH default, with full TTFT / SM / mem-BW / Prometheus telemetry. First mechanistic evidence of the GDN serialization barrier. |
| [`analysis.json`](./analysis.json) | 2026-04-24 | Deep profile | `0d7268d` | Aggregated JSON for `benchmark-e2e-2026-04-24.md`. |
| [`raw/`](./raw/) | 2026-04-24 | Deep profile | `0d7268d` | Per-request JSONL + pre/post `/metrics` + `/v1/status` scrapes + `nvidia-smi dmon` CSVs for the E2E pass. |

### 2.2 Historical

| File | Date | Notes |
|---|---|---|
| [`../../benchmarks/benchmark-qwen35-2026-03-07.md`](../../benchmarks/benchmark-qwen35-2026-03-07.md) | 2026-03-07 | First comprehensive Qwen3.5-35B-A3B characterisation. Source of the single-slot 80 tok/s prod number referenced in the top-level README. |
| [`../../benchmarks/bench_np1_96k_20260309_032953.md`](../../benchmarks/bench_np1_96k_20260309_032953.md) | 2026-03-09 | Single-slot 96 K context bench (`np=1`). KV-cache sweet spot measurement. |
| [`../../benchmarks/chimere-perf-phases-2026-03-19.md`](../../benchmarks/chimere-perf-phases-2026-03-19.md) | 2026-03-19 | Multi-phase performance review during the MTP port work. |
| [`../../benchmarks/hf-lora-scan-qwen35.md`](../../benchmarks/hf-lora-scan-qwen35.md) | 2026-03 | LoRA scanning benchmark for the engram/training pipeline. |
| [`../../benchmarks/quantification-sota-layer-by-layer.md`](../../benchmarks/quantification-sota-layer-by-layer.md) | 2026-03 | Per-layer quantization SOTA survey. |

The `chimere-server/benchmarks/` files (this directory) are the
"runtime" benchmarks; the `chimere/benchmarks/` files (one level up)
are the "model + quant" benchmarks. Both are tracked in git.

---

## 3. The `sweep/` harness

A llama-bench-style grid harness for chimere-server. Targets the three
knobs that actually move the needle on the hybrid GDN+MoE stack:

| Env var | Short name | What it controls |
|---|---|---|
| `CHIMERE_MULTISLOT` | M | Number of concurrent slots the native scheduler arms. 1 = single-slot (no scheduler). |
| `CHIMERE_NCMOE` | NCMOE | First N layers' MoE experts offloaded to CPU (frees VRAM at a TG cost). |
| `CHIMERE_MAX_PREFILL_CHUNK` | PCH | Max tokens admitted per driver tick during prefill. Also set via alias `CHIMERE_NATIVE_MAX_PREFILL_CHUNK` for older builds. |

### 3.1 Files

| File | Purpose |
|---|---|
| [`sweep/sweep-bench.sh`](./sweep/sweep-bench.sh) | Main entry point. Bash. CLI flags for each sweep axis + output dir. |
| [`sweep/driver_wrapper.py`](./sweep/driver_wrapper.py) | Per-cell driver. Loads `prompts.yaml`, calls `stream_bench.py`, writes `cell-summary.json`. |
| [`sweep/csv_append.py`](./sweep/csv_append.py) | Post-cell: reads `cell-summary.json` + `nvidia-smi-dmon-<tag>.csv`, appends one row to `sweep.csv`. |
| [`sweep/render_report.py`](./sweep/render_report.py) | Post-sweep: renders `REPORT.md` from `sweep.csv` + `REPORT-TEMPLATE.md.tmpl`. |
| [`sweep/prompts.yaml`](./sweep/prompts.yaml) | Canonical 4-prompt bank: short (~12 tok), medium (~420 tok), long (~1 900 tok), code (~280 tok). Round-robined by request index. |
| [`sweep/REPORT-TEMPLATE.md.tmpl`](./sweep/REPORT-TEMPLATE.md.tmpl) | Report template. `.tmpl` suffix avoids the "findings file" guardrail in the drafts pipeline. |
| [`sweep/README.md`](./sweep/README.md) | Harness-level docs: file layout, extension points, limitations. |

### 3.2 Prerequisites

1. `chimere-server` binary already built under the server repo:

   ```bash
   cd chimere-server
   cargo build --release --features server --bin chimere-server
   ```

2. `ik_llama.cpp` sm_120 shared libs on `LD_LIBRARY_PATH`. The harness
   auto-exports:

   ```
   $HOME/ik_llama.cpp/build_sm120/ggml/src
   $HOME/ik_llama.cpp/build_sm120/src
   /usr/local/cuda-12.8/lib64
   ```

   Override via the environment if your build tree lives elsewhere.

3. Model + tokenizer paths. Defaults point at the current prod
   model (`CHIMERE_MODEL`, `CHIMERE_TOKENIZER`). Override via
   the environment:

   ```bash
   CHIMERE_MODEL=/path/to/your.gguf \
   CHIMERE_TOKENIZER=/path/to/tokenizer.json \
   ./sweep/sweep-bench.sh --output-dir /tmp/chimere-sweep-mine
   ```

4. `nvidia-smi` in `PATH`.
5. `python3` (stdlib only, no pip, no PyYAML).
6. `:8082` free. The harness refuses `:8081` unless
   `--explicit-prod`.

### 3.3 Reproduction commands for the current studies

Each study lists its exact command in its own file. For quick reference:

#### Smoke run (~8 min, 3 cells)

```bash
cd chimere-server/benchmarks/sweep
./sweep-bench.sh --output-dir /tmp/chimere-sweep-min
```

Defaults to `M ∈ {1, 2, 4}`, `NCMOE = 0`, `PCH = 256`. Sanity check
that the harness works end-to-end on your box.

#### First focused sweep (~8 min, 6 cells, source of [`sweep-2026-04-24.md`](./sweep-2026-04-24.md))

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

#### Wide sweep (~6 min, 12 cells, source of the 12-cell table in [`2026-04-24-multislot-study.md`](./2026-04-24-multislot-study.md))

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

#### Dry run (print config, exit without running)

```bash
./sweep/sweep-bench.sh --output-dir /tmp/x --dry-run
```

### 3.4 Output layout

Every sweep writes the same tree. Paths below are relative to
`--output-dir`:

```
sweep.csv                   aggregate, one row per cell
REPORT.md                   auto-rendered from sweep.csv
raw/
  cell-<tag>/
    summary.json            stream_bench.py aggregate for this cell
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

The CSV schema is the same across tiers and is stable across sweep
runs. Column order:

```
cell_tag,git_sha,multislot,ncmoe,prefill_chunk,
n_reqs,conc,max_tokens,prompt_set_path,
wall_s,total_gen_tokens,agg_tok_per_s,
per_req_decode_p50,per_req_decode_p99,
ttft_ms_p50,ttft_ms_p99,
inter_tok_ms_p50,inter_tok_ms_p99,
vram_used_mib_p50,vram_used_mib_p95,
gpu_sm_p50,gpu_mem_p50,gpu_pwr_p50_w,
n_ok,n_err,errors_head
```

### 3.5 Re-rendering a report from an existing CSV

If you tweak the template or the scoring in `render_report.py` and
want to re-render without re-running the sweep:

```bash
cd chimere-server/benchmarks/sweep
python3 render_report.py \
    --csv /tmp/chimere-sweep-wide/sweep.csv \
    --template REPORT-TEMPLATE.md.tmpl \
    --output /tmp/chimere-sweep-wide/REPORT.md \
    --start "$(date -Iseconds)" --end "$(date -Iseconds)" \
    --git-sha "$(git -C ../.. rev-parse --short HEAD)" \
    --server-root "$(realpath ../..)" \
    --output-dir "/tmp/chimere-sweep-wide" \
    --model-path "$CHIMERE_MODEL" \
    --multislot-sweep "1 4 8" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "256 512 1024 2048" \
    --n-reqs-per-pass 12 \
    --max-tokens 128 \
    --prompt-set prompts.yaml
```

---

## 4. When to use what

### 4.1 "Is my fresh build functional?"

Smoke run (§3.3), read `REPORT.md`, confirm `n_ok/n` is `12/12` for
every cell. If a cell shows `0/0`, inspect
`logs/chimere-server-<tag>.log` — likely a model-load failure (see
§3.5 on the wider study for the known VRAM-release intermittent issue).

### 4.2 "What's the best config for my workload?"

Start from the decision tree in [`../../docs/perf-tuning.md`](../../docs/perf-tuning.md).
If it does not match your workload profile, fork `prompts.yaml` to
reflect your prompt sizes, then run a focused sweep on the relevant
axis (M or PCH or NCMOE) at the M value closest to your concurrency
level.

### 4.3 "Is the GPU the bottleneck?"

Deep profile with [`run-bench.sh`](./run-bench.sh) — it records SM
utilisation, memory-BW utilisation, power draw at 1 Hz alongside the
per-request TTFT / inter-token distribution. If SM is high (>80 %)
but memory-BW is low (<30 %), the kernel is launch-bound or dispatch-bound;
see [`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md)
for the current state of that investigation.

### 4.4 "I changed the backend, did I regress?"

Smoke run first (catches build/load regressions). Then rerun the same
cell of the focused sweep on the changed backend and diff the CSV rows.
`agg_tok_per_s` and `ttft_ms_p50` are the most load-bearing columns.
Differences under ~3 % are noise at N=12 (see the QW graph-reuse
experiment in [`2026-04-24-multislot-study.md`](./2026-04-24-multislot-study.md)
§3.5 for a cautionary tale).

### 4.5 "I want to benchmark a new arch (Mamba-2, Nemotron-H, Jamba, ...)"

The sweep harness is tokenizer-agnostic and takes `CHIMERE_MODEL` from
the environment, so:

```bash
CHIMERE_MODEL=/path/to/nemotron-q4_0.gguf \
CHIMERE_TOKENIZER=/path/to/nemotron-tokenizer.json \
CHIMERE_KV_MAX_SEQ=2048 \
./sweep/sweep-bench.sh --output-dir /tmp/chimere-sweep-nemotron \
    --multislot-sweep "1" --ncmoe-sweep "30" --prefill-chunk-sweep "256"
```

Note that `CHIMERE_MULTISLOT > 1` on non-Qwen3.5/3.6 architectures is
**not supported in the backend today** (`n_seqs == 1` is hardcoded in
`build_mamba2_layer`; see the "Known backend limitations" section of
the top-level [README.md](../../README.md)). Restrict the sweep to
`--multislot-sweep "1"` for Mamba-2 / Nemotron-H runs.

---

## 5. Shared limitations

Every bench in this directory shares these caveats. Each report also
restates them; we collect the consensus list here:

- **Small N per cell.** 12 to 40 requests is enough to rank, not
  enough to publish tight p99 bounds. Bump `--n-requests-per-pass`
  for tighter intervals.
- **One GPU, one model, one quant per study.** Generalisations to
  other SKUs / quants need their own runs.
- **Cold-start dominates per-cell wall time.** 30 to 60 s of model
  load per cell is excluded from the per-cell `wall_s` column by
  design; it lives in the harness's per-cell overhead.
- **VRAM is single-shot at end-of-pass.** No peak tracking. Use
  external monitoring if you need peak.
- **Native streaming only.** At `CHIMERE_MULTISLOT_NATIVE=1` the
  non-streaming path returns HTTP 400 (`native_mode_streaming_only`).
  All benches use streaming.
- **Scheduling gap unclosed.** Per-slot decode scales as 1/M.
  Aggregate is flat around ~85 tok/s on Qwen3.6-35B-A3B IQ3_S
  regardless of M. Fixing this is in-flight work in
  `ik_llama.cpp`; see
  [`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md).

---

## 6. Related documents

- [`../../docs/perf-tuning.md`](../../docs/perf-tuning.md) — operator-facing
  tuning guide. Decision tree keyed on deployment scenario.
- [`../../docs/scheduling-gap-analysis-2026-04-24.md`](../../docs/scheduling-gap-analysis-2026-04-24.md)
  — root-cause analysis of the multi-slot scheduling gap. Identifies
  `llama-delta-net.cpp:611-621` as the primary bottleneck.
- [`../../README.md`](../../README.md) — top-level project README with
  the 12-cell summary table in the Performance section.
- [`../../paper/`](../../paper/) — research-grade write-ups (DFlash
  drafter, GDN barrier paper, MTP negative paper).

---

<!-- reviewer-notes
This README is the index for chimere-server/benchmarks/. It synthesises:
- The existing directory listing (sweep-2026-04-24.md/.csv, benchmark-e2e-2026-04-24.md, analysis.json, raw/, stream_bench.py, run-bench.sh, sweep/*).
- Harness-level docs from sweep/README.md (expanded here for reachability from the top of the tree).
- References to the two new documents written in the same pass: 2026-04-24-multislot-study.md and docs/perf-tuning.md.
- References to the historical benchmarks/ directory one level up (chimere/benchmarks/*.md).

Things deliberately NOT included:
- A copy of the full 12-cell result table — those live in 2026-04-24-multislot-study.md §8 and should not be duplicated here.
- Any performance numbers in section 1 (the tier table) — those are budget numbers, not measurements.
- Tooling for arches we have not run (Mamba-2 + multi-slot etc.) — §4.5 flags the backend constraint instead of inventing a procedure.

Cross-links: every referenced file is a link, all relative paths resolve from benchmarks/. External refs (GitHub) use full URLs.
-->
