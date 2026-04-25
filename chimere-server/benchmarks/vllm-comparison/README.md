# Comparative benchmarks — chimere-server vs llama-server vs vLLM

This subfolder holds the apples-to-apples bench harness flagged in the audit
of 2026-04-25 ("doc claim 'au niveau de vLLM' is unverifiable without this").

## Why it's not trivial

The three engines do not load the same artifact:

| Engine          | Backend           | Model artifact                                              |
| --------------- | ----------------- | ----------------------------------------------------------- |
| `chimere-server`| Rust FFI ik_llama | `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (3.4 bpw)                  |
| `llama-server`  | stock llama.cpp   | identical GGUF                                              |
| `vLLM`          | PyTorch + AWQ     | `RedHatAI/Qwen3-30B-A3B-AWQ-INT4` (4 bpw, closest available) |

vLLM cannot ingest GGUF in production. The closest comparable size/family
that vLLM serves natively on consumer Blackwell is the AWQ-INT4 of the same
model family. We report this honestly in the result tables — **not** as
"identical model" but as "comparable size class on same hardware".

## Hardware

- RTX 5060 Ti, 16 GB, sm_120, driver 590.48, CUDA 12.8
- i5-14600KF, 32 GB DDR5, NVMe Gen4
- Linux 6.17, no other GPU consumers

## Workloads

| Name   | Prompt size (tok) | Gen size (tok) | Stresses             |
| ------ | ----------------- | -------------- | -------------------- |
| short  | ~30               | 256            | TTFT, small batch    |
| medium | ~50               | 1024           | sustained gen tok/s  |
| long   | ~8000             | 512            | long-context prefill, prefix cache |

## Concurrency sweep

`M ∈ {1, 2, 4}`. R=3 replicas per cell (cold + 2 warm). Median reported.

## How to run

Pre-reqs:

```bash
# stock llama-server (build adjacent to ik_llama)
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build ~/llama.cpp/build --target llama-server -j

# vLLM in dedicated venv (heavy; ~6 GB python deps)
python3 -m venv ~/.openclaw/venvs/vllm
source ~/.openclaw/venvs/vllm/bin/activate
pip install vllm  # pulls torch+CUDA stack
```

Run:

```bash
cd chimere-server/benchmarks/vllm-comparison
./run-comparison.sh
```

The script brings each engine up in turn (kills the previous), runs the
3×3×3 = 27 cells per engine, and writes one row per (engine, workload, M,
replica) into `results-YYYYMMDD-HHMMSS/results.csv`.

## Reading the CSV

Columns:

```
timestamp, engine, model, workload, M, replica, n_prompt, n_gen,
ttft_ms, total_ms, gen_tokps, prefill_tokps, vram_mb_peak, error
```

To produce a markdown summary table:

```bash
python3 analyze.py results-*/results.csv > BENCHMARKS.md
```

## What good results would look like

For chimere-server to legitimately claim "vLLM-level":

- **TTFT** (M=1, short): within 30 % of vLLM. (vLLM has heavier startup but
  typically faster per-token TTFT thanks to PagedAttn locality.)
- **gen_tokps** (M=4, medium): same ballpark as vLLM's `--max-num-seqs 4`
  baseline. Today the M=4 GDN barrier in chimere caps throughput well below
  M=1×4, so an honest result probably shows vLLM ahead until M2 RadixAttention
  lands and the GDN serialization is fixed.
- **VRAM peak**: chimere-server should be lower (no PagedAttn overhead, no
  PyTorch runtime) by ~1–2 GB.
- **long workload**: prefix cache ON in chimere (M2 branch) should approach
  vLLM RadixAttention on the warm replicas.

## Status

- [x] Harness written (this folder, 2026-04-25).
- [ ] First run — pending (waits for chimere-server idle window).
- [ ] `BENCHMARKS.md` summary committed at repo root.

## Skip-list

Things explicitly out of scope:

- Tensor parallelism (chimere is single-GPU by design).
- Multi-node / distributed.
- Mixed-precision activations (AWQ on vLLM uses W4A16, llama.cpp/chimere use
  W4A8 effectively via dequant→INT8 mma).
