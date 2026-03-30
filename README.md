# Chimere

> **Warning**: This project is under active development and may contain bugs. Use at your own risk. OpenClaw and Chimere are provided as-is with no warranty. Contributions and bug reports welcome.

**Rust-native MoE inference runtime + DFlash speculative decoding drafter for hybrid GDN/GQA language models on consumer Blackwell GPUs.**

Chimere runs Qwen3.5-35B-A3B (35B total, ~3.5B active per token) at **93 tok/s** on a single RTX 5060 Ti 16GB with custom CUDA sm_120 kernels, IQ3_S dequantization, and multi-tier Engram memory.

## What's in this repo

```
chimere-server/          Rust inference runtime (56K lines)
  src/                   Core engine: hybrid attention, MoE routing, Engram, entropy router
  ffi/                   FFI bindings to ggml for quantized GEMV (IQ3_S, Q8_0, Q5_K)
  kernels/               Custom CUDA kernels for sm_120 (Blackwell)
  docs/                  Architecture docs, profiling analysis

chimere-dflash/          Block diffusion drafter for speculative decoding (22K lines Python + C++)
  chimere/               Core library: modeling, engram, spec_decode, entropy_monitor
  scripts/               Training, benchmarking, evaluation
  extract/               C++ hidden state extractors (CMake)
  tests/                 Unit tests

patches/ik-llama-mtp/    5 patches adding MTP support to ik_llama.cpp for Qwen3.5 MoE
paper/                   Publication drafts and research documents
```

## Key results

| Metric | Value |
|--------|-------|
| Generation throughput | 93 tok/s (ik_llama backend), 83 tok/s (chimere-server HTTP) |
| Model | Qwen3.5-35B-A3B, IQ3_S custom-mix (14.71 GB, 3.56 BPW) |
| VRAM usage | ~14.2 GB / 16 GB |
| DFlash offline acceptance | τ = 9.4 tokens/step (+47% vs original paper) |
| Benchmark | 10/10 (kine, code, math, tools) |
| Hardware | RTX 5060 Ti 16GB, i5-14600KF, 32GB DDR5 |

## Architecture highlights

- **Hybrid GDN/GQA attention**: 3:1 ratio (24 GDN + 8 full attention layers), 4x KV cache reduction
- **Custom CUDA kernels**: IQ3_S dequant, Q8_0+dp4a GEMV, flash attention, fused MoE — all native sm_120
- **Multi-tier Engram memory**: Cuckoo filter (<10ns) → N-gram hash tables (O(1)) → FAISS semantic
- **Entropy-adaptive routing**: per-token compute allocation based on Shannon entropy
- **Adaptive Budget Forcing**: thinking budget management for quantized reasoning models
- **DFlash drafter**: block diffusion with mask ratio embedding + single-shot inference + exponential loss

## DFlash: the GDN State Barrier

We document a fundamental incompatibility between speculative decoding and hybrid SSM-attention architectures: GDN recurrent state cannot be trimmed via `llama_memory_seq_rm`, preventing online verification of draft tokens. This "GDN State Barrier" affects all hybrid linear/quadratic models (Jamba, RWKV-6, Qwen3.5 MoE). See `paper/02-dflash-moe-gdn-barrier-outline.md`.

## MTP patches for ik_llama.cpp

The `patches/ik-llama-mtp/` directory contains 5 patches adding Multi-Token Prediction support for Qwen3.5-35B-A3B MoE to [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). These fix 8 bugs in tensor mapping, compute buffer management, and MoE-specific handling. See patch commit messages for details.

## Building

```bash
# chimere-server (requires CUDA 12.8+ and ik_llama.cpp)
cd chimere-server
export IKLLAMACPP_DIR=/path/to/ik_llama.cpp/build_sm120
cargo build --release --features server

# chimere-dflash (Python)
cd chimere-dflash
pip install -e .
```

## Related repos

- [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) — Inference orchestrator (routing, Engram management, quality loop)
- [ramp-quant](https://github.com/AIdevsmartdata/ramp-quant) — RAMP mixed-precision quantization pipeline

## Publications

See `paper/00-publication-tracker.md` for the full list. Key papers in preparation:

1. **Chimere System Paper** — self-improving MoE system on consumer GPU (arXiv/MLSys)
2. **DFlash MoE + GDN State Barrier** — τ=9.4, structural incompatibility result (NeurIPS workshop)
3. **Engram on Consumer Hardware** — multi-tier N-gram memory (EMNLP demo)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kevin Remondiere** — Independent ML researcher, Bayonne, France
