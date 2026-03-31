# Chimere

**Rust-native MoE inference runtime with custom CUDA kernels for Blackwell GPUs.**

Chimere is a specialized inference engine for mixture-of-experts language models, achieving 93 tok/s on a single RTX 5060 Ti (16 GB VRAM) with custom sm_120 CUDA kernels.

## Chimere Distilled Models

We distilled Claude Opus 4.6 into Qwen3.5-35B-A3B (MoE, 256 experts, ~3.5B active params/token). This is the **first Opus distillation targeting a MoE architecture**.

### Benchmarks

| Metric | BF16 (65 GB) | RAMP-v3 GGUF (15 GB) | Base Qwen3.5 |
|--------|-------------|----------------------|-------------|
| **HumanEval (30)** | 97% | **97%** | — |
| **BFCL tool-calling (20)** | 85% | **85%** | 67.3% |
| **GSM8K (1,319)** | 52.8% | **64.9%** | — |
| **IFEval (15)** | 80% | **80%** | 91.9% |

### Download

| Model | Size | Link |
|-------|------|------|
| **GGUF (recommended)** | 15 GB | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF) |
| BF16 | 65 GB | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-BF16](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-BF16) |
| LoRA adapter | 15 GB | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-LoRA](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-LoRA) |

### Quick Start

```bash
# With llama.cpp / llama-server
llama-server \
    -m Qwen3.5-35B-A3B-Chimere-Distilled-RAMP-v3.gguf \
    -ngl 99 --n-cpu-moe 4 -c 32768 --jinja --port 8081
```

Runs at ~90 tok/s on RTX 5060 Ti (16 GB VRAM) with `-ngl 99 --n-cpu-moe 4`.

---

## Inference Runtime

### Architecture

- **chimere-server**: Rust inference engine (56K lines) — hybrid attention, expert routing, memory management
- **chimere-dflash**: Speculative decoding module (Python/C++, 22K lines)
- **patches/ik-llama-mtp**: Enhancement patches for upstream integration

### Technical Features

- Hybrid attention combining group-distributed and grouped-query approaches
- Custom sm_120 CUDA kernels for quantized operations
- Multi-tiered Engram memory system (cuckoo filters + semantic indexing)
- Entropy-based computational routing
- DFlash speculative decoding (+47% vs original paper)

### RAMP Quantization

Custom per-tensor quantization recipe:
- Critical paths (attention values, SSM params): Q8_0 / Q6_K
- MoE experts (256 per layer): IQ3_S
- Result: **15 GB** with zero quality loss on agentic benchmarks

### Requirements

- CUDA 12.8+ (Blackwell sm_120 native)
- ik_llama.cpp for server component
- 16 GB VRAM minimum (RTX 4080/5060 Ti class)

## Related Projects

- [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) — Inference orchestrator (intent classification, routing, quality gating)

## License

Apache 2.0 — Kevin Remondiere
