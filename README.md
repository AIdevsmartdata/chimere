# Chimere

**Rust inference engine with ik_llama FFI backend for Blackwell GPUs.**

Chimere is a specialized inference engine for mixture-of-experts language models. It wraps [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) via FFI for optimized CUDA inference and adds Engram memory, custom sampling (DRY penalty), entropy routing, and multi-token prediction on top.

**80 tok/s generation, 789 tok/s prefill** on a single RTX 5060 Ti (16 GB VRAM) with sm_120 CUDA kernels.

## Chimere Distilled Models

We distilled Claude Opus 4.6 into Qwen3.5-35B-A3B (MoE, 256 experts, ~3.5B active params/token). This is the **first Opus distillation targeting a MoE architecture**.

### Final Benchmark Results

| Benchmark | v1 RAMP (15 GB) | v3 RAMP (15 GB) | Base Qwen3.5 |
|-----------|-----------------|-----------------|-------------|
| **GSM8K CoT 8-shot** (lm-eval, 1319 qs) | 52.2% | **84.0%** | -- |
| **HumanEval** (30 problems, executed) | **97%** | 83% | -- |
| **BFCL tool-calling** (20 questions) | **90%** | 75% | 67.3% |
| **IFEval** (15 instruction tests) | 67% | **100%** | ~91.9% |
| **Edge cases** (15 adversarial tests) | 87% | **100%** | -- |
| **Speed** (RTX 5060 Ti) | 80 tok/s | 80 tok/s | ~75 tok/s |
| **Speed** (B200) | 154 tok/s | 152 tok/s | -- |
| **Prefill** (RTX 5060 Ti) | 789 tok/s | 789 tok/s | -- |
| **TTFT** | 80ms | 80ms | -- |

#### Qualitative Agentique Tests (3 complex real-world scenarios)

| Scenario | v1 | v3 | Max |
|----------|----|----|-----|
| Cybersecurity incident response (multi-tool chain) | 4 | 4 | 10 |
| ML pipeline architecture (RAG, 10K users) | 8 | 8 | 10 |
| Rust MoE runtime optimization | 7 | 8 | 10 |
| **Total** | **19** | **20** | **30** |

**Key finding:** v1 best for code+tools (97% HumanEval, 90% BFCL), v3 best for instructions+reasoning (100% IFEval, 84% GSM8K). A-LoRA routing (ODO classifies intent and selects the appropriate LoRA) gives the best of both.

### Download

| Model | Size | Best for | Link |
|-------|------|----------|------|
| **v1 RAMP GGUF** | 15 GB | Code + tool-calling | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF) |
| **v3 RAMP GGUF** | 15 GB | Instructions + reasoning | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF) |
| BF16 | 65 GB | Full precision | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-BF16](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-BF16) |
| LoRA adapter | 15 GB | Fine-tuning | [Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-LoRA](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-LoRA) |

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
