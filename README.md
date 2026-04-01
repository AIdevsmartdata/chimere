# Chimere

**The first self-improving Rust inference engine for MoE language models.**

Chimere is a complete inference stack that runs Qwen3.5-35B-A3B (256 experts, ~3.5B active params/token) at **80 tok/s on a single RTX 5060 Ti** (16 GB VRAM). It combines a Rust runtime with Engram hierarchical memory, entropy-adaptive routing, custom sampling, and an orchestration layer (ODO) that classifies intent and routes to specialized pipelines — all self-improving nightly via LoRA and Engram updates.

**No one else does this.** Existing solutions (vLLM, llama.cpp, TGI) are inference-only. Chimere is inference + intelligence: Engram injects domain knowledge at inference time, ODO routes prompts to specialized adapters, and the nightly pipeline learns from usage to improve quality over time.

## Build it and test it

```bash
# Clone
git clone https://github.com/AIdevsmartdata/chimere.git
cd chimere

# Docker (recommended) — full stack: Chimere + ODO + Web UI + Search
docker compose -f docker/docker-compose.yml up -d

# Or build from source (requires CUDA 12.8 + Rust)
cd chimere-server && cargo build --release --features server
```

Then download a model:
```bash
# Chimere v3 — Opus-distilled, 100% IFEval, 84% GSM8K
huggingface-cli download Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF chimere-v3-ramp.gguf

# Start
CHIMERE_MODEL=chimere-v3-ramp.gguf CHIMERE_LLAMA_BACKEND=1 chimere-server
```

## Chimere Distilled Models — Opus into MoE

We distilled **Claude Opus 4.6** into Qwen3.5-35B-A3B. This is the **first Opus distillation targeting a MoE architecture** (256 experts, hybrid SSM+attention). Two variants trained on different dataset compositions:

| Benchmark | v1 RAMP (15 GB) | v3 RAMP (15 GB) | Base Qwen3.5 |
|-----------|-----------------|-----------------|-------------|
| **GSM8K** (1319 qs, CoT 8-shot) | 52.2% | **84.0%** | -- |
| **HumanEval** (30 problems) | **97%** | 83% | -- |
| **BFCL tool-calling** (20 qs) | **90%** | 75% | 67.3% |
| **IFEval** (15 instruction tests) | 67% | **100%** | ~91.9% |
| **Edge cases** (15 adversarial) | 87% | **100%** | -- |

### Performance (RTX 5060 Ti, 16 GB VRAM)

| Metric | Value |
|--------|-------|
| **Generation** | 80 tok/s |
| **Prefill** | 789 tok/s |
| **TTFT** | 80ms |
| **Context** | 64K tokens |
| **VRAM** | 15.3 GB (560 MB free) |

v1 = best for code + tool-calling (97% HumanEval, 90% BFCL). v3 = best for instructions + reasoning (100% IFEval, 84% GSM8K). ODO routes to the right adapter automatically.

### Download

| Model | Size | Best for | Link |
|-------|------|----------|------|
| **v1 RAMP** | 15 GB | Code + tools | [HuggingFace](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-GGUF) |
| **v3 RAMP** | 15 GB | Instructions + reasoning | [HuggingFace](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF) |
| BF16 | 65 GB | Full precision | [HuggingFace](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-BF16) |
| LoRA | 15 GB | Fine-tuning | [HuggingFace](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-Distilled-LoRA) |

---

## Architecture

```
User → ODO (intent classification + routing) → Chimere Server
                                                    ├── Engram (hierarchical memory codebook)
                                                    ├── Custom sampling (DRY, temperature, entropy-aware)
                                                    ├── CUDA inference (sm_120 Blackwell kernels)
                                                    ├── A-LoRA hot-swap (conversation/code/tools adapters)
                                                    └── MTP speculative decoding
                                                          ↓
                                               Nightly pipeline (self-improvement)
                                                    ├── Training pair logging
                                                    ├── LoRA fine-tune on usage data
                                                    └── Engram codebook update (MDL compression)
```

### Chimere Server (Rust)

The inference engine. Loads GGUF models, manages KV cache, runs generation with custom sampling. OpenAI-compatible API (`/v1/chat/completions`).

**Key features no other engine has:**
- **Engram** — hierarchical memory codebook in Poincaré hyperbolic space. Entries near the origin = general concepts, near the boundary = specialized details. Injected at inference time, not just in the prompt.
- **Entropy routing** — dynamically switches between attention mechanisms based on sequence entropy
- **Custom sampling** — DRY penalty (anti-repetition), Engram-aware temperature, min-p
- **MTP** — multi-token prediction for speculative decode (+49.5% acceptance rate)

### ODO — One Door Orchestrator

Intent classifier + routing proxy. Classifies each prompt and routes to the right pipeline (code, research, kine, cyber, default) with appropriate:
- Thinking mode (enabled/disabled per route)
- Engram injection (domain-specific knowledge)
- Sampling profile (temperature, top_p per route)
- System prompt consolidation
- ABF (Adaptive Batch Formation)

### Nightly Pipeline

Self-improvement loop that runs daily:
1. Collects training pairs from ODO logs
2. LoRA fine-tune on high-quality pairs
3. Engram codebook update (MDL-gated: only adds entries when prediction error exceeds threshold)
4. Quality benchmarks to detect regression

---

## Configuration

All settings via environment variables:

### Model & Backend
| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERE_MODEL` | (required) | Path to GGUF model |
| `CHIMERE_TOKENIZER` | auto-detect | Path to tokenizer.json |
| `CHIMERE_LLAMA_BACKEND` | unset | Enable optimized FFI backend |

### Inference
| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERE_NCMOE` | 4 | MoE expert layers on CPU (0=all GPU) |
| `CHIMERE_KV_MAX_SEQ` | 65536 | Context length |
| `CHIMERE_BATCH` | 4096 | Batch size for prefill |
| `CHIMERE_UBATCH` | 512 | Micro-batch for decode |
| `CHIMERE_THREADS` | 14 | CPU threads |

### KV Cache
| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERE_KV_TYPE_K` | 8 (q8_0) | Key cache quantization |
| `CHIMERE_KV_TYPE_V` | 2 (q4_0) | Value cache quantization |
| `CHIMERE_KV_HADAMARD` | 1 | Hadamard rotation on keys |
| `CHIMERE_FLASH_ATTN` | 1 | Flash attention |

### Server
| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERE_PORT` | 8090 | API port |
| `CHIMERE_MAX_AGENTS` | 4 | Max concurrent requests |

### RAMP Quantization Sweet Spots (RTX 5060 Ti 16 GB)

| ncmoe | Context | Gen tok/s | VRAM free | Recommendation |
|-------|---------|-----------|-----------|----------------|
| 4 | 64K | 77 | 702 MB | Safe, production |
| **3** | **64K** | **80** | **560 MB** | **Recommended** |
| 2 | 64K | OOM | -- | Not viable on 16 GB |
| 4 | 32K | 80 | 1.2 GB | Maximum headroom |

---

## Requirements

- **GPU**: NVIDIA with 16+ GB VRAM (RTX 4080/5060 Ti/5070 Ti/5080 or better)
- **CUDA**: 12.8+ for Blackwell (sm_120). 12.4+ for Ada (sm_89)
- **RAM**: 32 GB recommended (MoE experts offloaded to CPU)
- **Rust**: 1.80+ (for build from source)

## Docker Stack

The `docker-compose.yml` brings up the full stack:

| Service | Port | Description |
|---------|------|-------------|
| **chimere** | 8081 | Inference engine |
| **odo** | 8084 | Intent router + orchestrator |
| **open-webui** | 3000 | Chat UI |
| **searxng** | 8888 | Web search |
| **chromadb** | (internal) | Vector DB for RAG |
| **nightly** | (internal) | Self-improvement pipeline |

```bash
# Download model first
mkdir -p models && cd models
huggingface-cli download Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF chimere-v3-ramp.gguf
# Also need tokenizer
huggingface-cli download Qwen/Qwen3.5-35B-A3B tokenizer.json

# Start everything
cd .. && docker compose -f docker/docker-compose.yml up -d

# Test
curl http://localhost:8081/health
curl http://localhost:8084/routes
```

## Related Projects

- [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) — ODO orchestrator
- [Chimere v3 RAMP GGUF](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF) — Opus-distilled model

## License

Apache 2.0 — Kevin Remondiere
