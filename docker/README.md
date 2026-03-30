# Chimere Stack

One-command self-improving LLM inference for consumer GPUs.

Run a 35-billion-parameter reasoning model at 90 tok/s on a single RTX card,
with overnight LoRA fine-tuning, domain memory, and intelligent routing --
all from `docker compose up`.

```
 User / App
     |
     v
 +---------+     +----------+     +-------------+
 |  Open   | --> |   ODO    | --> |  Inference  |
 | WebUI   |     | (router) |     | (ik_llama)  |
 | :3000   |     |  :8084   |     |    :8081    |
 +---------+     +----------+     +-------------+
                      |                  ^
                      v                  |
                 +----------+     +-------------+
                 | SearXNG  |     |   Nightly   |
                 | (search) |     | (self-impr) |
                 |  :8080   |     |  cron-based |
                 +----------+     +-------------+
```

## Quick Start

```bash
git clone https://github.com/AIdevsmartdata/chimere
cd chimere/docker

# 1. Detect your GPU and generate .env
python3 ../scripts/detect-gpu.py

# 2. Download the model (~15 GB)
./scripts/download-model.sh

# 3. Launch the stack
docker compose up -d

# 4. Open the UI
#    http://localhost:3000
```

The first launch builds the inference container (compiles ik_llama.cpp for your
GPU architecture). This takes 5-10 minutes. Subsequent starts are instant.

## What's Inside

### 5 Containers

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| **inference** | `chimere-inference` | 8081 | ik_llama.cpp server with full GPU offload, quantized KV cache, flash attention |
| **odo** | `chimere-odo` | 8084 | Intelligent request router: intent classification, Engram injection, sampling profiles, SOUL.md personality |
| **webui** | `open-webui` | 3000 | Chat interface with conversation history, file upload, code highlighting |
| **searxng** | `searxng/searxng` | 8080 | Privacy-respecting meta-search engine for web grounding |
| **nightly** | `chimere-nightly` | -- | Overnight self-improvement: LoRA training, Engram updates, DSPy prompt optimization |

### How They Connect

1. You chat through **Open WebUI** (port 3000) or any OpenAI-compatible client.
2. Requests go to **ODO**, which classifies intent (code, research, medical, general),
   selects sampling parameters, and injects relevant Engram context.
3. ODO forwards to **inference** (ik_llama.cpp) which runs the quantized model on GPU.
4. For web-grounded answers, ODO queries **SearXNG** and includes search results in context.
5. Every night, **nightly** reviews the day's responses, trains a LoRA adapter on the
   best ones, updates Engram memory tables, and optionally optimizes system prompts via DSPy.

## Supported GPUs

| GPU | VRAM | Quant | Context | Throughput | Notes |
|-----|------|-------|---------|------------|-------|
| RTX 3060 12GB | 12 GB | IQ3_S (14.7 GB) | 8K | ~25 tok/s | Needs `CHIMERE_NCMOE=8` to fit, reduced context |
| RTX 3090 / 4090 | 24 GB | Q4_K_M (19.5 GB) | 32K | ~60 tok/s | Comfortable fit, full context |
| RTX 4060 Ti 16GB | 16 GB | IQ3_S (14.7 GB) | 32K | ~50 tok/s | Sweet spot for price/performance |
| RTX 5060 Ti 16GB | 16 GB | RAMP-v2 (15.2 GB) | 32K | ~90 tok/s | Native sm_120 kernels, best perf/$ |
| RTX 5070 Ti 16GB | 16 GB | RAMP-v2 (15.2 GB) | 32K | ~100 tok/s | More SMs than 5060 Ti |
| RTX 5080 16GB | 16 GB | Q4_K_M (19.5 GB) | 32K | ~110 tok/s | Fits Q4 comfortably |
| RTX 5090 32GB | 32 GB | Q5_K_XL (25 GB) | 64K | ~130 tok/s | Full quality, maximum context |

Throughput estimates for Qwen3.5-35B-A3B. Actual numbers depend on CPU MoE offload
settings and system memory bandwidth.

## Requirements

- **Docker** with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **NVIDIA GPU** with 12 GB VRAM minimum (16 GB recommended)
- **32 GB system RAM** recommended (MoE experts spill to CPU)
- **25 GB free disk** for model file + container images
- Linux host (tested on Ubuntu 24.04/24.10)

### Verify GPU Access

```bash
# Should show your GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Configuration

### Environment Variables

Set these in `docker/.env` or pass them directly. The `detect-gpu.py` script
generates sensible defaults for your hardware.

**Inference:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERE_MODEL` | `/models/model.gguf` | Path to GGUF model inside container |
| `CHIMERE_CTX` | `32768` | Context window size |
| `CHIMERE_NGL` | `99` | GPU layers (99 = all) |
| `CHIMERE_NCMOE` | `4` | MoE expert layers offloaded to CPU |
| `CHIMERE_FLASH_ATTN` | `on` | Flash attention (on/off) |
| `CHIMERE_KV_K` | `q8_0` | Key cache quantization |
| `CHIMERE_KV_V` | `q4_0` | Value cache quantization |
| `CHIMERE_NP` | `1` | Number of parallel slots |
| `GPU_ARCH` | `120` | CUDA architecture (89, 120, etc.) |

**Nightly self-improvement:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NIGHTLY_SCHEDULE` | `00:30` | Time to run nightly jobs (HH:MM) |
| `NIGHTLY_LORA` | `true` | LoRA training on quality-validated pairs |
| `NIGHTLY_ENGRAM` | `true` | Engram n-gram memory updates |
| `NIGHTLY_DSPY` | `false` | DSPy system prompt optimization |
| `NIGHTLY_MEZO` | `false` | MeZO zeroth-order LoRA (fits 16 GB) |

### SOUL.md -- Personality

Create a file at `docker/config/soul/SOUL.md` to define the model's personality,
tone, and behavioral guidelines. ODO will load it as the system prompt.

```markdown
# SOUL.md

You are a helpful assistant specialized in ...
Your tone is professional yet approachable.
When uncertain, say so rather than guessing.
```

ODO loads SOUL.md and injects it as the system prompt for every request.

### Engram -- Domain Memory

Engram is a multi-tier memory system that gives the model domain-specific knowledge
without retraining:

- **Cuckoo filter** (<10ns lookup) -- fast membership test for known terms
- **N-gram hash tables** (O(1)) -- domain vocabulary, common patterns
- **FAISS semantic index** -- dense retrieval for longer knowledge chunks

Engram files live in `/data/engram/` (the `chimere-engram` volume). ODO
automatically selects the right Engram table based on intent classification.

To add domain knowledge:

```bash
# From the host
docker exec chimere-odo python3 /app/engram/engram_ingest.py \
    --input /data/engram/my-domain-corpus.txt \
    --route my-domain
```

### Nightly Self-Improvement

The `nightly` container runs four optional jobs on a daily schedule:

1. **Engram Write** -- Scores the day's responses, extracts quality patterns into
   Engram tables. Decays stale n-grams. Runs first because LoRA benefits from
   updated memory.

2. **LoRA Training** -- Collects quality-validated response pairs (score >= 4/5),
   deduplicates, converts to ShareGPT format, and trains a LoRA adapter.
   Requires GPU access; skips gracefully if the inference container is running.

3. **MeZO LoRA** -- Alternative to standard LoRA that uses zeroth-order optimization
   (gradient-free, 2 forward passes). Uses the same memory as inference -- fits on
   16 GB cards that cannot afford a backward pass.

4. **DSPy Optimization** -- Uses MIPROv2 Bayesian optimization to improve system
   prompts per domain. Light mode (default) runs 10 trials. Disabled by default
   because it requires a mature eval dataset.

Logs are written to `/data/logs/nightly/`. Check them with:

```bash
docker exec chimere-nightly cat /data/logs/nightly/nightly-$(date +%Y-%m-%d)*.log
```

### Code Mode

ODO automatically detects code-related requests and switches to optimized
sampling parameters:

- **Thinking enabled** with lower temperature (0.6 vs 1.0)
- **Structured output** via GBNF grammars when appropriate
- Engram code tables injected for language-specific patterns

No configuration needed -- intent classification handles it.

### Mobile Access via Tailscale

To access Chimere from your phone or other devices:

1. Install [Tailscale](https://tailscale.com/) on the host and your mobile device.
2. The stack binds to `0.0.0.0`, so it's reachable on the Tailscale IP.
3. Open `http://<tailscale-ip>:3000` in your mobile browser.
4. For a native experience, use any OpenAI-compatible mobile app pointed at
   `http://<tailscale-ip>:8084/v1` (ODO's OpenAI-compatible endpoint).

No port forwarding or public exposure required.

## Directory Structure

```
docker/
  docker-compose.yml       Main compose file
  .env                     Generated by detect-gpu.py
  config/
    searxng/               SearXNG configuration
  inference/
    Dockerfile             ik_llama.cpp multi-stage build
    entrypoint.sh          Env-to-flags adapter
  nightly/
    Dockerfile             Nightly self-improvement container
    scheduler.py           Simple Python scheduler
scripts/
  detect-gpu.py            Auto-detect GPU and generate .env
  download-model.sh        Download the GGUF model from HuggingFace
```

## Monitoring

```bash
# Container status
docker compose ps

# Inference server health
curl http://localhost:8081/health

# ODO stats (routes, request counts, latency)
curl http://localhost:8084/stats

# Inference metrics (Prometheus format)
curl http://localhost:8081/metrics

# Live logs
docker compose logs -f inference
docker compose logs -f odo
```

## Troubleshooting

**"Model file not found"** -- Make sure your GGUF model is in the `models/`
directory mounted as a volume. Check `CHIMERE_MODEL` in `.env`.

**Out of VRAM** -- Increase `CHIMERE_NCMOE` (offloads more MoE experts to CPU).
Try values 4, 8, 14, 18. Each increment frees ~200-500 MB VRAM at the cost of
~5-10% generation speed.

**Slow generation** -- Check `CHIMERE_NCMOE` is not too high. Verify
`GPU_ARCH` matches your card (wrong arch = software fallback). Run
`nvidia-smi` to confirm no other process is using the GPU.

**Nightly jobs fail** -- Check logs in `/data/logs/nightly/`. Common causes:
insufficient training pairs (need 50+), inference container holding the GPU
(LoRA skips gracefully), or missing eval datasets for DSPy.

**Build fails** -- The inference container compiles ik_llama.cpp from source.
Ensure `nvidia-container-toolkit` is installed and `docker run --gpus all`
works. For older GPUs, set `GPU_ARCH` explicitly (e.g., `89` for Ada).

## Links

- **GitHub**: [chimere](https://github.com/AIdevsmartdata/chimere) | [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) | [ramp-quant](https://github.com/AIdevsmartdata/ramp-quant)
- **HuggingFace models**: [Qwen3.5-35B-A3B RAMP-v2](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-RAMP-v2-15G)
- **Papers**: See `paper/` directory for drafts on the Chimere system, DFlash speculative decoding, and Engram memory

## License

Apache 2.0 -- see [LICENSE](../LICENSE).
