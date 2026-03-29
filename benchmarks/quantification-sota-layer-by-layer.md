RESEARCH: Find the best way to improve IQ3_S quantization quality on a Qwen3.5-35B-A3B MoE model using only a RTX 5060 Ti (16GB VRAM) + 32GB RAM.

CONTEXT:
- We have the BF16 model (67GB on disk, 14 safetensors shards)
- We already applied QuaRot Hadamard rotation shard-by-shard (works! 14/14 shards rotated)
- Current production: IQ3_S custom-mix (14.71 GB, 3.56 BPW) with imatrix + 317 tensor overrides
- The imatrix was generated with a Q4_K_M quant (BF16 imatrix crashes with llama-imatrix)
- SpinQuant (learned rotations) needs full model in memory for calibration → can't fit locally

THE USER'S IDEA: "Can we do SpinQuant/rotation/calibration LAYER BY LAYER?" — load one layer's weights, optimize rotation for that layer, save, move to next. Like we did for QuaRot shard-by-shard.

Search the web for:

1. **"layer-wise quantization" OR "per-layer calibration"** — calibrate each layer independently
2. **"SpinQuant layer by layer" OR "rotation optimization per layer"** — can Cayley SGD work per-layer?
3. **"GPTQ layer by layer"** — GPTQ already works layer-by-layer! How?
4. **"AWQ layer by layer"** — same question
5. **"imatrix generation low memory" OR "importance matrix streaming"** — generate imatrix without loading full model
6. **"llama-imatrix memory" OR "imatrix offload"** — tricks to reduce memory for imatrix
7. **"mixed precision quantization MoE"** — different bit-widths per expert
8. **"AQLM" OR "QuIP#" + "consumer GPU"** — codebook quantization that works locally
9. **"SqueezeLLM" + "sensitivity analysis per layer"** — non-uniform quantization
10. **"EfficientQAT" OR "quantization-aware training consumer"** — QAT on consumer GPU
11. **"RAMP" reinforcement learning mixed precision** — automated bit allocation
12. **"Leech lattice quantization" 2026** — latest 2-bit techniques

KEY INSIGHT: GPTQ works layer-by-layer (load one layer, calibrate with Hessian, quantize, save, next). Can we apply the same principle to:
- SpinQuant rotation learning (per-layer Cayley SGD)
- imatrix generation (per-layer importance scores)
- Mixed precision decisions (per-layer sensitivity analysis)

Also search:
- "llama.cpp quantize importance matrix shard" — can llama-imatrix work on shards?
- "auto-round" quantization (Intel) — claims better than GPTQ
- "HQQ half-quadratic quantization" — no calibration needed
- "quantization MoE experts different precision"
- "sensitive layers higher precision MoE"

For each technique:
- Memory requirement? Can it work on 16GB+32GB?
- Quality improvement over standard IQ3_S? (perplexity numbers)
- Code available? Compatible with GGUF/llama.cpp?
- Can it be done per-layer/per-shard?

THE IDEAL would be:
1. Per-layer sensitivity analysis (which layers/experts need more bits)
2. Per-layer SpinQuant rotation optimization (Cayley SGD, one layer at a time)
3. Generate imatrix per-layer (streaming, low memory)
4. Quantize with per-tensor overrides based on sensitivity
5. Result: better IQ3_S with lower perplexity, same file size

Give concrete, implementable solutions.