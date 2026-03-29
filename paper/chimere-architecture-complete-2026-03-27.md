# Chimère: a complete MoE language model architecture for consumer hardware

**Chimère is a hybrid SSM-attention Mixture-of-Experts language model designed to run entirely on a single RTX 5060 Ti (16GB VRAM) with 32GB DDR4-3200 RAM, targeting 5–7B active parameters per token from a 35–50B total parameter pool.** The architecture fuses five subsystems — Gated DeltaNet backbone, fine-grained MoE with 128 routed experts, a multi-tier Engram conditional memory, entropy-routed block diffusion for parallel generation, and SNN-inspired conditional computation — into a coherent system grounded in information theory, lattice mathematics, and statistical physics. This document presents the complete design, backed by over 80 papers published through March 2026, alongside honest feasibility assessments for every component.

---

## 1. The backbone: Gated DeltaNet in a 3:1 hybrid with full attention

The SSM landscape as of March 2026 has a clear frontrunner. **Gated DeltaNet (GDN)** (arXiv:2412.06464, ICLR 2025) combines Mamba-2's scalar gating with the delta update rule for precise key-value association overwriting in a single recurrence: M_t = α_t · M_{t-1} · (I − k_t q_t^T) + k_t q_t^T. This dual mechanism outperforms both pure Mamba-2 (16.42 vs 16.56 wiki perplexity at 1.3B) and DeltaNet (17.71) while maintaining linear-time O(n) inference with constant memory per token. GDN's production validation is decisive: it powers **Qwen3.5-397B-A17B**, **Qwen3-Next-80B-A3B**, and **OLMo Hybrid** (March 2026).

The strongest challenger is **Mamba-3 MIMO** (arXiv:2603.15569, ICLR 2026), which introduces exponential-trapezoidal discretization, complex-valued states via data-dependent RoPE, and a MIMO formulation. At 1.5B scale, Mamba-3 MIMO gains +1.8pp over GDN. However, it lacks production validation and its ecosystem is immature.

### The hybrid layer ratio that industry has converged on

The **3:1 (GDN:full attention)** ratio has emerged as the de facto standard. For Chimère's 32-layer model: **24 GDN layers + 8 full attention layers** (one every 4th layer). This **reduces KV cache by 4x** compared to pure attention, freeing VRAM for expert caching.

**Chimère backbone specification:**

| Component | Configuration |
|-----------|--------------|
| Architecture | GDN + Full Attention Hybrid (3:1) |
| Total layers | 32 (24 GDN + 8 full attention) |
| Hidden dimension | 4096 |
| GDN heads | 32, head_dim=128, state_size=128 |
| Attention | GQA with 32 Q-heads, 8 KV-heads, head_dim=128 |
| Normalization | RMSNorm pre-layer |
| Activation | SiLU throughout |

---

## 2. Fine-grained MoE with 128 experts and loss-free routing

**128 routed experts** hits the sweet spot for consumer hardware. Same count as Qwen3-30B-A3B, NVIDIA Nemotron-3-Nano-30B-A3B, and Llama 4 Maverick.

Chimère uses **top-4 routing with 1 shared expert**, yielding ~5.5-6B active parameters from ~35B total. The routing uses DeepSeek-V3's **auxiliary-loss-free load balancing** (arXiv:2408.15664): dynamically adjusted bias terms instead of auxiliary loss.

**D2-MoE delta decomposition** (ICML 2025): decomposes each expert into shared base + unique delta. Base stays in VRAM; only deltas (~50-70% smaller) transfer over PCIe. Effectively doubles offloading bandwidth.

| MoE parameter | Value |
|---------------|-------|
| Routed experts | 128 |
| Active routed | 4 per token |
| Shared experts | 1 (always in VRAM) |
| Expert hidden dim | 2048 (SwiGLU) |
| Shared expert dim | 3072 (1.5x) |
| Total params | ~35-40B |
| Active per token | ~5.5B |

---

## 3. Engram conditional memory: the knowledge pillar

DeepSeek Engram paper (arXiv:2601.07372): conditional memory module with hashed lookup + context-aware gating. Engram-27B gains +3.4 MMLU, +5.0 BBH, +3.7 ARC-C over iso-FLOPs MoE baseline. Optimal allocation: rho ~75-80% (20-25% to Engram, rest to MoE).

### Chimère's 4-tier memory hierarchy

- **Tier 0 — Cuckoo filter** (<10ns, 50MB GPU): existence check
- **Tier 1 — LSH/SimHash** (<1us, 200MB CPU): fuzzy semantic matching
- **Tier 2 — Multi-head hash tables** (O(1), 4GB CPU + 500MB GPU hot cache): core n-gram lookup
- **Tier 3 — VSA holographic memory** (400MB CPU): structured relational knowledge via FHRR

Total: ~850 MB GPU + 4.6 GB CPU

---

## 4. Block diffusion and entropy routing

### DFlash drafter
DFlash (arXiv:2602.06036): >6x lossless acceleration using block diffusion as parallel drafter. ~200-400M parameter drafter for Chimère (~0.5-0.8GB VRAM at FP4).

### Entropy router (novel contribution)
No existing paper implements this exact mechanism. Route tokens between AR (high entropy) and block diffusion (low entropy) based on Shannon entropy of vocabulary distribution. Swordsman (arXiv:2602.04399) demonstrated 81.50% GSM8K with 8.79x speedup using entropy-based block partitioning.

---

## 5. Three-level conditional computation

- **Level 1 — GateSkip** (layer skip, 0.004% params overhead, ~15% compute savings)
- **Level 2 — Entropy routing** (AR vs diffusion)
- **Level 3 — SNN sparsity** (90% activation sparsity via SpikingSSMs)

Orthogonal axes: depth x mode x width.

---

## 6. Training pipeline

### Realistic path
1. Start from Qwen2.5-7B or Qwen3-8B (fits 16GB for QLoRA)
2. Upcycle dense → MoE (duplicate FFN into experts)
3. Synthetic distillation from Qwen3.5-35B-A3B teacher (~800K samples)
4. QLoRA + RLVR (GRPO with verifiable rewards)
5. SPIN self-play iterations

Total: ~15-30B tokens, ~3-5 months continuous on RTX 5060 Ti.

---

## 7. Quantization

- SpinQuant rotation before quantization (45.1% gap reduction vs QuaRot)
- Mixed-precision per-expert: top 30% INT4, mid 40% E8 lattice 2BPW, bottom 30% ternary/pruned
- KV cache KIVI 2-bit (6-8x compression)
- Monarch matrices for expert compression (22x reduction)

---

## 8. VRAM budget (conservative, 8K context)

| Component | Size |
|-----------|------|
| Shared weights (FP4) | 2.0 GB |
| Hot expert cache | 3.0 GB |
| Expert transfer buffer | 0.4 GB |
| KV cache (2-bit) | 0.03 GB |
| Engram GPU | 0.85 GB |
| DFlash drafter | 0.5 GB |
| Activations + overhead | 1.1 GB |
| **Total** | **~8 GB** |
| **Headroom** | **~8 GB** |

---

## 9-12. Inference, RAG, Math connections, Benchmarks

See full document for details on:
- SM120 kernel constraints (no TMEM, uses extended mma.sync)
- Rust + FFI architecture
- HippoRAG 2 for graph-based retrieval
- Energy-Based Transformers connection to entropy routing
- Benchmark targets: 72-78 MMLU-Pro, 68-75 GPQA, 70-80 HumanEval

---

*Source: Claude App architecture proposal, March 27 2026*
*Based on 80+ papers through March 2026*
