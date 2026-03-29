# Chimère: A Self-Improving MoE Language Model System for Consumer GPU Hardware

## Paper Outline — System Paper (MLSys / OSDI / arXiv:cs.CL)

### Abstract
We present Chimère, an integrated system for running a 35B-parameter Mixture-of-Experts language model on a single RTX 5060 Ti (16GB VRAM, 32GB RAM) at 93 tokens/second, with continuous self-improvement through a nightly quality loop. The system combines five novel subsystems: (1) Multi-tier Engram memory with Cuckoo filter pre-screening, (2) entropy-based request classification for adaptive compute allocation, (3) Diverse Verifier Tree Search (DVTS) with step-level PRM scoring, (4) automatic tool injection from pipeline YAML definitions, and (5) a quality-gated training data pipeline feeding SPIN self-play and GRPO reinforcement learning. On a 10-question benchmark spanning physiotherapy, code generation, mathematics, and tool calling, Chimère achieves 100% pass rate with domain-specific Engram tables providing factual recall that matches frontier API models on covered domains. We release the full system as open source.

### 1. Introduction
- The gap between frontier API models and local deployment
- Consumer GPU constraints (16GB VRAM, PCIe 4.0 x8)
- Thesis: an integrated system of complementary techniques can close the gap on domain-specific tasks while maintaining privacy and $0.10/day operating cost
- Key result: 10/10 benchmark (was 5/10 before system integration)

### 2. System Architecture
- Architecture diagram: Telegram → Chimère → ODO → chimere-server → ThinkPRM
- Five subsystems overview
- Design principles: compositionality, graceful degradation, nightly self-improvement

### 3. Multi-Tier Engram Memory
- 3.1 N-gram hash tables (O(1) lookup, mmap'd)
- 3.2 Cuckoo filter Tier 0 (skip 97% lookups, <10ns)
- 3.3 FAISS semantic few-shot (Qwen3-Embedding, ~5ms)
- 3.4 Domain tables: kine (20MB, 314K n-grams), code (42MB), cyber (172KB)
- 3.5 Quality-gated nightly WRITE (auto-enrichment from production traffic)
- Ablation: with vs without Engram on kine benchmark

### 4. Adaptive Compute Allocation
- 4.1 Entropy router: 3-component heuristic (query complexity + classifier confidence + historical quality)
- 4.2 ABF (Adaptive Budget Forcing): thinking budget management
- 4.3 DVTS: K-candidate generation + ThinkPRM step-level scoring
- 4.4 GateSkip: scalar gates for layer-level compute skip
- 4.5 Interaction: entropy class determines thinking mode + DVTS activation

### 5. Tool Integration and Multi-Agent Pipeline
- 5.1 Automatic tool injection from pipeline YAML
- 5.2 Sequential multi-agent pipeline executor
- 5.3 4-step kine workflow: evidence → diagnostic → protocol → dosage

### 6. Self-Improvement Loop
- 6.1 Quality gate: ThinkPRM-1.5B step-level verifier
- 6.2 Training data pipeline: quality_scores.jsonl → training_pairs.jsonl
- 6.3 SPIN self-play: generate model responses, create DPO pairs
- 6.4 GRPO with verifiable rewards (code execution, JSON schema, ThinkPRM)
- 6.5 DSPy MIPROv2 weekly optimization
- 6.6 Engram WRITE nightly

### 7. Quantization and Inference
- 7.1 IQ3_S custom-mix with importance matrix (14.71 GB, 3.56 BPW)
- 7.2 ik_llama sm120 build (+23% throughput vs stock)
- 7.3 QuaRot Hadamard rotation for MoE (first on Qwen3.5 MoE)
- 7.4 NVFP4 analysis: why it fails on micro-experts (negative result)
- 7.5 GDN hybrid architecture: 3:1 ratio, 4× KV cache reduction

### 8. Experimental Results
- 8.1 Benchmark suite: 10 questions (kine, code, math, tools)
- 8.2 Before vs after system integration: 50% → 100%
- 8.3 Throughput: 93 tok/s sustained, ~20s for code, ~157s for kine (with thinking)
- 8.4 Quality scores: distribution analysis (2.83 → improved after ThinkPRM fix)
- 8.5 System uptime and nightly pipeline reliability
- 8.6 Comparison: what Opus 4.6 can vs cannot do that we match

### 9. Lessons Learned
- LoRA on 35B MoE doesn't fit 16GB even in 4-bit
- NVFP4/MXFP4 ineffective on 256 micro-experts
- DART Engram drafter incompatible with GDN seq_rm (3 attempts)
- Expert prefetch predictor (86% hit@8) is useless when CPU GEMV < 45% budget
- ThinkPRM markdown extraction critical for medical domain scoring
- np=1 server contention is the real bottleneck for self-improvement

### 10. Related Work
- KTransformers (MoE offloading)
- DeepSeek Engram (conditional memory)
- DFlash (block diffusion drafting)
- Autoresearch (Karpathy, self-improving)
- SpinQuant/QuaRot (rotation-based quantization)

### 11. Conclusion and Future Work
- Entropy routing AR↔diffusion (novel, no existing paper)
- DFlash integration for 400+ tok/s
- Cloud GPU for SPIN/GRPO training
- ARC-AGI-2 benchmark

### Appendix
- A. Full system configuration (services, ports, timers)
- B. Complete benchmark results
- C. VRAM budget breakdown
- D. Hardware specifications
