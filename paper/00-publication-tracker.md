# Chimère Publication Tracker — Mars 2026

## Papers par priorité

| # | Paper | Novelty | Prêt | Venue cible | Status |
|---|-------|---------|------|-------------|--------|
| 1 | **Chimère System Paper** | ★★★★ | 80% | MLSys/OSDI/arXiv | Outline done |
| 2 | **DFlash MoE + GDN Barrier** | ★★★★ | 75% | NeurIPS Workshop | Outline done |
| 3 | **Engram Multi-Tier Consumer HW** | ★★★★ | 60% | EMNLP Demo | TODO |
| 4 | **Self-Improving Quality Loop** | ★★★ | 70% | AAAI AI-HRI | TODO |
| 5 | **Expert Prefetch Negative Result** | ★★★ | 85% | Negative Results Workshop | TODO |
| 6 | **ABF Quantized Reasoning Budget** | ★★★ | 50% | arXiv short | TODO |
| 7 | **Entropy Router AR↔Diffusion** | ★★★★★ | 30% | ICML (when implemented) | Concept only |

## Paper 1: Chimère System Paper
- **File:** 01-chimere-system-paper-outline.md
- **Key result:** 10/10 benchmark, 93 tok/s, $0.10/day, self-improving
- **What's missing:** 30-day longitudinal data, formal ablation study
- **Estimated effort:** 2 weeks writing + 1 month data collection

## Paper 2: DFlash MoE + GDN Barrier
- **File:** 02-dflash-moe-gdn-barrier-outline.md
- **Key result:** τ=9.4 > paper's 6.4 (+47%), GDN state barrier formalized
- **What's missing:** Formalize GDN/seq_rm proof, minimal reproducer
- **Data available:** 72,908 blocks evaluated, training curves, 4 architecture versions
- **Estimated effort:** 1 week writing (data exists)

## Paper 3: Engram Multi-Tier for Consumer Hardware
- **Key result:** 4-tier Engram (Cuckoo → FAISS → N-gram → FHRR), 20MB kine table, O(1)+<10ns
- **Novel:** Multi-tier extension of DeepSeek Engram with domain routing
- **What's missing:** Ablation (with/without each tier), perplexity impact measurement
- **Estimated effort:** 1 week ablation + 1 week writing

## Paper 4: Self-Improving Quality Loop
- **Key result:** ThinkPRM scoring → training pairs → nightly LoRA/SPIN → Engram WRITE
- **Novel:** The model improves from production Telegram conversations while sleeping
- **What's missing:** 30+ days longitudinal data showing quality improvement
- **Estimated effort:** 1 month passive data + 1 week writing

## Paper 5: Expert Prefetch — Informative Negative Result
- **Key result:** MLP predictor 86.65% hit@8, but prefetch USELESS (CPU GEMV < 45% budget)
- **Novel:** Shows that expert prefetch is counter-productive on current hardware
- **Data available:** Trained predictor, A/B benchmarks, timing breakdown
- **Estimated effort:** 3 days writing (all data exists)

## Paper 6: ABF — Adaptive Budget Forcing for Quantized Reasoning
- **Key result:** ABF + CGRS + entropy routing save 15-30% thinking tokens
- **Novel:** Budget forcing adapted for quantized MoE with quality gate feedback
- **What's missing:** Formal benchmarks (MATH, GSM8K) with/without ABF
- **Estimated effort:** 1 week benchmarks + 1 week writing

## Paper 7: Entropy Router AR↔Diffusion (MOST NOVEL)
- **Key result:** No paper exists implementing this (confirmed Mar 2026)
- **Novel:** Route individual tokens between AR and block diffusion by Shannon entropy
- **What's missing:** Full implementation (DFlash drafter integration needed)
- **Estimated effort:** 1 month implementation + 2 weeks writing
- **Impact:** Could be the standout contribution if results are strong

---

## Shared Resources
- Hardware: RTX 5060 Ti 16GB, i5-14600KF, 32GB DDR5
- Model: Qwen3.5-35B-A3B IQ3_S custom-mix (14.71 GB)
- Codebase: chimere-rewrite (Rust), Chimère (Python)
- All code open-source ready
