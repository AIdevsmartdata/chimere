# Chimère Roadmap Q2-Q3 2026 — v4 (28 mars, post-marathon)

*Kevin (Tech Lead) & Claude Opus 4.6 (Lead Dev).*
*Intègre: 3 rapports recherche Opus, ablation Engram, logprobs FFI, 11 rapports domaine.*

---

## Situation au 28 mars 2026

| Métrique | v3 (27 mars matin) | v4 (28 mars) | Delta |
|----------|-------------------|--------------|-------|
| Benchmark ODO | 50% (5/10) | **100% (10/10)** | +50pp |
| Engram | v1 α=0.35 (nuit) | **v2 α=0.1 NEST response-only (88%)** | +11pp ablation |
| Logprobs | ❌ 0.0 partout | **✅ top-5 vrais log-softmax** | Débloqué |
| Quality scores | 19 | **69** | +50 |
| Training pairs | 50 | **68** | +18 |
| SPIN DPO pairs | 2 | **72** | +70 |
| Research reports | 0 | **11 deep + 3 Opus** | Nouveau |
| Modes ODO | — | **fast/quality/ultra** | Nouveau |
| Cross-encoder | — | **CPU 149M gte-modernbert** | Nouveau |
| Commits session | 0 | **18+** | ~12K lignes |

### Architecture prod v4

```
Telegram → Chimère Gateway (25443)
                ↓
          ODO (8084) ← mode fast/quality/ultra
                ↓           ├── Entropy router (3 composantes, seuil 0.52)
                ↓           ├── Confidence RAG trigger (logprob probe si web:false)
                ↓           ├── FAISS Semantic Few-shot (35 entries, skip factual)
                ↓           ├── RAG ChromaDB hybrid (dense+BM25+RRF+cross-encoder)
                ↓           ├── Web search SOTA (deep_search_sota + cross-encoder reranker)
                ↓           ├── Dynamic Engram (web→.engr per-query, quality/ultra only)
                ↓           ├── DVTS tree search (K=2/4, ThinkPRM scoring)
                ↓           ├── Multi-agent pipeline (4-step kine, opt-in)
                ↓           └── Tool injection auto (post-enrichment)
          chimere-server (8081) ← FFI ik_llama + Engram v2
                ↓                    ├── MultiEngram 3 tables + Cuckoo Tier 0
                ↓                    ├── NEST adaptive alpha (per-token RRC)
                ↓                    ├── Response-only bias (no thinking)
                ↓                    └── Logprobs FFI (top-5 log-softmax)
          ThinkPRM (8085, CPU) ← Step-level verifier async
                ↓
          quality_scores.jsonl + training_pairs.jsonl
                ↓ (nightly)
          Engram WRITE (04:00) → DSPy (lun 02:00) → RAG reindex (6h)
```

---

## Findings critiques session 27-28 mars

### Ablation Engram (mesuré)
```
Engram v1 (α=0.35, think+response):  77% ← NUIT
Engram OFF (α=0):                    85%
Engram v2 (α=0.1, response-only):   88% ← PRODUCTION
```
**Root causes v1** : bias pendant thinking, alpha trop haut, context 5 tokens, tables depuis documents pas réponses.

### Pipeline recherche (audit Opus)
- Tool injection SUPPRIMAIT l'enrichissement web (fixé)
- Web search timeout 30s (standard prend 60s) → 120s
- CRAG trop agressif (1/4 chunks survit)
- Few-shot sur factuel DÉGRADE (63%→11%) → skip factual

### LoRA sur 35B MoE 16GB (3 solutions identifiées)
1. **transformers-qwen3-moe-fused** — charge GGUF direct, Triton fused MoE, prouvé 16GB
2. **MeZO** — zeroth-order optimizer, même mémoire que inférence, pas de backward
3. **DeepSpeed ZeRO-2** — CPU offload optimizer states

### Quantification SOTA (3 solutions layer-by-layer)
1. **KurtBoost** — kurtosis par tenseur → tensor overrides data-driven, 0 GPU
2. **AutoRound Intel** — layer-wise optimization → GGUF direct, 16GB OK
3. **KurTail** — rotations APPRISES (vs QuaRot random), +15% PPL

### Logprobs + Engram (25 idées brainstormées)
- **C1**: Logprob-gated alpha (model confidence gates Engram) — 2h
- **B1**: DART Engram spec-dec (draft_sequence → batch verify) — 1 jour
- **J1**: Dynamic Engram from web (logit-level knowledge injection) — 2 jours, NOVEL
- **J4**: Self-growing Engram from inference logprobs — 6h, NOVEL

---

## Roadmap révisée v4

### SEMAINE 1 restante (28 mars - 2 avril) — LORA + QUANTIFICATION

**LoRA local (Plan A)** :
- [ ] Tester transformers-qwen3-moe-fused sur notre IQ3_S GGUF
- [ ] Si Qwen3.5 pas supporté → fallback MeZO (attention-only LoRA)
- [ ] Pipeline nightly : LoRA → merge → export GGUF adapter
- [ ] Timer 03:30 : stop chimere → train → restart

**Quantification améliorée** :
- [ ] KurtBoost : kurtosis par tenseur shard-by-shard (script Python, 30 min)
- [ ] Générer tensor overrides data-driven (remplace 317 overrides manuels)
- [ ] imatrix via Q8_0 rotaté + calibration FR domaine (1M+ tokens)
- [ ] Re-quantifier IQ3_S custom-mix v2 avec overrides KurtBoost

**Logprob quick wins** :
- [ ] C1 : logprob-gated alpha dans NEST (2h, 1 ligne)
- [ ] A3 : hallucination detector (confidence annotation streaming, 6h)

### SEMAINE 2 (3-9 avril) — ENGRAM SPEC-DEC + AUTOROUND

**DART Engram speculative decoding** :
- [ ] B1 : draft_sequence() → forward_prefill() batch verify (1 jour)
- [ ] E1 : logprob verification probabilistic acceptance (6h)
- [ ] Objectif : 1.3-1.5× speedup domaine kiné

**AutoRound Intel GGUF** :
- [ ] AutoScheme(avg_bits=3.5, IQ3_S+Q5_K_S) sur BF16-rotated
- [ ] Comparer PPL vs IQ3_S custom-mix actuel
- [ ] Si gain : swap en production

**Self-improving Engram** :
- [ ] J4 : grow Engram from inference logprobs (tokens haute confiance)
- [ ] C2 : disagreement logger (modèle vs Engram → training data ciblé)

### SEMAINE 3 (10-16 avril) — DYNAMIC ENGRAM + STRATEGY ROUTER

**Dynamic Engram from web search (NOVEL)** :
- [ ] J1 : web search → tokenize → build temp .engr → logit bias
- [ ] In-memory Engram builder (pas fichier)
- [ ] Comparer : RAG seul vs RAG + Dynamic Engram

**Per-token strategy router v2** :
- [ ] F1 : combine entropy + Engram confidence + logprobs
- [ ] Actions : skip forward (Engram draft), greedy AR, sampled AR, DVTS branch, pause+search
- [ ] ABF from logprob entropy (pas heuristique)

**KurTail rotations apprises** :
- [ ] Layer-wise forward pass pour collecte activations
- [ ] Cayley SGD par couche (quarot_rotate.py comme base)
- [ ] Re-quantifier avec rotations apprises

### SEMAINE 4-5 (17-30 avril) — DFLASH + BLOCK DIFFUSION

**DFlash drafter** :
- [ ] Intégrer z-lab/Qwen3.5-35B-A3B-DFlash
- [ ] Hybrid : Engram draft (facile) + DFlash draft (dur)
- [ ] Entropy-driven block boundaries (J2)

**Consensus speculative decoding** :
- [ ] J3 : MTP + Engram + Expert predictor triple-source
- [ ] Tokens dans l'intersection → >90% acceptance

### MOIS 2-3 — COMPRESSION + INTELLIGENCE + PAPERS

**Expert compression** : FloE, D²-MoE delta, Monarch matrices, RMT pruning
**Intelligence** : GenPRM, rStar-Math MCTS, HippoRAG 2
**Engram 50GB médical** : Common Crawl filtré → suffix array → domain tables
**Papers** : System paper + DFlash MoE + Dynamic Engram (3 submissions)

---

## Publications identifiées

| # | Paper | Novelty | Status |
|---|-------|---------|--------|
| 1 | **Chimère System Paper** | ★★★★ | Outline done |
| 2 | **DFlash MoE + GDN Barrier** | ★★★★ | Outline done |
| 3 | **Dynamic Engram: Logit-Level Knowledge Beyond Context** | ★★★★★ | Concept validé, novel confirmé |
| 4 | **NEST + Logprob-Gated Alpha** | ★★★★ | Implémenté, needs benchmark |
| 5 | **Self-Growing Engram from Inference Logprobs** | ★★★★ | Concept, 6h implem |
| 6 | **Per-Token Strategy Router** | ★★★★ | Framework en place |
| 7 | **Expert Prefetch Negative Result** | ★★★ | Data exists |

---

## Benchmark tracker

| Date | KINE | CODE | MATH | TOOLS | TOTAL | tok/s | Engram | Notes |
|------|------|------|------|-------|-------|-------|--------|-------|
| 25 mars | 1/2 | 2/2 | 1/1 | 2/2 | 7/8 (88%) | 83 | OFF | Baseline |
| 26 mars | 5/5 | 2/2 | 1/1 | 2/2 | 10/10 (100%) | 83 | OFF | ODO fixes |
| 27 mars (8081) | — | — | — | — | — | 93 | v1 α=0.35 | 77% ablation |
| 27 mars (8081) | — | — | — | — | — | 93 | OFF α=0 | 85% ablation |
| **27 mars (8081)** | — | — | — | — | — | 93 | **v2 α=0.1** | **88% ablation** |
| 27 mars quality | 5/5 | — | — | — | — | ~80 | v2+ODO | 100% quality mode |

---

*"The system improves in its sleep. Each token it generates teaches it what it doesn't know."*
*— Chimère v4 design philosophy*
