# Benchmark Qwen3.5 — RTX 5060 Ti 16GB
Date: 2026-03-14 (màj — ajout 27B Opus-Distilled IQ4_XS dense)

## Vitesses et VRAM

| Modele | Quant | Taille | Runtime | ncmoe/ngl | Slots | Ctx total | Gen tok/s | Prompt tok/s | VRAM utilise | VRAM libre |
|--------|-------|--------|---------|-----------|-------|-----------|-----------|-------------|-------------|------------|
| **27B Opus-Distilled** | **IQ4_XS (i1)** | **13.67G** | **ik_llama sm120** | **ngl 99** | **1** | **~8K** | **25.65** | **1067** | **~14.7G** | **~1.1G** |
| **27B Opus-Distilled** | **IQ4_XS (i1)** | **13.67G** | **stock latest** | **ngl 99** | **1** | **~0.5K** | **20.8** | **179.5** | **~14G** | **~1.5G** |
| **27B Opus-Distilled** | **IQ3_S custom-mix (ik_llama)** | **13.02G** | **ik_llama sm120** | **ngl 99** | **1** | **~12K** | **24.17** | **835.9** | **~13.2G** | **~2.6G** |
| **27B Opus-Distilled** | **IQ3_S MTP (stock quant)** | **11.42G** | **stock patched** | **ngl 64** | **1** | **4K** | **17.2** | **57.1** | **~11.7G** | **~3.7G** |
| **27B Opus-Distilled** | **IQ4_XS+Q6K+Q8 MTP (custom)** | **16G** | **stock patched** | **ngl 64** | **1** | **~0.5K** | **13.4** | **52.4** | **~15.5G** | **~0.3G** |
| **27B Opus-Distilled** | **IQ4_XS (i1)** | **13.67G** | **stock b8125** | **ngl 99** | **1** | **~8K** | **23.41** | **962** | **~14.7G** | **~1.1G** |
| **9B dense** | **Q8_0** | **8.9G** | **ik_llama sm120** | **ngl 99** | **1** | **64K** | **44.4** | **3370** | **10.0G** | **5.8G** |
| **9B dense** | **Q8_0** | **8.9G** | **stock b8125 sm120** | **ngl 99** | **1** | **64K** | **41.1** | **2815** | **~10G** | **~5.8G** |
| **35B-A3B (MoE)** | **IQ3_S custom-mix** | **15G** | **ik_llama sm120** | **ncmoe 2** | **1** | **10K** | **98.6** | **1357** | **15.2G** | **641M** |
| **35B-A3B (MoE)** | **IQ3_S custom-mix "Masterpiece"** | **15G** | **stock b8125** | **ncmoe 4** | **2** | **128K** | **72 (solo) / 45 (dual)** | **228 (solo) / 159 (dual)** | **15.2G** | **~1G** |
| 35B-A3B (MoE) | IQ3_S | 13G | ik_llama sm120 | 0 | 3 | 196K (3x64K) | **91** | **~180** | ~15.8G | ~500M |
| 35B-A3B (MoE) | IQ3_S | 13G | ik_llama sm120 | 0 | 1 | 64K | **91** | **184** | ~14.5G | ~1.8G |
| 35B-A3B (MoE) | IQ3_S (Unsloth UD) | 13G | stock b8125 | 0 | 3 | 196K (3x64K) | 69 | ~150 | ~15.8G | ~500M |
| **35B-A3B (MoE)** | **IQ3_S custom-mix** | **15G** | **stock b8125** | **ncmoe 6** | **3** | **196K (3x64K)** | **63.9** | **144.5** | **15.4G** | **417M** |
| 35B-A3B (MoE) | Q4_K_M | 21G | ik_llama sm120 | ncmoe 14 | 1 | **64K** | **62.3** | **314** | **15.1G** | **348M** |
| 35B-A3B (MoE) | Q4_K_M | 21G | ik_llama sm120 | ncmoe 14 | 2 | 32K (2x16K) | 61.4 | 187 | 15.4G | 476M |
| 35B-A3B (MoE) | Q4_K_M | 21G | ik_llama sm120 | ncmoe 14 | 1 | 16K | 61.4 | 187 | 15.2G | 669M |
| 35B-A3B (MoE) | Q4_K_M | 21G | stock b8125 | ncmoe 14 | 2 | 32K (2x16K) | 52 | 104 | 15.4G | 470M |
| 35B-A3B (MoE) | Q4_K_M | 21G | stock b8125 | ncmoe 14 | 1 | 16K | 52 | 104 | 15.2G | 679M |
| 35B-A3B (MoE) | Q4_K_M | 21G | stock b8125 | ncmoe 12 | 1 | 16K | - | - | OOM | OOM |
| **35B-A3B (MoE)** | **Q5_K_XL** | **25G** | **ik_llama sm120** | **ncmoe 18** | **1** | **64K** | **49.9** | **257** | **15.0G** | **454M** |
| 35B-A3B (MoE) | Q5_K_XL | 25G | stock b8125 | ncmoe 20 | 1 | 64K | 42 | 97 | ~14.6G | ~1.2G |
| ~~27B dense~~ | ~~IQ4_NL~~ | ~~15G~~ | ~~ik_llama sm120~~ | ~~ngl 99~~ | ~~1~~ | ~~4K~~ | ~~23.1~~ | ~~188~~ | ~~15.6G~~ | ~~208-526M~~ |

## Tool-Calling

| Modele | Quant | Runtime | Slots | Score | Notes |
|--------|-------|---------|-------|-------|-------|
| **35B-A3B** | **IQ3_S custom-mix "Masterpiece"** | **stock b8125** | **2** | **PASS** | **Référence SOTA (Q8/Q4 cache)** |
| 35B-A3B | IQ3_S | ik_llama sm120 | 1 | 20/20 | Parfait en solo |
| 35B-A3B | Q4_K_M | ik_llama sm120 | 1 | 3/3 | Parfait en solo |
| 35B-A3B | Q4_K_M | stock b8125 | 2 | 10/10 concurrent | Zero contamination |
| **35B-A3B** | **IQ3_S custom-mix** | **stock b8125** | **3** | **PASS** | **Tool call OK concurrent** |
| 35B-A3B | Q5_K_XL | stock b8125 | 1 | OK | Production actuelle |

## Concurrent Slots (contamination system prompt)

| Quant | Runtime | Slots | Slot 0 | Slot 1 | Slot 2 | Total | Verdict |
|-------|---------|-------|--------|--------|--------|-------|---------|
| **IQ3_S custom-mix "Masterpiece"** | **stock b8125** | **2** | **OK** | **OK** | - | **OK** | **STABLE 128K** |
| IQ3_S | ik_llama sm120 | 3 | 0/5 | 1/5 | 4/5 | 5/15 | BROKEN |
| IQ3_S | ik_llama sm120 | 3 | 2/9 (stress) | - | - | 2/9 | BROKEN |
| Q4_K_M | ik_llama sm120 | 2 | 0/5 | 4/5 | - | 4/10 | BROKEN |
| **Q4_K_M** | **stock b8125** | **2** | **5/5** | **5/5** | - | **10/10** | **OK** |
| IQ3_S (Unsloth UD) | stock b8125 | 3 | OK | OK | OK | OK | **OK** |
| **IQ3_S custom-mix** | **stock b8125** | **3** | **OK** | **OK** | **OK** | **3/3** | **OK** |

## Delta ik_llama vs stock (meme config)

| Metrique | ik_llama sm120 | stock b8125 sm120 | Delta |
|----------|---------------|-------------|-------|
| Gen tok/s (27B Opus IQ4_XS dense) | 25.65 | 23.41 | **+9.6%** |
| Prompt tok/s (27B Opus IQ4_XS dense) | 1067 | 962 | **+11%** |
| Gen tok/s (9B Q8_0 dense) | 44.4 | 41.1 | **+8%** |
| Prompt tok/s (9B Q8_0 dense) | 3370 | 2815 | **+20%** |
| Gen tok/s (Q4_K_M 35B MoE) | 61.4 | 52.0 | **+18%** |
| Prompt tok/s (Q4_K_M 35B MoE) | 187 | 104 | **+80%** |
| Gen tok/s (IQ3_S 35B MoE) | 91 | 69 | **+32%** |
| Gen tok/s (Q5_K_XL ncmoe18 vs ncmoe20) | 49.9 | 42 | **+19%** |
| Prompt tok/s (Q5_K_XL) | 257 | 97 | **+165%** |
| Concurrent fiabilite | BROKEN | 10/10 | ik_llama = bug |

**Note 9B dense** : ik_llama gagne aussi sur modèle dense (+8% TG, +20% PP) mais l'écart est plus faible que sur MoE (+18-32% TG). L'avantage ik_llama vient surtout des kernels MoE optimisés.

**Note 27B Opus-Distilled** : ik_llama gagne +9.6% TG et +11% PP sur ce dense 27B — cohérent avec le delta 9B dense (+8/+20%). Le 27B à 25.65 tok/s est **3.6x plus lent** que le 35B-A3B MoE (93 tok/s) car il active 27B params/token vs ~3B pour le MoE. Pas de tenseurs MTP dans le GGUF mradermacher (strippés à la quantification). Contexte max limité ~8K en full GPU (1.1 Go libre pour KV).

## IQ3_S Custom-Mix — Tests KV Cache (2026-03-09) ★ PRODUCTION MISE À JOUR

### Résultats réels (llama-bench r=3 + 15/15 validation qualité)

| Config | ncmoe | ctk/ctv | TG bench | PP bench | VRAM libre (96K) | Qualité | Statut |
|---|---|---|---|---|---|---|---|
| Ancienne prod | 2 | q4_0/q4_0 | 103.4 t/s | 1608 t/s | 257 MB | 15/15 | remplacée |
| **PRODUCTION** | **4** | **q8_0/q4_0** | **93.2 t/s** | **1182 t/s** | **562 MB** ✅ | **15/15** | **✅ ACTUELLE** |
| Test bf16 full | 6 | bf16/bf16 | 83.5 t/s ±6.5 | 885 t/s | 137 MB ⚠️ | 15/15 | ❌ trop risqué |

**Bilan test bf16/bf16 :**
- 15/15 mais variance élevée (±6.5 tok/s) — probablement dû aux transferts CPU-GPU ncmoe=6
- 137 MB VRAM libre = trop serré pour prod (pic compute possible → OOM)
- Qualité identique aux tests courts — différence bf16 n'est visible que >32K contexte rempli
- **q8_0/q4_0 est le sweet spot** : qualité Masterpiece level, 568 MB libres, viable

**Gains vs ancienne production :**
- VRAM : 257 → **568 MB libres** (+120%) — marge de sécurité doublée
- KV qualité : q4_0 keys → **q8_0 keys** — erreur 4-bit éliminée sur les clés d'attention
- TG : 103 → **93 tok/s** (-10%) — trade-off acceptable
- PP : 1608 → **1182 tok/s** (-26%) — impact prompt ingestion

**Paramètres VRAM réels mesurés (≠ estimations) :**
- ncmoe=4 q8_0/q4_0 96K : **15281 MiB used, 562 MiB free** (mesuré)
- ncmoe=6 bf16/bf16 96K : **15706 MiB used, 137 MiB free** (mesuré)
- Δ = +425 MiB pour bf16/bf16 vs q8_0/q4_0 (net après ncmoe savings)

---

## IQ3_S Custom-Mix "Masterpiece" — Synthèse (2026-03-08)

Le réglage ultime pour RTX 5060 Ti 16GB. Combine architecture GDN optimisée, quantification mixée expert et cache haute qualité.

| Metrique | IQ3_S Standard | **IQ3_S Masterpiece** | Delta |
|----------|---------------|----------------------|-------|
| Ctx total | 64K | **128K** | **x2** |
| Cache KV | q4_0 | **q8_0 Key / q4_0 Value** | **Qualité Unsloth** |
| Gen tok/s (solo) | 69 | **72.1** | **+4%** |
| Gen tok/s (dual) | - | **45.2** | **SOTA** |
| Prompt tok/s (solo) | 150 | **228** | **+52%** |
| ncmoe | 0 | **4** | Offload stratégique |
| Buffers | b 2048 | **b 1024 / ub 512** | **Agentic optim** |

## Recommandations Finales

### Option S (SOTA - RECOMMANDE) : IQ3_S custom-mix "Masterpiece"
- **2 slots, 128K ctx total, 72.1 tok/s, ncmoe 4**
- Cache KV q8_0/q4_0 (Zéro dégradation raisonnement long contexte)
- Buffers optimisés (b 1024/ub 512) pour réactivité agentique
- **Le meilleur équilibre Contexte / Qualité / Vitesse**

### Option A : IQ3_S custom-mix stock — Multi-agent 3 slots
- 3 slots, 196K ctx total (64K/slot), 63.9 tok/s, ncmoe 6
- **Idéal pour 3 agents parallèles légers**

### Option B : Q4_K_M 35B-A3B ik_llama solo — Vitesse brute
- 1 slot, 16K ctx, 61.4 tok/s gen, 187 tok/s prompt
- **Uniquement pour usage solo haute qualité ultra-rapide**

### Option C : Q5_K_XL stock — Qualité maximale
- 1 slot, 64K ctx, 42 tok/s
- **Pour tâches de raisonnement pur sans besoin de vitesse**

### Option D (à tester) : IQ3_S custom-mix ik_llama + bf16k/q4v — PRODUCTION AMÉLIORÉE
- ncmoe=4, np=1, ctx=96K, ctk=bf16, ctv=q4_0
- VRAM estimée libre : ~843 MB (safe)
- Vitesse estimée : ~89 tok/s (ik_llama +23% vs stock 72 tok/s ncmoe=4)
- **Meilleure qualité attention long contexte sans sacrifice vitesse majeur**

---

## Benchmarks Officiels Qwen3.5-35B-A3B (2026-03 — source: unsloth.ai)

Comparaison avec modèles concurrents. Colonne 35B-A3B = notre modèle (IQ3_S custom-mix).

| Benchmark | GPT-5-mini | GPT-OSS-120B | Qwen3-235B | Qwen3.5-122B | Qwen3.5-27B | **Qwen3.5-35B-A3B** |
|---|---|---|---|---|---|---|
| **Knowledge** |
| MMLU-Pro | 83.7 | 80.8 | 84.4 | 86.7 | 86.1 | **85.3** |
| MMLU-Redux | 93.7 | 91.0 | 93.8 | 94.0 | 93.2 | **93.3** |
| C-Eval | 82.2 | 76.2 | 92.1 | 91.9 | 90.5 | **90.2** |
| SuperGPQA | 58.6 | 54.6 | 64.9 | 67.1 | 65.6 | **63.4** |
| **Instruction Following** |
| IFEval | 93.9 | 88.9 | 87.8 | 93.4 | **95.0** | 91.9 |
| IFBench | 75.4 | 69.0 | 51.7 | 76.1 | **76.5** | 70.2 |
| MultiChallenge | 59.0 | 45.3 | 50.2 | **61.5** | 60.8 | 60.0 |
| **Long Context** |
| AA-LCR | 68.0 | 50.7 | 60.0 | **66.9** | 66.1 | 58.5 |
| LongBench v2 | 56.8 | 48.2 | 54.8 | **60.2** | **60.6** | 59.0 |
| **STEM & Reasoning** |
| HLE w/ CoT | 19.4 | 14.9 | 18.2 | **25.3** | 24.3 | 22.4 |
| GPQA Diamond | 82.8 | 80.1 | 81.1 | **86.6** | 85.5 | 84.2 |
| HMMT Feb 25 | 89.2 | 90.0 | 85.1 | **91.4** | **92.0** | 89.0 |
| HMMT Nov 25 | 84.2 | 90.0 | 89.5 | 90.3 | 89.8 | 89.2 |
| **Coding** |
| SWE-bench Verified | 72.0 | 62.0 | — | 72.0 | **72.4** | 69.2 |
| Terminal Bench 2 | 31.9 | 18.7 | — | **49.4** | 41.6 | 40.5 |
| LiveCodeBench v6 | 80.5 | **82.7** | 75.1 | 78.9 | 80.7 | 74.6 |
| CodeForces | 2160 | 2157 | 2146 | 2100 | 1899 | **2028** |
| OJBench | 40.4 | **41.5** | 32.7 | 39.5 | 40.1 | 36.0 |
| FullStackBench en | 30.6 | 58.9 | 61.1 | **62.6** | 60.1 | 58.1 |
| FullStackBench zh | 35.2 | 60.4 | **63.1** | 58.7 | 57.4 | 55.0 |
| **General Agent** |
| BFCL-V4 | 55.5 | — | 54.8 | **72.2** | 68.5 | 67.3 |
| TAU2-Bench | 69.8 | — | 58.5 | 79.5 | 79.0 | **81.2** |
| VITA-Bench | 13.9 | — | 31.6 | 33.6 | **41.9** | 31.9 |
| DeepPlanning | 17.9 | — | 17.1 | **24.1** | 22.6 | 22.8 |
| **Search Agent** |
| HLE w/ tool | 35.8 | 19.0 | — | 47.5 | **48.5** | 47.4 |
| Browsecomp | 48.1 | 41.1 | — | **63.8** | 61.0 | 61.0 |
| WideSearch | 47.2 | 40.4 | — | 60.5 | **61.1** | 57.1 |
| **Multilingualism** |
| MMMLU | 86.2 | 78.2 | 83.4 | **86.7** | 85.9 | 85.2 |
| MMLU-ProX | 78.5 | 74.5 | 77.9 | 82.2 | **82.2** | 81.0 |
| MAXIFE | 85.3 | 83.7 | 83.2 | **87.9** | **88.0** | 86.6 |

### Analyse pour notre usage (IQ3_S, 3.5B actifs)

**Points forts du 35B-A3B :**
- **TAU2-Bench 81.2** : meilleur de tous → agentic long horizon, notre cas Chimère ✅
- **GPQA Diamond 84.2** : raisonnement scientifique excellent
- **CodeForces 2028** : coding compétitif = bon pour les skills code
- **BFCL-V4 67.3** : tool-calling correct (27B=68.5, 122B=72.2 font mieux)

**Points à surveiller :**
- **BFCL-V4 67.3** : 4-5 pts sous 27B/122B → erreurs tool-calling sur edge cases → compensé par notre imatrix BFCL custom
- **Long Context (AA-LCR 58.5)** : plus faible que 122B/27B → **bf16 KV cache aide ici**
- **JSON long (FullStack 58.1)** : honnête mais pas le meilleur → batch par 20-30 items, temp 0.3-0.5

**Notre IQ3_S custom-mix vs Unsloth UD :**
- Imatrix calibré sur BFCL v3 (418 exemples tool-calling) → compense le déficit BFCL de base
- Q8_0 sur output.weight + attn_v → préserve la précision logits (JSON output, token choix)
- Tests locaux : 20/20 tool-calling (vs 20/20 Unsloth) mais meilleur raisonnement (MoT+Codeforces)

---

## 27B Opus-Distilled IQ4_XS — Benchmark (2026-03-14)

Source : `mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF`
Quant : IQ4_XS imatrix, 13.67 GiB, 4.25 BPW, 851 tenseurs, **0 tenseurs MTP**
Architecture : dense 27B (64 layers full attention, pas de GDN/SSM)

### Résultats

| Runtime | Test | tok/s |
|---|---|---|
| ik_llama sm120 | PP 512 | **1067** |
| ik_llama sm120 | PP 128 | 919 |
| ik_llama sm120 | TG 128 | **25.65** |
| ik_llama sm120 | TG 512 | 25.79 |
| stock b8125 | PP 512 | 962 |
| stock b8125 | TG 128 | 23.41 |

### Comparaison 27B dense vs 35B-A3B MoE

| Metrique | 35B-A3B IQ3_S (prod) | 27B Opus IQ4_XS | Ratio |
|---|---|---|---|
| Params actifs/token | ~3B | 27B | 9x |
| TG tok/s | 93 | 25.7 | 3.6x plus lent |
| PP tok/s | 1182 | 1067 | comparable |
| VRAM modèle | 15G | 13.67G | -1.3G |
| Ctx max (full GPU) | 64K (1 slot) | ~8K | 8x moins |
| MTP disponible | non (custom-mix pruned) | non (mradermacher stripped) | — |

### Verdict pour aider

- **25.7 tok/s** = utilisable mais pas confortable pour un éditeur interactif
- La qualité Opus-distilled pourrait compenser (meilleur raisonnement, meilleur code)
- Pour aider, le 35B-A3B reste **nettement supérieur** en vitesse (93 tok/s = 3.6x)
- Le 27B serait pertinent seulement si la qualité Opus-distilled est vraiment meilleure sur les taches code/raisonnement — test qualitatif nécessaire

---

## Test BF16 KV Cache — Plan (à venir)

### Budget VRAM (base : ncmoe=2, q4_0/q4_0, 96K → 257 MB libre)

| Config | ncmoe | VRAM freed | KV extra | Free estimé | Status |
|---|---|---|---|---|---|
| ncmoe=2 q4_0/q4_0 (actuel) | 2 | — | — | 257 MB | ✅ production |
| ncmoe=4 q8_0k/q4_0v | 4 | +1366 MB | +260 MB | **1363 MB** | ✅ test prioritaire |
| **ncmoe=4 bf16k/q4_0v** | **4** | **+1366 MB** | **+780 MB** | **843 MB** | **✅ cible** |
| ncmoe=4 bf16k/bf16v | 4 | +1366 MB | +1560 MB | 63 MB | ❌ OOM probable |
| ncmoe=6 bf16k/bf16v | 6 | +2732 MB | +1560 MB | 1429 MB | ✅ safe fallback |

### Ordre de test recommandé
1. `ncmoe=4 -ctk q8_0 -ctv q4_0` — quasi Masterpiece config sur ik_llama, ~89 tok/s estimé
2. `ncmoe=4 -ctk bf16 -ctv q4_0` — si 1 insuffisant, 843 MB libre, ~89 tok/s estimé
3. `ncmoe=6 -ctk bf16 -ctv bf16` — fallback qualité max si OOM à ncmoe=4, ~79 tok/s estimé

### Impact qualité bf16 KV
- Seulement **10 layers full-attention** (30/40 sont GDN/SSM = pas de KV)
- Mais ces 10 layers = tout le raisonnement long contexte (AA-LCR, LongBench)
- **Plus grande différence** sur contextes > 32K (kv_q4_0 perd ~3-5% qualité attention)
- Pour kine_router intent (<2K tokens) : impact nul — q4_0 est suffisant
