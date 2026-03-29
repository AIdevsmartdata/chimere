🚀 Commande de lancement (Moteur)
  Si tu dois relancer ton moteur manuellement :


   1 /home/remondiere/ik_llama.cpp/build/bin/llama-server \
   2   -m /home/remondiere/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_S.gguf \
   3   -ngl 99 -fa on -c 196608 -np 3 -b 4096 -ub 512 \
   4   --cache-type-k q8_0 --cache-type-v q4_0 \
   5   --reasoning-format deepseek --jinja --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 \
   6   --cont-batching --metrics --host 127.0.0.1 --port 8081 -t 14

# Chimere -- IA Frontiere sur PC Local (Mars 2026)

*Document de reference fusionné. Claude (Lead Dev) & Kevin (Tech Lead). 6 mars 2026.*
*Synthese de 14 agents d'audit + debat Architecte vs Critique.*

---

## 1. Ce que c'est VRAIMENT

Chimere est un **projet d'optimisation d'inference** pour Qwen3.5-35B-A3B sur RTX 5060 Ti 16GB, avec des ambitions de recherche en speculative decoding et entropy routing.

**Ce que ca N'EST PAS** : un projet de modele from scratch. On n'entraine pas un LLM, on optimise l'inference d'un modele existant et on developpe des techniques de speedup novel.

### Situation au 6 mars 2026

| Metrique | Valeur | Contexte |
|----------|--------|----------|
| Modele prod | Qwen3.5-35B-A3B **IQ3_S** | Full GPU, zero offload |
| Vitesse gen | **77.7 tok/s** | (vs 42.9 Q5 avant) |
| Contexte | **32K tokens** | (vs 4K Q5 avant) |
| VRAM | 12.6 GB / 16.3 GB | 3.7 GB libres |
| Vision | Non (mmproj pas charge) | Possible: +0.9 GB |
| DFlash tau | 27.6% (IQ3, online c2) | Pipeline nightly actif |
| Services | 8081 (llama), 8083 (router), 8084 (think) | Tous up |

**Le switch Q5 -> IQ3_S est le gain le plus impactant** : 1.8x vitesse, 8x contexte, zero offload CPU, VRAM libre pour vision/DFlash.

---

## 2. Architecture -- 7 Piliers (statut honnete)

### Pilier 1 : GatedDeltaNet 3:1 -- VALIDE, EN PROD
- Qwen3.5 et Kimi Linear utilisent le meme ratio 3:1 (arXiv:2507.06457)
- 75% couches GDN = pas de KV cache = enorme gain VRAM
- Kernels FLA Triton >2x Mamba-2 (arXiv:2503.14376), independant de Flash Attention
- **On ne l'a pas construit, on en beneficie via Qwen3.5**

### Pilier 2 : Gated Attention -- INTEGRE DANS QWEN3.5
- NeurIPS 2025 Best Paper (arXiv:2505.06708), sigmoid output gate
- Deja present dans les couches full-attention de Qwen3.5
- **Pertinent uniquement si on entraine from scratch. Parke pour l'instant**

### Pilier 3 : MoE Ultra-Sparse -- VALIDE, EN PROD
- Qwen3.5: 256 experts, 8 actifs/token. On le subit, on ne le construit pas
- **Le bottleneck CPU offload est RESOLU** : IQ3_S full GPU elimine le probleme
- FloE (9.3x compression) et D2-MoE (arXiv:2502.17298) = pistes futures
- **Risque** : expert prefetching (SP-MoE, arXiv:2510.10302) non implemente

### Pilier 4 : Engram -- PARKE (3 echecs)
- Sprint 1-2: frequency tables = noise (ADR-004)
- Sprint 3 SCONE: GPT-2 OK (-26.6% PPL) mais greffage Qwen3-4B ECHEC (5 runs)
- DeepSeek V4 integre Engram mais aucune replication externe n'existe
- **Decision : attendre V4 open-source avant de retenter**
- Concept valide (arXiv:2601.07372), execution hors de portee pour l'instant

### Pilier 5 : Block Diffusion / DFlash -- EN PROD, PIVOT VERS MTP
- DFlash v8 deepKV : 474M params, tau=27.6% (IQ3), pipeline nightly actif
- **MAIS** : drafter externe trop lent quand AR = 80 tok/s. Le overhead IPC + forward drafter > gain spec decode
- **Pivot** : MTP built-in Qwen3.5 (1 layer, ~260 MB) = drafter gratuit sans overhead
- DFlash reste actif pour accumulation de donnees (12.1K hidden states) et recherche
- Refs: arXiv:2602.06036 (>6x lossless), arXiv:2503.09573 (BD3LM, ICLR 2025 Oral)

### Pilier 6 : Entropy Routing AR <-> Diffusion -- RECHERCHE PURE
- **ZERO code. ZERO prototype. C'est une idee, pas un projet**
- Validation indirecte : SAGE (arXiv:2602.00523), EAD, EPIC, AdaSD
- Think Router (port 8084) fait du routing entropique think/no-think = prototype adjacent
- **Genuinely novel** : aucun paper ne propose AR<->diffusion routing
- **Statut realiste** : papier a ecrire, pas code a deployer. Parke comme direction de recherche

### Pilier 7 : MTP (Multi-Token Prediction) -- PRIORITE #1
- Qwen3.5 embarque 1 MTP layer native (785 tenseurs, ~260 MB IQ3_S)
- Port llama.cpp en cours (14 fichiers modifies, base PR #19937)
- Speedup attendu : 1.7-1.9x (2 tokens/forward)
- VRAM : IQ3_S + MTP = ~13.3 GB, 3 GB libres pour KV cache
- **Blocage identifie** : `llama_memory_seq_rm()` retourne false sur etat GDN recurrent
  - Solution : API save/restore EXISTE DEJA (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`)
  - Patch = 3 fichiers, ~50-100 lignes (speculative.cpp, server-context.cpp, llama-memory-hybrid.cpp)
  - PR #20075 tente ca mais CRASHE sur notre build (GGML_ASSERT view tensor)
  - PR #19493 = approche complementaire (server-side checkpointing)
- **C'est le seul chemin vers un speedup reel, deployable, sans entrainement**

---

## 3. Hardware Sweet Spot : RTX 5060 Ti + 32GB DDR5

| Spec | Valeur | Impact |
|------|--------|--------|
| Tensor Cores FP4 | 759 AI TOPS | **Unrestricted** (FP8/FP16 = half speed sur gaming) |
| VRAM | 16 GB GDDR7 | IQ3_S = 12.6 GB, reste 3.7 GB |
| Bandwidth | 448 GB/s | Bottleneck, compense par MoE sparsity |
| DDR5 | 32 GB | Reservoir potentiel Engram (futur) |
| TDP | 180W | Sustainable 24/7 |

**L'asymetrie FP4/FP8 est l'insight cle** : NVFP4 = plus petit ET plus rapide que FP8 (1.6x throughput, arXiv:2601.09527). IQ3_S est le format optimal actuel.

**Flash Attention = non-probleme** : FA3 n'a pas de support sm_120, mais 75% des couches sont GDN (FLA Triton) et 25% utilisent PyTorch SDPA.

**KV cache ultra-leger** : seulement 10 couches attention (pas 40) = 8.125 KB/token. 32K context = ~260 MB.

---

## 4. Roadmap Realiste

### Phase 1 : Optimisation Inference (Mars-Mai 2026)

| Tache | Statut | Impact |
|-------|--------|--------|
| **IQ3_S en production** | **FAIT** | 77.7 tok/s, 32K ctx |
| Fix cycle systemd (think-router) | **FAIT** | Services stables |
| MTP port llama.cpp | EN COURS (14 fichiers) | 1.7-1.9x speedup |
| Vision mmproj | PRET (ajouter 1 flag) | OCR Instagram repare |
| Gateway patches | PRET (1 commande) | Skills Telegram |
| DFlash nightly | ACTIF | Donnees accumulees |

**Critere de sortie** : > 120 tok/s effective (MTP), vision active, pipeline stable.

### Phase 2 : Recherche Appliquee (Juin-Sept 2026)

| Tache | Dependance | Realisme |
|-------|------------|----------|
| MoE Expert Prefetching | MTP Phase 1 | ELEVE (SP-MoE, papers existent) |
| Energy Verifier (~50M MLP) | DFlash hidden states | MOYEN |
| NVFP4 quantization | Toolchain NVIDIA | MOYEN (QAD dispo) |
| Engram retry (si V4 sort) | DeepSeek V4 open-source | FAIBLE |

### Phase 3 : Innovation (Oct 2026+)

| Tache | Realisme |
|-------|----------|
| AR<->diffusion routing paper | RECHERCHE PURE |
| Chimere-engine Rust GPU | BLOQUE (CUDA 12.8+ requis) |
| Modele custom pre-training | TRES AMBITIEUX |

---

## 5. Innovations -- Classement honnete

### Genuinely Novel (publiable)
1. **Entropy Routing AR<->Diffusion** -- Concept, pas code. Aucun paper existant
2. **DFlash pour Qwen3.5-35B-A3B** -- Premier drafter block diffusion pour GDN+MoE
3. **MTP + save/restore GDN** -- Notre patch llama.cpp (si merge upstream)

### Etat de l'art applique (novel dans le contexte)
4. IQ3_S full GPU = sweet spot quant pour MoE hybride sur 16GB
5. Think Router entropique (production, port 8084)
6. Nightly pipeline DFlash (capture + fine-tune automatise)

### Speculatif (non demontre)
7. Hex-hybride complet (7 piliers reunis) -- personne ne l'a fait
8. Energy Verifier -- MLP scoring via hidden states drafter
9. GDN state quantization -- territoire inexplore

---

## 6. Risques Majeurs

| Risque | Severite | Mitigation |
|--------|----------|------------|
| MTP crash (GGML_ASSERT) | HAUTE | PR #20075 existe, attendre merge ou patcher |
| Engram irreproductible | HAUTE | Parke. Attendre V4 open-source |
| Bandwidth 448 GB/s | CERTAINE | NVFP4 + MoE sparsity compensent |
| CUDA sm_120 pas natif | CERTAINE | PTX forward compat fonctionne |
| Equipe de 2 | HAUTE | Claude = multiplicateur 10-50x |

---

## 7. Stack Technique

| Composant | Techno |
|-----------|--------|
| Inference prod | llama.cpp b8125, IQ3_S, port 8081 |
| Training DFlash | PyTorch BF16, 8-layer Transformer |
| Orchestration | Chimère (Python), Think Router, Message Router |
| RAG | ChromaDB 5547 chunks, Qwen3-Embedding-0.6B (CPU) |
| Search | SearXNG (276 engines) + Brave + CRAG |
| Rust | chimere-deltanet (70/70 tests, CPU-only), chimere-engine (scaffold) |

---

## 8. Competitive Landscape

| Modele | GDN | MoE | Engram | Diffusion | Entropy | Local 16GB |
|--------|-----|-----|--------|-----------|---------|------------|
| **Chimere** | via Qwen3.5 | via Qwen3.5 | PARKE | DFlash tau=27.6% | Think Router | **OUI** |
| Qwen3.5-397B | OUI | OUI | NON | NON | NON | NON |
| DeepSeek V4 | ~OUI | OUI | OUI | NON | NON | NON |
| Mercury 2 | NON | ? | NON | OUI natif | NON | NON |

**Notre niche** : optimisation d'inference sur hardware consumer, avec techniques de speedup novel (MTP+DFlash, entropy routing). Aucun acteur ne cible ce creneau.

---

## Annexe : References cles

- GDN: arXiv:2412.06464 (ICLR 2025) | Ratio 3:1: arXiv:2507.06457
- Gated Attention: arXiv:2505.06708 (NeurIPS 2025 Best Paper)
- DFlash: arXiv:2602.06036 | BD3LM: arXiv:2503.09573 (ICLR 2025 Oral)
- Engram: arXiv:2601.07372 | NVFP4: arXiv:2601.09527
- SAGE: arXiv:2602.00523 | EPIC: arXiv:2601.01714 | EBT: arXiv:2507.02092
- MTP SD: arXiv:2602.06019 | SP-MoE: arXiv:2510.10302
- D2-MoE: arXiv:2502.17298 (ICML 2025) | FloE: OpenReview
- FLA: arXiv:2503.14376 | LLaDA 2.1: arXiv:2602.08676
