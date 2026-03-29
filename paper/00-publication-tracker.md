# Chimere Publication Tracker — Mars 2026

*Mis a jour le 30 mars apres verification complete des donnees.*

## Papers par priorite

| # | Paper | Novelty | Pret | Venue cible | Status |
|---|-------|---------|------|-------------|--------|
| 1 | **DFlash MoE + GDN State Barrier** | ★★★★ | **90%** | NeurIPS Workshop / arXiv | **.tex REECRIT, toutes donnees verifiees** |
| 2 | **Chimere System Paper** | ★★★★ | 75% | arXiv / MLSys | .tex ecrit, a enrichir avec history-system |
| 3 | **RAMP Mixed-Precision Quantization** | ★★★★ | 70% | arXiv | histoire reconstituee, .tex a ecrire |
| 4 | **Engram Multi-Tier Consumer HW** | ★★★★ | 60% | EMNLP Demo | ablation mesuree (77→85→88%) |
| 5 | **MTP on Qwen3.5 MoE (negative)** | ★★★ | 70% | arXiv note | 84.8% acceptance, 93→47 tok/s |
| 6 | **Expert Prefetch (negative)** | ★★★ | 85% | Workshop | 86.65% hit@8, zero speedup |
| 7 | **ODO Unified Orchestrator** | ★★★ | 60% | arXiv | code publie |
| 8 | **chimere-deltanet Rust Runtime** | ★★★★ | 65% | MLSys / arXiv | 56K lignes Rust |

## Corrections critiques (30 mars, post-verification)

### DFlash paper — CORRIGE
- **AVANT** : tau=9.4 (+47% vs DFlash) presente comme resultat principal
- **APRES** : tau=9.4 = train set (biaise). **Holdout tau=6.06** (v8 deepKV, comparable au paper original). **Wall-clock = 0.73x** (ralentissement, pas acceleration)
- 8 architectures, 7 extracteurs C++, 20 nightly runs — tout documente
- Nouveau titre : "Eight Architectures, One Structural Barrier"

### System paper — A ENRICHIR
- Benchmark 10/10 = smoke test (10 questions) → discute dans limitations
- Engram ablation verifiee : v1 77% → OFF 85% → v2 88%
- 104 quality scores (mean 3.04/5), 68 training pairs — verifie
- A integrer : sprint perf 9.1→93 tok/s, 7 bugs critiques, 8 dead ends

### RAMP paper — A ECRIRE
- 7 builds en une nuit, 4 echecs
- QuaRot PPL 49524 (dead end), OptRot/ParoQuant explosent
- RAMP-v2 : 15.2 GB, 3.78 BPW, 90 tok/s, 30/30 bench

## Histoires detaillees (Bureau, pour redaction)

| Fichier | Taille | Contenu |
|---------|--------|---------|
| `paper-history-dflash.md` | 37 KB | 8 versions, 27 jours, chaque echec |
| `paper-history-system.md` | ~30 KB | 7 semaines, 9.1→93 tok/s, 8 dead ends |
| `paper-history-ramp.md` | 35 KB | 20+ configs testees, 7 builds |

## Code et donnees publies

### GitHub (public)
- [chimere](https://github.com/AIdevsmartdata/chimere) — 96K lignes (Rust+Python+CUDA)
- [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) — 17K lignes (Python)
- [ramp-quant](https://github.com/AIdevsmartdata/ramp-quant) — 9K lignes (Python+C)

### HuggingFace
- [RAMP-v2-15G](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-RAMP-v2-15G) — GGUF 15.2 GB + imatrix
- [IQ3_S-MTP](https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-IQ3_S-MTP) — GGUF 11 GB avec tenseurs MTP
- [chimere-dflash-data](https://huggingface.co/Kevletesteur/chimere-dflash-data) — prompts d'entrainement
- [chimere-quality-scores](https://huggingface.co/datasets/Kevletesteur/chimere-quality-scores) — scores + pairs
- [chimere-engram-tables](https://huggingface.co/datasets/Kevletesteur/chimere-engram-tables) — tables n-gram
- [chimere-expert-predictor](https://huggingface.co/datasets/Kevletesteur/chimere-expert-predictor) — 4 modeles

## Hardware
- RTX 5060 Ti 16GB (Blackwell, sm_120), i5-14600KF, 32GB DDR5
- Total cost: ~$0.10/day electricity
