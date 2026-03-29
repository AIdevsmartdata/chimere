# Chimère-DFlash : Plan d'Implémentation Complet pour Claude CLI

## Contexte

On reconstruit DFlash from scratch pour créer un drafter block-diffusion
conditionné sur Qwen3.5-35B-A3B (notre modèle production).
L'objectif est d'avoir un prototype fonctionnel en ~1 semaine,
puis d'y intégrer nos innovations Chimère (GatedDeltaNet, entropy router, blocs adaptatifs).

## Architecture DFlash Reconstituée

D'après le paper, le blog z-lab, le code d'inférence public, et l'analyse Emergent Mind :

```
Target Model (Qwen3.5-35B-A3B, frozen, llama-server port 8081)
    │
    │ hidden_states de k=5 couches (uniformément réparties layers 2→37)
    ▼
┌──────────────────────────────────────┐
│          Feature Fusion              │
│  W_fuse : [h_l1; ...; h_l5] → c     │
│  Linear(5 * target_dim, drafter_dim) │
└──────────┬───────────────────────────┘
           │ context vector c ∈ R^d_c
           ▼
┌──────────────────────────────────────┐
│    Block Diffusion Drafter           │
│    5 layers Transformer bidir        │
│                                      │
│    Entrée: bloc de γ=16 tokens       │
│    bruités (embedding + noise)       │
│                                      │
│    Chaque layer reçoit c via         │
│    KV injection (K_proj, V_proj      │
│    du contexte → KV cache)           │
│                                      │
│    Attention: BIDIRECTIONNELLE        │
│    (pas de masque causal)            │
│                                      │
│    Sortie: logits ∈ R^(γ × vocab)    │
│    en UN SEUL forward pass           │
└──────────┬───────────────────────────┘
           │
           ▼
    Embedding + LM Head PARTAGÉS
    avec le target model (frozen)
```

### Spécifications techniques (pour Qwen3.5-35B-A3B)

- **Target hidden dim** : 2048 (Qwen3.5-35B-A3B)
- **Target layers** : 40 (cycle 3×GDN-MoE + 1×GQA-MoE × 10)
- **Target vocab** : 248,320
- **Drafter hidden dim** : 2048 (même que target, pour partager embed/LM head)
- **Drafter layers** : 5 (transformer bidirectionnel)
- **Drafter attention heads** : 16 (head_dim=128)
- **Block size γ** : 16 tokens
- **Feature extraction layers** : 5 layers uniformément entre layer 2 et layer 37
  → layers [2, 11, 20, 29, 37]
- **Diffusion** : Gaussien continu, 1 step denoise
- **Training data** : 289K samples (DFlash original), nous ciblerons ~100K-300K
- **Params entraînables** : ~100-200M (5 layers + fusion projections)
  L'embedding (248K × 2048) + LM head sont gelés/partagés

### Budget VRAM Training (RTX 5060 Ti 16GB)

**Phase 1 : Extraction features (Qwen3.5 chargé, ~5.2GB)**
- Qwen3.5 Q2_K : 5.2 GB
- Batch processing : ~2 GB
- Total : ~7.2 GB ✓

**Phase 2 : Training drafter (Qwen3.5 déchargé, 16GB dispo)**
- Drafter 5 layers (~200M params) : ~0.4 GB bf16
- Embedding + LM head frozen : ~1.0 GB
- Optimizer states (8-bit Adam) : ~0.6 GB
- Gradients : ~0.4 GB
- Activations (grad checkpointing, bs=8, seq=16) : ~0.5 GB
- CUDA overhead : ~0.5 GB
- **Total : ~3.4 GB** → TRÈS confortable

---

## Plan d'Exécution — 7 Jours

### JOUR 1 : Setup + Skeleton + Tests unitaires
- Structure projet complète
- config.py, diffusion.py, feature_fusion.py, modeling.py, spec_decode.py, data.py
- Tests unitaires : shapes, forward/backward, generate_block

### JOUR 2 : Data Pipeline + Feature Extraction
- generate_training_data.py (Qwen3.5 port 8081, 500+ samples)
- extract_features.py (Qwen2.5-0.5B proxy pour dev, swap Qwen3.5 pour prod)
- DFlashDataset avec collate et batching

### JOUR 3 : Training Loop
- train.py complet (wandb, 8-bit Adam, gradient checkpointing, cosine LR)
- Smoke test : 200 steps, loss < 5.0

### JOUR 4-5 : Training réel overnight
- 50K+ samples, 100K+ blocks
- Training overnight RTX 5060 Ti
- Évaluation acceptance length

### JOUR 6 : GatedDeltaNet bidirectionnel
- Remplacer Transformer layers par GDN dual-scan
- Benchmark forward speed

### JOUR 7 : Entropy Router + Benchmarks
- Adaptive block sizing via entropy
- Benchmarks complets vs AR pur

## Métriques de Succès

| Milestone | Métrique | Cible |
|---|---|---|
| Jour 1 | Tests passent | 100% |
| Jour 3 | Loss descend | < 5.0 après 500 steps |
| Jour 5 | Acceptance length | ≥3 tokens/block |
| Jour 7 | Speedup vs AR | ≥2× |
| Semaine 2 | GDN layers | ≥3× speedup |

## Références

- DFlash: arXiv 2602.06036 (Chen & Liu, Feb 2026)
- DFlash models: huggingface.co/collections/z-lab/dflash
- dllm framework: github.com/ZHZisZZ/dllm
- BD3LM: m-arriola.com/bd3lms/
- DEER: arXiv 2512.15176
- FLA: github.com/fla-org/flash-linear-attention
