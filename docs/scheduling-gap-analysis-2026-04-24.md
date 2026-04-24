# Multi-slot scheduling gap — investigation

Auditeur : agent Opus (2026-04-24, bloqué sur Write par sandbox → sauvé par la session principale).
Tip chimere-server : `main` @ `38ecf41`.
Tip ik_llama.cpp fork : `fix-issue-20225-hybrid-checkpoint-reset` (= build du `libllama.so` prod).

## Top-level résumé

**Single-sentence root cause** : le gap ~30 ms inter-step à M=4 vient du dispatch GDN dans ik_llama — `delta_net::build_layer_attn_linear` à `ik_llama.cpp/src/llama-delta-net.cpp:611-621` prend un chemin per-token subgraph dès que le batch a plusieurs `seq_id` distincts, émettant ~1800 ops ggml supplémentaires par decode step à M=4 vs M=1. Facteur secondaire : la réutilisation de graphe est désactivée inconditionnellement pour `n_tokens > 1` à `llama.cpp:561`, forçant un rebuild complet à chaque tick.

**Top-fix recommandation (Medium, 1-3 semaines)** : réécrire la boucle per-seq à `llama-delta-net.cpp:611-621` pour passer au kernel `ggml_delta_net` (déjà capable de n_seqs > 1, voir asserts à `:85,95`) un tenseur batché `(head_dim, head_k_dim/num_heads, 1, n_seqs=N)` au lieu de boucler. Pas de changement kernel — uniquement le dispatch, gather/scatter d'état via `ggml_get_rows`/`ggml_cpy`, et la gestion per-seq du reset-state.

**Gain attendu** : agrégat 95 tok/s → ~280 tok/s à M=4 (**2.9×**). Per-slot 24.4 → ~70 tok/s. Supprime les 120 subgraphs (30 GDN × 4 seqs) du decode step. Mem-BW devrait remonter de 23% à ~50%, confirmant qu'on a débloqué le plafond BW.

**Quick win (Small, 1-3 jours)** : relaxer `can_reuse_graph` à `llama.cpp:559-570` pour permettre reuse sur `n_tokens > 1` quand topologie stable (seq_id set, n_kv, n_outputs). `update_cache_copies` nécessite ~30 LOC pour gérer GDN state-dst views. Gain ~95 → ~120 tok/s (**+25%**). Zéro changement côté chimere-server, patch ik_llama-only, upstream-friendly.

**Effort** : QW = 1-3 j, Medium = 1-3 sem, tous deux dans ik_llama. Le slot_scheduler Rust et la frontière FFI ne sont PAS le bottleneck — ne pas modifier `chimere-server/src/slot_scheduler.rs` ni `llama_backend.rs`.

**Non-goals** : pas de rewrite Triton (bottleneck = dispatch, pas compute), pas de changement de `slot_scheduler.rs`/`llama_backend.rs`, garder `CHIMERE_MULTISLOT_NATIVE` opt-in, ne pas coupler ce travail au fix cosmétique du counter `chimere_gen_tokens_total`.

---

## 1. Symptômes observés

### 1.1 Débit agrégat plat

Source : `/home/remondiere/Bureau/chimere-drafts/e2e-profile-v1/benchmark-e2e-2026-04-24.md`

```
pass   wall_s   agg_tps   per-req p50   TTFT p50   inter-tok p50
M=1    54.23    94.41     98.7          4111 ms     10.17 ms
M=2    68.99    74.21     37.8          3509 ms     26.58 ms
M=4    53.73    95.29     24.4           130 ms     41.25 ms
```

Per-request rate chute en 1/N exactement : 98.7/37.8 = 2.61× (M=2), 98.7/24.4 = 4.05× (M=4). Agrégat ≈ 95 partout.

### 1.2 Distribution inter-token gap

p99/p50 = 1.04 (M=2) à 1.14 (M=4) — jitter quasi nul. Signature round-robin déterministe. Surcoût vs M=1 : +16.4 ms à M=2, +31.1 ms à M=4.

### 1.3 Télémétrie GPU

```
pass   SM p50 / moyenne   mem p50 / moyenne   pwr p50   pclk p50
M=1    89 / 88.1%         55 / 53.7%          129 W     2700 MHz
M=2    80 / 81.0%         27 / 26.6%           99 W     2737 MHz
M=4    77 / 76.5%         23 / 22.9%          103 W     2730 MHz
```

N=52, 66, 51 échantillons actifs. Mem BW chute 2.4× pendant que SM reste 75-89%.

### 1.4 Comptabilité bande passante

RTX 5060 Ti pic = 448 GB/s (128-bit @ 28 Gbps).
- M=1 : 448×0.55 = 246 GB/s / 98.7 tok/s = **2.49 GB/tok**.
- M=4 : 448×0.23 = 103 GB/s × 41.37 ms/step = **4.26 GB/step** = 1.06 GB/tok effectif.
- Trafic step M=4 (4.26 GB) / trafic token M=1 (2.49 GB) = **1.71×**.

Batching parfait donnerait 1.0× (même 2.49 GB/step). Série pure donnerait 4.0× (~10 GB/step à ~96 ms/step, NON observé). Le 1.71× observé dit : re-lecture partielle des poids + temps idle significatif entre kernels.

---

## 2. Walkthrough code — decode step à M=4

```
axum /v1/chat/completions
  -> server.rs chat_completions_handler
  -> Scheduler::admission_tx (mpsc, cap=64)
  -> slot_scheduler.rs NativeDriver::run()  [thread OS dédié]
       +-- admit_new / reap_draining
       +-- run_one_tick -> tick_generate_all  [hot path M=4]
            +-- build Vec<MultiSeqEntry> de 4 records
            +-- llama_backend.rs forward_multi_seq_borrow
                 +-- llama_batch_init(4, 0, 1)
                 +-- llama_decode(ctx, batch)     [C++, forward complet]
                 +-- llama_batch_free
            +-- per-slot : apply_engram_bias -> sample -> emit
```

### 2.1 Rust `tick_generate_all` (`chimere-server/src/slot_scheduler.rs:1552-1631`)

Rien de bloquant. Pas de mutex sur hot path (`std::thread::spawn`). `tick_us=0` en prod.

### 2.2 FFI `forward_multi_seq_borrow` (`chimere-server/src/llama_backend.rs:1036-1084`)

Single `llama_decode` à la ligne 1058. Pas de copie en retour (1068-1080 juste check présence). Pas le bottleneck.

### 2.3 ik_llama `llama_decode_internal` (`ik_llama.cpp/src/llama.cpp:3404-3690`)

- **3409-3446 mixed-seq fallback** : seulement si `any_diff && has_dup`. Decode M=4 avec seq_ids {0,1,2,3} uniques → fallback NON triggé. OK.
- **3524 `llama_kv_cache_find_slot`** : pour recurrent layers, `cache.head = min(seq_ids)`, `cache.n = max-min+1` (kv-cache.cpp:1095-1096). Pas de sérialisation.
- **3559 `can_reuse_graph`** (défini à llama.cpp:559-570) : **ligne 561 — `if (u_batch.n_tokens > 1) return false;`**. Réutilisation désactivée inconditionnellement pour n_tokens>1.

### 2.4 Boucle per-token GDN — LA CAUSE (`ik_llama.cpp/src/llama-delta-net.cpp:604-623`)

```cpp
if (all_same_seq) {
    bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
    return build_layer_attn_linear_core(ctx0, gf, cur, lctx.inp_s_seq_qnext,
        inp_out_ids, token_seq_ids.front(), reset_state, il, cb);
}
GGML_ASSERT(has_unique_seq_ids);
ggml_tensor * out = nullptr;
for (int64_t i = 0; i < batch.n_tokens; ++i) {
    ggml_tensor * cur_i = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (size_t) i * cur->nb[1]);
    ggml_tensor * inp_s_seq_qnext_i = ggml_view_2d(...);
    const bool reset_state_i = batch.pos != nullptr && batch.pos[i] == 0;
    const uint32_t state_seq_id_i = (uint32_t) token_seq_ids[i];
    ggml_tensor * out_i = build_layer_attn_linear_core(ctx0, gf, cur_i, ...);
    out = out == nullptr ? out_i : ggml_concat(ctx0, out, out_i, 1);
}
return out;
```

Chaque `build_layer_attn_linear_core` (lignes 430-585) se développe en ~15-20 ops ggml : attn_norm, build_qkvz, build_beta_gate, ssm_conv, silu, 2× l2_norm, 2× permute, build_fused_delta_net, cpy state writeback, gated_output.

Qwen3.6-35B a **30 layers GDN sur 40**. Pour decode M=4 :
- **120 sub-invocations per-seq par step** (30 layers × 4 seqs)
- **~1800-2400 ops ggml supplémentaires** dans le graphe vs M=1
- Toutes sérialisées topologiquement via `ggml_concat` par layer

Note : `build_fused_delta_net` (77-144) assertait déjà `q->ne[3] == n_seqs` (lignes 85, 95) — le kernel PEUT gérer des séquences batchées. Mais ligne 435 hard-code `const int64_t n_seqs = 1;` quand appelé depuis la boucle. **La capacité existe ; le dispatch ne l'utilise pas.**

### 2.5 Layers attention — nativement batchés (10 layers sur 40)

Standard Q/K/V matmul sur [n_embd, n_tokens]. KV cache gère paging per-seq. **Batchable : oui.**

### 2.6 FFN MoE — nativement batché

`ggml_top_k` + `ggml_mul_mat_id` / `ggml_moe_up_gate` sur [n_embd, n_tokens]. Routage per-token. **Batchable : oui.**

### 2.7 Table per-stage

| Stage | File:line | Batchable ? |
|---|---|:-:|
| Rust admission + tick dispatch | `slot_scheduler.rs:1271-1322` | oui |
| Rust tick_generate_all | `slot_scheduler.rs:1552-1588` | oui |
| FFI forward_multi_seq_borrow | `llama_backend.rs:1036-1084` | oui |
| ik_llama llama_decode_internal entry | `llama.cpp:3404-3446` | oui |
| KV slot find (recurrent) | `llama.cpp:1049-1100` | oui |
| **Graph reuse check** | **`llama.cpp:559-570`** | **NON** |
| Graph rebuild + sched_alloc | `llama.cpp:3570-3583` | NON |
| Attention (10 layers) | `llama-build-context.cpp:10505` | oui |
| **GDN layer (30 layers)** | **`llama-delta-net.cpp:604-623`** | **NON** |
| GDN core internals | `llama-delta-net.cpp:430-585` | partiel (n_seqs=1 hard-coded) |
| ggml_delta_net fused scan | `llama-delta-net.cpp:122` | oui (capable, sous-utilisé) |
| MoE FFN | `llama-build-context.cpp:1011-1281` | oui |
| Per-slot sample + emit | `slot_scheduler.rs:1591-1627` | négligeable |

---

## 3. Identification de la cause

### 3.1 Primaire : sérialisation GDN per-token (`llama-delta-net.cpp:611-621`)

Preuves :
- Inter-tok +31 ms à M=4 match 30 layers GDN × ~1 ms extra per-layer (120 subgraphs × ~250 µs launch+tiny-exec).
- Mem-BW 23% à M=4 = signature launch-bound (SM haut, BW bas = kernels nombreux ne saturant pas la mémoire).
- Jitter quasi nul (p99/p50 ≈ 1.04-1.14) = séquence subgraph déterministe per-step, pas de GC/mutex contention.
- `ggml_delta_net` fused kernel supporte n_seqs > 1 (asserts à delta-net.cpp:85, 95). **Le dispatch est le problème.**

### 3.2 Secondaire : graph rebuild à chaque tick multi-seq (`llama.cpp:561`)

Chaque tick n_tokens>1 trigger `reset_scheduler` → `build_graph` → `sched_alloc_graph`. Coût typique sur ce modèle : 5-10 ms/tick. À 41 ms/step ça fait 12-24% du step time. Passer à zéro → ~30 ms/step → ~117 tok/s agrégat → **+23% gain**.

Le gate est conservateur : reuse nécessite `all_seq_id`, `kv_self.n`, `n_outputs`, `cache_copies` stables. Pour decode steady-state M=N avec slot pool statique, TOUT est stable — le gate peut être relâché.

### 3.3 Confirmé PAS le bottleneck

- Scheduling Rust (mpsc, std::thread ; pas de mutex sur hot path)
- Overhead FFI (~5 µs/call)
- KV cache contention (pages per-seq, pas de conflit)
- MoE router (fully batched)
- Attention (10 layers, fully batched)
- Sampler (~100 µs × 4 seqs = 0.4 ms sur 41 ms)
- SSE emission (absorbée dans jitter)
- Counter `chimere_gen_tokens_total` cassé (cosmétique, zéro impact perf)

---

## 4. Options de fix, rangées effort/ROI

### 4.1 Quick win (S, 1-3 jours) : graph reuse pour decode multi-seq stable

**Fichier** : `ik_llama.cpp/src/llama.cpp:559-570` + `update_cache_copies` à 572-635.

Relâcher le gate `n_tokens > 1` pour autoriser reuse quand seq_id set + n_kv + n_outputs inchangés. Étendre `update_cache_copies` (~30 LOC) pour re-pointer GDN state_dst views selon le nouveau per-seq state_seq_id_i. Persister `seq_ids` dans `Prev` struct (llama-context.h).

**Risque** : topologie du graphe EST stable across steady-state ticks (vérifié : loop per-seq delta-net purement n_tokens-dépendante, n_tokens + seq_id set stables quand slot pool statique). Positions changent mais passent par `llama_set_inputs` qui gère déjà updates runtime.

**Gain attendu** : 41.25 → ~32-36 ms/step. Agrégat 95 → **~120 tok/s (+25%)**.

### 4.2 Medium (M, 1-3 semaines) : dispatch GDN batché

**Fichiers** : `ik_llama.cpp/src/llama-delta-net.cpp:604-623` (entry) + `:430-585` (core).

Remplacer la boucle per-token par un chemin batché qui :
1. Reshape input `[n_embd, n_tokens]` → `[n_embd, 1, n_tokens_per_seq, n_seqs]` (M=4 decode : n_tokens_per_seq=1, n_seqs=4).
2. Gather états via `ggml_get_rows(state_all, state_indices)` — `state_indices` nouveau tenseur input rempli par `llama_set_inputs` depuis `token_seq_ids`.
3. Appelle nouvelle `build_layer_attn_linear_core_batched` qui passe projections, fused scan, gated output avec bon n_seqs dim.
4. Scatter nouveaux états via `ggml_set_rows` ou N `ggml_cpy` (encore moins d'ops que N subgraphs complets aujourd'hui).
5. Gère per-seq reset-state via tenseur mask per-seq au lieu d'un bool scalaire.

Le kernel `ggml_delta_net` gère déjà n_seqs > 1 — pas de nouveau kernel. Les matmuls projection (`build_qkvz`, `build_beta_gate`) sont déjà batch-capables niveau matmul ; seuls les reshapes post-matmul changent.

**Risque** : plumbing layout délicat ; règles strides ggml strictes. Reset-state via mask per-seq au lieu de bool scalaire. Upstream (ik_llama) marqué "qwen3next TBD" — review non-triviale mais upstream-friendly.

**Gain attendu** : supprime les 120 subgraphs + 1800-2400 ops ggml du graphe M=4. Per-step ~34 ms → ~12 ms. Inter-tok ~14 ms à M=4. Agrégat **~280 tok/s (2.9×)**, per-slot ~70 tok/s.

### 4.3 Long (L, 2-4 mois) : varlen GDN + CUDA graph capture

1. Varlen GDN supportant différents n_tokens_per_seq (permet prefill+gen mixed ticks — déjà un TODO à `slot_scheduler.rs:49,54`). Référence : mamba-2 `ssm_scan_varlen`, flash-linear-attention.
2. CUDA graph capture du chemin decode fixed-topology une fois 4.1 en place. Single `cudaGraphLaunch` par tick.

**Gain attendu** : inter-tok 4-6 ms à M=4. Agrégat **~400-680 tok/s (4-7×)**, bornée par le plafond BW ~400 tok/s avec batching parfait.

### 4.4 Hors scope

- Rewrite Triton (compute n'est pas le bottleneck, dispatch l'est)
- Rewrite `llama_decode` API (API OK)
- Multi-GPU (single 5060 Ti)
- Swap modèle / quant (orthogonal)

---

## 5. Récap gains

| Fix | Effort | M=4 inter-tok | M=4 per-slot | M=4 agrégat | Speedup |
|---|---|--:|--:|--:|--:|
| Aujourd'hui | — | 41.25 ms | 24.4 tok/s | 95.3 tok/s | 1.00× |
| 4.1 QW (reuse) | 1-3 j | ~34 ms | ~30 tok/s | ~120 tok/s | 1.26× |
| 4.2 Medium | 1-3 sem | ~14 ms | ~70 tok/s | ~280 tok/s | 2.94× |
| 4.3 Long | 2-4 mo | ~5-6 ms | ~170 tok/s | ~400-680 tok/s | 4-7× |

Plafond BW à M=4 batching parfait : 448 GB/s × 0.55 / 2.49 GB/tok = ~100 tok/s par token concurrent × 4 = **~400 tok/s**. Le 680 de 4.3 suppose MoE expert overlap réduit trafic step sous 1 GB — workload-dépendant. Plafond réaliste 4.3 = ~400 tok/s soutenu.

---

## 6. Risques et non-goals

- **Anomalie M=2** (74 tok/s agg, pire que M=1) : probablement overhead per-step doublé sans bénéfice graph reuse. 4.1 seul devrait pousser M=2 à ~110 tok/s.
- Ne PAS fixer `chimere_gen_tokens_total` dans ce travail (cosmétique, déjà fixé séparément sur main 38ecf41).
- Ne PAS changer `slot_scheduler.rs` — driver Rust correct.
- Ne PAS introduire nouvelles ops ggml ; `ggml_delta_net`, `ggml_get_rows`, `ggml_cpy` déjà présentes.
- Garder `CHIMERE_MULTISLOT_NATIVE` opt-in ; prod single-slot doit rester bit-identique.
- Fixes vivent dans ik_llama, pas chimere-server.

---

## 7. Pointeurs reproduction

- Bench artifacts : `/home/remondiere/Bureau/chimere-drafts/e2e-profile-v1/raw/` (dmon CSVs, JSONL per-request, metrics/status snapshots).
- Bench driver : `/home/remondiere/Bureau/chimere-drafts/e2e-profile-v1/run-bench.sh` + `stream_bench.py` + `analyze.py`. Wall total ~10 min, downtime prod ~8 min.
- Attribution graph-rebuild : ik_llama a `#if IK_PRINT_TIMING` à llama.cpp:3542-3694. Rebuilder fork avec `-DIK_PRINT_TIMING=1` pour budgets token-par-token.
- Audit tag : chimere-server tip `38ecf41`, ik_llama fork `AIdevsmartdata/ik_llama.cpp` build_sm120, inspectés 2026-04-24.

---

## 8. Conclusion

Le gap scheduling ~30 ms entre decode steps à M=4 n'est PAS dans le slot_scheduler Rust ni à la frontière FFI. Il vit dans `llama_decode_internal` ik_llama, spécifiquement dans le dispatch Qwen3.6 GDN à `llama-delta-net.cpp:611-621`, qui prend un chemin per-token subgraph dès qu'il y a plusieurs seq_ids distincts, émettant ~1800 ops ggml supplémentaires par step M=4 vs M=1. Contributeur secondaire : graph reuse désactivée pour `n_tokens>1` à `llama.cpp:561`, forçant rebuild complet chaque tick. **Fix principal** : batcher le dispatch GDN via dim n_seqs déjà supportée par le kernel fused `ggml_delta_net` (1-3 sem, +2.9× agrégat). **Quick win** : relâcher gate graph-reuse (1-3 j, +25%). Aucun ne touche le code chimere-server.
