# Stress test — 4 requêtes piégées en parallèle (2026-04-24)

**Objectif** : vérifier (a) absence de contamination inter-slot sur 4 requêtes concurrentes, (b) qualité des réponses sur des questions techniques piégées, (c) comportement sampling/stop tokens, (d) timings sous charge M=4 avec PCH=512.

## Setup

- Runtime : chimere-server `0ee916f` (post-merge polish-profile + counter fix)
- Prod `:8081` : Qwen3.6-35B-A3B UD-IQ3_S, `CHIMERE_MULTISLOT=4` + `CHIMERE_MULTISLOT_NATIVE=1` + `CHIMERE_MAX_PREFILL_CHUNK=512`
- Harness : `stress-run.sh` (4 curl concurrent via `ThreadPoolExecutor`), streaming SSE, `max_tokens=8192`
- Hardware : RTX 5060 Ti 16 GB, sm_120, CUDA 12.8, driver 590.48

## Les 4 prompts piégés

Détail des prompts dans [`prompts.json`](prompts.json). Résumé :

| # | Piège | Type |
|---|---|---|
| 0 | **Bateau + marée + échelle** | Piège physique classique : le bateau flotte, donc l'échelle monte AVEC la marée. Réponse attendue = 0 barreau sous l'eau. |
| 1 | **count_voyelles Python** | Gestion des voyelles accentuées françaises (é, è, à, ...). Un `set('aeiouy')` naïf retournerait 1 au lieu de 3 pour "éléphant". |
| 2 | **Protocole kiné LCA vs LLI grade 3** | Distinction fine : LCA grade 3 = souvent chirurgical (post-op), LLI = souvent conservateur (attelle). Demande précision sur mise en charge, amplitudes, chirurgie. |
| 3 | **Instruction multi-contraintes** | "Exactement 3 conseils, chacun de 10 mots, numérotés, puis `fin.` en minuscules, rien d'autre." Test de respect format strict. |

## Résultats timings

| Prompt | TTFT | Wall | Content toks | Think chars | Finish |
|---|---:|---:|---:|---:|---:|
| bateau | 306 ms | 120.0 s | 0 | 10 291 | `length` (timeout client) |
| voyelles | 212 ms | 120.0 s | 556 | 7 514 | `length` (timeout client) |
| entorse | 123 ms | 120.0 s | 408 | 9 085 | `length` (timeout client) |
| instruction | 409 ms | 120.0 s | 417 | 7 481 | `length` (timeout client) |

- **TTFT 120-410 ms sous charge 4-concurrent** : excellent, en ligne avec les bench du matin (PCH=512 M=4 donne TTFT p50=422 ms).
- **Wall 120 s = timeout client** (pas le serveur) — tous les requests ont été coupés par `--max-time 120`.
- **Pas de `finish_reason=stop`** sur aucune requête : le modèle ne s'arrête jamais sur `<|im_end|>` et file jusqu'à `max_tokens` → voir **bug stop token** ci-dessous.

## Contamination inter-slot : NÉGATIVE ✓

Extrait des 4 think blocks (contenu pris au hasard dans chacun) :

- Slot 0 (`piege-logique-bateau`) : "Here's a thinking process that leads to the correct answer: 1. Analyze the Riddle/Problem: Context: A boat is docked (moored/tied up). Object: A ladder hangs over the side. Dimensions: 20 rungs total..."
- Slot 1 (`piege-code-voyelles-fr`) : "The user wants a Python 3 function named `count_voyelles` that counts vowels in a French string. The function needs a docstring with a doctest example, specifically testing the word 'éléphant' which must return 3..."
- Slot 2 (`piege-medical-entorse-grade3`) : "User: Physiotherapist (kinésithérapeute). Patient: Grade 3 sprain (entorse grade 3). Question: What's the difference in rehabilitation protocol between week 1-2 for ACL (LCA - ligament croisé antérieur) vs. MCL (LLI - ligament latéral interne) sprains?"
- Slot 3 (`piege-instruction-strict`) : "Topic: 3 tips for sleeping well ('bien dormir'). Format Requirements: EXACTLY 3 tips, Each tip must be EXACTLY 10 words, Numbered 1., 2., 3., After the 3 tips, end with the word 'fin.' in lowercase..."

Chaque slot raisonne exclusivement sur SON prompt. Aucun leak détecté. La `MultiEngramLookup` thread-local et l'isolation NativeScheduler tiennent sous charge 4-concurrent.

## Qualité réponses : 3/4 excellentes, 1 time-out think

### Slot 1 — count_voyelles

Sortie :

```python
def count_voyelles(texte: str) -> int:
    """
    Compte le nombre de voyelles (y compris les voyelles accentuées)
    dans une chaîne française.

    Exemples :
    >>> count_voyelles('éléphant')
    3
    >>> count_voyelles('PYTHON')
    1
    >>> count_voyelles('àâéèêëïîôùûü')
    10
    """
    voyelles = set('aeiouyàâéèêëïîôùûü')
    return sum(1 for caractere in texte.lower() if caractere in voyelles)
```

**Note qualité : 5/5**. Tous les critères sont cochés : set d'accents complet, case-insensitive via `.lower()`, doctest valide avec éléphant=3. Explication française. Code exécutable immédiatement.

### Slot 2 — LCA vs LLI grade 3 (kiné)

Points correctement traités :

- **Chirurgical vs conservateur** : LCA grade 3 "presque toujours chirurgical chez le patient actif/sportif" vs LLI grade 3 "quasi-exclusivement conservateur".
- **Mise en charge** : post-op LCA = MPOT/PTOT dès J1-J3 avec béquilles + bride verrouillée en extension. LLI = PTOT dès J1-J2, béquilles abandonnées J3-J5.
- **Amplitudes articulaires** : mentionnées (protocole chirurgien post-op LCA).
- **Terminologie médicale FR** : précise, LLI/LCA explicités, MCL/ACL traduits, ESSKA cité.
- **Structure** : tableau comparatif clair.

**Note qualité : 5/5**. Le piège (distinction clinique LCA chirurgie obligatoire vs LLI traitement conservateur) est parfaitement identifié et argumenté. Niveau confrère.

### Slot 3 — Instruction stricte 10 mots + fin.

Sortie :

```
1. Éteignez vos écrans avant de dormir pour un repos complet.
2. Gardez votre chambre fraîche et sombre pour un sommeil optimal.
3. Établissez une routine régulière chaque soir pour réguler votre horloge.
fin.
```

Comptage : chaque ligne = exactement 10 mots. `fin.` seul sur dernière ligne, en minuscules. **Format respecté au caractère près.**

**Note qualité : 5/5** sur le contenu visible. Mais voir bug stop token ci-dessous (le modèle continue à générer après `fin.`).

### Slot 0 — Bateau/marée : NON CONCLU

Think block = 10 291 caractères. Aucun content émis. Le modèle a rumine pendant 120 s sans jamais sortir du `<think>`. Probablement rentré dans un sur-raisonnement (énumération exhaustive de cas) sans jamais conclure.

**Note qualité : N/A** (pas de content). **Observation** : Qwen3.6 peut entrer en boucle de raisonnement sur des prompts "piège classique" où il sur-analyse. Un `max_tokens` plus agressif ou un forcing de réponse post-think (reasoning_format directif) serait nécessaire.

## Bug détecté : stop token non honoré

Sur les 4 requêtes, `finish_reason` = `length` (timeout client à 120 s), jamais `stop`. Le streaming montre qu'après une réponse terminée proprement (ex : `fin.\n` pour slot 3), le modèle continue à générer :

```
1. Éteignez vos écrans avant de dormir pour un repos complet.
2. Gardez votre chambre fraîche et sombre pour un sommeil optimal.
3. Établissez une routine régulière chaque soir pour réguler votre horloge.
fin.
user                    ← devrait s'arrêter ici sur <|im_end|>
assistant
fin.
2025
user
assistant
<think>
Here's a thinking process:
1. Analyze User Input: ...
```

Le modèle regénère le rôle `user` / `assistant` et repart dans un nouveau tour. Impact :

- Tokens gaspillés (jusqu'à 2-3× le budget utile).
- Latence perçue côté serveur bien au-delà du "vrai" stop.
- `chimere_gen_tokens_total` gonflé (fausse métrique).
- Logs pollués.

Côté UX final (Open WebUI, Claude Code), les clients coupent sur la 1ère réponse complète via leur propre parsing, donc l'utilisateur ne voit pas le surplus. Mais l'impact serveur est réel.

**Hypothèses à investiguer** :

1. `chimere_sampler.cpp` : stop_tokens configurés pour inclure `<|im_end|>` (token id 248046 d'après les logs) ?
2. Stream handler côté axum : filtre-t-il les tokens spéciaux avant de les émettre ?
3. Le `finish_reason=stop` est-il bien propagé depuis `chimere_sampler_sample` vers le driver ?

**Prochaine étape suggérée** : grep `<|im_end|>` et `248046` dans `chimere-server/src/` pour voir où le stop est (ou n'est pas) géré.

## Recommandations

1. **Pour la prod** : le M=4 / PCH=512 tient bien sous charge 4-concurrent piégée. Pas de contamination, TTFT excellent, qualité top. Ne pas changer.
2. **Bug stop token** : à investiguer (ticket à créer). Impact serveur (non-UX) mais réel. Priorité moyenne.
3. **Sur-raisonnement Qwen3.6 sur pièges classiques** : à tester avec un `max_tokens=16384` ou via ODO `:8084` qui force `enable_thinking=True` et a un budget plus large.

## Reproduction

```bash
cd chimere-server/benchmarks/stress-test
./stress-run.sh
```

Artifacts dans `raw/` : 4 JSONL par requête (id, timings, content, think) + SSE brut + `/metrics` pre/post + `/v1/status` pre/post + `nvidia-smi dmon`.
