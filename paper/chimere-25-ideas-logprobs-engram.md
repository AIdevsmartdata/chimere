# Chimère — 25 idées Logprobs × Engram × Stack complet

*Rapport détaillé avec avantages, limites, synergies et possibilités.*
*Basé sur l'audit de code chimere-rewrite + recherche SOTA mars 2026.*

---

## Vue d'ensemble : ce qu'on a en main

| Composant | État | Clé |
|-----------|------|-----|
| Logprobs | ✅ top-5 log-softmax par token | Entropie, confiance, distribution |
| Engram | ✅ 3 tables (kine 20MB, code 42MB, cyber 172KB) | O(1) lookup, Cuckoo filter |
| NEST | ✅ alpha adaptatif per-token | RRC confidence gating |
| Dynamic Engram | ✅ web → .engr per-query | Logit-level knowledge injection |
| ThinkPRM | ✅ step-level verifier CPU | Score qualité par étape |
| DVTS | ✅ K candidates + scoring | Tree search |
| MTP | ✅ 49.5% acceptance | Multi-token prediction |
| Expert predictor | ✅ 86% hit@8 | MLP prefetch |
| Entropy router | ✅ 5 stratégies | Route per-token |
| ABF | ✅ thinking budget | Adaptive Budget Forcing |

---

## A. CONTRÔLE DE GÉNÉRATION PAR LOGPROBS

### A1. Sampling adaptatif token-par-token

**Principe** : À chaque token généré, calculer l'entropie Shannon depuis les top-5 logprobs. Ajuster dynamiquement temperature, top_k, top_p pour le token SUIVANT.

**Ce que ça améliore** :
- Élimine les hallucinations dans les zones haute entropie (le modèle "invente" quand il hésite)
- Élimine le sur-sampling dans les zones basse entropie (code syntax, formules)
- Le modèle devient "conscient" de sa propre incertitude et adapte son comportement

**Avantages** :
- Zéro coût computationnel (entropie = 5 multiplications, déjà dans les logprobs)
- Pas de paramètre global à tuner — le système s'auto-régule
- Compatible avec TOUT le reste (s'ajoute sans modifier les autres composants)

**Limites** :
- Top-5 logprobs sous-estiment l'entropie totale (les 248K tokens restants contribuent)
- Les seuils (0.5, 2.0, 3.0 nats) nécessitent calibration par domaine
- Peut créer des oscillations temp↑↓ si l'entropie fluctue rapidement

**Combinaisons** :
- **+ F2 (ABF logprob)** : le même signal contrôle sampling ET budget thinking
- **+ A2 (recherche mid-gen)** : si très haute entropie, trigger la recherche ET change le sampling
- **+ H2 (perplexity abort)** : le monitoring alimente les deux systèmes

---

### A2. Recherche web mid-génération ("Entropy Spike = Knowledge Gap")

**Principe** : Quand l'entropie spike au-dessus d'un seuil pendant N tokens consécutifs, PAUSER la génération, lancer une recherche web, construire un Dynamic Engram, et reprendre la génération avec le biais factuel.

**Ce que ça améliore** :
- Le modèle ne génère JAMAIS de texte factuel sans vérification quand il est incertain
- Transforme le "je ne sais pas" implicite en action concrète (recherche)
- La réponse finale contient des faits vérifiés même si le prompt ne demandait pas de recherche

**Avantages** :
- Self-RAG sans modèle critique séparé (le modèle EST son propre critique via logprobs)
- Le Dynamic Engram injecte les faits au niveau logit (plus profond que le context RAG)
- L'utilisateur ne voit pas la pause (le streaming reprend naturellement)

**Limites** :
- Latence : une recherche web prend 10-30s, la réponse est bloquée
- Le modèle a déjà généré du texte avant la pause — incohérence possible
- Nécessite un mécanisme de "retour arrière" si le texte pré-pause est incorrect
- Complexité d'implémentation : interruption async du gen loop Rust

**Combinaisons** :
- **+ J1 (Dynamic Engram)** : la recherche ALIMENTE le Dynamic Engram qui BIASE le reste de la génération
- **+ A1 (sampling adaptatif)** : après la recherche, le sampling est plus confiant (entropie baisse)
- **+ H1 (confidence stream)** : le client voit "recherche en cours..." puis la confiance remonte

---

### A3. Détecteur d'hallucination temps réel

**Principe** : Identifier les tokens où le modèle a un logprob élevé (il choisit avec confiance) MAIS l'entropie est haute (beaucoup d'alternatives également plausibles). C'est le signal d'une "hallucination confiante".

**Ce que ça améliore** :
- Détection précoce des hallucinations AVANT qu'elles ne se propagent
- En médical : empêche de recommander un dosage inventé avec assurance
- En code : détecte les API qui n'existent pas mais "sonnent bien"

**Avantages** :
- Détection en temps réel (pas en post-processing)
- Compatible avec le streaming SSE (annotation par token)
- Combinable avec Engram : si Engram confirme le token → pas d'hallucination

**Limites** :
- Faux positifs sur les tokens créatifs (brainstorming, poésie) où l'entropie haute est VOULUE
- Le signal est bruité sur les tokens de liaison (articles, prépositions)
- Ne détecte pas les hallucinations "basse entropie" (fait incorrect mémorisé)

**Combinaisons** :
- **+ C3 (triangulation)** : Engram + ThinkPRM confirment/infirment le soupçon d'hallucination
- **+ H2 (abort)** : trop de tokens hallucinés → abort + retry
- **+ G1 (entropy-weighted training)** : les tokens flaggés alimentent le training ciblé

---

## B. ENGRAM COMME DRAFTER SPÉCULATIF

### B1. DART Engram speculative decoding

**Principe** : `draft_sequence(context, 5)` produit 5 draft tokens par chaining n-gram. Au lieu de vérifier un par un, envoyer les 5 en batch via `forward_prefill()`. Accepter le plus long préfixe qui match les logits du modèle.

**Ce que ça améliore** :
- Speedup 1.3-1.5× sur le domaine kiné (314K n-grams) — GRATUIT (pas de drafter model)
- Les phrases répétitives ("selon les recommandations HAS") sont générées en batch
- Aucun VRAM supplémentaire (l'Engram est sur CPU, mmap'd)

**Avantages** :
- Le Cuckoo filter fait que les lookups hors-domaine coûtent ~10ns (pas de overhead)
- `forward_prefill()` est optimisé pour le batch — le GPU est mieux utilisé
- Le multi-table merge donne des drafts plus diversifiés que le single-table

**Limites** :
- Acceptance rate ~0% hors domaine → aucun speedup pour du chat casual
- GDN recurrent state : si un draft est rejeté, l'état doit être rollback (même problème que DFlash)
- L'Engram prédit des séquences de MOTS, pas de RAISONNEMENT → ne draft que les parties factuelles

**Combinaisons** :
- **+ E1 (vérification probabiliste)** : au lieu d'exact match, accepter si le draft est dans le top-5 logprobs
- **+ B2 (hybrid MTP)** : si MTP et Engram sont d'accord, acceptance quasi-certaine
- **+ D2 (expert prefetch)** : les drafts Engram prédisent aussi les experts nécessaires

---

### B2. Hybrid MTP + Engram speculative decoding

**Principe** : Le MTP head et l'Engram font chacun une prédiction. Si les deux convergent sur le même token, c'est presque certainement correct. Deux sources indépendantes (apprise vs statistique) qui donnent le même résultat.

**Ce que ça améliore** :
- Acceptance rate boostée de +10-15pp quand consensus (deux sources ≠ corrélées)
- Détecte les cas "faciles" (consensus) vs "durs" (désaccord) automatiquement
- Quand il y a désaccord, les logprobs arbitrent → meilleure décision

**Avantages** :
- Coût zéro : les deux prédictions existent déjà (MTP intégré, Engram O(1))
- Le consensus signal peut alimenter le confidence stream (H1)
- Fonctionne comme un vote : 2/2 = fort, 1/2 = vérifier, 0/2 = AR normal

**Limites** :
- MTP n'est pas dispo sur toutes les paths (seulement quand MTP head est actif)
- Engram et MTP peuvent être corrélés (entraînés sur des données similaires)
- Le bénéfice ne s'additionne pas linéairement avec B1 (overlap)

**Combinaisons** :
- **+ J3 (triple consensus)** : ajouter l'expert predictor comme 3ème source → 3/3 = >95% acceptance
- **+ C1 (alpha gated)** : le consensus signal ajuste aussi l'alpha Engram dynamiquement

---

## C. LOGPROBS + ENGRAM COMBINÉS

### C1. Alpha gated par logprobs du modèle

**Principe** : `α_eff = base_alpha × engram_conf × (1 - model_conf)`. Quand le modèle est confiant (basse entropie), l'alpha Engram descend à ~0 (le modèle n'a pas besoin d'aide). Quand le modèle est incertain ET l'Engram est confiant, alpha monte.

**Ce que ça améliore** :
- Élimine le problème fondamental de l'Engram : overrider un modèle qui avait raison
- L'ablation a montré que l'Engram nuit quand il biaise des réponses correctes → ce fix cible exactement ça
- L'Engram ne "parle" que quand le modèle "écoute" (haute entropie)

**Avantages** :
- 1 ligne de code dans `nest_adaptive_alpha()`
- Combine deux signaux orthogonaux (interne modèle vs externe table)
- Auto-régulé : pas besoin de tuner l'alpha global (il s'adapte par token)

**Limites** :
- Logprobs top-5 à 0.0 (bug C++ — FIXÉ maintenant, mais vérifier les vrais valeurs en prod)
- Le signal d'entropie est bruité sur les premiers tokens (peu de contexte)
- Si le modèle est confiant ET faux (hallucination basse entropie), Engram ne corrige pas

**Combinaisons** :
- **+ A3 (hallucination detector)** : quand confiant+faux détecté, FORCER alpha up pour laisser Engram corriger
- **+ F1 (strategy router)** : le gating fait partie du routeur per-token complet

---

### C2. Logger de désaccord modèle vs Engram

**Principe** : Pour chaque token, comparer la prédiction top-1 du modèle (logprobs) avec la prédiction top-1 de l'Engram. Logger tous les désaccords où l'Engram a confiance > 0.8. Ces tokens sont exactement ceux où le modèle manque de connaissances domaine.

**Ce que ça améliore** :
- Identifie PRÉCISÉMENT quels concepts/termes le modèle ne maîtrise pas
- Génère des training data ciblées : au lieu de 500 prompts random/nuit, on cible les lacunes
- Mesure quantitative de la couverture domaine du modèle vs l'Engram

**Avantages** :
- Zéro overhead (logging en parallèle de la génération)
- Les paires (contexte, token_correct_engram, token_faux_modèle) sont du DPO data gratuit
- Longitudinal : on peut tracer la progression (le modèle s'améliore sur les tokens loggés)

**Limites** :
- L'Engram n'a pas toujours raison (il est statistique, pas factuel)
- Volume potentiellement énorme (des milliers de désaccords par session)
- Les désaccords sur les tokens de style (pas de contenu) sont du bruit

**Combinaisons** :
- **+ G1 (DFlash pondéré)** : pondérer le loss par la magnitude du désaccord
- **+ J4 (Engram auto-croissant)** : quand le modèle a raison et Engram tort → enrichir l'Engram
- **+ I2 (coverage map)** : les désaccords dessinent la carte des lacunes

---

### C3. Triangulation de confiance (Modèle + Engram + ThinkPRM)

**Principe** : Trois signaux indépendants de qualité :
1. Entropie logprob modèle (per-token, temps réel)
2. Confiance Engram (per-token, O(1))
3. Score ThinkPRM (per-step, CPU async)

Les trois d'accord = confiance maximale. Désaccord = alerte.

**Ce que ça améliore** :
- Ensemble de 3 "juges" sans exécuter 3 modèles (1 modèle + 1 table + 1 vérificateur léger)
- Chaque juge a un angle mort différent → la combinaison couvre plus de cas
- ThinkPRM apporte la vérification logique (les deux autres sont statistiques)

**Avantages** :
- ThinkPRM est sur CPU → pas de concurrence GPU
- Les 3 signaux opèrent à des échelles différentes (token, n-gram, step) → complémentaires
- Permet un "score de confiance" composite pour l'UI et le training

**Limites** :
- ThinkPRM est lent (~5-10s par step) → ne peut pas être per-token, seulement per-step
- Les 3 signaux peuvent être corrélés (tous entraînés sur du texte similaire)
- Complexité d'agrégation : comment combiner 3 scores de natures différentes ?

**Combinaisons** :
- **+ H1 (confidence stream)** : le score triangulé est le plus robuste pour l'UI
- **+ F1 (strategy router)** : les 3 signaux alimentent le routeur per-token
- **+ A2 (recherche mid-gen)** : désaccord 3-way → trigger recherche immédiat

---

## D. PREFETCH D'EXPERTS VIA LOGPROBS

### D1. Expert warming depuis logprobs

**Principe** : Les logprobs top-5 disent quels tokens sont probables. Chaque token active des experts spécifiques dans le MoE. Si on pré-mappe tokens → experts (table offline), on peut prefetcher les experts pendant le sampling.

**Ce que ça améliore** :
- Réduit la latence de chargement d'experts CPU → GPU (3.1ms par expert via PCIe)
- Le prefetch chevauche le sampling (parallélisme CPU/GPU naturel)
- Découplé du modèle : pas besoin d'accéder aux hidden states

**Avantages** :
- La table token→experts est petite (~248K × 8 experts = 2MB)
- Se construit offline en 1 heure (profiler le modèle sur un corpus)
- Compatible avec n'importe quel MoE, pas spécifique à Qwen3.5

**Limites** :
- La correspondance token→experts n'est pas 1:1 (le même token active des experts différents selon le contexte)
- À 93 tok/s, le sampling prend ~10ms — le prefetch PCIe aussi ~3ms → timing serré
- Notre config ncmoe=4 n'offloade que 4 layers → le gain est limité

**Combinaisons** :
- **+ D2 (Engram prefetch)** : l'Engram prédit 5 tokens → prefetch pour TOUS les 5 simultanément
- **+ B1 (DART spec-dec)** : les drafts Engram prédisent les experts des 5 prochains tokens d'un coup

---

### D2. Expert prefetch via Engram (zero-model lookahead)

**Principe** : L'Engram prédit les 5 prochains tokens en O(1) (~0.01ms). Depuis ces 5 tokens, on lookup les experts nécessaires et on lance le prefetch CPU→GPU AVANT que le modèle ne tourne. Le modèle statistique charge les poids du modèle neural.

**Ce que ça améliore** :
- Le prefetch commence ~0.01ms après le token courant (vs ~10ms avec D1 qui attend les logprobs)
- Pour ncmoe=4 : les 4 layers offloaded pourraient avoir leurs experts pré-chargés
- Le GPU n'attend jamais les experts — ils sont déjà là quand le forward pass arrive

**Avantages** :
- Latence de prédiction quasi-nulle (O(1) hash lookup)
- Peut couvrir 5 tokens d'avance (pipeline depth = 5)
- Fonctionne même quand le GPU est occupé (le prefetch est sur un stream CUDA séparé)

**Limites** :
- L'Engram n'a pas de prédiction hors-domaine → prefetch inutile pour le chat casual
- Le hit rate expert dépend du hit rate Engram (si le draft est faux, les experts aussi)
- Notre benchmark a montré que le prefetch NUIT à 40 tok/s (cudarc) mais pourrait aider à 93 tok/s (ik_llama)

**Combinaisons** :
- **+ B1 (DART spec-dec)** : les mêmes drafts Engram servent au spec-dec ET au prefetch → deux usages du même signal
- **+ J3 (consensus)** : si MTP + Engram convergent, le prefetch est quasi-certain d'être correct

---

## E. MTP DEPUIS ENGRAM

### E1. Vérification probabiliste par logprobs

**Principe** : Le spec-dec standard exige un exact match (draft == sampled). Mais avec les logprobs top-5, on peut faire du rejection sampling : si le draft token est dans le top-5 (mais pas top-1), l'accepter avec probabilité = exp(logprob_draft) / exp(logprob_top1). C'est mathématiquement correct et augmente drastiquement l'acceptance.

**Ce que ça améliore** :
- Acceptance rate passe de ~30% (exact match) à ~50% (probabilistic acceptance)
- Les tokens "presque corrects" (synonymes, variantes) sont acceptés au lieu d'être rejetés
- Le résultat reste mathématiquement non-biaisé (la correction distribution est respectée)

**Avantages** :
- S'applique à N'IMPORTE QUEL drafter (Engram, MTP, DFlash)
- La vérification coûte 0 (on a déjà les logprobs)
- La preuve mathématique est simple (Leviathan et al., 2023 + extension top-K)

**Limites** :
- Les logprobs top-5 ne couvrent que ~5 tokens → le draft doit être dans ce top-5
- Sur un vocabulaire de 248K, top-5 = 0.002% → les drafts rares sont toujours rejetés
- Le gain dépend de la distribution : si top-1 a logprob -0.01 et top-2 a -5.0, l'acceptance est quasi-nulle

**Combinaisons** :
- **+ B1 (DART spec-dec)** : le même mécanisme mais avec drafts Engram → boost gratuit
- **+ E2 (draft tree)** : vérification probabiliste sur chaque branche de l'arbre
- **+ B2 (hybrid)** : consensus MTP+Engram dans le top-5 → acceptance quasi-certaine

---

### E2. Draft cascadé avec cutoff confiance et arbre

**Principe** : Au lieu de chainer greedily le top-1, garder TOUS les candidats à chaque step → arbre de drafts. Couper les branches quand la confiance Engram passe sous 0.3. Vérifier l'arbre entier via un batch forward pass et trouver le plus long chemin valide.

**Ce que ça améliore** :
- Explore plus de possibilités que le greedy (le bon draft peut être au top-2 d'un step intermédiaire)
- Le cutoff de confiance évite de gaspiller du compute sur des branches improbables
- Un seul batch forward pour vérifier un arbre de 10-20 tokens

**Avantages** :
- La construction de l'arbre est O(1) par nœud (Engram lookup)
- `forward_prefill()` gère déjà le batch → pas de nouveau code GPU
- L'arbre est borné naturellement par la décroissance de confiance Engram

**Limites** :
- L'arbre peut être très large (top-3 × 5 levels = 243 feuilles) → explosion combinatoire
- Le batch forward pour 20+ tokens peut être plus coûteux que 5× single forward
- GDN state rollback pour les branches rejetées est complexe

**Combinaisons** :
- **+ E1 (probabilistic acceptance)** : chaque nœud de l'arbre vérifié probabilistiquement
- **+ J3 (consensus)** : filtrer l'arbre aux branches où MTP confirme

---

## F. STRATÉGIES DE GÉNÉRATION ADAPTATIVES

### F1. Routeur stratégie per-token v2

**Principe** : Combiner TOUS les signaux disponibles en une table de décision per-token :

| Entropie modèle | Engram confiance | Action |
|:---|:---|:---|
| Basse + match | Haute | Skip forward, accepter draft Engram |
| Basse + no match | — | Greedy AR |
| Moyenne + match | Haute | Draft + verify via logprobs |
| Moyenne + no match | — | Sampling normal + MTP |
| Haute + match | Haute | Engram narrow + extend thinking |
| Haute + no match | — | ABF extend, DVTS K=2 |
| Très haute | — | PAUSE → web search → Dynamic Engram |

**Ce que ça améliore** :
- Chaque token est traité avec la BONNE stratégie (pas de one-size-fits-all)
- Les tokens faciles vont PLUS VITE (skip forward pass)
- Les tokens durs reçoivent PLUS de compute (DVTS, recherche)
- Le système s'adapte automatiquement au domaine et à la question

**Avantages** :
- Subsume A1, A2, B1, C1, F2 en un framework unique
- Mesurable : on peut logger quelle stratégie est choisie et corréler avec la qualité
- Le routeur est extensible (ajouter des signaux = ajouter des colonnes)

**Limites** :
- Les seuils entre stratégies nécessitent calibration par domaine
- L'overhead du routing lui-même (~0.1ms/token) doit rester négligeable
- Les transitions entre stratégies (AR → spec-dec → AR) doivent être propres (pas de state corruption)

**Combinaisons** :
- C'est le **framework unificateur** de presque toutes les autres idées
- **+ C3 (triangulation)** : les 3 signaux alimentent le routeur
- **+ H1 (confidence stream)** : la stratégie choisie est exposée dans le stream

---

### F2. ABF depuis logprob entropy (pas heuristique)

**Principe** : Pendant le thinking, monitorer l'entropie des logprobs. Si l'entropie chute à ~0 pendant 20+ tokens consécutifs → le modèle a convergé → CUT thinking. Si l'entropie reste haute → le modèle explore encore → EXTEND budget.

**Ce que ça améliore** :
- Le thinking budget s'adapte à la DIFFICULTÉ RÉELLE de la question
- Questions faciles : thinking coupé après 200 tokens au lieu de 2048
- Questions dures : thinking étendu au-delà du budget initial

**Avantages** :
- Remplace l'heuristique ABF actuel (seuil fixe sur l'entropie de la probe) par un signal direct
- Économise 30-50% des tokens thinking sur les questions faciles → réponse plus rapide
- Aucune dégradation de qualité (le thinking est coupé APRÈS convergence, pas avant)

**Limites** :
- Le modèle peut "faussement converger" (basse entropie sur une mauvaise conclusion)
- Le seuil de convergence dépend du domaine (math vs chat vs code)
- Nécessite une fenêtre glissante (20+ tokens) → pas instantané

**Combinaisons** :
- **+ A1 (sampling adaptatif)** : le même signal contrôle thinking ET sampling
- **+ F1 (strategy router)** : la convergence thinking influence la stratégie de génération réponse

---

## G. SIGNAL DE TRAINING DEPUIS LOGPROBS

### G1. DFlash/MeZO pondéré par entropie

**Principe** : Lors du training (MeZO, DFlash, LoRA), pondérer la loss de chaque token par son entropie au moment de l'inférence. Tokens incertains = poids élevé (le modèle doit apprendre ÇA). Tokens certains = poids faible (il sait déjà).

**Ce que ça améliore** :
- Training 2-3× plus efficient (focus sur ce que le modèle ne sait PAS)
- Moins de steps nécessaires pour le même gain de qualité
- Évite d'over-fitter sur les tokens faciles (qui dominent statistiquement)

**Avantages** :
- Le signal d'entropie est gratuit (collecté pendant l'inférence prod)
- Compatible avec MeZO, LoRA, SFT, DPO — n'importe quel framework de training
- La pondération est un multiplicateur sur la loss → 1 ligne de code

**Limites** :
- Les tokens haute entropie ne sont pas toujours les plus importants (parfois l'entropie est haute sur des choix stylistiques, pas factuels)
- Nécessite de stocker l'entropie par token dans les training pairs → plus de stockage
- Le calibration data doit être représentatif du domaine cible

**Combinaisons** :
- **+ C2 (disagreement logger)** : les tokens de désaccord Engram/modèle sont exactement les haute-entropie
- **+ J4 (auto-growing Engram)** : l'Engram grandit sur les tokens incertains, le training cible les mêmes → double attaque

---

### G2. Engram auto-croissant depuis l'inférence

**Principe** : Pendant la génération, observer les logprobs. Quand le modèle génère un token avec haute confiance (logprob > -0.5) et que l'Engram N'AVAIT PAS de prédiction pour ce n-gram → AJOUTER le n-gram à l'Engram. L'Engram capture les patterns que le modèle maîtrise bien, pour accélérer les futures générations similaires.

**Ce que ça améliore** :
- L'Engram grandit organiquement sans intervention humaine
- Les patterns les plus fréquents ET les plus confiants sont capturés
- Au fil du temps, l'Engram couvre de plus en plus le domaine d'usage réel

**Avantages** :
- Apprentissage continu sans toucher aux poids du modèle
- L'Engram est une "distillation continue" des meilleures prédictions du modèle
- Zéro GPU — l'écriture Engram est une opération hash+mmap CPU

**Limites** :
- L'Engram peut capturer des hallucinations confiantes (le modèle a tort mais est sûr de lui)
- Croissance non-bornée → nécessite un mécanisme d'éviction (LRU ou TTL)
- Les n-grams capturés reflètent le biais du modèle, pas la vérité

**Combinaisons** :
- **+ A3 (hallucination detector)** : ne PAS ajouter les tokens flaggés comme hallucination
- **+ C2 (disagreement)** : quand le modèle a raison et Engram tort → ajouter. Quand Engram a raison et modèle tort → ne pas ajouter
- **+ I2 (coverage map)** : la croissance de l'Engram remplit les trous de la carte

---

## H. MONITORING QUALITÉ TEMPS RÉEL

### H1. Stream de confiance live

**Principe** : Chaque chunk SSE inclut un score de confiance composite :
```json
{"delta":{"content":"recommandations"},"confidence":0.95,"engram_match":true,"strategy":"greedy"}
```

**Ce que ça améliore** :
- L'utilisateur VOIT quand le modèle est incertain
- En médical : le kiné sait quelles parties de la recommandation sont fiables
- L'UI peut colorier les tokens (vert = confiant, orange = incertain, rouge = hallucination suspecte)

**Avantages** :
- Le signal est déjà calculé (logprobs + Engram match) → pas de coût
- Standard OpenAI étendu → compatible avec les clients existants qui ignorent le champ
- Logging automatique pour analyse post-hoc

**Limites** :
- La confiance "perçue" par l'UI peut inquiéter inutilement (beaucoup de tokens sont normalement à confiance moyenne)
- Les clients Telegram ne supportent pas le rich formatting par token
- Le champ supplémentaire augmente la taille des chunks SSE (~20% de bande passante)

**Combinaisons** :
- **+ C3 (triangulation)** : le score est la combinaison des 3 signaux (plus robuste)
- **+ F1 (strategy router)** : la stratégie choisie est visible dans le stream

---

### H2. Abort sur perplexité cumulative

**Principe** : Suivre la perplexité running (exp de la moyenne des log-probs négatifs). Si elle dépasse un seuil (le modèle génère du garbage confident) → ABORT → retry avec plus de contexte, ou temp plus basse, ou DVTS.

**Ce que ça améliore** :
- Empêche les longs paragraphes hallucinés
- Économise les tokens (pas besoin de générer 4096 tokens de garbage)
- Le retry automatique donne souvent une meilleure réponse

**Avantages** :
- Le seuil est objectif (perplexité > X = qualité dégradée, mesurable)
- Compatible avec le streaming (abort mid-stream)
- Le retry peut inclure des informations supplémentaires (RAG, web search)

**Limites** :
- Le seuil dépend du domaine (code a naturellement une perplexité plus basse que le chat)
- L'abort gaspille tout le compute déjà fait
- Le retry peut aussi échouer (boucle infinie si le seuil est trop strict)

**Combinaisons** :
- **+ A2 (recherche mid-gen)** : au lieu d'abort total, PAUSE + recherche + reprendre
- **+ F2 (ABF logprob)** : la perplexité alimente la décision ABF

---

## I. ENGRAM COMME BASE DE CONNAISSANCES

### I1. API Engram queryable

**Principe** : Endpoint HTTP pour interroger directement l'Engram :
`GET /engram/query?context=recommandations+HAS&domain=kine`
Retourne les top-K completions instantanément (~0.01ms, pas d'inférence modèle).

**Ce que ça améliore** :
- Auto-complétion domaine-spécifique ultra-rapide
- Le kiné tape "protocole de réédu..." et voit les completions instantanément
- Utilisable comme outil de recherche dans la base de connaissances n-gram

**Avantages** :
- Zéro GPU, zéro latence modèle
- L'Engram de 20MB kiné = un dictionnaire de 314K séquences médicales
- Pourrait alimenter un widget d'auto-complétion dans l'UI Telegram

**Limites** :
- Les completions sont des TOKENS, pas des phrases (nécessite post-traitement)
- L'Engram ne comprend pas le sens, juste les co-occurrences
- Limité à l'ordre n (5-grams) → pas de complétion longue

**Combinaisons** :
- **+ H1 (confidence stream)** : les completions Engram affichées en preview avant la réponse modèle
- **+ J4 (auto-growing)** : l'API montre en temps réel la couverture de l'Engram

---

### I2. Carte de couverture Engram

**Principe** : Pour chaque domaine, calculer et visualiser quels n-grams ont des prédictions confiantes (dense) et lesquels ont des trous. Produit une "carte de connaissances" du système.

**Ce que ça améliore** :
- Identifie les lacunes : "L'Engram kiné couvre 95% des protocoles HAS mais 20% de la pharmacologie"
- Guide les priorités d'ingestion de données
- Mesure quantitative de la progression du système au fil du temps

**Avantages** :
- Analyse offline, pas de coût runtime
- Visualisable en heatmap par domaine/sous-domaine
- Combinable avec les quality scores pour corréler couverture Engram ↔ qualité

**Limites** :
- Définir les "domaines" et "sous-domaines" pour la carte est subjectif
- Un Engram dense ≠ un Engram correct (peut avoir beaucoup de n-grams mais faux)

**Combinaisons** :
- **+ C2 (disagreement)** : les désaccords modèle/Engram marquent les trous
- **+ G2 (auto-growing)** : la carte montre la progression de la croissance

---

## J. COMBINAISONS INÉDITES

### J1. Dynamic Engram from web search at inference time

**Principe** : RAG injecte des connaissances dans le CONTEXTE (limité par la fenêtre). Dynamic Engram injecte dans les LOGITS (illimité, affecte TOUS les tokens). 100K tokens de résultats web → 100K n-grams de biais factuel sur TOUTE la génération.

**Ce que ça améliore** :
- La connaissance web ne disparaît PAS quand le context window est plein
- Chaque token de la réponse est biaisé vers la factualité des sources web
- Combiné avec RAG : le prompt a le contexte, les logits ont les patterns factuels

**Avantages** :
- **Aucun autre moteur d'inférence ne fait ça** → contribution novel publishable
- L'Engram est construit en ~1s à partir des résultats web (pipeline déjà implémenté)
- Se combine naturellement avec le RAG existant (pas de remplacement, complémentarité)

**Limites** :
- Le Dynamic Engram est temporaire (per-query) → pas de capitalisation entre requêtes
- Les résultats web peuvent contenir des erreurs → le biais propage les erreurs
- La qualité dépend de la qualité de la recherche web en amont

**Combinaisons** :
- **+ A2 (recherche mid-gen)** : si l'entropie spike, enrichir le Dynamic Engram avec une 2ème recherche
- **+ C1 (alpha gated)** : l'alpha du Dynamic Engram est gated par la confiance du modèle
- **+ J4 (auto-growing)** : les bons n-grams du Dynamic Engram migrent vers l'Engram permanent

---

### J2. Block diffusion adaptive par logprobs

**Principe** : La block diffusion génère des blocs de tokens en parallèle. Actuellement le block size est fixe. Avec les logprobs, on prédit OÙ l'entropie va monter et on place les frontières de blocs aux transitions.

**Ce que ça améliore** :
- Les blocs faciles (basse entropie) sont gros → maximum de parallélisme
- Les blocs durs (haute entropie) sont petits → raffinement maximal
- Rend la block diffusion viable (le problème actuel : blocs fixes = certains faciles, certains durs)

**Avantages** :
- L'historique d'entropie prédit assez bien les futurs spikes
- Compatible avec le Swordsman entropy-based partitioning (mais dynamique au lieu de fixe)
- Pourrait être la pièce manquante pour le DFlash × entropy router

**Limites** :
- La prédiction d'entropie future n'est pas exacte (les spikes sont partiellement imprévisibles)
- La block diffusion elle-même n'est pas encore en prod (elle est plus lente que AR actuellement)
- Les frontières de blocs mal placées dégradent la qualité

**Combinaisons** :
- **+ F1 (strategy router)** : le routeur décide AR vs block diffusion ET la taille du bloc
- **+ B1 (DART)** : les régions Engram-couvertes → AR/spec-dec rapide, le reste → block diffusion

---

### J3. Consensus speculative decoding (MTP + Engram + Expert predictor)

**Principe** : Trois sources de prédiction next-token :
1. MTP head (appris, hidden states)
2. Engram (statistique, n-gram hash)
3. Expert predictor (quels experts fire → quels tokens probables)

L'INTERSECTION de leurs top-K = tokens quasi-certains (>90% acceptance).

**Ce que ça améliore** :
- Acceptance rate maximale quand consensus (3 sources indépendantes)
- Détecte automatiquement les tokens "faciles" (consensus) vs "durs" (pas de consensus)
- Chaque source utilise un type d'information DIFFÉRENT → pas de corrélation

**Avantages** :
- Les 3 prédictions sont déjà calculées (MTP intégré, Engram O(1), expert predictor ~0.1ms)
- Le consensus à 3 est extrêmement fiable (faux positifs quasi-nuls)
- Progressif : consensus 3/3 → skip, 2/3 → verify, 1/3 → AR normal

**Limites** :
- L'expert predictor ne prédit pas directement les tokens (il prédit les experts, la correspondance experts→tokens est indirecte)
- Le MTP head n'est actif que sur certains paths
- Le consensus est rare hors-domaine → bénéfice limité au texte domaine

**Combinaisons** :
- **+ D2 (expert prefetch)** : si consensus → les experts sont déjà préfetchés (triple bénéfice du même signal)
- **+ B2 (hybrid)** : le consensus à 3 est l'extension naturelle du consensus à 2

---

### J4. Engram auto-améliorant par logprobs feedback loop

**Principe** : Après génération, scanner les logprobs. Pour chaque token :
- Logprob > -0.5 (confiant) + Engram vide (pas de prédiction) + n-gram nouveau
→ AJOUTER automatiquement à l'Engram.

L'Engram est une "distillation continue" du modèle en table statistique.

**Ce que ça améliore** :
- L'Engram grandit SANS intervention humaine ni pipeline de données
- Il capture les patterns les plus utilisés ET les plus fiables du modèle
- Au fil des jours, l'Engram couvre de plus en plus le vocabulaire d'usage

**Avantages** :
- Apprentissage continu sans toucher les poids
- Zéro GPU (écriture mmap CPU)
- Les n-grams ajoutés sont les plus utiles (haute confiance = fiable, usage réel = pertinent)

**Limites** :
- Hallucinations confiantes → l'Engram apprend des erreurs
- Croissance non-bornée → mécanisme d'éviction nécessaire
- Biais de domaine : l'Engram reflète les questions posées, pas la vérité objective

**Combinaisons** :
- **+ A3 (hallucination detector)** : gate : n'ajouter QUE les tokens non-flaggés
- **+ C2 (disagreement)** : quand modèle confiant ET Engram absent → ajouter. Quand modèle faux ET Engram correct → ne rien changer
- **+ I2 (coverage map)** : suivre la croissance et la couverture domaine

---

### J5. DVTS initialisé par Engram

**Principe** : DVTS génère K candidates (actuellement par sampling aléatoire avec temperature variée). Au lieu de ça, initialiser les K branches depuis les top-K prédictions Engram. L'espace de recherche est pré-orienté vers des completions pertinentes.

**Ce que ça améliore** :
- Les K branches démarrent de points INFORMÉS au lieu de random
- Moins de branches gaspillées sur des directions improbables
- ThinkPRM score mieux des candidates qui partent d'une base factuelle

**Avantages** :
- Le coût de l'initialisation Engram est O(1) (vs sampling qui nécessite un forward pass)
- Les branches Engram-initialisées sont plus diverses (chaque n-gram top-K est un point de départ différent)
- Réduit le K nécessaire pour trouver une bonne réponse (K=2 Engram-init ≈ K=4 random)

**Limites** :
- L'Engram ne prédit que les tokens suivants immédiats (pas la qualité globale de la branche)
- Les branches Engram peuvent être trop similaires (bias vers les patterns fréquents)
- DVTS avec np=1 reste séquentiel (K=2 = 2× le temps, pas parallélisable)

**Combinaisons** :
- **+ C3 (triangulation)** : scorer les branches avec les 3 signaux au lieu de ThinkPRM seul
- **+ E1 (probabilistic acceptance)** : les branches Engram sont vérifiées probabilistiquement

---

## Synergies et super-combinaisons

### Combinaison Alpha : "Chimère Adaptive Engine"
**C1 + A1 + F1 + B1** → Le système per-token complet :
chaque token → calculer entropie → gater Engram → choisir stratégie (draft/AR/DVTS) → ajuster sampling

### Combinaison Beta : "Self-Improving Memory"
**J4 + C2 + G1 + I2** → La boucle d'amélioration continue :
inférence → logger désaccords → pondérer training → Engram auto-grow → visualiser progression

### Combinaison Gamma : "Knowledge-Grounded Generation"
**J1 + A2 + C1 + H1** → Génération ancrée dans le web :
web search → Dynamic Engram → logprob-gated injection → mid-gen refresh si spike → confidence stream

### Combinaison Delta : "Maximum Speed"
**B1 + E1 + D2 + J3 + E2** → Speculative decoding maximal :
Engram draft tree → probabilistic verify → expert prefetch → consensus triple → cascade cutoff

---

## 3 papers publiables

1. **"Dynamic Engram: Logit-Level Knowledge Injection Beyond Context Windows"** (J1 + A2 + C1)
   RAG = contexte. Engram = logits. Les deux ensemble = complémentaires. Novel, mesurable, impactant.

2. **"Confidence Triangulation for Adaptive LLM Generation"** (C3 + F1 + H1)
   3 signaux indépendants → per-token strategy routing. Framework unificateur. Novel dans la combinaison.

3. **"Self-Growing N-gram Tables from Inference Logprobs"** (J4 + C2 + G2)
   Apprentissage continu sans training. L'Engram distille le modèle pendant l'inférence. Continual learning novel.

---

*"Each token the system generates teaches it what it doesn't know — and what it does."*
