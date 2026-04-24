# chimere-studio coding mode — roadmap M=4 native (2026-04-24)

**Vision en une phrase** — Un Cursor open-source, 100% local, où chaque action coding (build, review, refactor, test, doc) exploite les 4 slots concurrents de chimere-server pour livrer un workflow agentique en 30 s/round vs ~3-5 min en flat single-thread.

## Critères de succès mesurables

| Métrique | Cible | Mesure |
|---|---|---|
| Time-to-first-diff (build feature) | < 30 s | wall clock chimere-studio → 1er hunk diff visible |
| Time-to-test (générer + run) | < 45 s | + pytest exit code |
| Refactor batch 4 fichiers | < 25 s | 4 diffs prêts |
| Code review 4 axes | < 15 s | 4 critiques générées |
| User actions / minute | > 4 | actions agentiques complétées |
| MCP tools invocables | >= 8 | filesystem, git, run_tests, lint, format, search, doc, diff |
| Privacy | 100% local | aucun appel cloud, vérifié via `ss -tnp` |

## Architecture mentale "M=4 native"

Chaque pipeline ODO doit toujours réfléchir : "comment éclater ce travail sur 4 slots ?". 3 patterns canoniques :

### Pattern 1 — Pipeline parallel-then-sequential (build feature)
```
slot 1: Architect  -+
slot 2: DocSearch  -+--> slot 1: Reviewer --> slot 1: TechLead --> +-- slot 1: code_gen
                                                                    +-- slot 2: test_writer
                                                                    +-- slot 3: doc_gen
```
3 phases, 2-3 slots à chaque burst. Gain ~3x vs séquentiel.

### Pattern 2 — Fan-out (refactor batch / 4 angles review)
```
slot 1: file_a.py / security_review
slot 2: file_b.py / performance_review
slot 3: file_c.py / maintainability_review
slot 4: file_d.py / test_coverage_review
```
4 indépendants. Gain ~4x vs séquentiel.

### Pattern 3 — DVTS best-of-K (génération critique)
```
slot 1: candidate v1 (temp 0.6)
slot 2: candidate v2 (temp 0.7)
slot 3: candidate v3 (temp 0.8)
slot 4: ThinkPRM scorer (continu, ranks les 3 autres)
```
Latence = max(K) au lieu de somme. Qualité +5-15% mesuré.

---

## Milestones M1 -> M6

### M1 — OpenCode wrapper + smoke (1-2 jours, quick win)

Prouver que chimere-server est compatible avec un client agentique mature.

**Tâches**
- [ ] Installer OpenCode v0.20+
- [ ] Configurer endpoint local : `OPENAI_API_BASE=http://127.0.0.1:8081/v1`
- [ ] Tester 5 commandes courantes : plan, build, fix, test, commit
- [ ] Mesurer TTFT, qualité, latence end-to-end
- [ ] Documenter dans `chimere-server/integrations/opencode.md`

**Risque** faible. OpenCode supporte explicitement local llama-server.

**Critère de succès** 5 use cases marchent, pas de régression de qualité vs Cursor.

### M2 — Subagent isolation native dans ODO (3-5 jours)

SOTA 2026 dit "subagent context isolation > RAG sur 64K context". Aujourd'hui ODO partage le contexte entre toutes les étapes du pipeline. Anti-pattern.

**Scope**
- [ ] Nouveau concept `subagent_isolation: true` dans la YAML pipeline
- [ ] Quand activé : chaque step reçoit un contexte FRESH (system prompt + sa tâche), PAS l'historique du pipeline
- [ ] Le résultat de chaque sub-agent est résumé (~200 tokens) avant d'être passé à l'étape suivante
- [ ] Patcher `pipeline_executor.py` (~+150 LoC)

**Deliverables** `pipeline_executor.py` mode subagent + `code-agent-isolated.yaml` + test 4K tokens cap par step.

**Risque** medium. Peut perdre des informations entre steps. Mitigé par les summaries.

**Critère de succès** même qualité avec 3x moins de tokens consommés.

### M3 — chimere-mcp coding tools (5-7 jours)

`chimere-mcp` est mentionné dans les docs mais incomplet. Sans tools, l'agent peut écrire du code mais pas le tester ni le commit.

**Décision arch** : repo `AIdevsmartdata/chimere-mcp` ouvert public au démarrage M3, MIT.

**Tools à ajouter (MCP standard)**

| Tool | Description | LoC est. |
|---|---|---|
| `filesystem.read` | lire un fichier | reuse |
| `filesystem.write` | écrire/diff (avec confirmation user) | reuse |
| `filesystem.list` | tree/glob | reuse |
| `git.diff` | git diff staged/working | +30 |
| `git.commit` | commit avec message (confirm) | +30 |
| `shell.run_tests` | `pytest`, `cargo test`, `npm test` | +60 |
| `shell.lint` | `ruff`, `mypy`, `clippy`, `tsc` | +50 |
| `shell.format` | `ruff format`, `cargo fmt`, `prettier` | +40 |
| `code.search` | ripgrep + AST grep | +50 |

**Stack** FastMCP (Python). Server local sur `:9095`. **Transport** SSE (déjà standard chimere-server, scale better que stdio).

**Risque** medium-high. `shell.run_tests` exige sandbox (`bwrap` ou container) ou confirmation user à chaque exec.

**Critère de succès** un agent peut écrire du code, le linter, le tester, et committer en autonomie supervisée.

### M4 — chimere-studio coding-mode UI (10-14 jours, gros morceau)

**Décision arch** : embedder OpenCode comme dépendance via Tauri shell. NE PAS ré-implémenter.

**Layout cible**

```
+-----------------------------------------------------------------------------+
| TopBar: project | branch | model badge | M=4/PCH=512 status | settings      |
+--------------+---------------------------------+----------------------------+
| FileTree     | Editor + Diff Viewer            | Agent Conversation         |
| (50/100)     | (50/100, monaco)                | (30/100)                   |
|  > src/      |  +------------------+           |  > refactor this fn        |
|   > a.rs     |  | // current code  |           |  <- Architect: ...         |
|   > b.rs     |  |                  |           |  <- Reviewer: ...          |
|  > tests/    |  +------------------+           |  <- TechLead: synth        |
|              |  | // proposed diff |           |  [reply]                   |
| K candidates |  | + new line       |           |                            |
| [1][2][3][4] |  | - old line       |           |  Quick actions:            |
|              |  +------------------+           |  [build] [review]          |
|              |                                 |  [test]  [fix]             |
+--------------+---------------------------------+  [doc]   [refactor]        |
| Bottom: Terminal | Tasks | Memory | MCP Tools  |  [voice]                   |
+------------------------------------------------+----------------------------+
```

**Composants à coder**
- [ ] FileTree : `tauri-plugin-fs` + virtualized list
- [ ] Editor : Monaco React
- [ ] Diff viewer : `monaco-editor/react` diff mode
- [ ] Agent chat : reuse l'existant + ajouter quick actions
- [ ] Quick actions : 6 boutons -> POST `/v1/chat/completions` à ODO avec `odo_route` pinné
- [ ] Terminal : `xterm.js` + Tauri shell capability
- [ ] Tasks panel : montre les 4 slots actifs (gauges live `/v1/status`)
- [ ] DVTS K-candidates panel : voit les 4 alternatives, peut sélectionner

**Décision arch raccourcis clavier**
- Ctrl+B (build)
- Ctrl+R (review)
- Ctrl+T (test)
- Ctrl+F (fix)

**Tauri capabilities** : déjà window+shell+fs scopés. Ajouter `git` (read-only initialement).

**Risque** Tauri 2 plugins fragiles sur Linux. Tester chaque plugin standalone d'abord.

**Critère de succès** ouvrir un projet, demander "ajoute une fonction qui valide un email", obtenir un diff appliqué + tests + doc en < 60 s.

### M5 — Pipelines coding spécialisés M=4 (3-5 jours)

6 pipelines YAML qui exploitent natuellement les 4 slots :

| Pipeline | Use case | Slots utilisés | Wall cible |
|---|---|---|---|
| `code-build-feature.yaml` | "ajoute une feature X" | 1 -> 2 -> 1 -> 3 | 30 s |
| `code-refactor-batch.yaml` | "refactor ces 4 fichiers" | 4x parallel | 25 s |
| `code-review-multiangle.yaml` | "review ce commit" | 4x parallel | 15 s |
| `code-fix-bug.yaml` | "fix ce bug" | 1 (debug) -> 4 (tests) | 25 s |
| `code-write-tests.yaml` | "écris les tests pour X" | 4x parallel (1/fichier) | 20 s |
| `code-doc-generator.yaml` | "documente cette API" | 2 (doc + ex) // | 15 s |

**Chaque pipeline doit**
- Utiliser `subagent_isolation: true` (M2)
- Utiliser `parallel:` groups (F2 déjà fait)
- Activer `quality_gate` F6 sur les générations critiques
- Pin `odo_route: code-build-feature` etc. pour utilisation directe via API
- Spécifier MCP tools requis (M3)

**Risque** faible (pattern déjà éprouvé)

**Critère de succès** chaque pipeline atteint sa wall cible sur stress test (3 runs, 80% des cas)

### M6 — Plan/Build mode toggle + DVTS visualizer (5-7 jours)

**Plan/Build toggle** (style OpenCode)
- Mode "Plan" : architect + reviewer seulement, pas d'écriture fichier. Génère un plan markdown.
- Mode "Build" : applique le plan, écrit les diffs, lance les tests
- User confirme transition Plan -> Build

**DVTS viewer**
- Quand DVTS génère K=4 candidats, les afficher tous (pas juste le best)
- User peut sélectionner, comparer, merger
- ThinkPRM score visible pour chaque candidat

**Décision arch memory bank format** : Markdown (lisible, git-friendly, style Continue.dev) vs JSON.

**Risque** faible. UI work surtout.

**Critère de succès** user peut faire 1 plan, le valider, puis le builder, en restant <90 s total.

---

## Dépendances + ordre suggéré

```
M1 (OpenCode wrapper)       <- quick win, validation infra, indépendant

M2 (Subagent isolation)     <- prérequis M5 (pipelines exploitent isolation)

M3 (chimere-mcp tools)      <- prérequis M4 (UI a besoin des tools)

M4 (Studio coding-mode UI)  <- dépend M3 + M2

M5 (Pipelines coding)       <- dépend M2, complète M4

M6 (Plan/Build + DVTS view) <- dépend M4, polish final
```

Critique path : M2 -> M3 -> M4 -> M5 -> M6 ~ 5-6 semaines à un dev.
Quick wins en parallèle : M1 fait dès maintenant en attendant.

## Calendrier estimé

| Sem | Sprint |
|---|---|
| 1 | M1 OpenCode wrapper (1-2 j) + start M2 subagent isolation |
| 2 | Finir M2, start M3 chimere-mcp tools |
| 3 | Finir M3, start M4 studio coding-mode UI (gros) |
| 4 | M4 continue |
| 5 | Finir M4, M5 pipelines coding |
| 6 | M6 Plan/Build + DVTS viewer + polish |

Plus 1 semaine de buffer / bench / docs / dépublication = 7 semaines pour la version 1.

## Risques globaux

| Risque | Impact | Mitigation |
|---|---|---|
| chimere-mcp pas open-sourcé encore | bloque M3 | open-source EARLY (semaine 1), MIT |
| Tauri 2 plugins fragiles sur Linux | blocage M4 | tester chaque plugin standalone d'abord |
| `shell.run_tests` brèche sécurité | grave | sandbox bwrap obligatoire OU confirm user |
| Subagent summary perd info | qualité | A/B test avec/sans, dial le summary length |
| M=4 pas tenable si 1 slot bloqué | UX gaffe | timeout/retry par slot, kill-switch toujours |

## Décisions architecturales (validées 2026-04-24)

1. **chimere-mcp** : ouvrir public maintenant, repo `AIdevsmartdata/chimere-mcp`, licence MIT.
2. **Studio bundling** : embedder OpenCode comme dépendance via Tauri shell. NE PAS ré-implémenter.
3. **MCP transport** : SSE (déjà standard chimere-server, scale better que stdio).
4. **Memory bank format** : markdown (lisible, git-friendly).
5. **Quick action bindings** : Ctrl+B (build), Ctrl+R (review), Ctrl+T (test), Ctrl+F (fix).

## Status M1 — kickoff

**Lancé en agent Opus background** : étude OpenCode + draft `integrations/opencode.md` + smoke test plan. Reportable séparément quand l'agent termine.

---

<!-- reviewer-notes
- Plan synthétisé en session 2026-04-24 23h45
- 5 décisions architecturales validées par Kevin (option c)
- M1 lancé en agent background; M2-M6 sequencé
- Validation par stress test bench harness existant `chimere-server/benchmarks/sweep/`
-->
