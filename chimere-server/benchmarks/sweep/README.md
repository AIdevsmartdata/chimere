chimere-server sweep-bench harness (v1)
=========================================

A llama-bench-style grid harness for chimere-server. Sweeps across the
three knobs that actually move the needle on a hybrid GDN+MoE stack:

  - M     = CHIMERE_MULTISLOT            (how many slots)
  - NCMOE = CHIMERE_NCMOE                (how many MoE layers on CPU)
  - PCH   = CHIMERE_MAX_PREFILL_CHUNK    (prefill tokens per driver tick)

For each cell (M, NCMOE, PCH), the harness starts a fresh chimere-server
child on a bench port (default 8082), waits for /health, runs N
streaming requests through the repo's existing stream_bench.py, scrapes
/metrics + /v1/status + 1 Hz nvidia-smi telemetry, then stops the
server. Aggregate results go into sweep.csv and a REPORT.md is
auto-rendered from REPORT-TEMPLATE.md.tmpl.


Files
-----

  sweep-bench.sh               Main entry point (bash).
  driver_wrapper.py            Per-cell driver: loads prompts.yaml,
                               imports the repo's stream_bench.py, runs
                               one pass, writes cell-summary.json.
  csv_append.py                Post-cell: appends one row to sweep.csv.
  render_report.py             Post-sweep: fills REPORT-TEMPLATE.md.tmpl
                               from sweep.csv, writes REPORT.md.
  prompts.yaml                 4 canonical prompts (short, medium, long,
                               code). Edit or pass --prompt-set to use
                               a different bank.
  REPORT-TEMPLATE.md.tmpl      Report template (the .tmpl suffix avoids
                               the "findings file" guardrail in the
                               drafts pipeline — the harness still reads
                               it under that name).
  CHIMERE_MAX_PREFILL_CHUNK.patch
                               Small patch adding the shorter env var
                               alias. Not required to run the harness
                               (the harness sets both aliases). Apply it
                               after review so operators can discover
                               the knob via the top-of-file env table
                               in bin/chimere-server.rs.


Prerequisites
-------------

  1. chimere-server binary already built:
        SERVER_ROOT/target/release/chimere-server
     (default SERVER_ROOT is
      /home/remondiere/github-repos/chimere/chimere-server).
  2. ik_llama.cpp sm_120 shared libs on LD_LIBRARY_PATH. The harness
     sets this automatically to
        /home/remondiere/ik_llama.cpp/build_sm120/ggml/src
        /home/remondiere/ik_llama.cpp/build_sm120/src
        /usr/local/cuda-12.8/lib64
     Override LD_LIBRARY_PATH in the environment if your build tree is
     elsewhere.
  3. Model + tokenizer paths. Defaults point at the current prod model
     (CHIMERE_MODEL, CHIMERE_TOKENIZER) and can be overridden via env.
  4. nvidia-smi in PATH.
  5. python3 (stdlib only — no pip, no PyYAML).
  6. :8082 free. The harness refuses :8081 unless --explicit-prod.


Minimum sweep (recommended first run, ~10 min)
----------------------------------------------

  ./sweep-bench.sh --output-dir /tmp/chimere-sweep-min

This runs 3 cells: M in {1,2,4}, NCMOE=0, PCH=256. Good sanity check
that the harness works end-to-end on your box.


Prefill-chunk sweep (12 cells, ~30 min)
---------------------------------------

  ./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep-pch \
    --multislot-sweep "1 4" \
    --ncmoe-sweep "0" \
    --prefill-chunk-sweep "64 128 256 512 1024 2048"

Useful to find the prefill-chunk knee for your TTFT target.


NCMOE vs multislot (9 cells, ~25 min)
-------------------------------------

  ./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep-ncmoe \
    --multislot-sweep "1 4" \
    --ncmoe-sweep "0 4 8" \
    --prefill-chunk-sweep "256"

Answers "does NCMOE trade TG for context at multi-slot?". NCMOE=4 at
M=1 typically drops TG by ~15-20% but unlocks a larger KV cache. At M=4
the trade may flip (because decode is already serialized across slots).


Wide sweep (36 cells, ~75 min)
------------------------------

  ./sweep-bench.sh \
    --output-dir /tmp/chimere-sweep-wide \
    --multislot-sweep "1 2 4" \
    --ncmoe-sweep "0 4 8" \
    --prefill-chunk-sweep "128 256 512 1024"

Save for a known-idle window. The harness prints an estimated runtime
at start and heartbeat every 15 s during boot.


Dry-run
-------

  ./sweep-bench.sh --output-dir /tmp/x --dry-run

Prints the config and the cell count, exits immediately.


Skip-server (bench against a running server)
--------------------------------------------

  ./sweep-bench.sh --skip-server --port 8082 --output-dir /tmp/x

Useful to iterate on the driver itself without paying the 30-60 s
model-load cost per cell. The harness still snapshots metrics pre/post
per cell but does NOT restart the server, so per-cell server state
leaks across cells.


Safety
------

- Refuses :8081 unless --explicit-prod (matches existing run-bench.sh).
- Traps EXIT/INT/TERM and kills the child chimere-server cleanly.
- Uses `exec` in the child subshell so SIGTERM goes to chimere-server,
  not bash.
- Model-load timeout: 180 s (change in sweep-bench.sh if needed).


Output layout
-------------

  <output-dir>/
    sweep.csv            aggregate (one row per cell)
    REPORT.md            auto-rendered report
    raw/
      cell-<tag>/
        summary.json         from stream_bench.py
        cell-summary.json    condensed for csv_append.py
        raw-<tag>.jsonl      per-request observations
        metrics-pre.txt      Prometheus text
        metrics-post.txt
        status-pre.json      /v1/status
        status-post.json
      nvidia-smi-dmon-<tag>.csv    1 Hz GPU telemetry
    logs/
      chimere-server-<tag>.log     server stderr
      driver-<tag>.log             driver stderr


Re-rendering the report without re-running the sweep
----------------------------------------------------

If you tweak REPORT-TEMPLATE.md.tmpl or the scoring in render_report.py,
re-render from the existing CSV:

  python3 render_report.py \
      --csv /tmp/chimere-sweep-min/sweep.csv \
      --template REPORT-TEMPLATE.md.tmpl \
      --output /tmp/chimere-sweep-min/REPORT.md \
      --start "$(date -Iseconds)" --end "$(date -Iseconds)" \
      --git-sha "$(git -C SERVER_ROOT rev-parse --short HEAD)" \
      --server-root SERVER_ROOT \
      --output-dir /tmp/chimere-sweep-min \
      --model-path "$CHIMERE_MODEL" \
      --multislot-sweep "1 2 4" --ncmoe-sweep "0" \
      --prefill-chunk-sweep "256" --n-reqs-per-pass 20 \
      --max-tokens 128 --prompt-set prompts.yaml


Limitations (honest)
--------------------

- Per-cell model load = 30-60 s. A 48-cell sweep is ~60-90 min.
- N=20 reqs/cell is tight for TTFT p99 statistics. Bump for tighter
  error bars.
- VRAM column is one nvidia-smi query at end-of-pass, not a peak.
- The harness assumes Qwen3.5/3.6-style native-multislot streaming; it
  does NOT exercise the non-streaming legacy path.
- No back-pressure control: if chimere-server rate-limits requests,
  the driver's concurrency setting may over-saturate the admission
  queue. Inspect status-pre/post.json for queue_len if this matters.


Where to add new knobs
----------------------

Each new knob needs four touchpoints:

  1. sweep-bench.sh: add --<knob>-sweep CLI flag, thread through the
     three nested loops, export as env var in start_server().
  2. csv_append.py: add the column to the CSV header in sweep-bench.sh
     AND to the row assembly in csv_append.py.
  3. render_report.py: add to the grid table (render_grid_table) and
     the repro command (REPRO_COMMAND) if public-facing.
  4. REPORT-TEMPLATE.md.tmpl: no change needed — the grid table is
     auto-rendered.
