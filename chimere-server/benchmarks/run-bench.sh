#!/usr/bin/env bash
# run-bench.sh — E2E profile bench for chimere-server.
#
# Drives streaming /v1/chat/completions through chimere-server at three
# multi-slot settings (M=1, M=2, M=4) and records:
#   - /metrics pre/post (Prometheus counters + TTFT quantiles)
#   - /v1/status pre/post (JSON — includes per-slot state)
#   - per-request JSONL (TTFT, inter-token gaps, total ms, finish_reason)
#   - nvidia-smi dmon CSV concurrent with the load
#
# Runs against :8082 by default. Refuses :8081 unless EXPLICIT=1.
#
# Usage:
#   OUT_DIR=/tmp/e2e-profile ./run-bench.sh           # default sweep 1,2,4
#   MULTISLOT_SWEEP="4" ./run-bench.sh                # single pass M=4
#   PROMPT_SET=long ./run-bench.sh                    # long-prompt variant
#   EXPLICIT=1 BENCH_PORT=8081 ./run-bench.sh         # DO NOT normally use
#
# Environment:
#   OUT_DIR           Output directory (required, created if missing)
#   BENCH_PORT        Port to hit (default 8082)
#   MULTISLOT_SWEEP   Space-separated list of M values (default "1 2 4")
#   N_REQS            Requests per pass (default 40)
#   CONC              Concurrency per pass (default 4)
#   MAX_TOKENS        max_tokens per request (default 128)
#   PROMPT_SET        short|long (default short)
#   EXPLICIT          1 to allow :8081 / prod
#   SKIP_START_SERVER 1 if a server is already running on BENCH_PORT
#   SERVER_ROOT       chimere-server repo root (default /home/remondiere/github-repos/chimere/chimere-server)
#
set -euo pipefail

OUT_DIR="${OUT_DIR:?Set OUT_DIR to the output directory}"
BENCH_PORT="${BENCH_PORT:-8082}"
MULTISLOT_SWEEP="${MULTISLOT_SWEEP:-1 2 4}"
N_REQS="${N_REQS:-40}"
CONC="${CONC:-4}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT_SET="${PROMPT_SET:-short}"
EXPLICIT="${EXPLICIT:-0}"
SKIP_START_SERVER="${SKIP_START_SERVER:-0}"
SERVER_ROOT="${SERVER_ROOT:-/home/remondiere/github-repos/chimere/chimere-server}"

if [[ "$BENCH_PORT" == "8081" && "$EXPLICIT" != "1" ]]; then
    echo "[run-bench] refusing to target :8081 without EXPLICIT=1" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
STREAM_BENCH="$(dirname "$(readlink -f "$0")")/stream_bench.py"
[[ -x "$STREAM_BENCH" ]] || { echo "[run-bench] missing $STREAM_BENCH" >&2; exit 2; }

BASE_URL="http://127.0.0.1:${BENCH_PORT}"

# -----------------------------------------------------------------------------
# Server lifecycle helpers
# -----------------------------------------------------------------------------
SERVER_PID=""

start_server() {
    local m="$1"
    local log="$OUT_DIR/chimere-server-M${m}.log"
    echo "[run-bench] starting chimere-server M=$m on :$BENCH_PORT (log: $log)" >&2
    (
        export LD_LIBRARY_PATH=/home/remondiere/ik_llama.cpp/build_sm120/ggml/src:/home/remondiere/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64
        export CHIMERE_LLAMA_BACKEND=1
        export CHIMERE_MODEL=/home/remondiere/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
        export CHIMERE_TOKENIZER=/home/remondiere/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/main/tokenizer.json
        export CHIMERE_MMPROJ=/home/remondiere/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/mmproj-BF16.gguf
        export CHIMERE_PORT="$BENCH_PORT"
        export CHIMERE_NCMOE=0
        export CHIMERE_KV_MAX_SEQ=16384
        export CHIMERE_ENGRAM_DIR=/home/remondiere/.openclaw/data/engram
        export CHIMERE_MULTISLOT="$m"
        export CHIMERE_MULTISLOT_NATIVE=1
        export CHIMERE_SKIP_LEGACY_LLAMA=1
        export CHIMERE_SKIP_SAMPLER_INIT=0
        export CHIMERE_PROFILE=1
        export LLAMA_SET_ROWS=1
        exec "$SERVER_ROOT/target/release/chimere-server"
    ) >"$log" 2>&1 &
    SERVER_PID=$!
    echo "[run-bench] server pid=$SERVER_PID, waiting for /health…" >&2
    for i in $(seq 1 180); do
        if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
            echo "[run-bench] server ready after ${i}s" >&2
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[run-bench] server died; tail of log:" >&2
            tail -50 "$log" >&2
            exit 3
        fi
        sleep 1
    done
    echo "[run-bench] server did not become healthy in 180 s; tail:" >&2
    tail -80 "$log" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    exit 3
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run-bench] stopping chimere-server (pid $SERVER_PID)…" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        # Give up to 30s for graceful shutdown; then SIGKILL
        for i in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

cleanup() {
    stop_server
    if [[ -n "${DMON_PID:-}" ]]; then kill "$DMON_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
SUMMARY_JSON="$OUT_DIR/sweep-summary.json"
echo "[]" > "$SUMMARY_JSON"

for M in $MULTISLOT_SWEEP; do
    LABEL="M${M}"
    echo "=== [run-bench] pass $LABEL =============================" >&2

    if [[ "$SKIP_START_SERVER" != "1" ]]; then
        start_server "$M"
    fi

    # Start nvidia-smi dmon (GPU telemetry) for the duration of this pass
    DMON_CSV="$OUT_DIR/nvidia-smi-dmon-$LABEL.csv"
    nvidia-smi dmon -s puct -d 1 -c 1200 -o T > "$DMON_CSV" 2>&1 &
    DMON_PID=$!

    # Drive the bench
    python3 "$STREAM_BENCH" \
        --url "$BASE_URL" \
        --n "$N_REQS" \
        --conc "$CONC" \
        --max-tokens "$MAX_TOKENS" \
        --prompt-set "$PROMPT_SET" \
        --label "$LABEL" \
        --out-dir "$OUT_DIR" \
        --capture-metrics \
        $([[ "$BENCH_PORT" == "8081" ]] && echo --explicit-prod) \
        > "$OUT_DIR/bench-stdout-$LABEL.json" \
        2> "$OUT_DIR/bench-stderr-$LABEL.log" || {
            echo "[run-bench] bench driver failed for $LABEL; see stderr log" >&2
            kill "$DMON_PID" 2>/dev/null || true
            [[ "$SKIP_START_SERVER" != "1" ]] && stop_server
            continue
        }

    # Stop telemetry
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""

    # Stop server between passes
    if [[ "$SKIP_START_SERVER" != "1" ]]; then
        stop_server
        sleep 3
    fi

    echo "[run-bench] pass $LABEL complete; summary at $OUT_DIR/summary-$LABEL.json" >&2
done

# -----------------------------------------------------------------------------
# Final aggregate: concatenate per-pass summaries into a single list
# -----------------------------------------------------------------------------
echo "[run-bench] aggregating sweep summaries" >&2
python3 - <<'PY' "$OUT_DIR"
import json, sys, pathlib
out_dir = pathlib.Path(sys.argv[1])
summaries = []
for f in sorted(out_dir.glob("summary-M*.json")):
    summaries.append(json.loads(f.read_text()))
(out_dir / "sweep-summary.json").write_text(json.dumps(summaries, indent=2))
# Quick terminal table
print()
print(f"  {'pass':<6}  {'n_ok':>5}  {'wall_s':>7}  {'agg_tps':>8}  {'ttft_p50':>9}  {'ttft_p99':>9}  {'dec_p50':>8}  {'dec_p90':>8}")
for s in summaries:
    print(
        f"  {s['label']:<6}  {s['n_ok']:>5}  {s['wall_s']:>7}  "
        f"{s['agg_tok_per_s']:>8}  "
        f"{(s['ttft_ms']['p50'] or 0):>9.1f}  "
        f"{(s['ttft_ms']['p99'] or 0):>9.1f}  "
        f"{(s['per_req_decode_tok_per_s']['p50'] or 0):>8.1f}  "
        f"{(s['per_req_decode_tok_per_s']['p90'] or 0):>8.1f}"
    )
PY

echo "[run-bench] done; artifacts in $OUT_DIR" >&2
