#!/usr/bin/env bash
# sweep-bench.sh — llama-bench-style grid harness for chimere-server.
#
# chimere-server is a Rust FFI + native-scheduler stack, so the stock
# `llama-bench` does not apply. This script plays the same role for
# the knobs chimere actually honours:
#
#   M     = CHIMERE_MULTISLOT             (how many slots)
#   NCMOE = CHIMERE_NCMOE                 (how many MoE layers on CPU)
#   PCH   = CHIMERE_MAX_PREFILL_CHUNK     (prefill tokens per driver tick)
#
# For each cell of the grid it:
#   1. Starts a fresh chimere-server child on :8082 with that env.
#   2. Waits for /health (up to 180 s; model load is 30-60 s).
#   3. Runs N streaming requests through the existing stream_bench.py
#      at concurrency = min(M, --conc-cap), capturing /metrics and
#      /v1/status pre/post plus 1 Hz nvidia-smi telemetry.
#   4. Stops the server, records aggregated metrics into sweep.csv.
#   5. Moves on to the next cell.
#
# At the end, it emits REPORT.md (filled-in from REPORT-TEMPLATE.md)
# with tables and a sweet-spot recommendation.
#
# SAFETY:
#   - Refuses to bench against :8081 (production) without EXPLICIT=1.
#   - Traps EXIT/INT/TERM and kills the child chimere-server cleanly.
#   - Uses `exec` in the child so it is a direct pid we can kill.
#
# BUDGET:
#   - Model load is 30-60 s per cell.
#   - A 3×1×1=3-cell sweep is ~5-10 min.
#   - A 3×3×2=18-cell sweep is ~30-60 min.
#   - A 4×3×4=48-cell sweep is ~60-90 min.
#   Start small, expand later.
#
# USAGE:
#   ./sweep-bench.sh --output-dir /tmp/chimere-sweep
#   ./sweep-bench.sh --multislot-sweep "1 4" --ncmoe-sweep "0 4" \
#                    --prefill-chunk-sweep "128 256 512" \
#                    --output-dir /tmp/chimere-sweep-wide
#   ./sweep-bench.sh --dry-run --multislot-sweep "1 2 4 8"
#   ./sweep-bench.sh --skip-server --port 8082   # already-running server
#
set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults (all override-able via CLI flags below)
# -----------------------------------------------------------------------------
PORT=8082
MULTISLOT_SWEEP="1 2 4"
NCMOE_SWEEP="0"
PREFILL_CHUNK_SWEEP="256"
PROMPT_SET=""               # default: prompts.yaml next to this script
MAX_TOKENS=128
N_REQS_PER_PASS=20
CONC_CAP=0                  # 0 = use M as concurrency (cap at 8)
SKIP_SERVER=0
DRY_RUN=0
OUTPUT_DIR=""
EXPLICIT=0
SERVER_ROOT="${SERVER_ROOT:-/home/remondiere/github-repos/chimere/chimere-server}"
LD_LIBRARY_PATH_DEFAULT="/home/remondiere/ik_llama.cpp/build_sm120/ggml/src:/home/remondiere/ik_llama.cpp/build_sm120/src:/usr/local/cuda-12.8/lib64"

# Model defaults pulled from production systemd unit (see MEMORY.md).
# Overridable via pre-existing env.
: "${CHIMERE_MODEL:=/home/remondiere/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
: "${CHIMERE_TOKENIZER:=/home/remondiere/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/main/tokenizer.json}"
: "${CHIMERE_MMPROJ:=/home/remondiere/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/mmproj-BF16.gguf}"
: "${CHIMERE_KV_MAX_SEQ:=16384}"
: "${CHIMERE_ENGRAM_DIR:=/home/remondiere/.openclaw/data/engram}"

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
usage() {
    sed -n '2,45p' "$0"
    exit "${1:-0}"
}

while (( $# > 0 )); do
    case "$1" in
        --port)                 PORT="$2"; shift 2 ;;
        --multislot-sweep)      MULTISLOT_SWEEP="$2"; shift 2 ;;
        --ncmoe-sweep)          NCMOE_SWEEP="$2"; shift 2 ;;
        --prefill-chunk-sweep)  PREFILL_CHUNK_SWEEP="$2"; shift 2 ;;
        --prompt-set)           PROMPT_SET="$2"; shift 2 ;;
        --max-tokens)           MAX_TOKENS="$2"; shift 2 ;;
        --n-requests-per-pass)  N_REQS_PER_PASS="$2"; shift 2 ;;
        --conc-cap)             CONC_CAP="$2"; shift 2 ;;
        --skip-server)          SKIP_SERVER=1; shift ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=1; shift ;;
        --explicit-prod)        EXPLICIT=1; shift ;;
        -h|--help)              usage 0 ;;
        *)  echo "[sweep-bench] unknown arg: $1" >&2; usage 2 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "[sweep-bench] --output-dir is required" >&2
    exit 2
fi

if [[ "$PORT" == "8081" && "$EXPLICIT" != "1" ]]; then
    echo "[sweep-bench] refusing to target :8081 (prod). Use --explicit-prod to override." >&2
    exit 2
fi

HERE="$(dirname "$(readlink -f "$0")")"
# Default prompt-set: prompts.yaml shipped alongside this script.
if [[ -z "$PROMPT_SET" ]]; then
    PROMPT_SET="$HERE/prompts.yaml"
fi
if [[ ! -r "$PROMPT_SET" ]]; then
    echo "[sweep-bench] prompt-set not readable: $PROMPT_SET" >&2
    exit 2
fi

STREAM_BENCH="$SERVER_ROOT/benchmarks/stream_bench.py"
if [[ ! -x "$STREAM_BENCH" ]]; then
    # Fall back to python3 invocation if not +x
    if [[ -r "$STREAM_BENCH" ]]; then
        echo "[sweep-bench] note: $STREAM_BENCH not executable, will invoke via python3" >&2
    else
        echo "[sweep-bench] missing $STREAM_BENCH (check SERVER_ROOT)" >&2
        exit 2
    fi
fi

DRIVER_WRAPPER="$HERE/driver_wrapper.py"
if [[ ! -r "$DRIVER_WRAPPER" ]]; then
    echo "[sweep-bench] missing $DRIVER_WRAPPER — did you untar the harness?" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"/{raw,logs}

BASE_URL="http://127.0.0.1:${PORT}"
SWEEP_CSV="$OUTPUT_DIR/sweep.csv"
REPORT_MD="$OUTPUT_DIR/REPORT.md"
START_ISO="$(date -Iseconds)"
GIT_SHA="$(git -C "$SERVER_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# Total cell count — for progress output
n_cells=0
for _m in $MULTISLOT_SWEEP; do
    for _n in $NCMOE_SWEEP; do
        for _p in $PREFILL_CHUNK_SWEEP; do
            n_cells=$((n_cells + 1))
        done
    done
done

echo "========================================================"
echo "  chimere-server sweep-bench"
echo "  start:       $START_ISO"
echo "  git SHA:     $GIT_SHA"
echo "  output dir:  $OUTPUT_DIR"
echo "  port:        $PORT"
echo "  multislot:   $MULTISLOT_SWEEP"
echo "  ncmoe:       $NCMOE_SWEEP"
echo "  prefill_ch:  $PREFILL_CHUNK_SWEEP"
echo "  prompt-set:  $PROMPT_SET"
echo "  n_reqs/pass: $N_REQS_PER_PASS"
echo "  max_tokens:  $MAX_TOKENS"
echo "  cells:       $n_cells"
echo "  est. time:   $((n_cells * 2))-$((n_cells * 3)) minutes"
echo "========================================================"

if (( DRY_RUN )); then
    echo "[sweep-bench] --dry-run: not launching anything. Goodbye."
    exit 0
fi

# -----------------------------------------------------------------------------
# Server lifecycle
# -----------------------------------------------------------------------------
SERVER_PID=""
DMON_PID=""

cleanup() {
    local rc=$?
    [[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[sweep-bench] cleanup: stopping server pid=$SERVER_PID" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        # give 20s for graceful shutdown
        for i in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

start_server() {
    local M="$1" NCMOE="$2" PCH="$3" cell_tag="$4"
    local log="$OUTPUT_DIR/logs/chimere-server-${cell_tag}.log"

    echo "[sweep-bench] [$cell_tag] starting chimere-server on :$PORT ..." >&2
    echo "[sweep-bench] [$cell_tag]   log: $log" >&2

    (
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$LD_LIBRARY_PATH_DEFAULT}"
        export CHIMERE_LLAMA_BACKEND=1
        export CHIMERE_MODEL
        export CHIMERE_TOKENIZER
        export CHIMERE_MMPROJ
        export CHIMERE_PORT="$PORT"
        export CHIMERE_NCMOE="$NCMOE"
        export CHIMERE_KV_MAX_SEQ
        export CHIMERE_ENGRAM_DIR
        export CHIMERE_MULTISLOT="$M"
        # NativeScheduler only arms at M>=2; at M=1 the env is harmless (ignored).
        export CHIMERE_MULTISLOT_NATIVE=1
        export CHIMERE_SKIP_LEGACY_LLAMA=1
        export CHIMERE_SKIP_SAMPLER_INIT=0
        export CHIMERE_PROFILE=1
        export CHIMERE_MAX_PREFILL_CHUNK="$PCH"
        # Back-compat alias for versions without the CHIMERE_MAX_PREFILL_CHUNK
        # patch applied. Setting both is harmless — the newer wins when both
        # are read (see patch in this directory).
        export CHIMERE_NATIVE_MAX_PREFILL_CHUNK="$PCH"
        export LLAMA_SET_ROWS=1
        exec "$SERVER_ROOT/target/release/chimere-server"
    ) >"$log" 2>&1 &
    SERVER_PID=$!

    # Wait for /health (up to 180 s — 30-60 s model load + some slack)
    for i in $(seq 1 180); do
        if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
            echo "[sweep-bench] [$cell_tag]   healthy after ${i}s (pid=$SERVER_PID)" >&2
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[sweep-bench] [$cell_tag] server DIED during boot. Tail of log:" >&2
            tail -80 "$log" >&2
            SERVER_PID=""
            return 1
        fi
        # Progress heartbeat every 15 s
        if (( i % 15 == 0 )); then
            echo "[sweep-bench] [$cell_tag]   still booting ... (${i}s)" >&2
        fi
        sleep 1
    done

    echo "[sweep-bench] [$cell_tag] server did not become healthy in 180 s" >&2
    tail -80 "$log" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    return 1
}

stop_server() {
    [[ -z "${SERVER_PID:-}" ]] && return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[sweep-bench]   stopping server pid=$SERVER_PID ..." >&2
        kill "$SERVER_PID" 2>/dev/null || true
        for i in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    # Give the GPU a moment to release VRAM before the next cell boots
    sleep 3
}

# -----------------------------------------------------------------------------
# CSV header
# -----------------------------------------------------------------------------
{
    echo "cell_tag,git_sha,multislot,ncmoe,prefill_chunk,n_reqs,conc,max_tokens,prompt_set_path,wall_s,total_gen_tokens,agg_tok_per_s,per_req_decode_p50,per_req_decode_p99,ttft_ms_p50,ttft_ms_p99,inter_tok_ms_p50,inter_tok_ms_p99,vram_used_mib_p50,vram_used_mib_p95,gpu_sm_p50,gpu_mem_p50,gpu_pwr_p50_w,n_ok,n_err,errors_head"
} > "$SWEEP_CSV"

# -----------------------------------------------------------------------------
# Sweep loop
# -----------------------------------------------------------------------------
cell_idx=0
cells_ok=0
cells_fail=0

for M in $MULTISLOT_SWEEP; do
    for NCMOE in $NCMOE_SWEEP; do
        for PCH in $PREFILL_CHUNK_SWEEP; do
            cell_idx=$((cell_idx + 1))
            cell_tag="M${M}-N${NCMOE}-P${PCH}"
            echo ""
            echo "=== [sweep-bench] cell ${cell_idx}/${n_cells}: ${cell_tag} ==="

            # Concurrency: default to M (cap at CONC_CAP if >0, else cap at 8)
            if (( CONC_CAP > 0 )); then
                CONC="$(( M < CONC_CAP ? M : CONC_CAP ))"
            else
                CONC="$(( M < 8 ? M : 8 ))"
            fi
            # Always at least 1
            (( CONC < 1 )) && CONC=1

            if (( SKIP_SERVER == 0 )); then
                if ! start_server "$M" "$NCMOE" "$PCH" "$cell_tag"; then
                    echo "[sweep-bench] [$cell_tag] SKIP: server did not start" >&2
                    cells_fail=$((cells_fail + 1))
                    echo "${cell_tag},${GIT_SHA},${M},${NCMOE},${PCH},0,${CONC},${MAX_TOKENS},${PROMPT_SET},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,server_boot_failed" >> "$SWEEP_CSV"
                    continue
                fi
            fi

            # VRAM baseline right after boot (before load)
            VRAM_BASELINE="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' \n' || echo '0')"
            echo "[sweep-bench] [$cell_tag] VRAM post-boot: ${VRAM_BASELINE} MiB used"

            # 1 Hz GPU telemetry during the pass
            DMON_CSV="$OUTPUT_DIR/raw/nvidia-smi-dmon-${cell_tag}.csv"
            nvidia-smi dmon -s pumct -d 1 -c 1200 -o T > "$DMON_CSV" 2>&1 &
            DMON_PID=$!

            # Run the driver wrapper (expands prompts.yaml to a prompt list
            # for stream_bench.py, then aggregates and emits per-cell JSON).
            CELL_RAW="$OUTPUT_DIR/raw/cell-${cell_tag}"
            mkdir -p "$CELL_RAW"
            set +e
            python3 "$DRIVER_WRAPPER" \
                --url "$BASE_URL" \
                --stream-bench "$STREAM_BENCH" \
                --prompt-set "$PROMPT_SET" \
                --n "$N_REQS_PER_PASS" \
                --conc "$CONC" \
                --max-tokens "$MAX_TOKENS" \
                --label "$cell_tag" \
                --out-dir "$CELL_RAW" \
                > "$OUTPUT_DIR/logs/driver-${cell_tag}.log" 2>&1
            driver_rc=$?
            set -e

            # Stop telemetry
            kill "$DMON_PID" 2>/dev/null || true
            wait "$DMON_PID" 2>/dev/null || true
            DMON_PID=""

            if (( driver_rc != 0 )); then
                echo "[sweep-bench] [$cell_tag] DRIVER FAILED (rc=$driver_rc). Tail:" >&2
                tail -40 "$OUTPUT_DIR/logs/driver-${cell_tag}.log" >&2
                (( SKIP_SERVER == 0 )) && stop_server
                cells_fail=$((cells_fail + 1))
                echo "${cell_tag},${GIT_SHA},${M},${NCMOE},${PCH},0,${CONC},${MAX_TOKENS},${PROMPT_SET},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,driver_failed_rc${driver_rc}" >> "$SWEEP_CSV"
                continue
            fi

            # Aggregate into the CSV. The wrapper leaves us a
            # cell-summary.json next to the stream_bench artifacts.
            python3 "$HERE/csv_append.py" \
                --cell-tag "$cell_tag" \
                --git-sha "$GIT_SHA" \
                --multislot "$M" \
                --ncmoe "$NCMOE" \
                --prefill-chunk "$PCH" \
                --conc "$CONC" \
                --max-tokens "$MAX_TOKENS" \
                --prompt-set "$PROMPT_SET" \
                --cell-raw "$CELL_RAW" \
                --dmon-csv "$DMON_CSV" \
                --vram-baseline "$VRAM_BASELINE" \
                --csv "$SWEEP_CSV"

            echo "[sweep-bench] [$cell_tag] OK."
            cells_ok=$((cells_ok + 1))

            if (( SKIP_SERVER == 0 )); then
                stop_server
            fi
        done
    done
done

# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------
END_ISO="$(date -Iseconds)"
echo ""
echo "========================================================"
echo "  sweep done. ok=$cells_ok  fail=$cells_fail  total=$n_cells"
echo "  csv:    $SWEEP_CSV"
echo "========================================================"

python3 "$HERE/render_report.py" \
    --csv "$SWEEP_CSV" \
    --template "$HERE/REPORT-TEMPLATE.md.tmpl" \
    --output "$REPORT_MD" \
    --start "$START_ISO" \
    --end "$END_ISO" \
    --git-sha "$GIT_SHA" \
    --server-root "$SERVER_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$CHIMERE_MODEL" \
    --multislot-sweep "$MULTISLOT_SWEEP" \
    --ncmoe-sweep "$NCMOE_SWEEP" \
    --prefill-chunk-sweep "$PREFILL_CHUNK_SWEEP" \
    --n-reqs-per-pass "$N_REQS_PER_PASS" \
    --max-tokens "$MAX_TOKENS" \
    --prompt-set "$PROMPT_SET" \
    || echo "[sweep-bench] render_report.py failed (csv is still usable)"

echo "  report: $REPORT_MD"
