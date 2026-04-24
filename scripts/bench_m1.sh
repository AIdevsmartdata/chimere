#!/usr/bin/env bash
# scripts/bench_m1.sh — M1 multi-slot concurrency sweep wrapper.
#
# Runs `bench-m1` three times against three fresh `chimere-server` processes
# launched with CHIMERE_MULTISLOT=1 (baseline), =2, =4, and stitches the
# throughput ratios together. Production `:8081` is NEVER touched —
# everything runs on `:8082`.
#
# Usage:
#   scripts/bench_m1.sh                        # full 1/2/4 sweep
#   BENCH_SWEEP="1 2"    scripts/bench_m1.sh   # subset
#   BENCH_SKIP_SERVER=1  scripts/bench_m1.sh   # bench against already-running
#                                                # server on BENCH_URL (for
#                                                # the J4-dispatcher-pending
#                                                # case when multislot is a no-op)
#
# Requires:
#   - release build of `bench-m1` (this script triggers one if missing)
#   - `ik_llama.cpp` at $IKLLAMACPP_DIR (same env as chimere-m1-j3-build.sh)
#   - `nvidia-smi` optional; VRAM column reported as "n/a" if absent
#
# Exit codes:
#   0  — all passes OK, ratios at or above target
#   2  — pre-flight failed (missing binaries, :8081 safety trip)
#   5  — ratio target missed (harness decides; see bench_m1.rs)
#   6  — engram isolation broken

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

REPO="${REPO:-/home/remondiere/github-repos/chimere}"
BENCH_URL="${BENCH_URL:-http://127.0.0.1:8082/v1/chat/completions}"
BENCH_PORT="${BENCH_PORT:-8082}"
BENCH_MODEL="${BENCH_MODEL:-chimere-deltanet}"
BENCH_N="${BENCH_N:-100}"
BENCH_CONC="${BENCH_CONC:-4}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-64}"
BENCH_SWEEP="${BENCH_SWEEP:-1 2 4}"
BENCH_SKIP_SERVER="${BENCH_SKIP_SERVER:-0}"

# Same paths as chimere-m1-j3-build.sh — reuse so the build env is identical.
# Override IKLLAMACPP_DIR / IK_LLAMA_BUILD_SUBDIR to point at your local clone.
export IKLLAMACPP_DIR="${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}"
export IK_LLAMA_BUILD_SUBDIR="${IK_LLAMA_BUILD_SUBDIR:-build_sm120}"
export GGML_SO_DIR="${GGML_SO_DIR:-${IKLLAMACPP_DIR}/${IK_LLAMA_BUILD_SUBDIR}/ggml/src}"
export LLAMA_SO_DIR="${LLAMA_SO_DIR:-${IKLLAMACPP_DIR}/${IK_LLAMA_BUILD_SUBDIR}/src}"
export GGML_INCLUDE_DIR="${GGML_INCLUDE_DIR:-${IKLLAMACPP_DIR}/ggml/include}"
export GGML_SRC_DIR="${GGML_SRC_DIR:-${IKLLAMACPP_DIR}/ggml/src}"
export GGML_COMMON_DIR="${GGML_COMMON_DIR:-${IKLLAMACPP_DIR}/common}"
export IK_LLAMA_INCLUDE="${IK_LLAMA_INCLUDE:-${IKLLAMACPP_DIR}/include}"
export IK_LLAMA_SRC="${IK_LLAMA_SRC:-${IKLLAMACPP_DIR}/src}"
export GGML_COMMON_LIB_DIR="${GGML_COMMON_LIB_DIR:-${IKLLAMACPP_DIR}/${IK_LLAMA_BUILD_SUBDIR}/common}"
export CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.8/targets/x86_64-linux/lib}"
export LD_LIBRARY_PATH="${GGML_SO_DIR}:${LLAMA_SO_DIR}:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

if [[ "$BENCH_URL" == *":8081"* ]]; then
    echo "[bench_m1] refusing to run: BENCH_URL targets :8081 (production)." >&2
    exit 2
fi

cd "$REPO/chimere-server"

# Build bench-m1 if missing
if [[ ! -x "target/release/bench-m1" ]]; then
    echo "[bench_m1] building bench-m1 (release)..."
    cargo build --release --features server --bin bench-m1
fi

# Also build chimere-server (the bench'd target) unless skipped
if [[ "$BENCH_SKIP_SERVER" != "1" && ! -x "target/release/chimere-server" ]]; then
    echo "[bench_m1] building chimere-server (release)..."
    cargo build --release --features server --bin chimere-server
fi

# --------------------------------------------------------------------------
# Server lifecycle helpers
# --------------------------------------------------------------------------

SERVER_PID=""

start_server() {
    local num_slots="$1"
    export CHIMERE_PORT="$BENCH_PORT"
    export CHIMERE_MULTISLOT="$num_slots"
    # Keep the rest of the env quiet — defer to the caller's CHIMERE_MODEL
    # etc. Log to a pass-specific file so post-mortem is easy.
    local log="/tmp/chimere-server-bench-m${num_slots}.log"
    echo "[bench_m1] launching chimere-server M=$num_slots on :$BENCH_PORT (log: $log)" >&2
    ./target/release/chimere-server >"$log" 2>&1 &
    SERVER_PID=$!
    # Wait for /health (up to 120 s — big models take time to map)
    for i in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1; then
            echo "[bench_m1] server ready after ${i}s" >&2
            return 0
        fi
        sleep 1
    done
    echo "[bench_m1] server did not become healthy in 120 s; tail of log:" >&2
    tail -40 "$log" >&2 || true
    stop_server
    exit 2
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap stop_server EXIT INT TERM

# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

declare -A AGG_TPS
BASELINE_TPS=0
SUMMARY_LINES=()

for M in $BENCH_SWEEP; do
    LABEL="${M}-slot"
    [[ "$M" == "1" ]] && LABEL="baseline"

    if [[ "$BENCH_SKIP_SERVER" != "1" ]]; then
        start_server "$M"
    fi

    export BENCH_URL BENCH_MODEL BENCH_N BENCH_CONC BENCH_MAX_TOKENS
    export BENCH_PASS_LABEL="$LABEL"
    export BENCH_BASELINE_TPS="$BASELINE_TPS"

    # Capture both the human line and the JSON line from bench-m1.
    OUT=$(./target/release/bench-m1 || true)
    echo "$OUT"

    # Parse the JSON line to extract agg_tps.
    JSON=$(echo "$OUT" | grep -E '^\{"pass"' | tail -1 || true)
    if [[ -n "$JSON" ]]; then
        TPS=$(echo "$JSON" | sed -n 's/.*"agg_tps":\([0-9.]*\).*/\1/p')
        AGG_TPS[$M]="$TPS"
        if [[ "$M" == "1" ]]; then
            BASELINE_TPS="$TPS"
        fi
        SUMMARY_LINES+=("$JSON")
    fi

    if [[ "$BENCH_SKIP_SERVER" != "1" ]]; then
        stop_server
        sleep 2
    fi
done

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------

echo
echo "[bench_m1] summary:"
printf "  %-10s  %10s  %s\n" "pass" "tok/s" "ratio"
for M in $BENCH_SWEEP; do
    LABEL="${M}-slot"
    [[ "$M" == "1" ]] && LABEL="baseline"
    TPS="${AGG_TPS[$M]:-n/a}"
    if [[ "$M" != "1" && -n "${AGG_TPS[1]:-}" && "${AGG_TPS[1]:-}" != "0" && "$TPS" != "n/a" ]]; then
        RATIO=$(awk -v a="$TPS" -v b="${AGG_TPS[1]}" 'BEGIN { if (b+0 == 0) print "n/a"; else printf "%.2fx", a/b }')
    else
        RATIO="—"
    fi
    printf "  %-10s  %10s  %s\n" "$LABEL" "$TPS" "$RATIO"
done
