#!/usr/bin/env bash
# chimere-server vs llama-server vs vLLM — apples-to-apples bench harness.
#
# Targets a single hardware (RTX 5060 Ti, sm_120, 16 GB) and a single model
# family (Qwen3.6-35B-A3B). Each engine serves the same OpenAI-compatible
# /v1/chat/completions endpoint on different ports; this script brings each
# engine up in turn, hits it with the SAME workload, and dumps a CSV row.
#
# Models per engine:
#   - chimere-server : Qwen3.6-35B-A3B UD-IQ3_S (GGUF, ik_llama)
#   - llama-server   : Qwen3.6-35B-A3B UD-IQ3_S (GGUF, stock llama.cpp)
#   - vLLM           : RedHatAI/Qwen3-30B-A3B-* AWQ-INT4 (closest GGUF-equiv)
#
# Honesty caveat: vLLM does not load GGUF, so it serves a different (but
# closest) quantization of similar size. Reported as "comparable" not "identical".
#
# Workload: short (system-only, 256 tok), medium (3-turn, 1024 tok), long
# (8K context fill, 512 gen). Concurrency M = 1, 2, 4. Replicas R = 3.

set -euo pipefail

OUT_DIR="${OUT_DIR:-$(dirname "$0")/results-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"
RESULTS_CSV="$OUT_DIR/results.csv"
LOG="$OUT_DIR/run.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# -----------------------------------------------------------------------------
# Workloads
# -----------------------------------------------------------------------------
SHORT_PROMPT='Réponds en une phrase: quelle est la capitale du Japon ?'
MEDIUM_PROMPT='Explique en français en 200 mots ce qu est la technique Mamba-2 par rapport à Transformer.'
LONG_PROMPT='# Long-context fill (artificially padded to ~8K tokens for prefix-cache exercise) ...'

declare -A WORKLOADS=(
  [short]="$SHORT_PROMPT|256"
  [medium]="$MEDIUM_PROMPT|1024"
  [long]="$LONG_PROMPT|512"
)

# -----------------------------------------------------------------------------
# CSV header
# -----------------------------------------------------------------------------
if [ ! -f "$RESULTS_CSV" ]; then
  echo "timestamp,engine,model,workload,M,replica,n_prompt,n_gen,ttft_ms,total_ms,gen_tokps,prefill_tokps,vram_mb_peak,error" > "$RESULTS_CSV"
fi

# -----------------------------------------------------------------------------
# bench_one: $1=engine $2=port $3=workload $4=M $5=R
# -----------------------------------------------------------------------------
bench_one() {
  local engine=$1 port=$2 workload=$3 M=$4 R=$5
  local prompt="${WORKLOADS[$workload]%%|*}"
  local n_max="${WORKLOADS[$workload]##*|}"
  log "BENCH $engine $workload M=$M R=$R port=$port"

  # Capture VRAM peak in background
  local vram_log="$OUT_DIR/${engine}-${workload}-M${M}-R${R}-vram.txt"
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
    done ) > "$vram_log" &
  local vram_pid=$!

  # Run M concurrent requests R times
  python3 "$(dirname "$0")/bench-client.py" \
    --base-url "http://127.0.0.1:$port/v1" \
    --engine "$engine" \
    --workload "$workload" \
    --prompt "$prompt" \
    --max-tokens "$n_max" \
    --concurrency "$M" \
    --replicas "$R" \
    --output-csv "$RESULTS_CSV" \
    --vram-log "$vram_log" \
    2>&1 | tee -a "$LOG"

  kill "$vram_pid" 2>/dev/null || true
  wait "$vram_pid" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# Engine bring-up helpers
# -----------------------------------------------------------------------------
start_chimere() {
  log "Starting chimere-server on :8081 ..."
  systemctl --user start chimere-server.service
  sleep 30
  curl -sf http://127.0.0.1:8081/health > /dev/null
}
stop_chimere()  { log "Stopping chimere-server"; systemctl --user stop chimere-server.service || true; sleep 5; }

start_llama_server() {
  log "Starting stock llama-server on :8181 ..."
  ~/llama.cpp/build/bin/llama-server \
    -m ~/.openclaw/models/Qwen3.6-35B-A3B-IQ3_S/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
    --port 8181 -ngl 99 --flash-attn on -np 4 -c 16384 \
    --metrics > "$OUT_DIR/llama-server.log" 2>&1 &
  echo $! > "$OUT_DIR/llama-server.pid"
  sleep 60
  curl -sf http://127.0.0.1:8181/health > /dev/null
}
stop_llama_server() { log "Stopping llama-server"; kill "$(cat "$OUT_DIR/llama-server.pid")" 2>/dev/null || true; sleep 5; }

start_vllm() {
  log "Starting vLLM on :8281 ..."
  source ~/.openclaw/venvs/vllm/bin/activate
  vllm serve RedHatAI/Qwen3-30B-A3B-AWQ-INT4 \
    --port 8281 --max-model-len 16384 --gpu-memory-utilization 0.9 \
    --enable-prefix-caching --max-num-seqs 4 \
    > "$OUT_DIR/vllm.log" 2>&1 &
  echo $! > "$OUT_DIR/vllm.pid"
  sleep 90
  curl -sf http://127.0.0.1:8281/health > /dev/null
}
stop_vllm() { log "Stopping vLLM"; kill "$(cat "$OUT_DIR/vllm.pid")" 2>/dev/null || true; sleep 5; }

# -----------------------------------------------------------------------------
# Main bench matrix
# -----------------------------------------------------------------------------
ENGINES="${ENGINES:-chimere llama vllm}"
WORKLOAD_NAMES="${WORKLOAD_NAMES:-short medium long}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4}"
REPLICAS="${REPLICAS:-3}"

for engine in $ENGINES; do
  case $engine in
    chimere) start_chimere; PORT=8081 ;;
    llama)   start_llama_server; PORT=8181 ;;
    vllm)    start_vllm; PORT=8281 ;;
    *) log "unknown engine $engine"; continue ;;
  esac

  for workload in $WORKLOAD_NAMES; do
    for M in $CONCURRENCIES; do
      for R in $(seq 1 "$REPLICAS"); do
        bench_one "$engine" "$PORT" "$workload" "$M" "$R" || log "ERR $engine $workload M=$M R=$R"
      done
    done
  done

  case $engine in
    chimere) stop_chimere ;;
    llama)   stop_llama_server ;;
    vllm)    stop_vllm ;;
  esac
done

log "=== ALL DONE ==="
log "results: $RESULTS_CSV"
log "logs:    $OUT_DIR/"
