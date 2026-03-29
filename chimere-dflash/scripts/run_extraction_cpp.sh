#!/bin/bash
# run_extraction_cpp.sh — Auto-restart C++ extractor on CUDA OOM crashes
set -uo pipefail

MODEL="$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
EXTRACTOR="$HOME/chimere-dflash/extract/build/extract_single_position"
INPUT="$HOME/chimere-dflash/data/prompts_v6/combined_150k.jsonl"
OUTPUT="$HOME/chimere-dflash/data/features_q5"
LOG="$HOME/chimere-dflash/extraction_cpp.log"
MAX_RESTARTS=50
MAX_SEQ_LEN=512
RESUME_FROM="${1:-0}"

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "[$(date '+%H:%M:%S')] Starting extraction (resume=$RESUME_FROM, restart #$restart_count)" | tee -a "$LOG"

    "$EXTRACTOR" \
        -m "$MODEL" \
        -ngl 99 -ot "blk.[2-3][0-9].ffn_.*_exps.weight=CPU" \
        --flash-attn on -b 4096 -ub 4096 \
        --cache-type-k q8_0 --cache-type-v q4_0 \
        -c 576 \
        --layers 1,10,19,28,37 \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --resume-from "$RESUME_FROM" \
        >> "$LOG" 2>&1

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Extraction completed successfully" | tee -a "$LOG"
        break
    fi

    restart_count=$((restart_count + 1))

    # Find last processed prompt from log
    last_line=$(grep "prompts/s" "$LOG" | tail -1)
    last_prompt=$(echo "$last_line" | grep -oP '\[\K[0-9]+(?=/)' || echo "$RESUME_FROM")
    RESUME_FROM=$((last_prompt + 100))  # skip ahead past the problematic prompt

    echo "[$(date '+%H:%M:%S')] Crashed (exit=$exit_code). Restarting from $RESUME_FROM (restart $restart_count/$MAX_RESTARTS)" | tee -a "$LOG"
    sleep 3  # cooldown
done
