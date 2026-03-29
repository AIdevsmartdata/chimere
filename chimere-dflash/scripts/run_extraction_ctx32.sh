#!/bin/bash
# run_extraction_ctx32.sh — Auto-restart C++ extractor for ctx_len=32
#
# Usage: nohup bash scripts/run_extraction_ctx32.sh &
# All output goes to LOG file below. Do NOT redirect stdout/stderr externally.
set -uo pipefail

MODEL="$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
EXTRACTOR="$HOME/chimere-dflash/extract/build/extract_single_position"
INPUT="$HOME/chimere-dflash/data/prompts_v6/combined_150k.jsonl"
OUTPUT="$HOME/chimere-dflash/data/features_q5_ctx32"
LOG="$HOME/chimere-dflash/data/extraction_ctx32.log"
MAX_RESTARTS=50
MAX_SEQ_LEN=512
CTX_LEN=32
RESUME_FROM="${1:-0}"

# All output to LOG — wrapper handles everything
exec >> "$LOG" 2>&1

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo ""
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting extraction ctx32 (resume=$RESUME_FROM, restart #$restart_count)"
    echo "================================================================"

    "$EXTRACTOR" \
        -m "$MODEL" \
        -ngl 99 -ot "blk.2[3-9].ffn_.*_exps.weight=CPU" -ot "blk.3[0-9].ffn_.*_exps.weight=CPU" \
        --flash-attn on -b 512 -ub 512 \
        --cache-type-k q8_0 --cache-type-v q4_0 \
        -c 576 \
        --layers 1,10,19,28,37 \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --ctx-len "$CTX_LEN" \
        --block-size 16 \
        --resume-from "$RESUME_FROM"

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extraction completed successfully"
        break
    fi

    restart_count=$((restart_count + 1))

    # Find last processed prompt from log
    last_line=$(grep "prompts/s" "$LOG" | tail -1)
    last_prompt=$(echo "$last_line" | grep -oP '\[\K[0-9]+(?=/)' || echo "$RESUME_FROM")
    RESUME_FROM=$((last_prompt + 100))  # skip ahead past the problematic prompt

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Crashed (exit=$exit_code). Restarting from $RESUME_FROM (restart $restart_count/$MAX_RESTARTS)"
    echo "Waiting 15s for VRAM cleanup..."
    sleep 15  # cooldown — GPU needs time to release VRAM after OOM
done
