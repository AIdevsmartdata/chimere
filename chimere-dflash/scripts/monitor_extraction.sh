#!/bin/bash
# monitor_extraction.sh — Monitor extraction, continue with merged data if needed to reach 100K
set -euo pipefail

TARGET=100000
FEATURES_DIR="/home/remondiere/chimere-dflash/data/features_q5"
PROMPTS_V6="/home/remondiere/chimere-dflash/data/prompts_v6"
ORIG_INPUT="$PROMPTS_V6/ready_for_extraction.jsonl"
MERGED_INPUT="$PROMPTS_V6/merged_all.jsonl"
COMBINED_INPUT="$PROMPTS_V6/combined_for_extraction.jsonl"
LOGFILE="/home/remondiere/chimere-dflash/extraction_monitor.log"
MODEL="$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

count_samples() { ls "$FEATURES_DIR" | grep -c "^sample_" || echo 0; }

# Phase 1: Wait for current extraction to finish
log "=== Monitor started. Target: $TARGET samples ==="
while true; do
    n=$(count_samples)
    # Check if extract_single_position.py is still running
    if ! pgrep -f "extract_single_position.py" > /dev/null 2>&1; then
        log "Extraction process stopped. Samples: $n"
        break
    fi
    log "Running... $n samples"
    sleep 120
done

n=$(count_samples)
log "Phase 1 done: $n samples"

if [ "$n" -ge "$TARGET" ]; then
    log "Already at $n >= $TARGET. Done!"
    exit 0
fi

# Phase 2: Prepare combined input with merged data
log "Need more data. Combining original + merged datasets..."
if [ ! -f "$MERGED_INPUT" ]; then
    log "ERROR: merged_all.jsonl not found!"
    exit 1
fi

# Combine: original prompts + new merged prompts (dedup already done in merge_datasets.py)
cat "$ORIG_INPUT" "$MERGED_INPUT" > "$COMBINED_INPUT"
total_prompts=$(wc -l < "$COMBINED_INPUT")
log "Combined input: $total_prompts prompts"

# Phase 3: Relaunch extraction from where we left off
# resume-from = total prompts in original file (we already processed those)
orig_count=$(wc -l < "$ORIG_INPUT")
log "Relaunching extraction from prompt index $orig_count (start of merged data)..."

PYTHONUNBUFFERED=1 python /home/remondiere/chimere-dflash/scripts/extract_single_position.py \
    --input "$COMBINED_INPUT" \
    --output "$FEATURES_DIR" \
    --model "$MODEL" \
    --max-seq-len 512 \
    --resume-from "$orig_count" \
    >> /home/remondiere/chimere-dflash/extraction_phase2.log 2>&1 &

PHASE2_PID=$!
log "Phase 2 extraction started (PID $PHASE2_PID)"

# Phase 3b: Monitor until 100K
while true; do
    n=$(count_samples)
    if [ "$n" -ge "$TARGET" ]; then
        log "Reached $n >= $TARGET! Killing extraction..."
        kill "$PHASE2_PID" 2>/dev/null || true
        wait "$PHASE2_PID" 2>/dev/null || true
        break
    fi
    if ! kill -0 "$PHASE2_PID" 2>/dev/null; then
        log "Phase 2 extraction stopped. Samples: $n"
        break
    fi
    log "Phase 2 running... $n samples"
    sleep 120
done

n=$(count_samples)
log "=== Final count: $n samples ==="
if [ "$n" -ge "$TARGET" ]; then
    log "SUCCESS: $n samples ready for training!"
else
    log "WARNING: Only $n samples. May need more prompts."
fi
