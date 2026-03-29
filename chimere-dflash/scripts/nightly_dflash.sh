#!/bin/bash
# =============================================================================
# nightly_dflash.sh — Production nightly pipeline for DFlash IQ3 adaptation
#
# Steps:
#   1. Find latest checkpoint (auto-discover)
#   2. Capture on random subset of all_prompts (avoid repeats via offset rotation)
#   3. Fine-tune on FULL merged IQ3 buffer
#   4. Benchmark τ on holdout eval prompts
#   5. Log results
#
# Run via systemd timer or manually:
#   cd ~/chimere-dflash && bash scripts/nightly_dflash.sh
# =============================================================================
set -euo pipefail

PROJ=~/chimere-dflash
cd "$PROJ"

# Safety: restart IQ3 service on any exit (error or normal)
cleanup() {
    if [ "${IQ3_WAS_RUNNING:-false}" = true ]; then
        echo "  [cleanup] Restarting qwen35-iq3.service..."
        systemctl --user start qwen35-iq3.service || true
    fi
}
trap cleanup EXIT

LOGDIR="$PROJ/logs/nightly"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "================================================================"
echo " DFlash Nightly Pipeline — $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_IQ3=~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_S.gguf
PROMPTS="$PROJ/data/all_prompts.jsonl"
EVAL_PROMPTS="$PROJ/data/eval_prompts.jsonl"
BUFFER="$PROJ/data/online_buffer_iq3"
DAEMON="$PROJ/extract/build/target_daemon"
STATE_FILE="$PROJ/data/nightly_state.json"
CAPTURE_BATCH=500   # prompts per nightly run
MAX_TOKENS=64

# ---------------------------------------------------------------------------
# 1. Find latest checkpoint
# ---------------------------------------------------------------------------
find_latest_checkpoint() {
    local best=""
    local best_mtime=0
    for d in "$PROJ"/checkpoints_v8_online_iq3_*/best.pt; do
        [ -f "$d" ] || continue
        local mt
        mt=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [ "$mt" -gt "$best_mtime" ]; then
            best_mtime=$mt
            best="$d"
        fi
    done
    # Fallback to c4 Q5 checkpoint
    if [ -z "$best" ]; then
        best="$PROJ/checkpoints_v8_online_c4/best.pt"
    fi
    echo "$best"
}

CHECKPOINT=$(find_latest_checkpoint)
echo "  Checkpoint: $CHECKPOINT"

# ---------------------------------------------------------------------------
# 1b. Stop IQ3 production service if running (VRAM conflict with daemon)
# ---------------------------------------------------------------------------
IQ3_WAS_RUNNING=false
if systemctl --user is-active --quiet qwen35-iq3.service 2>/dev/null; then
    echo "  Stopping qwen35-iq3.service for capture..."
    systemctl --user stop qwen35-iq3.service
    IQ3_WAS_RUNNING=true
    sleep 2
fi

# ---------------------------------------------------------------------------
# 2. Determine prompt offset (rotate through all_prompts across nights)
# ---------------------------------------------------------------------------
TOTAL_PROMPTS=$(wc -l < "$PROMPTS")
OFFSET=0
if [ -f "$STATE_FILE" ]; then
    OFFSET=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('next_offset', 0))")
fi
# Wrap around
if [ "$OFFSET" -ge "$TOTAL_PROMPTS" ]; then
    OFFSET=0
fi
NEXT_OFFSET=$((OFFSET + CAPTURE_BATCH))
echo "  Prompts: $PROMPTS ($TOTAL_PROMPTS total, offset=$OFFSET, batch=$CAPTURE_BATCH)"

# ---------------------------------------------------------------------------
# 3. Capture
# ---------------------------------------------------------------------------
echo ""
echo "--- CAPTURE PHASE ---"
BUFFER_BEFORE=$(ls "$BUFFER" 2>/dev/null | grep -c sample || echo 0)

python3 scripts/bulk_capture_v2.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL_IQ3" \
    --daemon "$DAEMON" \
    --prompts "$PROMPTS" \
    --buffer-dir "$BUFFER" \
    --drafter-device cpu \
    --full-gpu \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$OFFSET" \
    --max-prompts "$CAPTURE_BATCH"

BUFFER_AFTER=$(ls "$BUFFER" | grep -c sample || echo 0)
NEW_SAMPLES=$((BUFFER_AFTER - BUFFER_BEFORE))
echo "  New samples: $NEW_SAMPLES (buffer: $BUFFER_BEFORE → $BUFFER_AFTER)"

# Save rotation state
python3 -c "
import json
state = {'next_offset': $NEXT_OFFSET, 'last_run': '$(date -Iseconds)', 'buffer_size': $BUFFER_AFTER}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"

# Skip fine-tune if too few new samples
if [ "$NEW_SAMPLES" -lt 50 ]; then
    echo "  Too few new samples ($NEW_SAMPLES), skipping fine-tune"
    echo "================================================================"
    echo " Nightly complete (capture only) — $(date)"
    echo "================================================================"
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. Fine-tune on FULL buffer
# ---------------------------------------------------------------------------
echo ""
echo "--- FINE-TUNE PHASE ---"

# Determine next cycle number
CYCLE_NUM=1
for d in "$PROJ"/checkpoints_v8_online_iq3_*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d" | sed 's/checkpoints_v8_online_iq3_c//')
    if [ "$n" -ge "$CYCLE_NUM" ] 2>/dev/null; then
        CYCLE_NUM=$((n + 1))
    fi
done
OUTPUT="$PROJ/checkpoints_v8_online_iq3_c${CYCLE_NUM}"
mkdir -p "$OUTPUT"
echo "  Output: $OUTPUT (cycle $CYCLE_NUM)"

python3 scripts/online_finetune.py \
    --checkpoint "$CHECKPOINT" \
    --buffer-dir "$BUFFER" \
    --output-dir "$OUTPUT" \
    --epochs 2 \
    --lr 8e-5 \
    --batch-size 4 \
    --grad-accum 4 \
    --bf16 \
    --no-rollback \
    --loss-type lk

# Check if new checkpoint was saved
if [ -f "$OUTPUT/best.pt" ]; then
    echo "  New checkpoint saved: $OUTPUT/best.pt"
    NEW_CHECKPOINT="$OUTPUT/best.pt"
else
    echo "  No improvement — keeping $CHECKPOINT"
    NEW_CHECKPOINT="$CHECKPOINT"
fi

# ---------------------------------------------------------------------------
# 5. Benchmark τ on eval holdout
# ---------------------------------------------------------------------------
echo ""
echo "--- BENCHMARK PHASE ---"

python3 scripts/benchmark_tau_v8.py \
    --checkpoint "$NEW_CHECKPOINT" \
    --features-dir "$PROJ/data/eval_features_iq3" \
    --n-samples 100 \
    2>/dev/null || echo "  (benchmark skipped — eval features not available)"

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " DFlash Nightly Complete — $(date)"
echo "================================================================"
echo "  Cycle:       $CYCLE_NUM"
echo "  Buffer:      $BUFFER_AFTER samples"
echo "  New samples: $NEW_SAMPLES"
echo "  Checkpoint:  $NEW_CHECKPOINT"
echo "  Log:         $LOGFILE"
echo "================================================================"

# IQ3 service restart handled by EXIT trap
