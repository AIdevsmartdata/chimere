#!/bin/bash
# Wait for c3 loop to finish, then fine-tune on FULL IQ3 buffer (5600+ samples)
set -euo pipefail
cd ~/chimere-dflash

echo "Waiting for iq3_training_loop.sh to finish..."
while pgrep -f "iq3_training_loop.sh" > /dev/null 2>&1; do
    PROGRESS=$(grep -v "^\[daemon\]" /tmp/iq3_loop.log 2>/dev/null | grep "running_τ" | tail -1 || echo "waiting...")
    echo "  $(date +%H:%M:%S) $PROGRESS"
    sleep 30
done
echo "c3 loop done."

# Wait for GPU to be free
sleep 5
echo "GPU freed. Starting full-buffer fine-tune..."

# Find best IQ3 checkpoint
CHECKPOINT=$(ls -t checkpoints_v8_online_iq3_*/best.pt 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="checkpoints_v8_online_c4/best.pt"
fi
echo "Starting from: $CHECKPOINT"

BUFFER="data/online_buffer_iq3"
N_SAMPLES=$(ls "$BUFFER" | grep -c sample)
echo "Buffer: $N_SAMPLES samples"

OUTPUT="checkpoints_v8_online_iq3_fullbuf"
mkdir -p "$OUTPUT"

PYTHONUNBUFFERED=1 python3 scripts/online_finetune.py \
    --checkpoint "$CHECKPOINT" \
    --buffer-dir "$BUFFER" \
    --output-dir "$OUTPUT" \
    --epochs 3 \
    --lr 8e-5 \
    --batch-size 4 \
    --grad-accum 4 \
    --bf16 \
    --no-rollback

echo ""
echo "Full-buffer fine-tune complete!"
echo "Checkpoint: $OUTPUT/best.pt"
