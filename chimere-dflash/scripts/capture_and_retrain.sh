#!/bin/bash
# Capture diverse prompts + retrain with anti-overfitting + benchmark
set -euo pipefail
cd ~/chimere-dflash

MODEL_IQ3=~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_S.gguf
DAEMON=extract/build/target_daemon
BUFFER=data/online_buffer_iq3
CHECKPOINT=checkpoints_v8_online_iq3_c2/best.pt  # start from c2 (pre-overfit)
OUTPUT=checkpoints_v8_online_iq3_c5

echo "================================================================"
echo " STEP 1: Capture diverse prompts into IQ3 buffer"
echo "================================================================"
BEFORE=$(ls "$BUFFER" 2>/dev/null | grep -c sample || echo 0)

python3 scripts/bulk_capture_v2.py \
  --checkpoint "$CHECKPOINT" \
  --model "$MODEL_IQ3" \
  --daemon "$DAEMON" \
  --prompts data/diverse_prompts.jsonl \
  --buffer-dir "$BUFFER" \
  --drafter-device cpu \
  --full-gpu \
  --max-tokens 64 \
  --max-prompts 140

AFTER=$(ls "$BUFFER" | grep -c sample || echo 0)
echo "  Buffer: $BEFORE → $AFTER (+$((AFTER - BEFORE)) new)"

echo ""
echo "================================================================"
echo " STEP 2: Retrain with anti-overfitting (1 epoch, low LR)"
echo "================================================================"

python3 scripts/online_finetune.py \
  --checkpoint "$CHECKPOINT" \
  --buffer-dir "$BUFFER" \
  --output-dir "$OUTPUT" \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 4 \
  --grad-accum 4 \
  --bf16 \
  --weight-decay 0.1 \
  --dropout 0.1 \
  --loss-type lk \
  --no-rollback

echo ""
echo "================================================================"
echo " STEP 3: Wall-clock benchmark"
echo "================================================================"

python3 scripts/benchmark_wallclock.py \
  --checkpoint "$OUTPUT/best.pt" \
  --model "$MODEL_IQ3" \
  --no-offload \
  --drafter-device cuda \
  --max-tokens 64 \
  --ctx-size 512 \
  --single-save

echo ""
echo "================================================================"
echo " DONE"
echo "================================================================"
