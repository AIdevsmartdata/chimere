#!/bin/bash
# Extract holdout in full-seq mode + benchmark τ on v8 checkpoint
# Run this after training completes (GPU must be free)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL="$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
HOLDOUT_JSONL="data/holdout_v8_500.jsonl"
HOLDOUT_FEATURES="data/features_holdout_fullseq"
CHECKPOINT="checkpoints_v8_10k/best.pt"

echo "============================================================"
echo " Step 1: Extract holdout (full-seq, 500 samples)"
echo "============================================================"

./extract/build/extract_single_position \
  -m "$MODEL" \
  --layers 1,10,19,28,37 \
  --input "$HOLDOUT_JSONL" \
  --output "$HOLDOUT_FEATURES" \
  --extract-all --max-seq-len 512 \
  -ngl 40 -ot "blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU" \
  --flash-attn on -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q4_0

echo ""
echo "============================================================"
echo " Step 2: Benchmark τ on holdout"
echo "============================================================"

python scripts/benchmark_tau_v8.py \
  --checkpoint "$CHECKPOINT" \
  --features-dir "$HOLDOUT_FEATURES" \
  --n-samples 500 --anchors-per-sample 10

echo ""
echo "============================================================"
echo " Step 3: Restart qwen35-llama"
echo "============================================================"
systemctl --user start qwen35-llama
echo "qwen35-llama restarted"
