#!/bin/bash
# IQ3 adaptation loop: capture → fine-tune → repeat
# Each cycle captures with the latest drafter, then fine-tunes on new data
set -e

MODEL=~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_S.gguf
PROMPTS_A=data/bootstrap_prompts.jsonl      # 393 code-heavy
PROMPTS_B=data/diverse_prompts_v2.jsonl     # 178 diverse
DAEMON=extract/build/target_daemon
MAX_CYCLES=5
START_CYCLE=3  # c1 and c2 already done

CHECKPOINT=checkpoints_v8_online_iq3_c2/best.pt

for cycle in $(seq $START_CYCLE $((START_CYCLE + MAX_CYCLES - 1))); do
    echo ""
    echo "================================================================"
    echo " IQ3 CYCLE $cycle — $(date)"
    echo "================================================================"

    BUFFER=data/online_buffer_iq3_c${cycle}
    OUTPUT=checkpoints_v8_online_iq3_c${cycle}
    mkdir -p "$BUFFER" "$OUTPUT"

    # Alternate prompt sets for diversity
    if (( cycle % 2 == 1 )); then
        PROMPTS=$PROMPTS_A
    else
        PROMPTS=$PROMPTS_B
    fi

    echo "[cycle $cycle] Capturing with $PROMPTS → $BUFFER"
    python3 scripts/bulk_capture_v2.py \
        --checkpoint "$CHECKPOINT" \
        --model "$MODEL" \
        --prompts "$PROMPTS" \
        --buffer-dir "$BUFFER" \
        --drafter-device cpu \
        --full-gpu \
        --max-tokens 64

    N_SAMPLES=$(ls "$BUFFER" | grep -c sample || echo 0)
    echo "[cycle $cycle] Captured $N_SAMPLES samples"

    if [ "$N_SAMPLES" -lt 50 ]; then
        echo "[cycle $cycle] Too few samples, skipping fine-tune"
        continue
    fi

    echo "[cycle $cycle] Fine-tuning → $OUTPUT"
    python3 scripts/online_finetune.py \
        --checkpoint "$CHECKPOINT" \
        --buffer-dir "$BUFFER" \
        --output-dir "$OUTPUT" \
        --epochs 2 --lr 1e-4 --batch-size 4 --grad-accum 2 \
        --bf16

    CHECKPOINT="$OUTPUT/best.pt"
    echo "[cycle $cycle] Done — new checkpoint: $CHECKPOINT"
done

echo ""
echo "================================================================"
echo " IQ3 TRAINING LOOP COMPLETE"
echo "================================================================"
echo "Final checkpoint: $CHECKPOINT"
