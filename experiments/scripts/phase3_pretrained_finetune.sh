#!/usr/bin/env bash
# Phase 3B — Load ImageNet-pretrained weights (via SHViT release), fine-tune whole model.
# Uses small LR. Requires 3-channel input (pretrained stem expects RGB).
set -e
source "$(dirname "$0")/_common.sh"

if [[ -z "$PRETRAINED_URL" ]]; then
  echo "[ERROR] No pretrained URL for MODEL=$MODEL. Use microvit_1 / microvit_2 / microvit_3."
  exit 1
fi

OUT="$OUTPUT_BASE/phase3_pretrained_finetune"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 128 \
  --epochs 80 \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-4 \
  --warmup-epochs 3 \
  --weight-decay 0.025 \
  --finetune "$PRETRAINED_URL" \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
