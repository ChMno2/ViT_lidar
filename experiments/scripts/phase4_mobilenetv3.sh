#!/usr/bin/env bash
# Phase 4 baseline — MobileNetV3-Small. Lightweight CNN, closer to MicroViT's parameter budget.
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase4_mobilenetv3"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model mobilenetv3_small_100 \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 128 \
  --epochs 50 \
  --num_workers "$NUM_WORKERS" \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --no-repeated-aug \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
