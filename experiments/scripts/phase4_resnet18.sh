#!/usr/bin/env bash
# Phase 4 baseline — ResNet-18. Classic CNN.
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase4_resnet18"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model resnet18 \
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
