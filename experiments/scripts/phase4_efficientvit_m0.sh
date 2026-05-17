#!/usr/bin/env bash
# Phase 4 baseline — EfficientViT-M0 (Microsoft Cream EfficientViT). Closest competitor to MicroViT.
# Model name in timm: efficientvit_m0  (timm >= 0.9)
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase4_efficientvit_m0"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model efficientvit_m0 \
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
