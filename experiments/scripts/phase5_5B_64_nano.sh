#!/usr/bin/env bash
# Phase 5B — 64x64, patch=4, halved dims/depths, 1-channel.
# Tests: input compression + moderate architecture shrink (~1.5M params target).
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase5_5B_64_nano"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model microvit_nano_64 \
  --in-chans 1 \
  --input-size 64 \
  --batch-size 128 \
  --epochs 100 \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --warmup-epochs 3 \
  --no-repeated-aug \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
