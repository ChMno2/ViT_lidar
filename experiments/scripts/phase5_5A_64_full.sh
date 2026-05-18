#!/usr/bin/env bash
# Phase 5A — 64x64 input, patch=4, same dims/depths as microvit_1, 1-channel.
# Tests: input compression alone (no architecture shrink).
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase5_5A_64_full"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model microvit_1_64 \
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
