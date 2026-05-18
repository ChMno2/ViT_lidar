#!/usr/bin/env bash
# Phase 5C — 64x64, patch=4, extreme-shrunk dims/depths, 1-channel.
# Tests: full FPGA-target compression (~0.5M params target).
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase5_5C_64_micro"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model microvit_micro_64 \
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
