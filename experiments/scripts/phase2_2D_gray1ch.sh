#!/usr/bin/env bash
# Phase 2D — Grayscale 1-channel input. Saves ~3x stem conv params/FLOPs.
# Required for the FPGA target (less compute, lower power).
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase2_2D_gray1ch"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 1 \
  --input-size 224 \
  --batch-size 128 \
  --epochs 50 \
  --num_workers "$NUM_WORKERS" \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --no-repeated-aug \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
