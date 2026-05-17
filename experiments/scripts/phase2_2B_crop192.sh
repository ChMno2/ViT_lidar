#!/usr/bin/env bash
# Phase 2B — Center-crop to 192x192, then upscale to 224. Tests whether edge info matters.
# Implemented by setting --input-size 192 (model will receive 192x192 directly).
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase2_2B_crop192"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 192 \
  --batch-size 128 \
  --epochs 50 \
  --num_workers "$NUM_WORKERS" \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --no-repeated-aug \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
