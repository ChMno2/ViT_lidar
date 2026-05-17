#!/usr/bin/env bash
# Phase 2E — Native-ish resolution. 320x320 input (preserves more spatial info than 224).
# Truly native 320x192 needs non-square patch-grid changes; 320x320 is the closest
# drop-in that doesn't require architecture surgery. Patch size 16 -> 20x20 token grid.
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase2_2E_native320"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 320 \
  --batch-size 64 \
  --epochs 50 \
  --num_workers "$NUM_WORKERS" \
  --lr 5e-4 \
  --warmup-epochs 3 \
  --no-repeated-aug \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
