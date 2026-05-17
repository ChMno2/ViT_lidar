#!/usr/bin/env bash
# Phase 3A — Train from scratch (no pretrained weights). Long training, full augmentation.
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase3_from_scratch"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 128 \
  --epochs 200 \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --warmup-epochs 5 \
  --weight-decay 0.025 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
