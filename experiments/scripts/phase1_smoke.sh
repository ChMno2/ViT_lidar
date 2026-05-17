#!/usr/bin/env bash
# Phase 1 — Smoke test. 1 epoch, tiny batch, verify pipeline runs end-to-end.
# Expected runtime: < 5 minutes on a single GPU.
set -e
source "$(dirname "$0")/_common.sh"

OUT="$OUTPUT_BASE/phase1_smoke"
mkdir -p "$OUT"

cd "$MICROVIT_DIR"

python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 8 \
  --epochs 1 \
  --num_workers 2 \
  --lr 1e-3 \
  --warmup-epochs 0 \
  --no-repeated-aug \
  --no-model-ema \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

echo "[phase1] SMOKE TEST DONE. If loss decreased and no NaN, pipeline OK."
echo "[phase1] Log: $OUT/train.log"
