#!/usr/bin/env bash
# Phase 3C — Pretrained backbone frozen, train classifier head only.
# Fast & robust when LiDAR dataset is small.
# NOTE: this script sets --set_bn_eval to freeze BN running stats.
set -e
source "$(dirname "$0")/_common.sh"

if [[ -z "$PRETRAINED_URL" ]]; then
  echo "[ERROR] No pretrained URL for MODEL=$MODEL."
  exit 1
fi

OUT="$OUTPUT_BASE/phase3_pretrained_head_only"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

# Higher LR since only the head trains.
python main.py \
  --data-set LIDAR \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 256 \
  --epochs 30 \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --warmup-epochs 1 \
  --weight-decay 0.0 \
  --finetune "$PRETRAINED_URL" \
  --set_bn_eval \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

echo "[phase3C] NOTE: this trains the full model unless you add a freeze-backbone hook."
echo "          See experiments/scripts/_freeze_backbone_patch.md for instructions."
