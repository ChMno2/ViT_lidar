#!/usr/bin/env bash
# ImageNet-1K — EVAL ONLY (reproduce the paper's Top-1 number using released weights).
# Runtime: ~5 minutes on a single GPU. No training, no GPU days burned.
#
# What this does:
#   1. Downloads the SHViT-released checkpoint (same arch as MicroViTv1).
#   2. Runs evaluation on ImageNet-1K validation set.
#   3. Reports Top-1 / Top-5 accuracy.
#
# Expected results (from paper Table):
#   microvit_1 -> Top-1 = 72.6%
#   microvit_2 -> Top-1 = 74.6%
#   microvit_3 -> Top-1 = 77.1%
#
# Prerequisites:
#   - ImageNet-1K validation set at $IMAGENET_PATH/val/<wnid>/*.JPEG (standard layout)
#   - Set: export IMAGENET_PATH=/path/to/imagenet
set -e
source "$(dirname "$0")/_common.sh"

: "${IMAGENET_PATH:?Set IMAGENET_PATH to your ImageNet-1K root (containing train/ val/).}"

if [[ -z "$PRETRAINED_URL" ]]; then
  echo "[ERROR] No pretrained URL for MODEL=$MODEL."
  exit 1
fi

OUT="$OUTPUT_BASE/imagenet1k_eval_$MODEL"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

python main.py \
  --eval \
  --data-set IMNET \
  --data-path "$IMAGENET_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size 256 \
  --num_workers "$NUM_WORKERS" \
  --finetune "$PRETRAINED_URL" \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"

echo ""
echo "[imagenet1k_eval] Compare reported Top-1 against paper Table:"
echo "                  microvit_1 -> 72.6 / microvit_2 -> 74.6 / microvit_3 -> 77.1"
