#!/usr/bin/env bash
# Shared variables for all experiment scripts. Override via env vars.

# Where the LiDAR dataset lives (ImageFolder layout with train/val/test).
# Override:  DATA_PATH=/path/to/your/dataset ./scripts/phaseX_*.sh
: "${DATA_PATH:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lidar_320x192_grayscale_5class_split}"

# Where to write checkpoints / logs (gitignored).
: "${OUTPUT_BASE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results}"

# Path to MicroViT repo (the cloned upstream code).
: "${MICROVIT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../MicroViT" && pwd)}"

# Number of CPU workers for data loading.
: "${NUM_WORKERS:=8}"

# Number of LiDAR classes (auto if 0).
: "${NB_CLASSES:=5}"

# Default model size for LiDAR experiments (S1=smallest, S3=largest).
# microvit_1 / microvit_2 / microvit_3
: "${MODEL:=microvit_1}"

# Pretrained weight URL (SHViT-released; same arch as MicroViTv1).
case "$MODEL" in
  microvit_1) PRETRAINED_URL="https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth";;
  microvit_2) PRETRAINED_URL="https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth";;
  microvit_3) PRETRAINED_URL="https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth";;
  *) PRETRAINED_URL="";;
esac
export PRETRAINED_URL

echo "============================================================"
echo "  DATA_PATH    = $DATA_PATH"
echo "  OUTPUT_BASE  = $OUTPUT_BASE"
echo "  MICROVIT_DIR = $MICROVIT_DIR"
echo "  MODEL        = $MODEL"
echo "============================================================"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[WARN] DATA_PATH does not exist: $DATA_PATH"
  echo "       Run Phase 0 first to inspect & split the dataset:"
  echo "       python experiments/phase0_health/check_dataset.py --data-root <raw_dir> --auto-split"
fi
