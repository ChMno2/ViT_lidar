#!/usr/bin/env bash
# ImageNet-1K — FROM-SCRATCH TRAINING. Reproduces the paper's training recipe.
# Locked to MicroViT-S1 (microvit_1), the smallest variant — 6.4M params, 231M FLOPs.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!  RESOURCE WARNING                                                       !!
# !!  - Dataset:    ImageNet-1K (~150 GB on disk)                             !!
# !!  - Compute:    4x A100/3090/4090 recommended (override NUM_GPU)          !!
# !!  - Wall time:  ~3-5 days on 4x A100 for 300 epochs (S1 is fastest)       !!
# !!  - Memory:     ~30 GB GPU VRAM total at batch 2048                       !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
set -e
# Force smallest model regardless of caller's $MODEL env.
export MODEL=microvit_1
source "$(dirname "$0")/_common.sh"

: "${IMAGENET_PATH:?Set IMAGENET_PATH to your ImageNet-1K root.}"
: "${NUM_GPU:=4}"
: "${ALL_BATCH_SIZE:=2048}"
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU

OUT="$OUTPUT_BASE/imagenet1k_train_${MODEL}"
mkdir -p "$OUT"
cd "$MICROVIT_DIR"

# Mirrors train.sh from the upstream repo. WandB can be enabled with --enable_wandb.
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  torchrun --nproc_per_node="$NUM_GPU" --master_port 12345 \
  main.py \
  --data-set IMNET \
  --data-path "$IMAGENET_PATH" \
  --model "$MODEL" \
  --in-chans 3 \
  --input-size 224 \
  --batch-size "$BATCH_SIZE" \
  --epochs 300 \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --weight-decay 0.032 \
  --aa rand-m9-mstd0.5-inc1 \
  --dist-eval \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"
