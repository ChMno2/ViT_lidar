#!/usr/bin/env bash
# ImageNet-1K — FROM-SCRATCH TRAINING. Reproduces the paper's training recipe.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!  RESOURCE WARNING                                                       !!
# !!                                                                          !!
# !!  - Dataset:    ImageNet-1K (~150 GB on disk)                             !!
# !!  - Compute:    4x A100/3090/4090 recommended; the paper used 4 GPUs      !!
# !!  - Wall time:  ~3-7 days on 4x A100 for 300 epochs                       !!
# !!  - Memory:     ~30 GB GPU VRAM total at batch 2048                       !!
# !!                                                                          !!
# !!  Before running, STRONGLY consider:                                      !!
# !!    1. Just run imagenet1k_eval_only.sh to validate the paper number      !!
# !!       (5 minutes, no training).                                          !!
# !!    2. Train on a subset (e.g. ImageNet-100) to validate the recipe       !!
# !!       end-to-end before committing 4-GPU days.                           !!
# !!                                                                          !!
# !!  See experiments/README.md for the recommended workflow.                 !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
set -e
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
