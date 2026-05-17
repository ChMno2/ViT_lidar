#!/usr/bin/env bash
# Master orchestrator — runs Phase 4 -> 2 -> 3 sequentially on a single GPU.
# Suitable for nohup background execution.
#
# Usage (on workstation):
#   cd /data2/m314831001/home/ViT_lidar_repo
#   DATA_PATH=/path/to/classification_dataset \
#   PATH=~/miniconda3/envs/lab8/bin:$PATH \
#   nohup ./experiments/scripts/run_all_lidar.sh > experiments/results/master.log 2>&1 &
#
# Each phase's log lives in experiments/results/<phase>/train.log.
# This master log just records the orchestration timeline.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }
banner() { echo ""; echo "================================================================"; echo "$(ts) $*"; echo "================================================================"; }

run_phase() {
  local name=$1
  local script=$2
  banner "START $name"
  if bash "$script"; then
    banner "DONE  $name"
  else
    banner "FAIL  $name (continuing anyway)"
  fi
}

banner "MASTER RUN — Phase 4 baselines first (fastest, sanity-check), then 2, then 3"

# Phase 4 — baselines (fast comparisons)
run_phase "phase4_resnet18"        "$SCRIPT_DIR/phase4_resnet18.sh"
run_phase "phase4_mobilenetv3"     "$SCRIPT_DIR/phase4_mobilenetv3.sh"
run_phase "phase4_efficientvit_m0" "$SCRIPT_DIR/phase4_efficientvit_m0.sh"

# Phase 2 — input strategies
run_phase "phase2_2A_resize224" "$SCRIPT_DIR/phase2_2A_resize224.sh"
run_phase "phase2_2B_crop192"   "$SCRIPT_DIR/phase2_2B_crop192.sh"
run_phase "phase2_2D_gray1ch"   "$SCRIPT_DIR/phase2_2D_gray1ch.sh"
run_phase "phase2_2E_native320" "$SCRIPT_DIR/phase2_2E_native320.sh"

# Phase 3 — training strategies (do pretrained first; from_scratch is slowest)
run_phase "phase3_pretrained_head_only" "$SCRIPT_DIR/phase3_pretrained_head_only.sh"
run_phase "phase3_pretrained_finetune"  "$SCRIPT_DIR/phase3_pretrained_finetune.sh"
run_phase "phase3_from_scratch"         "$SCRIPT_DIR/phase3_from_scratch.sh"

banner "MASTER RUN COMPLETE"
