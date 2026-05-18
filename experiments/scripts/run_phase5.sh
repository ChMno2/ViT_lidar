#!/usr/bin/env bash
# Phase 5 orchestrator — runs all 64x64 FPGA-target variants sequentially.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }
banner() { echo ""; echo "================================================================"; echo "$(ts) $*"; echo "================================================================"; }

run_phase() {
  banner "START $1"
  if bash "$2"; then banner "DONE  $1"; else banner "FAIL  $1 (continuing)"; fi
}

banner "PHASE 5 — 64x64 FPGA-target variants"
run_phase "phase5_5A_64_full"  "$SCRIPT_DIR/phase5_5A_64_full.sh"
run_phase "phase5_5B_64_nano"  "$SCRIPT_DIR/phase5_5B_64_nano.sh"
run_phase "phase5_5C_64_micro" "$SCRIPT_DIR/phase5_5C_64_micro.sh"
banner "PHASE 5 COMPLETE"
