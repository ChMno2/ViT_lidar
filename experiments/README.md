# Experiments

LiDAR-on-MicroViT experiment harness. All scripts are thin wrappers around the upstream [`MicroViT/main.py`](../MicroViT/main.py) with the LIDAR dataset branch and `--in-chans` flag added.

## Setup once

```bash
# 1. Python env
cd MicroViT && pip install -r requirements.txt && cd ..
pip install matplotlib   # for Phase 0 montage

# 2. Make all scripts executable
chmod +x experiments/scripts/*.sh
chmod +x experiments/phase0_health/*.py

# 3. Set dataset path (override every time, or export once)
export DATA_PATH=/path/to/lidar_320x192_grayscale_5class_split
```

The patches applied to upstream MicroViT:
- [MicroViT/data/datasets.py](../MicroViT/data/datasets.py) — added `LIDAR` branch and grayscale-aware transform
- [MicroViT/main.py](../MicroViT/main.py) — added `--in-chans`, `LIDAR` to `--data-set` choices, baseline-friendly `create_model` call

---

## Phase 0 — Data Health Check (do this FIRST)

Runs locally (no GPU). Inspects the dataset, builds a stratified train/val/test split, writes a report.

```bash
# If dataset is flat (no train/val/test dirs yet):
python experiments/phase0_health/check_dataset.py \
    --data-root /path/to/lidar_320x192_grayscale_5class \
    --auto-split

# If already split:
python experiments/phase0_health/check_dataset.py \
    --data-root /path/to/lidar_320x192_grayscale_5class
```

**Outputs:**
- `experiments/results/phase0/phase0_report.md` — per-class counts, pixel stats, duplicate check
- `experiments/results/phase0/class_montage.png` — visual sanity check

**Decisions you need to make from the report:**
- [ ] Class imbalance > 5×? Add `--weight-decay` adjustment or oversample.
- [ ] Near-duplicate groups spanning splits? Re-split or de-dup.
- [ ] Use the printed pixel mean/std as `Normalize` params for Phase 2D (grayscale).

---

## Phase 1 — Smoke Test (5 min)

```bash
./experiments/scripts/phase1_smoke.sh
```

Verifies: dataset loads, model forwards, loss decreases, no NaN. **Always run this before any long experiment.**

---

## Phase 2 — Input Strategy (~50 epochs each)

| Script | Resolution | Channels | Note |
|---|---|---|---|
| `phase2_2A_resize224.sh` | 224×224 (resized) | 3 | Baseline |
| `phase2_2B_crop192.sh`   | 192×192           | 3 | Tests if edge crop hurts |
| `phase2_2D_gray1ch.sh`   | 224×224           | 1 | **FPGA target** — saves stem params/FLOPs |
| `phase2_2E_native320.sh` | 320×320           | 3 | Preserves more spatial info (closest to native 320×192 without arch changes) |

```bash
for s in phase2_2A phase2_2B phase2_2D phase2_2E; do
  ./experiments/scripts/${s}_*.sh
done
```

Compare `val_acc1` across the 4 runs. Pick the best for Phase 3.

---

## Phase 3 — Training Strategy

Tells you whether ImageNet-pretrained helps for LiDAR (no obvious a-priori answer — LiDAR pixel distribution differs from natural images).

| Script | Init | Trainable | LR | Epochs |
|---|---|---|---|---|
| `phase3_from_scratch.sh`           | random | all | 1e-3 | 200 |
| `phase3_pretrained_finetune.sh`    | SHViT/ImageNet | all | 1e-4 | 80 |
| `phase3_pretrained_head_only.sh`   | SHViT/ImageNet | head only | 1e-3 | 30 |

Pretrained URLs are auto-selected by the `$MODEL` env var (microvit_1/2/3).

---

## Phase 4 — Baselines (fair comparison)

| Script | Model | Why |
|---|---|---|
| `phase4_resnet18.sh`        | resnet18              | Classic CNN floor |
| `phase4_mobilenetv3.sh`     | mobilenetv3_small_100 | Same parameter budget as MicroViT |
| `phase4_efficientvit_m0.sh` | efficientvit_m0       | Direct competitor (same paper family) |

Use **same epochs, batch size, augmentation** as your best Phase 2 config so the comparison is fair.

---

## ImageNet-1K Benchmark

### Option A — Validate the paper number (5 minutes, RECOMMENDED)

```bash
export IMAGENET_PATH=/path/to/imagenet
./experiments/scripts/imagenet1k_eval_only.sh
```

Loads the SHViT-released checkpoint (same architecture as MicroViTv1) and runs eval on ImageNet val. Should match published Top-1 ±0.2%:

| Model | Top-1 (paper) |
|---|---|
| microvit_1 | 72.6 |
| microvit_2 | 74.6 |
| microvit_3 | 77.1 |

### Option B — Train from scratch (DAYS, only if you must)

```bash
export IMAGENET_PATH=/path/to/imagenet
export NUM_GPU=4
./experiments/scripts/imagenet1k_train_from_scratch.sh
```

**Realistic cost:** 4× A100, batch 2048, 300 epochs, ~3–7 days wall time, ~150 GB disk for the dataset. Only do this if you have a reason to (modified architecture, ablation that needs a different recipe). Otherwise Option A is enough to claim "matches paper".

---

## Tweaking experiments

Override env vars without editing scripts:

```bash
MODEL=microvit_2 NUM_WORKERS=16 ./experiments/scripts/phase2_2A_resize224.sh
DATA_PATH=/other/dataset ./experiments/scripts/phase1_smoke.sh
```

See [`scripts/_common.sh`](scripts/_common.sh) for all overridable variables.

---

## Output layout

```
experiments/results/
├── phase0/            # health-check report + montage
├── phase1_smoke/      # smoke-test logs + ckpt
├── phase2_2A_resize224/
├── phase2_2B_crop192/
├── phase2_2D_gray1ch/
├── phase2_2E_native320/
├── phase3_from_scratch/
├── phase3_pretrained_finetune/
├── phase3_pretrained_head_only/
├── phase4_resnet18/
├── phase4_mobilenetv3/
├── phase4_efficientvit_m0/
├── imagenet1k_eval_microvit_1/
└── imagenet1k_train_microvit_1/
```

Each run writes `train.log` (full stdout), `checkpoint.pth`, and `log.txt` (per-epoch metrics in JSONL).

The `results/` directory is **gitignored** — pull metrics into your thesis manually.
