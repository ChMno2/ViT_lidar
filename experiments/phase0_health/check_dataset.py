"""
Phase 0 — LiDAR dataset health check.

Standalone script (no MicroViT dependency, only torchvision + PIL + matplotlib).
Inspects an ImageFolder-layout dataset and writes a report.

Expected layout:
    <data_root>/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png   (optional)

Or flat (no split yet):
    <data_root>/<class_name>/*.png    --> use --auto-split

Usage:
    python check_dataset.py --data-root /path/to/lidar_320x192_grayscale_5class
    python check_dataset.py --data-root /path/to/lidar_320x192_grayscale_5class --auto-split

Writes report to <out-dir>/phase0_report.md and a montage png per class.
"""
import argparse
import hashlib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def find_images(root: Path):
    """Return list of (path, class_name) tuples; supports split or flat layout."""
    samples = []
    splits_found = {d.name for d in root.iterdir() if d.is_dir() and d.name in {'train', 'val', 'test'}}
    if splits_found:
        for split in splits_found:
            split_dir = root / split
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for f in cls_dir.rglob('*'):
                    if f.suffix.lower() in IMG_EXTS:
                        samples.append((f, cls_dir.name, split))
    else:
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in cls_dir.rglob('*'):
                if f.suffix.lower() in IMG_EXTS:
                    samples.append((f, cls_dir.name, 'all'))
    return samples


def compute_stats(samples, n_sample=500):
    """Compute pixel mean/std over a random subset (n_sample images)."""
    rng = random.Random(42)
    subset = rng.sample(samples, min(n_sample, len(samples)))
    pixels = []
    sizes = Counter()
    modes = Counter()
    for path, _, _ in subset:
        with Image.open(path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            arr = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
            pixels.append(arr.flatten())
    flat = np.concatenate(pixels)
    return {
        'mean': float(flat.mean()),
        'std': float(flat.std()),
        'min': float(flat.min()),
        'max': float(flat.max()),
        'sizes': sizes.most_common(5),
        'modes': modes.most_common(),
        'sample_n': len(subset),
    }


def hash_image(path: Path, size=(32, 32)):
    """Cheap perceptual hash for near-duplicate detection."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L').resize(size), dtype=np.uint8)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


def detect_dupes(samples, max_check=2000):
    """Detect near-duplicate images via average-hash."""
    rng = random.Random(0)
    subset = rng.sample(samples, min(max_check, len(samples)))
    by_hash = defaultdict(list)
    for path, cls, split in subset:
        try:
            h = hash_image(path)
            by_hash[h].append((path, cls, split))
        except Exception:
            pass
    dupes = [v for v in by_hash.values() if len(v) > 1]
    return dupes


def save_montage(samples, out_path: Path, per_class=5):
    """Save a per-class image grid for visual inspection."""
    if not HAS_MPL:
        return False
    by_cls = defaultdict(list)
    for path, cls, _ in samples:
        by_cls[cls].append(path)
    classes = sorted(by_cls.keys())
    rng = random.Random(123)
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(per_class * 2, len(classes) * 2))
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    for i, cls in enumerate(classes):
        picks = rng.sample(by_cls[cls], min(per_class, len(by_cls[cls])))
        for j in range(per_class):
            ax = axes[i, j]
            ax.axis('off')
            if j < len(picks):
                with Image.open(picks[j]) as im:
                    ax.imshow(np.asarray(im.convert('L')), cmap='gray', vmin=0, vmax=255)
                if j == 0:
                    ax.set_ylabel(cls, fontsize=10, rotation=0, ha='right', va='center')
    fig.suptitle('Per-class samples (grayscale)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return True


def auto_split(samples, out_root: Path, ratios=(0.7, 0.15, 0.15), seed=42):
    """Stratified split into train/val/test directories using symlinks."""
    import shutil
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for path, cls, _ in samples:
        by_cls[cls].append(path)

    counts = {'train': 0, 'val': 0, 'test': 0}
    for cls, paths in by_cls.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        buckets = {
            'train': paths[:n_train],
            'val': paths[n_train:n_train + n_val],
            'test': paths[n_train + n_val:],
        }
        for split, files in buckets.items():
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = dst_dir / src.name
                if dst.exists() or dst.is_symlink():
                    continue
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    shutil.copy2(src, dst)
                counts[split] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Path to dataset root')
    ap.add_argument('--out-dir', default='experiments/results/phase0',
                    help='Where to write report and montage')
    ap.add_argument('--auto-split', action='store_true',
                    help='If dataset is flat (no train/val/test dirs), create split via symlinks')
    ap.add_argument('--split-out', default=None,
                    help='Where to write split (default: <data-root>_split)')
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[phase0] scanning {root} ...')
    samples = find_images(root)
    if not samples:
        raise SystemExit(f'No images found under {root}')

    # Per-class & per-split counts
    cls_counts = Counter(s[1] for s in samples)
    split_counts = Counter(s[2] for s in samples)
    has_split = set(split_counts) != {'all'}

    print(f'[phase0] found {len(samples)} images across {len(cls_counts)} classes')
    if not has_split and args.auto_split:
        split_out = Path(args.split_out or f'{root}_split').resolve()
        print(f'[phase0] auto-splitting to {split_out} ...')
        counts = auto_split(samples, split_out)
        print(f'[phase0] split counts: {counts}')
        samples = find_images(split_out)
        cls_counts = Counter(s[1] for s in samples)
        split_counts = Counter(s[2] for s in samples)
        has_split = True

    # Stats
    print('[phase0] computing pixel stats over 500 random images ...')
    stats = compute_stats(samples)
    print(f"[phase0] pixel mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"[phase0] image sizes (top 5): {stats['sizes']}")
    print(f"[phase0] image modes: {stats['modes']}")

    # Class balance check
    counts_sorted = sorted(cls_counts.items(), key=lambda kv: -kv[1])
    max_cnt = counts_sorted[0][1]
    min_cnt = counts_sorted[-1][1]
    imbalance = max_cnt / max(min_cnt, 1)

    # Duplicates
    print('[phase0] checking near-duplicates ...')
    dupes = detect_dupes(samples)

    # Montage
    montage_path = out_dir / 'class_montage.png'
    montage_ok = save_montage(samples, montage_path)
    if montage_ok:
        print(f'[phase0] wrote montage to {montage_path}')

    # Report
    report = out_dir / 'phase0_report.md'
    with report.open('w') as f:
        f.write(f'# Phase 0 — Health Check Report\n\n')
        f.write(f'**Dataset root:** `{root}`\n\n')
        f.write(f'## Summary\n\n')
        f.write(f'- Total images: **{len(samples)}**\n')
        f.write(f'- Classes ({len(cls_counts)}): {sorted(cls_counts)}\n')
        f.write(f'- Has train/val/test split: **{has_split}**\n\n')
        f.write(f'## Per-class counts\n\n')
        f.write('| Class | Count |\n|---|---:|\n')
        for cls, n in counts_sorted:
            f.write(f'| {cls} | {n} |\n')
        f.write(f'\n**Imbalance ratio (max/min): {imbalance:.2f}**')
        f.write(f' — {"OK" if imbalance < 3 else ("borderline" if imbalance < 5 else "needs reweighting")}\n\n')
        if has_split:
            f.write(f'## Per-split counts\n\n')
            f.write('| Split | Count |\n|---|---:|\n')
            for sp, n in sorted(split_counts.items()):
                f.write(f'| {sp} | {n} |\n')
            f.write('\n')
        f.write(f'## Image properties\n\n')
        f.write(f'- Pixel mean (grayscale, [0,1]): **{stats["mean"]:.4f}**\n')
        f.write(f'- Pixel std (grayscale, [0,1]): **{stats["std"]:.4f}**\n')
        f.write(f'- Pixel min/max: {stats["min"]:.4f} / {stats["max"]:.4f}\n')
        f.write(f'- Sizes (top 5): {stats["sizes"]}\n')
        f.write(f'- Modes: {stats["modes"]}\n\n')
        f.write(f'### Suggested normalization for grayscale 1-channel input\n\n')
        f.write(f'```python\n')
        f.write(f'transforms.Normalize(mean=({stats["mean"]:.4f},), std=({stats["std"]:.4f},))\n')
        f.write(f'```\n\n')
        f.write(f'## Near-duplicate check\n\n')
        f.write(f'- Groups of suspected near-duplicates: **{len(dupes)}**\n')
        if dupes:
            f.write(f'\nFirst 5 groups:\n\n')
            for grp in dupes[:5]:
                f.write(f'- ({len(grp)} files) splits={set(g[2] for g in grp)}, classes={set(g[1] for g in grp)}\n')
                for p, c, s in grp[:3]:
                    f.write(f'  - `{p}` (class={c}, split={s})\n')
            warn = any(len(set(g[2] for g in grp)) > 1 for grp in dupes)
            if warn:
                f.write(f'\n**WARNING: some duplicate groups span multiple splits (val leakage)!**\n')
        f.write(f'\n## Decisions to make\n\n')
        f.write(f'- [ ] Class imbalance handling (current ratio: {imbalance:.2f})\n')
        f.write(f'- [ ] Confirm train/val split is fixed (seed) and leak-free\n')
        f.write(f'- [ ] Decide single-channel (Phase 2D) vs 3-channel input\n')
        f.write(f'- [ ] Decide resolution (Phase 2A vs 2E)\n')

    print(f'[phase0] wrote report to {report}')
    print(f'[phase0] DONE.')


if __name__ == '__main__':
    main()
