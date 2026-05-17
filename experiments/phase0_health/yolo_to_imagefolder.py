"""
Convert a YOLO-detection-format dataset to an ImageFolder layout for classification.

Assumes the YOLO dataset has:
    <src>/<split>/images/<stem>.<ext>
    <src>/<split>/labels/<stem>.txt   (one bbox per line: "class_id cx cy w h")

For datasets where each image has exactly one object, the first label line's class_id
is used as the image's class label. Multi-object images use the FIRST object's class
(silent — if your data has multi-class images, this is wrong).

Output (symlinks, original files untouched):
    <dst>/<split>/class_<id>/<stem>.<ext>
"""
import argparse
import os
from collections import Counter
from pathlib import Path

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def convert(src: Path, dst: Path, splits=('train', 'val', 'test')):
    stats = {}
    for split in splits:
        img_dir = src / split / 'images'
        lbl_dir = src / split / 'labels'
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        counts = Counter()
        skipped = 0
        multi_obj = 0
        for lbl_path in lbl_dir.glob('*.txt'):
            try:
                with lbl_path.open() as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if not lines:
                    skipped += 1
                    continue
                if len(lines) > 1:
                    multi_obj += 1
                class_id = int(lines[0].split()[0])
            except (ValueError, IndexError):
                skipped += 1
                continue

            stem = lbl_path.stem
            img_path = next((img_dir / f'{stem}{ext}' for ext in IMG_EXTS
                             if (img_dir / f'{stem}{ext}').exists()), None)
            if img_path is None:
                skipped += 1
                continue

            dst_class_dir = dst / split / f'class_{class_id}'
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            link = dst_class_dir / img_path.name
            if not link.exists() and not link.is_symlink():
                os.symlink(img_path.resolve(), link)
            counts[class_id] += 1
        stats[split] = {'total': sum(counts.values()), 'per_class': dict(sorted(counts.items())),
                        'skipped': skipped, 'multi_object': multi_obj}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='YOLO dataset root (containing train/val with images/ + labels/)')
    ap.add_argument('--dst', required=True, help='Output ImageFolder root')
    args = ap.parse_args()
    stats = convert(Path(args.src).resolve(), Path(args.dst).resolve())
    for split, s in stats.items():
        print(f'{split}: total={s["total"]} skipped={s["skipped"]} '
              f'multi_object={s["multi_object"]} per_class={s["per_class"]}')
    print(f'\nDone. ImageFolder dataset at: {args.dst}')


if __name__ == '__main__':
    main()
