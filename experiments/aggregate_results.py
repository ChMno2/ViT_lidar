"""
Aggregate train.log files from experiments/results/<phase>/ into one markdown table.
Run after run_all_lidar.sh finishes.

Usage:
    python experiments/aggregate_results.py
    python experiments/aggregate_results.py --results-dir experiments/results --out summary.md
"""
import argparse
import re
from pathlib import Path


PHASES_ORDER = [
    'phase1_smoke',
    'phase4_resnet18', 'phase4_mobilenetv3', 'phase4_efficientvit_m0',
    'phase2_2A_resize224', 'phase2_2B_crop192', 'phase2_2D_gray1ch', 'phase2_2E_native320',
    'phase3_pretrained_head_only', 'phase3_pretrained_finetune', 'phase3_from_scratch',
]


def parse_log(log_path: Path) -> dict:
    info = {'name': log_path.parent.name, 'best_acc1': None, 'best_acc5': None,
            'final_acc1': None, 'params_m': None, 'macs_g': None, 'epochs_run': 0,
            'model': None, 'wall_time': None}
    if not log_path.exists():
        info['status'] = 'NO LOG'
        return info

    text = log_path.read_text(errors='ignore')

    m = re.search(r'(\S+),\s*params:\s*([\d.]+)\s*M,\s*macs:\s*([\d.]+)\s*G', text)
    if m:
        info['model'] = m.group(1)
        info['params_m'] = float(m.group(2))
        info['macs_g'] = float(m.group(3))

    accs = [float(x) for x in re.findall(r'\*\s*Acc@1\s+([\d.]+)', text)]
    accs5 = [float(x) for x in re.findall(r'\*\s*Acc@5\s+([\d.]+)', text)]
    if accs:
        info['best_acc1'] = max(accs)
        info['final_acc1'] = accs[-1]
        info['epochs_run'] = len(accs)
    if accs5:
        info['best_acc5'] = max(accs5)

    m = re.search(r'Training time\s+(\S+)', text)
    if m:
        info['wall_time'] = m.group(1)

    info['status'] = 'DONE' if info['best_acc1'] is not None else 'RUNNING/FAILED'
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='experiments/results')
    ap.add_argument('--out', default='experiments/results/SUMMARY.md')
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    phase_dirs = [p for p in results_dir.iterdir() if p.is_dir() and p.name != 'phase0']

    ordered = []
    seen = set()
    for name in PHASES_ORDER:
        d = results_dir / name
        if d.is_dir():
            ordered.append(d); seen.add(d.name)
    for d in sorted(phase_dirs):
        if d.name not in seen:
            ordered.append(d)

    rows = [parse_log(d / 'train.log') for d in ordered]

    out = Path(args.out).resolve()
    with out.open('w') as f:
        f.write('# Experiment Summary\n\n')
        f.write(f'Aggregated from `{results_dir}`.\n\n')
        f.write('| Phase | Model | Params (M) | MACs (G) | Epochs | Best Acc@1 | Final Acc@1 | Best Acc@5 | Status |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|---:|---|\n')
        for r in rows:
            f.write(f"| {r['name']} | {r['model'] or '-'} "
                    f"| {r['params_m'] or '-'} | {r['macs_g'] or '-'} "
                    f"| {r['epochs_run']} "
                    f"| {r['best_acc1'] if r['best_acc1'] is not None else '-'} "
                    f"| {r['final_acc1'] if r['final_acc1'] is not None else '-'} "
                    f"| {r['best_acc5'] if r['best_acc5'] is not None else '-'} "
                    f"| {r['status']} |\n")

        f.write('\n## Reading the table\n\n')
        f.write('- **Random baseline** for 5-class classification: 20% Acc@1\n')
        f.write('- **Phase 4** = non-ViT baselines (CNN floor)\n')
        f.write('- **Phase 2** = MicroViT-S1 with different input strategies (same training recipe)\n')
        f.write('- **Phase 3** = MicroViT-S1 with different init strategies (best Phase-2 config recommended)\n')
        f.write('- Acc@5 saturates at 100% on a 5-class task (every prediction has 5 logits, all classes appear in top-5)\n')

    print(f'Wrote {out}')
    for r in rows:
        print(f"  {r['name']:35s}  best_acc1={r['best_acc1']!s:6}  params={r['params_m']!s:5}M  status={r['status']}")


if __name__ == '__main__':
    main()
