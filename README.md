# ViT_lidar

Master's thesis project: applying / adapting **MicroViT** (Vision Transformer with low-complexity self-attention) to LiDAR data.

## Baseline

- Paper: [MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device (arXiv:2502.05800)](https://arxiv.org/abs/2502.05800)
- Upstream code: <https://github.com/novendrastywn/MicroViT>
- Local copy: [`MicroViT/`](MicroViT/)

## Layout

```
ViT_lidar/
├── MicroViT/          # cloned upstream baseline (modify here for LiDAR experiments)
├── 2502.05800v1.pdf   # reference paper
└── .vscode/           # local editor / SFTP config (sftp.json is gitignored)
```

## Notes

- `.vscode/sftp.json` contains lab server credentials and is excluded via `.gitignore` — never commit it.
