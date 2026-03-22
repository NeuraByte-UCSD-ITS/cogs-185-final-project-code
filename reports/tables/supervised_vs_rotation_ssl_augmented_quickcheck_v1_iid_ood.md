# Supervised vs Rotation-SSL (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_ssl_augmented_quickcheck_v1_1772598498`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.2474 | 0.3053 | 0.0750 | 0.2282 | 0.0750 |
| Rotation-SSL + Linear Probe | 0.1990 | 0.2476 | 0.0625 | 0.2114 | 0.0750 |
| Rotation-SSL + Fine-tune | 0.2755 | 0.3101 | 0.0625 | 0.2232 | 0.0625 |
