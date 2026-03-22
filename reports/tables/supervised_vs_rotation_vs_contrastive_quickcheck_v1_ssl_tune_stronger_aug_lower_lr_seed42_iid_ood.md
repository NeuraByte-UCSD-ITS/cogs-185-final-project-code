# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_stronger_aug_lower_lr_seed42_1772604292`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.2474 | 0.3053 | 0.0750 | 0.2282 | 0.0750 |
| Rotation-SSL + Fine-tune | 0.2474 | 0.3053 | 0.0875 | 0.2282 | 0.0875 |
| Contrastive-lite + Fine-tune | 0.2526 | 0.3077 | 0.0375 | 0.2248 | 0.0875 |
