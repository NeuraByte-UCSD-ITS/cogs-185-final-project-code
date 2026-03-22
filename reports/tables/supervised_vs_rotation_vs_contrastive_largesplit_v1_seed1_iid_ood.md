# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_largesplit_v1_seed1_1772601445`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.3342 | 0.3524 | 0.0800 | 0.3351 | 0.0200 |
| Rotation-SSL + Fine-tune | 0.3206 | 0.3091 | 0.1100 | 0.2961 | 0.0750 |
| Contrastive-lite + Fine-tune | 0.3044 | 0.3065 | 0.0500 | 0.3043 | 0.0650 |
