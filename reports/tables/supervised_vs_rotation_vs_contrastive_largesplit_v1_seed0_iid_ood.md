# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_largesplit_v1_seed0_1772600634`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.3682 | 0.4218 | 0.0600 | 0.3221 | 0.0600 |
| Rotation-SSL + Fine-tune | 0.2873 | 0.3333 | 0.0500 | 0.3058 | 0.0550 |
| Contrastive-lite + Fine-tune | 0.3391 | 0.2908 | 0.0400 | 0.2753 | 0.0300 |
