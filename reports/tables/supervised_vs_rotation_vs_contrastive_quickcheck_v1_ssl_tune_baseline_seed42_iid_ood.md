# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_baseline_seed42_1772603487`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.2474 | 0.3053 | 0.0750 | 0.2282 | 0.0750 |
| Rotation-SSL + Fine-tune | 0.2474 | 0.3053 | 0.0875 | 0.2282 | 0.0875 |
| Contrastive-lite + Fine-tune | 0.2551 | 0.3029 | 0.0750 | 0.2265 | 0.1000 |
