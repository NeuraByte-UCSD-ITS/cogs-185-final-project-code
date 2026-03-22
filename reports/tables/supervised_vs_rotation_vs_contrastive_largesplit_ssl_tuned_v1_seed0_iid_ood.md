# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_largesplit_ssl_tuned_v1_seed0_1772605529`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.3682 | 0.4218 | 0.0600 | 0.3221 | 0.0600 |
| Rotation-SSL + Fine-tune | 0.3164 | 0.2628 | 0.0850 | 0.3013 | 0.1000 |
| Contrastive-lite + Fine-tune | 0.3691 | 0.3367 | 0.0400 | 0.3110 | 0.0300 |
