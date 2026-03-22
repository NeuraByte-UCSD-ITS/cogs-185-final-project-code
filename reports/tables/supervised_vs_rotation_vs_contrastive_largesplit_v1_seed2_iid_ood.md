# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_vs_rotation_vs_contrastive_largesplit_v1_seed2_1772602242`
- Device: `cpu`

| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | 0.3688 | 0.3756 | 0.1250 | 0.3353 | 0.0200 |
| Rotation-SSL + Fine-tune | 0.3385 | 0.3646 | 0.1000 | 0.3307 | 0.0500 |
| Contrastive-lite + Fine-tune | 0.3584 | 0.3581 | 0.0900 | 0.2945 | 0.0400 |
