# SSL Tuning Sweep Summary

- Base config: `experiments/configs/supervised_vs_rotation_vs_contrastive_quickcheck.json`
- Seed: `42`

| Config | Best SSL Method | Supervised OOD | Rotation OOD | Contrastive OOD | Best SSL OOD | Delta (Best SSL - Supervised) | Run Dir |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | rotation_ssl_fine_tune | 0.2282 | 0.2282 | 0.2265 | 0.2282 | 0.0000 | `experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_baseline_seed42_1772603487` |
| stronger_aug_lower_lr | rotation_ssl_fine_tune | 0.2282 | 0.2282 | 0.2248 | 0.2282 | 0.0000 | `experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_stronger_aug_lower_lr_seed42_1772604292` |
| stronger_aug | rotation_ssl_fine_tune | 0.2282 | 0.2265 | 0.2198 | 0.2265 | -0.0017 | `experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_stronger_aug_seed42_1772604024` |
| longer_pretrain | contrastive_ssl_fine_tune | 0.2282 | 0.2181 | 0.2232 | 0.2232 | -0.0050 | `experiments/runs/supervised_vs_rotation_vs_contrastive_quickcheck_v1_ssl_tune_longer_pretrain_seed42_1772603758` |