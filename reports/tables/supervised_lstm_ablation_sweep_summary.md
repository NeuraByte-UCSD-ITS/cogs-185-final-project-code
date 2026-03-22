# Supervised LSTM Ablation Sweep Summary

- Base config: `experiments/configs/supervised_lstm_ablation_base.json`
- Seed: `42`

| Fraction | Train Episodes | Crop | Obstacles | IID Acc | OOD Acc | IID Success | OOD Success | Elapsed (s) | Run Dir |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 600 | 9 | 1 | 0.3291 | 0.3473 | 0.0417 | 0.0833 | 437.59 | `experiments/runs/supervised_lstm_ablation_v1_frac1p00_crop9_obs1_seed42_1772608356` |
| 1.00 | 600 | 9 | 2 | 0.2584 | 0.3186 | 0.0500 | 0.0833 | 458.02 | `experiments/runs/supervised_lstm_ablation_v1_frac1p00_crop9_obs2_seed42_1772608816` |
| 0.25 | 150 | 9 | 1 | 0.1878 | 0.3138 | 0.0417 | 0.0667 | 179.01 | `experiments/runs/supervised_lstm_ablation_v1_frac0p25_crop9_obs1_seed42_1772606735` |
| 0.25 | 150 | 9 | 2 | 0.1898 | 0.3138 | 0.0417 | 0.0583 | 180.39 | `experiments/runs/supervised_lstm_ablation_v1_frac0p25_crop9_obs2_seed42_1772606917` |
| 1.00 | 600 | 5 | 1 | 0.3242 | 0.2697 | 0.2083 | 0.1417 | 137.62 | `experiments/runs/supervised_lstm_ablation_v1_frac1p00_crop5_obs1_seed42_1772607770` |
| 1.00 | 600 | 5 | 2 | 0.2552 | 0.2661 | 0.0583 | 0.0750 | 144.26 | `experiments/runs/supervised_lstm_ablation_v1_frac1p00_crop5_obs2_seed42_1772607916` |
| 0.50 | 300 | 5 | 1 | 0.3579 | 0.2613 | 0.1083 | 0.0833 | 85.20 | `experiments/runs/supervised_lstm_ablation_v1_frac0p50_crop5_obs1_seed42_1772607004` |
| 0.50 | 300 | 5 | 2 | 0.3509 | 0.2613 | 0.0917 | 0.0667 | 87.83 | `experiments/runs/supervised_lstm_ablation_v1_frac0p50_crop5_obs2_seed42_1772607094` |
| 0.50 | 300 | 9 | 2 | 0.3333 | 0.2542 | 0.0750 | 0.0833 | 269.74 | `experiments/runs/supervised_lstm_ablation_v1_frac0p50_crop9_obs2_seed42_1772607631` |
| 0.50 | 300 | 9 | 1 | 0.3419 | 0.2518 | 0.1583 | 0.1000 | 263.57 | `experiments/runs/supervised_lstm_ablation_v1_frac0p50_crop9_obs1_seed42_1772607359` |
| 0.25 | 150 | 5 | 2 | 0.2265 | 0.2375 | 0.0667 | 0.0583 | 58.76 | `experiments/runs/supervised_lstm_ablation_v1_frac0p25_crop5_obs2_seed42_1772606555` |
| 0.25 | 150 | 5 | 1 | 0.2199 | 0.2327 | 0.0750 | 0.0583 | 58.05 | `experiments/runs/supervised_lstm_ablation_v1_frac0p25_crop5_obs1_seed42_1772606494` |