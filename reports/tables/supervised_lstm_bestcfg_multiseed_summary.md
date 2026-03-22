# Supervised LSTM Best-Config Multi-Seed Summary

- Base config: `experiments/configs/supervised_lstm_ablation_base.json`
- Fraction: `1.0`
- Crop size: `9`
- Obstacles: `1`

| Seed | IID Acc | OOD Acc | IID Success | OOD Success | Run Dir |
|---:|---:|---:|---:|---:|---|
| 11 | 0.2902 | 0.3049 | 0.1917 | 0.1000 | `experiments/runs/supervised_lstm_bestcfg_multiseed_v1_seed11_frac1p00_crop9_obs1_1772609340` |
| 22 | 0.2830 | 0.2837 | 0.0583 | 0.0667 | `experiments/runs/supervised_lstm_bestcfg_multiseed_v1_seed22_frac1p00_crop9_obs1_1772609888` |
| 33 | 0.2767 | 0.2969 | 0.0667 | 0.0333 | `experiments/runs/supervised_lstm_bestcfg_multiseed_v1_seed33_frac1p00_crop9_obs1_1772610564` |

- Mean IID acc: `0.2833`
- Mean OOD acc: `0.2952`