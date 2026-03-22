# Proposal Hyperparameter Sweep Summary

- Base config: `experiments/configs/supervised_lstm_ablation_base.json`
- Seed: `42`

| Axis | Value | IID Acc | OOD Acc | IID Success | OOD Success | Elapsed (s) | Run Dir |
|---|---|---:|---:|---:|---:|---:|---|
| optimizer | adam | 0.3242 | 0.2697 | 0.2083 | 0.1417 | 134.46 | `experiments/runs/proposal_hyper_sweep_v1_optimizer_adam_seed42_1772612362` |
| optimizer | sgd | 0.1878 | 0.3138 | 0.0417 | 0.0667 | 138.55 | `experiments/runs/proposal_hyper_sweep_v1_optimizer_sgd_seed42_1772612503` |
| conv_depth | 3 | 0.3242 | 0.2697 | 0.2083 | 0.1417 | 134.68 | `experiments/runs/proposal_hyper_sweep_v1_conv_depth_3_seed42_1772612639` |
| conv_depth | 5 | 0.3419 | 0.2578 | 0.0833 | 0.0917 | 285.11 | `experiments/runs/proposal_hyper_sweep_v1_conv_depth_5_seed42_1772612926` |
| embedding_size | 64 | 0.3242 | 0.2697 | 0.2083 | 0.1417 | 654.45 | `experiments/runs/proposal_hyper_sweep_v1_embedding_size_64_seed42_1772613582` |
| embedding_size | 128 | 0.3788 | 0.3019 | 0.1000 | 0.1083 | 136.74 | `experiments/runs/proposal_hyper_sweep_v1_embedding_size_128_seed42_1772613721` |
| lstm_hidden_size | 64 | 0.3242 | 0.2697 | 0.2083 | 0.1417 | 135.88 | `experiments/runs/proposal_hyper_sweep_v1_lstm_hidden_size_64_seed42_1772613858` |
| lstm_hidden_size | 128 | 0.2713 | 0.2673 | 0.0667 | 0.0583 | 138.11 | `experiments/runs/proposal_hyper_sweep_v1_lstm_hidden_size_128_seed42_1772613998` |