# Proposal Learning-Rate Axis Summary

- Base config: `experiments/configs/supervised_lstm_ablation_base.json`
- Seed: `42`

| Learning Rate | IID Acc | OOD Acc | IID Success | OOD Success | Run Dir |
|---:|---:|---:|---:|---:|---|
| 0.001 | 0.3242 | 0.2697 | 0.2083 | 0.1417 | `experiments/runs/proposal_lr_axis_v1_lr0p001_seed42_1773289455` |
| 0.0005 | 0.3307 | 0.2613 | 0.1167 | 0.0917 | `experiments/runs/proposal_lr_axis_v1_lr0p0005_seed42_1773289598` |