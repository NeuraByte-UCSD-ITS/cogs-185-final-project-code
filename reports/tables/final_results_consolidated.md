# Final Results Consolidated Table

## A) Core 3-way representation comparison (large split, multi-seed)

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/three_way_largesplit_multiseed_summary.md`


| Method | IID mean | IID std | OOD mean | OOD std |
|---|---:|---:|---:|---:|
| Supervised | 0.3832 | 0.0353 | 0.3308 | 0.0076 |
| Rotation SSL | 0.3357 | 0.0278 | 0.3109 | 0.0178 |
| Contrastive-lite SSL | 0.3185 | 0.0352 | 0.2914 | 0.0148 |

Figure: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/figures/three_way_iid_ood_multiseed.svg`

## B) Sequence architecture comparison (large split)

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/ff_vs_lstm_largesplit_summary.md`

| Model | Best Val Action Acc | Test Action Acc | Test Success Rate | Avg Steps to Goal |
|---|---:|---:|---:|---:|
| Feedforward | 0.2887 | 0.3043 | 0.0700 | 46.91 |
| LSTM | 0.5232 | 0.4928 | 0.3900 | 36.37 |

## C) Supervised-LSTM ablation best setting

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/supervised_lstm_ablation_sweep_summary.md` and `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/supervised_lstm_bestcfg_multiseed_summary.md`

- Best single-seed OOD from ablation sweep: fraction=1.00, crop=9, obstacles=1, OOD acc=0.3473
- Multi-seed confirmation (seed 11/22/33): mean OOD acc=0.2952

## D) Four-way representation run (adds predictive coding)

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/four_way_ssl_compare_quickcheck_v1_summary.md`

| Method | Linear Probe IID | Fine-tune IID | Fine-tune OOD | IID Success | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | - | 0.3053 | 0.2282 | 0.0750 | 0.0750 |
| Rotation SSL + Fine-tune | 0.2067 | 0.3077 | 0.2332 | 0.1000 | 0.0875 |
| Contrastive SSL + Fine-tune | 0.2885 | 0.3053 | 0.2282 | 0.0750 | 0.1000 |
| Predictive SSL + Fine-tune | 0.2644 | 0.2933 | 0.2315 | 0.0625 | 0.0250 |

## E) Proposal hyperparameter axes sweep

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/proposal_hyperparam_sweep_summary.md`

- Optimizer: SGD improved OOD acc over Adam in this sweep (OOD 0.3138 vs 0.2697).
- Embedding: 128 improved IID and OOD over 64 (IID 0.3788 vs 0.3242, OOD 0.3019 vs 0.2697).
- Conv depth: 5 did not improve OOD vs 3 in this sweep.
- LSTM hidden size: 128 did not improve OOD vs 64 in this sweep.

## F) Linear probe + compute/memory

Sources: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/four_way_linear_probe_summary.md` and `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/tables/four_way_compute_memory_summary.md`

| Method | Linear Probe Val Acc | Linear Probe IID Acc |
|---|---:|---:|
| Rotation SSL | 0.2551 | 0.2067 |
| Contrastive SSL | 0.2168 | 0.2885 |
| Predictive SSL | 0.3342 | 0.2644 |

| Method | Parameter Count | Parameter Memory (MB) |
|---|---:|---:|
| Supervised | 60740 | 0.232 |
| Rotation SSL | 61000 | 0.233 |
| Contrastive SSL | 66980 | 0.256 |
| Predictive SSL | 69060 | 0.263 |
- End-to-end elapsed seconds (whole four-way run): `255.23`

## G) Proposal-scale anchor run (10k/2k/2k)

Source: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_lstm_fullscale_candidate_v1_1773125770/metrics.json`

- Device: `mps`
- Best validation action accuracy: `0.6629`
- IID action accuracy: `0.6629`
- OOD action accuracy: `0.5848`
- IID success: `0.4005`
- OOD success: `0.2635`
- Elapsed seconds: `2286.71`