# Embedding Separability Summary (Cosine Similarity)

- Source checkpoint: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_lstm_fullscale_candidate_v1_1773125770/lstm_model_state_dict.pth`
- Source config: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/supervised_lstm_fullscale_candidate_v1_1773125770/config.json`
- Device used for embedding extraction: `cpu`

| Pair Type | Mean Cosine | Std Cosine | 25th Percentile | Median | 75th Percentile |
|---|---:|---:|---:|---:|---:|
| same action | 0.8100 | 0.1534 | 0.7040 | 0.8318 | 0.9561 |
| different action | 0.7941 | 0.1532 | 0.6964 | 0.8061 | 0.9304 |

Figure: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/reports/figures/embedding_separability_cosine_hist.svg`