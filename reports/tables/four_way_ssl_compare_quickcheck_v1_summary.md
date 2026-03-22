# Four-way Representation Comparison (Supervised + Rotation + Contrastive + Predictive)

- Run directory: `/Users/neurobit/Downloads/Academia/cogs185/project/cogs181-final/cogs-185/experiments/runs/four_way_ssl_compare_quickcheck_v1_1772612222`
- Device: `cpu`

| Method | Linear Probe IID | Fine-tune IID | Fine-tune OOD | IID Success | OOD Success |
|---|---:|---:|---:|---:|---:|
| Supervised end-to-end | - | 0.3053 | 0.2282 | 0.0750 | 0.0750 |
| Rotation SSL + Fine-tune | 0.2067 | 0.3077 | 0.2332 | 0.1000 | 0.0875 |
| Contrastive SSL + Fine-tune | 0.2885 | 0.3053 | 0.2282 | 0.0750 | 0.1000 |
| Predictive SSL + Fine-tune | 0.2644 | 0.2933 | 0.2315 | 0.0625 | 0.0250 |
