# Single vs Multi-Goal Checkpoint Summary

| branch | run | device | IID action acc | OOD action acc | IID success | OOD success | IID goal completion | OOD goal completion | elapsed sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-goal | supervised_lstm_fullscale_candidate_v1_1773125770 | mps | 0.6629 | 0.5848 | 0.4005 | 0.2635 | N/A | N/A | 2286.71 |
| multi-goal | supervised_lstm_multigoal_candidate_v1_1773294813 | mps | 0.5123 | 0.4696 | 0.0233 | 0.0133 | 0.1778 | 0.0756 | 384.52 |

Notes:
- single-goal success is target reached rate in single-goal rollout evaluation.
- multi-goal success is full sequence completion rate (all goals reached).
- multi-goal goal-completion ratio is partial progress metric (0 to 1).
