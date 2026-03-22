# Multi-Goal Demo Seed Sensitivity (Hybrid Policy)

Source log: `experiments/runs/terminal-outputs/demo-runs-multi-different-seeds.txt`

| Run | Seed | Max steps | Steps taken | Completed objective |
|---:|---:|---:|---:|---|
| 1 | 124 | 200 | 200 | no |
| 2 | 124 | 300 | 300 | no |
| 3 | 124 | 300 | 300 | no |
| 4 | 125 | 200 | 200 | no |
| 5 | 125 | 500 | 238 | yes |

Interpretation:
- Multi-goal hard OOD behavior is seed-sensitive even with `--policy-mode hybrid`.
- Increasing horizon (`max_steps`) can allow eventual completion in some seeds.
- This is expected for harder settings where policy + planner-rescue still faces obstacle-layout variability.
