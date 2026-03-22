import argparse
import json
import subprocess
from pathlib import Path


def _timestamp_from_path(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=str,
        default="experiments/configs/supervised_lstm_ablation_base.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    base_config_path = repository_root / args.base_config
    with open(base_config_path, "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    train_episode_base = int(base_config["data"]["split"]["train_episodes"])
    val_episodes = int(base_config["data"]["split"]["val_episodes"])
    test_episodes = int(base_config["data"]["split"]["test_episodes"])

    generated_config_dir = repository_root / "experiments" / "configs" / "generated"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    fractions = [0.25, 0.5, 1.0]
    crop_sizes = [5, 9]
    obstacle_counts = [1, 2]

    sweep_rows = []
    for fraction in fractions:
        for crop_size in crop_sizes:
            for obstacle_count in obstacle_counts:
                train_episodes = max(50, int(round(train_episode_base * fraction)))
                run_config = dict(base_config)
                run_config["seed"] = int(args.seed)
                run_config["experiment_name"] = (
                    f"supervised_lstm_ablation_v1_frac{fraction:.2f}_crop{crop_size}_obs{obstacle_count}_seed{args.seed}"
                ).replace(".", "p")
                run_config["data"] = dict(base_config["data"])
                run_config["data"]["split"] = {
                    "train_episodes": train_episodes,
                    "val_episodes": val_episodes,
                    "test_episodes": test_episodes,
                }
                run_config["data"]["observation_crop_size"] = int(crop_size)
                run_config["data"]["obstacle_count"] = int(obstacle_count)
                run_config["data"]["ood"] = dict(base_config["data"]["ood"])
                run_config["data"]["ood"]["obstacle_count"] = max(2, int(obstacle_count) + 1)

                generated_config_path = generated_config_dir / f"{run_config['experiment_name']}.json"
                with open(generated_config_path, "w", encoding="utf-8") as generated_file:
                    json.dump(run_config, generated_file, indent=2)

                command = [
                    "python",
                    "-m",
                    "src.train.train_supervised_lstm_only",
                    "--config",
                    str(generated_config_path.relative_to(repository_root)),
                ]
                print("Running:", " ".join(command))
                subprocess.run(command, check=True, cwd=repository_root)

                run_dirs = sorted((repository_root / "experiments" / "runs").glob(f"{run_config['experiment_name']}_*"))
                if not run_dirs:
                    raise FileNotFoundError(f"No run directory found for {run_config['experiment_name']}")
                latest_run_dir = max(run_dirs, key=_timestamp_from_path)
                metrics_path = latest_run_dir / "metrics.json"
                with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                    metrics = json.load(metrics_file)

                sweep_rows.append(
                    {
                        "fraction": fraction,
                        "train_episodes": train_episodes,
                        "crop_size": crop_size,
                        "obstacle_count": obstacle_count,
                        "iid_acc": float(metrics["lstm"]["iid_test_action_accuracy"]),
                        "ood_acc": float(metrics["lstm"]["ood_test_action_accuracy"]),
                        "iid_success": float(metrics["lstm"]["iid_test_success_rate"]),
                        "ood_success": float(metrics["lstm"]["ood_test_success_rate"]),
                        "elapsed_seconds": float(metrics["elapsed_seconds"]),
                        "run_dir": str(latest_run_dir.relative_to(repository_root)),
                    }
                )

    sweep_rows.sort(key=lambda row: row["ood_acc"], reverse=True)
    summary_path = repository_root / "reports" / "tables" / "supervised_lstm_ablation_sweep_summary.md"
    lines = [
        "# Supervised LSTM Ablation Sweep Summary",
        "",
        f"- Base config: `{args.base_config}`",
        f"- Seed: `{args.seed}`",
        "",
        "| Fraction | Train Episodes | Crop | Obstacles | IID Acc | OOD Acc | IID Success | OOD Success | Elapsed (s) | Run Dir |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['fraction']:.2f} | {row['train_episodes']} | {row['crop_size']} | {row['obstacle_count']} | {row['iid_acc']:.4f} | {row['ood_acc']:.4f} | {row['iid_success']:.4f} | {row['ood_success']:.4f} | {row['elapsed_seconds']:.2f} | `{row['run_dir']}` |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {summary_path}")
    if sweep_rows:
        best = sweep_rows[0]
        print(
            "Best OOD setting:",
            f"fraction={best['fraction']:.2f}, crop={best['crop_size']}, obstacles={best['obstacle_count']}, OOD={best['ood_acc']:.4f}",
        )


if __name__ == "__main__":
    main()
