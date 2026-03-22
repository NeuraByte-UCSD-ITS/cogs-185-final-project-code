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
    parser.add_argument("--base-config", type=str, default="experiments/configs/supervised_lstm_ablation_base.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--crop-size", type=int, default=9)
    parser.add_argument("--obstacle-count", type=int, default=1)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    base_config_path = repository_root / args.base_config
    with open(base_config_path, "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    generated_config_dir = repository_root / "experiments" / "configs" / "generated"
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    train_episode_base = int(base_config["data"]["split"]["train_episodes"])
    train_episodes = max(50, int(round(train_episode_base * args.fraction)))

    rows = []
    for seed_value in args.seeds:
        run_config = dict(base_config)
        run_config["seed"] = int(seed_value)
        run_config["experiment_name"] = (
            f"supervised_lstm_bestcfg_multiseed_v1_seed{seed_value}_frac{args.fraction:.2f}_crop{args.crop_size}_obs{args.obstacle_count}"
        ).replace(".", "p")
        run_config["data"] = dict(base_config["data"])
        run_config["data"]["split"] = {
            "train_episodes": train_episodes,
            "val_episodes": int(base_config["data"]["split"]["val_episodes"]),
            "test_episodes": int(base_config["data"]["split"]["test_episodes"]),
        }
        run_config["data"]["observation_crop_size"] = int(args.crop_size)
        run_config["data"]["obstacle_count"] = int(args.obstacle_count)
        run_config["data"]["ood"] = dict(base_config["data"]["ood"])
        run_config["data"]["ood"]["obstacle_count"] = max(2, int(args.obstacle_count) + 1)

        config_path = generated_config_dir / f"{run_config['experiment_name']}.json"
        with open(config_path, "w", encoding="utf-8") as generated_file:
            json.dump(run_config, generated_file, indent=2)

        command = ["python", "-m", "src.train.train_supervised_lstm_only", "--config", str(config_path.relative_to(repository_root))]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=repository_root)

        run_dirs = sorted((repository_root / "experiments" / "runs").glob(f"{run_config['experiment_name']}_*"))
        latest_run = max(run_dirs, key=_timestamp_from_path)
        with open(latest_run / "metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        rows.append(
            {
                "seed": seed_value,
                "iid_acc": float(metrics["lstm"]["iid_test_action_accuracy"]),
                "ood_acc": float(metrics["lstm"]["ood_test_action_accuracy"]),
                "iid_success": float(metrics["lstm"]["iid_test_success_rate"]),
                "ood_success": float(metrics["lstm"]["ood_test_success_rate"]),
                "run_dir": str(latest_run.relative_to(repository_root)),
            }
        )

    summary_path = repository_root / "reports" / "tables" / "supervised_lstm_bestcfg_multiseed_summary.md"
    rows = sorted(rows, key=lambda row: row["seed"])
    iid_mean = sum(row["iid_acc"] for row in rows) / len(rows)
    ood_mean = sum(row["ood_acc"] for row in rows) / len(rows)
    lines = [
        "# Supervised LSTM Best-Config Multi-Seed Summary",
        "",
        f"- Base config: `{args.base_config}`",
        f"- Fraction: `{args.fraction}`",
        f"- Crop size: `{args.crop_size}`",
        f"- Obstacles: `{args.obstacle_count}`",
        "",
        "| Seed | IID Acc | OOD Acc | IID Success | OOD Success | Run Dir |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['iid_acc']:.4f} | {row['ood_acc']:.4f} | {row['iid_success']:.4f} | {row['ood_success']:.4f} | `{row['run_dir']}` |"
        )
    lines.append("")
    lines.append(f"- Mean IID acc: `{iid_mean:.4f}`")
    lines.append(f"- Mean OOD acc: `{ood_mean:.4f}`")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {summary_path}")


if __name__ == "__main__":
    main()
