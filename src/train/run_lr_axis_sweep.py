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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    with open(repository_root / args.base_config, "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    generated_dir = repository_root / "experiments" / "configs" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = [1e-3, 5e-4]
    rows = []
    for learning_rate in learning_rates:
        config = json.loads(json.dumps(base_config))
        config["seed"] = int(args.seed)
        learning_rate_tag = str(learning_rate).replace(".", "p")
        config["experiment_name"] = f"proposal_lr_axis_v1_lr{learning_rate_tag}_seed{args.seed}"
        config["train"]["learning_rate"] = float(learning_rate)

        config_path = generated_dir / f"{config['experiment_name']}.json"
        with open(config_path, "w", encoding="utf-8") as generated_file:
            json.dump(config, generated_file, indent=2)

        command = [
            "python",
            "-m",
            "src.train.train_supervised_lstm_only",
            "--config",
            str(config_path.relative_to(repository_root)),
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=repository_root)

        run_dirs = sorted((repository_root / "experiments" / "runs").glob(f"{config['experiment_name']}_*"))
        latest_run = max(run_dirs, key=_timestamp_from_path)
        with open(latest_run / "metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        rows.append(
            {
                "learning_rate": learning_rate,
                "iid_acc": float(metrics["lstm"]["iid_test_action_accuracy"]),
                "ood_acc": float(metrics["lstm"]["ood_test_action_accuracy"]),
                "iid_success": float(metrics["lstm"]["iid_test_success_rate"]),
                "ood_success": float(metrics["lstm"]["ood_test_success_rate"]),
                "run_dir": str(latest_run.relative_to(repository_root)),
            }
        )

    rows = sorted(rows, key=lambda row: row["learning_rate"], reverse=True)
    output_table = repository_root / "reports" / "tables" / "proposal_lr_axis_summary.md"
    lines = [
        "# Proposal Learning-Rate Axis Summary",
        "",
        f"- Base config: `{args.base_config}`",
        f"- Seed: `{args.seed}`",
        "",
        "| Learning Rate | IID Acc | OOD Acc | IID Success | OOD Success | Run Dir |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['learning_rate']:.4g} | {row['iid_acc']:.4f} | {row['ood_acc']:.4f} | {row['iid_success']:.4f} | {row['ood_success']:.4f} | `{row['run_dir']}` |"
        )
    output_table.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {output_table}")


if __name__ == "__main__":
    main()
