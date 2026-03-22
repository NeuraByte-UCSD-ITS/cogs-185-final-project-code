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

    sweep_grid = [
        ("optimizer", "adam"),
        ("optimizer", "sgd"),
        ("conv_depth", 3),
        ("conv_depth", 5),
        ("embedding_size", 64),
        ("embedding_size", 128),
        ("lstm_hidden_size", 64),
        ("lstm_hidden_size", 128),
    ]

    generated_config_dir = repository_root / "experiments" / "configs" / "generated"
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for axis_name, axis_value in sweep_grid:
        run_config = json.loads(json.dumps(base_config))
        run_config["seed"] = int(args.seed)
        axis_tag = str(axis_value).replace(".", "p")
        run_config["experiment_name"] = f"proposal_hyper_sweep_v1_{axis_name}_{axis_tag}_seed{args.seed}"

        if axis_name == "optimizer":
            run_config["train"]["optimizer"] = str(axis_value)
        elif axis_name == "conv_depth":
            run_config["model"]["conv_depth"] = int(axis_value)
        elif axis_name == "embedding_size":
            run_config["model"]["embedding_size"] = int(axis_value)
        elif axis_name == "lstm_hidden_size":
            run_config["model"]["lstm_hidden_size"] = int(axis_value)

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
        latest_run_dir = max(run_dirs, key=_timestamp_from_path)
        with open(latest_run_dir / "metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        sweep_rows.append(
            {
                "axis": axis_name,
                "value": axis_value,
                "iid_acc": float(metrics["lstm"]["iid_test_action_accuracy"]),
                "ood_acc": float(metrics["lstm"]["ood_test_action_accuracy"]),
                "iid_success": float(metrics["lstm"]["iid_test_success_rate"]),
                "ood_success": float(metrics["lstm"]["ood_test_success_rate"]),
                "elapsed_seconds": float(metrics["elapsed_seconds"]),
                "run_dir": str(latest_run_dir.relative_to(repository_root)),
            }
        )

    summary_path = repository_root / "reports" / "tables" / "proposal_hyperparam_sweep_summary.md"
    lines = [
        "# Proposal Hyperparameter Sweep Summary",
        "",
        f"- Base config: `{args.base_config}`",
        f"- Seed: `{args.seed}`",
        "",
        "| Axis | Value | IID Acc | OOD Acc | IID Success | OOD Success | Elapsed (s) | Run Dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['axis']} | {row['value']} | {row['iid_acc']:.4f} | {row['ood_acc']:.4f} | {row['iid_success']:.4f} | {row['ood_success']:.4f} | {row['elapsed_seconds']:.2f} | `{row['run_dir']}` |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {summary_path}")


if __name__ == "__main__":
    main()
