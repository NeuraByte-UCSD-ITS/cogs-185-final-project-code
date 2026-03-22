import argparse
import json
import subprocess
from pathlib import Path


def _load_metrics(metrics_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    supervised_ood = float(metrics["supervised"]["ood_test_action_accuracy"])
    rotation_ood = float(metrics["rotation_ssl_fine_tune"]["ood_test_action_accuracy"])
    contrastive_ood = float(metrics["contrastive_ssl_fine_tune"]["ood_test_action_accuracy"])
    best_ssl_ood = max(rotation_ood, contrastive_ood)
    best_ssl_method = "rotation_ssl_fine_tune" if rotation_ood >= contrastive_ood else "contrastive_ssl_fine_tune"
    return {
        "supervised_ood": supervised_ood,
        "rotation_ood": rotation_ood,
        "contrastive_ood": contrastive_ood,
        "best_ssl_ood": best_ssl_ood,
        "best_ssl_method": best_ssl_method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=str,
        default="experiments/configs/supervised_vs_rotation_vs_contrastive_quickcheck.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    base_config_path = repository_root / args.base_config
    with open(base_config_path, "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    sweep_grid = [
        {"name": "baseline", "rotation_epochs": 8, "contrastive_epochs": 8, "ssl_lr": 1e-3, "max_shift": 2, "brightness_jitter": 0.15, "noise_std": 0.03},
        {"name": "longer_pretrain", "rotation_epochs": 12, "contrastive_epochs": 12, "ssl_lr": 1e-3, "max_shift": 2, "brightness_jitter": 0.15, "noise_std": 0.03},
        {"name": "stronger_aug", "rotation_epochs": 12, "contrastive_epochs": 12, "ssl_lr": 1e-3, "max_shift": 3, "brightness_jitter": 0.25, "noise_std": 0.05},
        {"name": "stronger_aug_lower_lr", "rotation_epochs": 12, "contrastive_epochs": 12, "ssl_lr": 5e-4, "max_shift": 3, "brightness_jitter": 0.25, "noise_std": 0.05},
    ]

    generated_config_dir = repository_root / "experiments" / "configs" / "generated"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for grid_item in sweep_grid:
        run_config = dict(base_config)
        run_config["seed"] = int(args.seed)
        run_config["experiment_name"] = f"{base_config['experiment_name']}_ssl_tune_{grid_item['name']}_seed{args.seed}"
        run_config["train"] = dict(base_config["train"])
        run_config["train"]["rotation_ssl_pretrain_epochs"] = int(grid_item["rotation_epochs"])
        run_config["train"]["contrastive_ssl_pretrain_epochs"] = int(grid_item["contrastive_epochs"])
        run_config["train"]["learning_rate_ssl"] = float(grid_item["ssl_lr"])
        run_config["ssl_augmentation"] = {
            "max_shift_pixels": int(grid_item["max_shift"]),
            "brightness_jitter": float(grid_item["brightness_jitter"]),
            "noise_std": float(grid_item["noise_std"]),
        }

        generated_config_path = generated_config_dir / f"{run_config['experiment_name']}.json"
        with open(generated_config_path, "w", encoding="utf-8") as generated_file:
            json.dump(run_config, generated_file, indent=2)

        command = [
            "python",
            "-m",
            "src.train.train_three_way_ssl_compare",
            "--config",
            str(generated_config_path.relative_to(repository_root)),
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=repository_root)

        run_dirs = sorted((repository_root / "experiments" / "runs").glob(f"{run_config['experiment_name']}_*"))
        if not run_dirs:
            raise FileNotFoundError(f"No run directory found for {run_config['experiment_name']}")
        latest_run_dir = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
        metrics_path = latest_run_dir / "metrics.json"
        metrics_summary = _load_metrics(metrics_path)
        sweep_results.append(
            {
                "config_name": grid_item["name"],
                "config_path": str(generated_config_path.relative_to(repository_root)),
                "run_dir": str(latest_run_dir.relative_to(repository_root)),
                "supervised_ood": metrics_summary["supervised_ood"],
                "rotation_ood": metrics_summary["rotation_ood"],
                "contrastive_ood": metrics_summary["contrastive_ood"],
                "best_ssl_ood": metrics_summary["best_ssl_ood"],
                "best_ssl_method": metrics_summary["best_ssl_method"],
                "delta_best_ssl_vs_supervised_ood": metrics_summary["best_ssl_ood"] - metrics_summary["supervised_ood"],
            }
        )

    sweep_results = sorted(sweep_results, key=lambda item: item["best_ssl_ood"], reverse=True)
    summary_path = repository_root / "reports" / "tables" / "ssl_tuning_sweep_summary.md"
    lines = [
        "# SSL Tuning Sweep Summary",
        "",
        f"- Base config: `{args.base_config}`",
        f"- Seed: `{args.seed}`",
        "",
        "| Config | Best SSL Method | Supervised OOD | Rotation OOD | Contrastive OOD | Best SSL OOD | Delta (Best SSL - Supervised) | Run Dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep_results:
        lines.append(
            f"| {row['config_name']} | {row['best_ssl_method']} | {row['supervised_ood']:.4f} | {row['rotation_ood']:.4f} | {row['contrastive_ood']:.4f} | {row['best_ssl_ood']:.4f} | {row['delta_best_ssl_vs_supervised_ood']:.4f} | `{row['run_dir']}` |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {summary_path}")
    print("Top configuration:", sweep_results[0]["config_name"])


if __name__ == "__main__":
    main()
