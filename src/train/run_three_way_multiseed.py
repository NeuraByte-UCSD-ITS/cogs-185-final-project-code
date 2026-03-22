import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=str,
        default="experiments/configs/supervised_vs_rotation_vs_contrastive_largesplit.json",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    base_config_path = repository_root / args.base_config
    with open(base_config_path, "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    generated_config_dir = repository_root / "experiments" / "configs" / "generated"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    for seed_value in args.seeds:
        run_config = dict(base_config)
        run_config["seed"] = int(seed_value)
        run_config["experiment_name"] = f"{base_config['experiment_name']}_seed{seed_value}"
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


if __name__ == "__main__":
    main()

