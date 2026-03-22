import json
from pathlib import Path

from src.utils.device import get_device


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "experiments" / "configs" / "baseline_smoke.json"

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    device = get_device()

    print("Loaded config:", config["experiment_name"])
    print("Proposal alignment:", config["proposal_alignment"])
    print("Resolved device:", device)
    print("--- Data ---")
    for key, value in config["data"].items():
        print(f"{key}: {value}")
    print("--- Model ---")
    for key, value in config["model"].items():
        print(f"{key}: {value}")
    print("--- Train ---")
    for key, value in config["train"].items():
        print(f"{key}: {value}")
    print("--- Eval metrics ---")
    for metric_name in config["eval"]["metrics"]:
        print(metric_name)


if __name__ == "__main__":
    main()

