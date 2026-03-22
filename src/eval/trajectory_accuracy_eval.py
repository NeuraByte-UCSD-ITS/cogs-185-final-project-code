import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.data.procedural_gridworld import GridworldDataConfig, _compute_oracle_action, crop_egocentric_observation
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.recurrent_policy import RecurrentPolicyNetwork


def _load_latest_fullscale(repository_root: Path):
    run_dirs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No fullscale run found. Provide --checkpoint and --config.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    return latest_run / "lstm_model_state_dict.pth", latest_run / "config.json"


def _evaluate_trajectory_accuracy(model, data_cfg, sequence_length: int, number_of_episodes: int, seed: int, device):
    random_state = np.random.RandomState(seed)
    env = ProceduralGridworldEnv(
        grid_size=data_cfg.grid_size,
        cell_size=data_cfg.cell_size,
        num_obstacles=data_cfg.obstacle_count,
        max_steps=data_cfg.episode_horizon_max,
    )
    exact_match_episodes = 0
    total_step_matches = 0
    total_steps = 0
    success_count = 0

    for _ in range(number_of_episodes):
        env.max_steps = int(random_state.randint(data_cfg.episode_horizon_min, data_cfg.episode_horizon_max + 1))
        observation = env.reset()
        done = False
        history = deque(maxlen=sequence_length)
        episode_all_steps_matched = True
        while not done:
            oracle_action = _compute_oracle_action(env.agent_pos, env.target_pos, env.obstacles, env.grid_size)
            crop = crop_egocentric_observation(observation, env.agent_pos, env.cell_size, data_cfg.observation_crop_size)
            history.append(crop)
            if len(history) < sequence_length:
                padded = [history[0]] * (sequence_length - len(history)) + list(history)
            else:
                padded = list(history)
            input_tensor = torch.from_numpy(np.stack(padded)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                predicted_action = int(torch.argmax(model(input_tensor), dim=1).item())

            step_match = int(predicted_action == oracle_action)
            total_step_matches += step_match
            total_steps += 1
            if step_match == 0:
                episode_all_steps_matched = False

            observation, reward, done, _ = env.step(predicted_action)
            if done and reward > 0:
                success_count += 1

        if episode_all_steps_matched:
            exact_match_episodes += 1

    return {
        "step_match_accuracy": total_step_matches / max(1, total_steps),
        "exact_trajectory_match_rate": exact_match_episodes / max(1, number_of_episodes),
        "success_rate": success_count / max(1, number_of_episodes),
        "episodes": number_of_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    if args.checkpoint:
        checkpoint_path = repository_root / args.checkpoint
        config_path = repository_root / args.config if args.config else checkpoint_path.parent / "config.json"
    else:
        checkpoint_path, config_path = _load_latest_fullscale(repository_root)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RecurrentPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=int(config["model"]["embedding_size"]),
        lstm_hidden_size=int(config["model"]["lstm_hidden_size"]),
        conv_depth=int(config["model"].get("conv_depth", 3)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    iid_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_test"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )
    ood_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["ood"]["grid_size_test"]),
        obstacle_count=int(config["data"]["ood"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )
    sequence_length = int(config["data"]["sequence_length"])

    iid_metrics = _evaluate_trajectory_accuracy(model, iid_cfg, sequence_length, args.episodes, args.seed, device)
    ood_metrics = _evaluate_trajectory_accuracy(model, ood_cfg, sequence_length, args.episodes, args.seed + 100, device)

    output_table = repository_root / "reports" / "tables" / "trajectory_accuracy_summary.md"
    lines = [
        "# Trajectory Accuracy Summary",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Config: `{config_path}`",
        f"- Device: `{device}`",
        f"- Episodes per split: `{args.episodes}`",
        "",
        "| Split | Step-match Accuracy | Exact Trajectory Match Rate | Success Rate |",
        "|---|---:|---:|---:|",
        f"| IID | {iid_metrics['step_match_accuracy']:.4f} | {iid_metrics['exact_trajectory_match_rate']:.4f} | {iid_metrics['success_rate']:.4f} |",
        f"| OOD | {ood_metrics['step_match_accuracy']:.4f} | {ood_metrics['exact_trajectory_match_rate']:.4f} | {ood_metrics['success_rate']:.4f} |",
    ]
    output_table.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {output_table}")


if __name__ == "__main__":
    main()
