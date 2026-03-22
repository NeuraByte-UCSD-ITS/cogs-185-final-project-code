import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.procedural_gridworld import GridworldDataConfig, crop_egocentric_observation
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.recurrent_policy import RecurrentPolicyNetwork


def _ascii_grid(grid_size: int, agent_pos, target_pos, obstacles):
    obstacle_set = {tuple(position) for position in obstacles}
    rows = []
    for row in range(grid_size):
        cells = []
        for column in range(grid_size):
            if [row, column] == agent_pos:
                cells.append("A")
            elif [row, column] == target_pos:
                cells.append("T")
            elif (row, column) in obstacle_set:
                cells.append("X")
            else:
                cells.append(".")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _load_latest_fullscale_checkpoint(repository_root: Path):
    run_dirs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No fullscale run found. Pass --checkpoint manually.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    return latest_run / "lstm_model_state_dict.pth", latest_run / "config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    if args.checkpoint:
        checkpoint_path = repository_root / args.checkpoint
        if args.config:
            config_path = repository_root / args.config
        else:
            config_path = checkpoint_path.parent / "config.json"
    else:
        checkpoint_path, config_path = _load_latest_fullscale_checkpoint(repository_root)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    data_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_test"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=min(int(config["data"]["episode_horizon_max"]), int(args.max_steps)),
    )
    sequence_length = int(config["data"]["sequence_length"])
    embedding_size = int(config["model"]["embedding_size"])
    lstm_hidden_size = int(config["model"]["lstm_hidden_size"])
    conv_depth = int(config["model"].get("conv_depth", 3))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RecurrentPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=embedding_size,
        lstm_hidden_size=lstm_hidden_size,
        conv_depth=conv_depth,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    np.random.seed(args.seed)
    environment = ProceduralGridworldEnv(
        grid_size=data_cfg.grid_size,
        cell_size=data_cfg.cell_size,
        num_obstacles=data_cfg.obstacle_count,
        max_steps=data_cfg.episode_horizon_max,
    )

    observation = environment.reset()
    history = []
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    done = False
    step_index = 0
    total_reward = 0.0

    print("=== Demo rollout started ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Grid size: {data_cfg.grid_size}, obstacles: {data_cfg.obstacle_count}, max steps: {environment.max_steps}")

    while not done and step_index < environment.max_steps:
        crop = crop_egocentric_observation(
            observation,
            environment.agent_pos,
            environment.cell_size,
            data_cfg.observation_crop_size,
        )
        history.append(crop)
        if len(history) < sequence_length:
            padded = [history[0]] * (sequence_length - len(history)) + list(history)
        else:
            padded = history[-sequence_length:]
        input_tensor = torch.from_numpy(np.stack(padded)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            action = int(torch.argmax(logits, dim=1).item())

        observation, reward, done, info = environment.step(action)
        total_reward += reward
        step_index += 1

        print("\n" + "=" * 42)
        print(f"Step {step_index} | Action: {action_names[action]} | Reward: {reward:.3f}")
        print(_ascii_grid(environment.grid_size, info["agent_pos"], info["target_pos"], info["obstacles"]))
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("\n=== Demo rollout ended ===")
    print(f"Steps taken: {step_index}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Reached target: {'yes' if environment.agent_pos == environment.target_pos else 'no'}")


if __name__ == "__main__":
    main()
