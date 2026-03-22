import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.data.procedural_gridworld import crop_egocentric_observation
from src.env.multi_goal_demo_env import MultiGoalDemoEnv
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.recurrent_policy import RecurrentPolicyNetwork


def _ascii_grid_single(grid_size: int, agent_pos, target_pos, obstacles):
    obstacle_set = {tuple(position) for position in obstacles}
    lines = []
    for row in range(grid_size):
        row_tokens = []
        for col in range(grid_size):
            if [row, col] == agent_pos:
                row_tokens.append("A")
            elif [row, col] == target_pos:
                row_tokens.append("T")
            elif (row, col) in obstacle_set:
                row_tokens.append("X")
            else:
                row_tokens.append(".")
        lines.append(" ".join(row_tokens))
    return "\n".join(lines)


def _ascii_grid_multi(grid_size: int, agent_pos, goal_positions, current_goal_index, obstacles):
    obstacle_set = {tuple(position) for position in obstacles}
    lines = []
    for row in range(grid_size):
        row_tokens = []
        for col in range(grid_size):
            if [row, col] == agent_pos:
                row_tokens.append("A")
            elif [row, col] in goal_positions:
                goal_index = goal_positions.index([row, col])
                row_tokens.append(f"T{goal_index + 1}" if goal_index >= current_goal_index else f"G{goal_index + 1}")
            elif (row, col) in obstacle_set:
                row_tokens.append("X")
            else:
                row_tokens.append(".")
        lines.append(" ".join(row_tokens))
    return "\n".join(lines)


def _load_checkpoint_and_config(repository_root: Path, checkpoint_arg: str, config_arg: str):
    if checkpoint_arg:
        checkpoint_path = repository_root / checkpoint_arg
        config_path = repository_root / config_arg if config_arg else checkpoint_path.parent / "config.json"
        return checkpoint_path, config_path
    run_dirs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No fullscale checkpoint found. Provide --checkpoint and --config.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    return latest_run / "lstm_model_state_dict.pth", latest_run / "config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--number-of-goals", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    checkpoint_path, config_path = _load_checkpoint_and_config(repository_root, args.checkpoint, args.config)
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

    np.random.seed(args.seed)
    if args.mode == "single":
        environment = ProceduralGridworldEnv(
            grid_size=args.grid_size,
            cell_size=8,
            num_obstacles=args.obstacles,
            max_steps=args.max_steps,
        )
    else:
        environment = MultiGoalDemoEnv(
            grid_size=args.grid_size,
            cell_size=8,
            num_obstacles=args.obstacles,
            number_of_goals=args.number_of_goals,
            max_steps=args.max_steps,
        )

    crop_cells = int(config["data"]["observation_crop_size"])
    sequence_length = int(config["data"]["sequence_length"])
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    observation = environment.reset()
    history = deque(maxlen=sequence_length)
    done = False
    total_reward = 0.0
    step_index = 0

    print("=== Demo rollout variant ===")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Grid size: {args.grid_size}, obstacles: {args.obstacles}, max steps: {args.max_steps}")
    if args.mode == "multi":
        print(f"Number of goals: {args.number_of_goals}")

    while not done and step_index < args.max_steps:
        crop = crop_egocentric_observation(observation, environment.agent_pos, environment.cell_size, crop_cells)
        history.append(crop)
        if len(history) < sequence_length:
            padded = [history[0]] * (sequence_length - len(history)) + list(history)
        else:
            padded = list(history)
        input_tensor = torch.from_numpy(np.stack(padded)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            action = int(torch.argmax(logits, dim=1).item())

        observation, reward, done, info = environment.step(action)
        total_reward += reward
        step_index += 1

        print("\n" + "=" * 56)
        print(f"Step {step_index} | Action: {action_names[action]} | Reward: {reward:.3f}")
        if args.mode == "single":
            print(_ascii_grid_single(environment.grid_size, info["agent_pos"], info["target_pos"], info["obstacles"]))
        else:
            print(
                _ascii_grid_multi(
                    environment.grid_size,
                    info["agent_pos"],
                    info["goal_positions"],
                    int(info["current_goal_index"]),
                    info["obstacles"],
                )
            )
            if info.get("reached_goal", False):
                print(f"Reached goal index: {info['current_goal_index']}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if args.mode == "single":
        reached = environment.agent_pos == environment.target_pos
    else:
        reached = environment.current_goal_index == environment.number_of_goals - 1 and environment.agent_pos == environment.target_pos
    print("\n=== Demo finished ===")
    print(f"Steps: {step_index}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Completed objective: {'yes' if reached else 'no'}")


if __name__ == "__main__":
    main()
