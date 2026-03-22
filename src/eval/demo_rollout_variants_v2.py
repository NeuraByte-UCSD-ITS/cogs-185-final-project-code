import argparse
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data.procedural_gridworld import crop_egocentric_observation
from src.env.multi_goal_demo_env import MultiGoalDemoEnv
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.recurrent_policy import RecurrentPolicyNetwork
from src.utils.device import get_device


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


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

    run_dirs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_*"))
    if not run_dirs:
        raise FileNotFoundError("No checkpoint found. Provide --checkpoint and --config.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    return latest_run / "lstm_model_state_dict.pth", latest_run / "config.json"


def _next_position(agent_pos: List[int], action: int, grid_size: int) -> Tuple[int, int]:
    row, col = agent_pos
    delta_row, delta_col = ACTION_DELTAS[action]
    next_row = min(grid_size - 1, max(0, row + delta_row))
    next_col = min(grid_size - 1, max(0, col + delta_col))
    return next_row, next_col


def _valid_actions(agent_pos: List[int], obstacles: List[List[int]], grid_size: int) -> List[int]:
    blocked = {tuple(position) for position in obstacles}
    actions = []
    for action in [0, 1, 2, 3]:
        candidate = _next_position(agent_pos, action, grid_size)
        if candidate not in blocked:
            actions.append(action)
    return actions


def _bfs_planner_action(agent_pos: List[int], target_pos: List[int], obstacles: List[List[int]], grid_size: int):
    start = tuple(agent_pos)
    target = tuple(target_pos)
    if start == target:
        return None

    blocked = {tuple(position) for position in obstacles}
    queue = deque([start])
    predecessor: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
    visited = {start}

    while queue:
        node = queue.popleft()
        if node == target:
            break
        for action in [0, 1, 2, 3]:
            next_node = _next_position([node[0], node[1]], action, grid_size)
            if next_node in blocked or next_node in visited:
                continue
            visited.add(next_node)
            predecessor[next_node] = (node, action)
            queue.append(next_node)

    if target not in predecessor:
        return None

    node = target
    first_action = None
    while node != start:
        prev_node, action = predecessor[node]
        first_action = action
        node = prev_node
    return first_action


def _select_action(
    logits: torch.Tensor,
    args,
    agent_pos: List[int],
    target_pos: List[int],
    obstacles: List[List[int]],
    grid_size: int,
    visited_counts: np.ndarray,
    recent_positions: deque,
):
    logits_1d = logits[0]
    valid_actions = _valid_actions(agent_pos, obstacles, grid_size)

    if args.policy_mode == "planner-bfs":
        action = _bfs_planner_action(agent_pos, target_pos, obstacles, grid_size)
        if action is not None:
            return action, "planner"

    if args.policy_mode == "epsilon-greedy" and random.random() < args.epsilon:
        return random.choice(valid_actions), "epsilon"

    if args.policy_mode == "temperature":
        temperature = max(1e-5, args.temperature)
        probabilities = torch.softmax(logits_1d / temperature, dim=0).detach().cpu().numpy()
        probabilities = probabilities / np.sum(probabilities)
        sampled_action = int(np.random.choice(np.arange(4), p=probabilities))
        if sampled_action in valid_actions:
            return sampled_action, "temperature"

    if args.policy_mode in {"hybrid", "novelty"}:
        novelty_scores = []
        for action in valid_actions:
            next_row, next_col = _next_position(agent_pos, action, grid_size)
            novelty_penalty = visited_counts[next_row, next_col]
            novelty_scores.append((action, float(logits_1d[action].item()) - args.novelty_weight * novelty_penalty))
        best_action = max(novelty_scores, key=lambda item: item[1])[0]
    else:
        masked_scores = [(action, float(logits_1d[action].item())) for action in valid_actions]
        best_action = max(masked_scores, key=lambda item: item[1])[0]

    if args.policy_mode == "hybrid":
        if len(recent_positions) >= args.stuck_window and len(set(recent_positions)) <= args.stuck_unique_threshold:
            planner_action = _bfs_planner_action(agent_pos, target_pos, obstacles, grid_size)
            if planner_action is not None:
                return planner_action, "hybrid-planner"

    return best_action, "model"


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
    parser.add_argument("--policy-mode", type=str, choices=["argmax", "epsilon-greedy", "temperature", "planner-bfs", "hybrid", "novelty"], default="argmax")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--novelty-weight", type=float, default=0.2)
    parser.add_argument("--stuck-window", type=int, default=8)
    parser.add_argument("--stuck-unique-threshold", type=int, default=2)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    checkpoint_path, config_path = _load_checkpoint_and_config(repository_root, args.checkpoint, args.config)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    model = RecurrentPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=int(config["model"]["embedding_size"]),
        lstm_hidden_size=int(config["model"]["lstm_hidden_size"]),
        conv_depth=int(config["model"].get("conv_depth", 3)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

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

    observation = environment.reset()
    history = deque(maxlen=sequence_length)
    recent_positions = deque(maxlen=max(args.stuck_window, 2))
    visited_counts = np.zeros((args.grid_size, args.grid_size), dtype=np.int32)

    done = False
    total_reward = 0.0
    step_index = 0

    print("=== Demo rollout variants v2 ===")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Policy mode: {args.policy_mode}")
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

        action, source = _select_action(
            logits,
            args,
            environment.agent_pos,
            environment.target_pos,
            environment.obstacles,
            environment.grid_size,
            visited_counts,
            recent_positions,
        )

        observation, reward, done, info = environment.step(action)
        total_reward += reward
        step_index += 1

        row, col = info["agent_pos"]
        visited_counts[row, col] += 1
        recent_positions.append(tuple(info["agent_pos"]))

        print("\n" + "=" * 56)
        print(f"Step {step_index} | Action: {ACTION_NAMES[action]} ({source}) | Reward: {reward:.3f}")
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
        completed = environment.agent_pos == environment.target_pos
    else:
        completed = (
            environment.current_goal_index == environment.number_of_goals - 1
            and environment.agent_pos == environment.target_pos
        )

    print("\n=== Demo finished ===")
    print(f"Steps: {step_index}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Completed objective: {'yes' if completed else 'no'}")


if __name__ == "__main__":
    main()
