import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.procedural_gridworld import (
    GridworldDataConfig,
    GridworldSequenceImitationDataset,
    crop_egocentric_observation,
    generate_sequence_imitation_samples,
)
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.recurrent_policy import RecurrentPolicyNetwork
from src.utils.device import get_device


def _set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for observation_sequence_batch, action_batch in loader:
        observation_sequence_batch = observation_sequence_batch.to(device)
        action_batch = action_batch.to(device)
        optimizer.zero_grad()
        logits = model(observation_sequence_batch)
        loss_value = criterion(logits, action_batch)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item() * action_batch.size(0)
        total_count += action_batch.size(0)
    return total_loss / max(1, total_count)


@torch.no_grad()
def _evaluate_action_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for observation_sequence_batch, action_batch in loader:
        observation_sequence_batch = observation_sequence_batch.to(device)
        action_batch = action_batch.to(device)
        logits = model(observation_sequence_batch)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == action_batch).sum().item()
        total += action_batch.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _evaluate_rollout(model, data_config: GridworldDataConfig, number_of_episodes: int, seed_value: int, sequence_length: int, device):
    random_state = np.random.RandomState(seed_value)
    environment = ProceduralGridworldEnv(
        grid_size=data_config.grid_size,
        cell_size=data_config.cell_size,
        num_obstacles=data_config.obstacle_count,
        max_steps=data_config.episode_horizon_max,
    )
    successful_episodes = 0
    step_counts = []
    model.eval()

    for _ in range(number_of_episodes):
        environment.max_steps = int(
            random_state.randint(data_config.episode_horizon_min, data_config.episode_horizon_max + 1)
        )
        full_observation = environment.reset()
        history = deque(maxlen=sequence_length)
        done = False
        steps_taken = 0
        while not done and steps_taken < environment.max_steps:
            crop = crop_egocentric_observation(
                full_observation,
                environment.agent_pos,
                environment.cell_size,
                data_config.observation_crop_size,
            )
            history.append(crop)
            if len(history) < sequence_length:
                padded = [history[0]] * (sequence_length - len(history)) + list(history)
            else:
                padded = list(history)

            input_tensor = torch.from_numpy(np.stack(padded)).unsqueeze(0).float().to(device)
            action = int(torch.argmax(model(input_tensor), dim=1).item())
            full_observation, reward_value, done, _ = environment.step(action)
            steps_taken += 1
            if done and reward_value > 0:
                successful_episodes += 1
        step_counts.append(steps_taken)

    return successful_episodes / max(1, number_of_episodes), float(np.mean(step_counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / args.config
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    seed_value = int(config["seed"])
    _set_seed(seed_value)
    device = get_device()

    sequence_length = int(config["data"]["sequence_length"])
    data_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_train"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )
    split_cfg = config["data"]["split"]

    train_x, train_y = generate_sequence_imitation_samples(int(split_cfg["train_episodes"]), data_cfg, seed_value, sequence_length)
    val_x, val_y = generate_sequence_imitation_samples(int(split_cfg["val_episodes"]), data_cfg, seed_value + 1, sequence_length)
    test_x, test_y = generate_sequence_imitation_samples(int(split_cfg["test_episodes"]), data_cfg, seed_value + 2, sequence_length)

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(GridworldSequenceImitationDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GridworldSequenceImitationDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GridworldSequenceImitationDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    ood_cfg = config["data"].get("ood", {})
    ood_enabled = bool(ood_cfg.get("enabled", False))
    if ood_enabled:
        ood_data_cfg = GridworldDataConfig(
            grid_size=int(ood_cfg.get("grid_size_test", config["data"]["grid_size_test"])),
            obstacle_count=int(ood_cfg.get("obstacle_count", config["data"]["obstacle_count"])),
            observation_crop_size=int(config["data"]["observation_crop_size"]),
            episode_horizon_min=int(config["data"]["episode_horizon_min"]),
            episode_horizon_max=int(config["data"]["episode_horizon_max"]),
        )
        ood_x, ood_y = generate_sequence_imitation_samples(int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])), ood_data_cfg, seed_value + 20, sequence_length)
        ood_loader = DataLoader(GridworldSequenceImitationDataset(ood_x, ood_y), batch_size=batch_size, shuffle=False)
    else:
        ood_data_cfg = None
        ood_loader = None

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Train/Val/Test samples: {len(train_y)}/{len(val_y)}/{len(test_y)}")
    if ood_enabled:
        print(f"OOD test samples: {len(ood_y)} (grid={ood_data_cfg.grid_size}, obstacles={ood_data_cfg.obstacle_count})")

    model = RecurrentPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=int(config["model"]["embedding_size"]),
        lstm_hidden_size=int(config["model"]["lstm_hidden_size"]),
        conv_depth=int(config["model"].get("conv_depth", 3)),
    ).to(device)
    optimizer_name = str(config["train"].get("optimizer", "adam")).lower()
    learning_rate = float(config["train"]["learning_rate"])
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    best_val_action_accuracy = 0.0
    best_state_dict = None
    history = []
    for epoch_index in range(int(config["train"]["epochs"])):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_action_accuracy = _evaluate_action_accuracy(model, val_loader, device)
        history.append(
            {"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_action_accuracy}
        )
        print(
            f"Epoch {epoch_index + 1}/{int(config['train']['epochs'])} "
            f"loss={train_loss:.4f} val_acc={val_action_accuracy:.4f}"
        )
        if val_action_accuracy > best_val_action_accuracy:
            best_val_action_accuracy = val_action_accuracy
            best_state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    iid_test_action_accuracy = _evaluate_action_accuracy(model, test_loader, device)
    iid_test_success_rate, iid_test_avg_steps = _evaluate_rollout(
        model, data_cfg, int(split_cfg["test_episodes"]), seed_value + 3, sequence_length, device
    )
    if ood_enabled:
        ood_test_action_accuracy = _evaluate_action_accuracy(model, ood_loader, device)
        ood_test_success_rate, ood_test_avg_steps = _evaluate_rollout(
            model, ood_data_cfg, int(ood_cfg["test_episodes"]), seed_value + 30, sequence_length, device
        )
    else:
        ood_test_action_accuracy = None
        ood_test_success_rate = None
        ood_test_avg_steps = None

    elapsed_seconds = time.time() - start_time
    run_directory = repository_root / "experiments" / "runs" / f"{config['experiment_name']}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment_name": config["experiment_name"],
        "seed": seed_value,
        "device": str(device),
        "settings": {
            "train_episodes": int(split_cfg["train_episodes"]),
            "val_episodes": int(split_cfg["val_episodes"]),
            "test_episodes": int(split_cfg["test_episodes"]),
            "observation_crop_size": int(config["data"]["observation_crop_size"]),
            "obstacle_count": int(config["data"]["obstacle_count"]),
            "sequence_length": sequence_length,
            "optimizer": optimizer_name,
            "conv_depth": int(config["model"].get("conv_depth", 3)),
            "embedding_size": int(config["model"]["embedding_size"]),
            "lstm_hidden_size": int(config["model"]["lstm_hidden_size"]),
        },
        "lstm": {
            "best_validation_action_accuracy": best_val_action_accuracy,
            "iid_test_action_accuracy": iid_test_action_accuracy,
            "iid_test_success_rate": iid_test_success_rate,
            "iid_test_avg_steps_to_goal": iid_test_avg_steps,
            "ood_test_action_accuracy": ood_test_action_accuracy,
            "ood_test_success_rate": ood_test_success_rate,
            "ood_test_avg_steps_to_goal": ood_test_avg_steps,
            "history": history,
        },
        "elapsed_seconds": elapsed_seconds,
    }

    with open(run_directory / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(run_directory / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    torch.save(model.state_dict(), run_directory / "lstm_model_state_dict.pth")

    print("\n=== Summary ===")
    print(f"IID action acc: {iid_test_action_accuracy:.4f}, success: {iid_test_success_rate:.4f}")
    if ood_test_action_accuracy is not None:
        print(f"OOD action acc: {ood_test_action_accuracy:.4f}, success: {ood_test_success_rate:.4f}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Saved run artifacts to: {run_directory}")


if __name__ == "__main__":
    main()
