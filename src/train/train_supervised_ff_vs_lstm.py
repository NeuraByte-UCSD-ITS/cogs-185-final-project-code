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
    GridworldImitationDataset,
    GridworldSequenceImitationDataset,
    crop_egocentric_observation,
    generate_imitation_samples,
    generate_sequence_imitation_samples,
)
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.feedforward_policy import FeedforwardPolicyNetwork
from src.models.recurrent_policy import RecurrentPolicyNetwork
from src.utils.device import get_device


def _set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        number_of_samples += inputs.size(0)
    return running_loss / max(1, number_of_samples)


@torch.no_grad()
def _evaluate_action_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _evaluate_rollout_feedforward(model, data_config: GridworldDataConfig, number_of_episodes: int, seed: int, device):
    np_random = np.random.RandomState(seed)
    env = ProceduralGridworldEnv(
        grid_size=data_config.grid_size,
        cell_size=data_config.cell_size,
        num_obstacles=data_config.obstacle_count,
        max_steps=data_config.episode_horizon_max,
    )
    successful = 0
    step_counts = []
    model.eval()
    for _ in range(number_of_episodes):
        env.max_steps = int(np_random.randint(data_config.episode_horizon_min, data_config.episode_horizon_max + 1))
        observation = env.reset()
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            crop = crop_egocentric_observation(observation, env.agent_pos, env.cell_size, data_config.observation_crop_size)
            x = torch.from_numpy(crop).unsqueeze(0).float().to(device)
            action = int(torch.argmax(model(x), dim=1).item())
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward > 0:
                successful += 1
        step_counts.append(steps)
    return successful / max(1, number_of_episodes), float(np.mean(step_counts))


@torch.no_grad()
def _evaluate_rollout_recurrent(
    model,
    data_config: GridworldDataConfig,
    number_of_episodes: int,
    seed: int,
    sequence_length: int,
    device,
):
    np_random = np.random.RandomState(seed)
    env = ProceduralGridworldEnv(
        grid_size=data_config.grid_size,
        cell_size=data_config.cell_size,
        num_obstacles=data_config.obstacle_count,
        max_steps=data_config.episode_horizon_max,
    )
    successful = 0
    step_counts = []
    model.eval()
    for _ in range(number_of_episodes):
        env.max_steps = int(np_random.randint(data_config.episode_horizon_min, data_config.episode_horizon_max + 1))
        observation = env.reset()
        done = False
        steps = 0
        history = deque(maxlen=sequence_length)
        while not done and steps < env.max_steps:
            crop = crop_egocentric_observation(observation, env.agent_pos, env.cell_size, data_config.observation_crop_size)
            history.append(crop)
            if len(history) < sequence_length:
                padded = [history[0]] * (sequence_length - len(history)) + list(history)
            else:
                padded = list(history)
            x = torch.from_numpy(np.stack(padded)).unsqueeze(0).float().to(device)
            action = int(torch.argmax(model(x), dim=1).item())
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward > 0:
                successful += 1
        step_counts.append(steps)
    return successful / max(1, number_of_episodes), float(np.mean(step_counts))


def _train_model(model, train_loader, val_loader, test_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val = 0.0
    best_state = None
    history = []
    for epoch_index in range(epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = _evaluate_action_accuracy(model, val_loader, device)
        history.append({"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_acc})
        print(f"Epoch {epoch_index + 1}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = _evaluate_action_accuracy(model, test_loader, device)
    return best_val, test_acc, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/supervised_ff_vs_lstm_smoke.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config["seed"])
    _set_seed(seed)
    device = get_device()

    data_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_train"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )
    split_cfg = config["data"]["split"]
    seq_len = int(config["data"]["sequence_length"])
    batch_size = int(config["train"]["batch_size"])
    epochs = int(config["train"]["epochs"])
    learning_rate = float(config["train"]["learning_rate"])

    # feedforward data
    ff_train_x, ff_train_y = generate_imitation_samples(int(split_cfg["train_episodes"]), data_cfg, seed)
    ff_val_x, ff_val_y = generate_imitation_samples(int(split_cfg["val_episodes"]), data_cfg, seed + 1)
    ff_test_x, ff_test_y = generate_imitation_samples(int(split_cfg["test_episodes"]), data_cfg, seed + 2)

    ff_train_loader = DataLoader(GridworldImitationDataset(ff_train_x, ff_train_y), batch_size=batch_size, shuffle=True)
    ff_val_loader = DataLoader(GridworldImitationDataset(ff_val_x, ff_val_y), batch_size=batch_size, shuffle=False)
    ff_test_loader = DataLoader(GridworldImitationDataset(ff_test_x, ff_test_y), batch_size=batch_size, shuffle=False)

    # recurrent data
    r_train_x, r_train_y = generate_sequence_imitation_samples(int(split_cfg["train_episodes"]), data_cfg, seed, seq_len)
    r_val_x, r_val_y = generate_sequence_imitation_samples(int(split_cfg["val_episodes"]), data_cfg, seed + 1, seq_len)
    r_test_x, r_test_y = generate_sequence_imitation_samples(int(split_cfg["test_episodes"]), data_cfg, seed + 2, seq_len)

    r_train_loader = DataLoader(GridworldSequenceImitationDataset(r_train_x, r_train_y), batch_size=batch_size, shuffle=True)
    r_val_loader = DataLoader(GridworldSequenceImitationDataset(r_val_x, r_val_y), batch_size=batch_size, shuffle=False)
    r_test_loader = DataLoader(GridworldSequenceImitationDataset(r_test_x, r_test_y), batch_size=batch_size, shuffle=False)

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"FF train/val/test samples: {len(ff_train_y)}/{len(ff_val_y)}/{len(ff_test_y)}")
    print(f"LSTM train/val/test samples: {len(r_train_y)}/{len(r_val_y)}/{len(r_test_y)}")

    start_time = time.time()

    ff_model = FeedforwardPolicyNetwork(input_channels=3, number_of_actions=4, embedding_size=64).to(device)
    print("\nTraining Feedforward baseline...")
    ff_best_val, ff_test_acc, ff_history = _train_model(
        ff_model, ff_train_loader, ff_val_loader, ff_test_loader, device, epochs, learning_rate
    )
    ff_success, ff_avg_steps = _evaluate_rollout_feedforward(
        ff_model, data_cfg, int(split_cfg["test_episodes"]), seed + 3, device
    )

    lstm_model = RecurrentPolicyNetwork(input_channels=3, number_of_actions=4, embedding_size=64, lstm_hidden_size=64).to(device)
    print("\nTraining LSTM baseline...")
    lstm_best_val, lstm_test_acc, lstm_history = _train_model(
        lstm_model, r_train_loader, r_val_loader, r_test_loader, device, epochs, learning_rate
    )
    lstm_success, lstm_avg_steps = _evaluate_rollout_recurrent(
        lstm_model, data_cfg, int(split_cfg["test_episodes"]), seed + 3, seq_len, device
    )

    elapsed_seconds = time.time() - start_time

    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "feedforward": {
            "best_validation_action_accuracy": ff_best_val,
            "test_action_accuracy": ff_test_acc,
            "test_success_rate": ff_success,
            "test_avg_steps_to_goal": ff_avg_steps,
            "epoch_history": ff_history,
        },
        "lstm": {
            "best_validation_action_accuracy": lstm_best_val,
            "test_action_accuracy": lstm_test_acc,
            "test_success_rate": lstm_success,
            "test_avg_steps_to_goal": lstm_avg_steps,
            "epoch_history": lstm_history,
        },
        "elapsed_seconds": elapsed_seconds,
    }

    run_directory = repo_root / "experiments" / "runs" / f"{config['experiment_name']}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)
    with open(run_directory / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(run_directory / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    torch.save(ff_model.state_dict(), run_directory / "feedforward_model_state_dict.pth")
    torch.save(lstm_model.state_dict(), run_directory / "lstm_model_state_dict.pth")

    table_path = repo_root / "reports" / "tables" / "ff_vs_lstm_smoke.md"
    table_text = (
        "# FF vs LSTM Smoke Comparison\n\n"
        f"- Run directory: `{run_directory}`\n"
        f"- Device: `{device}`\n\n"
        "| Model | Best Val Action Acc | Test Action Acc | Test Success Rate | Avg Steps to Goal |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Feedforward | {ff_best_val:.4f} | {ff_test_acc:.4f} | {ff_success:.4f} | {ff_avg_steps:.2f} |\n"
        f"| LSTM | {lstm_best_val:.4f} | {lstm_test_acc:.4f} | {lstm_success:.4f} | {lstm_avg_steps:.2f} |\n"
    )
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    print("\n=== Summary ===")
    print(f"Feedforward test action accuracy: {ff_test_acc:.4f}, success rate: {ff_success:.4f}")
    print(f"LSTM test action accuracy:       {lstm_test_acc:.4f}, success rate: {lstm_success:.4f}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Saved run artifacts to: {run_directory}")
    print(f"Updated table: {table_path}")


if __name__ == "__main__":
    main()

