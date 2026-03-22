import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.procedural_gridworld import (
    GridworldDataConfig,
    GridworldImitationDataset,
    crop_egocentric_observation,
    generate_imitation_samples,
)
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.feedforward_policy import FeedforwardPolicyNetwork
from src.utils.device import get_device


def _set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


@torch.no_grad()
def _evaluate_action_accuracy(model, data_loader, device):
    model.eval()
    number_of_correct_predictions = 0
    number_of_samples = 0

    for observation_batch, action_batch in data_loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)
        action_logits = model(observation_batch)
        predicted_actions = torch.argmax(action_logits, dim=1)
        number_of_correct_predictions += (predicted_actions == action_batch).sum().item()
        number_of_samples += action_batch.size(0)

    return number_of_correct_predictions / max(1, number_of_samples)


def _train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    sample_count = 0

    for observation_batch, action_batch in data_loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)

        optimizer.zero_grad()
        action_logits = model(observation_batch)
        loss_value = criterion(action_logits, action_batch)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item() * observation_batch.size(0)
        sample_count += observation_batch.size(0)

    return running_loss / max(1, sample_count)


@torch.no_grad()
def _evaluate_rollout_metrics(model, data_config: GridworldDataConfig, number_of_episodes: int, seed: int, device):
    random_state = np.random.RandomState(seed)
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
        done = False
        steps_taken = 0

        while not done and steps_taken < environment.max_steps:
            cropped_observation = crop_egocentric_observation(
                full_observation,
                environment.agent_pos,
                environment.cell_size,
                data_config.observation_crop_size,
            )
            observation_tensor = torch.from_numpy(cropped_observation).unsqueeze(0).float().to(device)
            action_logits = model(observation_tensor)
            predicted_action = int(torch.argmax(action_logits, dim=1).item())
            full_observation, reward_value, done, _ = environment.step(predicted_action)
            steps_taken += 1
            if done and reward_value > 0:
                successful_episodes += 1

        step_counts.append(steps_taken)

    success_rate = successful_episodes / max(1, number_of_episodes)
    average_steps = float(np.mean(step_counts)) if step_counts else 0.0
    return success_rate, average_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/baseline_smoke.json",
        help="Path to baseline smoke experiment config.",
    )
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / arguments.config
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    seed_value = int(config["seed"])
    _set_seed(seed_value)
    device = get_device()

    data_config = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_train"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )

    split_config = config["data"]["split"]
    train_observations, train_actions = generate_imitation_samples(
        number_of_episodes=int(split_config["train_episodes"]),
        data_config=data_config,
        seed=seed_value,
    )
    val_observations, val_actions = generate_imitation_samples(
        number_of_episodes=int(split_config["val_episodes"]),
        data_config=data_config,
        seed=seed_value + 1,
    )
    test_observations, test_actions = generate_imitation_samples(
        number_of_episodes=int(split_config["test_episodes"]),
        data_config=data_config,
        seed=seed_value + 2,
    )

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(GridworldImitationDataset(train_observations, train_actions), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GridworldImitationDataset(val_observations, val_actions), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GridworldImitationDataset(test_observations, test_actions), batch_size=batch_size, shuffle=False)

    model = FeedforwardPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=int(config["model"]["encoder_embedding_dim"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    number_of_epochs = int(config["train"]["epochs"])

    start_time = time.time()
    best_validation_accuracy = 0.0
    best_state_dictionary = None
    epoch_history = []

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Train/Val/Test sample counts: {len(train_actions)} / {len(val_actions)} / {len(test_actions)}")

    for epoch_index in range(number_of_epochs):
        training_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_accuracy = _evaluate_action_accuracy(model, val_loader, device)
        epoch_history.append(
            {
                "epoch": epoch_index + 1,
                "training_loss": training_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"Epoch {epoch_index + 1}/{number_of_epochs} "
            f"loss={training_loss:.4f} val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state_dictionary = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state_dictionary is not None:
        model.load_state_dict(best_state_dictionary)

    test_accuracy = _evaluate_action_accuracy(model, test_loader, device)
    rollout_success_rate, rollout_average_steps = _evaluate_rollout_metrics(
        model,
        data_config,
        number_of_episodes=int(split_config["test_episodes"]),
        seed=seed_value + 3,
        device=device,
    )
    elapsed_seconds = time.time() - start_time

    run_directory = repository_root / "experiments" / "runs" / f"{config['experiment_name']}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment_name": config["experiment_name"],
        "seed": seed_value,
        "device": str(device),
        "train_samples": int(len(train_actions)),
        "val_samples": int(len(val_actions)),
        "test_samples": int(len(test_actions)),
        "best_validation_accuracy": best_validation_accuracy,
        "test_action_accuracy": test_accuracy,
        "test_success_rate": rollout_success_rate,
        "test_avg_steps_to_goal": rollout_average_steps,
        "elapsed_seconds": elapsed_seconds,
        "epoch_history": epoch_history,
    }

    with open(run_directory / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    torch.save(model.state_dict(), run_directory / "model_state_dict.pth")
    with open(run_directory / "config.json", "w", encoding="utf-8") as copied_config_file:
        json.dump(config, copied_config_file, indent=2)

    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test action accuracy: {test_accuracy:.4f}")
    print(f"Test success rate: {rollout_success_rate:.4f}")
    print(f"Test avg steps to goal: {rollout_average_steps:.2f}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Saved run artifacts to: {run_directory}")


if __name__ == "__main__":
    main()
