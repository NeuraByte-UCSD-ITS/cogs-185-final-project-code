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
from src.models.rotation_ssl_policy import RotationSslPolicyModel
from src.utils.device import get_device


def _set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _train_supervised_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for observation_batch, action_batch in loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)
        optimizer.zero_grad()
        logits = model(observation_batch)
        loss_value = criterion(logits, action_batch)
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item() * observation_batch.size(0)
        number_of_samples += observation_batch.size(0)
    return running_loss / max(1, number_of_samples)


def _train_action_epoch_ssl_model(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for observation_batch, action_batch in loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)
        optimizer.zero_grad()
        logits = model.forward_action(observation_batch)
        loss_value = criterion(logits, action_batch)
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item() * observation_batch.size(0)
        number_of_samples += observation_batch.size(0)
    return running_loss / max(1, number_of_samples)


def _build_rotation_batch(observation_batch: torch.Tensor):
    batch_size = observation_batch.size(0)
    rotation_targets = torch.randint(0, 4, (batch_size,), device=observation_batch.device)
    rotated_batch = torch.empty_like(observation_batch)
    for sample_index in range(batch_size):
        rotated_batch[sample_index] = torch.rot90(
            observation_batch[sample_index],
            k=int(rotation_targets[sample_index].item()),
            dims=(1, 2),
        )
    return rotated_batch, rotation_targets


def _augment_for_ssl(observation_batch: torch.Tensor):
    """Lightweight SSL augmentations: translation jitter, brightness jitter, and Gaussian noise."""
    augmented_batch = observation_batch.clone()
    batch_size = augmented_batch.size(0)

    for sample_index in range(batch_size):
        sample = augmented_batch[sample_index]

        # Random translation jitter using roll in pixel-space.
        shift_y = int(torch.randint(-2, 3, (1,), device=sample.device).item())
        shift_x = int(torch.randint(-2, 3, (1,), device=sample.device).item())
        sample = torch.roll(sample, shifts=(shift_y, shift_x), dims=(1, 2))

        # Random brightness jitter.
        brightness_scale = 0.85 + 0.30 * torch.rand(1, device=sample.device).item()
        sample = sample * brightness_scale

        # Additive Gaussian noise.
        noise_sigma = 0.03
        sample = sample + noise_sigma * torch.randn_like(sample)

        augmented_batch[sample_index] = torch.clamp(sample, 0.0, 1.0)

    return augmented_batch


def _train_ssl_rotation_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for observation_batch, _ in loader:
        observation_batch = observation_batch.to(device)
        augmented_batch = _augment_for_ssl(observation_batch)
        rotated_batch, rotation_targets = _build_rotation_batch(augmented_batch)
        optimizer.zero_grad()
        rotation_logits = model.forward_rotation(rotated_batch)
        loss_value = criterion(rotation_logits, rotation_targets)
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item() * observation_batch.size(0)
        number_of_samples += observation_batch.size(0)
    return running_loss / max(1, number_of_samples)


@torch.no_grad()
def _evaluate_action_accuracy_supervised(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for observation_batch, action_batch in loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)
        logits = model(observation_batch)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == action_batch).sum().item()
        total += action_batch.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _evaluate_action_accuracy_ssl_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for observation_batch, action_batch in loader:
        observation_batch = observation_batch.to(device)
        action_batch = action_batch.to(device)
        logits = model.forward_action(observation_batch)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == action_batch).sum().item()
        total += action_batch.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _evaluate_rollout(model_predict_fn, data_config: GridworldDataConfig, number_of_episodes: int, seed: int, device):
    random_state = np.random.RandomState(seed)
    environment = ProceduralGridworldEnv(
        grid_size=data_config.grid_size,
        cell_size=data_config.cell_size,
        num_obstacles=data_config.obstacle_count,
        max_steps=data_config.episode_horizon_max,
    )
    successful_episodes = 0
    steps_per_episode = []

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
            action_logits = model_predict_fn(observation_tensor)
            predicted_action = int(torch.argmax(action_logits, dim=1).item())
            full_observation, reward_value, done, _ = environment.step(predicted_action)
            steps_taken += 1
            if done and reward_value > 0:
                successful_episodes += 1
        steps_per_episode.append(steps_taken)

    return successful_episodes / max(1, number_of_episodes), float(np.mean(steps_per_episode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/supervised_vs_rotation_ssl_quickcheck.json",
    )
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / arguments.config
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    seed_value = int(config["seed"])
    _set_seed(seed_value)
    device = get_device()

    data_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_train"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )

    split_cfg = config["data"]["split"]
    train_observations, train_actions = generate_imitation_samples(int(split_cfg["train_episodes"]), data_cfg, seed_value)
    val_observations, val_actions = generate_imitation_samples(int(split_cfg["val_episodes"]), data_cfg, seed_value + 1)
    test_observations, test_actions = generate_imitation_samples(int(split_cfg["test_episodes"]), data_cfg, seed_value + 2)

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(GridworldImitationDataset(train_observations, train_actions), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GridworldImitationDataset(val_observations, val_actions), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GridworldImitationDataset(test_observations, test_actions), batch_size=batch_size, shuffle=False)

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
        ood_test_observations, ood_test_actions = generate_imitation_samples(
            int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])),
            ood_data_cfg,
            seed_value + 20,
        )
        ood_test_loader = DataLoader(
            GridworldImitationDataset(ood_test_observations, ood_test_actions),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        ood_data_cfg = None
        ood_test_loader = None

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Train/Val/Test samples: {len(train_actions)}/{len(val_actions)}/{len(test_actions)}")
    if ood_enabled:
        print(f"OOD test samples: {len(ood_test_actions)} (grid={ood_data_cfg.grid_size}, obstacles={ood_data_cfg.obstacle_count})")

    start_time = time.time()

    # 1) Supervised baseline
    supervised_model = FeedforwardPolicyNetwork(
        input_channels=3,
        number_of_actions=4,
        embedding_size=int(config["model"]["embedding_size"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    supervised_optimizer = torch.optim.Adam(
        supervised_model.parameters(),
        lr=float(config["train"]["learning_rate_supervised"]),
    )
    supervised_best_val = 0.0
    supervised_history = []
    supervised_epochs = int(config["train"]["supervised_epochs"])

    for epoch_index in range(supervised_epochs):
        train_loss = _train_supervised_epoch(supervised_model, train_loader, supervised_optimizer, criterion, device)
        val_accuracy = _evaluate_action_accuracy_supervised(supervised_model, val_loader, device)
        supervised_history.append(
            {"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_accuracy}
        )
        supervised_best_val = max(supervised_best_val, val_accuracy)
        print(f"[Supervised] epoch {epoch_index + 1}/{supervised_epochs} loss={train_loss:.4f} val_acc={val_accuracy:.4f}")

    supervised_test_action_accuracy = _evaluate_action_accuracy_supervised(supervised_model, test_loader, device)
    supervised_success_rate, supervised_avg_steps = _evaluate_rollout(
        supervised_model,
        data_cfg,
        number_of_episodes=int(split_cfg["test_episodes"]),
        seed=seed_value + 3,
        device=device,
    )
    if ood_enabled:
        supervised_ood_action_accuracy = _evaluate_action_accuracy_supervised(supervised_model, ood_test_loader, device)
        supervised_ood_success_rate, supervised_ood_avg_steps = _evaluate_rollout(
            supervised_model,
            ood_data_cfg,
            number_of_episodes=int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])),
            seed=seed_value + 30,
            device=device,
        )
    else:
        supervised_ood_action_accuracy = None
        supervised_ood_success_rate = None
        supervised_ood_avg_steps = None

    # 2) Rotation SSL pretrain + linear probe + fine-tune
    ssl_model = RotationSslPolicyModel(
        input_channels=3,
        embedding_size=int(config["model"]["embedding_size"]),
        number_of_actions=4,
    ).to(device)
    ssl_optimizer = torch.optim.Adam(
        list(ssl_model.encoder.parameters()) + list(ssl_model.rotation_head.parameters()),
        lr=float(config["train"]["learning_rate_ssl"]),
    )
    ssl_epochs = int(config["train"]["ssl_pretrain_epochs"])
    ssl_history = []
    for epoch_index in range(ssl_epochs):
        ssl_loss = _train_ssl_rotation_epoch(ssl_model, train_loader, ssl_optimizer, criterion, device)
        ssl_history.append({"epoch": epoch_index + 1, "ssl_loss": ssl_loss})
        print(f"[Rotation SSL] epoch {epoch_index + 1}/{ssl_epochs} loss={ssl_loss:.4f}")

    # Linear probe
    for parameter in ssl_model.encoder.parameters():
        parameter.requires_grad = False
    linear_probe_optimizer = torch.optim.SGD(
        ssl_model.action_head.parameters(),
        lr=float(config["train"]["learning_rate_linear_probe"]),
        momentum=0.9,
    )
    linear_probe_epochs = int(config["train"]["linear_probe_epochs"])
    linear_probe_best_val = 0.0
    linear_probe_history = []
    for epoch_index in range(linear_probe_epochs):
        train_loss = _train_action_epoch_ssl_model(ssl_model, train_loader, linear_probe_optimizer, criterion, device)
        val_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, val_loader, device)
        linear_probe_best_val = max(linear_probe_best_val, val_accuracy)
        linear_probe_history.append(
            {"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_accuracy}
        )
        print(f"[Linear Probe] epoch {epoch_index + 1}/{linear_probe_epochs} loss={train_loss:.4f} val_acc={val_accuracy:.4f}")

    linear_probe_test_action_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, test_loader, device)
    linear_probe_success_rate, linear_probe_avg_steps = _evaluate_rollout(
        ssl_model.forward_action,
        data_cfg,
        number_of_episodes=int(split_cfg["test_episodes"]),
        seed=seed_value + 3,
        device=device,
    )
    if ood_enabled:
        linear_probe_ood_action_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, ood_test_loader, device)
        linear_probe_ood_success_rate, linear_probe_ood_avg_steps = _evaluate_rollout(
            ssl_model.forward_action,
            ood_data_cfg,
            number_of_episodes=int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])),
            seed=seed_value + 30,
            device=device,
        )
    else:
        linear_probe_ood_action_accuracy = None
        linear_probe_ood_success_rate = None
        linear_probe_ood_avg_steps = None

    # Fine-tune from SSL checkpoint
    for parameter in ssl_model.encoder.parameters():
        parameter.requires_grad = True
    fine_tune_optimizer = torch.optim.Adam(
        [
            {"params": ssl_model.encoder.parameters(), "lr": float(config["train"]["learning_rate_fine_tune_encoder"])},
            {"params": ssl_model.action_head.parameters(), "lr": float(config["train"]["learning_rate_fine_tune_head"])},
        ]
    )
    fine_tune_epochs = int(config["train"]["fine_tune_epochs"])
    fine_tune_best_val = 0.0
    fine_tune_history = []
    for epoch_index in range(fine_tune_epochs):
        train_loss = _train_action_epoch_ssl_model(ssl_model, train_loader, fine_tune_optimizer, criterion, device)
        val_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, val_loader, device)
        fine_tune_best_val = max(fine_tune_best_val, val_accuracy)
        fine_tune_history.append(
            {"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_accuracy}
        )
        print(f"[Fine-tune] epoch {epoch_index + 1}/{fine_tune_epochs} loss={train_loss:.4f} val_acc={val_accuracy:.4f}")

    fine_tune_test_action_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, test_loader, device)
    fine_tune_success_rate, fine_tune_avg_steps = _evaluate_rollout(
        ssl_model.forward_action,
        data_cfg,
        number_of_episodes=int(split_cfg["test_episodes"]),
        seed=seed_value + 3,
        device=device,
    )
    if ood_enabled:
        fine_tune_ood_action_accuracy = _evaluate_action_accuracy_ssl_model(ssl_model, ood_test_loader, device)
        fine_tune_ood_success_rate, fine_tune_ood_avg_steps = _evaluate_rollout(
            ssl_model.forward_action,
            ood_data_cfg,
            number_of_episodes=int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])),
            seed=seed_value + 30,
            device=device,
        )
    else:
        fine_tune_ood_action_accuracy = None
        fine_tune_ood_success_rate = None
        fine_tune_ood_avg_steps = None

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
        "supervised": {
            "best_validation_action_accuracy": supervised_best_val,
            "iid_test_action_accuracy": supervised_test_action_accuracy,
            "iid_test_success_rate": supervised_success_rate,
            "iid_test_avg_steps_to_goal": supervised_avg_steps,
            "ood_test_action_accuracy": supervised_ood_action_accuracy,
            "ood_test_success_rate": supervised_ood_success_rate,
            "ood_test_avg_steps_to_goal": supervised_ood_avg_steps,
            "history": supervised_history,
        },
        "rotation_ssl_linear_probe": {
            "best_validation_action_accuracy": linear_probe_best_val,
            "iid_test_action_accuracy": linear_probe_test_action_accuracy,
            "iid_test_success_rate": linear_probe_success_rate,
            "iid_test_avg_steps_to_goal": linear_probe_avg_steps,
            "ood_test_action_accuracy": linear_probe_ood_action_accuracy,
            "ood_test_success_rate": linear_probe_ood_success_rate,
            "ood_test_avg_steps_to_goal": linear_probe_ood_avg_steps,
            "history": linear_probe_history,
            "ssl_pretrain_history": ssl_history,
        },
        "rotation_ssl_fine_tune": {
            "best_validation_action_accuracy": fine_tune_best_val,
            "iid_test_action_accuracy": fine_tune_test_action_accuracy,
            "iid_test_success_rate": fine_tune_success_rate,
            "iid_test_avg_steps_to_goal": fine_tune_avg_steps,
            "ood_test_action_accuracy": fine_tune_ood_action_accuracy,
            "ood_test_success_rate": fine_tune_ood_success_rate,
            "ood_test_avg_steps_to_goal": fine_tune_ood_avg_steps,
            "history": fine_tune_history,
        },
        "elapsed_seconds": elapsed_seconds,
    }

    with open(run_directory / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(run_directory / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    torch.save(supervised_model.state_dict(), run_directory / "supervised_model_state_dict.pth")
    torch.save(ssl_model.state_dict(), run_directory / "rotation_ssl_model_state_dict.pth")

    table_path = repository_root / "reports" / "tables" / f"{config['experiment_name']}_iid_ood.md"
    table_text = (
        "# Supervised vs Rotation-SSL (IID and OOD Quickcheck)\n\n"
        f"- Run directory: `{run_directory}`\n"
        f"- Device: `{device}`\n\n"
        "| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Supervised end-to-end | {supervised_best_val:.4f} | {supervised_test_action_accuracy:.4f} | {supervised_success_rate:.4f} | {supervised_ood_action_accuracy if supervised_ood_action_accuracy is not None else float('nan'):.4f} | {supervised_ood_success_rate if supervised_ood_success_rate is not None else float('nan'):.4f} |\n"
        f"| Rotation-SSL + Linear Probe | {linear_probe_best_val:.4f} | {linear_probe_test_action_accuracy:.4f} | {linear_probe_success_rate:.4f} | {linear_probe_ood_action_accuracy if linear_probe_ood_action_accuracy is not None else float('nan'):.4f} | {linear_probe_ood_success_rate if linear_probe_ood_success_rate is not None else float('nan'):.4f} |\n"
        f"| Rotation-SSL + Fine-tune | {fine_tune_best_val:.4f} | {fine_tune_test_action_accuracy:.4f} | {fine_tune_success_rate:.4f} | {fine_tune_ood_action_accuracy if fine_tune_ood_action_accuracy is not None else float('nan'):.4f} | {fine_tune_ood_success_rate if fine_tune_ood_success_rate is not None else float('nan'):.4f} |\n"
    )
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write(table_text)

    print("\n=== Summary ===")
    print(f"Supervised test acc: {supervised_test_action_accuracy:.4f}, success: {supervised_success_rate:.4f}")
    print(f"SSL+LinearProbe test acc: {linear_probe_test_action_accuracy:.4f}, success: {linear_probe_success_rate:.4f}")
    print(f"SSL+FineTune test acc: {fine_tune_test_action_accuracy:.4f}, success: {fine_tune_success_rate:.4f}")
    if ood_enabled:
        print(f"Supervised OOD acc: {supervised_ood_action_accuracy:.4f}, success: {supervised_ood_success_rate:.4f}")
        print(f"SSL+LinearProbe OOD acc: {linear_probe_ood_action_accuracy:.4f}, success: {linear_probe_ood_success_rate:.4f}")
        print(f"SSL+FineTune OOD acc: {fine_tune_ood_action_accuracy:.4f}, success: {fine_tune_ood_success_rate:.4f}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Saved run artifacts to: {run_directory}")
    print(f"Updated table: {table_path}")


if __name__ == "__main__":
    main()
