import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.procedural_gridworld import (
    GridworldDataConfig,
    GridworldImitationDataset,
    crop_egocentric_observation,
    generate_imitation_samples,
)
from src.env.procedural_gridworld_env import ProceduralGridworldEnv
from src.models.contrastive_ssl_policy import ContrastiveSslPolicyModel
from src.models.feedforward_policy import FeedforwardPolicyNetwork
from src.models.rotation_ssl_policy import RotationSslPolicyModel
from src.utils.device import get_device


def _set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _augment_for_ssl(observation_batch: torch.Tensor, augmentation_config: dict | None = None):
    augmentation_config = augmentation_config or {}
    max_pixel_shift = int(augmentation_config.get("max_shift_pixels", 2))
    brightness_jitter = float(augmentation_config.get("brightness_jitter", 0.15))
    noise_std = float(augmentation_config.get("noise_std", 0.03))
    augmented_batch = observation_batch.clone()
    batch_size = augmented_batch.size(0)
    for sample_index in range(batch_size):
        sample = augmented_batch[sample_index]
        shift_y = int(torch.randint(-max_pixel_shift, max_pixel_shift + 1, (1,), device=sample.device).item())
        shift_x = int(torch.randint(-max_pixel_shift, max_pixel_shift + 1, (1,), device=sample.device).item())
        sample = torch.roll(sample, shifts=(shift_y, shift_x), dims=(1, 2))
        brightness_scale = (1.0 - brightness_jitter) + (2.0 * brightness_jitter) * torch.rand(1, device=sample.device).item()
        sample = sample * brightness_scale
        sample = sample + noise_std * torch.randn_like(sample)
        augmented_batch[sample_index] = torch.clamp(sample, 0.0, 1.0)
    return augmented_batch


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


def _train_rotation_ssl_epoch(model, loader, optimizer, criterion, device, augmentation_config: dict | None = None):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for observation_batch, _ in loader:
        observation_batch = observation_batch.to(device)
        augmented_batch = _augment_for_ssl(observation_batch, augmentation_config=augmentation_config)
        rotated_batch, rotation_targets = _build_rotation_batch(augmented_batch)
        optimizer.zero_grad()
        logits = model.forward_rotation(rotated_batch)
        loss_value = criterion(logits, rotation_targets)
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item() * observation_batch.size(0)
        number_of_samples += observation_batch.size(0)
    return running_loss / max(1, number_of_samples)


def _nt_xent_loss(projection_view_1: torch.Tensor, projection_view_2: torch.Tensor, temperature: float):
    projection_view_1 = F.normalize(projection_view_1, dim=1)
    projection_view_2 = F.normalize(projection_view_2, dim=1)
    representations = torch.cat([projection_view_1, projection_view_2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    batch_size = projection_view_1.size(0)
    total_size = 2 * batch_size
    diagonal_mask = torch.eye(total_size, dtype=torch.bool, device=similarity_matrix.device)
    similarity_matrix = similarity_matrix.masked_fill(diagonal_mask, -1e9)

    positive_indices = torch.arange(total_size, device=similarity_matrix.device)
    positive_indices = (positive_indices + batch_size) % total_size
    return F.cross_entropy(similarity_matrix, positive_indices)


def _train_contrastive_ssl_epoch(
    model,
    loader,
    optimizer,
    device,
    temperature: float,
    augmentation_config: dict | None = None,
):
    model.train()
    running_loss = 0.0
    number_of_samples = 0
    for observation_batch, _ in loader:
        observation_batch = observation_batch.to(device)
        view_1 = _augment_for_ssl(observation_batch, augmentation_config=augmentation_config)
        view_2 = _augment_for_ssl(observation_batch, augmentation_config=augmentation_config)
        projection_1 = model.forward_projection(view_1)
        projection_2 = model.forward_projection(view_2)
        loss_value = _nt_xent_loss(projection_1, projection_2, temperature)
        optimizer.zero_grad()
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
            cropped = crop_egocentric_observation(
                full_observation,
                environment.agent_pos,
                environment.cell_size,
                data_config.observation_crop_size,
            )
            input_tensor = torch.from_numpy(cropped).unsqueeze(0).float().to(device)
            logits = model_predict_fn(input_tensor)
            action = int(torch.argmax(logits, dim=1).item())
            full_observation, reward_value, done, _ = environment.step(action)
            steps_taken += 1
            if done and reward_value > 0:
                successful_episodes += 1
        steps_per_episode.append(steps_taken)
    return successful_episodes / max(1, number_of_episodes), float(np.mean(steps_per_episode))


def _run_fine_tune_phase(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    fine_tune_epochs: int,
    learning_rate_encoder: float,
    learning_rate_head: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": learning_rate_encoder},
            {"params": model.action_head.parameters(), "lr": learning_rate_head},
        ]
    )
    best_val = 0.0
    history = []
    for epoch_index in range(fine_tune_epochs):
        train_loss = _train_action_epoch_ssl_model(model, train_loader, optimizer, criterion, device)
        val_acc = _evaluate_action_accuracy_ssl_model(model, val_loader, device)
        best_val = max(best_val, val_acc)
        history.append({"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_acc})
        print(f"[Fine-tune] epoch {epoch_index + 1}/{fine_tune_epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")
    test_acc = _evaluate_action_accuracy_ssl_model(model, test_loader, device)
    return best_val, test_acc, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/supervised_vs_rotation_vs_contrastive_quickcheck.json",
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
    augmentation_config = config.get("ssl_augmentation", {})
    train_x, train_y = generate_imitation_samples(int(split_cfg["train_episodes"]), data_cfg, seed_value)
    val_x, val_y = generate_imitation_samples(int(split_cfg["val_episodes"]), data_cfg, seed_value + 1)
    test_x, test_y = generate_imitation_samples(int(split_cfg["test_episodes"]), data_cfg, seed_value + 2)

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(GridworldImitationDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GridworldImitationDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GridworldImitationDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

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
        ood_x, ood_y = generate_imitation_samples(int(ood_cfg.get("test_episodes", split_cfg["test_episodes"])), ood_data_cfg, seed_value + 20)
        ood_loader = DataLoader(GridworldImitationDataset(ood_x, ood_y), batch_size=batch_size, shuffle=False)
    else:
        ood_data_cfg = None
        ood_loader = None

    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Train/Val/Test samples: {len(train_y)}/{len(val_y)}/{len(test_y)}")
    if ood_enabled:
        print(f"OOD test samples: {len(ood_y)} (grid={ood_data_cfg.grid_size}, obstacles={ood_data_cfg.obstacle_count})")

    start_time = time.time()

    # Supervised baseline
    supervised_model = FeedforwardPolicyNetwork(input_channels=3, number_of_actions=4, embedding_size=int(config["model"]["embedding_size"])).to(device)
    criterion = nn.CrossEntropyLoss()
    supervised_optimizer = torch.optim.Adam(supervised_model.parameters(), lr=float(config["train"]["learning_rate_supervised"]))
    supervised_best_val = 0.0
    supervised_history = []
    for epoch_index in range(int(config["train"]["supervised_epochs"])):
        train_loss = _train_supervised_epoch(supervised_model, train_loader, supervised_optimizer, criterion, device)
        val_acc = _evaluate_action_accuracy_supervised(supervised_model, val_loader, device)
        supervised_best_val = max(supervised_best_val, val_acc)
        supervised_history.append({"epoch": epoch_index + 1, "train_loss": train_loss, "val_action_accuracy": val_acc})
        print(f"[Supervised] epoch {epoch_index + 1}/{int(config['train']['supervised_epochs'])} loss={train_loss:.4f} val_acc={val_acc:.4f}")
    supervised_iid_acc = _evaluate_action_accuracy_supervised(supervised_model, test_loader, device)
    supervised_iid_success, supervised_iid_steps = _evaluate_rollout(supervised_model, data_cfg, int(split_cfg["test_episodes"]), seed_value + 3, device)
    if ood_enabled:
        supervised_ood_acc = _evaluate_action_accuracy_supervised(supervised_model, ood_loader, device)
        supervised_ood_success, supervised_ood_steps = _evaluate_rollout(supervised_model, ood_data_cfg, int(ood_cfg["test_episodes"]), seed_value + 30, device)
    else:
        supervised_ood_acc = None
        supervised_ood_success = None
        supervised_ood_steps = None

    # Rotation SSL + fine-tune
    rotation_model = RotationSslPolicyModel(
        input_channels=3,
        embedding_size=int(config["model"]["embedding_size"]),
        number_of_actions=4,
    ).to(device)
    rotation_ssl_optimizer = torch.optim.Adam(
        list(rotation_model.encoder.parameters()) + list(rotation_model.rotation_head.parameters()),
        lr=float(config["train"]["learning_rate_ssl"]),
    )
    rotation_ssl_history = []
    for epoch_index in range(int(config["train"]["rotation_ssl_pretrain_epochs"])):
        ssl_loss = _train_rotation_ssl_epoch(
            rotation_model,
            train_loader,
            rotation_ssl_optimizer,
            criterion,
            device,
            augmentation_config=augmentation_config,
        )
        rotation_ssl_history.append({"epoch": epoch_index + 1, "ssl_loss": ssl_loss})
        print(f"[Rotation SSL] epoch {epoch_index + 1}/{int(config['train']['rotation_ssl_pretrain_epochs'])} loss={ssl_loss:.4f}")
    rotation_best_val, rotation_iid_acc, rotation_ft_history = _run_fine_tune_phase(
        rotation_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        fine_tune_epochs=int(config["train"]["fine_tune_epochs"]),
        learning_rate_encoder=float(config["train"]["learning_rate_fine_tune_encoder"]),
        learning_rate_head=float(config["train"]["learning_rate_fine_tune_head"]),
    )
    rotation_iid_success, rotation_iid_steps = _evaluate_rollout(rotation_model.forward_action, data_cfg, int(split_cfg["test_episodes"]), seed_value + 3, device)
    if ood_enabled:
        rotation_ood_acc = _evaluate_action_accuracy_ssl_model(rotation_model, ood_loader, device)
        rotation_ood_success, rotation_ood_steps = _evaluate_rollout(rotation_model.forward_action, ood_data_cfg, int(ood_cfg["test_episodes"]), seed_value + 30, device)
    else:
        rotation_ood_acc = None
        rotation_ood_success = None
        rotation_ood_steps = None

    # Contrastive-lite SSL + fine-tune
    contrastive_model = ContrastiveSslPolicyModel(
        input_channels=3,
        embedding_size=int(config["model"]["embedding_size"]),
        projection_size=int(config["model"]["projection_size"]),
        number_of_actions=4,
    ).to(device)
    contrastive_ssl_optimizer = torch.optim.Adam(
        list(contrastive_model.encoder.parameters()) + list(contrastive_model.projection_head.parameters()),
        lr=float(config["train"]["learning_rate_ssl"]),
    )
    contrastive_history = []
    for epoch_index in range(int(config["train"]["contrastive_ssl_pretrain_epochs"])):
        contrastive_loss = _train_contrastive_ssl_epoch(
            contrastive_model,
            train_loader,
            contrastive_ssl_optimizer,
            device,
            temperature=float(config["model"]["contrastive_temperature"]),
            augmentation_config=augmentation_config,
        )
        contrastive_history.append({"epoch": epoch_index + 1, "contrastive_loss": contrastive_loss})
        print(f"[Contrastive SSL] epoch {epoch_index + 1}/{int(config['train']['contrastive_ssl_pretrain_epochs'])} loss={contrastive_loss:.4f}")
    contrastive_best_val, contrastive_iid_acc, contrastive_ft_history = _run_fine_tune_phase(
        contrastive_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        fine_tune_epochs=int(config["train"]["fine_tune_epochs"]),
        learning_rate_encoder=float(config["train"]["learning_rate_fine_tune_encoder"]),
        learning_rate_head=float(config["train"]["learning_rate_fine_tune_head"]),
    )
    contrastive_iid_success, contrastive_iid_steps = _evaluate_rollout(contrastive_model.forward_action, data_cfg, int(split_cfg["test_episodes"]), seed_value + 3, device)
    if ood_enabled:
        contrastive_ood_acc = _evaluate_action_accuracy_ssl_model(contrastive_model, ood_loader, device)
        contrastive_ood_success, contrastive_ood_steps = _evaluate_rollout(contrastive_model.forward_action, ood_data_cfg, int(ood_cfg["test_episodes"]), seed_value + 30, device)
    else:
        contrastive_ood_acc = None
        contrastive_ood_success = None
        contrastive_ood_steps = None

    elapsed_seconds = time.time() - start_time
    run_directory = repository_root / "experiments" / "runs" / f"{config['experiment_name']}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment_name": config["experiment_name"],
        "seed": seed_value,
        "device": str(device),
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
        "test_samples": int(len(test_y)),
        "supervised": {
            "best_validation_action_accuracy": supervised_best_val,
            "iid_test_action_accuracy": supervised_iid_acc,
            "iid_test_success_rate": supervised_iid_success,
            "iid_test_avg_steps_to_goal": supervised_iid_steps,
            "ood_test_action_accuracy": supervised_ood_acc,
            "ood_test_success_rate": supervised_ood_success,
            "ood_test_avg_steps_to_goal": supervised_ood_steps,
            "history": supervised_history,
        },
        "rotation_ssl_fine_tune": {
            "best_validation_action_accuracy": rotation_best_val,
            "iid_test_action_accuracy": rotation_iid_acc,
            "iid_test_success_rate": rotation_iid_success,
            "iid_test_avg_steps_to_goal": rotation_iid_steps,
            "ood_test_action_accuracy": rotation_ood_acc,
            "ood_test_success_rate": rotation_ood_success,
            "ood_test_avg_steps_to_goal": rotation_ood_steps,
            "ssl_pretrain_history": rotation_ssl_history,
            "fine_tune_history": rotation_ft_history,
        },
        "contrastive_ssl_fine_tune": {
            "best_validation_action_accuracy": contrastive_best_val,
            "iid_test_action_accuracy": contrastive_iid_acc,
            "iid_test_success_rate": contrastive_iid_success,
            "iid_test_avg_steps_to_goal": contrastive_iid_steps,
            "ood_test_action_accuracy": contrastive_ood_acc,
            "ood_test_success_rate": contrastive_ood_success,
            "ood_test_avg_steps_to_goal": contrastive_ood_steps,
            "ssl_pretrain_history": contrastive_history,
            "fine_tune_history": contrastive_ft_history,
        },
        "elapsed_seconds": elapsed_seconds,
    }

    with open(run_directory / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(run_directory / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    torch.save(supervised_model.state_dict(), run_directory / "supervised_model_state_dict.pth")
    torch.save(rotation_model.state_dict(), run_directory / "rotation_ssl_model_state_dict.pth")
    torch.save(contrastive_model.state_dict(), run_directory / "contrastive_ssl_model_state_dict.pth")

    table_path = repository_root / "reports" / "tables" / f"{config['experiment_name']}_iid_ood.md"
    table_text = (
        "# Supervised vs Rotation-SSL vs Contrastive-lite (IID and OOD Quickcheck)\n\n"
        f"- Run directory: `{run_directory}`\n"
        f"- Device: `{device}`\n\n"
        "| Method | Best Val Action Acc | IID Test Acc | IID Success | OOD Test Acc | OOD Success |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Supervised end-to-end | {supervised_best_val:.4f} | {supervised_iid_acc:.4f} | {supervised_iid_success:.4f} | {supervised_ood_acc if supervised_ood_acc is not None else float('nan'):.4f} | {supervised_ood_success if supervised_ood_success is not None else float('nan'):.4f} |\n"
        f"| Rotation-SSL + Fine-tune | {rotation_best_val:.4f} | {rotation_iid_acc:.4f} | {rotation_iid_success:.4f} | {rotation_ood_acc if rotation_ood_acc is not None else float('nan'):.4f} | {rotation_ood_success if rotation_ood_success is not None else float('nan'):.4f} |\n"
        f"| Contrastive-lite + Fine-tune | {contrastive_best_val:.4f} | {contrastive_iid_acc:.4f} | {contrastive_iid_success:.4f} | {contrastive_ood_acc if contrastive_ood_acc is not None else float('nan'):.4f} | {contrastive_ood_success if contrastive_ood_success is not None else float('nan'):.4f} |\n"
    )
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write(table_text)

    print("\n=== Summary ===")
    print(f"Supervised IID acc: {supervised_iid_acc:.4f}, OOD acc: {supervised_ood_acc:.4f}")
    print(f"Rotation SSL IID acc: {rotation_iid_acc:.4f}, OOD acc: {rotation_ood_acc:.4f}")
    print(f"Contrastive SSL IID acc: {contrastive_iid_acc:.4f}, OOD acc: {contrastive_ood_acc:.4f}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Saved run artifacts to: {run_directory}")
    print(f"Updated table: {table_path}")


if __name__ == "__main__":
    main()
