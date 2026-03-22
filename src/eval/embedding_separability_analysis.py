import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.procedural_gridworld import GridworldDataConfig, generate_imitation_samples
from src.models.recurrent_policy import RecurrentPolicyNetwork


def _load_latest_fullscale(repository_root: Path):
    run_dirs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No fullscale run found. Provide --checkpoint and --config.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    return latest_run / "lstm_model_state_dict.pth", latest_run / "config.json"


def _sample_pair_similarities(embeddings: np.ndarray, labels: np.ndarray, number_of_pairs: int, same_label: bool, seed: int):
    random_state = np.random.RandomState(seed)
    label_to_indices = {}
    for index, label_value in enumerate(labels.tolist()):
        label_to_indices.setdefault(int(label_value), []).append(index)

    pairs = []
    if same_label:
        for _ in range(number_of_pairs * 3):
            label_value = random_state.choice(list(label_to_indices.keys()))
            if len(label_to_indices[label_value]) < 2:
                continue
            i, j = random_state.choice(label_to_indices[label_value], size=2, replace=False)
            pairs.append((i, j))
            if len(pairs) >= number_of_pairs:
                break
    else:
        label_values = list(label_to_indices.keys())
        for _ in range(number_of_pairs * 4):
            label_a, label_b = random_state.choice(label_values, size=2, replace=False)
            i = random_state.choice(label_to_indices[int(label_a)])
            j = random_state.choice(label_to_indices[int(label_b)])
            pairs.append((int(i), int(j)))
            if len(pairs) >= number_of_pairs:
                break

    if not pairs:
        return np.array([], dtype=np.float32)
    first = embeddings[[i for i, _ in pairs]]
    second = embeddings[[j for _, j in pairs]]
    first = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-8)
    second = second / np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-8)
    return np.sum(first * second, axis=1)


def _write_histogram_svg(output_path: Path, same_values: np.ndarray, diff_values: np.ndarray):
    width = 980
    height = 520
    margin_left = 70
    margin_right = 20
    margin_top = 60
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    bin_edges = np.linspace(-1.0, 1.0, 21)
    same_hist, _ = np.histogram(same_values, bins=bin_edges)
    diff_hist, _ = np.histogram(diff_values, bins=bin_edges)
    max_count = max(1, int(max(same_hist.max(), diff_hist.max())))

    def x_for_bin(bin_index: int):
        return margin_left + plot_width * (bin_index / (len(bin_edges) - 1))

    def y_for_count(count: int):
        return margin_top + plot_height * (1.0 - count / max_count)

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append('<text x="490" y="30" text-anchor="middle" font-size="20" font-family="Arial">Embedding Cosine Similarity Distribution</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>')

    bar_width = plot_width / (len(bin_edges) - 1)
    for index in range(len(bin_edges) - 1):
        same_count = int(same_hist[index])
        diff_count = int(diff_hist[index])
        x = x_for_bin(index)
        y_same = y_for_count(same_count)
        y_diff = y_for_count(diff_count)
        lines.append(f'<rect x="{x+1:.2f}" y="{y_same:.2f}" width="{bar_width/2-2:.2f}" height="{margin_top + plot_height - y_same:.2f}" fill="#4C78A8"/>')
        lines.append(f'<rect x="{x+bar_width/2+1:.2f}" y="{y_diff:.2f}" width="{bar_width/2-2:.2f}" height="{margin_top + plot_height - y_diff:.2f}" fill="#F58518"/>')

    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        x = margin_left + plot_width * ((tick + 1.0) / 2.0)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top+plot_height}" x2="{x:.2f}" y2="{margin_top+plot_height+5}" stroke="#333" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{margin_top+plot_height+20}" text-anchor="middle" font-size="12" font-family="Arial">{tick:.1f}</text>')

    lines.append('<rect x="760" y="75" width="14" height="14" fill="#4C78A8"/>')
    lines.append('<text x="782" y="87" font-size="12" font-family="Arial">same action pairs</text>')
    lines.append('<rect x="760" y="99" width="14" height="14" fill="#F58518"/>')
    lines.append('<text x="782" y="111" font-size="12" font-family="Arial">different action pairs</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--pairs", type=int, default=3000)
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

    data_cfg = GridworldDataConfig(
        grid_size=int(config["data"]["grid_size_train"]),
        obstacle_count=int(config["data"]["obstacle_count"]),
        observation_crop_size=int(config["data"]["observation_crop_size"]),
        episode_horizon_min=int(config["data"]["episode_horizon_min"]),
        episode_horizon_max=int(config["data"]["episode_horizon_max"]),
    )
    observation_array, action_array = generate_imitation_samples(number_of_episodes=200, data_config=data_cfg, seed=args.seed)
    if observation_array.shape[0] > args.samples:
        observation_array = observation_array[: args.samples]
        action_array = action_array[: args.samples]

    with torch.no_grad():
        observation_tensor = torch.from_numpy(observation_array).float().to(device)
        embedding_tensor = model.frame_encoder(observation_tensor)
    embeddings = embedding_tensor.detach().cpu().numpy()

    same_values = _sample_pair_similarities(embeddings, action_array, args.pairs, same_label=True, seed=args.seed)
    diff_values = _sample_pair_similarities(embeddings, action_array, args.pairs, same_label=False, seed=args.seed + 1)

    figure_path = repository_root / "reports" / "figures" / "embedding_separability_cosine_hist.svg"
    _write_histogram_svg(figure_path, same_values, diff_values)

    table_path = repository_root / "reports" / "tables" / "embedding_separability_summary.md"
    lines = [
        "# Embedding Separability Summary (Cosine Similarity)",
        "",
        f"- Source checkpoint: `{checkpoint_path}`",
        f"- Source config: `{config_path}`",
        f"- Device used for embedding extraction: `{device}`",
        "",
        "| Pair Type | Mean Cosine | Std Cosine | 25th Percentile | Median | 75th Percentile |",
        "|---|---:|---:|---:|---:|---:|",
        f"| same action | {float(np.mean(same_values)):.4f} | {float(np.std(same_values)):.4f} | {float(np.percentile(same_values, 25)):.4f} | {float(np.median(same_values)):.4f} | {float(np.percentile(same_values, 75)):.4f} |",
        f"| different action | {float(np.mean(diff_values)):.4f} | {float(np.std(diff_values)):.4f} | {float(np.percentile(diff_values, 25)):.4f} | {float(np.median(diff_values)):.4f} | {float(np.percentile(diff_values, 75)):.4f} |",
        "",
        f"Figure: `{figure_path}`",
    ]
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated table: {table_path}")
    print(f"Updated figure: {figure_path}")


if __name__ == "__main__":
    main()
