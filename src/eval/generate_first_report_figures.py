import json
from pathlib import Path
from statistics import mean, stdev


def _timestamp_from_run_dir(run_dir: Path) -> int:
    try:
        return int(run_dir.name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _load_latest_multiseed_runs(repo_root: Path):
    runs_root = repo_root / "experiments" / "runs"
    metrics_paths = sorted(
        runs_root.glob("supervised_vs_rotation_vs_contrastive_largesplit_v1_seed*_*/metrics.json")
    )
    latest_by_seed = {}
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        seed_value = int(metrics["seed"])
        current_timestamp = _timestamp_from_run_dir(run_dir)
        if seed_value not in latest_by_seed:
            latest_by_seed[seed_value] = (current_timestamp, metrics_path)
            continue
        if current_timestamp > latest_by_seed[seed_value][0]:
            latest_by_seed[seed_value] = (current_timestamp, metrics_path)

    selected_paths = [path for _, path in sorted(latest_by_seed.values(), key=lambda item: item[0])]
    selected_metrics = []
    for selected_path in selected_paths:
        with open(selected_path, "r", encoding="utf-8") as handle:
            selected_metrics.append(json.load(handle))
    return selected_paths, selected_metrics


def _write_grouped_bar_svg(
    output_path: Path,
    labels,
    series_names,
    series_values,
    y_max,
    title,
):
    width = 980
    height = 520
    margin_left = 80
    margin_right = 30
    margin_top = 60
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_for_group(group_index):
        step = plot_width / max(1, len(labels))
        return margin_left + step * group_index + step * 0.5

    def y_for_value(value):
        return margin_top + plot_height * (1.0 - min(max(value / y_max, 0.0), 1.0))

    legend_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    bars_per_group = len(series_names)
    group_step = plot_width / max(1, len(labels))
    bar_width = min(45, (group_step * 0.7) / max(1, bars_per_group))

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

    # axes
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>'
    )
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>')

    # y ticks
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = y_for_value(tick)
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#444" stroke-width="1"/>')
        lines.append(
            f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#333">{tick:.1f}</text>'
        )
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#EEE" stroke-width="1"/>'
        )

    for group_index, label in enumerate(labels):
        x_center = x_for_group(group_index)
        offset_start = -(bars_per_group - 1) * bar_width / 2.0
        for series_index, series_name in enumerate(series_names):
            value = float(series_values[series_name][group_index])
            x = x_center + offset_start + series_index * bar_width - bar_width / 2.0
            y = y_for_value(value)
            h = margin_top + plot_height - y
            color = legend_colors[series_index % len(legend_colors)]
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width-3:.2f}" height="{h:.2f}" fill="{color}"/>')
            lines.append(
                f'<text x="{x + (bar_width-3)/2:.2f}" y="{y-6:.2f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#111">{value:.3f}</text>'
            )
        lines.append(
            f'<text x="{x_center:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )

    # legend
    legend_x = width - margin_right - 220
    legend_y = margin_top + 10
    for i, series_name in enumerate(series_names):
        color = legend_colors[i % len(legend_colors)]
        y = legend_y + i * 22
        lines.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 22}" y="{y+12}" font-size="12" font-family="Arial">{series_name}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_two_bar_svg(output_path: Path, labels, first_name, first_values, second_name, second_values, y_max, title):
    series_values = {first_name: first_values, second_name: second_values}
    _write_grouped_bar_svg(output_path, labels, [first_name, second_name], series_values, y_max, title)


def _summarize_three_way(repo_root: Path):
    selected_paths, metrics_list = _load_latest_multiseed_runs(repo_root)
    if not metrics_list:
        raise FileNotFoundError("No three-way large-split run metrics found.")

    method_order = [
        ("supervised", "Supervised"),
        ("rotation_ssl_fine_tune", "Rotation SSL"),
        ("contrastive_ssl_fine_tune", "Contrastive-lite SSL"),
    ]
    iid_values = {label: [] for _, label in method_order}
    ood_values = {label: [] for _, label in method_order}

    for metrics in metrics_list:
        for key, label in method_order:
            iid_values[label].append(float(metrics[key]["iid_test_action_accuracy"]))
            ood_values[label].append(float(metrics[key]["ood_test_action_accuracy"]))

    seeds = [int(metrics["seed"]) for metrics in metrics_list]
    summary_rows = []
    for _, label in method_order:
        iid_array = [float(v) for v in iid_values[label]]
        ood_array = [float(v) for v in ood_values[label]]
        summary_rows.append(
            {
                "method": label,
                "iid_mean": float(mean(iid_array)),
                "iid_std": float(stdev(iid_array)) if len(iid_array) > 1 else 0.0,
                "ood_mean": float(mean(ood_array)),
                "ood_std": float(stdev(ood_array)) if len(ood_array) > 1 else 0.0,
            }
        )

    figure_path = repo_root / "reports" / "figures" / "three_way_iid_ood_multiseed.svg"
    labels = [row["method"] for row in summary_rows]
    iid_means = [row["iid_mean"] for row in summary_rows]
    ood_means = [row["ood_mean"] for row in summary_rows]
    _write_two_bar_svg(
        output_path=figure_path,
        labels=labels,
        first_name="IID mean accuracy",
        first_values=iid_means,
        second_name="OOD mean accuracy",
        second_values=ood_means,
        y_max=0.55,
        title="Three-way comparison (large split, multi-seed)",
    )

    table_path = repo_root / "reports" / "tables" / "three_way_largesplit_multiseed_summary.md"
    table_lines = [
        "# Three-way Large-Split Multi-Seed Summary",
        "",
        "Selected run files (latest per seed):",
        "",
    ]
    table_lines.extend([f"- `{path}`" for path in selected_paths])
    table_lines.append("")
    table_lines.append(f"Seeds: {seeds}")
    table_lines.append("")
    table_lines.append("| Method | IID mean | IID std | OOD mean | OOD std |")
    table_lines.append("|---|---:|---:|---:|---:|")
    for row in summary_rows:
        table_lines.append(
            f"| {row['method']} | {row['iid_mean']:.4f} | {row['iid_std']:.4f} | {row['ood_mean']:.4f} | {row['ood_std']:.4f} |"
        )
    table_lines.append("")
    table_lines.append(f"Figure: `{figure_path}`")
    table_path.write_text("\n".join(table_lines), encoding="utf-8")

    return table_path, figure_path


def _summarize_ff_vs_lstm(repo_root: Path):
    runs_root = repo_root / "experiments" / "runs"
    run_dirs = sorted(runs_root.glob("supervised_ff_vs_lstm_largesplit_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No FF vs LSTM large-split runs found.")
    latest_run = max(run_dirs, key=_timestamp_from_run_dir)
    metrics_path = latest_run / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    ff = metrics["feedforward"]
    lstm = metrics["lstm"]

    figure_path = repo_root / "reports" / "figures" / "ff_vs_lstm_largesplit.svg"
    model_labels = ["Feedforward", "LSTM"]
    accuracies = [float(ff["test_action_accuracy"]), float(lstm["test_action_accuracy"])]
    success_rates = [float(ff["test_success_rate"]), float(lstm["test_success_rate"])]
    _write_two_bar_svg(
        output_path=figure_path,
        labels=model_labels,
        first_name="Test action accuracy",
        first_values=accuracies,
        second_name="Test success rate",
        second_values=success_rates,
        y_max=0.7,
        title="FF vs LSTM on large split",
    )

    table_path = repo_root / "reports" / "tables" / "ff_vs_lstm_largesplit_summary.md"
    table_text = (
        "# FF vs LSTM Large-Split Summary\n\n"
        f"- Metrics source: `{metrics_path}`\n"
        f"- Figure: `{figure_path}`\n\n"
        "| Model | Best Val Action Acc | Test Action Acc | Test Success Rate | Avg Steps to Goal |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Feedforward | {float(ff['best_validation_action_accuracy']):.4f} | {float(ff['test_action_accuracy']):.4f} | {float(ff['test_success_rate']):.4f} | {float(ff['test_avg_steps_to_goal']):.2f} |\n"
        f"| LSTM | {float(lstm['best_validation_action_accuracy']):.4f} | {float(lstm['test_action_accuracy']):.4f} | {float(lstm['test_success_rate']):.4f} | {float(lstm['test_avg_steps_to_goal']):.2f} |\n"
    )
    table_path.write_text(table_text, encoding="utf-8")

    return table_path, figure_path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    report_figure_dir = repo_root / "reports" / "figures"
    report_table_dir = repo_root / "reports" / "tables"
    report_figure_dir.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)

    three_way_table, three_way_figure = _summarize_three_way(repo_root)
    ff_lstm_table, ff_lstm_figure = _summarize_ff_vs_lstm(repo_root)

    print("Generated artifacts:")
    print(f"- {three_way_table}")
    print(f"- {three_way_figure}")
    print(f"- {ff_lstm_table}")
    print(f"- {ff_lstm_figure}")


if __name__ == "__main__":
    main()
