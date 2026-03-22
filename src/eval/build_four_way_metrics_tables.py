import json
from pathlib import Path


def main():
    repository_root = Path(__file__).resolve().parents[2]
    run_dirs = sorted((repository_root / "experiments" / "runs").glob("four_way_ssl_compare_quickcheck_v1_*"))
    if not run_dirs:
        raise FileNotFoundError("No four_way_ssl_compare_quickcheck_v1 run directory found.")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    metrics_path = latest_run / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    compute_table_path = repository_root / "reports" / "tables" / "four_way_compute_memory_summary.md"
    lines = [
        "# Four-way Compute and Memory Summary",
        "",
        f"- Metrics source: `{metrics_path}`",
        "",
        "| Method | Parameter Count | Parameter Memory (MB) |",
        "|---|---:|---:|",
    ]
    for method_key, label in [
        ("supervised", "Supervised"),
        ("rotation", "Rotation SSL"),
        ("contrastive", "Contrastive SSL"),
        ("predictive", "Predictive SSL"),
    ]:
        lines.append(
            f"| {label} | {int(metrics['compute']['parameter_count'][method_key])} | {float(metrics['compute']['parameter_memory_mb'][method_key]):.3f} |"
        )
    lines.append("")
    lines.append(f"- End-to-end elapsed seconds (whole four-way run): `{float(metrics['compute']['elapsed_seconds']):.2f}`")
    compute_table_path.write_text("\n".join(lines), encoding="utf-8")

    linear_probe_table_path = repository_root / "reports" / "tables" / "four_way_linear_probe_summary.md"
    probe_lines = [
        "# Four-way Linear Probe Summary",
        "",
        f"- Metrics source: `{metrics_path}`",
        "",
        "| Method | Linear Probe Val Acc | Linear Probe IID Acc |",
        "|---|---:|---:|",
        f"| Rotation SSL | {float(metrics['rotation_ssl_fine_tune']['linear_probe_val_action_accuracy']):.4f} | {float(metrics['rotation_ssl_fine_tune']['linear_probe_iid_action_accuracy']):.4f} |",
        f"| Contrastive SSL | {float(metrics['contrastive_ssl_fine_tune']['linear_probe_val_action_accuracy']):.4f} | {float(metrics['contrastive_ssl_fine_tune']['linear_probe_iid_action_accuracy']):.4f} |",
        f"| Predictive SSL | {float(metrics['predictive_ssl_fine_tune']['linear_probe_val_action_accuracy']):.4f} | {float(metrics['predictive_ssl_fine_tune']['linear_probe_iid_action_accuracy']):.4f} |",
    ]
    linear_probe_table_path.write_text("\n".join(probe_lines), encoding="utf-8")

    print(f"Updated table: {compute_table_path}")
    print(f"Updated table: {linear_probe_table_path}")


if __name__ == "__main__":
    main()
