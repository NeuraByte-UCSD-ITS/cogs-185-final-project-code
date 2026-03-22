import json
from pathlib import Path


def _latest_metrics(run_glob, repository_root: Path):
    run_dirs = sorted((repository_root / "experiments" / "runs").glob(run_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for pattern: {run_glob}")
    latest_run = max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
    with open(latest_run / "metrics.json", "r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return latest_run, metrics


def main():
    repository_root = Path(__file__).resolve().parents[2]

    three_way_table_path = repository_root / "reports" / "tables" / "three_way_largesplit_multiseed_summary.md"
    ff_lstm_table_path = repository_root / "reports" / "tables" / "ff_vs_lstm_largesplit_summary.md"
    ablation_table_path = repository_root / "reports" / "tables" / "supervised_lstm_ablation_sweep_summary.md"
    ablation_multiseed_table_path = repository_root / "reports" / "tables" / "supervised_lstm_bestcfg_multiseed_summary.md"
    four_way_table_path = repository_root / "reports" / "tables" / "four_way_ssl_compare_quickcheck_v1_summary.md"
    hyper_table_path = repository_root / "reports" / "tables" / "proposal_hyperparam_sweep_summary.md"
    linear_probe_table_path = repository_root / "reports" / "tables" / "four_way_linear_probe_summary.md"
    compute_table_path = repository_root / "reports" / "tables" / "four_way_compute_memory_summary.md"

    consolidated_table_path = repository_root / "reports" / "tables" / "final_results_consolidated.md"
    lines = [
        "# Final Results Consolidated Table",
        "",
        "## A) Core 3-way representation comparison (large split, multi-seed)",
        "",
        f"Source: `{three_way_table_path}`",
        "",
    ]
    lines.extend(three_way_table_path.read_text(encoding="utf-8").splitlines()[-8:])
    lines.extend(
        [
            "",
            "## B) Sequence architecture comparison (large split)",
            "",
            f"Source: `{ff_lstm_table_path}`",
            "",
        ]
    )
    lines.extend(ff_lstm_table_path.read_text(encoding="utf-8").splitlines()[-4:])
    lines.extend(
        [
            "",
            "## C) Supervised-LSTM ablation best setting",
            "",
            f"Source: `{ablation_table_path}` and `{ablation_multiseed_table_path}`",
            "",
            "- Best single-seed OOD from ablation sweep: fraction=1.00, crop=9, obstacles=1, OOD acc=0.3473",
            "- Multi-seed confirmation (seed 11/22/33): mean OOD acc=0.2952",
            "",
            "## D) Four-way representation run (adds predictive coding)",
            "",
            f"Source: `{four_way_table_path}`",
            "",
        ]
    )
    lines.extend(four_way_table_path.read_text(encoding="utf-8").splitlines()[-6:])
    lines.extend(
        [
            "",
            "## E) Proposal hyperparameter axes sweep",
            "",
            f"Source: `{hyper_table_path}`",
            "",
            "- Optimizer: SGD improved OOD acc over Adam in this sweep (OOD 0.3138 vs 0.2697).",
            "- Embedding: 128 improved IID and OOD over 64 (IID 0.3788 vs 0.3242, OOD 0.3019 vs 0.2697).",
            "- Conv depth: 5 did not improve OOD vs 3 in this sweep.",
            "- LSTM hidden size: 128 did not improve OOD vs 64 in this sweep.",
            "",
            "## F) Linear probe + compute/memory",
            "",
            f"Sources: `{linear_probe_table_path}` and `{compute_table_path}`",
            "",
        ]
    )
    probe_lines = linear_probe_table_path.read_text(encoding="utf-8").splitlines()
    compute_lines = compute_table_path.read_text(encoding="utf-8").splitlines()
    lines.extend([line for line in probe_lines if line.startswith("|")][:5])
    lines.extend([""])
    lines.extend([line for line in compute_lines if line.startswith("|")][:6])
    lines.extend([line for line in compute_lines if line.startswith("- End-to-end elapsed")])
    lines.extend(["", "## G) Proposal-scale anchor run (10k/2k/2k)", ""])
    fullscale_runs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    if fullscale_runs:
        latest_fullscale_for_table = max(fullscale_runs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
        with open(latest_fullscale_for_table / "metrics.json", "r", encoding="utf-8") as metrics_file:
            fullscale_metrics_for_table = json.load(metrics_file)
        lines.extend(
            [
                f"Source: `{latest_fullscale_for_table / 'metrics.json'}`",
                "",
                f"- Device: `{fullscale_metrics_for_table['device']}`",
                f"- Best validation action accuracy: `{float(fullscale_metrics_for_table['lstm']['best_validation_action_accuracy']):.4f}`",
                f"- IID action accuracy: `{float(fullscale_metrics_for_table['lstm']['iid_test_action_accuracy']):.4f}`",
                f"- OOD action accuracy: `{float(fullscale_metrics_for_table['lstm']['ood_test_action_accuracy']):.4f}`",
                f"- IID success: `{float(fullscale_metrics_for_table['lstm']['iid_test_success_rate']):.4f}`",
                f"- OOD success: `{float(fullscale_metrics_for_table['lstm']['ood_test_success_rate']):.4f}`",
                f"- Elapsed seconds: `{float(fullscale_metrics_for_table['elapsed_seconds']):.2f}`",
            ]
        )
    else:
        lines.append("- Full-scale run not found.")
    consolidated_table_path.write_text("\n".join(lines), encoding="utf-8")

    # check full-scale status
    fullscale_runs = sorted((repository_root / "experiments" / "runs").glob("supervised_lstm_fullscale_candidate_v1_*"))
    status_path = repository_root / "reports" / "tables" / "proposal_scale_anchor_status.md"
    if fullscale_runs:
        latest_fullscale = max(fullscale_runs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))
        with open(latest_fullscale / "metrics.json", "r", encoding="utf-8") as metrics_file:
            fullscale_metrics = json.load(metrics_file)
        lines = [
            "# Proposal-Scale Anchor Status",
            "",
            f"- Completed run found: `{latest_fullscale}`",
            f"- Device: `{fullscale_metrics['device']}`",
            f"- IID action accuracy: `{float(fullscale_metrics['lstm']['iid_test_action_accuracy']):.4f}`",
            f"- OOD action accuracy: `{float(fullscale_metrics['lstm']['ood_test_action_accuracy']):.4f}`",
            f"- IID success: `{float(fullscale_metrics['lstm']['iid_test_success_rate']):.4f}`",
            f"- OOD success: `{float(fullscale_metrics['lstm']['ood_test_success_rate']):.4f}`",
            f"- Elapsed seconds: `{float(fullscale_metrics['elapsed_seconds']):.2f}`",
        ]
    else:
        lines = [
            "# Proposal-Scale Anchor Status",
            "",
            "- Full 10k/2k/2k anchor run is not yet completed in sandbox.",
            "- Recommended to run on local MPS using:",
            "```bash",
            "source ~/venvs/cogs185-hmm/bin/activate",
            "cd project/cogs181-final/cogs-185",
            "python -m src.train.train_supervised_lstm_only --config experiments/configs/supervised_lstm_fullscale_candidate.json",
            "```",
            "- Fallback large-anchor config also available:",
            "  `experiments/configs/supervised_lstm_large_anchor_fallback.json`",
        ]
    status_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Updated table: {consolidated_table_path}")
    print(f"Updated table: {status_path}")


if __name__ == "__main__":
    main()
