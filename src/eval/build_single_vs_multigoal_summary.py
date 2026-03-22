import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _latest_run_path(runs_root: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(runs_root.glob(f"{prefix}_*"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _single_row(run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    section = metrics.get("lstm", {})
    return {
        "branch": "single-goal",
        "run": run_dir.name,
        "device": metrics.get("device", "N/A"),
        "iid_action_acc": section.get("iid_test_action_accuracy"),
        "ood_action_acc": section.get("ood_test_action_accuracy"),
        "iid_success": section.get("iid_test_success_rate"),
        "ood_success": section.get("ood_test_success_rate"),
        "iid_goal_completion": None,
        "ood_goal_completion": None,
        "elapsed_seconds": metrics.get("elapsed_seconds"),
    }


def _multi_row(run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    section = metrics.get("lstm_multigoal", {})
    return {
        "branch": "multi-goal",
        "run": run_dir.name,
        "device": metrics.get("device", "N/A"),
        "iid_action_acc": section.get("iid_test_action_accuracy"),
        "ood_action_acc": section.get("ood_test_action_accuracy"),
        "iid_success": section.get("iid_full_success_rate"),
        "ood_success": section.get("ood_full_success_rate"),
        "iid_goal_completion": section.get("iid_mean_goal_completion_ratio"),
        "ood_goal_completion": section.get("ood_mean_goal_completion_ratio"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
    }


def _to_markdown(rows):
    lines = []
    lines.append("# Single vs Multi-Goal Checkpoint Summary")
    lines.append("")
    lines.append("| branch | run | device | IID action acc | OOD action acc | IID success | OOD success | IID goal completion | OOD goal completion | elapsed sec |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["branch"],
                    row["run"],
                    row["device"],
                    _fmt(row["iid_action_acc"]),
                    _fmt(row["ood_action_acc"]),
                    _fmt(row["iid_success"]),
                    _fmt(row["ood_success"]),
                    _fmt(row["iid_goal_completion"]),
                    _fmt(row["ood_goal_completion"]),
                    _fmt(row["elapsed_seconds"], digits=2),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- single-goal success is target reached rate in single-goal rollout evaluation.")
    lines.append("- multi-goal success is full sequence completion rate (all goals reached).")
    lines.append("- multi-goal goal-completion ratio is partial progress metric (0 to 1).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", type=str, default="")
    parser.add_argument("--multi-run", type=str, default="")
    parser.add_argument("--output", type=str, default="reports/tables/single_vs_multigoal_summary.md")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    runs_root = repository_root / "experiments" / "runs"

    if args.single_run:
        single_run_dir = repository_root / args.single_run
    else:
        single_run_dir = _latest_run_path(runs_root, "supervised_lstm_fullscale_candidate_v1")
        if single_run_dir is None:
            single_run_dir = _latest_run_path(runs_root, "supervised_lstm_ablation_base_v1")

    if args.multi_run:
        multi_run_dir = repository_root / args.multi_run
    else:
        multi_run_dir = _latest_run_path(runs_root, "supervised_lstm_multigoal")

    if single_run_dir is None or not (single_run_dir / "metrics.json").exists():
        raise FileNotFoundError("single-goal run not found. Provide --single-run with a run directory containing metrics.json")
    if multi_run_dir is None or not (multi_run_dir / "metrics.json").exists():
        raise FileNotFoundError("multi-goal run not found. Provide --multi-run with a run directory containing metrics.json")

    single_metrics = _load_json(single_run_dir / "metrics.json")
    multi_metrics = _load_json(multi_run_dir / "metrics.json")

    rows = [_single_row(single_run_dir, single_metrics), _multi_row(multi_run_dir, multi_metrics)]

    output_path = repository_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(_to_markdown(rows) + "\n")

    print(f"Wrote comparison table: {output_path}")
    print(f"Single run: {single_run_dir}")
    print(f"Multi run: {multi_run_dir}")


if __name__ == "__main__":
    main()
