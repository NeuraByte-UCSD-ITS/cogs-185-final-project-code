from pathlib import Path


def main():
    repository_root = Path(__file__).resolve().parents[2]
    summary_path = repository_root / "reports" / "tables" / "supervised_lstm_ablation_sweep_summary.md"
    figure_path = repository_root / "reports" / "figures" / "supervised_lstm_ablation_ood_ranked.svg"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary table: {summary_path}")

    rows = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "Fraction" in line or "---" in line:
            continue
        columns = [column.strip() for column in line.split("|")[1:-1]]
        rows.append(
            {
                "label": f"f{columns[0]}-c{columns[2]}-o{columns[3]}",
                "ood_acc": float(columns[5]),
                "iid_acc": float(columns[4]),
            }
        )

    rows = sorted(rows, key=lambda row: row["ood_acc"], reverse=True)
    if not rows:
        raise ValueError("No rows parsed from ablation summary table.")

    width = 1120
    height = 620
    margin_left = 70
    margin_right = 20
    margin_top = 70
    margin_bottom = 180
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_max = max(0.4, max(row["ood_acc"] for row in rows) + 0.05)

    def y_for_value(value: float) -> float:
        return margin_top + plot_height * (1.0 - min(max(value / y_max, 0.0), 1.0))

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="22" font-family="Arial">'
        "Supervised LSTM Ablation: OOD Accuracy Ranking"
        "</text>"
    )
    lines.append(
        f'<text x="{width/2:.1f}" y="56" text-anchor="middle" font-size="13" font-family="Arial" fill="#444">'
        "label format = f{fraction}-c{crop}-o{obstacles}"
        "</text>"
    )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>'
    )
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>')

    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = y_for_value(tick)
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#444" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{tick:.1f}</text>')
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#EEE" stroke-width="1"/>'
        )

    number_of_rows = len(rows)
    step = plot_width / max(1, number_of_rows)
    bar_width = max(10, min(40, step * 0.55))
    for index, row in enumerate(rows):
        center_x = margin_left + (index + 0.5) * step
        x = center_x - bar_width / 2
        ood_y = y_for_value(row["ood_acc"])
        iid_y = y_for_value(row["iid_acc"])
        ood_h = margin_top + plot_height - ood_y
        iid_h = margin_top + plot_height - iid_y
        lines.append(f'<rect x="{x:.2f}" y="{iid_y:.2f}" width="{bar_width:.2f}" height="{iid_h:.2f}" fill="#A0CBE8"/>')
        lines.append(f'<rect x="{x+3:.2f}" y="{ood_y:.2f}" width="{bar_width-6:.2f}" height="{ood_h:.2f}" fill="#4C78A8"/>')
        lines.append(
            f'<text x="{center_x:.2f}" y="{ood_y-6:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{row["ood_acc"]:.3f}</text>'
        )
        lines.append(
            f'<text transform="translate({center_x:.2f},{margin_top + plot_height + 16}) rotate(60)" text-anchor="start" font-size="10" font-family="Arial">{row["label"]}</text>'
        )

    lines.append('<rect x="830" y="84" width="14" height="14" fill="#A0CBE8"/>')
    lines.append('<text x="852" y="96" font-size="12" font-family="Arial">IID accuracy</text>')
    lines.append('<rect x="830" y="108" width="14" height="14" fill="#4C78A8"/>')
    lines.append('<text x="852" y="120" font-size="12" font-family="Arial">OOD accuracy</text>')
    lines.append("</svg>")

    figure_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated figure: {figure_path}")


if __name__ == "__main__":
    main()
