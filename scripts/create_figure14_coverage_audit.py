#!/usr/bin/env python3
"""
Create a provenance-oriented coverage audit figure for the repaired HumanEval runs.
"""

import matplotlib.pyplot as plt
import numpy as np

from figure_benchmark_utils import (
    ROOT,
    OUTPUT_DIR,
    ensure_output_dir,
    evaluation_results_coverage,
    repair_report_summary,
)


OUTPUT_PATH = OUTPUT_DIR / "figure14_coverage_audit"


def main() -> None:
    ensure_output_dir()

    coverage_specs = [
        ("GPT-2 Small", ROOT / "data/results_gpt2/evaluation_results.json"),
        ("GPT-2 Medium", ROOT / "data/results_gpt2_medium/evaluation_results.json"),
        ("CodeGen", ROOT / "data/results_codegen/evaluation_results.json"),
        ("CodeGen-NL", ROOT / "data/results_codegen_nl/evaluation_results.json"),
        ("CodeGen-Multi", ROOT / "data/results_codegen_multi/evaluation_results.json"),
        ("CodeGen-Mono", ROOT / "data/results_codegen_mono/evaluation_results.json"),
    ]
    repair_specs = [
        ("CodeGen main", ROOT / "outputs/tables/codegen_main_repair_report.json"),
        ("CodeGen-NL", ROOT / "outputs/tables/codegen_nl_repair_report.json"),
        ("CodeGen-Multi", ROOT / "outputs/tables/codegen_multi_repair_report.json"),
        ("CodeGen-Mono", ROOT / "outputs/tables/codegen_mono_repair_report.json"),
    ]

    coverage_rows = [(label, evaluation_results_coverage(path)) for label, path in coverage_specs]
    repair_rows = [(label, repair_report_summary(path)) for label, path in repair_specs]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    x_values = np.arange(len(coverage_rows))
    expected = [row["expected_samples"] for _, row in coverage_rows]
    actual = [row["samples"] for _, row in coverage_rows]
    colors = ["#adb5bd", "#f08c00", "#1c7ed6", "#868e96", "#1971c2", "#2f9e44"]

    ax_left.bar(
        x_values,
        expected,
        width=0.62,
        color="#f1f3f5",
        edgecolor="#adb5bd",
        linewidth=1.5,
        label="Expected completions",
    )
    bars = ax_left.bar(
        x_values,
        actual,
        width=0.48,
        color=colors,
        label="Scored completions",
    )
    ax_left.set_xticks(x_values)
    ax_left.set_xticklabels([label for label, _ in coverage_rows], rotation=20, ha="right", fontsize=10)
    ax_left.set_ylabel("Completions scored", fontsize=11)
    ax_left.set_title("Scored vs Intended HumanEval Coverage", fontsize=14, fontweight="bold")
    ax_left.grid(axis="y", alpha=0.25, linestyle="--")
    ax_left.legend(framealpha=0.96)

    for bar, (_, row) in zip(bars, coverage_rows):
        delta = row["expected_samples"] - row["samples"]
        label = f"{row['samples']}"
        if delta:
            label += f"\n(-{delta})"
        ax_left.text(
            bar.get_x() + bar.get_width() / 2.0,
            row["samples"] + 120,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    repair_labels = [label for label, _ in repair_rows]
    repair_counts = [row["repair_count"] for _, row in repair_rows]
    y_values = np.arange(len(repair_rows))
    ax_right.barh(y_values, repair_counts, color=["#e67700", "#868e96", "#1971c2", "#2f9e44"], alpha=0.92)
    ax_right.set_yticks(y_values)
    ax_right.set_yticklabels(repair_labels, fontsize=10)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Tasks flagged for repair", fontsize=11)
    ax_right.set_title("Repair Burden Before the Final Sync", fontsize=14, fontweight="bold")
    ax_right.grid(axis="x", alpha=0.25, linestyle="--")

    for index, count in enumerate(repair_counts):
        ax_right.text(count + 0.4, index, str(count), va="center", fontsize=9.5)

    note = (
        "Post-sync state:\n"
        "- Ladder checkpoints are back to 16,400 completions.\n"
        "- Main CodeGen HumanEval still has one empty task\n"
        "  (HumanEval/129), leaving 16,300 scored completions."
    )
    ax_right.text(
        0.98,
        0.05,
        note,
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ced4da"},
    )

    fig.suptitle(
        "Coverage Audit: The Ladder Is Repaired, but Main CodeGen HumanEval Still Has One Empty Task",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
