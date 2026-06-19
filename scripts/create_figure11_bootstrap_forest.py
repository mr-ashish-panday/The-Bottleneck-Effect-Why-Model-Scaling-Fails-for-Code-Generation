#!/usr/bin/env python3
"""
Create a forest-style confidence-interval figure for the main benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np

from figure_benchmark_utils import (
    MAIN_MODEL_ALIASES,
    MAIN_MODEL_COLORS,
    ROOT,
    OUTPUT_DIR,
    bootstrap_metric_rows,
    bootstrap_pairwise_rows,
    ensure_output_dir,
)


OUTPUT_PATH = OUTPUT_DIR / "figure11_bootstrap_forest"
PAIRWISE_ORDER = [
    "GPT-2 Small - GPT-2 Medium",
    "GPT-2 Small - CodeGen",
    "GPT-2 Medium - CodeGen",
]


def plot_estimates(ax, rows: list, title: str) -> None:
    rows = sorted(rows, key=lambda row: row["mean"])
    y_positions = np.arange(len(rows))
    for index, row in enumerate(rows):
        ax.errorbar(
            row["mean"],
            index,
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="o",
            color=MAIN_MODEL_COLORS[row["label"]],
            ecolor=MAIN_MODEL_COLORS[row["label"]],
            elinewidth=2.0,
            capsize=4,
            markersize=7,
        )
        ax.text(
            row["ci_high"] + 0.6,
            index,
            f"{row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")


def plot_pairwise(ax, rows: list, title: str) -> None:
    row_map = {row["label"]: row for row in rows}
    ordered = [row_map[label] for label in PAIRWISE_ORDER if label in row_map]
    y_positions = np.arange(len(ordered))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
    for index, row in enumerate(ordered):
        color = "#2f9e44" if row["ci_low"] > 0 or row["ci_high"] < 0 else "#495057"
        ax.errorbar(
            row["difference"],
            index,
            xerr=[[row["difference"] - row["ci_low"]], [row["ci_high"] - row["difference"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            markersize=7,
        )
        ax.text(
            row["ci_high"] + 0.9,
            index,
            f"p={row['p_value']:.4f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in ordered], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")


def main() -> None:
    ensure_output_dir()

    humaneval_rows = bootstrap_metric_rows(
        ROOT / "outputs/tables/bootstrap_significance.json",
        "success_rate",
        MAIN_MODEL_ALIASES,
    )
    humaneval_pairs = bootstrap_pairwise_rows(
        ROOT / "outputs/tables/bootstrap_significance.json",
        "success_rate",
        MAIN_MODEL_ALIASES,
    )
    mbpp_rows = bootstrap_metric_rows(
        ROOT / "outputs/tables/bootstrap_significance_mbpp_full.json",
        "success_rate",
        MAIN_MODEL_ALIASES,
    )
    mbpp_pairs = bootstrap_pairwise_rows(
        ROOT / "outputs/tables/bootstrap_significance_mbpp_full.json",
        "success_rate",
        MAIN_MODEL_ALIASES,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.05, 1.1]})

    plot_estimates(axes[0, 0], humaneval_rows, "HumanEval Model Estimates")
    plot_pairwise(axes[0, 1], humaneval_pairs, "HumanEval Pairwise Differences")
    plot_estimates(axes[1, 0], mbpp_rows, "MBPP Model Estimates")
    plot_pairwise(axes[1, 1], mbpp_pairs, "MBPP Pairwise Differences")

    axes[0, 0].set_xlabel("Success rate (%)", fontsize=11)
    axes[0, 1].set_xlabel("Difference in success rate (percentage points)", fontsize=11)
    axes[1, 0].set_xlabel("Success rate (%)", fontsize=11)
    axes[1, 1].set_xlabel("Difference in success rate (percentage points)", fontsize=11)

    axes[0, 0].set_xlim(-1, 43)
    axes[0, 1].set_xlim(-36, 4)
    axes[1, 0].set_xlim(-1, 10)
    axes[1, 1].set_xlim(-9, 1.5)

    fig.suptitle(
        "Bootstrap Intervals Make the Core Statistical Story Hard to Miss",
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
