#!/usr/bin/env python3
"""
Create a two-panel figure for CodeGen ladder benchmark performance.
"""

import matplotlib.pyplot as plt
import numpy as np

from figure_benchmark_utils import (
    LADDER_COLORS,
    ROOT,
    OUTPUT_DIR,
    bootstrap_metric_rows,
    bootstrap_pairwise_rows,
    ensure_output_dir,
)


OUTPUT_PATH = OUTPUT_DIR / "figure12_codegen_ladder_benchmarks"
MODEL_ORDER = ["CodeGen-NL", "CodeGen-Multi", "CodeGen-Mono"]
MBPP_ALIASES = {
    "CodeGen-NL MBPP": "CodeGen-NL",
    "CodeGen-Multi MBPP": "CodeGen-Multi",
    "CodeGen-Mono MBPP": "CodeGen-Mono",
}


def row_lookup(rows: list) -> dict:
    return {row["label"]: row for row in rows}


def significance_text(rows: list) -> str:
    lines = []
    for row in rows:
        if row["model_a"] == "CodeGen-NL" and row["model_b"] == "CodeGen-Multi":
            lines.append(f"NL vs Multi: p={row['p_value']:.4f}")
        elif row["model_b"] == "CodeGen-Mono":
            lines.append(f"{row['model_a'].split('-')[-1]} vs Mono: p={row['p_value']:.4f}")
    return "\n".join(lines)


def plot_panel(ax, rows: list, pairwise_rows: list, title: str, ylim: tuple) -> None:
    lookup = row_lookup(rows)
    x_values = np.arange(len(MODEL_ORDER))
    means = [lookup[label]["mean"] for label in MODEL_ORDER]
    yerr_low = [lookup[label]["mean"] - lookup[label]["ci_low"] for label in MODEL_ORDER]
    yerr_high = [lookup[label]["ci_high"] - lookup[label]["mean"] for label in MODEL_ORDER]

    for x_pos, label in enumerate(MODEL_ORDER):
        ax.errorbar(
            x_pos,
            means[x_pos],
            yerr=[[yerr_low[x_pos]], [yerr_high[x_pos]]],
            fmt="o",
            color=LADDER_COLORS[label],
            ecolor=LADDER_COLORS[label],
            elinewidth=2.0,
            capsize=5,
            markersize=8,
            zorder=3,
        )
    ax.plot(
        x_values,
        means,
        linewidth=2.5,
        color="#343a40",
        alpha=0.8,
        zorder=2,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(["NL", "Multi", "Mono"], fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for x_pos, mean in enumerate(means):
        ax.text(
            x_pos,
            mean + 0.03 * ylim[1],
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.text(
        0.03,
        0.96,
        significance_text(pairwise_rows),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ced4da"},
    )


def main() -> None:
    ensure_output_dir()

    humaneval_rows = bootstrap_metric_rows(
        ROOT / "outputs/tables/bootstrap_significance_codegen_ladder.json",
        "success_rate",
        {},
    )
    humaneval_pairs = bootstrap_pairwise_rows(
        ROOT / "outputs/tables/bootstrap_significance_codegen_ladder.json",
        "success_rate",
        {},
    )
    mbpp_rows = bootstrap_metric_rows(
        ROOT / "outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json",
        "success_rate",
        MBPP_ALIASES,
    )
    mbpp_pairs = bootstrap_pairwise_rows(
        ROOT / "outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json",
        "success_rate",
        MBPP_ALIASES,
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.6))

    plot_panel(ax_left, humaneval_rows, humaneval_pairs, "HumanEval Ladder", (0, 44))
    plot_panel(ax_right, mbpp_rows, mbpp_pairs, "MBPP Ladder", (0, 3.6))

    ax_left.set_ylabel("Success rate (%)", fontsize=11)
    ax_right.set_ylabel("Success rate (%)", fontsize=11)

    fig.suptitle(
        "Within-Family CodeGen Ladder: Mono Is Best, but HumanEval Is Not Perfectly Monotone",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
