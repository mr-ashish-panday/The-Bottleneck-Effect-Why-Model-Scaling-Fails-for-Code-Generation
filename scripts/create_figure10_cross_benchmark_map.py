#!/usr/bin/env python3
"""
Create a benchmark-overview figure spanning classical and strict benchmarks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_benchmark_utils import (
    MAIN_MODEL_ALIASES,
    MAIN_MODEL_COLORS,
    ROOT,
    OUTPUT_DIR,
    bootstrap_metric_rows,
    ensure_output_dir,
    load_livecodebench_pass_metrics,
    parse_evalplus_log,
)


OUTPUT_PATH = OUTPUT_DIR / "figure10_cross_benchmark_map"


def row_map(rows: list) -> dict:
    return {row["label"]: row["mean"] for row in rows}


def annotate_bars(ax, bars, decimals: int) -> None:
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            y = 0.03 * max(ax.get_ylim()[1], 1.0)
            label = "0.0"
        else:
            y = height + 0.02 * ax.get_ylim()[1]
            label = f"{height:.{decimals}f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=90,
            color="#212529",
        )


def main() -> None:
    ensure_output_dir()

    humaneval = row_map(
        bootstrap_metric_rows(
            ROOT / "outputs/tables/bootstrap_significance.json",
            "pass@1",
            MAIN_MODEL_ALIASES,
        )
    )
    mbpp = row_map(
        bootstrap_metric_rows(
            ROOT / "outputs/tables/bootstrap_significance_mbpp_full.json",
            "pass@1",
            MAIN_MODEL_ALIASES,
        )
    )

    humaneval_plus = {
        "GPT-2 Small": parse_evalplus_log(ROOT / "outputs/logs/evalplus_gpt2_humaneval.log")["plus_pass@1"],
        "GPT-2 Medium": parse_evalplus_log(ROOT / "outputs/logs/evalplus_gpt2_medium_humaneval.log")["plus_pass@1"],
        "CodeGen": parse_evalplus_log(ROOT / "outputs/logs/evalplus_codegen_humaneval.log")["plus_pass@1"],
    }
    mbpp_plus = {
        "GPT-2 Small": parse_evalplus_log(ROOT / "outputs/logs/evalplus_gpt2_mbppplus.log")["plus_pass@1"],
        "GPT-2 Medium": parse_evalplus_log(ROOT / "outputs/logs/evalplus_gpt2_medium_mbppplus.log")["plus_pass@1"],
        "CodeGen": parse_evalplus_log(ROOT / "outputs/logs/evalplus_codegen_mbppplus.log")["plus_pass@1"],
    }
    livecodebench = {
        "GPT-2 Small": load_livecodebench_pass_metrics(ROOT / "outputs/tables/livecodebench_gpt2_summary.json")["pass@1"],
        "GPT-2 Medium": load_livecodebench_pass_metrics(ROOT / "outputs/tables/livecodebench_gpt2_medium_summary.json")["pass@1"],
        "CodeGen": load_livecodebench_pass_metrics(ROOT / "outputs/tables/livecodebench_codegen_summary.json")["pass@1"],
    }

    model_order = ["GPT-2 Small", "GPT-2 Medium", "CodeGen"]
    classical_names = ["HumanEval", "MBPP"]
    strict_names = ["HumanEval+", "MBPP+", "LiveCodeBench"]
    classical_data = [humaneval, mbpp]
    strict_data = [humaneval_plus, mbpp_plus, livecodebench]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    width = 0.24
    x_left = np.arange(len(classical_names))
    x_right = np.arange(len(strict_names))

    for index, model in enumerate(model_order):
        offsets = (index - 1) * width
        left_values = [dataset[model] for dataset in classical_data]
        right_values = [dataset[model] for dataset in strict_data]
        bars_left = ax_left.bar(
            x_left + offsets,
            left_values,
            width=width,
            color=MAIN_MODEL_COLORS[model],
            label=model,
            alpha=0.92,
        )
        bars_right = ax_right.bar(
            x_right + offsets,
            right_values,
            width=width,
            color=MAIN_MODEL_COLORS[model],
            label=model,
            alpha=0.92,
        )
        annotate_bars(ax_left, bars_left, decimals=1)
        annotate_bars(ax_right, bars_right, decimals=2)

    ax_left.set_title("Classical Benchmarks (pass@1)", fontsize=14, fontweight="bold")
    ax_left.set_ylabel("Score (%)", fontsize=11)
    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels(classical_names, fontsize=10)
    ax_left.set_ylim(0, 42)
    ax_left.grid(axis="y", alpha=0.25, linestyle="--")

    ax_right.set_title("Strict and Contamination-Aware Benchmarks (pass@1)", fontsize=14, fontweight="bold")
    ax_right.set_xticks(x_right)
    ax_right.set_xticklabels(strict_names, fontsize=10)
    ax_right.set_ylim(0, 2.5)
    ax_right.grid(axis="y", alpha=0.25, linestyle="--")

    handles, labels = ax_left.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 1.02),
    )
    for text in legend.get_texts():
        text.set_fontsize(10)

    ax_left.text(
        0.98,
        0.95,
        "Main message:\nGPT-2 Small and Medium stay close,\nwhile CodeGen is the only strong\nsmall-model code baseline.",
        transform=ax_left.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ced4da"},
    )
    ax_right.text(
        0.98,
        0.95,
        "Strict tests preserve the ordering\nbut compress all three models toward zero.",
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ced4da"},
    )

    fig.suptitle(
        "Benchmark Map: Code-Specialized Pretraining Helps, but Stricter Evaluation Compresses Everyone",
        fontsize=15,
        fontweight="bold",
        y=1.07,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
