#!/usr/bin/env python3
"""
Create a benchmark-strictness cascade figure using pass@1 across base and strict settings.
"""

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


OUTPUT_PATH = OUTPUT_DIR / "figure13_strictness_cascade"
MODEL_ORDER = ["GPT-2 Small", "GPT-2 Medium", "CodeGen"]


def row_map(rows: list) -> dict:
    return {row["label"]: row["mean"] for row in rows}


def plot_family(ax, x_labels: list, values_by_model: dict, title: str) -> None:
    x_values = np.arange(len(x_labels))
    for model in MODEL_ORDER:
        values = values_by_model[model]
        ax.plot(
            x_values,
            values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=MAIN_MODEL_COLORS[model],
            label=model,
        )
        for x_pos, value in zip(x_values, values):
            label = f"{value:.2f}" if value < 10 else f"{value:.1f}"
            ax.text(
                x_pos,
                value + (0.05 if value < 1 else 0.4),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MAIN_MODEL_COLORS[model],
            )

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")


def main() -> None:
    ensure_output_dir()

    humaneval_pass1 = row_map(
        bootstrap_metric_rows(ROOT / "outputs/tables/bootstrap_significance.json", "pass@1", MAIN_MODEL_ALIASES)
    )
    mbpp_pass1 = row_map(
        bootstrap_metric_rows(ROOT / "outputs/tables/bootstrap_significance_mbpp_full.json", "pass@1", MAIN_MODEL_ALIASES)
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

    humaneval_family = {
        model: [humaneval_pass1[model], humaneval_plus[model], livecodebench[model]]
        for model in MODEL_ORDER
    }
    mbpp_family = {
        model: [mbpp_pass1[model], mbpp_plus[model], livecodebench[model]]
        for model in MODEL_ORDER
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.8))

    plot_family(
        ax_left,
        ["HumanEval", "HumanEval+", "LiveCodeBench"],
        humaneval_family,
        "HumanEval Family (pass@1)",
    )
    plot_family(
        ax_right,
        ["MBPP", "MBPP+", "LiveCodeBench"],
        mbpp_family,
        "MBPP Family (pass@1)",
    )

    ax_left.set_ylabel("Score (%) on symlog scale", fontsize=11)

    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Strictness Cascade: CodeGen Keeps the Only Nonzero Signal, but the Drop Is Still Severe",
        fontsize=15,
        fontweight="bold",
        y=1.08,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
