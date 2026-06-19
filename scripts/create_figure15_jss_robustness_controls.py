#!/usr/bin/env python3
"""
Create a four-panel robustness-control figure for the JSS retargeting package.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_benchmark_utils import OUTPUT_DIR, ROOT, ensure_output_dir


OUTPUT_PATH = OUTPUT_DIR / "figure15_jss_robustness_controls"

MODEL_ORDER_HE = ["GPT-2 Medium", "CodeGen Mono", "Qwen2.5-Coder"]
MODEL_ORDER_MBPP = ["CodeGen Mono", "Qwen2.5-Coder"]

PROMPT_CONDITIONS = [
    ("Canonical", "canonical", "#495057"),
    ("Signature", "signature_only", "#1971c2"),
    ("Comment+sig.", "comment_plus_signature", "#2f9e44"),
]

DECODING_CONDITIONS = [
    ("Low temp", "low_temp", "#0ca678"),
    ("Standard", "standard", "#495057"),
    ("High temp", "high_temp", "#e8590c"),
]


def load_aggregate(relative_path: str) -> list[dict]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["summaries"]


def model_from_label(label: str) -> str:
    if "gpt2_medium" in label:
        return "GPT-2 Medium"
    if "codegen_mono_350m" in label:
        return "CodeGen Mono"
    if "qwen25_coder_05b" in label:
        return "Qwen2.5-Coder"
    raise ValueError(f"Unknown model in label: {label}")


def prompt_condition_from_label(label: str) -> str:
    if "comment_plus_signature" in label:
        return "comment_plus_signature"
    if "signature_only" in label:
        return "signature_only"
    if "canonical" in label:
        return "canonical"
    raise ValueError(f"Unknown prompt condition in label: {label}")


def decoding_condition_from_label(label: str) -> str:
    if "low_temp" in label:
        return "low_temp"
    if "high_temp" in label:
        return "high_temp"
    if "standard" in label:
        return "standard"
    raise ValueError(f"Unknown decoding condition in label: {label}")


def index_rows(rows: list[dict], condition_parser) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        model = model_from_label(row["label"])
        condition = condition_parser(row["label"])
        indexed[(model, condition)] = row
    return indexed


def metric_percent(row: dict, metric: str) -> float:
    if metric == "sample_success":
        return 100.0 * row["success_samples"] / row["total_samples"]
    if metric == "task_coverage":
        return 100.0 * row["problems_with_success"] / row["tasks"]
    raise ValueError(metric)


def grouped_bars(
    ax,
    indexed: dict[tuple[str, str], dict],
    models: list[str],
    conditions: list[tuple[str, str, str]],
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    x_values = np.arange(len(models))
    width = 0.23 if len(conditions) == 3 else 0.32
    offsets = np.linspace(-width, width, len(conditions))

    for offset, (display, key, color) in zip(offsets, conditions):
        values = [metric_percent(indexed[(model, key)], metric) for model in models]
        bars = ax.bar(
            x_values + offset,
            values,
            width=width,
            label=display,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            if value >= 8.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.8,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#212529",
                )

    ax.set_xticks(x_values)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    ensure_output_dir()

    prompt_rows = load_aggregate("outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json")
    humaneval_decoding_rows = load_aggregate(
        "outputs/tables/jss_decoding_robustness_20s/aggregate_summary.json"
    )
    mbpp_decoding_rows = load_aggregate("outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json")

    prompt_index = index_rows(prompt_rows, prompt_condition_from_label)
    humaneval_decoding_index = index_rows(humaneval_decoding_rows, decoding_condition_from_label)
    mbpp_decoding_index = index_rows(mbpp_decoding_rows, decoding_condition_from_label)

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4))

    grouped_bars(
        axes[0, 0],
        prompt_index,
        MODEL_ORDER_HE,
        PROMPT_CONDITIONS,
        "sample_success",
        "HumanEval Prompt Format",
        "Successful samples (%)",
        (0, 76),
    )
    grouped_bars(
        axes[0, 1],
        humaneval_decoding_index,
        MODEL_ORDER_HE,
        DECODING_CONDITIONS,
        "sample_success",
        "HumanEval Decoding Temperature",
        "Successful samples (%)",
        (0, 62),
    )
    grouped_bars(
        axes[1, 0],
        mbpp_decoding_index,
        MODEL_ORDER_MBPP,
        DECODING_CONDITIONS,
        "sample_success",
        "MBPP Decoding: Sample Success",
        "Successful samples (%)",
        (0, 32),
    )
    grouped_bars(
        axes[1, 1],
        mbpp_decoding_index,
        MODEL_ORDER_MBPP,
        DECODING_CONDITIONS,
        "task_coverage",
        "MBPP Decoding: Distinct-Task Coverage",
        "Tasks with >=1 success (%)",
        (0, 60),
    )

    handles, labels = axes[0, 1].get_legend_handles_labels()
    prompt_handles, prompt_labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(prompt_handles, prompt_labels, loc="upper left", fontsize=8.5, frameon=False)
    axes[0, 1].legend(handles, labels, loc="upper left", fontsize=8.5, frameon=False)
    axes[1, 0].legend(handles, labels, loc="upper right", fontsize=8.5, frameon=False)
    axes[1, 1].legend(handles, labels, loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Robustness Controls: Prompt and Decoding Choices Change the Bottleneck",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Bars show local execution success. MBPP panels expose the sample-success vs distinct-task-coverage tradeoff.",
        ha="center",
        fontsize=10,
        color="#495057",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
