#!/usr/bin/env python3
"""
Create a steering-specificity figure with target responses and matched controls.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOP5_PATH = ROOT / "data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json"
CONTROLS_PATH = ROOT / "data/results_gpt2_medium/ablation/activation_steering_controls_10x5_summary.json"
OUTPUT_PATH = ROOT / "outputs/figures/figure6_steering_controls"


SUCCESS_COLOR = "#2f9e44"
SYNTAX_COLOR = "#c92a2a"
RUNTIME_COLOR = "#1c7ed6"
CONTROL_COLOR = "#495057"
TARGET_COLOR = "#f08c00"


def metric(entry: dict, name: str) -> float:
    if "category_percentages" in entry:
        return float(entry["category_percentages"].get(name, 0.0))
    return float(entry.get(name, 0.0))


def main() -> None:
    with TOP5_PATH.open("r", encoding="utf-8") as handle:
        top5_report = json.load(handle)
    with CONTROLS_PATH.open("r", encoding="utf-8") as handle:
        controls_report = json.load(handle)

    baseline = top5_report["baseline"]
    target_results = sorted(top5_report["results"], key=lambda item: float(item["alpha"]))

    x_values = [0.0] + [float(item["alpha"]) for item in target_results]
    success = [metric(baseline, "success_pct")] + [metric(item, "success_pct") for item in target_results]
    syntax = [metric(baseline, "syntax_error_pct")] + [metric(item, "syntax_error_pct") for item in target_results]
    runtime = [metric(baseline, "runtime_error_pct")] + [metric(item, "runtime_error_pct") for item in target_results]

    random_controls = controls_report["random_controls"]
    random_success = [float(item["success_pct"]) for item in random_controls]
    random_mean = float(controls_report["random_control_summary"]["success_mean"])
    target_plus_two = next(item for item in controls_report["target_conditions"] if float(item["alpha"]) == 2.0)
    target_minus_two = next(item for item in controls_report["target_conditions"] if float(item["alpha"]) == -2.0)
    specificity = controls_report["specificity_test"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    ax_left.plot(x_values, success, marker="o", linewidth=2.6, markersize=7, color=SUCCESS_COLOR, label="Success")
    ax_left.plot(x_values, syntax, marker="s", linewidth=2.2, markersize=6, color=SYNTAX_COLOR, label="Syntax")
    ax_left.plot(x_values, runtime, marker="^", linewidth=2.2, markersize=7, color=RUNTIME_COLOR, label="Runtime")
    ax_left.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

    for x_pos, y_pos in zip(x_values, success):
        ax_left.annotate(
            f"{y_pos:.0f}",
            (x_pos, y_pos),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=SUCCESS_COLOR,
            fontweight="bold",
        )

    ax_left.set_title("Target Steering Response", fontsize=14, fontweight="bold")
    ax_left.set_xlabel("Steering coefficient (alpha)", fontsize=11)
    ax_left.set_ylabel("Percentage of samples", fontsize=11)
    ax_left.set_xticks(x_values)
    ax_left.set_ylim(0, 100)
    ax_left.grid(axis="y", alpha=0.25, linestyle="--")
    ax_left.legend(framealpha=0.95, loc="center right")

    x_controls = np.ones(len(random_success))
    jitter = np.linspace(-0.12, 0.12, len(random_success))
    ax_right.scatter(
        x_controls + jitter,
        random_success,
        color=CONTROL_COLOR,
        alpha=0.8,
        s=42,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Random controls",
    )
    ax_right.boxplot(
        [random_success],
        positions=[1.0],
        widths=0.32,
        patch_artist=True,
        boxprops={"facecolor": "#dee2e6", "edgecolor": CONTROL_COLOR, "linewidth": 1.4},
        medianprops={"color": CONTROL_COLOR, "linewidth": 1.8},
        whiskerprops={"color": CONTROL_COLOR},
        capprops={"color": CONTROL_COLOR},
    )
    ax_right.axhline(metric(baseline, "success_pct"), color="#343a40", linestyle="--", linewidth=1.4, label="Baseline")
    ax_right.axhline(random_mean, color=CONTROL_COLOR, linestyle="-.", linewidth=1.4, label="Random mean")
    ax_right.axhline(float(target_minus_two["success_pct"]), color="#e67700", linestyle=":", linewidth=2.0, label="Target -2.0")
    ax_right.axhline(float(target_plus_two["success_pct"]), color=TARGET_COLOR, linestyle="-", linewidth=2.2, label="Target +2.0")

    ax_right.set_title("Matched Sparse Random Controls", fontsize=14, fontweight="bold")
    ax_right.set_ylabel("Success rate (%)", fontsize=11)
    ax_right.set_xticks([1.0])
    ax_right.set_xticklabels(["20 controls\n(alpha=+2.0)"], fontsize=10)
    ax_right.set_xlim(0.65, 1.35)
    ax_right.set_ylim(0, max(max(random_success), float(target_plus_two["success_pct"])) + 6)
    ax_right.grid(axis="y", alpha=0.25, linestyle="--")
    ax_right.legend(framealpha=0.95, loc="upper left", fontsize=9)

    summary_text = (
        f"Target +2.0: {float(target_plus_two['success_pct']):.1f}%\n"
        f"Random mean: {random_mean:.1f}%\n"
        f"Random max: {max(random_success):.1f}%\n"
        f"Beats: {specificity['num_controls_beaten']}/{specificity['num_controls']}\n"
        f"Empirical p <= {specificity['empirical_p_upper_bound']:.3f}"
    )
    ax_right.text(
        0.98,
        0.05,
        summary_text,
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#ced4da"},
    )

    fig.suptitle(
        "GPT-2 Medium Layer-12 Steering Is Constructive but Only Suggestively Specific",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
