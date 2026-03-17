#!/usr/bin/env python3
"""
Create a three-panel figure for CodeGen ladder scaled follow-ups.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    (
        ROOT / "data/results_codegen_nl/ablation/scaled_layer11_20x10_summary.json",
        "CodeGen-NL",
        "Layer 11",
        "",
    ),
    (
        ROOT / "data/results_codegen_multi/ablation/scaled_layer7_20x10_summary.json",
        "CodeGen-Multi",
        "Layer 7",
        "Anomalous 0-scale rebound",
    ),
    (
        ROOT / "data/results_codegen_mono/ablation/scaled_layer13_20x10_summary.json",
        "CodeGen-Mono",
        "Layer 13",
        "",
    ),
]
OUTPUT_PATH = ROOT / "outputs/figures/figure8_codegen_ladder_followups"


SUCCESS_COLOR = "#2f9e44"
SYNTAX_COLOR = "#c92a2a"
RUNTIME_COLOR = "#1c7ed6"


def load_conditions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    conditions = [
        {
            "label": "1.0",
            "success_pct": float(report["baseline"]["success_pct"]),
            "syntax_error_pct": float(report["baseline"]["syntax_error_pct"]),
            "runtime_error_pct": float(report["baseline"]["runtime_error_pct"]),
        }
    ]
    for item in report["layer_summaries"][0]["conditions"]:
        conditions.append(
            {
                "label": f"{float(item['scale']):.2f}".rstrip("0").rstrip("."),
                "success_pct": float(item["success_pct"]),
                "syntax_error_pct": float(item["syntax_error_pct"]),
                "runtime_error_pct": float(item["runtime_error_pct"]),
            }
        )
    return conditions


def draw_panel(ax: plt.Axes, title: str, layer_label: str, note: str, conditions: list[dict]) -> None:
    x = np.arange(len(conditions))
    success = np.array([item["success_pct"] for item in conditions])
    runtime = np.array([item["runtime_error_pct"] for item in conditions])
    syntax = np.array([item["syntax_error_pct"] for item in conditions])

    ax.bar(x, success, color=SUCCESS_COLOR, edgecolor="black", linewidth=0.8)
    ax.bar(x, runtime, bottom=success, color=RUNTIME_COLOR, edgecolor="black", linewidth=0.8)
    ax.bar(x, syntax, bottom=success + runtime, color=SYNTAX_COLOR, edgecolor="black", linewidth=0.8)

    for idx, value in enumerate(success):
        if value >= 8.0:
            ax.text(idx, max(value / 2, 2.0), f"{value:.1f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    for idx, value in enumerate(runtime):
        if value >= 4.0:
            ax.text(idx, success[idx] + value / 2, f"{value:.1f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    title_text = f"{title}\n{layer_label}"
    if note:
        title_text += f"\n{note}"
    ax.set_title(title_text, fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in conditions], fontsize=10)
    ax.set_xlabel("Layer scale", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    if note:
        ax.text(
            0.98,
            0.98,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#c92a2a",
            bbox={"boxstyle": "round", "facecolor": "#fff5f5", "edgecolor": "#ffa8a8", "alpha": 0.95},
        )


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)

    for ax, (path, title, layer_label, note) in zip(axes, REPORTS):
        draw_panel(ax, title, layer_label, note, load_conditions(path))

    axes[0].set_ylabel("Percentage of samples", fontsize=11)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SUCCESS_COLOR, ec="black"),
        plt.Rectangle((0, 0), 1, 1, color=RUNTIME_COLOR, ec="black"),
        plt.Rectangle((0, 0), 1, 1, color=SYNTAX_COLOR, ec="black"),
    ]
    fig.legend(handles, ["Success", "Runtime", "Syntax"], loc="upper center", ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Scaled Follow-Ups Across the CodeGen Continued-Pretraining Ladder",
        fontsize=15,
        fontweight="bold",
        y=1.08,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
