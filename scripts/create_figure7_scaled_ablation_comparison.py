#!/usr/bin/env python3
"""
Create a paired graded-ablation comparison for GPT-2 Medium and CodeGen.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GPT2_PATH = ROOT / "data/results_gpt2_medium/ablation/scaled_layer12_10x5_summary.json"
CODEGEN_PATH = ROOT / "data/results_codegen/ablation/scaled_layer13_10x5_summary.json"
OUTPUT_PATH = ROOT / "outputs/figures/figure7_scaled_ablation_comparison"


SUCCESS_COLOR = "#2f9e44"
SYNTAX_COLOR = "#c92a2a"
RUNTIME_COLOR = "#1c7ed6"


def load_conditions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    baseline = {
        "label": "1.0",
        "success_pct": float(report["baseline"]["success_pct"]),
        "syntax_error_pct": float(report["baseline"]["syntax_error_pct"]),
        "runtime_error_pct": float(report["baseline"]["runtime_error_pct"]),
    }
    conditions = [baseline]
    for item in report["layer_summaries"][0]["conditions"]:
        conditions.append(
            {
                "label": f"{float(item['scale']):.2f}".rstrip("0").rstrip("."),
                "success_pct": float(item["success_pct"]),
                "syntax_error_pct": float(item["syntax_error_pct"]),
                "runtime_error_pct": float(item["runtime_error_pct"]),
            }
        )
    return {"metadata": report["metadata"], "conditions": conditions}


def draw_panel(ax: plt.Axes, title: str, subtitle: str, conditions: list[dict]) -> None:
    x = np.arange(len(conditions))
    success = np.array([item["success_pct"] for item in conditions])
    runtime = np.array([item["runtime_error_pct"] for item in conditions])
    syntax = np.array([item["syntax_error_pct"] for item in conditions])

    ax.bar(x, success, color=SUCCESS_COLOR, edgecolor="black", linewidth=0.8, label="Success")
    ax.bar(x, runtime, bottom=success, color=RUNTIME_COLOR, edgecolor="black", linewidth=0.8, label="Runtime")
    ax.bar(x, syntax, bottom=success + runtime, color=SYNTAX_COLOR, edgecolor="black", linewidth=0.8, label="Syntax")

    for idx, value in enumerate(success):
        if value >= 2.0:
            ax.text(idx, max(value / 2, 1.4), f"{value:.0f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for idx, value in enumerate(runtime):
        if value >= 4.0:
            ax.text(idx, success[idx] + value / 2, f"{value:.0f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in conditions], fontsize=10)
    ax.set_xlabel("Layer scale", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def main() -> None:
    gpt2 = load_conditions(GPT2_PATH)
    codegen = load_conditions(CODEGEN_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    draw_panel(
        axes[0],
        "GPT-2 Medium",
        "Layer 12, 10 problems x 5 samples",
        gpt2["conditions"],
    )
    draw_panel(
        axes[1],
        "CodeGen-350M",
        "Layer 13, 10 problems x 5 samples",
        codegen["conditions"],
    )

    axes[0].set_ylabel("Percentage of samples", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Graded Ablation Reveals Thresholded Collapse in GPT-2 and Residual Executable Support in CodeGen",
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
