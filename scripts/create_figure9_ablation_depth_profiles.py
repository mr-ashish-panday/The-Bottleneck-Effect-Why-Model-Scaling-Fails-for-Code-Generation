#!/usr/bin/env python3
"""
Create depth-profile plots for success and runtime under layer ablation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    ("GPT-2 Small", ROOT / "data/results/ablation/layer_ablation_results.json", "#8b5e34"),
    ("GPT-2 Medium", ROOT / "data/results_gpt2_medium/ablation/layer_ablation_results.json", "#c92a2a"),
    ("CodeGen-350M", ROOT / "data/results_codegen/ablation/layer_ablation_results.json", "#1c7ed6"),
]
OUTPUT_PATH = ROOT / "outputs/figures/figure9_ablation_depth_profiles"


def load_series(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    num_layers = len(rows)
    depths = np.linspace(0.0, 100.0, num_layers)
    success = np.array([float(row.get("success_pct", 0.0)) for row in rows])
    runtime = np.array([float(row.get("runtime_error_pct", 0.0)) for row in rows])

    best_success_idx = int(np.argmax(success))
    best_runtime_idx = int(np.argmax(runtime))

    return {
        "depths": depths,
        "success": success,
        "runtime": runtime,
        "best_success_idx": best_success_idx,
        "best_runtime_idx": best_runtime_idx,
    }


def annotate_signature(ax: plt.Axes, depths: np.ndarray, values: np.ndarray, idx: int, label: str, color: str) -> None:
    ax.scatter(depths[idx], values[idx], color=color, edgecolors="black", linewidths=0.8, s=70, zorder=5)
    ax.annotate(
        label,
        (depths[idx], values[idx]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=9,
        color=color,
        fontweight="bold",
    )


def main() -> None:
    loaded = [(label, load_series(path), color) for label, path, color in REPORTS]

    fig, (ax_success, ax_runtime) = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True)

    for label, series, color in loaded:
        ax_success.plot(series["depths"], series["success"], marker="o", markersize=4.5, linewidth=2.2, color=color, label=label)
        ax_runtime.plot(series["depths"], series["runtime"], marker="o", markersize=4.5, linewidth=2.2, color=color, label=label)

        success_label = f"L{series['best_success_idx']}"
        runtime_label = f"L{series['best_runtime_idx']}"
        annotate_signature(ax_success, series["depths"], series["success"], series["best_success_idx"], success_label, color)
        annotate_signature(ax_runtime, series["depths"], series["runtime"], series["best_runtime_idx"], runtime_label, color)

    ax_success.set_title("Residual Success by Normalized Depth", fontsize=13, fontweight="bold")
    ax_runtime.set_title("Runtime-Error Shift by Normalized Depth", fontsize=13, fontweight="bold")

    ax_success.set_ylabel("Percentage of samples", fontsize=11)
    ax_success.set_xlabel("Layer depth (%)", fontsize=11)
    ax_runtime.set_xlabel("Layer depth (%)", fontsize=11)

    ax_success.set_ylim(0, 35)
    ax_runtime.set_ylim(0, 26)
    ax_success.grid(alpha=0.25, linestyle="--")
    ax_runtime.grid(alpha=0.25, linestyle="--")

    fig.legend(loc="upper center", ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Layer Ablation Profiles Show Narrower Bottlenecks in GPT-2 and Broader Residual Support in CodeGen",
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
