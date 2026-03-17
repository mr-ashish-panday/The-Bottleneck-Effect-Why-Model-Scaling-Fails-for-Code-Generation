#!/usr/bin/env python3
"""
Create a steering-response plot from activation steering results.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def safe_metric(condition: dict, metric_name: str) -> float:
    return condition.get("category_percentages", {}).get(metric_name, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a figure summarizing activation steering outcomes"
    )
    parser.add_argument(
        "--input_file",
        default="data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json",
    )
    parser.add_argument(
        "--output_file",
        default="outputs/figures/figure5_activation_steering_response.png",
    )
    parser.add_argument(
        "--title",
        default="GPT-2 Medium Layer-12 Activation Steering",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    with open(input_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    conditions = []
    baseline = report.get("baseline")
    if baseline:
        conditions.append(
            {
                "alpha": 0.0,
                "label": "baseline",
                "success_pct": safe_metric(baseline, "success_pct"),
                "syntax_error_pct": safe_metric(baseline, "syntax_error_pct"),
                "runtime_error_pct": safe_metric(baseline, "runtime_error_pct"),
            }
        )

    for result in sorted(report.get("results", []), key=lambda item: float(item["alpha"])):
        conditions.append(
            {
                "alpha": float(result["alpha"]),
                "label": f"{float(result['alpha']):+g}",
                "success_pct": safe_metric(result, "success_pct"),
                "syntax_error_pct": safe_metric(result, "syntax_error_pct"),
                "runtime_error_pct": safe_metric(result, "runtime_error_pct"),
            }
        )

    if not conditions:
        raise ValueError("No conditions found in steering report")

    x_values = [item["alpha"] for item in conditions]
    success = [item["success_pct"] for item in conditions]
    syntax = [item["syntax_error_pct"] for item in conditions]
    runtime = [item["runtime_error_pct"] for item in conditions]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    ax.plot(x_values, success, marker="o", linewidth=2.5, color="#2ca02c", label="Success")
    ax.plot(x_values, syntax, marker="s", linewidth=2.0, color="#d62728", label="Syntax")
    ax.plot(x_values, runtime, marker="^", linewidth=2.0, color="#1f77b4", label="Runtime")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(0.02, 0.03, "baseline", transform=ax.transAxes, fontsize=10, color="black")

    for alpha, value in zip(x_values, success):
        ax.annotate(
            f"{value:.0f}",
            (alpha, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
            color="#1d6f1d",
        )

    ax.set_xlabel("Steering coefficient (alpha)", fontsize=12)
    ax.set_ylabel("Percentage of samples", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.set_xticks(sorted(set(x_values)))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.95)

    metadata = report.get("metadata", {})
    summary = (
        f"Top-{metadata.get('num_dims', '?')} dims: {metadata.get('selected_dims', [])}\n"
        f"{metadata.get('num_problems', '?')} problems x "
        f"{metadata.get('samples_per_problem', '?')} samples"
    )
    ax.text(
        0.98,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
