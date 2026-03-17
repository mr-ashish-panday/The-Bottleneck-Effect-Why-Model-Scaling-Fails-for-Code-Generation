#!/usr/bin/env python3
"""
Summarize contrastive activation steering results.
"""

import argparse
import json
from pathlib import Path


def safe_metric(condition: dict, metric_name: str) -> float:
    return condition.get("category_percentages", {}).get(metric_name, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Analyze activation steering outputs")
    parser.add_argument(
        "--input_file",
        required=True,
        help="activation_steering_results.json produced by scripts/contrastive_activation_steering.py",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Where to save the compact summary JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(
        args.output_file
        or input_path.with_name("activation_steering_summary.json")
    )

    with open(input_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    baseline = report.get("baseline") or {}
    baseline_success = safe_metric(baseline, "success_pct")
    baseline_syntax = safe_metric(baseline, "syntax_error_pct")
    baseline_runtime = safe_metric(baseline, "runtime_error_pct")

    condition_summaries = []
    for result in sorted(report.get("results", []), key=lambda item: float(item["alpha"])):
        success_pct = safe_metric(result, "success_pct")
        syntax_pct = safe_metric(result, "syntax_error_pct")
        runtime_pct = safe_metric(result, "runtime_error_pct")
        condition_summaries.append(
            {
                "alpha": result["alpha"],
                "success_pct": success_pct,
                "syntax_error_pct": syntax_pct,
                "runtime_error_pct": runtime_pct,
                "success_shift": success_pct - baseline_success,
                "syntax_shift": syntax_pct - baseline_syntax,
                "runtime_shift": runtime_pct - baseline_runtime,
            }
        )

    best_condition = max(condition_summaries, key=lambda item: item["success_pct"]) if condition_summaries else None

    summary = {
        "metadata": report.get("metadata", {}),
        "baseline": {
            "success_pct": baseline_success,
            "syntax_error_pct": baseline_syntax,
            "runtime_error_pct": baseline_runtime,
        },
        "best_condition": best_condition,
        "conditions": condition_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 80)
    print("ACTIVATION STEERING SUMMARY")
    print("=" * 80)
    print(
        f"Baseline: success {baseline_success:.1f}% | "
        f"syntax {baseline_syntax:.1f}% | "
        f"runtime {baseline_runtime:.1f}%"
    )
    print()
    for condition in condition_summaries:
        print(
            f"alpha {condition['alpha']:>5}: "
            f"success {condition['success_pct']:5.1f}% "
            f"({condition['success_shift']:+5.1f}), "
            f"syntax {condition['syntax_error_pct']:5.1f}% "
            f"({condition['syntax_shift']:+5.1f}), "
            f"runtime {condition['runtime_error_pct']:5.1f}% "
            f"({condition['runtime_shift']:+5.1f})"
        )

    if best_condition:
        print()
        print(
            f"Best alpha: {best_condition['alpha']} "
            f"with success {best_condition['success_pct']:.1f}%"
        )

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
