#!/usr/bin/env python3
"""
Summarize scaled layer ablation results into a compact report.
"""

import argparse
import json
from pathlib import Path


def safe_metric(condition: dict, metric_name: str) -> float:
    return condition.get("category_percentages", {}).get(metric_name, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Analyze scaled ablation outputs")
    parser.add_argument(
        "--input_file",
        required=True,
        help="scaled_layer_ablation_results.json produced by scripts/scaled_layer_ablation.py",
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
        or input_path.with_name("scaled_layer_ablation_summary.json")
    )

    with open(input_path, "r") as handle:
        report = json.load(handle)

    baseline = report.get("baseline") or {}
    baseline_success = safe_metric(baseline, "success_pct")
    baseline_syntax = safe_metric(baseline, "syntax_error_pct")
    baseline_runtime = safe_metric(baseline, "runtime_error_pct")

    by_layer = {}
    for condition in report.get("results", []):
        by_layer.setdefault(condition["layer"], []).append(condition)

    layer_summaries = []
    for layer, conditions in sorted(by_layer.items()):
        ordered = sorted(conditions, key=lambda item: float(item["scale"]), reverse=True)
        max_runtime = max(safe_metric(item, "runtime_error_pct") for item in ordered)
        max_syntax = max(safe_metric(item, "syntax_error_pct") for item in ordered)
        min_success = min(safe_metric(item, "success_pct") for item in ordered)

        layer_summaries.append(
            {
                "layer": layer,
                "num_conditions": len(ordered),
                "max_runtime_shift": max_runtime - baseline_runtime,
                "max_syntax_shift": max_syntax - baseline_syntax,
                "max_success_drop": baseline_success - min_success,
                "conditions": [
                    {
                        "scale": item["scale"],
                        "success_pct": safe_metric(item, "success_pct"),
                        "syntax_error_pct": safe_metric(item, "syntax_error_pct"),
                        "runtime_error_pct": safe_metric(item, "runtime_error_pct"),
                    }
                    for item in ordered
                ],
            }
        )

    summary = {
        "metadata": report.get("metadata", {}),
        "baseline": {
            "success_pct": baseline_success,
            "syntax_error_pct": baseline_syntax,
            "runtime_error_pct": baseline_runtime,
        },
        "layer_summaries": layer_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 80)
    print("SCALED ABLATION SUMMARY")
    print("=" * 80)
    print(
        f"Baseline: success {baseline_success:.1f}% | "
        f"syntax {baseline_syntax:.1f}% | "
        f"runtime {baseline_runtime:.1f}%"
    )
    print()
    for layer_summary in layer_summaries:
        print(
            f"Layer {layer_summary['layer']:>2}: "
            f"success drop {layer_summary['max_success_drop']:.1f} | "
            f"syntax shift {layer_summary['max_syntax_shift']:.1f} | "
            f"runtime shift {layer_summary['max_runtime_shift']:.1f}"
        )

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
