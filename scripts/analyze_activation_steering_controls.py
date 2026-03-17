#!/usr/bin/env python3
"""
Summarize activation-steering control experiments.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def safe_metric(condition: dict, metric_name: str) -> float:
    return condition.get("category_percentages", {}).get(metric_name, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze learned steering vs matched random controls"
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(
        args.output_file
        or input_path.with_name("activation_steering_controls_summary.json")
    )

    with open(input_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    baseline = report.get("baseline", {})
    baseline_success = safe_metric(baseline, "success_pct")
    baseline_syntax = safe_metric(baseline, "syntax_error_pct")
    baseline_runtime = safe_metric(baseline, "runtime_error_pct")

    target_conditions = []
    for condition in sorted(
        report.get("target_conditions", []),
        key=lambda item: float(item["alpha"]),
    ):
        target_conditions.append(
            {
                "alpha": float(condition["alpha"]),
                "success_pct": safe_metric(condition, "success_pct"),
                "syntax_error_pct": safe_metric(condition, "syntax_error_pct"),
                "runtime_error_pct": safe_metric(condition, "runtime_error_pct"),
            }
        )

    random_controls = []
    for control in sorted(
        report.get("random_controls", []),
        key=lambda item: int(item["control_id"]),
    ):
        random_controls.append(
            {
                "control_id": int(control["control_id"]),
                "success_pct": safe_metric(control, "success_pct"),
                "syntax_error_pct": safe_metric(control, "syntax_error_pct"),
                "runtime_error_pct": safe_metric(control, "runtime_error_pct"),
                "selected_dims": control.get("selected_dims", []),
            }
        )

    target_alpha_to_condition = {item["alpha"]: item for item in target_conditions}
    positive_target = max(
        (item for item in target_conditions if item["alpha"] > 0),
        key=lambda item: item["success_pct"],
        default=None,
    )

    random_success = [item["success_pct"] for item in random_controls]
    random_syntax = [item["syntax_error_pct"] for item in random_controls]
    random_runtime = [item["runtime_error_pct"] for item in random_controls]

    summary = {
        "metadata": report.get("metadata", {}),
        "baseline": {
            "success_pct": baseline_success,
            "syntax_error_pct": baseline_syntax,
            "runtime_error_pct": baseline_runtime,
        },
        "target_conditions": target_conditions,
        "random_controls": random_controls,
        "random_control_summary": {
            "num_controls": len(random_controls),
            "success_mean": mean(random_success) if random_success else None,
            "success_std": pstdev(random_success) if len(random_success) > 1 else 0.0,
            "syntax_mean": mean(random_syntax) if random_syntax else None,
            "runtime_mean": mean(random_runtime) if random_runtime else None,
        },
    }

    if positive_target and random_success:
        num_controls_beaten = sum(
            1 for score in random_success if positive_target["success_pct"] > score
        )
        num_controls_tied_or_beaten = sum(
            1 for score in random_success if positive_target["success_pct"] <= score
        )
        empirical_p = (num_controls_tied_or_beaten + 1) / (len(random_success) + 1)
        summary["specificity_test"] = {
            "target_alpha": positive_target["alpha"],
            "target_success_pct": positive_target["success_pct"],
            "baseline_success_pct": baseline_success,
            "target_success_shift": positive_target["success_pct"] - baseline_success,
            "random_control_success_mean": mean(random_success),
            "random_control_success_max": max(random_success),
            "num_controls_beaten": num_controls_beaten,
            "num_controls": len(random_success),
            "empirical_p_upper_bound": empirical_p,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 80)
    print("ACTIVATION STEERING CONTROL SUMMARY")
    print("=" * 80)
    print(
        f"Baseline: success {baseline_success:.1f}% | "
        f"syntax {baseline_syntax:.1f}% | "
        f"runtime {baseline_runtime:.1f}%"
    )
    print()
    for condition in target_conditions:
        print(
            f"Target alpha {condition['alpha']:>5}: "
            f"success {condition['success_pct']:5.1f}% | "
            f"syntax {condition['syntax_error_pct']:5.1f}% | "
            f"runtime {condition['runtime_error_pct']:5.1f}%"
        )

    if random_controls:
        print()
        print(
            "Random controls: "
            f"mean success {mean(random_success):.1f}% | "
            f"max success {max(random_success):.1f}% | "
            f"std {pstdev(random_success) if len(random_success) > 1 else 0.0:.2f}"
        )

    if "specificity_test" in summary:
        result = summary["specificity_test"]
        print(
            "Specificity: "
            f"target alpha {result['target_alpha']} shift {result['target_success_shift']:+.1f} "
            f"vs random-control mean {result['random_control_success_mean']:.1f}% "
            f"(empirical p upper bound {result['empirical_p_upper_bound']:.3f})"
        )

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
