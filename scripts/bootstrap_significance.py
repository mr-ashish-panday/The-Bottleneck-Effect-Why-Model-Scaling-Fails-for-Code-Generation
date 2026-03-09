#!/usr/bin/env python3
"""
Compute problem-level confidence intervals and paired significance tests.

This script treats HumanEval tasks as the independent unit and uses saved
evaluation outputs to estimate:
  - mean category rates with bootstrap confidence intervals
  - pass@k with bootstrap confidence intervals
  - paired permutation tests between models
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_MODELS = {
    "GPT-2 (124M)": "data/results_gpt2",
    "GPT-2 Medium (355M)": "data/results_gpt2_medium",
    "CodeGen-350M": "data/results_codegen",
}

DEFAULT_RATE_METRICS = [
    "success_rate",
    "syntax_error_rate",
    "runtime_error_rate",
    "wrong_output_rate",
    "timeout_rate",
]

DEFAULT_PASS_K = [1, 10, 100]


def parse_model_spec(model_spec: str) -> Tuple[str, Path]:
    """Parse NAME=PATH model specifications."""
    if "=" not in model_spec:
        raise ValueError(f"Invalid model spec '{model_spec}'. Use NAME=PATH.")
    name, path = model_spec.split("=", 1)
    return name.strip(), Path(path.strip())


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """HumanEval pass@k estimator."""
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0

    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - (k / value)
    return 1.0 - product


def load_problem_metrics(results_dir: Path, pass_k_values: Iterable[int]) -> Dict[str, Dict[str, float]]:
    """Load problem-level metrics from one evaluation_results.json file."""
    eval_file = results_dir / "evaluation_results.json"
    with open(eval_file, "r") as handle:
        evaluation_results = json.load(handle)

    problem_metrics = {}
    for problem in evaluation_results:
        task_id = problem["task_id"]
        category_counts = Counter(
            sample.get("category", "unknown_error")
            for sample in problem.get("samples", [])
        )
        total_samples = sum(category_counts.values())
        if total_samples == 0:
            continue

        metrics = {
            "success_rate": category_counts.get("success", 0) / total_samples,
            "syntax_error_rate": category_counts.get("syntax_error", 0) / total_samples,
            "runtime_error_rate": category_counts.get("runtime_error", 0) / total_samples,
            "wrong_output_rate": category_counts.get("wrong_output", 0) / total_samples,
            "timeout_rate": category_counts.get("timeout", 0) / total_samples,
            "samples_per_task": total_samples,
            "num_correct": category_counts.get("success", 0),
        }

        for k in pass_k_values:
            metrics[f"pass@{k}"] = estimate_pass_at_k(
                total_samples,
                category_counts.get("success", 0),
                k,
            )

        problem_metrics[task_id] = metrics

    return problem_metrics


def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> Dict[str, float]:
    """Bootstrap a mean and percentile confidence interval."""
    if values.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    sample_indices = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    sampled_means = values[sample_indices].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(sampled_means, 2.5)),
        "ci_high": float(np.percentile(sampled_means, 97.5)),
    }


def paired_bootstrap_difference(
    deltas: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> Dict[str, float]:
    """Bootstrap paired differences with tasks as the resampling unit."""
    sample_indices = rng.integers(0, deltas.size, size=(num_bootstrap, deltas.size))
    sampled_deltas = deltas[sample_indices].mean(axis=1)
    return {
        "difference": float(deltas.mean()),
        "ci_low": float(np.percentile(sampled_deltas, 2.5)),
        "ci_high": float(np.percentile(sampled_deltas, 97.5)),
    }


def paired_permutation_p_value(
    deltas: np.ndarray,
    rng: np.random.Generator,
    num_permutations: int,
) -> float:
    """Sign-flip permutation test for paired model comparisons."""
    observed = abs(float(deltas.mean()))
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float64),
        size=(num_permutations, deltas.size),
    )
    permuted = np.abs((signs * deltas).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (num_permutations + 1))


def summarize_model(
    problem_metrics: Dict[str, Dict[str, float]],
    rate_metrics: List[str],
    pass_k_values: List[int],
    rng: np.random.Generator,
    num_bootstrap: int,
) -> Dict[str, object]:
    """Build confidence intervals for one model."""
    task_ids = sorted(problem_metrics)
    summary = {
        "num_tasks": len(task_ids),
        "samples_per_task": int(problem_metrics[task_ids[0]]["samples_per_task"]) if task_ids else 0,
        "metrics": {},
    }

    metric_names = rate_metrics + [f"pass@{k}" for k in pass_k_values]
    for metric_name in metric_names:
        values = np.array([problem_metrics[task_id][metric_name] for task_id in task_ids], dtype=np.float64)
        summary["metrics"][metric_name] = bootstrap_mean_confidence_interval(
            values,
            rng,
            num_bootstrap,
        )

    return summary


def compare_models(
    model_a_name: str,
    model_a_metrics: Dict[str, Dict[str, float]],
    model_b_name: str,
    model_b_metrics: Dict[str, Dict[str, float]],
    metric_names: List[str],
    rng: np.random.Generator,
    num_bootstrap: int,
    num_permutations: int,
) -> List[Dict[str, object]]:
    """Compute paired bootstrap intervals and p-values."""
    shared_task_ids = sorted(set(model_a_metrics) & set(model_b_metrics))
    if not shared_task_ids:
        return []

    results = []
    for metric_name in metric_names:
        deltas = np.array(
            [
                model_a_metrics[task_id][metric_name] - model_b_metrics[task_id][metric_name]
                for task_id in shared_task_ids
            ],
            dtype=np.float64,
        )
        paired_bootstrap = paired_bootstrap_difference(deltas, rng, num_bootstrap)
        p_value = paired_permutation_p_value(deltas, rng, num_permutations)

        results.append(
            {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "metric": metric_name,
                "num_shared_tasks": len(shared_task_ids),
                **paired_bootstrap,
                "p_value": p_value,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap significance tests from saved evaluation outputs")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model specification in NAME=PATH format. Repeat for multiple models.",
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--num_permutations",
        type=int,
        default=10000,
        help="Number of paired permutations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--pass_k",
        action="append",
        type=int,
        default=[],
        help="Pass@k values to compute. Repeat for multiple values.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/tables/bootstrap_significance.json",
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    pass_k_values = args.pass_k or DEFAULT_PASS_K
    model_specs = args.model or [
        f"{name}={path}"
        for name, path in DEFAULT_MODELS.items()
    ]

    rng = np.random.default_rng(args.seed)
    all_problem_metrics = {}
    report = {
        "metadata": {
            "num_bootstrap": args.num_bootstrap,
            "num_permutations": args.num_permutations,
            "seed": args.seed,
        },
        "models": {},
        "pairwise_tests": [],
    }

    print("=" * 80)
    print("BOOTSTRAP SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    for model_spec in model_specs:
        model_name, results_dir = parse_model_spec(model_spec)
        problem_metrics = load_problem_metrics(results_dir, pass_k_values)
        all_problem_metrics[model_name] = problem_metrics

        summary = summarize_model(
            problem_metrics,
            DEFAULT_RATE_METRICS,
            pass_k_values,
            rng,
            args.num_bootstrap,
        )
        report["models"][model_name] = summary

        print(f"\n{model_name}")
        print(f"  Tasks: {summary['num_tasks']}")
        print(f"  Samples/task: {summary['samples_per_task']}")
        preview_metrics = ["success_rate", "syntax_error_rate", "runtime_error_rate"]
        preview_metrics.extend([f"pass@{k}" for k in pass_k_values[:2]])
        for metric_name in preview_metrics:
            metric = summary["metrics"][metric_name]
            print(
                f"  {metric_name:18s} "
                f"{metric['mean'] * 100:6.2f}% "
                f"[{metric['ci_low'] * 100:6.2f}, {metric['ci_high'] * 100:6.2f}]"
            )

    metric_names = DEFAULT_RATE_METRICS + [f"pass@{k}" for k in pass_k_values]
    model_names = list(all_problem_metrics)
    for index, model_a_name in enumerate(model_names):
        for model_b_name in model_names[index + 1:]:
            pairwise = compare_models(
                model_a_name,
                all_problem_metrics[model_a_name],
                model_b_name,
                all_problem_metrics[model_b_name],
                metric_names,
                rng,
                args.num_bootstrap,
                args.num_permutations,
            )
            report["pairwise_tests"].extend(pairwise)

    print("\n" + "=" * 80)
    print("PAIRWISE SUCCESS-RATE DIFFERENCES")
    print("=" * 80)
    for test in report["pairwise_tests"]:
        if test["metric"] != "success_rate":
            continue
        print(
            f"{test['model_a']} vs {test['model_b']}: "
            f"{test['difference'] * 100:+6.2f}% "
            f"[{test['ci_low'] * 100:+6.2f}, {test['ci_high'] * 100:+6.2f}], "
            f"p={test['p_value']:.4f}"
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
