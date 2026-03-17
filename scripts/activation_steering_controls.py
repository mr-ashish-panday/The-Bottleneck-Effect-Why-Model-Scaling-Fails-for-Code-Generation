#!/usr/bin/env python3
"""
Run matched control experiments for activation steering.

The learned steering vector is compared against sparse random controls with the
same coefficient magnitudes and sparsity. This tests whether observed gains are
specific to the learned dimensions rather than a generic side-effect of adding
noise to the residual stream.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

import sys

sys.path.append(".")

from scripts.contrastive_activation_steering import (
    evaluate_condition,
    load_steering_vector,
)
from src.data.dataset_loader import DatasetLoader
from src.models.model_wrapper import CodeGenerationModel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare learned activation steering against matched random controls"
    )
    parser.add_argument("--config", default="config_gpt2_medium.yaml")
    parser.add_argument(
        "--analysis_file",
        default="data/results_gpt2_medium/ablation/layer12_analysis_real.json",
    )
    parser.add_argument(
        "--dimensions_file",
        default="data/results_gpt2_medium/ablation/activation_classification_real.json",
    )
    parser.add_argument(
        "--vector_mode",
        choices=["top_dims", "full_layer"],
        default="top_dims",
    )
    parser.add_argument("--num_dims", type=int, default=5)
    parser.add_argument(
        "--normalization",
        choices=["none", "l2", "mean_abs"],
        default="none",
    )
    parser.add_argument(
        "--target_alphas",
        default="-2.0,2.0",
        help="Comma-separated alphas for the learned vector",
    )
    parser.add_argument(
        "--control_alpha",
        type=float,
        default=2.0,
        help="Alpha used for the random control vectors",
    )
    parser.add_argument(
        "--num_random_controls",
        type=int,
        default=5,
        help="How many matched sparse random controls to test",
    )
    parser.add_argument(
        "--allow_target_dims",
        action="store_true",
        help="Allow random controls to reuse the target steering dimensions",
    )
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--samples_per_problem", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file",
        default="data/results_gpt2_medium/ablation/activation_steering_controls_10x5.json",
    )
    return parser


def parse_float_list(raw_value: str) -> List[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def load_existing_output(output_path: Path) -> Dict[str, object]:
    if not output_path.exists():
        return {"metadata": {}, "baseline": None, "target_conditions": [], "random_controls": []}
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sparse_random_controls(
    target_vector: np.ndarray,
    target_dims: List[int],
    num_controls: int,
    seed: int,
    allow_target_dims: bool,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    hidden_size = int(target_vector.shape[0])
    nonzero_dims = np.flatnonzero(target_vector)
    nonzero_values = target_vector[nonzero_dims].copy()

    candidate_dims = np.arange(hidden_size)
    if not allow_target_dims:
        candidate_dims = np.setdiff1d(candidate_dims, np.asarray(target_dims, dtype=np.int32))

    if len(candidate_dims) < len(nonzero_values):
        raise ValueError("Not enough candidate dimensions to build matched sparse controls")

    controls = []
    for control_id in range(num_controls):
        chosen_dims = rng.choice(candidate_dims, size=len(nonzero_values), replace=False)
        permuted_values = nonzero_values[rng.permutation(len(nonzero_values))]

        control_vector = np.zeros_like(target_vector)
        control_vector[chosen_dims] = permuted_values

        controls.append(
            {
                "control_id": control_id,
                "control_type": "random_sparse_matched",
                "selected_dims": [int(dim) for dim in chosen_dims.tolist()],
                "vector": control_vector,
                "vector_l2_norm": float(np.linalg.norm(control_vector)),
            }
        )

    return controls


def condition_key(prefix: str, alpha: float, control_id: int = None) -> str:
    if control_id is None:
        return f"{prefix}:{alpha}"
    return f"{prefix}:{control_id}:{alpha}"


def main() -> None:
    args = build_arg_parser().parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layer_index, steering_vector, selected_dims = load_steering_vector(
        analysis_file=Path(args.analysis_file),
        dimensions_file=Path(args.dimensions_file) if args.dimensions_file else None,
        vector_mode=args.vector_mode,
        num_dims=args.num_dims,
        normalization=args.normalization,
    )
    target_alphas = parse_float_list(args.target_alphas)

    report = load_existing_output(output_path)
    report["metadata"] = {
        "config": args.config,
        "analysis_file": args.analysis_file,
        "dimensions_file": args.dimensions_file,
        "layer_index": layer_index,
        "vector_mode": args.vector_mode,
        "num_dims": args.num_dims,
        "selected_dims": selected_dims,
        "normalization": args.normalization,
        "target_alphas": target_alphas,
        "control_alpha": args.control_alpha,
        "num_random_controls": args.num_random_controls,
        "allow_target_dims": args.allow_target_dims,
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "seed": args.seed,
        "target_vector_l2_norm": float(np.linalg.norm(steering_vector)),
    }

    completed_keys = set()
    for result in report.get("target_conditions", []):
        completed_keys.add(condition_key("target", float(result["alpha"])))
    for result in report.get("random_controls", []):
        completed_keys.add(
            condition_key(
                "control",
                float(result["alpha"]),
                int(result["control_id"]),
            )
        )

    print("=" * 80)
    print("ACTIVATION STEERING SPECIFICITY CONTROLS")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Layer index: {layer_index}")
    print(f"Target dims: {selected_dims[:10]}")
    print(f"Target alphas: {target_alphas}")
    print(f"Random controls: {args.num_random_controls} at alpha={args.control_alpha}")
    print(f"Output: {output_path}")

    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()

    loader = DatasetLoader(args.config)
    problems = loader.load(num_problems=args.num_problems)

    timeout = config["execution"]["timeout_seconds"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    temperature = config["model"]["temperature"]
    top_p = config["model"]["top_p"]

    if not report.get("baseline"):
        print("\nRunning baseline condition")
        report["baseline"] = evaluate_condition(
            problems=problems,
            model_wrapper=model_wrapper,
            timeout=timeout,
            num_samples=args.samples_per_problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            base_seed=args.seed,
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    for alpha in target_alphas:
        key = condition_key("target", float(alpha))
        if key in completed_keys:
            print(f"Skipping existing target condition alpha={alpha}")
            continue

        print(f"\nRunning learned-vector condition alpha={alpha}")
        result = evaluate_condition(
            problems=problems,
            model_wrapper=model_wrapper,
            timeout=timeout,
            num_samples=args.samples_per_problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            base_seed=args.seed,
            layer_index=layer_index,
            steering_vector=steering_vector,
            alpha=alpha,
        )
        result["condition_type"] = "target"
        report.setdefault("target_conditions", []).append(result)
        completed_keys.add(key)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    controls = build_sparse_random_controls(
        target_vector=steering_vector,
        target_dims=selected_dims,
        num_controls=args.num_random_controls,
        seed=args.seed + 1000,
        allow_target_dims=args.allow_target_dims,
    )

    for control in controls:
        key = condition_key("control", args.control_alpha, int(control["control_id"]))
        if key in completed_keys:
            print(f"Skipping existing random control #{control['control_id']}")
            continue

        print(
            "\nRunning random matched control "
            f"#{control['control_id']} with dims {control['selected_dims']}"
        )
        result = evaluate_condition(
            problems=problems,
            model_wrapper=model_wrapper,
            timeout=timeout,
            num_samples=args.samples_per_problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            base_seed=args.seed + (control["control_id"] * 10000),
            layer_index=layer_index,
            steering_vector=control["vector"],
            alpha=args.control_alpha,
        )
        result["condition_type"] = "random_sparse_matched"
        result["control_id"] = int(control["control_id"])
        result["selected_dims"] = control["selected_dims"]
        result["vector_l2_norm"] = control["vector_l2_norm"]
        report.setdefault("random_controls", []).append(result)
        completed_keys.add(key)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
