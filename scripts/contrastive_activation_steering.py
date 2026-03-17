#!/usr/bin/env python3
"""
Contrastive activation steering for code-generation behavior.

This experiment uses the difference between successful and failed layer
activations as a steering vector. Positive coefficients move the residual stream
toward the "successful code" direction, while negative coefficients move it
away. The goal is to test whether the identified subspace is not only
diagnostic, but causally useful.
"""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys

sys.path.append(".")

from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import categorize_failure, execute_code
from src.models.model_wrapper import CodeGenerationModel


def parse_float_list(raw_value: str) -> List[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_layer_module(model, layer_index: int):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_index]
    raise AttributeError("Unsupported architecture: expected model.transformer.h")


def load_steering_vector(
    analysis_file: Path,
    dimensions_file: Path,
    vector_mode: str,
    num_dims: int,
    normalization: str,
) -> Tuple[int, np.ndarray, List[int]]:
    with open(analysis_file, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)

    difference = np.asarray(analysis["difference"], dtype=np.float32)
    layer_index = int(analysis["layer_index"])

    if vector_mode == "full_layer":
        vector = difference.copy()
        selected_dims = list(range(vector.shape[0]))
    else:
        if dimensions_file:
            with open(dimensions_file, "r", encoding="utf-8") as handle:
                dimensions_data = json.load(handle)
            selected_dims = dimensions_data["top_dimensions"][:num_dims]
        else:
            selected_dims = analysis["top_discriminative_dims"][:num_dims]

        vector = np.zeros_like(difference)
        vector[selected_dims] = difference[selected_dims]

    if normalization == "l2":
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
    elif normalization == "mean_abs":
        scale = float(np.mean(np.abs(vector[np.nonzero(vector)]))) if np.count_nonzero(vector) else 0.0
        if scale > 0:
            vector = vector / scale

    return layer_index, vector.astype(np.float32), [int(dim) for dim in selected_dims]


def generate_samples(
    model_wrapper: CodeGenerationModel,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    layer_index: int = None,
    steering_vector: np.ndarray = None,
    alpha: float = 0.0,
) -> List[str]:
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model_wrapper.max_length,
    )
    inputs = {key: value.to(model_wrapper.device) for key, value in inputs.items()}

    handle = None
    if layer_index is not None and steering_vector is not None and alpha != 0.0:
        layer = resolve_layer_module(model, layer_index)
        steering_tensor = torch.tensor(
            steering_vector,
            device=model_wrapper.device,
            dtype=model.dtype,
        ).view(1, 1, -1)

        def hook_fn(module, module_input, module_output):
            if isinstance(module_output, tuple):
                hidden_states = module_output[0] + (alpha * steering_tensor)
                return (hidden_states, *module_output[1:])
            return module_output + (alpha * steering_tensor)

        handle = layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        decoded = []
        for output in outputs:
            decoded.append(
                tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            )
        return decoded
    finally:
        if handle is not None:
            handle.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_percentages(category_counts: Counter, total_samples: int) -> Dict[str, float]:
    return {
        f"{category}_pct": (count / total_samples) * 100 if total_samples else 0.0
        for category, count in category_counts.items()
    }


def evaluate_condition(
    problems,
    model_wrapper: CodeGenerationModel,
    timeout: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    layer_index: int = None,
    steering_vector: np.ndarray = None,
    alpha: float = 0.0,
) -> Dict[str, object]:
    category_counts = Counter()
    per_problem = []
    total_samples = 0

    for problem_index, problem in enumerate(tqdm(problems, leave=False)):
        condition_seed = base_seed + (problem_index * 1000)
        if alpha != 0.0:
            condition_seed += int((alpha + 10.0) * 100)
        set_seed(condition_seed)

        generated_samples = generate_samples(
            model_wrapper=model_wrapper,
            prompt=problem.prompt,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            layer_index=layer_index,
            steering_vector=steering_vector,
            alpha=alpha,
        )

        problem_counts = Counter()
        for sample in generated_samples:
            full_code = problem.prompt + sample
            result = execute_code(full_code, problem.test, timeout=timeout)
            category = categorize_failure(result)
            problem_counts[category] += 1
            category_counts[category] += 1
            total_samples += 1

        per_problem.append(
            {
                "task_id": problem.task_id,
                "category_counts": dict(problem_counts),
                **compute_percentages(problem_counts, sum(problem_counts.values())),
            }
        )

    return {
        "alpha": alpha,
        "layer": layer_index,
        "total_samples": total_samples,
        "category_counts": dict(category_counts),
        "category_percentages": compute_percentages(category_counts, total_samples),
        "problems": per_problem,
    }


def load_existing_output(output_path: Path) -> Dict[str, object]:
    if not output_path.exists():
        return {"metadata": {}, "baseline": None, "results": []}
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Contrastive activation steering for code generation")
    parser.add_argument("--config", default="config_gpt2_medium.yaml", help="Model/dataset config")
    parser.add_argument(
        "--analysis_file",
        default="data/results_gpt2_medium/ablation/layer12_analysis_real.json",
        help="Layer analysis JSON with success/failure mean activations",
    )
    parser.add_argument(
        "--dimensions_file",
        default="data/results_gpt2_medium/ablation/activation_classification_real.json",
        help="Optional classifier JSON with ranked steering dimensions",
    )
    parser.add_argument(
        "--vector_mode",
        choices=["top_dims", "full_layer"],
        default="top_dims",
        help="Use a sparse top-dimension vector or the full difference vector",
    )
    parser.add_argument("--num_dims", type=int, default=5, help="Number of top dimensions to keep")
    parser.add_argument(
        "--normalization",
        choices=["none", "l2", "mean_abs"],
        default="none",
        help="Optional steering-vector normalization",
    )
    parser.add_argument(
        "--alphas",
        default="-1.0,-0.5,0.5,1.0",
        help="Comma-separated steering strengths",
    )
    parser.add_argument("--num_problems", type=int, default=10, help="Number of benchmark problems")
    parser.add_argument("--samples_per_problem", type=int, default=5, help="Samples per problem")
    parser.add_argument("--output_file", default=None, help="Where to store results")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline generation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(config["paths"]["results_dir"])
    output_path = Path(
        args.output_file
        or results_dir / "ablation" / "activation_steering_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layer_index, steering_vector, selected_dims = load_steering_vector(
        analysis_file=Path(args.analysis_file),
        dimensions_file=Path(args.dimensions_file) if args.dimensions_file else None,
        vector_mode=args.vector_mode,
        num_dims=args.num_dims,
        normalization=args.normalization,
    )
    alphas = parse_float_list(args.alphas)

    print("=" * 80)
    print("CONTRASTIVE ACTIVATION STEERING")
    print("=" * 80)
    print(f"Model config: {args.config}")
    print(f"Layer index: {layer_index}")
    print(f"Vector mode: {args.vector_mode}")
    print(f"Selected dims: {selected_dims[:10]}")
    print(f"Normalization: {args.normalization}")
    print(f"Alphas: {alphas}")
    print(f"Problems: {args.num_problems}")
    print(f"Samples/problem: {args.samples_per_problem}")
    print(f"Output: {output_path}")

    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()

    loader = DatasetLoader(args.config)
    problems = loader.load(num_problems=args.num_problems)

    timeout = config["execution"]["timeout_seconds"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    temperature = config["model"]["temperature"]
    top_p = config["model"]["top_p"]

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
        "vector_l2_norm": float(np.linalg.norm(steering_vector)),
        "vector_mean_abs": float(np.mean(np.abs(steering_vector))),
        "alphas": alphas,
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "seed": args.seed,
    }

    completed_alphas = {float(result["alpha"]) for result in report.get("results", [])}

    if not args.skip_baseline and not report.get("baseline"):
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

    for alpha in alphas:
        if float(alpha) in completed_alphas:
            print(f"Skipping existing condition alpha={alpha}")
            continue

        print(f"\nRunning steering condition alpha={alpha}")
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
        report.setdefault("results", []).append(result)
        completed_alphas.add(float(alpha))

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        success_pct = result["category_percentages"].get("success_pct", 0.0)
        syntax_pct = result["category_percentages"].get("syntax_error_pct", 0.0)
        runtime_pct = result["category_percentages"].get("runtime_error_pct", 0.0)
        print(
            f"  Success {success_pct:5.1f}% | "
            f"Syntax {syntax_pct:5.1f}% | "
            f"Runtime {runtime_pct:5.1f}%"
        )

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
