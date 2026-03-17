#!/usr/bin/env python3
"""
Scaled layer ablation study.

This is a stronger follow-up than full zeroing because it tests whether layer
importance appears gradually as we attenuate a block instead of removing it
completely. That is the cleanest causal next step for the current paper.
"""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from tqdm import tqdm

import sys
sys.path.append(".")

from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import categorize_failure, execute_code
from src.models.model_wrapper import CodeGenerationModel


def parse_int_list(raw_value: str) -> List[int]:
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_float_list(raw_value: str) -> List[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transformer_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Unsupported architecture: expected model.transformer.h")


def scale_output(output, scale: float):
    if isinstance(output, tuple):
        hidden_states = output[0] * scale
        return (hidden_states, *output[1:])
    return output * scale


def generate_samples(
    model_wrapper: CodeGenerationModel,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    layer_index: int = None,
    scale: float = 1.0,
) -> List[str]:
    """Generate samples, optionally attenuating one transformer layer."""
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
    if layer_index is not None:
        layers = get_transformer_layers(model)
        handle = layers[layer_index].register_forward_hook(
            lambda module, module_inputs, output: scale_output(output, scale)
        )

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
    scale: float = 1.0,
) -> Dict[str, object]:
    """Run one baseline or scaled-ablation condition."""
    category_counts = Counter()
    per_problem = []
    total_samples = 0

    for problem_index, problem in enumerate(tqdm(problems, leave=False)):
        condition_seed = base_seed + (problem_index * 1000)
        if layer_index is not None:
            condition_seed += (layer_index + 1) * 100
            condition_seed += int(scale * 100)
        set_seed(condition_seed)

        generated_samples = generate_samples(
            model_wrapper=model_wrapper,
            prompt=problem.prompt,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            layer_index=layer_index,
            scale=scale,
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
        "layer": layer_index,
        "scale": scale,
        "total_samples": total_samples,
        "category_counts": dict(category_counts),
        "category_percentages": compute_percentages(category_counts, total_samples),
        "problems": per_problem,
    }


def load_existing_output(output_path: Path) -> Dict[str, object]:
    if not output_path.exists():
        return {"metadata": {}, "baseline": None, "results": []}
    with open(output_path, "r") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run scaled layer ablation")
    parser.add_argument("--config", default="config.yaml", help="Dataset config")
    parser.add_argument("--model_config", default="config_gpt2_medium.yaml", help="Model config")
    parser.add_argument("--layers", default="", help="Comma-separated layer indices. Empty means all layers.")
    parser.add_argument("--scales", default="0.75,0.5,0.25,0.0", help="Comma-separated layer scales")
    parser.add_argument("--num_problems", type=int, default=50, help="Number of benchmark problems")
    parser.add_argument("--samples_per_problem", type=int, default=20, help="Samples per problem")
    parser.add_argument("--output_file", default=None, help="Where to store results")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline generation")
    args = parser.parse_args()

    with open(args.model_config, "r") as handle:
        config = yaml.safe_load(handle)

    output_path = Path(
        args.output_file
        or Path(config["paths"]["results_dir"]) / "ablation" / "scaled_layer_ablation_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCALED LAYER ABLATION")
    print("=" * 80)
    print(f"Model config: {args.model_config}")
    print(f"Dataset config: {args.config}")
    print(f"Problems: {args.num_problems}")
    print(f"Samples/problem: {args.samples_per_problem}")
    print(f"Scales: {args.scales}")
    print(f"Output: {output_path}")

    model_wrapper = CodeGenerationModel(args.model_config)
    model_wrapper.load_model()

    loader = DatasetLoader(args.config)
    problems = loader.load(num_problems=args.num_problems)

    layers = get_transformer_layers(model_wrapper.model)
    target_layers = parse_int_list(args.layers) if args.layers else list(range(len(layers)))
    scales = parse_float_list(args.scales)

    timeout = config["execution"]["timeout_seconds"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    temperature = config["model"]["temperature"]
    top_p = config["model"]["top_p"]

    report = load_existing_output(output_path)
    report["metadata"] = {
        "model_config": args.model_config,
        "dataset_config": args.config,
        "model_name": config["model"]["name"],
        "num_layers": len(layers),
        "target_layers": target_layers,
        "scales": scales,
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "seed": args.seed,
    }

    completed_conditions = {
        (result["layer"], float(result["scale"]))
        for result in report.get("results", [])
    }

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
        with open(output_path, "w") as handle:
            json.dump(report, handle, indent=2)

    for layer_index in target_layers:
        for scale in scales:
            condition_key = (layer_index, float(scale))
            if condition_key in completed_conditions:
                print(f"Skipping existing condition layer={layer_index}, scale={scale}")
                continue

            print(f"\nRunning layer {layer_index} at scale {scale}")
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
                scale=scale,
            )
            report.setdefault("results", []).append(result)
            completed_conditions.add(condition_key)

            with open(output_path, "w") as handle:
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
