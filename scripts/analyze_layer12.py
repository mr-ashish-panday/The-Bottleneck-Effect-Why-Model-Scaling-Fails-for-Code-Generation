#!/usr/bin/env python3
"""
Analyze a target GPT-2 Medium layer and save both summary statistics and
sample-level activations for linear probing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import yaml

import sys

sys.path.append(".")

from src.models.model_wrapper import CodeGenerationModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_gpt2_medium.yaml")
    parser.add_argument(
        "--layer_index",
        type=int,
        default=12,
        help="Transformer block index to analyze.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Total number of samples to analyze across both classes.",
    )
    parser.add_argument(
        "--failure_category",
        default="syntax_error",
        help="Negative class to compare against successful generations.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last_token"],
        default="mean",
        help="How to collapse sequence activations into a single vector.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--samples_output", default=None)
    return parser.parse_args()


def sample_vector_from_layer(model, tokenizer, prompt, code, layer_index, pooling):
    """Return a single activation vector for one completion."""
    full_code = prompt + code
    inputs = tokenizer(full_code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    captured = None

    def hook_fn(module, module_input, module_output):
        nonlocal captured
        captured = module_output[0] if isinstance(module_output, tuple) else module_output

    handle = model.transformer.h[layer_index].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    if captured is None:
        raise RuntimeError(f"No activation captured for layer {layer_index}")

    if pooling == "mean":
        vector = captured.mean(dim=1).squeeze(0)
    else:
        vector = captured[:, -1, :].squeeze(0)

    return vector.detach().cpu().numpy().astype(np.float32)


def collect_balanced_samples(all_results, num_per_class, failure_category, seed):
    success_candidates = []
    failure_candidates = []

    for problem in all_results:
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        for sample in problem["samples"]:
            category = sample.get("category")
            record = {
                "task_id": task_id,
                "sample_id": sample.get("sample_id"),
                "prompt": prompt,
                "code": sample["code"],
                "category": category,
            }

            if category == "success":
                success_candidates.append(record)
            elif category == failure_category:
                failure_candidates.append(record)

    if len(success_candidates) < num_per_class:
        raise ValueError(
            f"Requested {num_per_class} success samples, found {len(success_candidates)}"
        )
    if len(failure_candidates) < num_per_class:
        raise ValueError(
            f"Requested {num_per_class} {failure_category} samples, "
            f"found {len(failure_candidates)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(success_candidates)
    rng.shuffle(failure_candidates)

    return success_candidates[:num_per_class], failure_candidates[:num_per_class]


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(config["paths"]["results_dir"])
    ablation_dir = results_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    summary_output = Path(args.summary_output or ablation_dir / "layer12_analysis.json")
    samples_output = Path(
        args.samples_output or ablation_dir / "layer12_probe_samples.json"
    )

    num_per_class = max(1, args.num_samples // 2)

    print("=" * 60)
    print("TARGET LAYER ACTIVATION ANALYSIS")
    print("=" * 60)
    print(f"Model config: {args.config}")
    print(f"Layer index: {args.layer_index}")
    print(f"Pooling: {args.pooling}")
    print(f"Failure category: {args.failure_category}")
    print(f"Requested samples per class: {num_per_class}")
    print()

    print("Loading model...")
    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()

    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "r", encoding="utf-8") as handle:
        all_results = json.load(handle)

    success_samples, failure_samples = collect_balanced_samples(
        all_results=all_results,
        num_per_class=num_per_class,
        failure_category=args.failure_category,
        seed=args.seed,
    )

    print(
        f"Collected {len(success_samples)} success and "
        f"{len(failure_samples)} {args.failure_category} samples"
    )

    sample_records = []
    success_activations = []
    failure_activations = []

    for label, samples, target_list in [
        ("success", success_samples, success_activations),
        (args.failure_category, failure_samples, failure_activations),
    ]:
        for sample in tqdm(samples, desc=f"Extracting {label}"):
            activation = sample_vector_from_layer(
                model=model_wrapper.model,
                tokenizer=model_wrapper.tokenizer,
                prompt=sample["prompt"],
                code=sample["code"],
                layer_index=args.layer_index,
                pooling=args.pooling,
            )

            target_list.append(activation)
            sample_records.append(
                {
                    "task_id": sample["task_id"],
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "layer_index": args.layer_index,
                    "pooling": args.pooling,
                    "activation": activation.tolist(),
                }
            )

    success_matrix = np.stack(success_activations)
    failure_matrix = np.stack(failure_activations)

    success_mean = success_matrix.mean(axis=0)
    failure_mean = failure_matrix.mean(axis=0)
    difference = success_mean - failure_mean
    abs_difference = np.abs(difference)
    top_dims = np.argsort(abs_difference)[::-1][:20]

    summary = {
        "layer_index": args.layer_index,
        "layer_label": f"transformer.h[{args.layer_index}]",
        "pooling": args.pooling,
        "failure_category": args.failure_category,
        "seed": args.seed,
        "num_success_samples": int(success_matrix.shape[0]),
        "num_failure_samples": int(failure_matrix.shape[0]),
        "sample_activations_file": str(samples_output),
        "success_mean": success_mean.tolist(),
        "failure_mean": failure_mean.tolist(),
        "difference": difference.tolist(),
        "top_discriminative_dims": [int(dim) for dim in top_dims],
        "difference_magnitude": float(abs_difference.mean()),
    }

    with open(summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(samples_output, "w", encoding="utf-8") as handle:
        json.dump(sample_records, handle, indent=2)

    print()
    print("Analysis complete")
    print(f"Mean absolute difference: {summary['difference_magnitude']:.4f}")
    print(f"Top 5 discriminative dimensions: {summary['top_discriminative_dims'][:5]}")
    print(f"Summary saved to: {summary_output}")
    print(f"Sample activations saved to: {samples_output}")


if __name__ == "__main__":
    main()
