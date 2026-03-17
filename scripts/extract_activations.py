#!/usr/bin/env python3
"""
Extract hidden-state activations for saved generations.
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
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Maximum samples to extract per category.",
    )
    parser.add_argument("--output_file", default=None)
    parser.add_argument(
        "--representation",
        choices=["last_token", "mean"],
        default="last_token",
        help="How to collapse sequence activations into a single vector.",
    )
    return parser.parse_args()


def extract_sample_activations(model, tokenizer, prompt, code, layers, representation):
    """Extract activations for selected transformer blocks."""
    full_code = prompt + code
    inputs = tokenizer(full_code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    activations = {}

    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for layer_idx in layers:
            # hidden_states[0] is the embedding output, so block i lives at i + 1.
            hidden_state = hidden_states[layer_idx + 1][0]
            if representation == "mean":
                vector = hidden_state.mean(dim=0)
            else:
                vector = hidden_state[-1]
            activations[f"layer_{layer_idx}"] = vector.cpu().numpy().astype(np.float32).tolist()

    return activations


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(config["paths"]["results_dir"])
    eval_file = results_dir / "evaluation_results.json"
    output_file = Path(args.output_file or results_dir / "activations.json")
    layers = [0, 3, 6, 9, 11]

    print("=" * 60)
    print("ACTIVATION EXTRACTION")
    print("=" * 60)
    print(f"Representation: {args.representation}")
    print(f"Layers: {layers}")
    print(f"Max samples per category: {args.num_samples}")
    print()

    print("Loading model...")
    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()

    with open(eval_file, "r", encoding="utf-8") as handle:
        all_samples = json.load(handle)

    samples_by_category = {"success": [], "syntax_error": [], "runtime_error": []}
    for problem in all_samples:
        for sample in problem["samples"]:
            category = sample.get("category")
            if category not in samples_by_category:
                continue
            if len(samples_by_category[category]) >= args.num_samples:
                continue

            samples_by_category[category].append(
                {
                    "task_id": problem["task_id"],
                    "sample_id": sample["sample_id"],
                    "prompt": problem["prompt"],
                    "code": sample["code"],
                    "category": category,
                }
            )

    print(
        "Collected samples: "
        + ", ".join(f"{cat}={len(items)}" for cat, items in samples_by_category.items())
    )
    print()

    records = []
    for category, samples in samples_by_category.items():
        print(f"Extracting {category} activations...")
        for sample in tqdm(samples, desc=category):
            try:
                activations = extract_sample_activations(
                    model=model_wrapper,
                    tokenizer=model_wrapper.tokenizer,
                    prompt=sample["prompt"],
                    code=sample["code"],
                    layers=layers,
                    representation=args.representation,
                )
            except Exception as exc:
                print(f"Error processing {sample['task_id']} sample {sample['sample_id']}: {exc}")
                continue

            records.append(
                {
                    "task_id": sample["task_id"],
                    "sample_id": sample["sample_id"],
                    "category": category,
                    "representation": args.representation,
                    "activations": activations,
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print()
    print(f"Extracted {len(records)} activation samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
