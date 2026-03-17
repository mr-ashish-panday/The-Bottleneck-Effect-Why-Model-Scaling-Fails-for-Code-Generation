#!/usr/bin/env python3
"""
Compute prompt perplexity on a benchmark prompt set for one model/config.

This is a low-cost within-family follow-up for the CodeGen NL -> Multi -> Mono
ladder. It measures how well each checkpoint models the benchmark prompt
distribution before any generation or execution happens.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

import sys

sys.path.append(".")

from src.data.dataset_loader import DatasetLoader
from src.models.model_wrapper import CodeGenerationModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model/dataset config")
    parser.add_argument("--num_problems", type=int, default=None, help="Optional cap")
    parser.add_argument("--output_file", default=None, help="Where to save JSON")
    return parser.parse_args()


def shift_nll_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, float]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)

    total_nll = float(token_losses.sum().item())
    total_tokens = int(shift_labels.numel())
    avg_nll = total_nll / total_tokens if total_tokens else 0.0
    perplexity = math.exp(avg_nll) if total_tokens else float("inf")
    return {
        "num_tokens": total_tokens,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
    }


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_path = Path(
        args.output_file
        or Path(config["paths"]["results_dir"]) / "prompt_perplexity.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader(args.config)
    problems = loader.load(
        num_problems=args.num_problems or config["feasibility_check"]["num_problems"]
    )

    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    per_problem: List[Dict[str, object]] = []
    total_nll = 0.0
    total_tokens = 0

    for problem in tqdm(problems, desc="Prompt perplexity"):
        encoded = tokenizer(
            problem.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model_wrapper.max_length,
        )
        encoded = {key: value.to(model_wrapper.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        metrics = shift_nll_from_logits(outputs.logits, encoded["input_ids"])
        record = {
            "task_id": problem.task_id,
            "prompt_length_chars": len(problem.prompt),
            **metrics,
        }
        per_problem.append(record)

        total_nll += metrics["total_nll"]
        total_tokens += metrics["num_tokens"]

    mean_perplexity = (
        sum(record["perplexity"] for record in per_problem) / len(per_problem)
        if per_problem
        else float("inf")
    )
    token_weighted_avg_nll = total_nll / total_tokens if total_tokens else 0.0
    token_weighted_perplexity = math.exp(token_weighted_avg_nll) if total_tokens else float("inf")

    summary = {
        "config": args.config,
        "model_name": config["model"]["name"],
        "dataset_name": config["dataset"]["name"],
        "num_problems": len(per_problem),
        "mean_prompt_perplexity": mean_perplexity,
        "token_weighted_avg_nll": token_weighted_avg_nll,
        "token_weighted_prompt_perplexity": token_weighted_perplexity,
        "per_problem": per_problem,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote prompt perplexity summary to {output_path}")
    print(f"Mean prompt perplexity: {mean_perplexity:.4f}")
    print(f"Token-weighted prompt perplexity: {token_weighted_perplexity:.4f}")


if __name__ == "__main__":
    main()
