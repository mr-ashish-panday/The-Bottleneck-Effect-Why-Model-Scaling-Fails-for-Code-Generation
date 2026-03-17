#!/usr/bin/env python3
"""
Generate MBPP+ completions directly from the official EvalPlus task source.

This avoids the task-ID and split mismatch that appeared when we tried to
re-score the earlier sanitized MBPP run with EvalPlus. The script mirrors the
saved artifact format used elsewhere in the repo so the existing exporters and
analysis scripts can reuse it.
"""

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
import yaml

import sys

sys.path.append(".")

from src.models.model_wrapper import CodeGenerationModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model config path")
    parser.add_argument("--resume", action="store_true", help="Resume from saved JSON")
    parser.add_argument("--num_problems", type=int, default=None, help="Optional task cap")
    parser.add_argument("--num_samples", type=int, default=None, help="Samples per task")
    parser.add_argument("--output_dir", default=None, help="Override results_dir")
    return parser.parse_args()


def task_sort_key(task_id: str):
    numbers = re.findall(r"\d+", str(task_id))
    if numbers:
        return (0, int(numbers[-1]))
    return (1, str(task_id))


def load_mbppplus_tasks(cache_path: Path) -> List[Dict[str, str]]:
    if cache_path.exists():
        tasks = []
        with open(cache_path, "r", encoding="utf-8") as handle:
            for line in handle:
                tasks.append(json.loads(line))
        return sorted(tasks, key=lambda item: task_sort_key(item["task_id"]))

    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for clean MBPP+ generation. "
            "Install it in the server venv before running this script."
        ) from exc

    raw_tasks = get_mbpp_plus()
    tasks = []
    for task_id, problem in raw_tasks.items():
        tasks.append(
            {
                "task_id": str(problem.get("task_id", task_id)),
                "prompt": problem["prompt"],
                "entry_point": problem.get("entry_point"),
                "canonical_solution": problem.get("canonical_solution"),
            }
        )

    tasks = sorted(tasks, key=lambda item: task_sort_key(item["task_id"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")
    return tasks


def save_results(results_file: Path, all_results: List[Dict]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


def generate_with_memory_cleanup(
    model_wrapper: CodeGenerationModel,
    prompt: str,
    num_samples: int,
) -> List[str]:
    try:
        return model_wrapper.generate(prompt=prompt, num_samples=num_samples)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chunk_size = max(1, num_samples // 2)
        samples: List[str] = []
        for start in range(0, num_samples, chunk_size):
            current = min(chunk_size, num_samples - start)
            samples.extend(model_wrapper.generate(prompt=prompt, num_samples=current))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return samples


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(args.output_dir or config["paths"]["results_dir"])
    results_file = results_dir / "generated_samples.json"
    cache_path = Path(
        config.get("dataset", {}).get(
            "cache_path",
            Path(config["paths"]["raw_data"]) / "mbppplus_evalplus.jsonl",
        )
    )

    existing_results = {}
    if args.resume and results_file.exists():
        with open(results_file, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                existing_results[record["task_id"]] = record
        print(f"Resuming with {len(existing_results)} completed MBPP+ tasks")

    tasks = load_mbppplus_tasks(cache_path)
    if args.num_problems is not None:
        tasks = tasks[: args.num_problems]
    else:
        tasks = tasks[: config["feasibility_check"]["num_problems"]]

    pending_tasks = [task for task in tasks if task["task_id"] not in existing_results]
    print(f"Need to generate {len(pending_tasks)} MBPP+ tasks")

    if not pending_tasks:
        print("All requested MBPP+ tasks are already generated.")
        return

    num_samples = args.num_samples or config["feasibility_check"]["num_samples_per_problem"]

    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()

    all_results = list(existing_results.values())

    for task in pending_tasks:
        print(f"Generating {task['task_id']}")
        samples = generate_with_memory_cleanup(
            model_wrapper=model_wrapper,
            prompt=task["prompt"],
            num_samples=num_samples,
        )

        record = {
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "canonical_solution": task.get("canonical_solution"),
            "test": None,
            "entry_point": task.get("entry_point"),
            "samples": [
                {
                    "sample_id": index,
                    "code": sample,
                }
                for index, sample in enumerate(samples)
            ],
        }
        all_results.append(record)
        save_results(results_file, all_results)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved MBPP+ generations to {results_file}")


if __name__ == "__main__":
    main()
