#!/usr/bin/env python3
"""
Identify missing or undersampled tasks in a generated/evaluated result file.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

import sys

sys.path.append(".")

from src.data.dataset_loader import DatasetLoader


def load_results(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["task_id"]: item for item in data}


def sample_count(record: dict) -> int:
    return len(record.get("samples", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find missing or short tasks in generation outputs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--input_file",
        default=None,
        help="JSON file to inspect; defaults to results_dir/generated_samples.json",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Expected dataset prefix length. Defaults to feasibility_check.num_problems.",
    )
    parser.add_argument(
        "--expected_samples",
        type=int,
        default=None,
        help="Expected samples per task. Defaults to feasibility_check.num_samples_per_problem.",
    )
    parser.add_argument(
        "--task_ids_out",
        default=None,
        help="Optional file to write task IDs needing repair, one per line.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(config["paths"]["results_dir"])
    input_path = Path(args.input_file or results_dir / "generated_samples.json")
    expected_samples = args.expected_samples or config["feasibility_check"]["num_samples_per_problem"]
    num_problems = args.num_problems or config["feasibility_check"]["num_problems"]

    loader = DatasetLoader(args.config)
    expected_problems = loader.load(num_problems=num_problems)
    expected_task_ids = [problem.task_id for problem in expected_problems]
    expected_task_set = set(expected_task_ids)

    found = load_results(input_path)

    missing_task_ids: List[str] = []
    short_task_ids: List[str] = []
    extra_task_ids: List[str] = sorted(set(found) - expected_task_set)

    for task_id in expected_task_ids:
        record = found.get(task_id)
        if record is None:
            missing_task_ids.append(task_id)
            continue
        if sample_count(record) < expected_samples:
            short_task_ids.append(task_id)

    repair_task_ids = missing_task_ids + [task_id for task_id in short_task_ids if task_id not in missing_task_ids]

    report = {
        "input_file": str(input_path),
        "expected_num_problems": len(expected_task_ids),
        "expected_samples_per_task": expected_samples,
        "found_num_tasks": len(found),
        "missing_task_ids": missing_task_ids,
        "short_task_ids": short_task_ids,
        "extra_task_ids": extra_task_ids,
        "repair_task_ids": repair_task_ids,
    }

    print(json.dumps(report, indent=2))

    if args.task_ids_out:
        output_path = Path(args.task_ids_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for task_id in repair_task_ids:
                handle.write(f"{task_id}\n")
        print(f"\nWrote repair task list to: {output_path}")


if __name__ == "__main__":
    main()
