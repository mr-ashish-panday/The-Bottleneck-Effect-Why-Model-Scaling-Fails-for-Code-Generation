#!/usr/bin/env python3
"""
Convert saved generations into EvalPlus sample format.
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--input_file", default=None, help="generated_samples.json path")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["paths"]["results_dir"])
    input_file = Path(args.input_file or results_dir / "generated_samples.json")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        all_samples = json.load(f)

    row_count = 0
    with open(output_file, "w") as f:
        for problem in all_samples:
            prompt = problem["prompt"]
            for sample in problem["samples"]:
                record = {
                    "task_id": problem["task_id"],
                    "solution": prompt + sample["code"],
                }
                f.write(json.dumps(record) + "\n")
                row_count += 1

    print(f"Exported {row_count} samples to {output_file}")


if __name__ == "__main__":
    main()
