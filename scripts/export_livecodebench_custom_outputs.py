#!/usr/bin/env python3
"""
Convert saved generations into LiveCodeBench custom_evaluator format.
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--input_file", default=None, help="generated_samples.json path")
    parser.add_argument("--output_file", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["paths"]["results_dir"])
    input_file = Path(args.input_file or results_dir / "generated_samples.json")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        all_samples = json.load(f)

    outputs = []
    for problem in all_samples:
        prompt = problem["prompt"]
        code_list = [prompt + sample["code"] for sample in problem["samples"]]
        outputs.append({
            "question_id": str(problem["task_id"]),
            "code_list": code_list,
        })

    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Exported {len(outputs)} LiveCodeBench records to {output_file}")


if __name__ == "__main__":
    main()
