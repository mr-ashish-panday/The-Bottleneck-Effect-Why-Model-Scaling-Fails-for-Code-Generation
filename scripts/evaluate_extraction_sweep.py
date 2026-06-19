#!/usr/bin/env python3
"""
Evaluate whether output-extraction choices change model ranking.

This is a confound-control script. The original evaluator uses raw prompt plus
generated text. This script also tries simple forgiving extraction strategies
without mutating the saved generations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import yaml
from tqdm import tqdm

import sys

sys.path.append(".")

from src.evaluation.code_executor import execute_code


def strip_markdown_fences(code: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return code


def truncate_at_common_stops(code: str) -> str:
    stops = [
        "\nif __name__",
        "\n# Example",
        "\n# Test",
        "\nassert ",
        "\nprint(",
        "\n```",
    ]
    best = len(code)
    for marker in stops:
        index = code.find(marker)
        if index > 0:
            best = min(best, index)
    return code[:best].rstrip() + "\n"


def prompt_preamble(prompt: str) -> str:
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            return "\n".join(lines[:index]).rstrip() + "\n"
    return ""


def keep_first_generated_function_or_body(prompt: str, code: str) -> str:
    """Use a full generated function if present; otherwise use the body."""
    cleaned = strip_markdown_fences(code)
    match = re.search(r"(^|\n)(def\s+\w+\s*\(.*)", cleaned, flags=re.DOTALL)
    if match:
        return prompt_preamble(prompt) + match.group(2)
    return prompt + cleaned


def strategy_raw(prompt: str, code: str) -> str:
    return prompt + code


def strategy_strip_fences(prompt: str, code: str) -> str:
    return prompt + strip_markdown_fences(code)


def strategy_truncate_stops(prompt: str, code: str) -> str:
    return prompt + truncate_at_common_stops(strip_markdown_fences(code))


def strategy_generated_function_or_body(prompt: str, code: str) -> str:
    return truncate_at_common_stops(keep_first_generated_function_or_body(prompt, code))


STRATEGIES: Dict[str, Callable[[str, str], str]] = {
    "raw": strategy_raw,
    "strip_fences": strategy_strip_fences,
    "truncate_stops": strategy_truncate_stops,
    "generated_function_or_body": strategy_generated_function_or_body,
}


def categorize(result: Dict) -> str:
    if result.get("success"):
        return "success"
    error_type = result.get("error_type")
    if error_type == "syntax_error":
        return "syntax_error"
    if error_type == "timeout":
        return "timeout"
    if error_type == "assertion_error":
        return "wrong_output"
    if error_type == "runtime_error":
        return "runtime_error"
    return str(error_type or "unknown_error")


def iter_samples(records: List[Dict]) -> Iterable[Dict]:
    for problem in records:
        for sample in problem.get("samples", []):
            yield {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "test": problem["test"],
                "sample_id": sample.get("sample_id"),
                "code": sample.get("code", ""),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=list(STRATEGIES.keys()),
        choices=list(STRATEGIES.keys()),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results_dir = Path(config["paths"]["results_dir"])
    input_file = Path(args.input_file or results_dir / "generated_samples.json")
    timeout = int(config["execution"]["timeout_seconds"])

    with input_file.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    samples = list(iter_samples(records))
    summary = {
        "config": args.config,
        "input_file": str(input_file).replace("\\", "/"),
        "num_tasks": len(records),
        "num_samples": len(samples),
        "strategies": {},
    }

    for strategy_name in args.strategies:
        transform = STRATEGIES[strategy_name]
        counts: Counter = Counter()
        task_successes: Counter = Counter()

        for sample in tqdm(samples, desc=f"Extraction {strategy_name}"):
            full_code = transform(sample["prompt"], sample["code"])
            try:
                result = execute_code(full_code, sample["test"], timeout=timeout)
                category = categorize(result)
            except Exception as exc:
                category = "evaluation_error"
                result = {"success": False, "error_message": str(exc)}

            counts[category] += 1
            if result.get("success"):
                task_successes[sample["task_id"]] += 1

        success_count = counts["success"]
        summary["strategies"][strategy_name] = {
            "sample_success_count": success_count,
            "sample_success_pct": (success_count / len(samples) * 100.0) if samples else 0.0,
            "task_success_count": len(task_successes),
            "task_success_pct": (len(task_successes) / len(records) * 100.0) if records else 0.0,
            "category_counts": dict(counts),
        }

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote extraction sweep to {output_file}")


if __name__ == "__main__":
    main()
