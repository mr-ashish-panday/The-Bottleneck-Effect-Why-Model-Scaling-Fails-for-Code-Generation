#!/usr/bin/env python3
"""
Summarize the CodeGen pretraining ladder (NL -> Multi -> Mono).
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_row(label: str, results_dir: Path):
    feasibility = load_json(results_dir / "feasibility_report.json")
    syntax = load_json(results_dir / "syntax_analysis.json")

    category_distribution = feasibility["category_distribution"]
    total = sum(category_distribution.values())

    row = {
        "label": label,
        "results_dir": str(results_dir),
        "success_pct": category_distribution.get("success", 0) / total * 100 if total else 0.0,
        "syntax_pct": category_distribution.get("syntax_error", 0) / total * 100 if total else 0.0,
        "runtime_pct": category_distribution.get("runtime_error", 0) / total * 100 if total else 0.0,
        "timeout_pct": category_distribution.get("timeout", 0) / total * 100 if total else 0.0,
        "top_syntax_errors": sorted(
            syntax["category_distribution"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5],
    }
    return row


def parse_model_arg(raw_value: str):
    label, path = raw_value.split("=", 1)
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Format: Label=path/to/results_dir",
    )
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    rows = []
    for raw_model in args.model:
        label, results_dir = parse_model_arg(raw_model)
        rows.append(compute_row(label, results_dir))

    summary = {"models": rows}
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote CodeGen ladder summary to {output_path}")
    for row in rows:
        print(
            f"{row['label']}: success={row['success_pct']:.2f}% "
            f"syntax={row['syntax_pct']:.2f}% runtime={row['runtime_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
