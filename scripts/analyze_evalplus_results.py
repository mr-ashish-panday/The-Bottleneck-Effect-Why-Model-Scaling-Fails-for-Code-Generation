#!/usr/bin/env python3
"""
Summarize EvalPlus result artifacts and logs into a compact JSON report.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


KEYWORDS = ("pass", "base", "plus", "score", "total", "correct")


def flatten_metrics(obj: Any, prefix: str = "") -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_metrics(value, child_prefix))
    elif isinstance(obj, list):
        if obj and all(isinstance(item, (int, float, str, bool)) for item in obj):
            if any(keyword in prefix.lower() for keyword in KEYWORDS):
                metrics[prefix] = obj
        else:
            for index, value in enumerate(obj):
                child_prefix = f"{prefix}[{index}]"
                metrics.update(flatten_metrics(value, child_prefix))
    elif isinstance(obj, (int, float, str, bool)):
        if any(keyword in prefix.lower() for keyword in KEYWORDS):
            metrics[prefix] = obj
    return metrics


def parse_log(log_file: Path) -> Dict[str, Any]:
    if not log_file.exists():
        return {}

    metric_pattern = re.compile(
        r"(?P<label>(base|plus|pass@1|pass@5|pass@10)[^:\n]*)[:=]\s*(?P<value>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    parsed: Dict[str, Any] = {}
    for line in log_file.read_text().splitlines():
        for match in metric_pattern.finditer(line):
            parsed[match.group("label").strip()] = float(match.group("value"))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_root", required=True, help="Directory to search")
    parser.add_argument("--log_file", default=None, help="Optional EvalPlus log file")
    parser.add_argument("--output_file", required=True, help="Summary JSON path")
    args = parser.parse_args()

    search_root = Path(args.search_root)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for candidate in sorted(search_root.rglob("*.json")):
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        metrics = flatten_metrics(payload)
        if metrics:
            summaries.append({
                "file": str(candidate),
                "metrics": metrics,
            })

    summary = {
        "search_root": str(search_root),
        "json_summaries": summaries,
        "log_metrics": parse_log(Path(args.log_file)) if args.log_file else {},
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote EvalPlus summary to {output_file}")


if __name__ == "__main__":
    main()
