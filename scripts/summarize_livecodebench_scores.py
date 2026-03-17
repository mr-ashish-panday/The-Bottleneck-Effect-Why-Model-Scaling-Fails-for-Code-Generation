#!/usr/bin/env python3
"""
Summarize LiveCodeBench score artifacts and logs into a compact JSON report.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KEYWORDS = ("pass", "score", "release", "total", "correct")


def flatten_metrics(obj: Any, prefix: str = "") -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_metrics(value, child_prefix))
    elif isinstance(obj, list):
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

    parsed: Dict[str, Any] = {}
    metric_pattern = re.compile(
        r"(?P<label>pass@1|pass@5|score[^:\n]*)[:=]\s*(?P<value>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for line in log_file.read_text().splitlines():
        for match in metric_pattern.finditer(line):
            parsed[match.group("label").strip()] = float(match.group("value"))
    return parsed


def infer_label(log_file: Optional[Path]) -> Optional[str]:
    if log_file is None:
        return None

    match = re.match(r"livecodebench_(.+)", log_file.stem)
    if not match:
        return None

    return match.group(1)


def iter_candidate_files(search_root: Path) -> Iterable[Path]:
    seen = set()
    roots = [search_root]

    fallback_root = Path.cwd() / "outputs" / "livecodebench"
    if fallback_root != search_root:
        roots.append(fallback_root)

    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*.json")):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def extract_eval_metrics(candidate: Path, payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        return None

    first = payload[0]
    if not isinstance(first, dict):
        return None

    metrics: Dict[str, Any] = {}
    for key in ("pass@1", "pass@5", "pass@10"):
        value = first.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    if candidate.name.endswith("_eval_all.json"):
        pass_values: List[float] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            value = row.get("pass@1")
            if isinstance(value, bool):
                pass_values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                pass_values.append(float(value))
        if pass_values:
            metrics["tasks_scored"] = len(pass_values)
            metrics["pass@1_mean"] = sum(pass_values) / len(pass_values)

    return metrics or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_root", required=True, help="Directory to search")
    parser.add_argument("--log_file", default=None, help="Optional LiveCodeBench log")
    parser.add_argument("--output_file", required=True, help="Summary JSON path")
    args = parser.parse_args()

    search_root = Path(args.search_root)
    log_file = Path(args.log_file) if args.log_file else None
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    label = infer_label(log_file)
    name_prefix = f"{label}_custom_outputs" if label else None

    for candidate in iter_candidate_files(search_root):
        if name_prefix and not candidate.name.startswith(name_prefix):
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue

        metrics = extract_eval_metrics(candidate, payload)
        if metrics is None:
            metrics = flatten_metrics(payload)
        if metrics:
            summaries.append({
                "file": str(candidate),
                "metrics": metrics,
            })

    summary = {
        "search_root": str(search_root),
        "label": label,
        "json_summaries": summaries,
        "log_metrics": parse_log(log_file) if log_file else {},
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote LiveCodeBench summary to {output_file}")


if __name__ == "__main__":
    main()
