#!/usr/bin/env python3
"""
Shared helpers for benchmark-summary figures.
"""

import json
import re
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "figures"

MAIN_MODEL_ALIASES = {
    "GPT-2 (124M)": "GPT-2 Small",
    "GPT-2 Medium (355M)": "GPT-2 Medium",
    "CodeGen-350M": "CodeGen",
    "GPT-2 MBPP": "GPT-2 Small",
    "GPT-2 Medium MBPP": "GPT-2 Medium",
    "CodeGen MBPP": "CodeGen",
}

MAIN_MODEL_COLORS = {
    "GPT-2 Small": "#495057",
    "GPT-2 Medium": "#e67700",
    "CodeGen": "#1c7ed6",
}

LADDER_COLORS = {
    "CodeGen-NL": "#868e96",
    "CodeGen-Multi": "#1971c2",
    "CodeGen-Mono": "#2f9e44",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def alias_label(raw_label: str, aliases: Dict[str, str]) -> str:
    return aliases.get(raw_label, raw_label)


def bootstrap_metric_rows(path: Path, metric_name: str, aliases: Dict[str, str]) -> List[dict]:
    payload = load_json(path)
    rows = []
    for raw_label, model_data in payload["models"].items():
        metric = model_data["metrics"][metric_name]
        rows.append(
            {
                "raw_label": raw_label,
                "label": alias_label(raw_label, aliases),
                "mean": float(metric["mean"]) * 100.0,
                "ci_low": float(metric["ci_low"]) * 100.0,
                "ci_high": float(metric["ci_high"]) * 100.0,
                "num_tasks": int(model_data["num_tasks"]),
                "samples_per_task": int(model_data["samples_per_task"]),
            }
        )
    return rows


def bootstrap_pairwise_rows(path: Path, metric_name: str, aliases: Dict[str, str]) -> List[dict]:
    payload = load_json(path)
    rows = []
    for test in payload["pairwise_tests"]:
        if test["metric"] != metric_name:
            continue
        model_a = alias_label(test["model_a"], aliases)
        model_b = alias_label(test["model_b"], aliases)
        rows.append(
            {
                "label": f"{model_a} - {model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "difference": float(test["difference"]) * 100.0,
                "ci_low": float(test["ci_low"]) * 100.0,
                "ci_high": float(test["ci_high"]) * 100.0,
                "p_value": float(test["p_value"]),
                "num_shared_tasks": int(test["num_shared_tasks"]),
            }
        )
    return rows


def parse_evalplus_log(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    current_prefix = None
    pattern = re.compile(r"pass@(\d+):\s*([0-9]*\.?[0-9]+)")
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lower()
        if "(base tests)" in line:
            current_prefix = "base"
            continue
        if "(base + extra tests)" in line:
            current_prefix = "plus"
            continue
        match = pattern.search(line)
        if match and current_prefix:
            metric_name = f"{current_prefix}_pass@{match.group(1)}"
            metrics[metric_name] = float(match.group(2)) * 100.0
    return metrics


def load_livecodebench_pass_metrics(summary_path: Path) -> Dict[str, float]:
    payload = load_json(summary_path)
    for summary in payload["json_summaries"]:
        metrics = summary["metrics"]
        if "pass@1" in metrics:
            return {name: float(value) * 100.0 for name, value in metrics.items() if name.startswith("pass@")}
    raise ValueError(f"No pass metrics found in {summary_path}")


def evaluation_results_coverage(path: Path, expected_samples_per_task: int = 100) -> dict:
    rows = load_json(path)
    total_tasks = len(rows)
    total_samples = sum(len(row.get("samples", [])) for row in rows)
    zero_tasks = [row["task_id"] for row in rows if len(row.get("samples", [])) == 0]
    short_tasks = [
        row["task_id"]
        for row in rows
        if 0 < len(row.get("samples", [])) < expected_samples_per_task
    ]
    return {
        "tasks": total_tasks,
        "samples": total_samples,
        "expected_samples": total_tasks * expected_samples_per_task,
        "zero_tasks": zero_tasks,
        "short_tasks": short_tasks,
    }


def repair_report_summary(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(raw_text)
    missing = payload.get("missing_task_ids", [])
    short = payload.get("short_task_ids", [])
    repair = payload.get("repair_task_ids", [])
    return {
        "missing_count": len(missing),
        "short_count": len(short),
        "repair_count": len(repair),
        "missing_task_ids": missing,
        "short_task_ids": short,
        "repair_task_ids": repair,
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
