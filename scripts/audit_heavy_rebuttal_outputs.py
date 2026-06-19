#!/usr/bin/env python3
"""
Audit heavy rebuttal run coverage without re-running expensive jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_generated(results_file: Path) -> Dict[str, int]:
    if not results_file.exists():
        return {"tasks": 0, "samples": 0, "empty_tasks": 0}

    with results_file.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    tasks = len(records)
    samples = 0
    empty_tasks = 0
    for record in records:
        sample_count = len(record.get("samples", []))
        samples += sample_count
        if sample_count == 0:
            empty_tasks += 1

    return {"tasks": tasks, "samples": samples, "empty_tasks": empty_tasks}


def audit_config(config_path: Path) -> Dict[str, Any]:
    config = load_yaml(config_path)
    meta = config.get("heavy_rebuttal", {})
    results_dir = Path(config["paths"]["results_dir"])
    expected_tasks = int(config["feasibility_check"]["num_problems"])
    expected_samples_per_task = int(config["feasibility_check"]["num_samples_per_problem"])
    expected_samples = expected_tasks * expected_samples_per_task

    generated_file = results_dir / "generated_samples.json"
    evaluation_file = results_dir / "evaluation_results.json"
    counts = count_generated(generated_file)

    status = "missing"
    if counts["tasks"] == expected_tasks and counts["samples"] == expected_samples and counts["empty_tasks"] == 0:
        status = "complete"
    elif counts["tasks"] > 0:
        status = "partial"

    return {
        "status": status,
        "config": str(config_path).replace("\\", "/"),
        "phase": meta.get("phase"),
        "model": meta.get("model_key"),
        "benchmark": meta.get("benchmark_key"),
        "decoding": meta.get("decoding_key"),
        "prompt_style": meta.get("humaneval_prompt_style"),
        "results_dir": str(results_dir).replace("\\", "/"),
        "expected_tasks": expected_tasks,
        "actual_tasks": counts["tasks"],
        "expected_samples": expected_samples,
        "actual_samples": counts["samples"],
        "empty_tasks": counts["empty_tasks"],
        "has_generated": generated_file.exists(),
        "has_local_eval": evaluation_file.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs/heavy_rebuttal")
    parser.add_argument("--phase", default=None)
    parser.add_argument("--output_file", default="outputs/tables/heavy_rebuttal_coverage_audit.json")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    rows: List[Dict[str, Any]] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        row = audit_config(config_path)
        if args.phase and row["phase"] != args.phase:
            continue
        rows.append(row)

    summary: Dict[str, Any] = {
        "config_dir": str(config_dir).replace("\\", "/"),
        "phase": args.phase,
        "total_jobs": len(rows),
        "status_counts": {},
        "jobs": rows,
    }
    for row in rows:
        summary["status_counts"][row["status"]] = summary["status_counts"].get(row["status"], 0) + 1

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Audited {len(rows)} jobs")
    for status, count in sorted(summary["status_counts"].items()):
        print(f"  {status}: {count}")
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
