#!/usr/bin/env python3
"""
Build config files and a resumable server run script for the heavy rebuttal suite.

The manifest is the source of truth. This script expands model x benchmark x
decoding x prompt-style combinations into YAML configs and a shell runner that
can execute one phase at a time on the GPU server.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class Job:
    phase: str
    job_id: str
    label: str
    model_key: str
    benchmark_key: str
    decoding_key: str
    humaneval_prompt_style: str
    config_path: Path
    results_dir: str


def slug(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("@", "at")
    )


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_config(
    manifest: Dict[str, Any],
    phase_name: str,
    model_key: str,
    benchmark_key: str,
    decoding_key: str,
    humaneval_prompt_style: str,
    results_dir: str,
) -> Dict[str, Any]:
    defaults = manifest["defaults"]
    model = manifest["models"][model_key]
    benchmark = manifest["benchmarks"][benchmark_key]
    decoding = manifest["decoding"][decoding_key]

    dataset_name = benchmark["dataset"]
    if dataset_name == "mbppplus":
        # The clean EvalPlus generator uses its own task loader but still needs
        # a stable config-shaped dataset name.
        dataset_name = "mbpp"

    dataset_options = copy.deepcopy(benchmark.get("dataset_options", {}))
    if benchmark["dataset"] == "humaneval":
        dataset_options["humaneval_prompt_style"] = humaneval_prompt_style

    config: Dict[str, Any] = {
        "project": {
            "name": manifest["project"]["name"],
            "version": "0.2.0",
            "phase": phase_name,
        },
        "paths": merge_dict(
            defaults["paths"],
            {
                "results_dir": results_dir,
            },
        ),
        "hardware": copy.deepcopy(defaults["hardware"]),
        "model": merge_dict(
            defaults["model"],
            {
                "name": model["hf_name"],
                "pretrained_path": model["hf_name"],
                "temperature": decoding["temperature"],
                "top_p": decoding["top_p"],
                "family": model["family"],
                "parameter_millions": model["parameter_millions"],
                "role": model["role"],
            },
        ),
        "dataset": merge_dict(
            {
                "name": dataset_name,
                "seed": defaults["seed"],
            },
            dataset_options,
        ),
        "generation": copy.deepcopy(defaults["generation"]),
        "execution": copy.deepcopy(defaults["execution"]),
        "feasibility_check": {
            "num_problems": benchmark["num_problems"],
            "num_samples_per_problem": benchmark["num_samples"],
            "decision_threshold": 0.7,
        },
        "heavy_rebuttal": {
            "phase": phase_name,
            "model_key": model_key,
            "benchmark_key": benchmark_key,
            "decoding_key": decoding_key,
            "humaneval_prompt_style": humaneval_prompt_style,
        },
    }

    return config


def expand_jobs(
    manifest: Dict[str, Any],
    config_dir: Path,
    selected_phases: Iterable[str] | None = None,
) -> List[Job]:
    selected = set(selected_phases or manifest["phases"].keys())
    jobs: List[Job] = []

    for phase_name, phase in manifest["phases"].items():
        if phase_name not in selected:
            continue
        for model_key in phase["models"]:
            for benchmark_key in phase["benchmarks"]:
                prompt_styles = phase.get("humaneval_prompt_styles", ["canonical"])
                if manifest["benchmarks"][benchmark_key]["dataset"] != "humaneval":
                    prompt_styles = ["canonical"]
                for decoding_key in phase["decoding"]:
                    for prompt_style in prompt_styles:
                        pieces = [
                            phase_name,
                            model_key,
                            benchmark_key,
                            decoding_key,
                        ]
                        if manifest["benchmarks"][benchmark_key]["dataset"] == "humaneval":
                            pieces.append(prompt_style)
                        job_id = "__".join(slug(piece) for piece in pieces)
                        label = "__".join(
                            slug(piece)
                            for piece in (
                                phase_name,
                                model_key,
                                benchmark_key,
                                decoding_key,
                                prompt_style,
                            )
                        )
                        config_path = config_dir / f"{job_id}.yaml"
                        results_dir = f"data/results_heavy_rebuttal/{phase_name}/{job_id}"
                        jobs.append(
                            Job(
                                phase=phase_name,
                                job_id=job_id,
                                label=label,
                                model_key=model_key,
                                benchmark_key=benchmark_key,
                                decoding_key=decoding_key,
                                humaneval_prompt_style=prompt_style,
                                config_path=config_path,
                                results_dir=results_dir,
                            )
                        )

    return jobs


def write_configs(manifest: Dict[str, Any], jobs: List[Job]) -> None:
    for job in jobs:
        job.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_config(
            manifest=manifest,
            phase_name=job.phase,
            model_key=job.model_key,
            benchmark_key=job.benchmark_key,
            decoding_key=job.decoding_key,
            humaneval_prompt_style=job.humaneval_prompt_style,
            results_dir=job.results_dir,
        )
        with job.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_runner(manifest: Dict[str, Any], jobs: List[Job]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "ROOT=\"${ROOT:-$SCRIPT_DIR}\"",
        "PHASE=\"${PHASE:-core_scaling}\"",
        "FORCE=\"${FORCE:-0}\"",
        "RUN_PROMPT_PPL=\"${RUN_PROMPT_PPL:-0}\"",
        "SMOKE=\"${SMOKE:-0}\"",
        "CONFIRM_PAID_RUN=\"${CONFIRM_PAID_RUN:-0}\"",
        "SMOKE_NUM_PROBLEMS=\"${SMOKE_NUM_PROBLEMS:-2}\"",
        "SMOKE_NUM_SAMPLES=\"${SMOKE_NUM_SAMPLES:-2}\"",
        "VENV_PATH=\"${VENV_PATH:-$ROOT/venv}\"",
        "LCB_DIR=\"${LCB_DIR:-$ROOT/external/LiveCodeBench}\"",
        "RELEASE_VERSION=\"${RELEASE_VERSION:-release_v2}\"",
        "",
        "cd \"$ROOT\"",
        "if [[ \"$SMOKE\" != \"1\" && \"$CONFIRM_PAID_RUN\" != \"1\" ]]; then",
        "  echo \"Refusing full paid run. Set CONFIRM_PAID_RUN=1 after explicit approval, or use SMOKE=1.\"",
        "  exit 2",
        "fi",
        "export PYTHONUNBUFFERED=1",
        "export TOKENIZERS_PARALLELISM=\"${TOKENIZERS_PARALLELISM:-false}\"",
        "export HF_HOME=\"${HF_HOME:-$ROOT/.cache/huggingface}\"",
        "if [[ -n \"${VIRTUAL_ENV:-}\" ]]; then",
        "  echo \"[$(date '+%F %T')] Using active virtualenv: $VIRTUAL_ENV\"",
        "elif [[ -f \"$VENV_PATH/bin/activate\" ]]; then",
        "  source \"$VENV_PATH/bin/activate\"",
        "  echo \"[$(date '+%F %T')] Activated virtualenv: $VENV_PATH\"",
        "else",
        "  echo \"[$(date '+%F %T')] No venv found at $VENV_PATH; using current Python: $(command -v python)\"",
        "fi",
        "mkdir -p outputs/logs outputs/tables outputs/evalplus outputs/livecodebench external",
        "",
        "num_problems_for() {",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"$SMOKE_NUM_PROBLEMS\"",
        "  else",
        "    echo \"$1\"",
        "  fi",
        "}",
        "",
        "num_samples_for() {",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"$SMOKE_NUM_SAMPLES\"",
        "  else",
        "    echo \"$1\"",
        "  fi",
        "}",
        "",
        "ensure_evalplus() {",
        "  if ! python -c \"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('evalplus') else 1)\"; then",
        "    python -m pip install \"evalplus==0.3.1\"",
        "  fi",
        "}",
        "",
        "should_run_phase() {",
        "  local job_phase=\"$1\"",
        "  [[ \"$PHASE\" == \"all\" || \"$PHASE\" == \"$job_phase\" ]]",
        "}",
        "",
        "run_generation() {",
        "  local config=\"$1\"",
        "  local num_problems=\"$2\"",
        "  local num_samples=\"$3\"",
        "  python scripts/generate_samples_safe.py --config \"$config\" --resume --num_problems \"$num_problems\" --num_samples \"$num_samples\"",
        "}",
        "",
        "run_local_eval() {",
        "  local config=\"$1\"",
        "  python scripts/run_evaluation.py --config \"$config\"",
        "}",
        "",
        "run_extraction_sweep() {",
        "  local config=\"$1\"",
        "  local output_file=\"$2\"",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"[$(date '+%F %T')] Smoke mode: skip extraction sweep\"",
        "    return 0",
        "  fi",
        "  if [[ \"$FORCE\" != \"1\" && -s \"$output_file\" ]]; then",
        "    echo \"[$(date '+%F %T')] Skip extraction sweep; exists: $output_file\"",
        "    return 0",
        "  fi",
        "  python scripts/evaluate_extraction_sweep.py --config \"$config\" --output_file \"$output_file\"",
        "}",
        "",
        "run_prompt_ppl() {",
        "  local config=\"$1\"",
        "  local output_file=\"$2\"",
        "  if [[ \"$RUN_PROMPT_PPL\" != \"1\" ]]; then",
        "    return 0",
        "  fi",
        "  if [[ \"$FORCE\" != \"1\" && -s \"$output_file\" ]]; then",
        "    echo \"[$(date '+%F %T')] Skip prompt PPL; exists: $output_file\"",
        "    return 0",
        "  fi",
        "  python scripts/compute_prompt_perplexity.py --config \"$config\" --output_file \"$output_file\"",
        "}",
        "",
        "run_evalplus_rescore() {",
        "  local label=\"$1\"",
        "  local dataset=\"$2\"",
        "  local config=\"$3\"",
        "  local max_samples=\"$4\"",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"[$(date '+%F %T')] Smoke mode: skip EvalPlus scoring\"",
        "    return 0",
        "  fi",
        "  ensure_evalplus",
        "  local case_dir=\"outputs/evalplus/${label}_${dataset}\"",
        "  local samples_file=\"${case_dir}/samples.jsonl\"",
        "  local log_file=\"outputs/logs/evalplus_${label}_${dataset}.log\"",
        "  local summary_file=\"outputs/tables/evalplus_${label}_${dataset}_summary.json\"",
        "  if [[ \"$FORCE\" != \"1\" && -s \"$summary_file\" ]]; then",
        "    echo \"[$(date '+%F %T')] Skip EvalPlus; exists: $summary_file\"",
        "    return 0",
        "  fi",
        "  mkdir -p \"$case_dir\"",
        "  python scripts/export_evalplus_samples.py --config \"$config\" --max_samples_per_task \"$max_samples\" --output_file \"$samples_file\"",
        "  (cd \"$case_dir\" && evalplus.evaluate \"$dataset\" --samples samples.jsonl) | tee \"$log_file\"",
        "  python scripts/analyze_evalplus_results.py --search_root \"$case_dir\" --log_file \"$log_file\" --output_file \"$summary_file\"",
        "}",
        "",
        "run_clean_mbppplus() {",
        "  local label=\"$1\"",
        "  local config=\"$2\"",
        "  local num_problems=\"$3\"",
        "  local num_samples=\"$4\"",
        "  num_problems=\"$(num_problems_for \"$num_problems\")\"",
        "  num_samples=\"$(num_samples_for \"$num_samples\")\"",
        "  ensure_evalplus",
        "  local case_dir=\"outputs/evalplus/${label}_mbppplus\"",
        "  local samples_file=\"${case_dir}/samples.jsonl\"",
        "  local log_file=\"outputs/logs/evalplus_${label}_mbppplus.log\"",
        "  local summary_file=\"outputs/tables/evalplus_${label}_mbppplus_summary.json\"",
        "  if [[ \"$FORCE\" != \"1\" && -s \"$summary_file\" ]]; then",
        "    echo \"[$(date '+%F %T')] Skip MBPP+; exists: $summary_file\"",
        "    return 0",
        "  fi",
        "  python scripts/generate_mbppplus_evalplus.py --config \"$config\" --resume --num_problems \"$num_problems\" --num_samples \"$num_samples\"",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"[$(date '+%F %T')] Smoke mode: generated MBPP+ samples only; skip EvalPlus scoring\"",
        "    return 0",
        "  fi",
        "  mkdir -p \"$case_dir\"",
        "  python scripts/export_evalplus_samples.py --config \"$config\" --output_file \"$samples_file\"",
        "  (cd \"$case_dir\" && evalplus.evaluate mbpp --samples samples.jsonl) | tee \"$log_file\"",
        "  python scripts/analyze_evalplus_results.py --search_root \"$case_dir\" --log_file \"$log_file\" --output_file \"$summary_file\"",
        "}",
        "",
        "ensure_livecodebench() {",
        "  if [[ ! -f \"data/raw/livecodebench_${RELEASE_VERSION}.jsonl\" ]]; then",
        "    python scripts/download_data.py --dataset livecodebench --version_tag \"$RELEASE_VERSION\"",
        "  fi",
        "  if [[ ! -d \"$LCB_DIR/.git\" ]]; then",
        "    git clone https://github.com/LiveCodeBench/LiveCodeBench.git \"$LCB_DIR\"",
        "  fi",
        "}",
        "",
        "run_livecodebench_case() {",
        "  local label=\"$1\"",
        "  local config=\"$2\"",
        "  local num_problems=\"$3\"",
        "  local num_samples=\"$4\"",
        "  num_problems=\"$(num_problems_for \"$num_problems\")\"",
        "  num_samples=\"$(num_samples_for \"$num_samples\")\"",
        "  local output_json=\"$ROOT/outputs/livecodebench/${label}_custom_outputs.json\"",
        "  local log_file=\"$ROOT/outputs/logs/livecodebench_${label}.log\"",
        "  local summary_file=\"$ROOT/outputs/tables/livecodebench_${label}_summary.json\"",
        "  if [[ \"$FORCE\" != \"1\" && -s \"$summary_file\" ]]; then",
        "    echo \"[$(date '+%F %T')] Skip LiveCodeBench; exists: $summary_file\"",
        "    return 0",
        "  fi",
        "  ensure_livecodebench",
        "  run_generation \"$config\" \"$num_problems\" \"$num_samples\"",
        "  if [[ \"$SMOKE\" == \"1\" ]]; then",
        "    echo \"[$(date '+%F %T')] Smoke mode: generated LiveCodeBench samples only; skip external scoring\"",
        "    return 0",
        "  fi",
        "  python scripts/export_livecodebench_custom_outputs.py --config \"$config\" --output_file \"$output_json\"",
        "  (cd \"$LCB_DIR\" && PYTHONPATH=\"$LCB_DIR:${PYTHONPATH:-}\" python -m lcb_runner.runner.custom_evaluator --custom_output_file \"$output_json\" --release_version \"$RELEASE_VERSION\") | tee \"$log_file\"",
        "  local eval_all_file",
        "  eval_all_file=\"$(find \"$LCB_DIR\" -type f -name '*eval_all*.json' | sort | tail -n 1 || true)\"",
        "  if [[ -n \"$eval_all_file\" ]]; then",
        "    (cd \"$LCB_DIR\" && PYTHONPATH=\"$LCB_DIR:${PYTHONPATH:-}\" python -m lcb_runner.evaluation.compute_scores --eval_all_file \"$eval_all_file\") | tee -a \"$log_file\" || true",
        "  fi",
        "  python scripts/summarize_livecodebench_scores.py --search_root \"$LCB_DIR\" --log_file \"$log_file\" --output_file \"$summary_file\"",
        "}",
        "",
        "echo \"[$(date '+%F %T')] Starting heavy rebuttal suite phase=$PHASE\"",
        "",
    ]

    benchmark_lookup = manifest["benchmarks"]
    for job in jobs:
        benchmark = benchmark_lookup[job.benchmark_key]
        config = str(job.config_path).replace("\\", "/")
        num_problems = str(benchmark["num_problems"])
        num_samples = str(benchmark["num_samples"])
        label = job.label
        job_num_problems = f"$(num_problems_for {num_problems})"
        job_num_samples = f"$(num_samples_for {num_samples})"

        lines.extend(
            [
                f"if should_run_phase {shell_quote(job.phase)}; then",
                f"  echo \"[$(date '+%F %T')] Job {job.job_id}\"",
            ]
        )

        if benchmark.get("clean_evalplus_generation"):
            lines.append(
                f"  run_clean_mbppplus {shell_quote(label)} {shell_quote(config)} {num_problems} {num_samples}"
            )
        elif benchmark.get("livecodebench_evaluation"):
            lines.append(
                f"  run_livecodebench_case {shell_quote(label)} {shell_quote(config)} {num_problems} {num_samples}"
            )
        else:
            lines.append(
                f"  run_generation {shell_quote(config)} {job_num_problems} {job_num_samples}"
            )
            if benchmark.get("local_evaluation"):
                lines.append(f"  run_local_eval {shell_quote(config)}")
            if benchmark.get("extraction_sweep"):
                output_file = f"outputs/tables/heavy_rebuttal/{label}_extraction_sweep.json"
                lines.append(
                    f"  run_extraction_sweep {shell_quote(config)} {shell_quote(output_file)}"
                )
            if benchmark.get("dataset") in {"humaneval", "mbpp"}:
                ppl_file = f"outputs/tables/heavy_rebuttal/{label}_prompt_perplexity.json"
                lines.append(
                    f"  run_prompt_ppl {shell_quote(config)} {shell_quote(ppl_file)}"
                )
            if benchmark.get("evalplus_dataset"):
                evalplus_dataset = benchmark["evalplus_dataset"]
                lines.append(
                    f"  run_evalplus_rescore {shell_quote(label)} {shell_quote(evalplus_dataset)} {shell_quote(config)} {job_num_samples}"
                )

        lines.extend(["fi", ""])

    lines.append("echo \"[$(date '+%F %T')] Heavy rebuttal suite completed\"")
    lines.append("")
    return "\n".join(lines)


def write_job_index(jobs: List[Job], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "phase": job.phase,
            "job_id": job.job_id,
            "label": job.label,
            "model_key": job.model_key,
            "benchmark_key": job.benchmark_key,
            "decoding_key": job.decoding_key,
            "humaneval_prompt_style": job.humaneval_prompt_style,
            "config_path": str(job.config_path).replace("\\", "/"),
            "results_dir": job.results_dir,
        }
        for job in jobs
    ]
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/heavy_rebuttal_manifest.yaml",
        help="Manifest YAML path.",
    )
    parser.add_argument(
        "--config_dir",
        default="configs/heavy_rebuttal",
        help="Directory where generated configs are written.",
    )
    parser.add_argument(
        "--runner",
        default="run_heavy_rebuttal_suite.sh",
        help="Output shell runner path.",
    )
    parser.add_argument(
        "--job_index",
        default="outputs/tables/heavy_rebuttal_job_index.json",
        help="Output JSON index of generated jobs.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        default=None,
        help="Limit generation to a phase. May be repeated.",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    config_dir = Path(args.config_dir)
    jobs = expand_jobs(manifest, config_dir, selected_phases=args.phase)
    write_configs(manifest, jobs)

    runner_path = Path(args.runner)
    runner_path.write_text(render_runner(manifest, jobs), encoding="utf-8")
    write_job_index(jobs, Path(args.job_index))

    phase_counts: Dict[str, int] = {}
    for job in jobs:
        phase_counts[job.phase] = phase_counts.get(job.phase, 0) + 1

    print(f"Wrote {len(jobs)} configs to {config_dir}")
    print(f"Wrote runner to {runner_path}")
    print(f"Wrote job index to {args.job_index}")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} jobs")


if __name__ == "__main__":
    main()
