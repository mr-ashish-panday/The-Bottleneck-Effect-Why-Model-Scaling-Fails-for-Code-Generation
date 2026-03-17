#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

NUM_PROBLEMS="${NUM_PROBLEMS:-164}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

repair_one() {
  local config="$1"
  local task_file="$2"

  echo "[$(date '+%F %T')] Auditing gaps for ${config}"

  if [[ ! -s "${task_file}" ]]; then
    echo "[$(date '+%F %T')] No repair tasks listed for ${config}; skipping generation"
  else
    echo "[$(date '+%F %T')] Repairing $(wc -l < "${task_file}") task(s) for ${config}"
    python scripts/generate_samples_safe.py \
      --config "${config}" \
      --resume \
      --num_problems "${NUM_PROBLEMS}" \
      --num_samples "${NUM_SAMPLES}" \
      --task_ids_file "${task_file}" \
      --force_selected
  fi

  python scripts/run_evaluation.py --config "${config}"
  python scripts/analyze_failures.py --config "${config}"
  python scripts/deep_syntax_analysis.py --config "${config}"
}

python scripts/identify_generation_gaps.py \
  --config config_codegen.yaml \
  --num_problems "${NUM_PROBLEMS}" \
  --expected_samples "${NUM_SAMPLES}" \
  --task_ids_out outputs/tables/codegen_main_repair_tasks.txt > outputs/tables/codegen_main_repair_report.json

python scripts/identify_generation_gaps.py \
  --config config_codegen_nl.yaml \
  --num_problems "${NUM_PROBLEMS}" \
  --expected_samples "${NUM_SAMPLES}" \
  --task_ids_out outputs/tables/codegen_nl_repair_tasks.txt > outputs/tables/codegen_nl_repair_report.json

python scripts/identify_generation_gaps.py \
  --config config_codegen_multi.yaml \
  --num_problems "${NUM_PROBLEMS}" \
  --expected_samples "${NUM_SAMPLES}" \
  --task_ids_out outputs/tables/codegen_multi_repair_tasks.txt > outputs/tables/codegen_multi_repair_report.json

python scripts/identify_generation_gaps.py \
  --config config_codegen_mono.yaml \
  --num_problems "${NUM_PROBLEMS}" \
  --expected_samples "${NUM_SAMPLES}" \
  --task_ids_out outputs/tables/codegen_mono_repair_tasks.txt > outputs/tables/codegen_mono_repair_report.json

repair_one "config_codegen.yaml" "outputs/tables/codegen_main_repair_tasks.txt"
repair_one "config_codegen_nl.yaml" "outputs/tables/codegen_nl_repair_tasks.txt"
repair_one "config_codegen_multi.yaml" "outputs/tables/codegen_multi_repair_tasks.txt"
repair_one "config_codegen_mono.yaml" "outputs/tables/codegen_mono_repair_tasks.txt"

python scripts/bootstrap_significance.py \
  --model "GPT-2 (124M)=data/results_gpt2" \
  --model "GPT-2 Medium (355M)=data/results_gpt2_medium" \
  --model "CodeGen-350M=data/results_codegen" \
  --pass_k 1 \
  --pass_k 10 \
  --pass_k 100 \
  --output_file outputs/tables/bootstrap_significance.json

python scripts/bootstrap_significance.py \
  --model "CodeGen-NL=data/results_codegen_nl" \
  --model "CodeGen-Multi=data/results_codegen_multi" \
  --model "CodeGen-Mono=data/results_codegen_mono" \
  --pass_k 1 \
  --pass_k 5 \
  --pass_k 10 \
  --output_file outputs/tables/bootstrap_significance_codegen_ladder.json

python scripts/summarize_codegen_ladder.py \
  --model "CodeGen-NL=data/results_codegen_nl" \
  --model "CodeGen-Multi=data/results_codegen_multi" \
  --model "CodeGen-Mono=data/results_codegen_mono" \
  --output_file outputs/tables/codegen_ladder_summary.json

echo "[$(date '+%F %T')] CodeGen HumanEval repairs completed"
