#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables outputs/evalplus

NUM_PROBLEMS="${NUM_PROBLEMS:-378}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

if ! python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('evalplus') else 1)"; then
  python -m pip install "evalplus==0.3.1"
fi

run_case() {
  local label="$1"
  local config="$2"
  local case_dir="outputs/evalplus/${label}_mbppplus"
  local samples_file="${case_dir}/samples.jsonl"
  local log_file="outputs/logs/evalplus_${label}_mbppplus.log"
  local summary_file="outputs/tables/evalplus_${label}_mbppplus_summary.json"

  echo "[$(date '+%F %T')] Generating clean MBPP+ samples for ${label}"
  python scripts/generate_mbppplus_evalplus.py \
    --config "$config" \
    --resume \
    --num_problems "$NUM_PROBLEMS" \
    --num_samples "$NUM_SAMPLES"

  mkdir -p "$case_dir"
  python scripts/export_evalplus_samples.py \
    --config "$config" \
    --output_file "$samples_file"

  echo "[$(date '+%F %T')] Running EvalPlus MBPP+ scoring for ${label}"
  (
    cd "$case_dir"
    evalplus.evaluate mbpp --samples "samples.jsonl"
  ) | tee "$log_file"

  python scripts/analyze_evalplus_results.py \
    --search_root "$case_dir" \
    --log_file "$log_file" \
    --output_file "$summary_file"
}

echo "[$(date '+%F %T')] Starting clean MBPP+ validation"

run_case "gpt2" "config_mbppplus_gpt2.yaml"
run_case "gpt2_medium" "config_mbppplus_gpt2_medium.yaml"
run_case "codegen" "config_mbppplus_codegen.yaml"

echo "[$(date '+%F %T')] Clean MBPP+ validation completed"
