#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables outputs/evalplus

if ! python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('evalplus') else 1)"; then
  python -m pip install "evalplus==0.2.1"
fi

export EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-20}"

run_case() {
  local label="$1"
  local dataset="$2"
  local config="$3"
  local case_dir="outputs/evalplus/${label}_${dataset}"
  local samples_file="${case_dir}/samples.jsonl"
  local log_file="outputs/logs/evalplus_${label}_${dataset}.log"
  local summary_file="outputs/tables/evalplus_${label}_${dataset}_summary.json"

  mkdir -p "$case_dir"

  echo "[$(date '+%F %T')] Exporting ${label} ${dataset} samples"
  python scripts/export_evalplus_samples.py \
    --config "$config" \
    --output_file "$samples_file"

  echo "[$(date '+%F %T')] Running EvalPlus for ${label} ${dataset}"
  (
    cd "$case_dir"
    evalplus.evaluate --dataset "$dataset" --samples "samples.jsonl"
  ) | tee "$log_file"

  python scripts/analyze_evalplus_results.py \
    --search_root "$case_dir" \
    --log_file "$log_file" \
    --output_file "$summary_file"
}

echo "[$(date '+%F %T')] Starting EvalPlus rescoring"

run_case "gpt2" "humaneval" "config.yaml"
run_case "gpt2_medium" "humaneval" "config_gpt2_medium.yaml"
run_case "codegen" "humaneval" "config_codegen.yaml"
run_case "gpt2" "mbpp" "config_mbpp_full_gpt2.yaml"
run_case "gpt2_medium" "mbpp" "config_mbpp_full_gpt2_medium.yaml"
run_case "codegen" "mbpp" "config_mbpp_full_codegen.yaml"

echo "[$(date '+%F %T')] EvalPlus rescoring completed"
