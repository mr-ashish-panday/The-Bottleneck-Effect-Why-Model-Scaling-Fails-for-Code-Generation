#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

HUMANEVAL_PROBLEMS="${HUMANEVAL_PROBLEMS:-164}"
MBPP_PROBLEMS="${MBPP_PROBLEMS:-257}"

run_case() {
  local config="$1"
  local num_problems="$2"
  local output_file="$3"

  echo "[$(date '+%F %T')] Prompt perplexity for ${config}"
  python scripts/compute_prompt_perplexity.py \
    --config "${config}" \
    --num_problems "${num_problems}" \
    --output_file "${output_file}"
}

run_case "config_codegen_nl.yaml" "${HUMANEVAL_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_nl_humaneval.json"
run_case "config_codegen_multi.yaml" "${HUMANEVAL_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_multi_humaneval.json"
run_case "config_codegen_mono.yaml" "${HUMANEVAL_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_mono_humaneval.json"

run_case "config_mbpp_full_codegen_nl.yaml" "${MBPP_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_nl_mbpp.json"
run_case "config_mbpp_full_codegen_multi.yaml" "${MBPP_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_multi_mbpp.json"
run_case "config_mbpp_full_codegen_mono.yaml" "${MBPP_PROBLEMS}" "outputs/tables/prompt_ppl_codegen_mono_mbpp.json"

echo "[$(date '+%F %T')] CodeGen ladder prompt perplexity completed"
