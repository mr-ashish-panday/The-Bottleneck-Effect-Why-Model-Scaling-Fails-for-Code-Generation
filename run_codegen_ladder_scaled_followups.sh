#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ashish/paper11_code_execution_failures}"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

NUM_PROBLEMS="${NUM_PROBLEMS:-20}"
SAMPLES_PER_PROBLEM="${SAMPLES_PER_PROBLEM:-10}"
SCALES="${SCALES:-0.75,0.5,0.25,0.0}"

run_followup() {
  local config="$1"
  local layer="$2"
  local prefix="$3"

  echo "[$(date '+%F %T')] Starting scaled follow-up for ${prefix} layer ${layer}"

  python scripts/scaled_layer_ablation.py \
    --config "${config}" \
    --model_config "${config}" \
    --layers "${layer}" \
    --scales "${SCALES}" \
    --num_problems "${NUM_PROBLEMS}" \
    --samples_per_problem "${SAMPLES_PER_PROBLEM}" \
    --output_file "data/${prefix}/ablation/scaled_layer${layer}_${NUM_PROBLEMS}x${SAMPLES_PER_PROBLEM}.json"

  python scripts/analyze_scaled_ablation.py \
    --input_file "data/${prefix}/ablation/scaled_layer${layer}_${NUM_PROBLEMS}x${SAMPLES_PER_PROBLEM}.json" \
    --output_file "data/${prefix}/ablation/scaled_layer${layer}_${NUM_PROBLEMS}x${SAMPLES_PER_PROBLEM}_summary.json"
}

echo "[$(date '+%F %T')] Starting CodeGen ladder scaled follow-ups"
run_followup "config_codegen_nl.yaml" "11" "results_codegen_nl"
run_followup "config_codegen_multi.yaml" "7" "results_codegen_multi"
run_followup "config_codegen_mono.yaml" "13" "results_codegen_mono"
echo "[$(date '+%F %T')] CodeGen ladder scaled follow-ups finished"
