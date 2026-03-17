#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

DISCOVERY_PROBLEMS="${DISCOVERY_PROBLEMS:-50}"
DISCOVERY_SAMPLES="${DISCOVERY_SAMPLES:-20}"
FOLLOWUP_PROBLEMS="${FOLLOWUP_PROBLEMS:-10}"
FOLLOWUP_SAMPLES="${FOLLOWUP_SAMPLES:-5}"
FOLLOWUP_SCALES="${FOLLOWUP_SCALES:-0.75,0.5,0.25,0.0}"

echo "[$(date '+%F %T')] Starting CodeGen ladder mechanism rerun"

for config in "config_codegen_nl.yaml" "config_codegen_multi.yaml" "config_codegen_mono.yaml"; do
  echo "[$(date '+%F %T')] Full layer ablation discovery for ${config}"
  python scripts/layer_ablation.py \
    --config "${config}" \
    --model_config "${config}" \
    --num_problems "${DISCOVERY_PROBLEMS}" \
    --samples_per_problem "${DISCOVERY_SAMPLES}"
done

run_scaled_if_requested() {
  local config="$1"
  local layers="$2"
  local output_file="$3"
  local label="$4"

  if [[ -z "$layers" ]]; then
    return 0
  fi

  echo "[$(date '+%F %T')] Targeted scaled ablation for ${label}: layers ${layers}"
  python scripts/scaled_layer_ablation.py \
    --config "${config}" \
    --model_config "${config}" \
    --layers "${layers}" \
    --scales "${FOLLOWUP_SCALES}" \
    --num_problems "${FOLLOWUP_PROBLEMS}" \
    --samples_per_problem "${FOLLOWUP_SAMPLES}" \
    --output_file "${output_file}"
}

run_scaled_if_requested \
  "config_codegen_nl.yaml" \
  "${NL_LAYERS:-}" \
  "data/results_codegen_nl/ablation/scaled_followup.json" \
  "results_codegen_nl"
run_scaled_if_requested \
  "config_codegen_multi.yaml" \
  "${MULTI_LAYERS:-}" \
  "data/results_codegen_multi/ablation/scaled_followup.json" \
  "results_codegen_multi"
run_scaled_if_requested \
  "config_codegen_mono.yaml" \
  "${MONO_LAYERS:-}" \
  "data/results_codegen_mono/ablation/scaled_followup.json" \
  "results_codegen_mono"

echo "[$(date '+%F %T')] CodeGen ladder mechanism rerun finished"
