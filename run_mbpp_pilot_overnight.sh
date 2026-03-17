#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs

NUM_PROBLEMS="${NUM_PROBLEMS:-30}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"

echo "[$(date '+%F %T')] Starting MBPP overnight pilot"
echo "[$(date '+%F %T')] Problems: ${NUM_PROBLEMS} | Samples/problem: ${NUM_SAMPLES}"

if [ ! -f data/raw/mbpp.jsonl ]; then
  echo "[$(date '+%F %T')] MBPP missing, downloading dataset"
  python scripts/download_data.py --dataset mbpp
else
  echo "[$(date '+%F %T')] MBPP dataset already present"
fi

CONFIGS=(
  "config_mbpp_gpt2.yaml"
  "config_mbpp_gpt2_medium.yaml"
  "config_mbpp_codegen.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Running pipeline for ${config}"
  python scripts/generate_samples.py \
    --config "${config}" \
    --num_problems "${NUM_PROBLEMS}" \
    --num_samples "${NUM_SAMPLES}"

  python scripts/run_evaluation.py --config "${config}"
  python scripts/analyze_failures.py --config "${config}"
  python scripts/deep_syntax_analysis.py --config "${config}"
done

echo "[$(date '+%F %T')] MBPP overnight pilot completed"
