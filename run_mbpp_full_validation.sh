#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

NUM_PROBLEMS="${NUM_PROBLEMS:-257}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

echo "[$(date '+%F %T')] Starting full-coverage MBPP validation"
echo "[$(date '+%F %T')] Problems: ${NUM_PROBLEMS} | Samples/problem: ${NUM_SAMPLES}"

if [ ! -f data/raw/mbpp.jsonl ]; then
  echo "[$(date '+%F %T')] MBPP missing, downloading dataset"
  python scripts/download_data.py --dataset mbpp
else
  echo "[$(date '+%F %T')] MBPP dataset already present"
fi

CONFIGS=(
  "config_mbpp_full_gpt2.yaml"
  "config_mbpp_full_gpt2_medium.yaml"
  "config_mbpp_full_codegen.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Running safe generation for ${config}"
  python scripts/generate_samples_safe.py \
    --config "${config}" \
    --resume \
    --num_problems "${NUM_PROBLEMS}" \
    --num_samples "${NUM_SAMPLES}"

  echo "[$(date '+%F %T')] Evaluating ${config}"
  python scripts/run_evaluation.py --config "${config}"
  python scripts/analyze_failures.py --config "${config}"
  python scripts/deep_syntax_analysis.py --config "${config}"
done

python scripts/bootstrap_significance.py \
  --model "GPT-2 MBPP=data/results_mbpp_full_gpt2" \
  --model "GPT-2 Medium MBPP=data/results_mbpp_full_gpt2_medium" \
  --model "CodeGen MBPP=data/results_mbpp_full_codegen" \
  --pass_k 1 \
  --pass_k 5 \
  --pass_k 10 \
  --pass_k 20 \
  --output_file outputs/tables/bootstrap_significance_mbpp_full.json

echo "[$(date '+%F %T')] Full-coverage MBPP validation completed"
