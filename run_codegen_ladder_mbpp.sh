#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

NUM_PROBLEMS="${NUM_PROBLEMS:-257}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

CONFIGS=(
  "config_mbpp_full_codegen_nl.yaml"
  "config_mbpp_full_codegen_multi.yaml"
  "config_mbpp_full_codegen_mono.yaml"
)

if [ ! -f data/raw/mbpp.jsonl ]; then
  python scripts/download_data.py --dataset mbpp
fi

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Running CodeGen ladder MBPP validation for ${config}"
  python scripts/generate_samples_safe.py \
    --config "${config}" \
    --resume \
    --num_problems "${NUM_PROBLEMS}" \
    --num_samples "${NUM_SAMPLES}"

  python scripts/run_evaluation.py --config "${config}"
  python scripts/analyze_failures.py --config "${config}"
  python scripts/deep_syntax_analysis.py --config "${config}"
done

python scripts/bootstrap_significance.py \
  --model "CodeGen-NL MBPP=data/results_mbpp_full_codegen_nl" \
  --model "CodeGen-Multi MBPP=data/results_mbpp_full_codegen_multi" \
  --model "CodeGen-Mono MBPP=data/results_mbpp_full_codegen_mono" \
  --pass_k 1 \
  --pass_k 5 \
  --pass_k 10 \
  --pass_k 20 \
  --output_file outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json

python scripts/summarize_codegen_ladder.py \
  --model "CodeGen-NL MBPP=data/results_mbpp_full_codegen_nl" \
  --model "CodeGen-Multi MBPP=data/results_mbpp_full_codegen_multi" \
  --model "CodeGen-Mono MBPP=data/results_mbpp_full_codegen_mono" \
  --output_file outputs/tables/codegen_ladder_mbpp_summary.json

echo "[$(date '+%F %T')] CodeGen ladder MBPP validation completed"
