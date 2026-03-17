#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

NUM_PROBLEMS="${NUM_PROBLEMS:-164}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

CONFIGS=(
  "config_codegen_nl.yaml"
  "config_codegen_multi.yaml"
  "config_codegen_mono.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Running CodeGen ladder benchmark for ${config}"
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

echo "[$(date '+%F %T')] CodeGen pretraining ladder completed"
