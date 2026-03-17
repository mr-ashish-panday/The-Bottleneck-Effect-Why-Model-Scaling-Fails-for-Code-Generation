#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

echo "[$(date '+%F %T')] Resuming remaining CodeGen ladder queue from CodeGen-Mono MBPP"

python scripts/generate_samples_safe.py \
  --config config_mbpp_full_codegen_mono.yaml \
  --resume \
  --num_problems 257 \
  --num_samples 20

python scripts/run_evaluation.py --config config_mbpp_full_codegen_mono.yaml
python scripts/analyze_failures.py --config config_mbpp_full_codegen_mono.yaml
python scripts/deep_syntax_analysis.py --config config_mbpp_full_codegen_mono.yaml

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

echo "[$(date '+%F %T')] Running prompt perplexity diagnostics"
bash run_codegen_ladder_prompt_ppl.sh

echo "[$(date '+%F %T')] Running mechanism discovery rerun"
bash run_codegen_ladder_mechanism.sh

echo "[$(date '+%F %T')] Resume queue finished"
