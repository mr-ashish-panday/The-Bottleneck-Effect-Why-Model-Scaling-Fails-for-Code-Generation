#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "config_mbpp_gpt2.yaml"
  "config_mbpp_gpt2_medium.yaml"
  "config_mbpp_codegen.yaml"
)

python scripts/download_data.py --dataset mbpp

for config in "${CONFIGS[@]}"; do
  python scripts/generate_samples.py --config "$config"
  python scripts/run_evaluation.py --config "$config"
  python scripts/analyze_failures.py --config "$config"
  python scripts/deep_syntax_analysis.py --config "$config"
done
