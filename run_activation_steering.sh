#!/usr/bin/env bash
set -euo pipefail

python scripts/contrastive_activation_steering.py \
  --config config_gpt2_medium.yaml \
  --analysis_file data/results_gpt2_medium/ablation/layer12_analysis_real.json \
  --dimensions_file data/results_gpt2_medium/ablation/activation_classification_real.json \
  --vector_mode top_dims \
  --num_dims 5 \
  --alphas -1.0,-0.5,0.5,1.0 \
  --num_problems 10 \
  --samples_per_problem 5 \
  --output_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json

python scripts/analyze_activation_steering.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json \
  --output_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5_summary.json
