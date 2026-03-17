#!/usr/bin/env bash
set -euo pipefail

python scripts/activation_steering_controls.py \
  --config config_gpt2_medium.yaml \
  --analysis_file data/results_gpt2_medium/ablation/layer12_analysis_real.json \
  --dimensions_file data/results_gpt2_medium/ablation/activation_classification_real.json \
  --vector_mode top_dims \
  --num_dims 5 \
  --target_alphas=-2.0,2.0 \
  --control_alpha 2.0 \
  --num_random_controls 5 \
  --num_problems 10 \
  --samples_per_problem 5 \
  --output_file data/results_gpt2_medium/ablation/activation_steering_controls_10x5.json

python scripts/analyze_activation_steering_controls.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_controls_10x5.json \
  --output_file data/results_gpt2_medium/ablation/activation_steering_controls_10x5_summary.json
