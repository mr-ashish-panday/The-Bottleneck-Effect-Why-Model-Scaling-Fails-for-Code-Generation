#!/usr/bin/env bash
set -euo pipefail

python scripts/bootstrap_significance.py

python scripts/scaled_layer_ablation.py \
  --config config.yaml \
  --model_config config_gpt2_medium.yaml \
  --layers 12 \
  --scales 0.75,0.5,0.25,0.0 \
  --num_problems 50 \
  --samples_per_problem 20

python scripts/scaled_layer_ablation.py \
  --config config.yaml \
  --model_config config.yaml \
  --layers 2 \
  --scales 0.75,0.5,0.25,0.0 \
  --num_problems 50 \
  --samples_per_problem 20

python scripts/scaled_layer_ablation.py \
  --config config.yaml \
  --model_config config_codegen.yaml \
  --layers 5,7,13 \
  --scales 0.75,0.5,0.25,0.0 \
  --num_problems 50 \
  --samples_per_problem 20

python scripts/contrastive_activation_steering.py \
  --config config_gpt2_medium.yaml \
  --analysis_file data/results_gpt2_medium/ablation/layer12_analysis_real.json \
  --dimensions_file data/results_gpt2_medium/ablation/activation_classification_real.json \
  --vector_mode top_dims \
  --num_dims 5 \
  --alphas=-2.0,-1.0,1.0,2.0 \
  --num_problems 10 \
  --samples_per_problem 5 \
  --output_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json

python scripts/analyze_activation_steering.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json \
  --output_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5_summary.json
