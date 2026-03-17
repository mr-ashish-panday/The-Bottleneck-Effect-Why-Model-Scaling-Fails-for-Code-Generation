#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

mkdir -p outputs/logs

wait_for_controls() {
  while pgrep -f "scripts/activation_steering_controls.py" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] Waiting for activation_steering_controls.py to finish..."
    sleep 60
  done
}

source venv/bin/activate

echo "[$(date '+%F %T')] Overnight follow-up queue started"
wait_for_controls

echo "[$(date '+%F %T')] Summarizing steering outputs"
python scripts/analyze_activation_steering.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json \
  --output_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5_summary.json

python scripts/analyze_activation_steering_controls.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_controls_10x5.json \
  --output_file data/results_gpt2_medium/ablation/activation_steering_controls_10x5_summary.json

python scripts/create_figure5_activation_steering.py \
  --input_file data/results_gpt2_medium/ablation/activation_steering_top5_10x5.json \
  --output_file outputs/figures/figure5_activation_steering_response.png

echo "[$(date '+%F %T')] Running GPT-2 small scaled ablation"
python scripts/scaled_layer_ablation.py \
  --config config.yaml \
  --model_config config.yaml \
  --layers 2 \
  --scales 0.75,0.5,0.25,0.0 \
  --num_problems 10 \
  --samples_per_problem 5 \
  --output_file data/results/ablation/scaled_layer2_10x5.json

python scripts/analyze_scaled_ablation.py \
  --input_file data/results/ablation/scaled_layer2_10x5.json \
  --output_file data/results/ablation/scaled_layer2_10x5_summary.json

echo "[$(date '+%F %T')] Running CodeGen scaled ablation"
python scripts/scaled_layer_ablation.py \
  --config config.yaml \
  --model_config config_codegen.yaml \
  --layers 13 \
  --scales 0.75,0.5,0.25,0.0 \
  --num_problems 10 \
  --samples_per_problem 5 \
  --output_file data/results_codegen/ablation/scaled_layer13_10x5.json

python scripts/analyze_scaled_ablation.py \
  --input_file data/results_codegen/ablation/scaled_layer13_10x5.json \
  --output_file data/results_codegen/ablation/scaled_layer13_10x5_summary.json

echo "[$(date '+%F %T')] Overnight follow-up queue completed"
