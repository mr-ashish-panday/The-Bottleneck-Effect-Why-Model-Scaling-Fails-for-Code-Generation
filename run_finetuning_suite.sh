#!/bin/bash
set -e

echo "=========================================="
echo "FINE-TUNING DYNAMICS EXPERIMENT SUITE"
echo "Started: $(date)"
echo "Estimated time: 40-50 GPU hours"
echo "=========================================="

# Experiment 1: GPT-2 Small fine-tuning
echo ""
echo "[1/3] Fine-tuning GPT-2 Small..."
python scripts/finetune_dynamics.py \
    --model gpt2 \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_gpt2

echo "✅ GPT-2 Small complete"

# Experiment 2: GPT-2 Medium fine-tuning
echo ""
echo "[2/3] Fine-tuning GPT-2 Medium..."
python scripts/finetune_dynamics.py \
    --model gpt2-medium \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_gpt2_medium

echo "✅ GPT-2 Medium complete"

# Experiment 3: CodeGen fine-tuning
echo ""
echo "[3/3] Fine-tuning CodeGen..."
python scripts/finetune_dynamics.py \
    --model Salesforce/codegen-350M-mono \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_codegen

echo "✅ CodeGen complete"

echo ""
echo "=========================================="
echo "ALL FINE-TUNING EXPERIMENTS COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
