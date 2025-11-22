#!/bin/bash
set -e

echo "=========================================="
echo "FINE-TUNING REMAINING MODELS"
echo "Started: $(date)"
echo "=========================================="

# GPT-2 Medium (restart)
echo ""
echo "[1/2] Fine-tuning GPT-2 Medium..."
python scripts/finetune_dynamics.py \
    --model gpt2-medium \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_gpt2_medium

echo "✅ GPT-2 Medium complete"

# CodeGen
echo ""
echo "[2/2] Fine-tuning CodeGen..."
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
echo "ALL REMAINING MODELS COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
