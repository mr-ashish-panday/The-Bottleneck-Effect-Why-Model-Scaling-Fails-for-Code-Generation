#!/bin/bash
set -e

echo "=========================================="
echo "FINE-TUNING DYNAMICS SUITE V2"
echo "Started: $(date)"
echo "=========================================="
echo "Changes from V1:"
echo "  - Batch size: 1 (prevents OOM)"
echo "  - All models: 10 epochs (comparable)"
echo "  - No checkpoints saved (disk space)"
echo "  - Results: training_trajectory.json"
echo "=========================================="
echo ""

# Experiment 1: GPT-2 Small
echo "[1/3] Fine-tuning GPT-2 Small (10 epochs)..."
python scripts/finetune_dynamics_v2.py \
    --model gpt2 \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_gpt2_v2

echo "✅ GPT-2 Small complete"
echo ""

# Experiment 2: GPT-2 Medium
echo "[2/3] Fine-tuning GPT-2 Medium (10 epochs)..."
python scripts/finetune_dynamics_v2.py \
    --model gpt2-medium \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_gpt2_medium_v2

echo "✅ GPT-2 Medium complete"
echo ""

# Experiment 3: CodeGen (NOW 10 EPOCHS)
echo "[3/3] Fine-tuning CodeGen (10 epochs)..."
python scripts/finetune_dynamics_v2.py \
    --model Salesforce/codegen-350M-mono \
    --num_train_problems 130 \
    --num_eval_problems 30 \
    --num_epochs 10 \
    --eval_samples 20 \
    --output_dir data/results_finetuning_codegen_v2

echo "✅ CodeGen complete"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  data/results_finetuning_gpt2_v2/training_trajectory.json"
echo "  data/results_finetuning_gpt2_medium_v2/training_trajectory.json"
echo "  data/results_finetuning_codegen_v2/training_trajectory.json"
echo ""
echo "No checkpoints saved (disk space optimized)"
echo "=========================================="
