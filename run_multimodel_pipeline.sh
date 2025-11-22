#!/bin/bash
set -e

echo "=========================================="
echo "MULTI-MODEL COMPARISON PIPELINE"
echo "Models: GPT-2, GPT-2 Medium, CodeGen-350M"
echo "Started: $(date)"
echo "=========================================="

# Model 1: GPT-2 (already done)
echo ""
echo "Model 1/3: GPT-2 (ALREADY COMPLETE)"
echo "Results in: data/results_gpt2/"

# Model 2: GPT-2 Medium
echo ""
echo "=========================================="
echo "Model 2/3: GPT-2 MEDIUM (355M)"
echo "=========================================="
python scripts/generate_samples.py --config config_gpt2_medium.yaml --num_problems 164 --num_samples 100
python scripts/run_evaluation.py --config config_gpt2_medium.yaml
python scripts/analyze_failures.py --config config_gpt2_medium.yaml
python scripts/deep_syntax_analysis.py --config config_gpt2_medium.yaml
python scripts/extract_activations.py --config config_gpt2_medium.yaml --num_samples 1000

echo "✅ GPT-2 Medium complete"

# Model 3: CodeGen
echo ""
echo "=========================================="
echo "Model 3/3: CODEGEN-350M"
echo "=========================================="
python scripts/generate_samples.py --config config_codegen.yaml --num_problems 164 --num_samples 100
python scripts/run_evaluation.py --config config_codegen.yaml
python scripts/analyze_failures.py --config config_codegen.yaml
python scripts/deep_syntax_analysis.py --config config_codegen.yaml
python scripts/extract_activations.py --config config_codegen.yaml --num_samples 1000

echo "✅ CodeGen complete"

echo ""
echo "=========================================="
echo "ALL MODELS COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  GPT-2:        data/results_gpt2/"
echo "  GPT-2 Medium: data/results_gpt2_medium/"
echo "  CodeGen:      data/results_codegen/"
echo ""
echo "Next: Run comparison analysis"
echo "  python scripts/compare_models.py"
