#!/bin/bash
set -e

echo "=========================================="
echo "EXTENDED SAMPLING FOR STATISTICAL POWER"
echo "Models: GPT-2 Small + CodeGen (skip Medium - OOM issues)"
echo "Started: $(date)"
echo "Estimated time: 60-80 GPU hours"
echo "=========================================="

# Generate 500 samples per problem (instead of 100)
# This gives you Pass@k metrics (k=1,5,10,50,100,200,500)

echo ""
echo "[1/2] GPT-2 Small: 164 problems × 500 samples = 82,000 samples"
python scripts/generate_samples.py \
    --config config.yaml \
    --num_problems 164 \
    --num_samples 500

echo "Evaluating GPT-2..."
python scripts/run_evaluation.py --config config.yaml

echo "Analyzing GPT-2..."
python scripts/analyze_failures.py --config config.yaml

echo "✅ GPT-2 Small complete"

echo ""
echo "[2/2] CodeGen: 164 problems × 500 samples = 82,000 samples"
python scripts/generate_samples.py \
    --config config_codegen.yaml \
    --num_problems 164 \
    --num_samples 500

echo "Evaluating CodeGen..."
python scripts/run_evaluation.py --config config_codegen.yaml

echo "Analyzing CodeGen..."
python scripts/analyze_failures.py --config config_codegen.yaml

echo "✅ CodeGen complete"

echo ""
echo "=========================================="
echo "EXTENDED SAMPLING COMPLETE!"
echo "=========================================="
echo "Total samples: 164,000 (2 models × 82,000)"
echo "GPT-2 results:  data/results_gpt2/"
echo "CodeGen results: data/results_codegen/"
echo "Finished: $(date)"
echo ""
echo "PASS@K ANALYSIS:"
python << 'PYEOF'
import json
from pathlib import Path

for model_name, results_dir in [("GPT-2", "data/results_gpt2"), ("CodeGen", "data/results_codegen")]:
    try:
        with open(f"{results_dir}/evaluation_results.json") as f:
            data = json.load(f)
        
        # Calculate Pass@k
        total_problems = len(data)
        
        for k in [1, 5, 10, 50, 100]:
            problems_with_k_success = 0
            for problem in data:
                successes = sum(1 for s in problem['samples'] if s.get('category') == 'success')
                if successes >= k:
                    problems_with_k_success += 1
            
            pass_at_k = problems_with_k_success / total_problems * 100
            print(f"{model_name} Pass@{k:3d}: {pass_at_k:5.1f}%")
        
        print()
    except Exception as e:
        print(f"Could not compute Pass@k for {model_name}: {e}")
        print()
PYEOF

echo "=========================================="
