#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="outputs/logs"
mkdir -p $LOGDIR

echo "=========================================="
echo "PAPER 11: HEAVY ANALYSIS PIPELINE"
echo "Started: $(date)"
echo "Estimated time: 4-6 hours"
echo "=========================================="

# Step 1: Deep syntax analysis (30-45 mins)
echo ""
echo "[1/3] Deep syntax error analysis..."
python scripts/deep_syntax_analysis.py \
    --config config.yaml \
    2>&1 | tee $LOGDIR/syntax_analysis_$TIMESTAMP.log

# Step 2: Extract activations (2-3 hours - HEAVY!)
echo ""
echo "[2/3] Extracting activations (THIS IS HEAVY - 2-3 hours)..."
python scripts/extract_activations.py \
    --config config.yaml \
    --num_samples 1000 \
    2>&1 | tee $LOGDIR/activation_extraction_$TIMESTAMP.log

# Step 3: Final analysis report
echo ""
echo "[3/3] Generating final analysis report..."
python scripts/analyze_failures.py \
    --config config.yaml \
    2>&1 | tee $LOGDIR/final_analysis_$TIMESTAMP.log

echo ""
echo "=========================================="
echo "HEAVY PIPELINE COMPLETE!"
echo "Finished: $(date)"
echo "Check results in: data/results/"
echo "=========================================="
