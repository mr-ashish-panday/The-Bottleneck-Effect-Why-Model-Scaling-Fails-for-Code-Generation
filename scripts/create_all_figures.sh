#!/bin/bash
set -e

echo "=========================================="
echo "CREATING ALL PAPER FIGURES"
echo "=========================================="

echo ""
echo "[1/4] Creating Figure 1: Model Comparison..."
python scripts/create_figure1.py

echo ""
echo "[2/4] Creating Figure 2: Layer Ablation Heatmap..."
python scripts/create_figure2.py

echo ""
echo "[3/4] Creating Figure 3: Error Distribution..."
python scripts/create_figure3.py

echo ""
echo "[4/4] Creating Figure 4: Activation Projection..."
python scripts/create_figure4_improved.py

echo ""
echo "=========================================="
echo "ALL FIGURES COMPLETE!"
echo "=========================================="
echo "Saved to: outputs/figures/"
ls -lh outputs/figures/
