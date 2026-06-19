#!/bin/bash
set -e

echo "=========================================="
echo "CREATING ALL PAPER FIGURES"
echo "=========================================="

echo ""
echo "[1/10] Creating Figure 1: Model Comparison..."
python scripts/create_figure1.py

echo ""
echo "[2/10] Creating Figure 2: Layer Ablation Heatmap..."
python scripts/create_figure2.py

echo ""
echo "[3/10] Creating Figure 3: Error Distribution..."
python scripts/create_figure3.py

echo ""
echo "[4/10] Creating Figure 4: Activation Projection..."
python scripts/create_figure4_improved.py

echo ""
echo "[5/10] Creating Figure 10: Cross-Benchmark Map..."
python scripts/create_figure10_cross_benchmark_map.py

echo ""
echo "[6/10] Creating Figure 11: Bootstrap Forest Plot..."
python scripts/create_figure11_bootstrap_forest.py

echo ""
echo "[7/10] Creating Figure 12: CodeGen Ladder Benchmarks..."
python scripts/create_figure12_codegen_ladder_benchmarks.py

echo ""
echo "[8/10] Creating Figure 13: Strictness Cascade..."
python scripts/create_figure13_strictness_cascade.py

echo ""
echo "[9/10] Creating Figure 14: Coverage Audit..."
python scripts/create_figure14_coverage_audit.py

echo ""
echo "[10/10] Creating Figure 15: JSS Robustness Controls..."
python scripts/create_figure15_jss_robustness_controls.py

echo ""
echo "=========================================="
echo "ALL FIGURES COMPLETE!"
echo "=========================================="
echo "Saved to: outputs/figures/"
ls -lh outputs/figures/
