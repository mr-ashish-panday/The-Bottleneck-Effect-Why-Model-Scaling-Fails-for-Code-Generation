#!/usr/bin/env python3
"""
Figure 1: Model Performance Comparison
Bar chart showing syntax %, success %, runtime % across 3 models
Uses REAL data from evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING FIGURE 1: MODEL PERFORMANCE COMPARISON")
print("="*60)

# Load evaluation results for each model
models_data = {
    "GPT-2 Small\n(124M)": {
        "file": "data/results_gpt2/evaluation_results.json",
        "feasibility": "data/results_gpt2/feasibility_report.json"
    },
    "GPT-2 Medium\n(355M)": {
        "file": "data/results_gpt2_medium/evaluation_results.json",
        "feasibility": "data/results_gpt2_medium/feasibility_report.json"
    },
    "CodeGen-350M": {
        "file": "data/results_codegen/evaluation_results.json",
        "feasibility": "data/results_codegen/feasibility_report.json"
    }
}

# Extract data
model_names = []
syntax_pcts = []
success_pcts = []
runtime_pcts = []
total_samples = []

for model_name, paths in models_data.items():
    try:
        # Load feasibility report (has summary stats)
        with open(paths["feasibility"]) as f:
            report = json.load(f)
        
        cat_dist = report["category_distribution"]
        total = report["summary"]["total_samples"]
        
        # Calculate percentages
        syntax_pct = (cat_dist.get("syntax_error", 0) / total) * 100
        success_pct = (cat_dist.get("success", 0) / total) * 100
        runtime_pct = (cat_dist.get("runtime_error", 0) / total) * 100
        
        model_names.append(model_name)
        syntax_pcts.append(syntax_pct)
        success_pcts.append(success_pct)
        runtime_pcts.append(runtime_pct)
        total_samples.append(total)
        
        print(f"\n{model_name}:")
        print(f"  Total samples: {total}")
        print(f"  Syntax errors: {syntax_pct:.1f}%")
        print(f"  Success: {success_pct:.1f}%")
        print(f"  Runtime errors: {runtime_pct:.1f}%")
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(model_names))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, syntax_pcts, width, label='Syntax Errors', 
               color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, success_pcts, width, label='Success', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, runtime_pcts, width, label='Runtime Errors', 
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 1:  # Only label if significant
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Formatting
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Generations (%)', fontsize=14, fontweight='bold')
ax.set_title('Code Generation Performance Across Models\n(HumanEval, 16,400 generations per model)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add text annotation about scaling paradox
ax.annotate('Scaling Paradox:\nMedium (355M) worse\nthan Small (124M)', 
            xy=(0.5, 50), xytext=(0.5, 70),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='yellow', alpha=0.7))

plt.tight_layout()

# Save
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "figure1_model_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "figure1_model_comparison.pdf", bbox_inches='tight')

print(f"\n✅ Figure 1 saved to {output_dir}/figure1_model_comparison.png")

plt.close()
