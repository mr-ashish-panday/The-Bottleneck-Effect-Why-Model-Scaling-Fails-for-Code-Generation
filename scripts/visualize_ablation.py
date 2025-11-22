#!/usr/bin/env python3
"""Create publication-quality ablation comparison figure."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
gpt2_small = json.load(open("data/results/ablation/layer_ablation_results.json"))
gpt2_medium = json.load(open("data/results_gpt2_medium/ablation/layer_ablation_results.json"))
codegen = json.load(open("data/results_codegen/ablation/layer_ablation_results.json"))

# Extract success rates
models = {
    "GPT-2 Small (12L)": [r.get('success_pct', 0) for r in gpt2_small],
    "GPT-2 Medium (24L)": [r.get('success_pct', 0) for r in gpt2_medium],
    "CodeGen-350M (20L)": [r.get('success_pct', 0) for r in codegen],
}

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Success rate per layer
for name, success_rates in models.items():
    layers = range(len(success_rates))
    ax1.plot(layers, success_rates, marker='o', label=name, linewidth=2)

ax1.set_xlabel('Layer Index', fontsize=12)
ax1.set_ylabel('Success Rate When Layer Ablated (%)', fontsize=12)
ax1.set_title('Layer Ablation Impact Across Models', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1, 35)

# Plot 2: Heatmap
data = []
labels = []
for name, rates in models.items():
    data.append(rates)
    labels.append(name.split('(')[0].strip())

im = ax2.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=30)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels)
ax2.set_xlabel('Layer Index', fontsize=12)
ax2.set_title('Layer Criticality Heatmap', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('outputs/figures/layer_ablation_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure saved to outputs/figures/layer_ablation_comparison.png")
plt.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for name, rates in models.items():
    max_success = max(rates)
    best_layer = rates.index(max_success)
    print(f"\n{name}:")
    print(f"  Best layer: {best_layer}")
    print(f"  Max success when ablated: {max_success:.1f}%")
    print(f"  Avg success when ablated: {np.mean(rates):.1f}%")
