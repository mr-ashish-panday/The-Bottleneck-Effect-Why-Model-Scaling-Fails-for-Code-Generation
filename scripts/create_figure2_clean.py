#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

print("="*60)
print("CREATING FIGURE 2: LAYER ABLATION HEATMAP (CLEAN VERSION)")
print("="*60)

ablation_files = {
    "GPT-2 Small (12L)": "data/results/ablation/layer_ablation_results.json",
    "GPT-2 Medium (24L)": "data/results_gpt2_medium/ablation/layer_ablation_results.json",
    "CodeGen-350M (20L)": "data/results_codegen/ablation/layer_ablation_results.json"
}

models_data = []

for model_name, filepath in ablation_files.items():
    try:
        with open(filepath) as f:
            results = json.load(f)
        
        success_rates = [r.get("success_pct", 0) for r in results]
        num_layers = len(success_rates)
        
        models_data.append({
            "name": model_name,
            "success_rates": success_rates,
            "num_layers": num_layers
        })
        
        print(f"\n{model_name}:")
        print(f"  Layers: {num_layers}")
        print(f"  Max success when ablated: {max(success_rates):.1f}%")
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Create figure - BAR CHART INSTEAD OF HEATMAP
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for idx, model in enumerate(models_data):
    ax = axes[idx]
    
    layers = range(model["num_layers"])
    success_rates = model["success_rates"]
    
    # Color bars based on success rate
    colors = []
    for rate in success_rates:
        if rate == 0:
            colors.append('#d62728')  # Red - total failure
        elif rate < 5:
            colors.append('#ff7f0e')  # Orange - minimal survival
        elif rate < 15:
            colors.append('#ffdd57')  # Yellow - partial survival
        else:
            colors.append('#2ca02c')  # Green - strong survival
    
    bars = ax.bar(layers, success_rates, color=colors, edgecolor='black', linewidth=0.8)
    
    # Highlight critical layers
    for i, rate in enumerate(success_rates):
        if rate > 2.0:
            ax.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
            # Add star
            ax.plot(i, rate + 0.5, '*', color='gold', markersize=15, 
                   markeredgecolor='black', markeredgewidth=1)
    
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(model["name"], fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, model["num_layers"] - 0.5)

# Shared x-axis label
axes[-1].set_xlabel('Layer Index', fontsize=12, fontweight='bold')

# Overall title
fig.suptitle('Layer Ablation Results: Success Rate When Each Layer Is Removed\n(Red = Catastrophic Failure, Green = Strong Survival)', 
            fontsize=14, fontweight='bold', y=0.995)

# Legend
legend_elements = [
    mpatches.Patch(color='#d62728', label='0% (Total Failure)'),
    mpatches.Patch(color='#ff7f0e', label='<5% (Minimal Survival)'),
    mpatches.Patch(color='#ffdd57', label='5-15% (Partial Survival)'),
    mpatches.Patch(color='#2ca02c', label='>15% (Strong Survival)')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
          bbox_to_anchor=(0.98, 0.98), framealpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.98])

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "figure2_ablation_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "figure2_ablation_heatmap.pdf", bbox_inches='tight')

print(f"\n✅ Figure 2 saved with clean bar chart design!")
plt.close()
