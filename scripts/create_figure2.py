#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING FIGURE 2: LAYER ABLATION HEATMAP")
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

# Normalize to 100 positions
heatmap_data = []
model_labels = []

for model in models_data:
    normalized = np.zeros(100)
    
    for i, success_rate in enumerate(model["success_rates"]):
        norm_pos = int((i / model["num_layers"]) * 100)
        if norm_pos < 100:
            normalized[norm_pos] = success_rate
    
    heatmap_data.append(normalized)
    model_labels.append(model["name"])

heatmap_data = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(16, 6))

im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
               vmin=0, vmax=30, interpolation='nearest')

ax.set_yticks(range(len(model_labels)))
ax.set_yticklabels(model_labels, fontsize=12)
ax.set_xlabel('Normalized Layer Position (% through model)', fontsize=14, fontweight='bold')
ax.set_title('Layer Ablation Impact: Success Rate When Each Layer Is Removed\n(Dark Red = Catastrophic Failure, Yellow/Green = Partial Survival)', 
             fontsize=16, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Success Rate When Layer Ablated (%)', 
               rotation=270, labelpad=25, fontsize=12, fontweight='bold')

xticks = [0, 25, 50, 75, 99]
ax.set_xticks(xticks)
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)

ax.set_xticks(range(0, 100, 10), minor=True)
ax.grid(which='minor', axis='x', alpha=0.2, linestyle='-', linewidth=0.5)

for i, model in enumerate(models_data):
    for layer_idx, success_rate in enumerate(model["success_rates"]):
        if success_rate > 2.0:
            norm_pos = int((layer_idx / model["num_layers"]) * 100)
            ax.plot(norm_pos, i, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1)

plt.tight_layout()

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "figure2_ablation_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "figure2_ablation_heatmap.pdf", bbox_inches='tight')

print(f"\n✅ Figure 2 saved")
plt.close()
