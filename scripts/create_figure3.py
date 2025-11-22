#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING FIGURE 3: SYNTAX ERROR DISTRIBUTION")
print("="*60)

syntax_files = {
    "GPT-2\nSmall": "data/results_gpt2/syntax_analysis.json",
    "GPT-2\nMedium": "data/results_gpt2_medium/syntax_analysis.json",
    "CodeGen\n350M": "data/results_codegen/syntax_analysis.json"
}

models_data = []

for model_name, filepath in syntax_files.items():
    try:
        with open(filepath) as f:
            results = json.load(f)
        
        cat_dist = results["category_distribution"]
        total_syntax = results["total_syntax_errors"]
        
        percentages = {cat: (count / total_syntax * 100) 
                      for cat, count in cat_dist.items()}
        
        models_data.append({
            "name": model_name,
            "percentages": percentages,
            "total": total_syntax
        })
        
        print(f"\n{model_name.replace(chr(10), ' ')}:")
        print(f"  Total syntax errors: {total_syntax}")
        for cat, pct in sorted(percentages.items(), key=lambda x: -x[1])[:3]:
            print(f"    {cat}: {pct:.1f}%")
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

all_categories = set()
for model in models_data:
    all_categories.update(model["percentages"].keys())

category_order = ["indentation", "bracket_mismatch", "quote_mismatch", 
                 "keyword_error", "colon_missing", "operator_error", "other"]

categories = [c for c in category_order if c in all_categories]

model_names = [m["name"] for m in models_data]
data_matrix = []

for category in categories:
    row = [m["percentages"].get(category, 0) for m in models_data]
    data_matrix.append(row)

data_matrix = np.array(data_matrix)

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#8b4513', '#ff69b4', '#708090', '#9acd32', '#00ced1', '#ff8c00', '#dc143c']

x = np.arange(len(model_names))
width = 0.6
bottom = np.zeros(len(model_names))

bars_list = []
for i, category in enumerate(categories):
    bars = ax.bar(x, data_matrix[i], width, bottom=bottom, 
                  label=category.replace('_', ' ').title(),
                  color=colors[i % len(colors)], alpha=0.85,
                  edgecolor='black', linewidth=1.5)
    bars_list.append(bars)
    
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 5:
            y_pos = bottom[j] + height/2
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    bottom += data_matrix[i]

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Syntax Errors (%)', fontsize=14, fontweight='bold')
ax.set_title('Syntax Error Type Distribution Across Models\n(Breakdown shows different failure patterns from pretraining)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12, fontweight='bold')

# Fixed legend (removed linewidth parameter)
legend = ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.95)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)

ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')

output_dir = Path("outputs/figures")
plt.savefig(output_dir / "figure3_error_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "figure3_error_distribution.pdf", bbox_inches='tight')

print(f"\n✅ Figure 3 saved")
plt.close()
