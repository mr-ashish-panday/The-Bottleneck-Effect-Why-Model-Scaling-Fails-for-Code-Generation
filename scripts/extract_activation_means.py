#!/usr/bin/env python3
"""
Extract actual mean activation values for Table 4.
This replaces fabricated numbers with real data.
"""

import json
import numpy as np
from pathlib import Path

# Load layer 12 analysis
analysis_file = Path("data/results_gpt2_medium/ablation/layer12_analysis.json")

with open(analysis_file) as f:
    data = json.load(f)

success_mean = np.array(data['success_mean'])
failure_mean = np.array(data['failure_mean'])
difference = np.array(data['difference'])

# Top 5 discriminative dimensions
top_dims = data['top_discriminative_dims'][:5]

print("="*60)
print("TABLE 4 DATA: TOP DISCRIMINATIVE DIMENSIONS")
print("="*60)

table_data = []

for i, dim in enumerate(top_dims):
    succ_val = success_mean[dim]
    fail_val = failure_mean[dim]
    diff_val = abs(succ_val - fail_val)
    
    table_data.append({
        "dimension": int(dim),
        "mean_success": float(succ_val),
        "mean_failure": float(fail_val),
        "difference": float(diff_val),
    })
    
    print(f"\nDimension {dim}:")
    print(f"  Mean (Success):  {succ_val:+.4f}")
    print(f"  Mean (Failure):  {fail_val:+.4f}")
    print(f"  Difference:      {diff_val:.4f}")

# Calculate percentage of total signal
total_diff = np.sum(np.abs(difference))
for item in table_data:
    item['pct_signal'] = (item['difference'] / total_diff) * 100

print("\n" + "="*60)
print("PERCENTAGE OF TOTAL DISCRIMINATIVE SIGNAL")
print("="*60)
for item in table_data:
    print(f"Dimension {item['dimension']:4d}: {item['pct_signal']:5.1f}%")

top5_signal = sum(item['pct_signal'] for item in table_data)
print(f"\nTop 5 dimensions: {top5_signal:.1f}% of total signal")
print(f"Remaining 1019 dimensions: {100 - top5_signal:.1f}% of total signal")

# Save for paper
output = {
    "table_data": table_data,
    "top5_signal_pct": top5_signal,
    "total_dimensions": 1024,
    "discriminative_dimensions": 5,
}

output_file = Path("data/results_gpt2_medium/ablation/table4_data.json")
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Data saved to: {output_file}")
print("\nUse this data for Table 4 in the paper!")
