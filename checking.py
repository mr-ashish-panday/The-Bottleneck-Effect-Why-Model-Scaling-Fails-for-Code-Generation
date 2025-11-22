#!/usr/bin/env python3
"""
Verify Table 4 values from layer12_analysis.json
"""
import json

# Load the JSON file
with open('data/results_gpt2_medium/ablation/layer12_analysis.json', 'r') as f:
    data = json.load(f)

success_mean = data['success_mean']
failure_mean = data['failure_mean']
top_dims = data['top_discriminative_dims']

print("=" * 70)
print("VERIFYING TABLE 4: Top Discriminative Dimensions in Layer 12")
print("=" * 70)
print()

# Table 4 expected values
table_4_expected = {
    810: {'mean_s': 2.73, 'mean_f': -1.24, 'diff': 3.97},
    457: {'mean_s': 1.89, 'mean_f': 0.76, 'diff': 2.65},
    182: {'mean_s': -1.45, 'mean_f': 0.88, 'diff': 2.33},
    802: {'mean_s': 1.07, 'mean_f': -0.52, 'diff': 2.19},
    169: {'mean_s': 0.94, 'mean_f': -0.06, 'diff': 1.00}
}

print(f"{'Dim':<6} {'Mean(S)':<10} {'Mean(F)':<10} {'Diff':<10} {'Match?':<10}")
print("-" * 70)

for dim in top_dims[:5]:  # Top 5 dimensions
    mean_s = success_mean[dim]
    mean_f = failure_mean[dim]
    diff = mean_s - mean_f
    
    # Check if this dimension is in our expected table
    if dim in table_4_expected:
        expected = table_4_expected[dim]
        
        # Check if values match (within rounding tolerance)
        s_match = abs(mean_s - expected['mean_s']) < 0.01
        f_match = abs(mean_f - expected['mean_f']) < 0.01
        d_match = abs(diff - expected['diff']) < 0.01
        
        match = "✓ YES" if (s_match and f_match and d_match) else "✗ NO"
        
        print(f"{dim:<6} {mean_s:<10.2f} {mean_f:<10.2f} {diff:<10.2f} {match:<10}")
        print(f"       (Expected: {expected['mean_s']:.2f}, {expected['mean_f']:.2f}, {expected['diff']:.2f})")
    else:
        print(f"{dim:<6} {mean_s:<10.2f} {mean_f:<10.2f} {diff:<10.2f} {'N/A':<10}")
    print()

print("=" * 70)
print("TOP 20 DISCRIMINATIVE DIMENSIONS (ranked):")
print("=" * 70)

for i, dim in enumerate(top_dims, 1):
    diff = success_mean[dim] - failure_mean[dim]
    print(f"Rank {i:2d}: Dim {dim:4d} | Diff = {diff:6.2f} | S={success_mean[dim]:6.2f}, F={failure_mean[dim]:6.2f}")