#!/usr/bin/env python3
"""
Compare results across GPT-2, GPT-2 Medium, and CodeGen.
"""

import json
from pathlib import Path
from collections import Counter

def load_results(results_dir):
    """Load evaluation and analysis results."""
    with open(results_dir / "evaluation_results.json") as f:
        eval_results = json.load(f)
    
    with open(results_dir / "feasibility_report.json") as f:
        analysis = json.load(f)
    
    with open(results_dir / "syntax_analysis.json") as f:
        syntax = json.load(f)
    
    return eval_results, analysis, syntax

def main():
    models = {
        "GPT-2 (124M)": Path("data/results_gpt2"),
        "GPT-2 Medium (355M)": Path("data/results_gpt2_medium"),
        "CodeGen-350M": Path("data/results_codegen"),
    }
    
    print("="*80)
    print("MULTI-MODEL COMPARISON: CODE EXECUTION FAILURES")
    print("="*80)
    print()
    
    all_results = {}
    
    for model_name, results_dir in models.items():
        if not results_dir.exists():
            print(f"⚠️  {model_name}: Results not found")
            continue
        
        eval_results, analysis, syntax = load_results(results_dir)
        
        all_results[model_name] = {
            "category_dist": analysis["category_distribution"],
            "syntax_breakdown": syntax["category_distribution"],
            "error_positions": syntax["position_distribution"],
        }
        
        print(f"✅ {model_name}: Loaded")
    
    print()
    print("="*80)
    print("CATEGORY DISTRIBUTION COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Syntax Error':<15} {'Success':<15} {'Runtime Error':<15}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        dist = results["category_dist"]
        total = sum(dist.values())
        
        syntax_pct = dist.get("syntax_error", 0) / total * 100
        success_pct = dist.get("success", 0) / total * 100
        runtime_pct = dist.get("runtime_error", 0) / total * 100
        
        print(f"{model_name:<25} {syntax_pct:>6.1f}%         {success_pct:>6.1f}%         {runtime_pct:>6.1f}%")
    
    print()
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Calculate differences
    gpt2_syntax = all_results["GPT-2 (124M)"]["category_dist"]["syntax_error"] / 16400 * 100
    
    if "GPT-2 Medium (355M)" in all_results:
        medium_syntax = all_results["GPT-2 Medium (355M)"]["category_dist"]["syntax_error"] / 16400 * 100
        print(f"1. Model size impact: {gpt2_syntax:.1f}% → {medium_syntax:.1f}% syntax errors")
        print(f"   (3× parameters = {gpt2_syntax - medium_syntax:.1f}% reduction)")
    
    if "CodeGen-350M" in all_results:
        codegen_syntax = all_results["CodeGen-350M"]["category_dist"]["syntax_error"] / 16400 * 100
        print(f"2. Code pretraining impact: {gpt2_syntax:.1f}% → {codegen_syntax:.1f}% syntax errors")
        print(f"   (Code-specific training = {gpt2_syntax - codegen_syntax:.1f}% reduction)")
    
    print()
    print("="*80)
    print("SYNTAX ERROR BREAKDOWN COMPARISON")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        syntax_breakdown = results["syntax_breakdown"]
        total_syntax = sum(syntax_breakdown.values())
        
        for category, count in sorted(syntax_breakdown.items(), key=lambda x: -x[1])[:5]:
            pct = count / total_syntax * 100
            print(f"  {category:<25}: {pct:>5.1f}%")
    
    print()
    print("="*80)
    print("✅ Comparison complete!")
    print("This data is ready for your paper!")
    print("="*80)

if __name__ == "__main__":
    main()
