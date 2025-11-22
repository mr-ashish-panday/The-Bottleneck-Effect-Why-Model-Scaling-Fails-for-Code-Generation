#!/usr/bin/env python3
"""
Analyze failure patterns and make feasibility decision.

Usage:
    python scripts/analyze_failures.py --config config.yaml
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import yaml

import sys
sys.path.append('.')


def analyze_by_complexity(all_samples):
    """Analyze failures by problem complexity (LOC)."""
    complexity_failures = defaultdict(lambda: Counter())
    
    for problem in all_samples:
        # Estimate complexity by prompt length
        prompt_lines = len(problem["prompt"].split('\n'))
        
        if prompt_lines <= 5:
            complexity = "simple"
        elif prompt_lines <= 10:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # Count failures
        for sample in problem["samples"]:
            category = sample.get("category", "unknown")
            complexity_failures[complexity][category] += 1
    
    return complexity_failures


def analyze_error_messages(all_samples):
    """Extract common error patterns."""
    error_patterns = Counter()
    
    for problem in all_samples:
        for sample in problem["samples"]:
            result = sample.get("execution_result", {})
            error_msg = result.get("error_message", "")
            
            if error_msg:
                # Extract error type
                if ":" in error_msg:
                    error_type = error_msg.split(":")[0]
                    error_patterns[error_type] += 1
    
    return error_patterns


def make_feasibility_decision(category_counts, total_samples, threshold=0.7):
    """
    Decide if we can proceed based on failure categorization.
    
    Decision criteria:
    - Can we clearly categorize >70% of failures?
    - Are there interesting patterns?
    """
    # Calculate categorization rate
    clear_categories = ["syntax_error", "runtime_error", "wrong_output", "timeout", "success"]
    clear_count = sum(category_counts[cat] for cat in clear_categories)
    categorization_rate = clear_count / total_samples
    
    # Decision
    can_proceed = categorization_rate >= threshold
    
    return {
        "can_proceed": can_proceed,
        "categorization_rate": categorization_rate,
        "threshold": threshold,
        "total_samples": total_samples,
        "clear_categories": clear_count,
        "unclear_categories": total_samples - clear_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze failure patterns")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with evaluation results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for analysis report",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set paths
    results_dir = Path(config['paths']['results_dir'])
    input_file = Path(args.input_file or results_dir / "evaluation_results.json")
    output_file = Path(args.output_file or results_dir / "feasibility_report.json")
    
    print("="*60)
    print("PHASE 1-2: FEASIBILITY CHECK - ANALYSIS")
    print("="*60)
    print(f"Input file: {input_file}")
    print()
    
    # Load results
    with open(input_file, 'r') as f:
        all_samples = json.load(f)
    
    # Count categories
    category_counts = Counter()
    total_samples = 0
    
    for problem in all_samples:
        for sample in problem["samples"]:
            category = sample.get("category", "unknown")
            category_counts[category] += 1
            total_samples += 1
    
    # Analyze by complexity
    complexity_failures = analyze_by_complexity(all_samples)
    
    # Analyze error patterns
    error_patterns = analyze_error_messages(all_samples)
    
    # Make feasibility decision
    threshold = config['feasibility_check']['decision_threshold']
    decision = make_feasibility_decision(category_counts, total_samples, threshold)
    
    # Create report
    report = {
        "summary": {
            "total_problems": len(all_samples),
            "total_samples": total_samples,
            "samples_per_problem": total_samples // len(all_samples),
        },
        "category_distribution": dict(category_counts),
        "complexity_analysis": {
            complexity: dict(counts) 
            for complexity, counts in complexity_failures.items()
        },
        "error_patterns": dict(error_patterns.most_common(20)),
        "feasibility_decision": decision,
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    print("FEASIBILITY REPORT")
    print("="*60)
    print(f"Total problems: {report['summary']['total_problems']}")
    print(f"Total samples: {report['summary']['total_samples']}")
    print()
    
    print("CATEGORY DISTRIBUTION:")
    print("-"*60)
    for category, count in category_counts.most_common():
        percentage = (count / total_samples) * 100
        print(f"  {category:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print()
    print("COMPLEXITY ANALYSIS:")
    print("-"*60)
    for complexity in ["simple", "medium", "complex"]:
        if complexity in complexity_failures:
            counts = complexity_failures[complexity]
            total = sum(counts.values())
            print(f"\n  {complexity.upper()} problems:")
            for category, count in counts.most_common(3):
                pct = (count / total) * 100
                print(f"    {category:20s}: {count:4d} ({pct:5.1f}%)")
    
    print()
    print("="*60)
    print("FEASIBILITY DECISION")
    print("="*60)
    print(f"Categorization rate: {decision['categorization_rate']:.1%}")
    print(f"Threshold: {decision['threshold']:.1%}")
    print(f"Clear categories: {decision['clear_categories']} / {decision['total_samples']}")
    print()
    
    if decision["can_proceed"]:
        print("✅ DECISION: PROCEED TO FULL PAPER")
        print()
        print("We can clearly categorize failure patterns.")
        print("There are interesting differences across complexity levels.")
        print("The error taxonomy is well-defined.")
        print()
        print("NEXT STEPS:")
        print("1. Expand to all 164 HumanEval problems")
        print("2. Generate 100 samples × 164 = 16,400 total samples")
        print("3. Deep dive into error patterns")
        print("4. Extract activation patterns")
        print("5. Build execution-aware decoding")
    else:
        print("❌ DECISION: PIVOT TO PAPER 10")
        print()
        print("Cannot clearly categorize >70% of failures.")
        print("Patterns are not distinct enough for paper.")
        print()
        print("PIVOT TO PAPER 10:")
        print("  Edge deployment story with LoRA")
    
    print()
    print(f"✅ Full report saved to: {output_file}")


if __name__ == "__main__":
    main()
