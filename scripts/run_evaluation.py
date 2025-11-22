#!/usr/bin/env python3
"""
Evaluate generated samples and categorize failures.

Usage:
    python scripts/run_evaluation.py --config config.yaml
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import Counter

import sys
sys.path.append('.')

from src.evaluation.code_executor import execute_code


def categorize_failure(result: dict) -> str:
    """
    Categorize failure type based on execution result.
    
    Categories:
    - syntax_error: Code doesn't parse
    - runtime_error: Crashes during execution
    - assertion_error: Wrong output
    - timeout: Execution too slow
    - success: Passed all tests
    """
    if result["success"]:
        return "success"
    
    error_type = result["error_type"]
    
    if error_type == "syntax_error":
        return "syntax_error"
    elif error_type == "timeout":
        return "timeout"
    elif error_type == "assertion_error":
        return "wrong_output"
    elif error_type == "runtime_error":
        return "runtime_error"
    else:
        return "unknown_error"


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated samples")
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
        help="Input file with generated samples",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for evaluation results",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set paths
    results_dir = Path(config['paths']['results_dir'])
    input_file = Path(args.input_file or results_dir / "generated_samples.json")
    output_file = Path(args.output_file or results_dir / "evaluation_results.json")
    
    print("="*60)
    print("PHASE 1-2: FEASIBILITY CHECK - EVALUATION")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Load generated samples
    print("Loading generated samples...")
    with open(input_file, 'r') as f:
        all_samples = json.load(f)
    print(f"✅ Loaded {len(all_samples)} problems")
    print()
    
    # Evaluate each sample
    print("Evaluating samples...")
    timeout = config['execution']['timeout_seconds']
    
    total_samples = 0
    category_counts = Counter()
    
    for problem in tqdm(all_samples, desc="Problems"):
        task_id = problem["task_id"]
        test_code = problem["test"]
        
        # Evaluate each sample
        for sample in problem["samples"]:
            total_samples += 1
            code = sample["code"]
            
            # Combine prompt + generated code
            full_code = problem["prompt"] + code
            
            # Execute
            try:
                result = execute_code(full_code, test_code, timeout=timeout)
                category = categorize_failure(result)
                
                # Store results
                sample["execution_result"] = result
                sample["category"] = category
                
                category_counts[category] += 1
            
            except Exception as e:
                sample["execution_result"] = {
                    "success": False,
                    "error_type": "evaluation_error",
                    "error_message": str(e),
                }
                sample["category"] = "evaluation_error"
                category_counts["evaluation_error"] += 1
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Print statistics
    print()
    print("="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"✅ Evaluated {total_samples} samples")
    print(f"✅ Results saved to: {output_file}")
    print()
    print("CATEGORY DISTRIBUTION:")
    print("-"*60)
    
    for category, count in category_counts.most_common():
        percentage = (count / total_samples) * 100
        print(f"  {category:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print()
    print("NEXT STEP: Analyze failures")
    print("  python scripts/analyze_failures.py --config config.yaml")


if __name__ == "__main__":
    main()
