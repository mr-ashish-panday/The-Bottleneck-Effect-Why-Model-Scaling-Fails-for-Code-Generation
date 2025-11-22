#!/usr/bin/env python3
"""
Generate code samples for HumanEval problems.
This is the main script for Phase 1-2 feasibility check.

Usage:
    python scripts/generate_samples.py --num_problems 50 --num_samples 100
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import yaml

import sys
sys.path.append('.')

from src.data.dataset_loader import DatasetLoader
from src.models.model_wrapper import CodeGenerationModel


def main():
    parser = argparse.ArgumentParser(description="Generate code samples")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to use (default: use config)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Samples per problem (default: use config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: use config)",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    num_problems = args.num_problems or config['feasibility_check']['num_problems']
    num_samples = args.num_samples or config['feasibility_check']['num_samples_per_problem']
    output_dir = Path(args.output_dir or config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PHASE 1-2: FEASIBILITY CHECK - SAMPLE GENERATION")
    print("="*60)
    print(f"Number of problems: {num_problems}")
    print(f"Samples per problem: {num_samples}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    loader = DatasetLoader(args.config)
    problems = loader.load(num_problems=num_problems)
    print(f"✅ Loaded {len(problems)} problems")
    print()
    
    # Load model
    print("Loading model...")
    model = CodeGenerationModel(args.config)
    model.load_model()
    print()
    
    # Generate samples
    print("Generating samples...")
    all_results = []
    
    for problem in tqdm(problems, desc="Problems"):
        problem_results = {
            "task_id": problem.task_id,
            "prompt": problem.prompt,
            "canonical_solution": problem.canonical_solution,
            "test": problem.test,
            "entry_point": problem.entry_point,
            "samples": [],
        }
        
        # Generate samples for this problem
        try:
            samples = model.generate(
                prompt=problem.prompt,
                num_samples=num_samples,
            )
            
            for i, sample in enumerate(samples):
                problem_results["samples"].append({
                    "sample_id": i,
                    "code": sample,
                })
        
        except Exception as e:
            print(f"❌ Error generating for {problem.task_id}: {e}")
            continue
        
        all_results.append(problem_results)
        
        # Save incrementally (in case of crash)
        if len(all_results) % 10 == 0:
            temp_output = output_dir / f"samples_temp_{len(all_results)}.json"
            with open(temp_output, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Save final results
    output_file = output_dir / "generated_samples.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print("="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"✅ Generated samples for {len(all_results)} problems")
    print(f"✅ Total samples: {len(all_results) * num_samples}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("NEXT STEP: Run evaluation")
    print("  python scripts/run_evaluation.py --config config.yaml")


if __name__ == "__main__":
    main()
