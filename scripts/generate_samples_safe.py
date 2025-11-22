#!/usr/bin/env python3
"""
Safe generation with GPU memory management and automatic recovery.
"""

import argparse
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
import yaml

import sys
sys.path.append('.')

from src.data.dataset_loader import DatasetLoader
from src.models.model_wrapper import CodeGenerationModel


def generate_with_memory_cleanup(model, prompt, num_samples):
    """Generate samples with automatic memory cleanup on OOM."""
    try:
        samples = model.generate(prompt=prompt, num_samples=num_samples)
        return samples, None
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear cache and retry with smaller batch
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Retry with half the samples at a time
                samples = []
                for i in range(0, num_samples, num_samples // 2):
                    batch_samples = model.generate(
                        prompt=prompt,
                        num_samples=min(num_samples // 2, num_samples - i)
                    )
                    samples.extend(batch_samples)
                    torch.cuda.empty_cache()
                
                return samples, None
            except Exception as retry_error:
                return [], str(retry_error)
        else:
            return [], str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_codegen.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from failures")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['paths']['results_dir'])
    
    # Load existing results if resuming
    existing_results = {}
    results_file = results_dir / "generated_samples.json"
    if args.resume and results_file.exists():
        with open(results_file) as f:
            existing_data = json.load(f)
            existing_results = {item['task_id']: item for item in existing_data}
        print(f"Resuming: Found {len(existing_results)} existing problems")
    
    # Load dataset
    loader = DatasetLoader(args.config)
    all_problems = loader.load(num_problems=164)
    
    # Filter to only missing problems
    problems_to_generate = [p for p in all_problems if p.task_id not in existing_results]
    print(f"Need to generate: {len(problems_to_generate)} problems")
    
    if not problems_to_generate:
        print("All problems already generated!")
        return
    
    # Load model
    model = CodeGenerationModel(args.config)
    model.load_model()
    
    num_samples = config['feasibility_check']['num_samples_per_problem']
    all_results = list(existing_results.values())
    
    for problem in tqdm(problems_to_generate, desc="Problems"):
        samples, error = generate_with_memory_cleanup(model, problem.prompt, num_samples)
        
        problem_result = {
            "task_id": problem.task_id,
            "prompt": problem.prompt,
            "canonical_solution": problem.canonical_solution,
            "test": problem.test,
            "entry_point": problem.entry_point,
            "samples": [],
        }
        
        if error:
            print(f"\n⚠️  {problem.task_id}: {error}")
        
        for i, sample in enumerate(samples):
            problem_result["samples"].append({
                "sample_id": i,
                "code": sample,
            })
        
        all_results.append(problem_result)
        
        # Save after each problem
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n✅ Complete: {len(all_results)} problems")


if __name__ == "__main__":
    main()
