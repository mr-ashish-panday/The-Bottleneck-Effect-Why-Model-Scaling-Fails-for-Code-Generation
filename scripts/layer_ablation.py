#!/usr/bin/env python3
"""
Layer ablation study - which layers control which error types?
CRITICAL EXPERIMENT: Explains WHY GPT-2 Medium is worse.
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import gc

import sys
sys.path.append('.')

from src.models.model_wrapper import CodeGenerationModel
from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import execute_code, categorize_failure

def generate_with_layer_ablation(model, tokenizer, prompt, layer_to_ablate, num_samples=10):
    """Generate samples with one layer's output zeroed."""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Hook to zero out layer
    activations = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            # Zero out hidden states
            zeroed = torch.zeros_like(output[0])
            activations.append(output[0].detach().cpu())  # Save original
            return (zeroed, *output[1:])
        zeroed = torch.zeros_like(output)
        activations.append(output.detach().cpu())
        return zeroed
    
    # Register hook
    layer = model.transformer.h[layer_to_ablate]
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        samples = []
        prompt_length = inputs['input_ids'].shape[1]
        for output in outputs:
            generated_ids = output[prompt_length:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            samples.append(text)
    
    finally:
        handle.remove()
        torch.cuda.empty_cache()
    
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_config", default="config_gpt2_medium.yaml", 
                       help="Which model to ablate (gpt2 or gpt2-medium)")
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--samples_per_problem", type=int, default=20)
    args = parser.parse_args()
    
    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['paths']['results_dir']) / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {config['model']['name']}")
    model_wrapper = CodeGenerationModel(args.model_config)
    model_wrapper.load_model()
    
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    
    # Load data
    loader = DatasetLoader(args.config)
    problems = loader.load(num_problems=args.num_problems)
    
    num_layers = len(model.transformer.h)
    print(f"\nModel has {num_layers} layers")
    print(f"Testing ablation on {len(problems)} problems")
    print(f"Generating {args.samples_per_problem} samples per problem\n")
    
    all_results = []
    
    # Test each layer
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        layer_stats = {
            "layer": layer_idx,
            "total_samples": 0,
            "syntax_errors": 0,
            "successes": 0,
            "runtime_errors": 0,
            "timeouts": 0,
            "problems": []
        }
        
        for problem in tqdm(problems, desc=f"Layer {layer_idx}", leave=False):
            try:
                # Generate with ablation
                samples = generate_with_layer_ablation(
                    model,
                    tokenizer,
                    problem.prompt,
                    layer_idx,
                    num_samples=args.samples_per_problem
                )
                
                # Evaluate
                problem_stats = {
                    "task_id": problem.task_id,
                    "syntax_errors": 0,
                    "successes": 0,
                    "runtime_errors": 0,
                }
                
                for sample in samples:
                    full_code = problem.prompt + sample
                    result = execute_code(full_code, problem.test, timeout=5)
                    category = categorize_failure(result)
                    
                    layer_stats["total_samples"] += 1
                    problem_stats[category + "s"] = problem_stats.get(category + "s", 0) + 1
                    
                    if category == "syntax_error":
                        layer_stats["syntax_errors"] += 1
                    elif category == "success":
                        layer_stats["successes"] += 1
                    elif category == "runtime_error":
                        layer_stats["runtime_errors"] += 1
                    elif category == "timeout":
                        layer_stats["timeouts"] += 1
                
                layer_stats["problems"].append(problem_stats)
                
            except Exception as e:
                print(f"\n⚠️ Error on {problem.task_id} layer {layer_idx}: {e}")
                continue
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        # Calculate percentages
        total = layer_stats["total_samples"]
        if total > 0:
            layer_stats["syntax_error_pct"] = layer_stats["syntax_errors"] / total * 100
            layer_stats["success_pct"] = layer_stats["successes"] / total * 100
            layer_stats["runtime_error_pct"] = layer_stats["runtime_errors"] / total * 100
        
        all_results.append(layer_stats)
        
        # Save incrementally
        output_file = results_dir / "layer_ablation_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nLayer {layer_idx}: {layer_stats['syntax_error_pct']:.1f}% syntax errors, "
              f"{layer_stats['success_pct']:.1f}% success")
    
    # Summary
    print("\n" + "="*60)
    print("LAYER ABLATION SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"Layer {result['layer']:2d}: "
              f"Syntax {result.get('syntax_error_pct', 0):5.1f}%  "
              f"Success {result.get('success_pct', 0):5.1f}%  "
              f"Runtime {result.get('runtime_error_pct', 0):5.1f}%")
    
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
