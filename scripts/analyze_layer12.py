#!/usr/bin/env python3
"""
Deep dive into Layer 12 - the bottleneck layer.
"""

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

import sys
sys.path.append('.')

from src.models.model_wrapper import CodeGenerationModel
from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import execute_code, categorize_failure

def extract_layer12_activations(model, tokenizer, prompt, code):
    """Extract Layer 12 hidden states."""
    
    full_code = prompt + code
    inputs = tokenizer(full_code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    layer12_activation = None
    
    def hook_fn(module, input, output):
        nonlocal layer12_activation
        if isinstance(output, tuple):
            layer12_activation = output[0].detach().cpu()
        else:
            layer12_activation = output.detach().cpu()
    
    # Hook Layer 12
    handle = model.transformer.h[12].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    # Return mean activation over sequence
    return layer12_activation.mean(dim=1).squeeze().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_gpt2_medium.yaml")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()
    
    # Load evaluation results
    results_file = Path(config['paths']['results_dir']) / "evaluation_results.json"
    with open(results_file) as f:
        all_results = json.load(f)
    
    # Collect successful and failed samples
    success_samples = []
    failure_samples = []
    
    print("Collecting samples...")
    for problem in all_results:
        for sample in problem['samples']:
            if len(success_samples) >= args.num_samples // 2 and len(failure_samples) >= args.num_samples // 2:
                break
            
            category = sample.get('category')
            if category == 'success' and len(success_samples) < args.num_samples // 2:
                success_samples.append({
                    'prompt': problem['prompt'],
                    'code': sample['code'],
                    'category': 'success'
                })
            elif category == 'syntax_error' and len(failure_samples) < args.num_samples // 2:
                failure_samples.append({
                    'prompt': problem['prompt'],
                    'code': sample['code'],
                    'category': 'syntax_error'
                })
    
    print(f"Extracted {len(success_samples)} successes, {len(failure_samples)} failures")
    
    # Extract activations
    print("Extracting Layer 12 activations...")
    
    success_activations = []
    for sample in tqdm(success_samples, desc="Success samples"):
        act = extract_layer12_activations(
            model_wrapper.model,
            model_wrapper.tokenizer,
            sample['prompt'],
            sample['code']
        )
        success_activations.append(act)
    
    failure_activations = []
    for sample in tqdm(failure_samples, desc="Failure samples"):
        act = extract_layer12_activations(
            model_wrapper.model,
            model_wrapper.tokenizer,
            sample['prompt'],
            sample['code']
        )
        failure_activations.append(act)
    
    # Analyze differences
    success_mean = np.mean(success_activations, axis=0)
    failure_mean = np.mean(failure_activations, axis=0)
    
    difference = success_mean - failure_mean
    
    # Find most discriminative dimensions
    top_dims = np.argsort(np.abs(difference))[-20:]
    
    results = {
        'success_mean': success_mean.tolist(),
        'failure_mean': failure_mean.tolist(),
        'difference': difference.tolist(),
        'top_discriminative_dims': top_dims.tolist(),
        'difference_magnitude': float(np.mean(np.abs(difference))),
    }
    
    # Save
    output_file = Path(config['paths']['results_dir']) / "ablation" / "layer12_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete")
    print(f"Mean absolute difference: {results['difference_magnitude']:.4f}")
    print(f"Top discriminative dimensions: {top_dims[:5]}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
