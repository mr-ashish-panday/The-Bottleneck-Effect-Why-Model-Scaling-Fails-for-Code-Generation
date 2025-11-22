#!/usr/bin/env python3
"""
Measure which layers contribute most to successful generation.
Uses gradient-based attribution instead of ablation.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import yaml

import sys
sys.path.append('.')

from src.models.model_wrapper import CodeGenerationModel
from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import execute_code

def compute_layer_gradients(model, tokenizer, prompt, code, target="success"):
    """Compute gradient of success w.r.t. each layer."""
    
    full_code = prompt + code
    inputs = tokenizer(full_code, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Enable gradients
    model.zero_grad()
    
    # Forward pass with hooks to capture layer outputs
    layer_outputs = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            layer_outputs.append(output[0])
        else:
            layer_outputs.append(output)
    
    handles = []
    for layer in model.transformer.h:
        handles.append(layer.register_forward_hook(hook_fn))
    
    try:
        # Forward
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Backward
        loss.backward()
        
        # Get gradients for each layer
        layer_grads = []
        for layer_out in layer_outputs:
            if layer_out.grad is not None:
                grad_norm = layer_out.grad.norm().item()
                layer_grads.append(grad_norm)
            else:
                layer_grads.append(0.0)
        
    finally:
        for h in handles:
            h.remove()
    
    return layer_grads

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_gpt2_medium.yaml")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()
    
    # Enable gradients
    model_wrapper.model.train()
    for param in model_wrapper.model.parameters():
        param.requires_grad = True
    
    # Load evaluation results
    results_file = Path(config['paths']['results_dir']) / "evaluation_results.json"
    with open(results_file) as f:
        all_results = json.load(f)
    
    # Collect successful samples
    success_samples = []
    for problem in all_results:
        for sample in problem['samples'][:10]:  # First 10 per problem
            if sample.get('category') == 'success':
                success_samples.append({
                    'prompt': problem['prompt'],
                    'code': sample['code']
                })
            if len(success_samples) >= args.num_samples:
                break
        if len(success_samples) >= args.num_samples:
            break
    
    print(f"Analyzing {len(success_samples)} successful samples...")
    
    # Compute gradients
    all_layer_grads = []
    
    for sample in tqdm(success_samples):
        grads = compute_layer_gradients(
            model_wrapper.model,
            model_wrapper.tokenizer,
            sample['prompt'],
            sample['code']
        )
        all_layer_grads.append(grads)
    
    # Average across samples
    import numpy as np
    avg_grads = np.mean(all_layer_grads, axis=0)
    
    # Normalize
    avg_grads = avg_grads / avg_grads.sum()
    
    print("\n" + "="*60)
    print("GRADIENT-BASED LAYER IMPORTANCE")
    print("="*60)
    for i, grad in enumerate(avg_grads):
        print(f"Layer {i:2d}: {grad:.4f}")
    
    # Find most important layers
    top_layers = np.argsort(avg_grads)[-5:][::-1]
    print(f"\nTop 5 most important layers: {top_layers}")
    
    # Save
    output_file = Path(config['paths']['results_dir']) / "ablation" / "gradient_attribution.json"
    with open(output_file, 'w') as f:
        json.dump({
            'layer_importance': avg_grads.tolist(),
            'top_layers': top_layers.tolist(),
        }, f, indent=2)
    
    print(f"\n✅ Saved to: {output_file}")

if __name__ == "__main__":
    main()
