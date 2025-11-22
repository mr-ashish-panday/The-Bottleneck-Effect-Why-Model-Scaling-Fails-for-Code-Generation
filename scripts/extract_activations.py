#!/usr/bin/env python3
"""
Extract hidden state activations from model for successful vs failed samples.
This is VERY HEAVY - processes model activations for thousands of samples.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np

import sys
sys.path.append('.')

from src.models.model_wrapper import CodeGenerationModel


def extract_sample_activations(model, tokenizer, prompt, code, layers=[0, 3, 6, 9, 11]):
    """Extract activations at specific layers for a code sample."""
    
    full_code = prompt + code
    inputs = tokenizer(full_code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activations = {}
    
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        for layer_idx in layers:
            # Get last token representation
            layer_act = hidden_states[layer_idx][0, -1, :].cpu().numpy()
            activations[f"layer_{layer_idx}"] = layer_act.tolist()
    
    return activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num_samples", type=int, default=1000, help="Samples to extract (per category)")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['paths']['results_dir'])
    eval_file = results_dir / "evaluation_results.json"
    output_file = Path(args.output_file or results_dir / "activations.json")
    
    print("="*60)
    print("ACTIVATION EXTRACTION (HEAVY COMPUTATION)")
    print("="*60)
    print(f"Extracting activations for {args.num_samples} samples per category")
    print("This will take 2-3 hours...")
    print()
    
    # Load model
    print("Loading model...")
    model_wrapper = CodeGenerationModel(args.config)
    model_wrapper.load_model()
    
    # Load evaluation results
    with open(eval_file) as f:
        all_samples = json.load(f)
    
    # Collect samples by category
    samples_by_category = {"success": [], "syntax_error": [], "runtime_error": []}
    
    for problem in all_samples:
        for sample in problem["samples"]:
            category = sample.get("category")
            if category in samples_by_category and len(samples_by_category[category]) < args.num_samples:
                samples_by_category[category].append({
                    "task_id": problem["task_id"],
                    "sample_id": sample["sample_id"],
                    "prompt": problem["prompt"],
                    "code": sample["code"],
                    "category": category,
                })
    
    print(f"Collected samples: {[(cat, len(samps)) for cat, samps in samples_by_category.items()]}")
    print()
    
    # Extract activations
    all_activations = []
    
    for category, samples in samples_by_category.items():
        print(f"Extracting activations for {category}...")
        
        for sample in tqdm(samples, desc=category):
            try:
                activations = extract_sample_activations(
                    model_wrapper,
                    model_wrapper.tokenizer,
                    sample["prompt"],
                    sample["code"]
                )
                
                all_activations.append({
                    "task_id": sample["task_id"],
                    "sample_id": sample["sample_id"],
                    "category": category,
                    "activations": activations,
                })
            except Exception as e:
                print(f"Error processing {sample['task_id']}: {e}")
                continue
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(all_activations, f, indent=2)
    
    print()
    print(f"✅ Extracted {len(all_activations)} activation samples")
    print(f"✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
