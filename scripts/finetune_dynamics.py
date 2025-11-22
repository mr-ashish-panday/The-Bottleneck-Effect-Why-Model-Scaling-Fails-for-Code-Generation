#!/usr/bin/env python3
"""
Track how error distribution changes during fine-tuning.
Saves checkpoints at different epochs for analysis.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from pathlib import Path
from tqdm import tqdm
import yaml

import sys
sys.path.append('.')

from src.data.dataset_loader import DatasetLoader
from src.evaluation.code_executor import execute_code, categorize_failure


class CodeDataset(Dataset):
    """Dataset for code fine-tuning."""
    
    def __init__(self, problems, tokenizer, max_length=512):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        # Combine prompt + solution
        full_text = problem.prompt + problem.canonical_solution
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }


def evaluate_checkpoint(model, tokenizer, eval_problems, device, num_samples=20):
    """Evaluate model at current checkpoint."""
    
    model.eval()
    
    results = {
        'syntax_errors': 0,
        'successes': 0,
        'runtime_errors': 0,
        'total': 0,
    }
    
    for problem in tqdm(eval_problems, desc="Evaluating", leave=False):
        # Generate samples
        inputs = tokenizer(problem.prompt, return_tensors="pt").to(device)
        
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
        
        # Decode and evaluate
        for output in outputs:
            generated_text = tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            full_code = problem.prompt + generated_text
            result = execute_code(full_code, problem.test, timeout=5)
            category = categorize_failure(result)
            
            results['total'] += 1
            if category == 'success':
                results['successes'] += 1
            elif category == 'syntax_error':
                results['syntax_errors'] += 1
            else:
                results['runtime_errors'] += 1
    
    # Calculate percentages
    total = results['total']
    results['syntax_error_pct'] = results['syntax_errors'] / total * 100
    results['success_pct'] = results['successes'] / total * 100
    results['runtime_error_pct'] = results['runtime_errors'] / total * 100
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model to fine-tune")
    parser.add_argument("--num_train_problems", type=int, default=130, help="Training set size")
    parser.add_argument("--num_eval_problems", type=int, default=30, help="Eval set size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--eval_samples", type=int, default=20, help="Samples per problem for eval")
    parser.add_argument("--output_dir", default="data/results_finetuning")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FINE-TUNING DYNAMICS EXPERIMENT")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training problems: {args.num_train_problems}")
    print(f"Eval problems: {args.num_eval_problems}")
    print(f"Epochs: {args.num_epochs}")
    print()
    
    # Load data
    loader = DatasetLoader("config.yaml")
    all_problems = loader.load(num_problems=164)
    
    # Split train/eval
    train_problems = all_problems[:args.num_train_problems]
    eval_problems = all_problems[args.num_train_problems:args.num_train_problems + args.num_eval_problems]
    
    print(f"Train: {len(train_problems)} problems")
    print(f"Eval: {len(eval_problems)} problems")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Create dataset
    train_dataset = CodeDataset(train_problems, tokenizer)
    
    # Evaluate BEFORE fine-tuning (Epoch 0)
    print("\n" + "="*60)
    print("EPOCH 0 (Pretrained)")
    print("="*60)
    
    epoch0_results = evaluate_checkpoint(
        model, tokenizer, eval_problems, device, args.eval_samples
    )
    
    print(f"Syntax errors: {epoch0_results['syntax_error_pct']:.1f}%")
    print(f"Success: {epoch0_results['success_pct']:.1f}%")
    print(f"Runtime errors: {epoch0_results['runtime_error_pct']:.1f}%")
    
    all_results = [{'epoch': 0, **epoch0_results}]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        save_steps=len(train_dataset) // 2,  # Save every epoch
        save_total_limit=args.num_epochs,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=100,
        report_to="none",
    )
    
    # Custom trainer to evaluate at each epoch
    class CustomTrainer(Trainer):
        def __init__(self, *args, eval_func=None, eval_args=None, results_list=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_func = eval_func
            self.eval_args = eval_args
            self.results_list = results_list
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if self.eval_func is not None:
                epoch = int(state.epoch)
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch}")
                print("="*60)
                
                results = self.eval_func(
                    self.model,
                    *self.eval_args
                )
                
                print(f"Syntax errors: {results['syntax_error_pct']:.1f}%")
                print(f"Success: {results['success_pct']:.1f}%")
                print(f"Runtime errors: {results['runtime_error_pct']:.1f}%")
                
                self.results_list.append({'epoch': epoch, **results})
                
                # Save results
                with open(output_dir / "training_dynamics.json", 'w') as f:
                    json.dump(self.results_list, f, indent=2)
            
            return super().on_epoch_end(args, state, control, **kwargs)
    
    # Train
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_func=evaluate_checkpoint,
        eval_args=(tokenizer, eval_problems, device, args.eval_samples),
        results_list=all_results,
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    trainer.train()
    
    # Final save
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    with open(output_dir / "training_dynamics.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir / 'training_dynamics.json'}")
    
    # Print summary
    print("\nTRAINING TRAJECTORY:")
    print("-"*60)
    print(f"{'Epoch':<8} {'Syntax %':<12} {'Success %':<12} {'Runtime %':<12}")
    print("-"*60)
    for r in all_results:
        print(f"{r['epoch']:<8} {r['syntax_error_pct']:<12.1f} {r['success_pct']:<12.1f} {r['runtime_error_pct']:<12.1f}")

if __name__ == "__main__":
    main()
