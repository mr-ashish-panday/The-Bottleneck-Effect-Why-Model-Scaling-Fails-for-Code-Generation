#!/usr/bin/env python3
"""
Fine-tuning dynamics experiment - Version 2
- Batch size 1 (prevents OOM)
- No checkpoint saving (saves disk space)
- All models run 10 epochs (comparable data)
- Results saved to training_trajectory.json (new name)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from pathlib import Path
from tqdm import tqdm
import gc

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
        'timeouts': 0,
        'total': 0,
    }
    
    for problem in tqdm(eval_problems, desc="Evaluating", leave=False):
        try:
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
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode and evaluate
            prompt_length = inputs['input_ids'].shape[1]
            for output in outputs:
                generated_ids = output[prompt_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                full_code = problem.prompt + generated_text
                result = execute_code(full_code, problem.test, timeout=5)
                category = categorize_failure(result)
                
                results['total'] += 1
                if category == 'success':
                    results['successes'] += 1
                elif category == 'syntax_error':
                    results['syntax_errors'] += 1
                elif category == 'timeout':
                    results['timeouts'] += 1
                else:
                    results['runtime_errors'] += 1
            
            # Clear memory after each problem
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\n⚠️ Error evaluating {problem.task_id}: {e}")
            continue
    
    # Calculate percentages
    total = results['total']
    if total > 0:
        results['syntax_error_pct'] = results['syntax_errors'] / total * 100
        results['success_pct'] = results['successes'] / total * 100
        results['runtime_error_pct'] = results['runtime_errors'] / total * 100
        results['timeout_pct'] = results['timeouts'] / total * 100
    else:
        results['syntax_error_pct'] = 0
        results['success_pct'] = 0
        results['runtime_error_pct'] = 0
        results['timeout_pct'] = 0
    
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
    print("FINE-TUNING DYNAMICS EXPERIMENT V2")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training problems: {args.num_train_problems}")
    print(f"Eval problems: {args.num_eval_problems}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: 1 (memory safe)")
    print(f"Checkpoint saving: DISABLED (saves disk space)")
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
    
    # Training arguments - NO CHECKPOINTS
    training_args = TrainingArguments(
        output_dir=str(output_dir / "temp_checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,  # BATCH SIZE 1
        gradient_accumulation_steps=2,   # Accumulate gradients
        save_strategy="no",               # NO CHECKPOINT SAVING
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=100,
        report_to="none",
        fp16=True,  # Mixed precision for memory efficiency
        dataloader_num_workers=0,  # Prevent memory issues
    )
    
    # Custom trainer to evaluate at each epoch
    class CustomTrainer(Trainer):
        def __init__(self, *args, eval_func=None, eval_args=None, results_list=None, output_path=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_func = eval_func
            self.eval_args = eval_args
            self.results_list = results_list
            self.output_path = output_path
        
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
                
                # Save results incrementally to NEW FILENAME
                with open(self.output_path, 'w') as f:
                    json.dump(self.results_list, f, indent=2)
                
                print(f"💾 Saved to {self.output_path}")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return super().on_epoch_end(args, state, control, **kwargs)
    
    # Train with custom trainer
    output_json = output_dir / "training_trajectory.json"  # NEW FILENAME
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_func=evaluate_checkpoint,
        eval_args=(tokenizer, eval_problems, device, args.eval_samples),
        results_list=all_results,
        output_path=output_json,
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("⚠️  NO CHECKPOINTS WILL BE SAVED (disk space optimization)")
    print(f"📊 Results will be saved to: {output_json}")
    print()
    
    trainer.train()
    
    # Final save
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Final results saved to {output_json}")
    
    # Print summary
    print("\nTRAINING TRAJECTORY:")
    print("-"*60)
    print(f"{'Epoch':<8} {'Syntax %':<12} {'Success %':<12} {'Runtime %':<12}")
    print("-"*60)
    for r in all_results:
        print(f"{r['epoch']:<8} {r['syntax_error_pct']:<12.1f} {r['success_pct']:<12.1f} {r['runtime_error_pct']:<12.1f}")
    
    # Clean up temp directory
    import shutil
    temp_dir = output_dir / "temp_checkpoints"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n🗑️  Cleaned up temporary files")

if __name__ == "__main__":
    main()
