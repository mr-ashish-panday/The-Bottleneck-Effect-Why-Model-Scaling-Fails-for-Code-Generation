#!/usr/bin/env python3
"""
Download HumanEval and MBPP datasets.

Usage:
    python scripts/download_data.py --dataset humaneval
    python scripts/download_data.py --dataset mbpp
    python scripts/download_data.py --dataset all
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm


def download_humaneval(output_dir: Path) -> None:
    """Download HumanEval dataset from HuggingFace."""
    print("Downloading HumanEval dataset...")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("openai_humaneval", split="test")
    
    # Convert to list of dicts
    data = []
    for item in tqdm(dataset, desc="Processing HumanEval"):
        data.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        })
    
    # Save as JSONL
    output_file = output_dir / "humaneval.jsonl"
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ HumanEval saved to {output_file}")
    print(f"   Total problems: {len(data)}")


def download_mbpp(output_dir: Path) -> None:
    """Download MBPP dataset from HuggingFace."""
    print("Downloading MBPP dataset...")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("mbpp", "sanitized", split="test")
    
    # Convert to list of dicts
    data = []
    for item in tqdm(dataset, desc="Processing MBPP"):
        data.append({
            "task_id": f"MBPP/{item['task_id']}",
            "prompt": item["text"],
            "code": item["code"],
            "test_list": item["test_list"],
            "test_setup_code": item.get("test_setup_code", ""),
            "challenge_test_list": item.get("challenge_test_list", []),
        })
    
    # Save as JSONL
    output_file = output_dir / "mbpp.jsonl"
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ MBPP saved to {output_file}")
    print(f"   Total problems: {len(data)}")


def verify_dataset(dataset_path: Path) -> Dict[str, int]:
    """Verify downloaded dataset integrity."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    stats = {
        "total_problems": len(data),
        "avg_prompt_length": sum(len(item["prompt"]) for item in data) // len(data),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download code generation benchmarks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["humaneval", "mbpp", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for datasets",
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    if args.dataset in ["humaneval", "all"]:
        try:
            download_humaneval(output_dir)
            stats = verify_dataset(output_dir / "humaneval.jsonl")
            print(f"   Verification: {stats}")
        except Exception as e:
            print(f"❌ Failed to download HumanEval: {e}")
    
    if args.dataset in ["mbpp", "all"]:
        try:
            download_mbpp(output_dir)
            stats = verify_dataset(output_dir / "mbpp.jsonl")
            print(f"   Verification: {stats}")
        except Exception as e:
            print(f"❌ Failed to download MBPP: {e}")
    
    print("\n✅ Download complete!")
    print(f"   Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()