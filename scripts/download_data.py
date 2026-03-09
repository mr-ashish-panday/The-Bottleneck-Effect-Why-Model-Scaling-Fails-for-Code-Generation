#!/usr/bin/env python3
"""
Download HumanEval, MBPP, and LiveCodeBench datasets.

Usage:
    python scripts/download_data.py --dataset humaneval
    python scripts/download_data.py --dataset mbpp
    python scripts/download_data.py --dataset livecodebench --version_tag release_v2
    python scripts/download_data.py --dataset all
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

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
        prompt = item.get("prompt") or item.get("text")
        if prompt is None:
            raise KeyError("MBPP example is missing both 'prompt' and 'text'")

        test_setup_code = item.get("test_setup_code", "")
        if not test_setup_code and item.get("test_imports"):
            imports = item["test_imports"]
            if isinstance(imports, list):
                rendered_imports = []
                for entry in imports:
                    entry = str(entry).strip()
                    if not entry:
                        continue
                    if entry.startswith("import ") or entry.startswith("from "):
                        rendered_imports.append(entry)
                    else:
                        rendered_imports.append(f"import {entry}")
                test_setup_code = "\n".join(rendered_imports)
            else:
                test_setup_code = str(imports)

        data.append({
            "task_id": f"MBPP/{item['task_id']}",
            "prompt": prompt,
            "code": item["code"],
            "test_list": item["test_list"],
            "test_setup_code": test_setup_code,
            "challenge_test_list": item.get("challenge_test_list", []),
        })
    
    # Save as JSONL
    output_file = output_dir / "mbpp.jsonl"
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ MBPP saved to {output_file}")
    print(f"   Total problems: {len(data)}")


def _load_livecodebench_dataset(version_tag: str):
    """Load the official LiveCodeBench code_generation_lite dataset."""
    try:
        return load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=version_tag,
            split="test",
        )
    except TypeError:
        return load_dataset(
            "livecodebench/code_generation_lite",
            version_tag,
            split="test",
        )
    except ValueError:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=version_tag,
        )
        if "test" in dataset:
            return dataset["test"]
        return dataset


def download_livecodebench(output_dir: Path, version_tag: str) -> None:
    """Download LiveCodeBench code_generation_lite from Hugging Face."""
    print(f"Downloading LiveCodeBench ({version_tag}) dataset...")

    dataset = _load_livecodebench_dataset(version_tag)

    data = []
    for item in tqdm(dataset, desc="Processing LiveCodeBench"):
        question_content = item.get("question_content", "")
        starter_code = item.get("starter_code", "") or ""
        data.append({
            "task_id": str(item["question_id"]),
            "question_id": str(item["question_id"]),
            "prompt": question_content,
            "question_title": item.get("question_title", ""),
            "question_content": question_content,
            "starter_code": starter_code,
            "platform": item.get("platform", ""),
            "contest_id": item.get("contest_id", ""),
            "contest_date": item.get("contest_date", ""),
            "difficulty": item.get("difficulty", ""),
            "metadata": item.get("metadata", ""),
        })

    output_file = output_dir / f"livecodebench_{version_tag}.jsonl"
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ LiveCodeBench saved to {output_file}")
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
        choices=["humaneval", "mbpp", "livecodebench", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--version_tag",
        type=str,
        default="release_v2",
        help="LiveCodeBench version tag",
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

    if args.dataset in ["livecodebench", "all"]:
        try:
            download_livecodebench(output_dir, args.version_tag)
            stats = verify_dataset(output_dir / f"livecodebench_{args.version_tag}.jsonl")
            print(f"   Verification: {stats}")
        except Exception as e:
            print(f"❌ Failed to download LiveCodeBench: {e}")
    
    print("\n✅ Download complete!")
    print(f"   Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
