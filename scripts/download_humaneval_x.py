#!/usr/bin/env python3
"""Download HumanEval-X for multilingual validation."""

from datasets import load_dataset
import json
from pathlib import Path

def download_language(lang, output_dir):
    """Download HumanEval-X for specific language."""
    print(f"Downloading HumanEval-X ({lang})...")
    
    # Load from HuggingFace
    dataset = load_dataset("THUDM/humaneval-x", lang, split="test")
    
    # Convert to our format
    data = []
    for item in dataset:
        data.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "declaration": item.get("declaration", ""),
            "language": lang,
        })
    
    # Save
    output_file = output_dir / f"humaneval_x_{lang}.jsonl"
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Saved {len(data)} problems to {output_file}")
    return len(data)

if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Java and JavaScript
    for lang in ["java", "js"]:
        try:
            count = download_language(lang, output_dir)
            print(f"  {lang}: {count} problems")
        except Exception as e:
            print(f"❌ Failed to download {lang}: {e}")
