#!/usr/bin/env python3
"""
Deep syntax error analysis - classify syntax errors into subcategories.
This is HEAVY - analyzes all 15,489 syntax errors in detail.
"""

import argparse
import json
import ast
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import yaml


def classify_syntax_error(code: str, error_msg: str) -> dict:
    """
    Classify syntax error into detailed subcategories.
    
    Categories:
    - indentation: Wrong indentation
    - bracket_mismatch: Missing/extra brackets (), [], {}
    - quote_mismatch: Unclosed strings
    - colon_missing: Missing : after if/def/for/while
    - operator_error: Wrong operators (= vs ==, etc.)
    - keyword_error: Misspelled keywords
    - incomplete_statement: Incomplete lines
    - other: Other syntax issues
    """
    
    classification = {
        "category": "other",
        "subcategory": "unknown",
        "line_number": None,
        "error_token": None,
    }
    
    # Extract line number from error message
    if "line" in error_msg.lower():
        try:
            line_num = int(re.search(r'line (\d+)', error_msg).group(1))
            classification["line_number"] = line_num
        except:
            pass
    
    error_lower = error_msg.lower()
    
    # Indentation errors
    if "indent" in error_lower or "unexpected indent" in error_lower:
        classification["category"] = "indentation"
        classification["subcategory"] = "unexpected_indent"
        return classification
    
    # Colon errors
    if "expected ':'" in error_msg or "invalid syntax. Perhaps you forgot a comma" in error_msg:
        classification["category"] = "colon_missing"
        classification["subcategory"] = "missing_colon"
        return classification
    
    # Bracket/paren errors
    if any(x in error_lower for x in ["')'", "'('", "']'", "'['", "'}'", "'{'"]):
        classification["category"] = "bracket_mismatch"
        
        if "'('" in error_msg or "')'" in error_msg:
            classification["subcategory"] = "parenthesis"
        elif "'['" in error_msg or "']'" in error_msg:
            classification["subcategory"] = "square_bracket"
        elif "'{'" in error_msg or "'}'" in error_msg:
            classification["subcategory"] = "curly_brace"
        
        return classification
    
    # String/quote errors
    if "unterminated string" in error_lower or "eol while scanning" in error_lower:
        classification["category"] = "quote_mismatch"
        classification["subcategory"] = "unclosed_string"
        return classification
    
    # Operator errors
    if "==" in error_msg or "'='" in error_msg and "or" in error_msg:
        classification["category"] = "operator_error"
        classification["subcategory"] = "assignment_vs_comparison"
        return classification
    
    # Keyword errors
    if "invalid syntax" in error_lower and any(kw in code.lower() for kw in ["dfe", "fi", "retrun", "pritn"]):
        classification["category"] = "keyword_error"
        classification["subcategory"] = "misspelled_keyword"
        return classification
    
    # Incomplete statements
    if "incomplete" in error_lower or "unexpected EOF" in error_lower:
        classification["category"] = "incomplete_statement"
        classification["subcategory"] = "truncated_code"
        return classification
    
    # Extract error token if present
    if "'" in error_msg:
        try:
            token = re.search(r"'([^']+)'", error_msg).group(1)
            classification["error_token"] = token
        except:
            pass
    
    return classification


def analyze_error_position(code: str, error_classification: dict) -> dict:
    """Analyze WHERE in the code the error occurred."""
    
    lines = code.split('\n')
    total_lines = len(lines)
    line_num = error_classification.get("line_number")
    
    position_info = {
        "total_lines": total_lines,
        "error_line": line_num,
        "error_position": "unknown",
        "code_prefix_length": 0,
    }
    
    if line_num is None:
        return position_info
    
    # Calculate position (early, middle, late)
    if line_num <= total_lines * 0.33:
        position_info["error_position"] = "early"
    elif line_num <= total_lines * 0.67:
        position_info["error_position"] = "middle"
    else:
        position_info["error_position"] = "late"
    
    # Count tokens before error
    prefix = '\n'.join(lines[:line_num])
    position_info["code_prefix_length"] = len(prefix)
    
    return position_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['paths']['results_dir'])
    input_file = Path(args.input_file or results_dir / "evaluation_results.json")
    output_file = Path(args.output_file or results_dir / "syntax_analysis.json")
    
    print("="*60)
    print("DEEP SYNTAX ERROR ANALYSIS")
    print("="*60)
    print(f"Input: {input_file}")
    print()
    
    # Load data
    with open(input_file) as f:
        all_samples = json.load(f)
    
    # Analyze all syntax errors
    syntax_errors = []
    category_counts = Counter()
    subcategory_counts = Counter()
    position_counts = Counter()
    
    print("Analyzing syntax errors in detail...")
    
    for problem in tqdm(all_samples, desc="Problems"):
        for sample in problem["samples"]:
            if sample.get("category") == "syntax_error":
                code = problem["prompt"] + sample["code"]
                error_msg = sample["execution_result"]["error_message"]
                
                # Classify error
                classification = classify_syntax_error(code, error_msg)
                
                # Analyze position
                position = analyze_error_position(code, classification)
                
                # Count
                category_counts[classification["category"]] += 1
                subcategory = f"{classification['category']}:{classification['subcategory']}"
                subcategory_counts[subcategory] += 1
                position_counts[position["error_position"]] += 1
                
                # Store
                syntax_errors.append({
                    "task_id": problem["task_id"],
                    "sample_id": sample["sample_id"],
                    "classification": classification,
                    "position": position,
                    "error_message": error_msg,
                })
    
    # Create report
    report = {
        "total_syntax_errors": len(syntax_errors),
        "category_distribution": dict(category_counts),
        "subcategory_distribution": dict(subcategory_counts.most_common(20)),
        "position_distribution": dict(position_counts),
        "detailed_errors": syntax_errors[:100],  # First 100 for inspection
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print()
    print("="*60)
    print("SYNTAX ERROR BREAKDOWN")
    print("="*60)
    print(f"Total syntax errors: {len(syntax_errors)}")
    print()
    print("MAIN CATEGORIES:")
    for cat, count in category_counts.most_common():
        pct = count / len(syntax_errors) * 100
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")
    
    print()
    print("TOP SUBCATEGORIES:")
    for subcat, count in subcategory_counts.most_common(10):
        pct = count / len(syntax_errors) * 100
        print(f"  {subcat:35s}: {count:5d} ({pct:5.1f}%)")
    
    print()
    print("ERROR POSITIONS:")
    for pos, count in position_counts.most_common():
        pct = count / len(syntax_errors) * 100
        print(f"  {pos:10s}: {count:5d} ({pct:5.1f}%)")
    
    print()
    print(f"✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
