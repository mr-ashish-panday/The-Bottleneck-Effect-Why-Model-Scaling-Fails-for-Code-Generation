#!/usr/bin/env python3
"""
Extract the real top-dimension activation statistics used in the manuscript table.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_file",
        default="data/results_gpt2_medium/ablation/layer12_analysis.json",
    )
    parser.add_argument(
        "--output_file",
        default="data/results_gpt2_medium/ablation/table4_data.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.analysis_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    success_mean = np.asarray(data["success_mean"], dtype=np.float32)
    failure_mean = np.asarray(data["failure_mean"], dtype=np.float32)
    difference = success_mean - failure_mean

    top_dims = [int(dim) for dim in data["top_discriminative_dims"][:5]]
    total_abs_signal = float(np.abs(difference).sum())

    table_data = []
    for dim in top_dims:
        succ_val = float(success_mean[dim])
        fail_val = float(failure_mean[dim])
        diff_val = float(abs(difference[dim]))
        table_data.append(
            {
                "dimension": dim,
                "mean_success": succ_val,
                "mean_failure": fail_val,
                "difference": diff_val,
                "pct_signal": (diff_val / total_abs_signal) * 100 if total_abs_signal else 0.0,
            }
        )

    output = {
        "analysis_file": str(Path(args.analysis_file)),
        "layer_index": data.get("layer_index"),
        "pooling": data.get("pooling"),
        "table_data": table_data,
        "top5_signal_pct": float(sum(item["pct_signal"] for item in table_data)),
        "total_dimensions": int(success_mean.shape[0]),
        "discriminative_dimensions": len(table_data),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("=" * 60)
    print("TABLE 4 DATA")
    print("=" * 60)
    for item in table_data:
        print(
            f"Dim {item['dimension']:4d}: "
            f"success={item['mean_success']:+.4f}, "
            f"failure={item['mean_failure']:+.4f}, "
            f"|delta|={item['difference']:.4f}"
        )
    print(f"Top-5 signal share: {output['top5_signal_pct']:.2f}%")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
