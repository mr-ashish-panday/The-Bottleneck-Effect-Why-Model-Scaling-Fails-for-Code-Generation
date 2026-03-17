#!/usr/bin/env python3
"""
Train real linear probes on saved activation samples.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_file",
        default="data/results_gpt2_medium/ablation/layer12_analysis.json",
    )
    parser.add_argument("--samples_file", default=None)
    parser.add_argument(
        "--layer_key",
        default=None,
        help="Layer key for activations.json-style inputs, e.g. layer_11.",
    )
    parser.add_argument("--positive_category", default="success")
    parser.add_argument("--negative_category", default="syntax_error")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--output_file",
        default="data/results_gpt2_medium/ablation/activation_classification.json",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_samples_path(analysis_path, analysis_data, explicit_path):
    if explicit_path:
        return Path(explicit_path)

    sample_path = analysis_data.get("sample_activations_file")
    if sample_path:
        return Path(sample_path)

    return analysis_path.parent / "layer12_probe_samples.json"


def get_activation_vector(record, layer_key=None):
    if "activation" in record:
        return np.asarray(record["activation"], dtype=np.float32)

    activations = record.get("activations")
    if not activations:
        raise ValueError("Activation record is missing both 'activation' and 'activations'")

    if layer_key:
        if layer_key not in activations:
            raise KeyError(f"Layer key '{layer_key}' not found in activation record")
        return np.asarray(activations[layer_key], dtype=np.float32)

    if len(activations) == 1:
        return np.asarray(next(iter(activations.values())), dtype=np.float32)

    raise ValueError("Multiple activation layers available. Provide --layer_key.")


def prepare_dataset(records, positive_category, negative_category, layer_key):
    features = []
    labels = []

    for record in records:
        category = record.get("category")
        if category == positive_category:
            label = 1
        elif category == negative_category:
            label = 0
        else:
            continue

        features.append(get_activation_vector(record, layer_key=layer_key))
        labels.append(label)

    if not features:
        raise ValueError("No activation records matched the requested categories")

    return np.stack(features), np.asarray(labels, dtype=np.int32)


def fit_probe(features, labels, dimensions, test_size, random_state):
    if dimensions is None:
        probe_features = features
        selected_dimensions = list(range(features.shape[1]))
    else:
        selected_dimensions = [int(dim) for dim in dimensions]
        probe_features = features[:, selected_dimensions]

    x_train, x_test, y_train, y_test = train_test_split(
        probe_features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    classifier = LogisticRegression(max_iter=5000, random_state=random_state)
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    return {
        "dimensions": selected_dimensions,
        "num_dimensions": len(selected_dimensions),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    args = parse_args()

    analysis_path = Path(args.analysis_file)
    analysis_data = load_json(analysis_path)
    samples_path = resolve_samples_path(analysis_path, analysis_data, args.samples_file)
    sample_records = load_json(samples_path)

    features, labels = prepare_dataset(
        records=sample_records,
        positive_category=args.positive_category,
        negative_category=args.negative_category,
        layer_key=args.layer_key,
    )

    top_dimensions = [int(dim) for dim in analysis_data["top_discriminative_dims"]]
    top2 = top_dimensions[:2]
    top5 = top_dimensions[:5]

    metrics = {
        "top2": fit_probe(
            features,
            labels,
            dimensions=top2,
            test_size=args.test_size,
            random_state=args.random_state,
        ),
        "top5": fit_probe(
            features,
            labels,
            dimensions=top5,
            test_size=args.test_size,
            random_state=args.random_state,
        ),
        "full_layer": fit_probe(
            features,
            labels,
            dimensions=None,
            test_size=args.test_size,
            random_state=args.random_state,
        ),
    }

    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    results = {
        "analysis_file": str(analysis_path),
        "samples_file": str(samples_path),
        "layer_index": analysis_data.get("layer_index"),
        "layer_label": analysis_data.get("layer_label"),
        "pooling": analysis_data.get("pooling"),
        "positive_category": args.positive_category,
        "negative_category": args.negative_category,
        "total_samples": int(len(labels)),
        "class_counts": {
            args.positive_category: positives,
            args.negative_category: negatives,
        },
        "top_dimensions": top_dimensions[:5],
        "metrics": metrics,
        "headline_result": {
            "probe": "top5",
            "accuracy": metrics["top5"]["accuracy"],
        },
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("=" * 60)
    print("ACTIVATION CLASSIFIER RESULTS")
    print("=" * 60)
    print(f"Samples file: {samples_path}")
    print(
        f"Classes: {args.positive_category}={positives}, "
        f"{args.negative_category}={negatives}"
    )
    print(f"Top 2 dims {top2}:  {metrics['top2']['accuracy']:.4f} accuracy")
    print(f"Top 5 dims {top5}:  {metrics['top5']['accuracy']:.4f} accuracy")
    print(f"Full layer probe:   {metrics['full_layer']['accuracy']:.4f} accuracy")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
