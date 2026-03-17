#!/usr/bin/env python3
"""
Create Figure 4 from real activation samples rather than simulated points.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_file",
        default="data/results_gpt2_medium/ablation/layer12_analysis.json",
    )
    parser.add_argument("--samples_file", default=None)
    parser.add_argument(
        "--classification_file",
        default="data/results_gpt2_medium/ablation/activation_classification.json",
    )
    parser.add_argument("--positive_category", default="success")
    parser.add_argument("--negative_category", default="syntax_error")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--output_file",
        default="outputs/figures/figure4_activation_projection.png",
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


def get_activation_vector(record):
    if "activation" in record:
        return np.asarray(record["activation"], dtype=np.float32)
    activations = record.get("activations")
    if activations and len(activations) == 1:
        return np.asarray(next(iter(activations.values())), dtype=np.float32)
    raise ValueError("Sample record does not contain a single activation vector")


def prepare_points(sample_records, positive_category, negative_category, dimensions):
    xs = []
    ys = []

    for record in sample_records:
        category = record.get("category")
        if category == positive_category:
            label = 1
        elif category == negative_category:
            label = 0
        else:
            continue

        activation = get_activation_vector(record)
        xs.append(activation[dimensions])
        ys.append(label)

    if not xs:
        raise ValueError("No samples matched the requested categories")

    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def build_plot_statistics(points, labels, positive_category, negative_category, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        points,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    held_out_model = LogisticRegression(max_iter=5000, random_state=random_state)
    held_out_model.fit(x_train, y_train)
    held_out_accuracy = float(accuracy_score(y_test, held_out_model.predict(x_test)))

    plot_model = LogisticRegression(max_iter=5000, random_state=random_state)
    plot_model.fit(points, labels)

    positive_points = points[labels == 1]
    negative_points = points[labels == 0]
    positive_mean = positive_points.mean(axis=0)
    negative_mean = negative_points.mean(axis=0)
    separation = float(np.linalg.norm(positive_mean - negative_mean))

    return {
        "held_out_accuracy": held_out_accuracy,
        "plot_model": plot_model,
        "positive_points": positive_points,
        "negative_points": negative_points,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "class_counts": {
            positive_category: int(len(positive_points)),
            negative_category: int(len(negative_points)),
        },
        "separation": separation,
    }


def draw_decision_boundary(ax, classifier, points):
    coef = classifier.coef_[0]
    intercept = classifier.intercept_[0]

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    padding_x = max(0.5, 0.05 * (x_max - x_min))
    padding_y = max(0.5, 0.05 * (y_max - y_min))

    x_values = np.linspace(x_min - padding_x, x_max + padding_x, 200)
    if abs(coef[1]) < 1e-8:
        x_boundary = np.full(200, -intercept / coef[0])
        y_values = np.linspace(y_min - padding_y, y_max + padding_y, 200)
        ax.plot(x_boundary, y_values, "--", color="black", linewidth=2, label="Linear boundary")
    else:
        y_values = -(coef[0] * x_values + intercept) / coef[1]
        ax.plot(x_values, y_values, "--", color="black", linewidth=2, label="Linear boundary")

    ax.set_xlim(x_min - padding_x, x_max + padding_x)
    ax.set_ylim(y_min - padding_y, y_max + padding_y)


def main():
    args = parse_args()

    analysis_path = Path(args.analysis_file)
    analysis_data = load_json(analysis_path)
    samples_path = resolve_samples_path(analysis_path, analysis_data, args.samples_file)
    sample_records = load_json(samples_path)

    top2_dims = [int(dim) for dim in analysis_data["top_discriminative_dims"][:2]]
    points, labels = prepare_points(
        sample_records=sample_records,
        positive_category=args.positive_category,
        negative_category=args.negative_category,
        dimensions=top2_dims,
    )

    stats = build_plot_statistics(
        points=points,
        labels=labels,
        positive_category=args.positive_category,
        negative_category=args.negative_category,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    classification_path = Path(args.classification_file)
    classification_data = None
    if classification_path.exists():
        classification_data = load_json(classification_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        stats["positive_points"][:, 0],
        stats["positive_points"][:, 1],
        c="#2ca02c",
        alpha=0.45,
        s=28,
        label=args.positive_category.replace("_", " ").title(),
        edgecolors="none",
    )
    ax.scatter(
        stats["negative_points"][:, 0],
        stats["negative_points"][:, 1],
        c="#d62728",
        alpha=0.45,
        s=28,
        label=args.negative_category.replace("_", " ").title(),
        edgecolors="none",
    )

    ax.scatter(
        [stats["positive_mean"][0]],
        [stats["positive_mean"][1]],
        c="#1b6e1b",
        marker="*",
        s=450,
        edgecolors="black",
        linewidths=1.2,
        label=f"{args.positive_category.replace('_', ' ').title()} mean",
        zorder=5,
    )
    ax.scatter(
        [stats["negative_mean"][0]],
        [stats["negative_mean"][1]],
        c="#8b1e1e",
        marker="*",
        s=450,
        edgecolors="black",
        linewidths=1.2,
        label=f"{args.negative_category.replace('_', ' ').title()} mean",
        zorder=5,
    )

    draw_decision_boundary(ax, stats["plot_model"], points)

    top2_accuracy = stats["held_out_accuracy"]
    top5_accuracy = None
    if classification_data:
        top2_accuracy = classification_data["metrics"]["top2"]["accuracy"]
        top5_accuracy = classification_data["metrics"]["top5"]["accuracy"]

    annotation_lines = [f"Top-2 held-out accuracy: {top2_accuracy:.1%}"]
    if top5_accuracy is not None:
        annotation_lines.append(f"Top-5 held-out accuracy: {top5_accuracy:.1%}")
    annotation_lines.append(f"2D class-mean separation: {stats['separation']:.2f}")

    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax.set_xlabel(f"Activation dimension {top2_dims[0]}", fontsize=13)
    ax.set_ylabel(f"Activation dimension {top2_dims[1]}", fontsize=13)
    ax.set_title(
        "GPT-2 Medium target-layer activation space\n"
        "Real success vs. syntax-error samples",
        fontsize=15,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.95)

    plt.tight_layout()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    stats_output = output_path.with_name("figure4_activation_projection_stats.json")
    stats_payload = {
        "analysis_file": str(analysis_path),
        "samples_file": str(samples_path),
        "classification_file": str(classification_path),
        "dimensions": top2_dims,
        "held_out_accuracy": top2_accuracy,
        "top5_accuracy": top5_accuracy,
        "class_counts": stats["class_counts"],
        "class_mean_separation": stats["separation"],
    }
    with open(stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)

    print("=" * 60)
    print("FIGURE 4 CREATED")
    print("=" * 60)
    print(f"Dimensions: {top2_dims}")
    print(f"Top-2 held-out accuracy: {top2_accuracy:.4f}")
    if top5_accuracy is not None:
        print(f"Top-5 held-out accuracy: {top5_accuracy:.4f}")
    print(f"Saved figure to: {output_path}")
    print(f"Saved stats to: {stats_output}")


if __name__ == "__main__":
    main()
