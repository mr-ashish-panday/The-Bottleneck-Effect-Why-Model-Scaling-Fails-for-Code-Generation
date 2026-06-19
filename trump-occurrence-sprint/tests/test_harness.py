import pandas as pd

from src.evaluate import bootstrap_brier_ci, metric_block, slice_metrics


def test_metric_harness_dummy_constant():
    labels = pd.DataFrame(
        {
            "transcript_id": ["a", "a", "b", "b"],
            "target": ["x", "y", "x", "y"],
            "t_pct": [0, 10, 0, 10],
            "target_band": ["high", "rare", "high", "rare"],
            "label_occurs_after": [1, 0, 1, 0],
        }
    )
    preds = labels[["transcript_id", "target", "t_pct"]].copy()
    preds["p_pred"] = 0.5
    preds["model_name"] = "dummy"
    metrics = metric_block(labels, preds, bins=10)
    assert metrics["brier"] == 0.25
    assert metrics["ece"] is not None
    slices = slice_metrics(labels, preds, bins=10)
    assert "overall" in slices
    ci = bootstrap_brier_ci(labels, preds, reps=10, seed=1)
    assert "brier_ci95" in ci
