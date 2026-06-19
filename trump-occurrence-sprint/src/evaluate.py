from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


KEYS = ["transcript_id", "target", "t_pct"]


def clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), 1e-6, 1 - 1e-6)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        if mask.any():
            rows.append(
                {
                    "bin": i,
                    "p_low": float(low),
                    "p_high": float(high),
                    "n": int(mask.sum()),
                    "mean_pred": float(p[mask].mean()),
                    "empirical_rate": float(y[mask].mean()),
                    "abs_gap": float(abs(p[mask].mean() - y[mask].mean())),
                }
            )
        else:
            rows.append(
                {
                    "bin": i,
                    "p_low": float(low),
                    "p_high": float(high),
                    "n": 0,
                    "mean_pred": None,
                    "empirical_rate": None,
                    "abs_gap": None,
                }
            )
    return rows


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = reliability_table(y, p, bins)
    total = len(y)
    return float(
        sum((row["n"] / total) * row["abs_gap"] for row in table if row["n"] and row["abs_gap"] is not None)
    )


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": None, "calibration_slope": None}
    logits = np.log(clip_probs(p) / (1 - clip_probs(p))).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def metric_block(labels: pd.DataFrame, preds: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    merged = labels[KEYS + ["label_occurs_after"]].merge(preds[KEYS + ["p_pred", "model_name"]], on=KEYS, how="inner")
    if len(merged) != len(preds):
        raise ValueError(f"Prediction/label merge mismatch: labels={len(labels)} preds={len(preds)} merged={len(merged)}")
    y = merged["label_occurs_after"].astype(int).to_numpy()
    p = clip_probs(merged["p_pred"].to_numpy())
    brier = float(np.mean((p - y) ** 2))
    out: dict[str, Any] = {
        "n_rows": int(len(merged)),
        "n_transcripts": int(merged["transcript_id"].nunique()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "brier": brier,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "ece": ece_score(y, p, bins) if len(y) else None,
        "reliability": reliability_table(y, p, bins),
    }
    out.update(calibration_slope_intercept(y, p))
    try:
        out["auc_diagnostic"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    except ValueError:
        out["auc_diagnostic"] = None
    return out


def slice_metrics(labels: pd.DataFrame, preds: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    merged = labels.merge(preds[KEYS + ["p_pred", "model_name"]], on=KEYS, how="inner")
    out: dict[str, Any] = {"overall": metric_block(labels, preds, bins)}
    out["by_t_pct"] = {}
    for t_pct, part in merged.groupby("t_pct"):
        out["by_t_pct"][str(t_pct)] = metric_block(part, part[KEYS + ["p_pred", "model_name"]], bins)
    out["by_target_band"] = {}
    for band, part in merged.groupby("target_band"):
        out["by_target_band"][str(band)] = metric_block(part, part[KEYS + ["p_pred", "model_name"]], bins)
    out["t_zero_vs_after"] = {}
    for name, part in (("t0", merged[merged["t_pct"] == 0]), ("t_gt_0", merged[merged["t_pct"] > 0])):
        if len(part):
            out["t_zero_vs_after"][name] = metric_block(part, part[KEYS + ["p_pred", "model_name"]], bins)
    return out


def bootstrap_brier_ci(
    labels: pd.DataFrame,
    preds: pd.DataFrame,
    reps: int,
    seed: int,
    baseline_preds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    merged = labels[KEYS + ["label_occurs_after"]].merge(preds[KEYS + ["p_pred"]], on=KEYS, how="inner")
    merged = merged.rename(columns={"p_pred": "p_pred_model"})
    if baseline_preds is not None:
        merged = merged.merge(
            baseline_preds[KEYS + ["p_pred"]].rename(columns={"p_pred": "p_pred_baseline"}),
            on=KEYS,
            how="inner",
        )
    transcript_ids = merged["transcript_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    deltas = []
    for _ in range(reps):
        sample_ids = rng.choice(transcript_ids, size=len(transcript_ids), replace=True)
        sample = pd.concat([merged[merged["transcript_id"] == tid] for tid in sample_ids], ignore_index=True)
        y = sample["label_occurs_after"].astype(int).to_numpy()
        p = clip_probs(sample["p_pred_model"].to_numpy())
        brier = float(np.mean((p - y) ** 2))
        values.append(brier)
        if baseline_preds is not None:
            p_base = clip_probs(sample["p_pred_baseline"].to_numpy())
            deltas.append(float(np.mean((p - y) ** 2) - np.mean((p_base - y) ** 2)))
    result: dict[str, Any] = {
        "reps": reps,
        "brier_ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }
    if baseline_preds is not None:
        result["model_minus_baseline_brier_delta_ci95"] = [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ]
    return result
