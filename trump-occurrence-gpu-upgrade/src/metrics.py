from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        n = int(mask.sum())
        if n == 0:
            table.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": 0})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(conf - acc)
        ece += (n / len(y)) * gap
        table.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": n, "mean_pred": conf, "empirical": acc, "gap": gap})
    return float(ece), table


def metric_block(labels: pd.DataFrame, preds: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    merged = labels[["transcript_id", "target", "t_pct", "label_occurs_after"]].merge(
        preds[["transcript_id", "target", "t_pct", "p_pred"]],
        on=["transcript_id", "target", "t_pct"],
        how="inner",
    )
    y = merged["label_occurs_after"].astype(int).to_numpy()
    p = np.clip(merged["p_pred"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    ece, reliability = expected_calibration_error(y, p, bins=bins)
    out = {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece,
        "reliability": reliability,
    }
    try:
        out["auc_diagnostic"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc_diagnostic"] = None
    try:
        X = logit(p).reshape(-1, 1)
        cal = LogisticRegression(solver="lbfgs").fit(X, y)
        out["calibration_intercept"] = float(cal.intercept_[0])
        out["calibration_slope"] = float(cal.coef_[0][0])
    except Exception:
        out["calibration_intercept"] = None
        out["calibration_slope"] = None
    return out


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str = "isotonic") -> Any:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if method == "platt":
        model = LogisticRegression(solver="lbfgs").fit(logit(p).reshape(-1, 1), y)
        return {"method": "platt", "model": model}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return {"method": "isotonic", "model": iso}


def apply_calibrator(calibrator: Any, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if calibrator["method"] == "platt":
        return np.clip(calibrator["model"].predict_proba(logit(p).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(calibrator["model"].predict(p), 1e-6, 1 - 1e-6)

