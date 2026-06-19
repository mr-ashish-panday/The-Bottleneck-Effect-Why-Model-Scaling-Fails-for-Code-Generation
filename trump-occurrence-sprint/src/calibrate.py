from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


@dataclass
class IdentityCalibrator:
    constant: float | None = None

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.constant is not None:
            return np.full_like(p, self.constant, dtype=float)
        return np.clip(p, 1e-6, 1 - 1e-6)


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str = "isotonic"):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return IdentityCalibrator(float(y.mean()) if len(y) else 0.5)
    if method == "isotonic" and len(np.unique(p)) >= 3:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return iso
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model.fit(logits, y)
    return PlattCalibrator(model)


def apply_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    if hasattr(calibrator, "predict"):
        out = calibrator.predict(np.asarray(p, dtype=float))
    else:
        out = calibrator.transform(np.asarray(p, dtype=float))
    return np.clip(np.asarray(out, dtype=float), 1e-6, 1 - 1e-6)
