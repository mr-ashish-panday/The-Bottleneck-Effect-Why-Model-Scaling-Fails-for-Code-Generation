from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from src.baselines import predict_timing
from src.calibrate import apply_calibrator, fit_calibrator
from src.llm_features import FEATURE_COLUMNS, build_content_features, fit_feature_artifact, save_pickle


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_offset_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    l2: float = 1.0,
    nonnegative_indices: set[int] | None = None,
) -> dict[str, Any]:
    y = y.astype(float)
    n_features = X.shape[1]
    nonnegative_indices = nonnegative_indices or set()

    def objective(theta):
        intercept = theta[0]
        beta = theta[1:]
        z = offset + intercept + X @ beta
        p = sigmoid(z)
        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += 0.5 * l2 * float(beta @ beta) / max(1, len(y))
        grad_common = (p - y) / max(1, len(y))
        grad_intercept = np.sum(grad_common)
        grad_beta = X.T @ grad_common + (l2 / max(1, len(y))) * beta
        grad = np.concatenate([[grad_intercept], grad_beta])
        return loss, grad

    result = minimize(
        lambda th: objective(th)[0],
        np.zeros(n_features + 1, dtype=float),
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None) if i in nonnegative_indices else (None, None) for i in range(n_features)],
        options={"maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Offset logistic failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "beta": result.x[1:].astype(float),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "l2": float(l2),
    }


def duration_adjustment(X: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    gamma = float(artifact.get("fixed_duration_gamma", 0.0) or 0.0)
    idx = artifact.get("duration_feature_index")
    if gamma == 0.0 or idx is None:
        return np.zeros(X.shape[0], dtype=float)
    return gamma * X[:, int(idx)]


def predict_offset_logistic(X: np.ndarray, offset: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    return sigmoid(offset + artifact["intercept"] + X @ artifact["beta"] + duration_adjustment(X, artifact))


def fit_challenger(
    corpus: list[dict[str, Any]],
    splits: dict[str, Any],
    targets: list[dict[str, Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    baseline_artifacts: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    feature_artifact = fit_feature_artifact(corpus, splits, targets, config)
    selected_features = config["challenger"].get("selected_features") or FEATURE_COLUMNS
    nonnegative_names = set(config["challenger"].get("nonnegative_features") or [])
    nonnegative_indices = {i for i, name in enumerate(selected_features) if name in nonnegative_names}
    duration_feature_index = (
        selected_features.index("log_expected_remaining_words")
        if "log_expected_remaining_words" in selected_features
        else None
    )
    X_train_df = build_content_features(train_df, corpus, targets, feature_artifact, blank_prefix=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df[selected_features].to_numpy(dtype=float))
    y_train = train_df["label_occurs_after"].astype(int).to_numpy()
    p_base_train = predict_timing(train_df, baseline_artifacts["stats"], baseline_artifacts["timing_artifact"])
    offset_train = logit(p_base_train)
    offset_model = fit_offset_logistic(
        X_train,
        y_train,
        offset_train,
        l2=float(config["challenger"]["l2"]),
        nonnegative_indices=nonnegative_indices,
    )
    offset_model["fixed_duration_gamma"] = float(config["challenger"].get("fixed_duration_gamma", 0.0) or 0.0)
    offset_model["duration_feature_index"] = duration_feature_index

    X_val_df = build_content_features(val_df, corpus, targets, feature_artifact, blank_prefix=False)
    X_val = scaler.transform(X_val_df[selected_features].to_numpy(dtype=float))
    p_base_val = predict_timing(val_df, baseline_artifacts["stats"], baseline_artifacts["timing_artifact"])
    p_val_raw = predict_offset_logistic(X_val, logit(p_base_val), offset_model)
    calibrator = fit_calibrator(
        p_val_raw,
        val_df["label_occurs_after"].astype(int).to_numpy(),
        method=config["baselines"]["calibration_method"],
    )

    return {
        "feature_artifact": feature_artifact,
        "feature_scaler": scaler,
        "offset_model": offset_model,
        "calibrator": calibrator,
        "feature_columns": selected_features,
        "nonnegative_features": sorted(nonnegative_names),
        "available_feature_columns": FEATURE_COLUMNS,
        "train_feature_means": dict(zip(FEATURE_COLUMNS, X_train_df.mean().astype(float))),
    }


def predict_challenger(
    df: pd.DataFrame,
    corpus: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    baseline_artifacts: dict[str, Any],
    challenger_artifact: dict[str, Any],
    blank_prefix: bool = False,
    calibrate: bool = True,
) -> np.ndarray:
    Xdf = build_content_features(
        df,
        corpus,
        targets,
        challenger_artifact["feature_artifact"],
        blank_prefix=blank_prefix,
    )
    selected_features = challenger_artifact.get("feature_columns", FEATURE_COLUMNS)
    X = challenger_artifact["feature_scaler"].transform(Xdf[selected_features].to_numpy(dtype=float))
    p_base = predict_timing(df, baseline_artifacts["stats"], baseline_artifacts["timing_artifact"])
    raw = predict_offset_logistic(X, logit(p_base), challenger_artifact["offset_model"])
    if calibrate:
        return apply_calibrator(challenger_artifact["calibrator"], raw)
    return raw
