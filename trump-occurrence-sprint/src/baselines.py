from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.matching import first_occurrence, occurrences, tokenize


def save_pickle(path: str | Path, obj: Any) -> None:
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def compute_train_target_stats(
    corpus: list[dict[str, Any]],
    train_ids: set[str],
    targets: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    train_rows = [row for row in corpus if row["transcript_id"] in train_ids]
    total_words = sum(int(row["n_words"]) for row in train_rows)
    format_counts = Counter(row["format"] for row in train_rows)
    target_stats: dict[str, dict[str, float]] = {}
    format_rates: dict[tuple[str, str], float] = {}
    global_occurrences = 0
    global_trials = len(train_rows) * len(targets)

    token_cache = {row["transcript_id"]: tokenize(row["text"]) for row in train_rows}

    for target in targets:
        name = target["target"]
        doc_occ = 0
        total_count = 0
        first_pcts: list[float] = []
        early = 0
        by_format_occ: Counter[str] = Counter()
        for row in train_rows:
            toks = token_cache[row["transcript_id"]]
            hits = occurrences(name, toks)
            total_count += len(hits)
            if hits:
                doc_occ += 1
                by_format_occ[row["format"]] += 1
                first_pct = hits[0] / max(1, len(toks))
                first_pcts.append(first_pct)
                if first_pct <= 0.25:
                    early += 1
        global_occurrences += doc_occ
        doc_rate_unsmoothed = doc_occ / max(1, len(train_rows))
        target_stats[name] = {
            "target_doc_occ": float(doc_occ),
            "target_doc_rate": float(doc_rate_unsmoothed),
            "target_total_count": float(total_count),
            "count_per_1k": float(1000.0 * total_count / max(1, total_words)),
            "mean_first_pct": float(np.mean(first_pcts)) if first_pcts else 1.0,
            "median_first_pct": float(np.median(first_pcts)) if first_pcts else 1.0,
            "early_rate": float(early / max(1, doc_occ)),
        }
        target_rate = (doc_occ + alpha * 0.5) / (len(train_rows) + alpha)
        for fmt, n_fmt in format_counts.items():
            fmt_occ = by_format_occ[fmt]
            format_rates[(name, fmt)] = float((fmt_occ + alpha * target_rate) / (n_fmt + alpha))

    global_rate = global_occurrences / max(1, global_trials)
    for target in targets:
        name = target["target"]
        doc_occ = target_stats[name]["target_doc_occ"]
        target_stats[name]["target_rate_smoothed"] = float((doc_occ + alpha * global_rate) / (len(train_rows) + alpha))

    return {
        "target_stats": target_stats,
        "format_rates": format_rates,
        "global_rate": float(global_rate),
        "format_counts": dict(format_counts),
        "n_train": len(train_rows),
        "alpha": alpha,
    }


def predict_constant(df: pd.DataFrame, stats: dict[str, Any]) -> np.ndarray:
    out = []
    for row in df.itertuples(index=False):
        target = getattr(row, "target")
        fmt = getattr(row, "format")
        p = stats["format_rates"].get((target, fmt))
        if p is None:
            p = stats["target_stats"].get(target, {}).get("target_rate_smoothed", stats["global_rate"])
        out.append(p)
    return np.clip(np.asarray(out, dtype=float), 1e-6, 1 - 1e-6)


NUMERIC_FEATURES = [
    "target_doc_rate",
    "target_rate_smoothed",
    "count_per_1k",
    "mean_first_pct",
    "median_first_pct",
    "early_rate",
    "t_pct",
    "elapsed_words",
    "expected_remaining_words",
    "log_elapsed_words",
    "log_expected_remaining_words",
]


def build_timing_matrix(
    df: pd.DataFrame,
    stats: dict[str, Any],
    columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, list[str], StandardScaler]:
    base = pd.DataFrame(index=df.index)
    for feature in NUMERIC_FEATURES:
        if feature.startswith("target_") or feature in {"count_per_1k", "mean_first_pct", "median_first_pct", "early_rate"}:
            base[feature] = [stats["target_stats"].get(t, {}).get(feature, 0.0) for t in df["target"]]
    base["t_pct"] = df["t_pct"].astype(float) / 100.0
    base["elapsed_words"] = df["elapsed_words"].astype(float)
    base["expected_remaining_words"] = df["expected_remaining_words"].astype(float)
    base["log_elapsed_words"] = np.log1p(df["elapsed_words"].astype(float))
    base["log_expected_remaining_words"] = np.log1p(df["expected_remaining_words"].astype(float))
    cats = pd.get_dummies(df[["format", "target_band"]].astype(str), prefix=["format", "band"], dtype=float)
    Xdf = pd.concat([base[NUMERIC_FEATURES], cats], axis=1)
    if columns is None:
        columns = list(Xdf.columns)
    Xdf = Xdf.reindex(columns=columns, fill_value=0.0)
    if fit or scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(Xdf.to_numpy(dtype=float))
    else:
        X = scaler.transform(Xdf.to_numpy(dtype=float))
    return X, columns, scaler


def fit_timing_model(train_df: pd.DataFrame, stats: dict[str, Any], c: float, max_iter: int) -> dict[str, Any]:
    X, columns, scaler = build_timing_matrix(train_df, stats, fit=True)
    y = train_df["label_occurs_after"].astype(int).to_numpy()
    model = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
    model.fit(X, y)
    return {"model": model, "columns": columns, "scaler": scaler}


def predict_timing(df: pd.DataFrame, stats: dict[str, Any], artifact: dict[str, Any]) -> np.ndarray:
    X, _, _ = build_timing_matrix(df, stats, columns=artifact["columns"], scaler=artifact["scaler"], fit=False)
    return np.clip(artifact["model"].predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
