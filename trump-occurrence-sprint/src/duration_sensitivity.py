from __future__ import annotations

import numpy as np
import pandas as pd

from src.challenger import predict_challenger


def duration_sensitivity_report(
    val_df: pd.DataFrame,
    corpus,
    targets,
    baseline_artifacts,
    challenger_artifact,
    sample_rows: int,
    multipliers: list[float],
    seed: int,
) -> dict:
    sample = val_df.sample(n=min(sample_rows, len(val_df)), random_state=seed).copy()
    preds_by_mult = {}
    base_remaining = sample["expected_remaining_words"].astype(float).to_numpy()
    for mult in multipliers:
        mod = sample.copy()
        mod["expected_remaining_words"] = np.maximum(0, np.round(base_remaining * mult)).astype(int)
        preds_by_mult[str(mult)] = predict_challenger(
            mod, corpus, targets, baseline_artifacts, challenger_artifact, blank_prefix=False, calibrate=True
        )
    ordered = [preds_by_mult[str(m)] for m in multipliers]
    monotonic = []
    sensitivities = []
    for vals in zip(*ordered):
        vals = list(vals)
        monotonic.append(all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)))
        sensitivities.append(vals[-1] - vals[0])
    return {
        "sample_rows": int(len(sample)),
        "multipliers": multipliers,
        "monotonic_fraction": float(np.mean(monotonic)) if monotonic else None,
        "mean_sensitivity_high_minus_low": float(np.mean(sensitivities)) if sensitivities else None,
        "median_sensitivity_high_minus_low": float(np.median(sensitivities)) if sensitivities else None,
    }
