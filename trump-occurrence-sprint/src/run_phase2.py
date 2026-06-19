from __future__ import annotations

import pandas as pd

from src.baselines import compute_train_target_stats, fit_timing_model, predict_constant, predict_timing, save_pickle
from src.calibrate import apply_calibrator, fit_calibrator
from src.common import ROOT, ensure_dirs, load_config, read_json, read_jsonl, write_json
from src.evaluate import bootstrap_brier_ci, slice_metrics


def prediction_frame(labels: pd.DataFrame, p, model_name: str) -> pd.DataFrame:
    out = labels[["transcript_id", "target", "t_pct"]].copy()
    out["p_pred"] = p
    out["model_name"] = model_name
    return out


def main() -> None:
    config = load_config()
    ensure_dirs(ROOT / "models", ROOT / "predictions", ROOT / "metrics")
    corpus = read_jsonl(ROOT / "data" / "corpus.jsonl")
    splits = read_json(ROOT / "data" / "splits.json")
    targets = read_json(ROOT / "data" / "targets.json")["targets"]
    train_df = pd.read_parquet(ROOT / "benchmark" / "checkpoints_train.parquet")
    val_df = pd.read_parquet(ROOT / "benchmark" / "checkpoints_val.parquet")

    baseline_cfg = config["baselines"]
    stats = compute_train_target_stats(
        corpus,
        train_ids=set(splits["train"]["transcript_ids"]),
        targets=targets,
        alpha=float(baseline_cfg["smoothing_alpha"]),
    )

    y_val = val_df["label_occurs_after"].astype(int).to_numpy()

    p_const_raw = predict_constant(val_df, stats)
    const_cal = fit_calibrator(p_const_raw, y_val, method=baseline_cfg["calibration_method"])
    p_const = apply_calibrator(const_cal, p_const_raw)
    pred_const = prediction_frame(val_df, p_const, "constant")
    pred_const.to_parquet(ROOT / "predictions" / "val_constant.parquet", index=False)

    timing_artifact = fit_timing_model(
        train_df,
        stats,
        c=float(baseline_cfg["logistic_c"]),
        max_iter=int(baseline_cfg["max_iter"]),
    )
    p_timing_raw = predict_timing(val_df, stats, timing_artifact)
    timing_cal = fit_calibrator(p_timing_raw, y_val, method=baseline_cfg["calibration_method"])
    p_timing = apply_calibrator(timing_cal, p_timing_raw)
    pred_timing = prediction_frame(val_df, p_timing, "timing")
    pred_timing.to_parquet(ROOT / "predictions" / "val_timing.parquet", index=False)

    save_pickle(
        ROOT / "models" / "baselines.pkl",
        {
            "stats": stats,
            "constant_calibrator": const_cal,
            "timing_artifact": timing_artifact,
            "timing_calibrator": timing_cal,
            "config": baseline_cfg,
        },
    )

    reps = int(config["evaluation"]["bootstrap_reps_dev"])
    metrics = {
        "constant": slice_metrics(val_df, pred_const, bins=int(config["evaluation"]["ece_bins"])),
        "timing": slice_metrics(val_df, pred_timing, bins=int(config["evaluation"]["ece_bins"])),
        "bootstrap": {
            "constant": bootstrap_brier_ci(val_df, pred_const, reps=reps, seed=int(config["seed"])),
            "timing": bootstrap_brier_ci(
                val_df,
                pred_timing,
                reps=reps,
                seed=int(config["seed"]) + 1,
                baseline_preds=pred_const,
            ),
        },
    }
    b_const = metrics["constant"]["overall"]["brier"]
    b_timing = metrics["timing"]["overall"]["brier"]
    metrics["gate_2"] = {
        "baseline_b_beats_a": bool(b_timing < b_const),
        "constant_brier": b_const,
        "timing_brier": b_timing,
        "absolute_improvement": b_const - b_timing,
    }
    write_json(ROOT / "metrics" / "val_metrics.json", metrics)
    print(
        "Phase 2 complete: "
        f"constant_brier={b_const:.6f} timing_brier={b_timing:.6f} "
        f"gate_2={metrics['gate_2']['baseline_b_beats_a']}"
    )
    if not metrics["gate_2"]["baseline_b_beats_a"]:
        raise SystemExit("Gate 2 failed: timing baseline did not beat constant baseline on validation Brier.")


if __name__ == "__main__":
    main()
