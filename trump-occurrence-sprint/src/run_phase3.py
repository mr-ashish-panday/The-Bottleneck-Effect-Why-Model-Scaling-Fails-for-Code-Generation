from __future__ import annotations

import time

import pandas as pd

from src.baselines import load_pickle, save_pickle
from src.challenger import fit_challenger, predict_challenger
from src.common import ROOT, ensure_dirs, load_config, read_json, read_jsonl, write_json
from src.contamination import contamination_report
from src.duration_sensitivity import duration_sensitivity_report
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
    baseline_artifacts = load_pickle(ROOT / "models" / "baselines.pkl")

    start = time.perf_counter()
    challenger_artifact = fit_challenger(
        corpus,
        splits,
        targets,
        train_df,
        val_df,
        baseline_artifacts,
        config,
    )
    fit_seconds = time.perf_counter() - start
    save_pickle(ROOT / "models" / "challenger.pkl", challenger_artifact)

    pred_start = time.perf_counter()
    p_full = predict_challenger(val_df, corpus, targets, baseline_artifacts, challenger_artifact, blank_prefix=False)
    pred_seconds = time.perf_counter() - pred_start
    p_blanked = predict_challenger(val_df, corpus, targets, baseline_artifacts, challenger_artifact, blank_prefix=True)

    pred_full = prediction_frame(val_df, p_full, "challenger_full")
    pred_blanked = prediction_frame(val_df, p_blanked, "challenger_blanked")
    pred_full.to_parquet(ROOT / "predictions" / "val_challenger.parquet", index=False)
    pred_blanked.to_parquet(ROOT / "predictions" / "val_challenger_blanked.parquet", index=False)

    pred_timing = pd.read_parquet(ROOT / "predictions" / "val_timing.parquet")
    val_metrics = read_json(ROOT / "metrics" / "val_metrics.json")
    val_metrics["challenger_full"] = slice_metrics(val_df, pred_full, bins=int(config["evaluation"]["ece_bins"]))
    val_metrics["challenger_blanked"] = slice_metrics(val_df, pred_blanked, bins=int(config["evaluation"]["ece_bins"]))
    val_metrics["bootstrap"]["challenger_full"] = bootstrap_brier_ci(
        val_df,
        pred_full,
        reps=int(config["evaluation"]["bootstrap_reps_dev"]),
        seed=int(config["seed"]) + 2,
        baseline_preds=pred_timing,
    )
    b_timing = val_metrics["timing"]["overall"]["brier"]
    b_ch = val_metrics["challenger_full"]["overall"]["brier"]
    val_metrics["gate_3_validation"] = {
        "challenger_beats_timing": bool(b_ch < b_timing),
        "timing_brier": b_timing,
        "challenger_brier": b_ch,
        "absolute_improvement": b_timing - b_ch,
    }
    write_json(ROOT / "metrics" / "val_metrics.json", val_metrics)

    imp_full = b_timing - b_ch
    b_blank = val_metrics["challenger_blanked"]["overall"]["brier"]
    imp_blank = b_timing - b_blank
    content_share = None if imp_full == 0 else (imp_full - imp_blank) / imp_full
    write_json(
        ROOT / "metrics" / "ablation.json",
        {
            "validation": {
                "timing_brier": b_timing,
                "challenger_full_brier": b_ch,
                "challenger_blanked_brier": b_blank,
                "imp_full": imp_full,
                "imp_blank": imp_blank,
                "content_share": content_share,
            }
        },
    )

    duration = duration_sensitivity_report(
        val_df,
        corpus,
        targets,
        baseline_artifacts,
        challenger_artifact,
        sample_rows=int(config["duration_sensitivity"]["sample_rows"]),
        multipliers=[float(x) for x in config["duration_sensitivity"]["multipliers"]],
        seed=int(config["seed"]),
    )
    write_json(ROOT / "metrics" / "duration_sensitivity.json", duration)

    write_json(
        ROOT / "metrics" / "cost_latency.json",
        {
            "api_dollars": 0.0,
            "feature_kind": challenger_artifact["feature_artifact"]["semantic"]["kind"],
            "train_fit_seconds": fit_seconds,
            "validation_prediction_seconds": pred_seconds,
            "validation_rows": int(len(val_df)),
            "seconds_per_validation_prediction": pred_seconds / max(1, len(val_df)),
            "total_calls": 0,
            "tokens_per_prediction": None,
            "window_words_per_prediction": {
                "trailing_window_words": int(config["challenger"]["trailing_window_words"]),
                "full_prefix_window_words": int(config["challenger"]["full_prefix_window_words"]),
            },
            "plausible_at_benchmark_scale": True,
        },
    )

    contamination = contamination_report(corpus, splits["val"]["transcript_ids"], config)
    write_json(ROOT / "metrics" / "contamination_report.json", contamination)

    print(
        "Phase 3 challenger complete: "
        f"timing_brier={b_timing:.6f} challenger_brier={b_ch:.6f} "
        f"gate_3={b_ch < b_timing} duration_monotonic={duration['monotonic_fraction']} "
        f"contamination_method={contamination['method']}"
    )
    if not (b_ch < b_timing):
        raise SystemExit("Gate 3 failed: challenger did not beat timing baseline on validation Brier.")


if __name__ == "__main__":
    main()
