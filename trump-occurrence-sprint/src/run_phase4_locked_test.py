from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.baselines import load_pickle, predict_constant, predict_timing
from src.calibrate import apply_calibrator
from src.challenger import predict_challenger
from src.common import ROOT, ensure_dirs, load_config, read_json, read_jsonl, sha256_file, write_json
from src.contamination import contamination_report
from src.evaluate import bootstrap_brier_ci, slice_metrics


def prediction_frame(labels: pd.DataFrame, p, model_name: str) -> pd.DataFrame:
    out = labels[["transcript_id", "target", "t_pct"]].copy()
    out["p_pred"] = p
    out["model_name"] = model_name
    return out


def hash_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def main() -> None:
    config = load_config()
    ensure_dirs(ROOT / "metrics", ROOT / "predictions")

    freeze_manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "single_locked_test_pass",
        "config_sha256": hash_if_exists(ROOT / "config.yaml"),
        "corpus_sha256": hash_if_exists(ROOT / "data" / "corpus.jsonl"),
        "splits_sha256": hash_if_exists(ROOT / "data" / "splits.json"),
        "targets_sha256": hash_if_exists(ROOT / "data" / "targets.json"),
        "baseline_model_sha256": hash_if_exists(ROOT / "models" / "baselines.pkl"),
        "challenger_model_sha256": hash_if_exists(ROOT / "models" / "challenger.pkl"),
        "phase3_val_metrics_sha256": hash_if_exists(ROOT / "metrics" / "val_metrics.json"),
        "note": "Written before loading test checkpoint labels in this script.",
    }
    write_json(ROOT / "metrics" / "freeze_manifest.json", freeze_manifest)

    corpus = read_jsonl(ROOT / "data" / "corpus.jsonl")
    splits = read_json(ROOT / "data" / "splits.json")
    targets = read_json(ROOT / "data" / "targets.json")["targets"]
    test_df = pd.read_parquet(ROOT / "benchmark" / "checkpoints_test.parquet")
    baseline_artifacts = load_pickle(ROOT / "models" / "baselines.pkl")
    challenger_artifact = load_pickle(ROOT / "models" / "challenger.pkl")

    p_const_raw = predict_constant(test_df, baseline_artifacts["stats"])
    p_const = apply_calibrator(baseline_artifacts["constant_calibrator"], p_const_raw)
    pred_const = prediction_frame(test_df, p_const, "constant")
    pred_const.to_parquet(ROOT / "predictions" / "test_constant.parquet", index=False)

    p_timing_raw = predict_timing(test_df, baseline_artifacts["stats"], baseline_artifacts["timing_artifact"])
    p_timing = apply_calibrator(baseline_artifacts["timing_calibrator"], p_timing_raw)
    pred_timing = prediction_frame(test_df, p_timing, "timing")
    pred_timing.to_parquet(ROOT / "predictions" / "test_timing.parquet", index=False)

    p_full = predict_challenger(test_df, corpus, targets, baseline_artifacts, challenger_artifact, blank_prefix=False)
    pred_full = prediction_frame(test_df, p_full, "challenger_full")
    pred_full.to_parquet(ROOT / "predictions" / "test_challenger_full.parquet", index=False)

    p_blanked = predict_challenger(test_df, corpus, targets, baseline_artifacts, challenger_artifact, blank_prefix=True)
    pred_blanked = prediction_frame(test_df, p_blanked, "challenger_blanked")
    pred_blanked.to_parquet(ROOT / "predictions" / "test_challenger_blanked.parquet", index=False)

    bins = int(config["evaluation"]["ece_bins"])
    test_metrics = {
        "constant": slice_metrics(test_df, pred_const, bins=bins)["overall"],
        "timing": slice_metrics(test_df, pred_timing, bins=bins)["overall"],
        "challenger_full": slice_metrics(test_df, pred_full, bins=bins)["overall"],
        "challenger_blanked": slice_metrics(test_df, pred_blanked, bins=bins)["overall"],
    }
    write_json(ROOT / "metrics" / "test_metrics.json", test_metrics)

    slice_all = {
        "constant": slice_metrics(test_df, pred_const, bins=bins),
        "timing": slice_metrics(test_df, pred_timing, bins=bins),
        "challenger_full": slice_metrics(test_df, pred_full, bins=bins),
        "challenger_blanked": slice_metrics(test_df, pred_blanked, bins=bins),
    }
    write_json(ROOT / "metrics" / "slice_metrics.json", slice_all)
    write_json(
        ROOT / "metrics" / "reliability.json",
        {name: metrics["reliability"] for name, metrics in test_metrics.items()},
    )

    reps = int(config["evaluation"]["bootstrap_reps_final"])
    bootstrap = {
        "constant": bootstrap_brier_ci(test_df, pred_const, reps=reps, seed=int(config["seed"]) + 10),
        "timing": bootstrap_brier_ci(test_df, pred_timing, reps=reps, seed=int(config["seed"]) + 11),
        "challenger_full": bootstrap_brier_ci(
            test_df, pred_full, reps=reps, seed=int(config["seed"]) + 12, baseline_preds=pred_timing
        ),
        "challenger_blanked": bootstrap_brier_ci(
            test_df, pred_blanked, reps=reps, seed=int(config["seed"]) + 13, baseline_preds=pred_timing
        ),
    }
    write_json(ROOT / "metrics" / "bootstrap_deltas.json", bootstrap)

    contamination = contamination_report(corpus, splits["test"]["transcript_ids"], config)
    write_json(ROOT / "metrics" / "test_contamination_report.json", contamination)
    flagged = set(contamination["flagged_transcript_ids"])
    pruned_df = test_df[~test_df["transcript_id"].isin(flagged)].copy()
    contamination_pruned = {
        "n_flagged_transcripts_removed": len(flagged),
        "flagged_transcript_ids": sorted(flagged),
        "constant": slice_metrics(pruned_df, pred_const[~pred_const["transcript_id"].isin(flagged)], bins=bins)["overall"],
        "timing": slice_metrics(pruned_df, pred_timing[~pred_timing["transcript_id"].isin(flagged)], bins=bins)["overall"],
        "challenger_full": slice_metrics(pruned_df, pred_full[~pred_full["transcript_id"].isin(flagged)], bins=bins)["overall"],
        "challenger_blanked": slice_metrics(pruned_df, pred_blanked[~pred_blanked["transcript_id"].isin(flagged)], bins=bins)["overall"],
    }
    write_json(ROOT / "metrics" / "contamination_pruned_metrics.json", contamination_pruned)

    ablation_path = ROOT / "metrics" / "ablation.json"
    ablation = read_json(ablation_path) if ablation_path.exists() else {}
    b_ts = test_metrics["timing"]["brier"]
    b_full = test_metrics["challenger_full"]["brier"]
    b_blank = test_metrics["challenger_blanked"]["brier"]
    imp_full = b_ts - b_full
    imp_blank = b_ts - b_blank
    ablation["test"] = {
        "timing_brier": b_ts,
        "challenger_full_brier": b_full,
        "challenger_blanked_brier": b_blank,
        "imp_full": imp_full,
        "imp_blank": imp_blank,
        "content_share": None if imp_full == 0 else (imp_full - imp_blank) / imp_full,
    }
    write_json(ablation_path, ablation)

    print(
        "Phase 4 locked test complete: "
        f"timing_brier={b_ts:.6f} challenger_brier={b_full:.6f} "
        f"delta={b_full - b_ts:.6f} contamination_flags={len(flagged)}"
    )


if __name__ == "__main__":
    main()
