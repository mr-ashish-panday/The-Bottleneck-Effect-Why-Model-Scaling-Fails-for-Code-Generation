from __future__ import annotations

import pandas as pd

from src.checkpoints import build_checkpoint_rows
from src.common import ROOT, ensure_dirs, load_config, read_jsonl, write_json
from src.evaluate import bootstrap_brier_ci, metric_block
from src.splits import chronological_splits
from src.targets import build_targets


def main() -> None:
    config = load_config()
    ensure_dirs(ROOT / "benchmark", ROOT / "data", ROOT / "metrics", ROOT / "predictions")
    corpus = read_jsonl(ROOT / "data" / "corpus.jsonl")
    splits = chronological_splits(
        corpus,
        train_frac=float(config["splits"]["train_frac"]),
        val_frac=float(config["splits"]["val_frac"]),
    )
    write_json(ROOT / "data" / "splits.json", splits)

    targets_obj = build_targets(corpus, splits, config)
    write_json(ROOT / "data" / "targets.json", targets_obj)

    frames = build_checkpoint_rows(corpus, splits, targets_obj, config["checkpoints"]["grid_pct"])
    for split, frame in frames.items():
        frame.to_parquet(ROOT / "benchmark" / f"checkpoints_{split}.parquet", index=False)

    val_labels = frames["val"]
    dummy = val_labels[["transcript_id", "target", "t_pct"]].copy()
    dummy["p_pred"] = float(val_labels["label_occurs_after"].mean()) if len(val_labels) else 0.5
    dummy["model_name"] = "dummy_constant"
    dummy.to_parquet(ROOT / "predictions" / "val_dummy_constant.parquet", index=False)
    metrics = {
        "dummy_constant": metric_block(val_labels, dummy, bins=int(config["evaluation"]["ece_bins"])),
        "dummy_bootstrap": bootstrap_brier_ci(
            val_labels,
            dummy,
            reps=int(config["evaluation"]["bootstrap_reps_dev"]),
            seed=int(config["seed"]),
        ),
        "checkpoint_rows": {split: int(len(frame)) for split, frame in frames.items()},
        "targets": targets_obj["counts_by_band"],
    }
    write_json(ROOT / "metrics" / "dummy_metrics.json", metrics)
    print(
        "Phase 1 complete: "
        f"train_rows={len(frames['train'])} val_rows={len(frames['val'])} test_rows={len(frames['test'])} "
        f"targets={len(targets_obj['targets'])}"
    )


if __name__ == "__main__":
    main()
