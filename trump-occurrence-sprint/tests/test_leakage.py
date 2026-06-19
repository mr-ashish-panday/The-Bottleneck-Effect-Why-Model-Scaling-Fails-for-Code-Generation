from pathlib import Path

import pandas as pd
import pytest

from src.common import ROOT, read_json, read_jsonl


def require_phase1():
    if not (ROOT / "data" / "splits.json").exists():
        pytest.skip("Phase 1 artifacts not built yet")


def test_transcript_ids_do_not_cross_splits():
    require_phase1()
    splits = read_json(ROOT / "data" / "splits.json")
    sets = {name: set(splits[name]["transcript_ids"]) for name in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


def test_targets_are_train_only():
    require_phase1()
    splits = read_json(ROOT / "data" / "splits.json")
    targets = read_json(ROOT / "data" / "targets.json")["targets"]
    train_ids = set(splits["train"]["transcript_ids"])
    val_test = set(splits["val"]["transcript_ids"]) | set(splits["test"]["transcript_ids"])
    for target in targets:
        assert target["source_split"] == "train"
        assert set(target["source_transcript_ids"]).issubset(train_ids)
        assert not (set(target["source_transcript_ids"]) & val_test)


def test_model_facing_checkpoint_fields_do_not_include_future_label_fields():
    require_phase1()
    path = ROOT / "benchmark" / "checkpoints_train.parquet"
    df = pd.read_parquet(path)
    forbidden_model_fields = {"first_occurrence_index", "label_occurs_after"}
    model_facing = {
        "transcript_id",
        "target",
        "target_band",
        "t_pct",
        "elapsed_words",
        "expected_remaining_words",
        "format",
        "title",
        "date",
        "source_url",
        "n_words",
    }
    assert forbidden_model_fields.issubset(df.columns)
    assert "first_occurrence_index" not in model_facing
    assert "label_occurs_after" not in model_facing
