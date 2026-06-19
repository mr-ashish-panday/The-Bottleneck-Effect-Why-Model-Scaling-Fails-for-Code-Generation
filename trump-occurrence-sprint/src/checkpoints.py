from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import pandas as pd

from src.matching import first_occurrence, tokenize


def build_duration_tables(corpus: list[dict[str, Any]], train_ids: set[str]) -> dict[str, list[int]]:
    by_format: defaultdict[str, list[int]] = defaultdict(list)
    global_lengths: list[int] = []
    for row in corpus:
        if row["transcript_id"] in train_ids:
            by_format[row["format"]].append(int(row["n_words"]))
            global_lengths.append(int(row["n_words"]))
    by_format["__global__"] = global_lengths
    return {k: sorted(v) for k, v in by_format.items() if v}


def expected_remaining_words(format_name: str, elapsed_words: int, duration_tables: dict[str, list[int]]) -> int:
    lengths = duration_tables.get(format_name) or duration_tables["__global__"]
    eligible = [n for n in lengths if n > elapsed_words]
    if not eligible:
        eligible = duration_tables["__global__"]
    median_total = statistics.median(eligible)
    return max(0, int(round(median_total - elapsed_words)))


def build_checkpoint_rows(
    corpus: list[dict[str, Any]],
    splits: dict[str, Any],
    targets_obj: dict[str, Any],
    grid_pct: list[int],
) -> dict[str, pd.DataFrame]:
    targets = targets_obj["targets"]
    train_ids = set(splits["train"]["transcript_ids"])
    duration_tables = build_duration_tables(corpus, train_ids)
    by_split_ids = {split: set(splits[split]["transcript_ids"]) for split in ("train", "val", "test")}
    frames: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for row in corpus:
        split = next((s for s, ids in by_split_ids.items() if row["transcript_id"] in ids), None)
        if not split:
            continue
        tokens = tokenize(row["text"])
        n_words = len(tokens)
        for target in targets:
            first_idx = first_occurrence(target["target"], tokens)
            for pct in grid_pct:
                elapsed = int(round(n_words * (pct / 100.0)))
                elapsed = min(max(elapsed, 0), n_words)
                if first_idx is not None and first_idx < elapsed:
                    continue
                frames[split].append(
                    {
                        "transcript_id": row["transcript_id"],
                        "target": target["target"],
                        "target_band": target["target_band"],
                        "t_pct": int(pct),
                        "elapsed_words": int(elapsed),
                        "expected_remaining_words": expected_remaining_words(row["format"], elapsed, duration_tables),
                        "format": row["format"],
                        "title": row["title"],
                        "date": row["date"],
                        "source_url": row["source_url"],
                        "n_words": int(n_words),
                        "first_occurrence_index": None if first_idx is None else int(first_idx),
                        "label_occurs_after": bool(first_idx is not None and first_idx >= elapsed),
                    }
                )
    return {split: pd.DataFrame(rows) for split, rows in frames.items()}
