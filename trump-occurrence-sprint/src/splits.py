from __future__ import annotations

from collections import Counter
from typing import Any


def chronological_splits(
    corpus: list[dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    rows = sorted(corpus, key=lambda r: (r["date"], r["transcript_id"]))
    n = len(rows)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "transcript_ids": [r["transcript_id"] for r in items],
            "count": len(items),
            "date_min": items[0]["date"] if items else None,
            "date_max": items[-1]["date"] if items else None,
            "format_counts": dict(Counter(r["format"] for r in items)),
        }

    return {
        "method": "chronological_transcript_level",
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "train": pack(train),
        "val": pack(val),
        "test": pack(test),
    }


def split_lookup(splits: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for transcript_id in splits[split]["transcript_ids"]:
            lookup[transcript_id] = split
    return lookup
