from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_corpus_map(path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["transcript_id"]: row for row in read_jsonl(path)}


def checkpoint_prompt(
    row: pd.Series,
    corpus: dict[str, dict[str, Any]],
    max_prefix_words: int,
    blank_prefix: bool = False,
) -> str:
    meta = corpus[str(row["transcript_id"])]
    tokens = simple_tokenize(meta["text"])
    elapsed = int(row["elapsed_words"])
    prefix = "" if blank_prefix else " ".join(tokens[:elapsed][-max_prefix_words:])
    if not prefix:
        prefix = "[empty prefix]"
    title = meta.get("title", "")
    date = meta.get("date", "")
    fmt = meta.get("format", "unknown")
    target = row["target"]
    return (
        "Task: estimate whether a target word or phrase will appear later in this transcript.\n"
        f"Title: {title}\n"
        f"Date: {date}\n"
        f"Format: {fmt}\n"
        f"Checkpoint percent: {row['t_pct']}\n"
        f"Elapsed words: {row['elapsed_words']}\n"
        f"Expected remaining words: {row['expected_remaining_words']}\n"
        f"Target: {target}\n"
        "Transcript prefix:\n"
        f"{prefix}\n"
        "Question: Will the target appear after this checkpoint before the transcript ends?"
    )


def prediction_frame(labels: pd.DataFrame, p_pred: np.ndarray, model_name: str) -> pd.DataFrame:
    out = labels[["transcript_id", "target", "t_pct"]].copy()
    out["p_pred"] = p_pred.astype(float)
    out["model_name"] = model_name
    return out


def load_frame(path: str | Path, row_cap: int | None = None, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if row_cap and row_cap > 0 and len(df) > row_cap:
        df = df.sample(n=row_cap, random_state=seed).sort_index().reset_index(drop=True)
    return df.reset_index(drop=True)
