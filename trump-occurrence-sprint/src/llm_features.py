from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, StandardScaler

from src.matching import first_occurrence, target_tokens, tokenize


PREFIX_CONTENT_FEATURES = [
    "trailing_target_cosine",
    "fullprefix_target_cosine",
    "adjacency_prefix_score",
    "adjacency_trailing_score",
    "adjacency_prefix_density",
    "adjacency_trailing_density",
    "adjacency_trailing_max",
    "target_positive_centroid_trailing_cosine",
    "target_positive_centroid_full_cosine",
    "target_pos_minus_neg_trailing",
    "target_pos_minus_neg_full",
]
METADATA_CONTENT_FEATURES = [
    "title_target_cosine",
    "title_contains_target",
    "log_expected_remaining_words",
    "expected_remaining_per_1k",
]
FEATURE_COLUMNS = PREFIX_CONTENT_FEATURES + METADATA_CONTENT_FEATURES


def save_pickle(path: str | Path, obj: Any) -> None:
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def window_text(tokens: list[str], end: int, size: int) -> str:
    end = max(0, min(end, len(tokens)))
    start = max(0, end - size)
    return " ".join(tokens[start:end])


def fit_semantic_artifact(corpus: list[dict[str, Any]], train_ids: set[str], targets: list[dict[str, Any]], config: dict[str, Any]):
    cfg = config["challenger"]
    docs: list[str] = []
    for row in corpus:
        if row["transcript_id"] in train_ids:
            toks = tokenize(row["text"])
            docs.append(row["title"])
            docs.append(" ".join(toks[: int(cfg["full_prefix_window_words"])]))
            docs.append(" ".join(toks[-int(cfg["full_prefix_window_words"]) :]))
    docs.extend(t["target"] for t in targets)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    n_components = min(128, max(2, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=int(config["seed"]))
    normalizer = Normalizer(copy=False)
    svd.fit(matrix)
    return {"vectorizer": vectorizer, "svd": svd, "normalizer": normalizer, "kind": "tfidf_svd"}


def embed_texts(artifact: dict[str, Any], texts: list[str]) -> np.ndarray:
    X = artifact["vectorizer"].transform(texts)
    X = artifact["svd"].transform(X)
    X = artifact["normalizer"].transform(X)
    return np.asarray(X, dtype=float)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def build_adjacency_tables(
    corpus: list[dict[str, Any]],
    train_ids: set[str],
    targets: list[dict[str, Any]],
    window_words: int,
) -> dict[str, dict[str, float]]:
    target_names = [t["target"] for t in targets]
    tables: dict[str, Counter[str]] = {target: Counter() for target in target_names}
    train_rows = [row for row in corpus if row["transcript_id"] in train_ids]
    for row in train_rows:
        toks = tokenize(row["text"])
        for target in target_names:
            idx = first_occurrence(target, toks)
            if idx is None:
                continue
            start = max(0, idx - window_words)
            end = min(len(toks), idx + len(target_tokens(target)) + window_words)
            for tok in toks[start:end]:
                if len(tok) >= 3 and tok not in set(target_tokens(target)):
                    tables[target][tok] += 1
    weighted: dict[str, dict[str, float]] = {}
    for target, counter in tables.items():
        total = sum(counter.values()) or 1
        weighted[target] = {tok: count / total for tok, count in counter.most_common(80)}
    return weighted


def adjacency_score(tokens: list[str], weights: dict[str, float]) -> float:
    if not tokens or not weights:
        return 0.0
    counts = Counter(tokens)
    score = sum(counts[tok] * weight for tok, weight in weights.items())
    return float(np.log1p(score))


def adjacency_density(tokens: list[str], weights: dict[str, float]) -> float:
    if not tokens or not weights:
        return 0.0
    score = sum(weights.get(tok, 0.0) for tok in tokens)
    return float(score / max(1, len(tokens)))


def adjacency_max(tokens: list[str], weights: dict[str, float]) -> float:
    if not tokens or not weights:
        return 0.0
    return float(max((weights.get(tok, 0.0) for tok in tokens), default=0.0))


def build_target_centroids(
    corpus: list[dict[str, Any]],
    train_ids: set[str],
    targets: list[dict[str, Any]],
    semantic_artifact: dict[str, Any],
) -> dict[str, dict[str, list[float] | None]]:
    train_rows = [row for row in corpus if row["transcript_id"] in train_ids]
    doc_texts = [" ".join(tokenize(row["text"])) for row in train_rows]
    doc_embs = embed_texts(semantic_artifact, doc_texts)
    token_cache = [tokenize(row["text"]) for row in train_rows]
    centroids: dict[str, dict[str, list[float] | None]] = {}
    zero = np.zeros(doc_embs.shape[1], dtype=float)
    for target in [t["target"] for t in targets]:
        positives = []
        negatives = []
        for idx, toks in enumerate(token_cache):
            if first_occurrence(target, toks) is None:
                negatives.append(idx)
            else:
                positives.append(idx)
        pos = doc_embs[positives].mean(axis=0) if positives else zero.copy()
        neg = doc_embs[negatives].mean(axis=0) if negatives else zero.copy()
        pos_norm = np.linalg.norm(pos)
        neg_norm = np.linalg.norm(neg)
        if pos_norm > 0:
            pos = pos / pos_norm
        if neg_norm > 0:
            neg = neg / neg_norm
        centroids[target] = {
            "positive": pos.astype(float).tolist(),
            "negative": neg.astype(float).tolist(),
            "positive_docs": len(positives),
            "negative_docs": len(negatives),
        }
    return centroids


def fit_feature_artifact(
    corpus: list[dict[str, Any]],
    splits: dict[str, Any],
    targets: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    train_ids = set(splits["train"]["transcript_ids"])
    semantic = fit_semantic_artifact(corpus, train_ids, targets, config)
    adjacency = build_adjacency_tables(
        corpus,
        train_ids,
        targets,
        window_words=int(config["challenger"]["cooccurrence_window_words"]),
    )
    centroids = build_target_centroids(corpus, train_ids, targets, semantic)
    return {
        "semantic": semantic,
        "adjacency": adjacency,
        "target_centroids": centroids,
        "feature_columns": FEATURE_COLUMNS,
        "config": config["challenger"],
    }


def build_content_features(
    df: pd.DataFrame,
    corpus: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    artifact: dict[str, Any],
    blank_prefix: bool = False,
) -> pd.DataFrame:
    cfg = artifact["config"]
    rows_by_id = {row["transcript_id"]: row for row in corpus}
    token_cache = {row["transcript_id"]: tokenize(row["text"]) for row in corpus}
    target_texts = [t["target"] for t in targets]
    target_set = set(target_texts)

    target_emb = dict(zip(target_texts, embed_texts(artifact["semantic"], target_texts)))
    unique_titles = sorted(set(df["title"].astype(str)))
    title_emb = dict(zip(unique_titles, embed_texts(artifact["semantic"], unique_titles)))

    unique_windows = sorted(set(zip(df["transcript_id"], df["elapsed_words"])))
    trailing_texts: list[str] = []
    full_texts: list[str] = []
    for tid, elapsed in unique_windows:
        toks = token_cache[tid]
        trailing_texts.append("" if blank_prefix else window_text(toks, int(elapsed), int(cfg["trailing_window_words"])))
        full_texts.append("" if blank_prefix else window_text(toks, int(elapsed), int(cfg["full_prefix_window_words"])))
    trailing_embs = embed_texts(artifact["semantic"], trailing_texts)
    full_embs = embed_texts(artifact["semantic"], full_texts)
    trailing_map = dict(zip(unique_windows, trailing_embs))
    full_map = dict(zip(unique_windows, full_embs))

    feature_rows: list[dict[str, float]] = []
    for row in df.itertuples(index=False):
        target = getattr(row, "target")
        tid = getattr(row, "transcript_id")
        elapsed = int(getattr(row, "elapsed_words"))
        toks = token_cache[tid]
        prefix_toks = [] if blank_prefix else toks[:elapsed]
        trailing_toks = [] if blank_prefix else toks[max(0, elapsed - int(cfg["trailing_window_words"])) : elapsed]
        t_emb = target_emb[target]
        title = getattr(row, "title")
        weights = artifact["adjacency"].get(target, {})
        centroids = artifact["target_centroids"].get(target, {})
        pos_centroid = np.asarray(centroids.get("positive"), dtype=float)
        neg_centroid = np.asarray(centroids.get("negative"), dtype=float)
        target_tok = target_tokens(target)
        title_toks = tokenize(title)
        trailing_vec = trailing_map[(tid, elapsed)]
        full_vec = full_map[(tid, elapsed)]
        pos_trailing = float(np.dot(trailing_vec, pos_centroid)) if pos_centroid.size else 0.0
        pos_full = float(np.dot(full_vec, pos_centroid)) if pos_centroid.size else 0.0
        neg_trailing = float(np.dot(trailing_vec, neg_centroid)) if neg_centroid.size else 0.0
        neg_full = float(np.dot(full_vec, neg_centroid)) if neg_centroid.size else 0.0
        feature_rows.append(
            {
                "trailing_target_cosine": float(np.dot(trailing_vec, t_emb)),
                "fullprefix_target_cosine": float(np.dot(full_vec, t_emb)),
                "adjacency_prefix_score": adjacency_score(prefix_toks, weights),
                "adjacency_trailing_score": adjacency_score(trailing_toks, weights),
                "adjacency_prefix_density": adjacency_density(prefix_toks, weights),
                "adjacency_trailing_density": adjacency_density(trailing_toks, weights),
                "adjacency_trailing_max": adjacency_max(trailing_toks, weights),
                "target_positive_centroid_trailing_cosine": pos_trailing,
                "target_positive_centroid_full_cosine": pos_full,
                "target_pos_minus_neg_trailing": pos_trailing - neg_trailing,
                "target_pos_minus_neg_full": pos_full - neg_full,
                "title_target_cosine": float(np.dot(title_emb[title], t_emb)),
                "title_contains_target": float(
                    any(title_toks[i : i + len(target_tok)] == target_tok for i in range(0, len(title_toks) - len(target_tok) + 1))
                    if target_tok
                    else 0.0
                ),
                "log_expected_remaining_words": float(np.log1p(max(0, getattr(row, "expected_remaining_words")))),
                "expected_remaining_per_1k": float(max(0, getattr(row, "expected_remaining_words")) / 1000.0),
            }
        )
    out = pd.DataFrame(feature_rows, index=df.index)
    return out[FEATURE_COLUMNS]
