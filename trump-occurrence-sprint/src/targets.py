from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.matching import occurrences, tokenize


STOPWORDS = {
    "the", "and", "for", "that", "you", "are", "with", "this", "have", "not", "but", "was", "they",
    "will", "our", "all", "can", "just", "from", "what", "about", "there", "their", "would", "when",
    "who", "has", "had", "were", "been", "your", "his", "her", "she", "him", "them", "than", "then",
    "very", "much", "one", "two", "three", "say", "said", "get", "got", "going", "because", "into",
    "out", "over", "under", "also", "more", "most", "some", "any", "every", "really", "like", "know",
    "think", "want", "thank", "thanks", "yes", "yeah", "well", "look", "good", "great",
}


def valid_unigram(tok: str, min_len: int) -> bool:
    return len(tok) >= min_len and tok not in STOPWORDS and not tok.isdigit()


def valid_phrase(parts: tuple[str, ...], min_len: int) -> bool:
    content = [p for p in parts if p not in STOPWORDS]
    return len(content) >= 1 and all(len(p) >= min_len and not p.isdigit() for p in content)


def count_train_terms(train_rows: list[dict[str, Any]], min_len: int, max_phrase_len: int):
    total_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    doc_ids: defaultdict[str, list[str]] = defaultdict(list)
    phrase_counts: Counter[str] = Counter()
    phrase_doc_counts: Counter[str] = Counter()
    phrase_doc_ids: defaultdict[str, list[str]] = defaultdict(list)

    for row in train_rows:
        toks = tokenize(row["text"])
        seen: set[str] = set()
        for tok in toks:
            if valid_unigram(tok, min_len):
                total_counts[tok] += 1
                seen.add(tok)
        for tok in seen:
            doc_counts[tok] += 1
            doc_ids[tok].append(row["transcript_id"])

        phrase_seen: set[str] = set()
        for n in range(2, max_phrase_len + 1):
            for i in range(0, len(toks) - n + 1):
                parts = tuple(toks[i : i + n])
                if valid_phrase(parts, min_len):
                    phrase = " ".join(parts)
                    phrase_counts[phrase] += 1
                    phrase_seen.add(phrase)
        for phrase in phrase_seen:
            phrase_doc_counts[phrase] += 1
            phrase_doc_ids[phrase].append(row["transcript_id"])

    return total_counts, doc_counts, doc_ids, phrase_counts, phrase_doc_counts, phrase_doc_ids


def pick_bands(
    total_counts: Counter[str],
    doc_counts: Counter[str],
    doc_ids: dict[str, list[str]],
    high_n: int,
    mid_n: int,
    rare_n: int,
    cold_n: int,
    cold_max_docs: int,
) -> list[dict[str, Any]]:
    terms = [t for t, df in doc_counts.items() if df > 0]
    by_df_desc = sorted(terms, key=lambda t: (doc_counts[t], total_counts[t], t), reverse=True)
    high = by_df_desc[:high_n]

    remaining = [t for t in by_df_desc if t not in set(high)]
    mid_start = max(0, len(remaining) // 3)
    mid_pool = remaining[mid_start:]
    mid = sorted(mid_pool, key=lambda t: (abs(doc_counts[t] - 8), -total_counts[t], t))[:mid_n]

    used = set(high) | set(mid)
    rare_pool = [t for t in terms if t not in used and 2 <= doc_counts[t] <= max(10, cold_max_docs + 4)]
    rare = sorted(rare_pool, key=lambda t: (doc_counts[t], -total_counts[t], t))[:rare_n]

    used |= set(rare)
    cold_pool = [t for t in terms if t not in used and doc_counts[t] <= cold_max_docs]
    cold = sorted(cold_pool, key=lambda t: (doc_counts[t], -total_counts[t], t))[:cold_n]

    targets: list[dict[str, Any]] = []
    for band, items in (("high", high), ("mid", mid), ("rare", rare), ("cold", cold)):
        for target in items:
            targets.append(
                {
                    "target": target,
                    "target_band": band,
                    "kind": "unigram",
                    "train_doc_freq": int(doc_counts[target]),
                    "train_total_count": int(total_counts[target]),
                    "source_split": "train",
                    "source_transcript_ids": doc_ids[target],
                }
            )
    return targets


def pick_phrases(
    phrase_counts: Counter[str],
    phrase_doc_counts: Counter[str],
    phrase_doc_ids: dict[str, list[str]],
    phrase_n: int,
) -> list[dict[str, Any]]:
    pool = [p for p, df in phrase_doc_counts.items() if 2 <= df <= 25 and phrase_counts[p] >= 2]
    chosen = sorted(pool, key=lambda p: (phrase_doc_counts[p], phrase_counts[p], p), reverse=True)[:phrase_n]
    return [
        {
            "target": phrase,
            "target_band": "phrase",
            "kind": "phrase",
            "train_doc_freq": int(phrase_doc_counts[phrase]),
            "train_total_count": int(phrase_counts[phrase]),
            "source_split": "train",
            "source_transcript_ids": phrase_doc_ids[phrase],
        }
        for phrase in chosen
    ]


def build_targets(corpus: list[dict[str, Any]], splits: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["targets"]
    train_ids = set(splits["train"]["transcript_ids"])
    train_rows = [row for row in corpus if row["transcript_id"] in train_ids]
    counts = count_train_terms(
        train_rows,
        min_len=int(cfg["min_token_len"]),
        max_phrase_len=int(cfg["max_phrase_len"]),
    )
    total_counts, doc_counts, doc_ids, phrase_counts, phrase_doc_counts, phrase_doc_ids = counts
    unigram_targets = pick_bands(
        total_counts,
        doc_counts,
        doc_ids,
        high_n=int(cfg["high_unigrams"]),
        mid_n=int(cfg["mid_unigrams"]),
        rare_n=int(cfg["rare_unigrams"]),
        cold_n=int(cfg["cold"]),
        cold_max_docs=int(cfg["cold_max_train_transcripts"]),
    )
    phrase_targets = pick_phrases(
        phrase_counts,
        phrase_doc_counts,
        phrase_doc_ids,
        phrase_n=int(cfg["phrases"]),
    )
    targets = unigram_targets + phrase_targets
    seen: set[str] = set()
    deduped = []
    for item in targets:
        if item["target"] not in seen:
            seen.add(item["target"])
            deduped.append(item)
    return {
        "method": "train_only_frequency_bands",
        "train_transcript_count": len(train_rows),
        "targets": deduped,
        "counts_by_band": dict(Counter(t["target_band"] for t in deduped)),
    }
