from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
HYPHEN_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
HYPHEN_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]")


@dataclass(frozen=True)
class MatchConfig:
    split_hyphens: bool = True
    strip_possessive: bool = True


def normalize_for_match(text: str, split_hyphens: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    if split_hyphens:
        text = HYPHEN_RE.sub(" ", text)
    else:
        text = HYPHEN_RE.sub("-", text)
    return text


def normalize_token(token: str, strip_possessive: bool = True) -> str:
    token = normalize_for_match(token, split_hyphens=False)
    if strip_possessive and (token.endswith("'s") or token.endswith("s'")):
        token = token[:-2]
    return token


def tokenize(text: str, split_hyphens: bool = True, strip_possessive: bool = True) -> list[str]:
    text = normalize_for_match(text, split_hyphens=split_hyphens)
    token_re = WORD_RE if split_hyphens else HYPHEN_WORD_RE
    return [normalize_token(m.group(0), strip_possessive=strip_possessive) for m in token_re.finditer(text)]


def target_tokens(target: str, config: MatchConfig | None = None) -> list[str]:
    cfg = config or MatchConfig()
    return tokenize(target, split_hyphens=cfg.split_hyphens, strip_possessive=cfg.strip_possessive)


def occurrences(target: str, tokens: list[str], config: MatchConfig | None = None) -> list[int]:
    cfg = config or MatchConfig()
    needle = target_tokens(target, cfg)
    if not needle:
        return []
    norm_tokens = [normalize_token(tok, strip_possessive=cfg.strip_possessive) for tok in tokens]
    n = len(needle)
    hits: list[int] = []
    for i in range(0, len(norm_tokens) - n + 1):
        if norm_tokens[i : i + n] == needle:
            hits.append(i)
    return hits


def first_occurrence(target: str, tokens: list[str], config: MatchConfig | None = None) -> int | None:
    hits = occurrences(target, tokens, config=config)
    return hits[0] if hits else None
