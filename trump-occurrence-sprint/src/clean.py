from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.common import normalize_text, parse_date, sha256_text, stable_id
from src.matching import tokenize


DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b"
)
SPEAKER_RE = re.compile(r"^(.{1,90}?)\s*\(?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)?\s*:?\s*$")
INLINE_SPEAKER_RE = re.compile(
    r"(?:^|\n)(?P<speaker>[A-Z][A-Za-z0-9 .,'&\-]{1,80}?)(?::\s*)?\(\s*"
    r"(?:(?:\d{1,2}:)?\d{2}:\d{2})?\s*\)\s*:?\s*",
    re.M,
)
BARE_TIME_RE = re.compile(r"^\(?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)?\s*:?\s*$")
ANNOTATION_RE = re.compile(r"\[(?:applause|inaudible|crosstalk|crowd noise|laughter|music|cheering)[^\]]*\]", re.I)
INLINE_TIME_RE = re.compile(r"\(?\b\d{1,2}:\d{2}(?::\d{2})?\b\)?")


@dataclass
class ParsedTranscript:
    transcript_id: str
    title: str
    date: str | None
    source_url: str
    format: str
    n_words: int
    text: str
    primary_speaker: str | None
    all_speakers: list[str]
    raw_path: str | None = None


def infer_format(title: str) -> str:
    t = title.lower()
    if any(x in t for x in ["rally", "remarks at", "delivers remarks"]):
        return "rally"
    if any(x in t for x in ["speech", "address", "commencement"]):
        return "speech"
    if any(x in t for x in ["interview", "60 minutes", "town hall"]):
        return "interview"
    if "debate" in t:
        return "debate"
    if any(x in t for x in ["press conference", "news conference", "gaggle", "speaks to press", "takes questions"]):
        return "press_event"
    if any(x in t for x in ["trial", "court", "deposition", "supreme court"]):
        return "legal"
    if any(x in t for x in ["meeting", "announcement", "executive order", "signs", "ceremony"]):
        return "press_event"
    return "unknown"


def extract_title(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if h1:
        title = normalize_text(h1.get_text(" ", strip=True))
        if title:
            return title
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string.replace("| Rev", "").replace("Transcript", ""))
    return None


def extract_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    return parse_date(match.group(0).replace("Sept.", "Sep")) if match else None


def strip_boilerplate_line(line: str) -> bool:
    bad = [
        "hungry for more",
        "subscribe to",
        "thank you for subscribing",
        "share this post",
        "copyright disclaimer",
        "under title 17",
        "transcripts home",
        "contact support",
        "request a demo",
        "try rev free",
        "no items found",
        "read the transcript here",
        "view all",
        "services",
        "industries",
        "resources",
        "about rev",
    ]
    low = line.lower()
    return any(x in low for x in bad)


def clean_utterance(line: str) -> str:
    line = ANNOTATION_RE.sub(" ", line)
    line = re.sub(r"\[.*?\]", " ", line)
    line = re.sub(r"\(\s*inaudible\s+[^)]*\)", " ", line, flags=re.I)
    line = INLINE_TIME_RE.sub(" ", line)
    line = normalize_text(line)
    return line


def parse_turns(soup: BeautifulSoup) -> list[tuple[str | None, str]]:
    text = soup.get_text("\n", strip=True)
    lines = [normalize_text(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    start = 0
    for i, line in enumerate(lines):
        if "copyright disclaimer" in line.lower():
            start = i + 1
            break
    lines = lines[start:]

    body_lines: list[str] = []
    for line in lines:
        if strip_boilerplate_line(line) and body_lines:
            break
        if not strip_boilerplate_line(line):
            body_lines.append(line)

    body = "\n".join(body_lines)
    inline_matches = list(INLINE_SPEAKER_RE.finditer(body))
    if inline_matches:
        turns: list[tuple[str | None, str]] = []
        for i, match in enumerate(inline_matches):
            speaker = normalize_text(match.group("speaker").strip(":- "))
            start_pos = match.end()
            end_pos = inline_matches[i + 1].start() if i + 1 < len(inline_matches) else len(body)
            utterance = clean_utterance(body[start_pos:end_pos])
            if speaker and utterance:
                turns.append((speaker, utterance))
        if turns:
            return turns

    turns: list[tuple[str | None, str]] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            joined = clean_utterance(" ".join(buffer))
            if joined:
                turns.append((current_speaker, joined))
        buffer = []

    for line in lines:
        if strip_boilerplate_line(line):
            if len(turns) > 3:
                break
            continue
        if BARE_TIME_RE.match(line):
            continue
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            label = normalize_text(speaker_match.group(1).strip(":- "))
            if label and len(label.split()) <= 8:
                flush()
                current_speaker = label
                continue
        if len(line) > 1:
            buffer.append(line)
    flush()
    return turns


def choose_primary_speaker(
    turns: list[tuple[str | None, str]],
    prefer_donald_trump: bool = True,
    dominant_min_share: float = 0.35,
) -> str | None:
    counts: Counter[str] = Counter()
    for speaker, utterance in turns:
        if speaker:
            counts[speaker] += len(tokenize(utterance))
    if not counts:
        return None
    if prefer_donald_trump:
        for speaker in counts:
            if "donald trump" in speaker.lower() or speaker.lower() == "trump":
                return speaker
    speaker, count = counts.most_common(1)[0]
    total = sum(counts.values())
    if total and count / total >= dominant_min_share:
        return speaker
    return None


def parse_transcript_html(
    html: str,
    source_url: str,
    raw_path: str | None = None,
    prefer_donald_trump: bool = True,
    dominant_speaker_min_share: float = 0.35,
) -> ParsedTranscript | None:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)
    title = extract_title(soup)
    date = extract_date(page_text)
    if not title or not date:
        return None

    turns = parse_turns(soup)
    primary = choose_primary_speaker(turns, prefer_donald_trump, dominant_speaker_min_share)
    if turns and primary:
        text_parts = [utterance for speaker, utterance in turns if speaker == primary]
    elif turns:
        text_parts = [utterance for _, utterance in turns]
    else:
        text_parts = [page_text]

    text = clean_utterance(" ".join(text_parts))
    tokens = tokenize(text)
    if not tokens:
        return None

    transcript_id = stable_id(source_url, title, date, prefix="rev")
    speakers = sorted({speaker for speaker, _ in turns if speaker})
    return ParsedTranscript(
        transcript_id=transcript_id,
        title=title,
        date=date,
        source_url=source_url,
        format=infer_format(title),
        n_words=len(tokens),
        text=text,
        primary_speaker=primary,
        all_speakers=speakers,
        raw_path=raw_path,
    )


def parsed_to_row(parsed: ParsedTranscript) -> dict[str, Any]:
    return {
        "transcript_id": parsed.transcript_id,
        "title": parsed.title,
        "date": parsed.date,
        "source_url": parsed.source_url,
        "format": parsed.format,
        "n_words": parsed.n_words,
        "text": parsed.text,
        "primary_speaker": parsed.primary_speaker,
        "all_speakers": parsed.all_speakers,
        "raw_path": parsed.raw_path,
        "text_sha256": sha256_text(parsed.text),
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    title_date_seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda r: (r["date"], r["title"], r["source_url"])):
        key = (row["title"].lower(), row["date"])
        text_hash = row["text_sha256"]
        if text_hash in seen_hashes:
            dropped.append({"source_url": row["source_url"], "reason": "duplicate_text", "kept": seen_hashes[text_hash]})
            continue
        if key in title_date_seen:
            dropped.append({"source_url": row["source_url"], "reason": "duplicate_title_date"})
            continue
        seen_hashes[text_hash] = row["source_url"]
        title_date_seen.add(key)
        kept.append(row)
    return kept, dropped


def summarize_corpus(rows: list[dict[str, Any]], dropped: list[dict[str, Any]], discovery_note: str) -> str:
    dates = sorted(row["date"] for row in rows)
    formats = Counter(row["format"] for row in rows)
    lengths = sorted(row["n_words"] for row in rows)

    def pct(p: float) -> int | None:
        if not lengths:
            return None
        idx = min(len(lengths) - 1, max(0, int(round((len(lengths) - 1) * p))))
        return lengths[idx]

    lines = [
        "# Data Card",
        "",
        f"Transcripts kept: {len(rows)}",
        f"Transcripts dropped: {len(dropped)}",
        f"Date range: {dates[0] if dates else 'n/a'} to {dates[-1] if dates else 'n/a'}",
        "",
        "## Discovery",
        "",
        discovery_note,
        "",
        "## Format Distribution",
        "",
    ]
    for fmt, count in sorted(formats.items()):
        lines.append(f"- {fmt}: {count}")
    lines.extend(
        [
            "",
            "## Length Distribution",
            "",
            f"- min: {lengths[0] if lengths else 'n/a'}",
            f"- p25: {pct(0.25)}",
            f"- median: {pct(0.50)}",
            f"- p75: {pct(0.75)}",
            f"- max: {lengths[-1] if lengths else 'n/a'}",
            "",
            "## Cleaning Decisions",
            "",
            "- Parsed Rev speaker/timestamp text from saved raw HTML.",
            "- Preferred `Donald Trump` speaker turns where that speaker label was present.",
            "- Otherwise kept the dominant labeled speaker when a clear dominant speaker existed.",
            "- Fell back to all parsed turns where no reliable primary speaker was available.",
            "- Stripped bracketed annotations, inline timestamps, and visible Rev boilerplate.",
            "- Deduplicated exact normalized text hashes and repeated title/date pairs.",
            "",
            "## Known Gaps",
            "",
        ]
    )
    if len(rows) < 60:
        lines.append("- UNDERPOWERED CORPUS: fewer than 60 transcripts were available after cleaning.")
    else:
        lines.append("- Corpus meets the 60-transcript floor.")
    if len(rows) < 100:
        lines.append("- Corpus is below the 100-transcript target.")
    if dropped:
        lines.extend(["", "## Dropped Items", ""])
        for item in dropped[:80]:
            lines.append(f"- {item.get('source_url')}: {item.get('reason')}")
    return "\n".join(lines) + "\n"
