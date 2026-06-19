from __future__ import annotations

from pathlib import Path

from src.clean import dedupe_rows, summarize_corpus
from src.common import ROOT, ensure_dirs, load_config, write_jsonl
from src.scrape import scrape


def main() -> None:
    config = load_config()
    ensure_dirs(ROOT / "data", ROOT / "data" / "raw", ROOT / "src", ROOT / "tests")
    rows = scrape()
    kept, dropped = dedupe_rows(rows)
    discovery_note_path = ROOT / "data" / "discovery_note.txt"
    discovery_note = discovery_note_path.read_text(encoding="utf-8") if discovery_note_path.exists() else ""
    write_jsonl(ROOT / "data" / "corpus.jsonl", kept)
    data_card = summarize_corpus(kept, dropped, discovery_note)
    (ROOT / "data" / "data_card.md").write_text(data_card, encoding="utf-8", newline="\n")
    print(f"Phase 0 complete: kept={len(kept)} dropped={len(dropped)}")
    floor = int(config["scrape"]["min_transcripts"])
    if len(kept) < floor:
        print(f"WARNING: corpus below floor ({len(kept)} < {floor}); documented as underpowered.")


if __name__ == "__main__":
    main()
