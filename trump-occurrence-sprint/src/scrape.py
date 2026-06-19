from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.clean import parse_transcript_html, parsed_to_row
from src.common import ROOT, ensure_dirs, load_config, slugify, write_jsonl


TRANSCRIPT_RE = re.compile(r"/transcripts/[A-Za-z0-9_.~/-]+")


def fetch(url: str, timeout: int, user_agent: str) -> tuple[int, str]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    return response.status_code, response.text


def extract_transcript_urls_from_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/transcripts/" in href:
            urls.append(urljoin(base_url, href.split("#")[0]))
    for match in TRANSCRIPT_RE.finditer(html):
        urls.append(urljoin(base_url, match.group(0)))
    return list(dict.fromkeys(urls))


def extract_transcript_urls_from_sitemap(xml: str) -> list[str]:
    urls = re.findall(r"<loc>(.*?)</loc>", xml)
    filtered = [
        url
        for url in urls
        if "/transcripts/" in url and any(term in url.lower() for term in ("trump", "donald"))
    ]
    return list(dict.fromkeys(filtered))


def raw_path_for_url(raw_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    slug = slugify(parsed.path.replace("/transcripts/", "").strip("/"))
    return raw_dir / f"{slug}.html"


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def discover_urls(config: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    scrape_cfg = config["scrape"]
    timeout = int(scrape_cfg["request_timeout_seconds"])
    user_agent = scrape_cfg["user_agent"]
    category_url = scrape_cfg["category_url"]
    sitemap_url = scrape_cfg["sitemap_url"]

    status, category_html = fetch(category_url, timeout, user_agent)
    if status != 200:
        raise RuntimeError(f"Category fetch failed: {status} {category_url}")
    category_urls = extract_transcript_urls_from_html(category_html, category_url)

    discovered: list[tuple[str, str]] = [(url, "category_page") for url in category_urls]
    discovery_note = (
        f"Started from {category_url}. The category HTML yielded {len(category_urls)} unique transcript URLs."
    )

    if len(category_urls) < int(scrape_cfg["min_transcripts"]):
        status, sitemap_xml = fetch(sitemap_url, timeout, user_agent)
        if status != 200:
            raise RuntimeError(f"Sitemap fetch failed: {status} {sitemap_url}")
        sitemap_urls = extract_transcript_urls_from_sitemap(sitemap_xml)
        known = {url for url, _ in discovered}
        for url in sitemap_urls:
            if url not in known:
                discovered.append((url, "rev_sitemap_trump_slug"))
                known.add(url)
        discovery_note += (
            f" Because this was below the 60-transcript floor, Rev's own sitemap at {sitemap_url} "
            f"was filtered to transcript URLs with `trump` or `donald` in the slug, adding "
            f"{len(discovered) - len(category_urls)} candidate URLs."
        )

    return discovered, discovery_note


def scrape(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    scrape_cfg = config["scrape"]
    clean_cfg = config["cleaning"]
    raw_dir = ROOT / "data" / "raw"
    ensure_dirs(raw_dir)

    discovered, discovery_note = discover_urls(config)
    max_pages = int(scrape_cfg["max_candidate_pages"])
    target = int(scrape_cfg["target_transcripts"])
    candidates = discovered[:max_pages]

    timeout = int(scrape_cfg["request_timeout_seconds"])
    user_agent = scrape_cfg["user_agent"]
    min_words = int(clean_cfg["min_words"])

    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def fetch_one(item: tuple[str, str]) -> dict[str, Any]:
        url, source = item
        raw_path = raw_path_for_url(raw_dir, url)
        try:
            if raw_path.exists():
                html = raw_path.read_text(encoding="utf-8", errors="replace")
                status = 200
                cached = True
            else:
                status, html = fetch(url, timeout, user_agent)
                cached = False
                if status == 200:
                    save_text(raw_path, html)
                time.sleep(0.05)
            parsed = None
            reason = None
            if status == 200:
                parsed_obj = parse_transcript_html(
                    html,
                    source_url=url,
                    raw_path=str(raw_path.relative_to(ROOT)),
                    prefer_donald_trump=bool(clean_cfg["prefer_donald_trump_speaker"]),
                    dominant_speaker_min_share=float(clean_cfg["dominant_speaker_min_share"]),
                )
                if parsed_obj and parsed_obj.n_words >= min_words:
                    parsed = parsed_to_row(parsed_obj)
                elif parsed_obj:
                    reason = f"too_short_{parsed_obj.n_words}"
                else:
                    reason = "parse_failed_or_missing_date"
            else:
                reason = f"http_{status}"
            return {
                "source_url": url,
                "discovered_via": source,
                "status_code": status,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "cached": cached,
                "kept": parsed is not None,
                "drop_reason": reason,
                "row": parsed,
            }
        except Exception as exc:  # pragma: no cover - defensive scrape logging
            return {
                "source_url": url,
                "discovered_via": source,
                "status_code": None,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "cached": False,
                "kept": False,
                "drop_reason": f"exception_{type(exc).__name__}: {exc}",
                "row": None,
            }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_one, item) for item in candidates]
        for future in tqdm(as_completed(futures), total=len(futures), desc="scrape"):
            result = future.result()
            manifest.append({k: v for k, v in result.items() if k != "row"})
            if result["row"]:
                rows.append(result["row"])
            if len(rows) >= target:
                # Already submitted futures may complete; we still keep their manifest.
                pass

    manifest.sort(key=lambda r: (r["discovered_via"], r["source_url"]))
    rows.sort(key=lambda r: (r["date"], r["title"], r["source_url"]))
    write_jsonl(ROOT / "data" / "raw_manifest.jsonl", manifest)
    write_jsonl(ROOT / "data" / "scraped_rows_unfiltered.jsonl", rows)
    (ROOT / "data" / "discovery_note.txt").write_text(discovery_note + "\n", encoding="utf-8")
    return rows


def main() -> None:
    rows = scrape()
    print(f"Scraped parseable rows: {len(rows)}")


if __name__ == "__main__":
    main()
