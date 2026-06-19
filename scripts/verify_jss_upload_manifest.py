#!/usr/bin/env python3
"""Verify the exact JSS portal upload files against the manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "JSS_UPLOAD_FILE_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    failures: list[str] = []
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))

    for row in rows:
        rel = row["relative_path"]
        path = ROOT / rel
        role = row["portal_role"]
        required = row["required"].lower() == "yes"

        if not path.exists():
            status = "missing required" if required else "missing optional"
            failures.append(f"{role}: {status}: {rel}")
            continue

        actual_size = path.stat().st_size
        expected_size = int(row["bytes"])
        if actual_size != expected_size:
            failures.append(f"{role}: size mismatch {rel}: actual={actual_size} expected={expected_size}")

        actual_hash = sha256(path)
        expected_hash = row["sha256"].upper()
        if actual_hash != expected_hash:
            failures.append(f"{role}: sha256 mismatch {rel}: actual={actual_hash} expected={expected_hash}")

    print(f"upload_manifest_checks={len(rows)} failed={len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
