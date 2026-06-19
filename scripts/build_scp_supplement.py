#!/usr/bin/env python3
"""Build the SCP reviewer-facing supplementary artifact ZIP."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission_scp_20260601"
OUT = SUBMISSION / "scp_supplement_artifact_full.zip"
TMP_OUT = OUT.with_suffix(OUT.suffix + ".tmp")
LISTING = SUBMISSION / "supplement_zip_listing.txt"
ZIP_DATE = (2026, 6, 1, 0, 0, 0)

ROOT_FILES = [
    "ARTIFACT_README.md",
    "SCP_ARTIFACT_README.md",
    "SCP_CLAIM_ARTIFACT_AUDIT.md",
    "SCP_PORTAL_METADATA.md",
    "README.md",
    "bottleneck.pdf",
    "bottleneck.tex",
    "config.yaml",
    "scp_cover_letter.txt",
    "scp_highlights.txt",
    "references.bib",
    "requirements-lightning.txt",
    "requirements.txt",
]

ROOT_GLOBS = [
    "config_*.yaml",
]

DIRS = [
    "configs",
    "data",
    "outputs",
    "scripts",
    "src",
]

EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
}

EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".pyc",
    ".pyo",
}

EXCLUDED_DIR_PREFIXES = {
    "outputs/pdf_checks",
}

EXCLUDED_REL_PATHS = {
    "scripts/RUN_FORCE_CODEX_CHAT_RESTORE_AS_ADMIN.cmd",
    "scripts/audit_jss_completion.py",
    "scripts/build_isse_source_package.py",
    "scripts/build_isse_supplement.py",
    "scripts/build_jss_source_package.py",
    "scripts/build_jss_supplement.py",
    "scripts/check_lightning_status.ps1",
    "scripts/create_figure15_jss_robustness_controls.py",
    "scripts/force_codex_chat_interface_restore.ps1",
    "scripts/jss_submission_status.py",
    "scripts/open_jss_final_submit_focus.ps1",
    "scripts/open_jss_submission_workspace.ps1",
    "scripts/record_jss_manual_confirmations.py",
    "scripts/record_jss_standard_proof.py",
    "scripts/record_jss_submission.py",
    "scripts/run_jss_preflight.py",
    "scripts/setup_lightning_ai.sh",
    "scripts/stop_lightning_keepalive_after_jss_submit.ps1",
    "scripts/verify_jss_upload_manifest.py",
    "outputs/figures/figure15_jss_robustness_controls.pdf",
    "outputs/figures/figure15_jss_robustness_controls.png",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDED_REL_PATHS:
        return True
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in EXCLUDED_DIR_PREFIXES)


def iter_files() -> list[Path]:
    files: set[Path] = set()

    for rel in ROOT_FILES:
        path = ROOT / rel
        if path.exists() and path.is_file() and not is_excluded(path):
            files.add(path)

    for pattern in ROOT_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and not is_excluded(path):
                files.add(path)

    for dirname in DIRS:
        root = ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not is_excluded(path):
                files.add(path)

    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> int:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    files = iter_files()

    manifest_rows: list[dict[str, str | int]] = []
    payloads: list[tuple[str, bytes]] = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        payloads.append((rel, data))
        manifest_rows.append({"RelativePath": rel, "Bytes": len(data), "SHA256": sha256_bytes(data)})

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["RelativePath", "Bytes", "SHA256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest_rows)
    manifest_text = buffer.getvalue().encode("utf-8")

    with zipfile.ZipFile(TMP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel, data in payloads:
            zf.writestr(zip_info(rel), data)
        zf.writestr(zip_info("PACKAGE_MANIFEST.csv"), manifest_text)
    TMP_OUT.replace(OUT)

    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    LISTING.write_text("\n".join(names) + "\n", encoding="utf-8")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest().upper()
    print(f"supplement={OUT}")
    print(f"files={len(names)}")
    print(f"bad={bad}")
    print(f"bytes={OUT.stat().st_size}")
    print(f"sha256={digest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
