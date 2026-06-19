#!/usr/bin/env python3
"""Audit whether the JSS submission goal is actually complete.

This script is intentionally lightweight: it checks existing files and proof
artifacts only. It does not rebuild LaTeX, regenerate archives, or run GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "JSS_UPLOAD_FILE_MANIFEST.csv"
TRACKER = ROOT / "JSS_POST_SUBMISSION_TRACKER.md"
AUDIT = ROOT / "JSS_FINAL_SUBMISSION_AUDIT.md"
TEX = ROOT / "bottleneck.tex"
MANUAL_FORM = ROOT / "JSS_MANUAL_CONFIRMATION_FORM.md"
PROOF_DIR = ROOT / "submission_jss_20260512_135646" / "post_submission_proof"
PROOF_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".eml", ".html", ".htm"}
REQUIRED_PROOF_LABELS = {
    "submission_confirmation": "Submission confirmation screenshot/PDF",
    "confirmation_email": "Confirmation email",
    "uploaded_file_list": "Final portal uploaded-file list",
    "portal_pdf_proof": "Portal-generated PDF proof",
}


@dataclass(frozen=True)
class Check:
    requirement: str
    evidence: str
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the JSS submission goal is actually complete."
    )
    parser.add_argument(
        "--report",
        help="Optional Markdown report path to write the current audit results.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def has_filled_tracker_field(text: str, field: str) -> bool:
    match = re.search(rf"^- {re.escape(field)}:[ \t]*(?P<value>[^\r\n]*)$", text, flags=re.MULTILINE)
    if not match:
        return False
    value = match.group("value").strip()
    return bool(value and value.lower() not in {"todo", "tbd", "n/a", "none"})


def valid_proof_file(path: Path) -> bool:
    proof_root = PROOF_DIR.resolve()
    try:
        path.resolve().relative_to(proof_root)
    except ValueError:
        return False
    return (
        path.exists()
        and path.is_file()
        and path.name.lower() != "readme.md"
        and path.suffix.lower() in PROOF_SUFFIXES
        and path.stat().st_size > 0
    )


def recorded_proofs(text: str) -> dict[str, Path]:
    proofs: dict[str, Path] = {}
    pattern = re.compile(r"^- (?P<label>[a-z_]+): `(?P<path>[^`]+)`$", flags=re.MULTILINE)
    for match in pattern.finditer(text):
        label = match.group("label")
        raw_path = Path(match.group("path"))
        proofs[label] = raw_path if raw_path.is_absolute() else ROOT / raw_path
    return proofs


def required_proof_files_are_distinct(proofs: dict[str, Path]) -> bool:
    required_paths: list[Path] = []
    for label in REQUIRED_PROOF_LABELS:
        path = proofs.get(label)
        if not path or not valid_proof_file(path):
            return False
        required_paths.append(path.resolve())
    return len(required_paths) == len(set(required_paths))


def manifest_checks() -> list[Check]:
    if not MANIFEST.exists():
        return [Check("Upload manifest exists", str(MANIFEST), False)]

    checks: list[Check] = [Check("Upload manifest exists", str(MANIFEST), True)]
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row.get("required", "").lower() != "yes":
            continue
        path = ROOT / row["relative_path"]
        if not path.exists():
            checks.append(Check(f"Required upload file exists: {row['portal_role']}", str(path), False))
            continue
        size_ok = path.stat().st_size == int(row["bytes"])
        hash_ok = sha256(path) == row["sha256"].upper()
        checks.append(
            Check(
                f"Required upload file matches manifest: {row['portal_role']}",
                f"{row['relative_path']} size_ok={size_ok} sha256_ok={hash_ok}",
                size_ok and hash_ok,
            )
        )
    return checks


def required_upload_paths() -> list[str]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return [
            row["relative_path"]
            for row in csv.DictReader(handle)
            if row.get("required", "").lower() == "yes"
        ]


def upload_checked(text: str, relative_path: str) -> bool:
    escaped = re.escape(relative_path)
    pattern = rf"^- \[[xX]\] `{escaped}`$"
    return bool(re.search(pattern, text, flags=re.MULTILINE))


def manual_confirmation_counts(text: str) -> tuple[int, int]:
    required_unchecked = 0
    checked = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ["):
            continue
        if "if the portal allows it" in stripped:
            continue
        if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            checked += 1
        elif stripped.startswith("- [ ]"):
            required_unchecked += 1
    return checked, required_unchecked


def write_report(path_text: str, checks: list[Check], failed: list[Check]) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    lines = [
        "# JSS Completion Audit Report",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- Total checks: `{len(checks)}`",
        f"- Failed checks: `{len(failed)}`",
        f"- Completion status: `{'complete' if not failed else 'not_complete'}`",
        "",
        "## Checks",
        "",
        "| Status | Requirement | Evidence |",
        "|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        evidence = check.evidence.replace("|", "\\|")
        requirement = check.requirement.replace("|", "\\|")
        lines.append(f"| {status} | {requirement} | `{evidence}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    tracker_text = TRACKER.read_text(encoding="utf-8") if TRACKER.exists() else ""
    audit_text = AUDIT.read_text(encoding="utf-8") if AUDIT.exists() else ""
    tex_text = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    manual_text = MANUAL_FORM.read_text(encoding="utf-8") if MANUAL_FORM.exists() else ""
    saved_proof = recorded_proofs(tracker_text)
    manual_checked, manual_unchecked = manual_confirmation_counts(manual_text)

    checks: list[Check] = []
    checks.append(
        Check(
            "Target journal is Journal of Systems and Software",
            "bottleneck.tex contains journal declaration",
            "\\journal{Journal of Systems and Software}" in tex_text,
        )
    )
    checks.extend(manifest_checks())
    checks.append(
        Check(
            "Final audit explicitly says not to mark the proof bundle complete too early",
            "JSS_FINAL_SUBMISSION_AUDIT.md",
            (
                "Do Not Mark The Overall Goal Complete Yet" in audit_text
                or "Do Not Mark The Proof Bundle Complete Yet" in audit_text
            ),
        )
    )
    checks.append(
        Check(
            "Post-submission tracker exists",
            str(TRACKER),
            TRACKER.exists(),
        )
    )
    checks.append(
        Check(
            "Manual confirmation form exists",
            str(MANUAL_FORM),
            MANUAL_FORM.exists(),
        )
    )
    checks.append(
        Check(
            "All required manual confirmation boxes are checked",
            f"checked={manual_checked} unchecked_required={manual_unchecked}; optional portal-allowed artifacts ignored",
            MANUAL_FORM.exists() and manual_checked > 0 and manual_unchecked == 0,
        )
    )
    checks.append(
        Check(
            "Manuscript ID is recorded after portal submission",
            "JSS_POST_SUBMISSION_TRACKER.md field: Manuscript ID",
            has_filled_tracker_field(tracker_text, "Manuscript ID"),
        )
    )
    checks.append(
        Check(
            "Corresponding author email is recorded after portal submission",
            "JSS_POST_SUBMISSION_TRACKER.md field: Corresponding author email",
            has_filled_tracker_field(tracker_text, "Corresponding author email"),
        )
    )
    checks.append(
        Check(
            "Recorded proof section exists",
            "JSS_POST_SUBMISSION_TRACKER.md section: Recorded Proof Files",
            "## Recorded Proof Files" in tracker_text,
        )
    )
    for label, description in REQUIRED_PROOF_LABELS.items():
        path = saved_proof.get(label)
        checks.append(
            Check(
                f"{description} is recorded and saved",
                f"{label}: {path if path else 'missing'}; accepted suffixes={sorted(PROOF_SUFFIXES)}",
                bool(path and valid_proof_file(path)),
            )
        )
    checks.append(
        Check(
            "At least all required portal proof categories are present",
            ", ".join(REQUIRED_PROOF_LABELS),
            all(label in saved_proof and valid_proof_file(saved_proof[label]) for label in REQUIRED_PROOF_LABELS),
        )
    )
    checks.append(
        Check(
            "Required portal proof categories use distinct files",
            ", ".join(REQUIRED_PROOF_LABELS),
            required_proof_files_are_distinct(saved_proof),
        )
    )
    checks.append(
        Check(
            "Tracker status changed away from not submitted",
            "JSS_POST_SUBMISSION_TRACKER.md status line",
            "Status: not submitted yet." not in tracker_text,
        )
    )
    for relative_path in required_upload_paths():
        checks.append(
            Check(
                f"Required portal upload is checked in tracker: {relative_path}",
                "JSS_POST_SUBMISSION_TRACKER.md Final Uploaded Files",
                upload_checked(tracker_text, relative_path),
            )
        )
    required_paths = required_upload_paths()
    checks.append(
        Check(
            "All required portal uploads are checked in tracker",
            ", ".join(required_paths),
            bool(required_paths) and all(upload_checked(tracker_text, path) for path in required_paths),
        )
    )

    failed = [check for check in checks if not check.passed]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}: {check.requirement}")
        print(f"  evidence: {check.evidence}")
    print(f"jss_completion_checks={len(checks)} failed={len(failed)}")
    if args.report:
        write_report(args.report, checks, failed)
        print(f"report={Path(args.report)}")
    if failed:
        print("completion_status=not_complete")
        return 1
    print("completion_status=complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
