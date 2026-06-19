#!/usr/bin/env python3
"""Record a completed JSS submission from standard proof filenames.

This is a convenience wrapper for the post-portal step. It does not submit the
paper and it refuses to run unless the four required proof files already exist
with the standard names in the post-submission proof folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROOF_DIR = ROOT / "submission_jss_20260512_135646" / "post_submission_proof"
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".eml", ".html", ".htm"}
STANDARD_PROOFS = {
    "confirmation-proof": "01_submission_confirmation",
    "email-proof": "02_confirmation_email",
    "uploaded-file-list-proof": "03_uploaded_file_list",
    "portal-pdf-proof": "04_portal_pdf_proof",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record JSS proof using standard post_submission_proof filenames."
    )
    parser.add_argument("--manuscript-id", required=True, help="JSS/Editorial Manager manuscript ID.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass through to record_jss_submission.py when the tracker already has a manuscript ID.",
    )
    parser.add_argument(
        "--confirm-all-manual-gates",
        action="store_true",
        help=(
            "Confirm author metadata, reviewer conflicts, portal classifications, "
            "required uploads, portal PDF proof, and saved proof are all complete."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show detected proof files without writing.")
    return parser.parse_args()


def find_standard_proof(stem: str) -> Path:
    matches = [
        path
        for path in PROOF_DIR.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES and path.stat().st_size > 0
    ]
    if not matches:
        raise SystemExit(f"missing required proof file: {PROOF_DIR / (stem + '.*')}")
    if len(matches) > 1:
        formatted = "\n".join(f"- {path}" for path in sorted(matches))
        raise SystemExit(f"multiple proof files found for {stem}; keep exactly one:\n{formatted}")
    return matches[0]


def run_command(args: list[str]) -> None:
    print("+ " + " ".join(args))
    completed = subprocess.run(args, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    if not PROOF_DIR.exists():
        raise SystemExit(f"proof folder does not exist: {PROOF_DIR}")
    if not args.confirm_all_manual_gates and not args.dry_run:
        raise SystemExit("refusing to record: pass --confirm-all-manual-gates after the portal checks are truly complete")

    detected = {arg_name: find_standard_proof(stem) for arg_name, stem in STANDARD_PROOFS.items()}
    print("standard_proof_status=found")
    for arg_name, path in detected.items():
        print(f"{arg_name}={path.relative_to(ROOT).as_posix()}")
    if args.dry_run:
        print("dry_run=true")
        return 0

    record_args = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "record_jss_submission.py"),
        "--manuscript-id",
        args.manuscript_id,
    ]
    if args.force:
        record_args.append("--force")
    for arg_name, path in detected.items():
        record_args.extend([f"--{arg_name}", str(path.relative_to(ROOT))])
    run_command(record_args)

    run_command(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "record_jss_manual_confirmations.py"),
            "--confirm-author-metadata",
            "--confirm-no-reviewer-conflicts",
            "--confirm-portal-classifications",
            "--confirm-required-uploads",
            "--confirm-portal-proof",
            "--confirm-proof-saved",
        ]
    )

    run_command(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "audit_jss_completion.py"),
            "--report",
            "JSS_COMPLETION_AUDIT_REPORT.md",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
