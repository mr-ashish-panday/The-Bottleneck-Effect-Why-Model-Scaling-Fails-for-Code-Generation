#!/usr/bin/env python3
"""Record JSS portal submission proof after the manuscript is submitted.

This script refuses to run without a manuscript ID and the required saved proof
files in the post-submission proof folder. It is meant for after portal submit,
not for preparing or simulating a submission.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "JSS_POST_SUBMISSION_TRACKER.md"
AUDIT = ROOT / "JSS_FINAL_SUBMISSION_AUDIT.md"
CHECKLIST = ROOT / "JSS_SUBMISSION_CHECKLIST.md"
PROOF_DIR = ROOT / "submission_jss_20260512_135646" / "post_submission_proof"
DEFAULT_EMAIL = "ashishpanday9818@gmail.com"
DEFAULT_ARTICLE_TYPE = "Regular research article"
DEFAULT_TITLE = "The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation"
NPT = timezone(timedelta(hours=5, minutes=45))
PROOF_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".eml", ".html", ".htm"}
REQUIRED_PROOF_ARGS = {
    "submission_confirmation": "Saved submission confirmation screenshot/PDF.",
    "confirmation_email": "Saved confirmation email.",
    "uploaded_file_list": "Saved final portal uploaded-file list.",
    "portal_pdf_proof": "Saved portal-generated PDF proof.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a completed JSS submission only after proof exists."
    )
    parser.add_argument("--manuscript-id", required=True, help="JSS/Editorial Manager manuscript ID.")
    parser.add_argument("--confirmation-proof", required=True, help=REQUIRED_PROOF_ARGS["submission_confirmation"])
    parser.add_argument("--email-proof", required=True, help=REQUIRED_PROOF_ARGS["confirmation_email"])
    parser.add_argument("--uploaded-file-list-proof", required=True, help=REQUIRED_PROOF_ARGS["uploaded_file_list"])
    parser.add_argument("--portal-pdf-proof", required=True, help=REQUIRED_PROOF_ARGS["portal_pdf_proof"])
    parser.add_argument(
        "--ack-proof",
        action="append",
        default=[],
        help="Optional saved editor/system acknowledgement proof file.",
    )
    parser.add_argument("--email", default=DEFAULT_EMAIL, help="Corresponding author email.")
    parser.add_argument("--portal-account", default=DEFAULT_EMAIL, help="Portal account used.")
    parser.add_argument("--article-type", default=DEFAULT_ARTICLE_TYPE, help="Article type selected in portal.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Final submitted title.")
    parser.add_argument(
        "--submission-date",
        default=datetime.now(NPT).strftime("%Y-%m-%d"),
        help="Submission date, default is current Nepal date.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing recorded manuscript ID.")
    return parser.parse_args()


def resolve_proof(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else ROOT / raw
    path = path.resolve()
    proof_root = PROOF_DIR.resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"proof file does not exist: {path}")
    try:
        path.relative_to(proof_root)
    except ValueError as exc:
        raise SystemExit(f"proof file must be inside {proof_root}: {path}") from exc
    if path.name.lower() == "readme.md":
        raise SystemExit("README.md is not submission proof")
    if path.suffix.lower() not in PROOF_SUFFIXES:
        raise SystemExit(f"proof file must use one of {sorted(PROOF_SUFFIXES)}: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"proof file is empty: {path}")
    return path


def replace_field(text: str, field: str, value: str) -> str:
    pattern = rf"^- {re.escape(field)}:[^\r\n]*$"
    replacement = f"- {field}: {value}"
    next_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"tracker field not found: {field}")
    return next_text


def has_existing_manuscript_id(text: str) -> bool:
    match = re.search(r"^- Manuscript ID:[ \t]*(?P<value>[^\r\n]*)$", text, flags=re.MULTILINE)
    return bool(match and match.group("value").strip())


def mark_upload_checkboxes(text: str) -> str:
    upload_files = [
        "bottleneck.pdf",
        "submission_jss_20260512_135646/jss_source_package.zip",
        "jss_highlights.txt",
        "jss_cover_letter.txt",
        "submission_jss_20260512_135646/jss_supplement_artifact_full.zip",
    ]
    for filename in upload_files:
        text = text.replace(f"- [ ] `{filename}`", f"- [x] `{filename}`")
    return text


def append_recorded_proofs(text: str, proof_items: list[tuple[str, Path]]) -> str:
    section = "\n## Recorded Proof Files\n\n"
    section += "\n".join(
        f"- {label}: `{path.relative_to(ROOT).as_posix()}`" for label, path in proof_items
    )
    section += "\n"
    if "## Recorded Proof Files" in text:
        return re.sub(
            r"\n## Recorded Proof Files\n\n(?:- [a-z_]+: `.*`\n?)*",
            section,
            text,
            count=1,
        )
    return text.rstrip() + "\n" + section


def require_distinct_required_proofs(proof_items: list[tuple[str, Path]]) -> None:
    required = [(label, path.resolve()) for label, path in proof_items if label != "acknowledgement"]
    seen: dict[Path, str] = {}
    for label, path in required:
        if path in seen:
            raise SystemExit(
                f"required proof file reused for {seen[path]} and {label}: {path}"
            )
        seen[path] = label


def update_submission_checklist(
    text: str,
    manuscript_id: str,
    submission_date: str,
    proof_items: list[tuple[str, Path]],
) -> str:
    proof_lines = "".join(
        f"- {label}: `{path.relative_to(ROOT).as_posix()}`\n" for label, path in proof_items
    )
    section = (
        "## Submitted To JSS\n\n"
        f"- Submission date: `{submission_date}`\n"
        f"- Manuscript ID: `{manuscript_id}`\n"
        "- Proof files recorded by `scripts/record_jss_submission.py`:\n"
        f"{proof_lines}"
    )
    if "## Submitted To JSS" in text:
        return re.sub(
            r"## Submitted To JSS\n[\s\S]*?(?=\n## |\Z)",
            section,
            text,
            count=1,
        )
    marker = "\n## Do Not Reopen Without New Evidence"
    if marker in text:
        return text.replace(marker, "\n" + section + marker, 1)
    return text.rstrip() + "\n\n" + section + "\n"


def main() -> int:
    args = parse_args()
    proof_items = [
        ("submission_confirmation", resolve_proof(args.confirmation_proof)),
        ("confirmation_email", resolve_proof(args.email_proof)),
        ("uploaded_file_list", resolve_proof(args.uploaded_file_list_proof)),
        ("portal_pdf_proof", resolve_proof(args.portal_pdf_proof)),
    ]
    proof_items.extend(("acknowledgement", resolve_proof(item)) for item in args.ack_proof)
    require_distinct_required_proofs(proof_items)
    tracker_text = TRACKER.read_text(encoding="utf-8")
    if has_existing_manuscript_id(tracker_text) and not args.force:
        raise SystemExit("tracker already has a manuscript ID; pass --force to update it")

    tracker_text = tracker_text.replace(
        "Status: not submitted yet.",
        f"Status: submitted to JSS on {args.submission_date}; manuscript ID `{args.manuscript_id}`.",
        1,
    )
    tracker_text = replace_field(tracker_text, "Submission date", args.submission_date)
    tracker_text = replace_field(tracker_text, "Manuscript ID", args.manuscript_id)
    tracker_text = replace_field(tracker_text, "Portal account used", args.portal_account)
    tracker_text = replace_field(tracker_text, "Corresponding author email", args.email)
    tracker_text = replace_field(tracker_text, "Article type", args.article_type)
    tracker_text = replace_field(tracker_text, "Final title", args.title)
    tracker_text = mark_upload_checkboxes(tracker_text)
    tracker_text = append_recorded_proofs(tracker_text, proof_items)
    TRACKER.write_text(tracker_text, encoding="utf-8", newline="\n")

    audit_text = AUDIT.read_text(encoding="utf-8")
    record = (
        "\n## Recorded Portal Submission\n\n"
        f"- Submission date: `{args.submission_date}`\n"
        f"- Manuscript ID: `{args.manuscript_id}`\n"
        f"- Corresponding email: `{args.email}`\n"
        "- Proof files:\n"
        + "".join(
            f"  - {label}: `{path.relative_to(ROOT).as_posix()}`\n"
            for label, path in proof_items
        )
    )
    if "## Recorded Portal Submission" in audit_text:
        audit_text = re.sub(r"\n## Recorded Portal Submission\n[\s\S]*$", record, audit_text, count=1)
    else:
        audit_text = audit_text.rstrip() + "\n" + record
    AUDIT.write_text(audit_text, encoding="utf-8", newline="\n")

    checklist_text = CHECKLIST.read_text(encoding="utf-8")
    checklist_text = update_submission_checklist(
        checklist_text,
        args.manuscript_id,
        args.submission_date,
        proof_items,
    )
    CHECKLIST.write_text(checklist_text, encoding="utf-8", newline="\n")

    print(f"recorded_manuscript_id={args.manuscript_id}")
    for label, path in proof_items:
        print(f"recorded_proof_{label}={path.relative_to(ROOT).as_posix()}")
    print(f"updated_checklist={CHECKLIST.relative_to(ROOT).as_posix()}")
    print("next=python -B scripts/audit_jss_completion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
