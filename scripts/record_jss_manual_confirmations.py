#!/usr/bin/env python3
"""Record explicit human confirmations in JSS_MANUAL_CONFIRMATION_FORM.md.

This script is guarded on purpose. It only checks required boxes when every
confirmation category is explicitly acknowledged by command-line flag. Optional
artifact boxes containing "if the portal allows it" are left unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORM = ROOT / "JSS_MANUAL_CONFIRMATION_FORM.md"
NPT = timezone(timedelta(hours=5, minutes=45))


REQUIRED_FLAGS = {
    "confirm_author_metadata": "author metadata is correct",
    "confirm_no_reviewer_conflicts": "no suggested-reviewer conflicts exist",
    "confirm_portal_classifications": "portal classifications were matched",
    "confirm_required_uploads": "required files were attached in the portal",
    "confirm_portal_proof": "portal-generated proof was inspected",
    "confirm_proof_saved": "post-submit proof files were saved",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check required JSS manual-confirmation boxes only after explicit confirmation."
    )
    for flag, help_text in REQUIRED_FLAGS.items():
        parser.add_argument(f"--{flag.replace('_', '-')}", action="store_true", help=f"Confirm {help_text}.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    return parser.parse_args()


def required_flags_present(args: argparse.Namespace) -> list[str]:
    return [flag for flag in REQUIRED_FLAGS if getattr(args, flag)]


def mark_required_boxes(text: str) -> tuple[str, int, int]:
    changed = 0
    optional_skipped = 0
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            if "if the portal allows it" in stripped:
                optional_skipped += 1
                lines.append(line)
                continue
            line = line.replace("- [ ]", "- [x]", 1)
            changed += 1
        lines.append(line)
    return "\n".join(lines) + "\n", changed, optional_skipped


def append_record(text: str, args: argparse.Namespace, changed: int, optional_skipped: int) -> str:
    timestamp = datetime.now(NPT).strftime("%Y-%m-%d %H:%M:%S %Z")
    confirmed = ", ".join(REQUIRED_FLAGS[flag] for flag in REQUIRED_FLAGS if getattr(args, flag))
    section = (
        "\n## Recorded Manual Confirmation\n\n"
        f"- Recorded at: `{timestamp}`\n"
        f"- Required boxes checked by script: `{changed}`\n"
        f"- Optional portal-allowed boxes left unchanged: `{optional_skipped}`\n"
        f"- Confirmed categories: {confirmed}\n"
    )
    if "## Recorded Manual Confirmation" in text:
        head = text.split("## Recorded Manual Confirmation", 1)[0].rstrip()
        return head + section
    return text.rstrip() + "\n" + section


def main() -> int:
    args = parse_args()
    present = required_flags_present(args)
    missing = [flag for flag in REQUIRED_FLAGS if flag not in present]
    if missing:
        print("manual_confirmation_status=refused")
        for flag in missing:
            print(f"missing_flag=--{flag.replace('_', '-')}")
        return 1

    text = FORM.read_text(encoding="utf-8")
    updated, changed, optional_skipped = mark_required_boxes(text)
    updated = append_record(updated, args, changed, optional_skipped)

    print(f"manual_confirmation_status={'dry_run' if args.dry_run else 'recorded'}")
    print(f"required_boxes_to_check={changed}")
    print(f"optional_boxes_left_unchanged={optional_skipped}")
    if not args.dry_run:
        FORM.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
