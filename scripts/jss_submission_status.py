#!/usr/bin/env python3
"""Print the current JSS submission status without rebuilding artifacts."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "JSS_UPLOAD_FILE_MANIFEST.csv"
PROOF_DIR = ROOT / "submission_jss_20260512_135646" / "post_submission_proof"
SUPPORT_ESCALATION = ROOT / "JSS_PORTAL_SUPPORT_ESCALATION.md"


def c_drive_free_gib() -> float | None:
    try:
        return shutil.disk_usage("C:\\").free / (1024**3)
    except Exception:
        return None


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    print("JSS submission status: submitted")
    print("Manuscript ID: JSSOFTWARE-D-26-01113")
    print("Submission date: 2026-05-24")
    print("Target: Journal of Systems and Software")
    print("Official page: https://www.sciencedirect.com/journal/journal-of-systems-and-software")
    print("Editorial Manager: https://www.editorialmanager.com/jssoftware/default.aspx")
    print("Submit route: https://www.editorialmanager.com/jssoftware/submit_manuscript.asp")
    print("")
    print("Upload files:")
    for row in rows:
        required = "required" if row["required"].lower() == "yes" else "optional"
        print(f"- {row['portal_role']} ({required}): {row['relative_path']}")
        print(f"  bytes={row['bytes']} sha256={row['sha256']}")
    print("")
    print("Pre-submit gates already passed on the submitted package:")
    print("- python scripts/verify_jss_upload_manifest.py")
    print("- python scripts/run_jss_preflight.py")
    print("- Do not rebuild packages unless one of these gates fails.")
    free_gib = c_drive_free_gib()
    if free_gib is not None:
            print(f"- Current C: free space: {free_gib:.2f} GiB; avoid rebuilds unless a concrete gate fails.")
    print("")
    print("Remaining proof closeout:")
    print("- Save confirmation screen/PDF if not already saved.")
    print("- Reconstructed confirmation-email text proof is saved; replace with raw .eml later if available.")
    print("- Save final uploaded-file list if not already saved.")
    print("- Keep portal-generated PDF proof in the proof folder.")
    print("")
    proof_files = []
    if PROOF_DIR.exists():
        proof_files = sorted(
            path.name for path in PROOF_DIR.iterdir() if path.is_file() and path.name.lower() != "readme.md"
        )
    print("Proof folder:")
    print(f"- {PROOF_DIR.relative_to(ROOT)}")
    if proof_files:
        for name in proof_files:
            print(f"- proof_present: {name}")
    else:
        print("- no confirmation screenshot/email, portal proof, or uploaded-file list found")
    print("")
    print("Portal warning fallback:")
    print(f"- {SUPPORT_ESCALATION.name}")
    print("- Use only if the authenticated author workflow is blocked by the development warning.")
    print("")
    print("Manual confirmation recorder:")
    print("- python -B scripts/record_jss_manual_confirmations.py --confirm-author-metadata --confirm-no-reviewer-conflicts --confirm-portal-classifications --confirm-required-uploads --confirm-portal-proof --confirm-proof-saved")
    print("")
    print("Completion audit:")
    print("- python -B scripts/audit_jss_completion.py")
    print("- python -B scripts/audit_jss_completion.py --report JSS_COMPLETION_AUDIT_REPORT.md")
    print("- Expected until all proof files are saved: completion_status=not_complete")
    print("")
    print("Proof recorder after proof files are saved:")
    print("- python -B scripts/record_jss_submission.py --manuscript-id JSSOFTWARE-D-26-01113 --force --confirmation-proof <path> --email-proof <path> --uploaded-file-list-proof <path> --portal-pdf-proof <path>")
    print("")
    print("Lightning status helper:")
    print("- $env:LIGHTNING_SSH_TARGET = '<current Lightning SSH target>'")
    print("- .\\scripts\\check_lightning_status.ps1")
    print("")
    print("Post-submit keepalive stop helper:")
    print("- .\\scripts\\stop_lightning_keepalive_after_jss_submit.ps1 -ConfirmStop")
    print("- Refuses to run until python -B scripts/audit_jss_completion.py passes.")
    print("")
    print("Open submission workspace/status links:")
    print("- .\\scripts\\open_jss_submission_workspace.ps1")
    print("")
    print("After proof files are saved:")
    print("- Run scripts/record_jss_submission.py with the saved proof files and --force.")
    print("- Run scripts/record_jss_manual_confirmations.py with all six explicit confirmation flags.")
    print("- Run scripts/audit_jss_completion.py --report JSS_COMPLETION_AUDIT_REPORT.md and require it to pass.")
    print("- Stop the paid Lightning instance unless a specific new run is approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
