#!/usr/bin/env python3
"""Run the full JSS pre-upload validation gate."""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "JSS_PREFLIGHT_REPORT.md"

BAD_LOG_PATTERNS = [
    "undefined references",
    "undefined citations",
    "Rerun to get cross-references",
    "There were undefined",
    "Overfull",
    "LaTeX Warning",
]

BAD_TEXT_PATTERNS = [
    r"BEGIN (RSA|OPENSSH|PRIVATE)",
    r"AKIA[0-9A-Z]{16}",
    r"hf_[A-Za-z0-9_\-]{20,}",
    r"ssh\.lightning\.ai",
    r"s_01k",
    r"lightning_rsa",
    r"github\.com/mr-ashish-panday/The-Bottleneck",
    r"publicartifacturl",
    r"repository includes",
]

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def run_command(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode == 0, proc.stdout.strip()


def read_zip_text(zip_path: Path, member: str) -> str:
    normalized_member = member.replace("\\", "/").lstrip("./")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.replace("\\", "/").lstrip("./") == normalized_member:
                with zf.open(name) as f:
                    return f.read().decode("utf-8")
    raise KeyError(f"{member!r} not found in {zip_path}")


def check_archives() -> tuple[bool, str]:
    source = ROOT / "submission_jss_20260512_135646" / "jss_source_package.zip"
    supplement = ROOT / "submission_jss_20260512_135646" / "jss_supplement_artifact_full.zip"
    expected = [(source, 23), (supplement, 398)]
    lines: list[str] = []
    ok = True

    for path, expected_count in expected:
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                names = [info.filename for info in zf.infolist()]
                count = len(names)
            noise_hits = [
                name
                for name in names
                if "__pycache__" in name or name.endswith((".pyc", ".pyo", ".log"))
            ]
            path_ok = bad is None and count == expected_count and not noise_hits
            ok = ok and path_ok
            lines.append(
                f"{path.name}: entries={count}, expected={expected_count}, "
                f"bad={bad}, noise_hits={len(noise_hits)}"
            )
        except Exception as exc:
            ok = False
            lines.append(f"{path.name}: error={exc}")

    return ok, "\n".join(lines)


def check_latex_log() -> tuple[bool, str]:
    log = ROOT / "bottleneck.log"
    text = log.read_text(encoding="utf-8", errors="ignore")
    hits = [pattern for pattern in BAD_LOG_PATTERNS if pattern in text]
    if hits:
        return False, "bad log patterns: " + ", ".join(hits)
    return True, "no unresolved refs/cites, rerun warnings, LaTeX warnings, or overfull boxes"


def check_data_availability() -> tuple[bool, str]:
    source_zip = ROOT / "submission_jss_20260512_135646" / "jss_source_package.zip"
    supplement_zip = ROOT / "submission_jss_20260512_135646" / "jss_supplement_artifact_full.zip"
    sources = {
        "local bottleneck.tex": (ROOT / "bottleneck.tex").read_text(encoding="utf-8"),
        "source zip bottleneck.tex": read_zip_text(source_zip, "bottleneck.tex"),
        "supplement zip bottleneck.tex": read_zip_text(supplement_zip, "bottleneck.tex"),
    }
    ok = True
    lines: list[str] = []
    for name, text in sources.items():
        has_curated = "curated supplementary artifact archive" in text
        has_stale = "publicartifacturl" in text or "project artifact repository" in text
        ok = ok and has_curated and not has_stale
        lines.append(f"{name}: curated={has_curated}, stale_repo_reference={has_stale}")
    return ok, "\n".join(lines)


def scan_text_files() -> tuple[bool, str]:
    targets = [
        ROOT / "bottleneck.tex",
        ROOT / "JSS_PORTAL_METADATA.md",
        ROOT / "JSS_PORTAL_UPLOAD_RUNBOOK.md",
        ROOT / "JSS_FINAL_SUBMISSION_AUDIT.md",
        ROOT / "JSS_MANUAL_CONFIRMATION_FORM.md",
        ROOT / "JSS_SUBMISSION_CHECKLIST.md",
        ROOT / "JSS_SUGGESTED_REVIEWERS.md",
        ROOT / "LIGHTNING_COST_CONTROL_HANDOFF.md",
        ROOT / "JSS_UPLOAD_FILE_MANIFEST.csv",
    ]
    compiled = [re.compile(pattern) for pattern in BAD_TEXT_PATTERNS]
    hits: list[str] = []

    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for regex in compiled:
            if regex.search(text):
                hits.append(f"{path.name}: {regex.pattern}")

    for zip_name in ["jss_source_package.zip", "jss_supplement_artifact_full.zip"]:
        zip_path = ROOT / "submission_jss_20260512_135646" / zip_name
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                suffix = Path(info.filename).suffix.lower()
                if suffix not in TEXT_SUFFIXES:
                    continue
                try:
                    text = zf.read(info).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                for regex in compiled:
                    if regex.search(text):
                        hits.append(f"{zip_name}:{info.filename}: {regex.pattern}")

    if hits:
        return False, "\n".join(hits)
    return True, "no sensitive strings or stale artifact-repository references found"


def write_report(rows: list[tuple[str, bool, str]]) -> None:
    passed = sum(1 for _, ok, _ in rows if ok)
    lines = [
        "# JSS Preflight Report",
        "",
        f"Checks passed: {passed}",
        f"Checks failed: {len(rows) - passed}",
        "",
        "| Status | Check | Evidence |",
        "|---|---|---|",
    ]
    for name, ok, evidence in rows:
        status = "PASS" if ok else "FAIL"
        safe_evidence = evidence.replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {status} | {name} | `{safe_evidence}` |")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows: list[tuple[str, bool, str]] = []

    for name, command in [
        ("claim-to-artifact audit", [sys.executable, "scripts/audit_jss_submission_artifacts.py"]),
        ("upload manifest verifier", [sys.executable, "scripts/verify_jss_upload_manifest.py"]),
    ]:
        ok, evidence = run_command(command)
        rows.append((name, ok, evidence))

    for name, check in [
        ("archive readability and counts", check_archives),
        ("latex log health", check_latex_log),
        ("data availability wording", check_data_availability),
        ("sensitive/stale text scan", scan_text_files),
    ]:
        ok, evidence = check()
        rows.append((name, ok, evidence))

    write_report(rows)
    failed = [name for name, ok, _ in rows if not ok]
    print(f"jss_preflight_checks={len(rows)} failed={len(failed)}")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        print(f"report={REPORT}")
        return 1
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
