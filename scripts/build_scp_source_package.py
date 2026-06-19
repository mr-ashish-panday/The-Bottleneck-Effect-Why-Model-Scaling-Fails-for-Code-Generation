#!/usr/bin/env python3
"""Build the SCP manuscript source-package ZIP."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission_scp_20260601"
OUT = SUBMISSION / "scp_source_package.zip"
TMP_OUT = OUT.with_suffix(OUT.suffix + ".tmp")
LISTING = SUBMISSION / "source_zip_listing.txt"
ZIP_DATE = (2026, 6, 1, 0, 0, 0)

SOURCE_FILES = [
    "bottleneck.pdf",
    "bottleneck.tex",
    "scp_cover_letter.txt",
    "scp_highlights.txt",
    "references.bib",
    "outputs/figures/figure10_cross_benchmark_map.pdf",
    "outputs/figures/figure11_bootstrap_forest.pdf",
    "outputs/figures/figure12_codegen_ladder_benchmarks.pdf",
    "outputs/figures/figure13_strictness_cascade.pdf",
    "outputs/figures/figure14_coverage_audit.pdf",
    "outputs/figures/figure15_targeted_robustness_controls.pdf",
    "outputs/figures/figure2_ablation_heatmap.png",
    "outputs/figures/figure3_error_distribution.png",
    "outputs/figures/figure4_activation_projection_real.png",
    "outputs/figures/figure5_activation_steering_response.png",
    "outputs/figures/figure6_steering_controls.pdf",
    "outputs/figures/figure7_scaled_ablation_comparison.pdf",
    "outputs/figures/figure8_codegen_ladder_followups.pdf",
    "outputs/figures/figure9_ablation_depth_profiles.pdf",
]

ZIP_ORDER = [
    "./",
    "./outputs/",
    "./outputs/figures/",
    "./bottleneck.pdf",
    "./bottleneck.tex",
    "./scp_cover_letter.txt",
    "./scp_highlights.txt",
    "./PACKAGE_MANIFEST.csv",
    "./references.bib",
    "./outputs/figures/figure10_cross_benchmark_map.pdf",
    "./outputs/figures/figure11_bootstrap_forest.pdf",
    "./outputs/figures/figure12_codegen_ladder_benchmarks.pdf",
    "./outputs/figures/figure13_strictness_cascade.pdf",
    "./outputs/figures/figure14_coverage_audit.pdf",
    "./outputs/figures/figure15_targeted_robustness_controls.pdf",
    "./outputs/figures/figure2_ablation_heatmap.png",
    "./outputs/figures/figure3_error_distribution.png",
    "./outputs/figures/figure4_activation_projection_real.png",
    "./outputs/figures/figure5_activation_steering_response.png",
    "./outputs/figures/figure6_steering_controls.pdf",
    "./outputs/figures/figure7_scaled_ablation_comparison.pdf",
    "./outputs/figures/figure8_codegen_ladder_followups.pdf",
    "./outputs/figures/figure9_ablation_depth_profiles.pdf",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def zip_info(name: str, is_dir: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o40755 if is_dir else 0o100644) << 16
    return info


def build_manifest(payloads: dict[str, bytes]) -> bytes:
    rows = []
    for rel in SOURCE_FILES:
        data = payloads[rel]
        rows.append({"RelativePath": rel, "Bytes": len(data), "SHA256": sha256_bytes(data)})

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["RelativePath", "Bytes", "SHA256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    SUBMISSION.mkdir(parents=True, exist_ok=True)

    missing = [rel for rel in SOURCE_FILES if not (ROOT / rel).is_file()]
    if missing:
        print("missing source package files:")
        for rel in missing:
            print(f"- {rel}")
        return 1

    payloads = {rel: (ROOT / rel).read_bytes() for rel in SOURCE_FILES}
    payloads["PACKAGE_MANIFEST.csv"] = build_manifest(payloads)

    with zipfile.ZipFile(TMP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ZIP_ORDER:
            is_dir = name.endswith("/")
            rel = name.removeprefix("./")
            data = b"" if is_dir else payloads[rel]
            zf.writestr(zip_info(name, is_dir=is_dir), data)
    TMP_OUT.replace(OUT)

    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        bad = zf.testzip()

    LISTING.write_text("\n".join(names) + "\n", encoding="utf-8")
    digest = sha256_bytes(OUT.read_bytes())
    print(f"source={OUT}")
    print(f"files={len(names)}")
    print(f"bad={bad}")
    print(f"bytes={OUT.stat().st_size}")
    print(f"sha256={digest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
