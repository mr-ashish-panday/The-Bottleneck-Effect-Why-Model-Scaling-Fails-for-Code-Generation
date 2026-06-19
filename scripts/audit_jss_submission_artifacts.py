#!/usr/bin/env python3
"""Audit JSS submission artifacts against manuscript-facing claims."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "JSS_CLAIM_ARTIFACT_AUDIT.md"
TMP_REPORT = REPORT.with_suffix(REPORT.suffix + ".tmp")


def load_json(path: str) -> object:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> float:
    return value * 100.0


def close(actual: float, expected: float, tol: float = 0.06) -> bool:
    return abs(actual - expected) <= tol


def add_check(checks: list[dict], name: str, ok: bool, evidence: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "evidence": evidence})


def model_by_substring(models: dict, text: str) -> dict:
    matches = [v for k, v in models.items() if text.lower() in k.lower()]
    if not matches:
        raise KeyError(f"No model key contains {text!r}; keys={list(models)}")
    return matches[0]


def summary_by_label(summaries: list[dict], text: str) -> dict:
    matches = [row for row in summaries if text in row["label"]]
    if not matches:
        raise KeyError(f"No summary label contains {text!r}")
    return matches[0]


def evalplus_status_rate(path: str, status_key: str = "plus_status") -> float:
    data = load_json(path)
    metrics = data["json_summaries"][0]["metrics"]
    values = [v for k, v in metrics.items() if k.endswith(status_key)]
    if not values:
        raise ValueError(f"No {status_key} entries in {path}")
    return sum(1 for v in values if v == "pass") / len(values)


def syntax_pct(path: str, category: str) -> float:
    data = load_json(path)
    return data["category_distribution"][category] / data["total_syntax_errors"] * 100.0


def audit_figures(checks: list[dict]) -> None:
    tex = (ROOT / "bottleneck.tex").read_text(encoding="utf-8")
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for fig in figures:
        path = ROOT / "outputs" / "figures" / fig
        add_check(checks, f"Figure exists: {fig}", path.exists(), str(path))


def audit_archives(checks: list[dict]) -> None:
    packages = sorted(ROOT.glob("submission_jss_*/jss_source_package.zip"))
    supplements = sorted(ROOT.glob("submission_jss_*/jss_supplement_artifact_full.zip"))
    for label, candidates, minimum_entries in [
        ("source package", packages, 20),
        ("supplement package", supplements, 398),
    ]:
        if not candidates:
            add_check(checks, f"{label} exists", False, "missing")
            continue
        path = candidates[-1]
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                count = len(zf.infolist())
            ok = bad is None and count >= minimum_entries
            evidence = f"{path} entries={count} bad={bad}"
        except Exception as exc:  # pragma: no cover - diagnostic path
            ok = False
            evidence = f"{path} zip error: {exc}"
        add_check(checks, f"{label} readable", ok, evidence)


def audit_numeric_claims(checks: list[dict]) -> None:
    human = load_json("outputs/tables/bootstrap_significance.json")
    models = human["models"]
    small = model_by_substring(models, "GPT-2 (124M)")
    medium = model_by_substring(models, "GPT-2 Medium")
    codegen = model_by_substring(models, "CodeGen")
    values = {
        "GPT-2 Small HumanEval success 5.2%": pct(small["metrics"]["success_rate"]["mean"]),
        "GPT-2 Medium HumanEval success 4.8%": pct(medium["metrics"]["success_rate"]["mean"]),
        "CodeGen HumanEval success 37.4%": pct(codegen["metrics"]["success_rate"]["mean"]),
    }
    expected = {
        "GPT-2 Small HumanEval success 5.2%": 5.2,
        "GPT-2 Medium HumanEval success 4.8%": 4.8,
        "CodeGen HumanEval success 37.4%": 37.4,
    }
    for name, actual in values.items():
        add_check(checks, name, close(actual, expected[name]), f"actual={actual:.3f}%")

    mbpp = load_json("outputs/tables/bootstrap_significance_mbpp_full.json")
    mbpp_models = mbpp["models"]
    gpt2_mbpp = model_by_substring(mbpp_models, "GPT-2 MBPP")
    medium_mbpp = model_by_substring(mbpp_models, "GPT-2 Medium")
    codegen_mbpp = model_by_substring(mbpp_models, "CodeGen")
    add_check(
        checks,
        "MBPP GPT-2 variants 0.0% success",
        pct(gpt2_mbpp["metrics"]["success_rate"]["mean"]) == 0.0
        and pct(medium_mbpp["metrics"]["success_rate"]["mean"]) == 0.0,
        "GPT-2=0.0%, GPT-2 Medium=0.0%",
    )
    add_check(
        checks,
        "MBPP CodeGen success 7.39%",
        close(pct(codegen_mbpp["metrics"]["success_rate"]["mean"]), 7.39),
        f"actual={pct(codegen_mbpp['metrics']['success_rate']['mean']):.3f}%",
    )
    add_check(
        checks,
        "MBPP CodeGen pass@5 22.96%",
        close(pct(codegen_mbpp["metrics"]["pass@5"]["mean"]), 22.96),
        f"actual={pct(codegen_mbpp['metrics']['pass@5']['mean']):.3f}%",
    )

    ladder = load_json("outputs/tables/codegen_ladder_summary.json")["models"]
    ladder_values = {row["label"]: row["success_pct"] for row in ladder}
    add_check(
        checks,
        "CodeGen HumanEval ladder 31.60 -> 30.66 -> 39.00",
        close(ladder_values["CodeGen-NL"], 31.60)
        and close(ladder_values["CodeGen-Multi"], 30.66)
        and close(ladder_values["CodeGen-Mono"], 39.00),
        str(ladder_values),
    )

    ladder_mbpp = load_json("outputs/tables/codegen_ladder_mbpp_summary.json")["models"]
    ladder_mbpp_values = {row["label"]: row["success_pct"] for row in ladder_mbpp}
    add_check(
        checks,
        "CodeGen MBPP ladder 0.02 -> 1.23 -> 2.76",
        close(ladder_mbpp_values["CodeGen-NL MBPP"], 0.02)
        and close(ladder_mbpp_values["CodeGen-Multi MBPP"], 1.23)
        and close(ladder_mbpp_values["CodeGen-Mono MBPP"], 2.76),
        str(ladder_mbpp_values),
    )

    lcb_codegen = load_json("outputs/tables/livecodebench_codegen_summary.json")
    lcb_gpt2 = load_json("outputs/tables/livecodebench_gpt2_summary.json")
    lcb_medium = load_json("outputs/tables/livecodebench_gpt2_medium_summary.json")
    codegen_metrics = lcb_codegen["json_summaries"][0]["metrics"]
    gpt2_metrics = lcb_gpt2["json_summaries"][0]["metrics"]
    medium_metrics = lcb_medium["json_summaries"][0]["metrics"]
    add_check(
        checks,
        "LiveCodeBench GPT-2 variants 0.0 pass@1/pass@5/pass@10",
        all(gpt2_metrics[k] == 0.0 and medium_metrics[k] == 0.0 for k in ["pass@1", "pass@5", "pass@10"]),
        f"gpt2={gpt2_metrics}, medium={medium_metrics}",
    )
    add_check(
        checks,
        "LiveCodeBench CodeGen 0.02/0.10/0.20",
        close(pct(codegen_metrics["pass@1"]), 0.02)
        and close(pct(codegen_metrics["pass@5"]), 0.10)
        and close(pct(codegen_metrics["pass@10"]), 0.20),
        str({k: pct(codegen_metrics[k]) for k in ["pass@1", "pass@5", "pass@10"]}),
    )

    prompt = load_json("outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json")["summaries"]
    codegen_sig = summary_by_label(prompt, "codegen_mono_350m__humaneval__standard__signature_only")
    codegen_comment = summary_by_label(prompt, "codegen_mono_350m__humaneval__standard__comment_plus_signature")
    add_check(
        checks,
        "JSS prompt CodeGen signature 42.6% vs comment 71.2%",
        close(pct(codegen_sig["success_sample_rate"]), 42.6)
        and close(pct(codegen_comment["success_sample_rate"]), 71.2),
        f"signature={pct(codegen_sig['success_sample_rate']):.2f}, comment={pct(codegen_comment['success_sample_rate']):.2f}",
    )

    mbpp_dec = load_json("outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json")["summaries"]
    cg_low = summary_by_label(mbpp_dec, "codegen_mono_350m__mbpp__low_temp")
    cg_std = summary_by_label(mbpp_dec, "codegen_mono_350m__mbpp__standard")
    q_low = summary_by_label(mbpp_dec, "qwen25_coder_05b__mbpp__low_temp")
    q_std = summary_by_label(mbpp_dec, "qwen25_coder_05b__mbpp__standard")
    ok = (
        cg_low["success_samples"] > cg_std["success_samples"]
        and cg_low["problems_with_success"] < cg_std["problems_with_success"]
        and q_low["success_samples"] > q_std["success_samples"]
        and q_low["problems_with_success"] < q_std["problems_with_success"]
    )
    add_check(
        checks,
        "MBPP low-temp sample-success vs coverage tradeoff",
        ok,
        (
            f"CodeGen samples {cg_low['success_samples']} > {cg_std['success_samples']}, "
            f"coverage {cg_low['problems_with_success']} < {cg_std['problems_with_success']}; "
            f"Qwen samples {q_low['success_samples']} > {q_std['success_samples']}, "
            f"coverage {q_low['problems_with_success']} < {q_std['problems_with_success']}"
        ),
    )

    codegen_plus = pct(evalplus_status_rate("outputs/tables/evalplus_codegen_humaneval_summary.json"))
    gpt2_plus = pct(evalplus_status_rate("outputs/tables/evalplus_gpt2_humaneval_summary.json"))
    medium_plus = pct(evalplus_status_rate("outputs/tables/evalplus_gpt2_medium_humaneval_summary.json"))
    add_check(
        checks,
        "HumanEval+ first-five ordering GPT-2=0, Medium=0, CodeGen ~=2.1",
        gpt2_plus == 0.0 and medium_plus == 0.0 and close(codegen_plus, 2.1),
        f"gpt2={gpt2_plus:.2f}, medium={medium_plus:.2f}, codegen={codegen_plus:.2f}",
    )

    syntax_expected = {
        "GPT-2 syntax profile": {
            "path": "data/results_gpt2/syntax_analysis.json",
            "indentation": 22.0,
            "bracket_mismatch": 6.5,
            "quote_mismatch": 7.9,
            "keyword_error": 8.6,
            "colon_missing": 0.3,
            "other": 54.4,
        },
        "GPT-2 Medium syntax profile": {
            "path": "data/results_gpt2_medium/syntax_analysis.json",
            "indentation": 28.7,
            "bracket_mismatch": 3.0,
            "quote_mismatch": 8.0,
            "keyword_error": 8.1,
            "colon_missing": 0.3,
            "other": 51.7,
        },
        "CodeGen syntax profile": {
            "path": "data/results_codegen/syntax_analysis.json",
            "indentation": 7.4,
            "bracket_mismatch": 23.1,
            "quote_mismatch": 5.3,
            "keyword_error": 2.4,
            "colon_missing": 4.7,
            "other": 57.0,
        },
    }
    for name, spec in syntax_expected.items():
        path = spec["path"]
        actual = {category: syntax_pct(path, category) for category in spec if category != "path"}
        ok = all(close(actual[category], expected) for category, expected in spec.items() if category != "path")
        evidence = ", ".join(f"{category}={actual[category]:.1f}" for category in actual)
        add_check(checks, name, ok, evidence)


def write_report(checks: list[dict]) -> None:
    ok_count = sum(1 for row in checks if row["ok"])
    fail_count = len(checks) - ok_count
    lines = [
        "# JSS Claim-to-Artifact Audit",
        "",
        f"Checks passed: {ok_count}",
        f"Checks failed: {fail_count}",
        "",
        "| Status | Check | Evidence |",
        "|---|---|---|",
    ]
    for row in checks:
        status = "PASS" if row["ok"] else "FAIL"
        evidence = row["evidence"].replace("|", "\\|")
        lines.append(f"| {status} | {row['name']} | `{evidence}` |")
    lines.append("")
    if fail_count:
        lines.append("Do not submit until failed checks are resolved or explicitly justified.")
    else:
        lines.append("All automated audit checks passed.")
    TMP_REPORT.write_text("\n".join(lines), encoding="utf-8")
    TMP_REPORT.replace(REPORT)


def main() -> int:
    checks: list[dict] = []
    audit_figures(checks)
    audit_archives(checks)
    audit_numeric_claims(checks)
    write_report(checks)
    failed = [row for row in checks if not row["ok"]]
    print(f"checks={len(checks)} passed={len(checks)-len(failed)} failed={len(failed)}")
    for row in failed:
        print(f"FAIL: {row['name']} -- {row['evidence']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
