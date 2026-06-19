# ISSE Claim-to-Artifact Audit

Checks passed: 32
Checks failed: 0

This audit records the same evidence gate used for the prior submission package, with the target-specific packaging renamed for the ISSE retarget. The automated checks verify that manuscript-facing figures, source packages, supplement packages, benchmark summaries, syntax profiles, and robustness-control values are backed by local artifacts.

| Status | Check | Evidence |
|---|---|---|
| PASS | Figure exists: figure10_cross_benchmark_map.pdf | `outputs/figures/figure10_cross_benchmark_map.pdf` |
| PASS | Figure exists: figure11_bootstrap_forest.pdf | `outputs/figures/figure11_bootstrap_forest.pdf` |
| PASS | Figure exists: figure12_codegen_ladder_benchmarks.pdf | `outputs/figures/figure12_codegen_ladder_benchmarks.pdf` |
| PASS | Figure exists: figure15_targeted_robustness_controls.pdf | `outputs/figures/figure15_targeted_robustness_controls.pdf` |
| PASS | Figure exists: figure8_codegen_ladder_followups.pdf | `outputs/figures/figure8_codegen_ladder_followups.pdf` |
| PASS | Figure exists: figure2_ablation_heatmap.png | `outputs/figures/figure2_ablation_heatmap.png` |
| PASS | Figure exists: figure9_ablation_depth_profiles.pdf | `outputs/figures/figure9_ablation_depth_profiles.pdf` |
| PASS | Figure exists: figure3_error_distribution.png | `outputs/figures/figure3_error_distribution.png` |
| PASS | Figure exists: figure4_activation_projection_real.png | `outputs/figures/figure4_activation_projection_real.png` |
| PASS | Figure exists: figure5_activation_steering_response.png | `outputs/figures/figure5_activation_steering_response.png` |
| PASS | Figure exists: figure6_steering_controls.pdf | `outputs/figures/figure6_steering_controls.pdf` |
| PASS | Figure exists: figure7_scaled_ablation_comparison.pdf | `outputs/figures/figure7_scaled_ablation_comparison.pdf` |
| PASS | Figure exists: figure13_strictness_cascade.pdf | `outputs/figures/figure13_strictness_cascade.pdf` |
| PASS | Figure exists: figure14_coverage_audit.pdf | `outputs/figures/figure14_coverage_audit.pdf` |
| PASS | GPT-2 Small HumanEval success 5.2% | `actual=5.152%` |
| PASS | GPT-2 Medium HumanEval success 4.8% | `actual=4.787%` |
| PASS | CodeGen HumanEval success 37.4% | `actual=37.421%` |
| PASS | MBPP GPT-2 variants 0.0% success | `GPT-2=0.0%, GPT-2 Medium=0.0%` |
| PASS | MBPP CodeGen success 7.39% | `actual=7.393%` |
| PASS | MBPP CodeGen pass@5 22.96% | `actual=22.960%` |
| PASS | CodeGen HumanEval ladder 31.60 -> 30.66 -> 39.00 | `CodeGen-NL=31.60, CodeGen-Multi=30.66, CodeGen-Mono=39.00` |
| PASS | CodeGen MBPP ladder 0.02 -> 1.23 -> 2.76 | `CodeGen-NL=0.02, CodeGen-Multi=1.23, CodeGen-Mono=2.76` |
| PASS | LiveCodeBench GPT-2 variants 0.0 pass@1/pass@5/pass@10 | `GPT-2=0.0, GPT-2 Medium=0.0` |
| PASS | LiveCodeBench CodeGen 0.02/0.10/0.20 | `pass@1=0.0196, pass@5=0.0978, pass@10=0.1957` |
| PASS | Prompt robustness CodeGen signature 42.6% vs comment 71.2% | `signature=42.62, comment=71.16` |
| PASS | MBPP low-temperature sample-success vs coverage tradeoff | `CodeGen samples 371 > 220, coverage 79 < 88; Qwen samples 741 > 488, coverage 128 < 141` |
| PASS | HumanEval+ first-five ordering GPT-2=0, Medium=0, CodeGen ~=2.1 | `GPT-2=0.00, GPT-2 Medium=0.00, CodeGen=2.07` |
| PASS | GPT-2 syntax profile | `indentation=22.0, bracket_mismatch=6.5, quote_mismatch=7.9, keyword_error=8.6, colon_missing=0.3, other=54.4` |
| PASS | GPT-2 Medium syntax profile | `indentation=28.7, bracket_mismatch=3.0, quote_mismatch=8.0, keyword_error=8.1, colon_missing=0.3, other=51.7` |
| PASS | CodeGen syntax profile | `indentation=7.4, bracket_mismatch=23.1, quote_mismatch=5.3, keyword_error=2.4, colon_missing=4.7, other=57.0` |
| PASS | Manuscript retargeted to ISSE | `bottleneck.tex` uses `\journal{Innovations in Systems and Software Engineering}` |
| PASS | ISSE-facing portal metadata exists | `ISSE_PORTAL_METADATA.md` |

All automated and manual audit checks recorded here passed for the current evidence package.
