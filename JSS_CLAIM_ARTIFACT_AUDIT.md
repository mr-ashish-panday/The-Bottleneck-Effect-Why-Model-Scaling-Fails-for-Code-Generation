# JSS Claim-to-Artifact Audit

Checks passed: 32
Checks failed: 0

| Status | Check | Evidence |
|---|---|---|
| PASS | Figure exists: figure10_cross_benchmark_map.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure10_cross_benchmark_map.pdf` |
| PASS | Figure exists: figure11_bootstrap_forest.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure11_bootstrap_forest.pdf` |
| PASS | Figure exists: figure12_codegen_ladder_benchmarks.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure12_codegen_ladder_benchmarks.pdf` |
| PASS | Figure exists: figure15_jss_robustness_controls.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure15_jss_robustness_controls.pdf` |
| PASS | Figure exists: figure8_codegen_ladder_followups.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure8_codegen_ladder_followups.pdf` |
| PASS | Figure exists: figure2_ablation_heatmap.png | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure2_ablation_heatmap.png` |
| PASS | Figure exists: figure9_ablation_depth_profiles.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure9_ablation_depth_profiles.pdf` |
| PASS | Figure exists: figure3_error_distribution.png | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure3_error_distribution.png` |
| PASS | Figure exists: figure4_activation_projection_real.png | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure4_activation_projection_real.png` |
| PASS | Figure exists: figure5_activation_steering_response.png | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure5_activation_steering_response.png` |
| PASS | Figure exists: figure6_steering_controls.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure6_steering_controls.pdf` |
| PASS | Figure exists: figure7_scaled_ablation_comparison.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure7_scaled_ablation_comparison.pdf` |
| PASS | Figure exists: figure13_strictness_cascade.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure13_strictness_cascade.pdf` |
| PASS | Figure exists: figure14_coverage_audit.pdf | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\outputs\figures\figure14_coverage_audit.pdf` |
| PASS | source package readable | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\submission_jss_20260512_135646\jss_source_package.zip entries=23 bad=None` |
| PASS | supplement package readable | `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\submission_jss_20260512_135646\jss_supplement_artifact_full.zip entries=398 bad=None` |
| PASS | GPT-2 Small HumanEval success 5.2% | `actual=5.152%` |
| PASS | GPT-2 Medium HumanEval success 4.8% | `actual=4.787%` |
| PASS | CodeGen HumanEval success 37.4% | `actual=37.421%` |
| PASS | MBPP GPT-2 variants 0.0% success | `GPT-2=0.0%, GPT-2 Medium=0.0%` |
| PASS | MBPP CodeGen success 7.39% | `actual=7.393%` |
| PASS | MBPP CodeGen pass@5 22.96% | `actual=22.960%` |
| PASS | CodeGen HumanEval ladder 31.60 -> 30.66 -> 39.00 | `{'CodeGen-NL': 31.597560975609756, 'CodeGen-Multi': 30.664634146341463, 'CodeGen-Mono': 39.0}` |
| PASS | CodeGen MBPP ladder 0.02 -> 1.23 -> 2.76 | `{'CodeGen-NL MBPP': 0.019455252918287938, 'CodeGen-Multi MBPP': 1.2256809338521402, 'CodeGen-Mono MBPP': 2.762645914396887}` |
| PASS | LiveCodeBench GPT-2 variants 0.0 pass@1/pass@5/pass@10 | `gpt2={'pass@1': 0.0, 'pass@5': 0.0, 'pass@10': 0.0}, medium={'pass@1': 0.0, 'pass@5': 0.0, 'pass@10': 0.0}` |
| PASS | LiveCodeBench CodeGen 0.02/0.10/0.20 | `{'pass@1': 0.01956947162426614, 'pass@5': 0.09784735812133072, 'pass@10': 0.19569471624266144}` |
| PASS | JSS prompt CodeGen signature 42.6% vs comment 71.2% | `signature=42.62, comment=71.16` |
| PASS | MBPP low-temp sample-success vs coverage tradeoff | `CodeGen samples 371 > 220, coverage 79 < 88; Qwen samples 741 > 488, coverage 128 < 141` |
| PASS | HumanEval+ first-five ordering GPT-2=0, Medium=0, CodeGen ~=2.1 | `gpt2=0.00, medium=0.00, codegen=2.07` |
| PASS | GPT-2 syntax profile | `indentation=22.0, bracket_mismatch=6.5, quote_mismatch=7.9, keyword_error=8.6, colon_missing=0.3, other=54.4` |
| PASS | GPT-2 Medium syntax profile | `indentation=28.7, bracket_mismatch=3.0, quote_mismatch=8.0, keyword_error=8.1, colon_missing=0.3, other=51.7` |
| PASS | CodeGen syntax profile | `indentation=7.4, bracket_mismatch=23.1, quote_mismatch=5.3, keyword_error=2.4, colon_missing=4.7, other=57.0` |

All automated audit checks passed.