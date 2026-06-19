# CODEX WORKING MEMORY

Purpose: short factual memory for ongoing Codex collaboration on this paper repo.

## Stable project state

- Repo: `C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation`
- Main paper source: `bottleneck.tex`
- Main compiled PDF: `bottleneck.pdf`
- Current focus: make the paper submission-ready through evidence alignment, figures, captions, layout, and discussion tightening.

## Verified manuscript/evidence state

- Latest synced repo includes repaired CodeGen ladder artifacts and updated paper/tooling.
- Main CodeGen HumanEval artifact still has no saved completions for `HumanEval/129`, so main CodeGen HumanEval uses `16,300` scored completions even though the bootstrap summary still carries `164` task IDs.
- CodeGen ladder HumanEval is now repaired and full coverage:
  - `CodeGen-NL`: `31.60%`
  - `CodeGen-Multi`: `30.66%`
  - `CodeGen-Mono`: `39.00%`
- MBPP ladder remains monotone: `0.02% -> 1.23% -> 2.76%`
- Steering provenance is synced locally:
  - `20` random controls
  - control mean `6.2%`
  - random max `14.0%`
  - empirical `p <= 0.095`

## Prior collaboration decisions

- We agreed the paper should not be finalized until evidence-repair issues were addressed first.
- We agreed to avoid making new figures from stale or missing logs.
- We added and kept the CodeGen ladder scaled-followup story in the manuscript:
  - NL layer 11 is useful evidence.
  - Mono layer 13 is useful evidence.
  - Multi layer 7 is anomalous and should not be sold as clean mechanistic support.
- We previously added safe figure scripts and figures before the repaired sync, then re-synced the repo and re-aligned the manuscript to the repaired artifacts.

## 2026-03-17 paper-polish pass

- Goal: captions, figure placement, and tighter discussion wording now that repaired evidence is in the repo.
- `bottleneck.tex` updates made:
  - changed result-heavy tables/figures from `[h]` to `[t]` to reduce float fights
  - wrapped two wide tables in `\resizebox{\columnwidth}{!}{...}`:
    - `tab:codegen_ladder_scaled`
    - `tab:steering_controls`
  - inserted `\clearpage` before `\section{Analysis and Discussion}`
  - rewrote figure captions to be shorter and claim-first for:
    - main HumanEval comparison
    - CodeGen ladder followups
    - ablation heatmap
    - normalized-depth ablation profiles
    - syntax error profile
    - activation projection
    - steering response
    - steering-controls comparison
    - graded GPT-2 Medium vs CodeGen ablation comparison
  - tightened discussion sections:
    - `Why Does Scaling Fail for GPT-2?`
    - `How Does CodeGen Achieve Robustness?`
    - `What the CodeGen Ladder Adds`
    - `What the LiveCodeBench Result Changes`
    - `Why a Small Probe Still Works`
- Rebuild status:
  - rebuilt sequentially with `pdflatex` twice
  - final log shows no `Overfull` entries and no `LaTeX Warning` entries
  - remaining noise is underfull-box output only
  - final PDF timestamp after this pass: `2026-03-17 19:06` local time

## Current editable state

- Modified files from this pass:
  - `bottleneck.tex`
  - `bottleneck.pdf`

## 2026-03-17 benchmark-figure pass

- Goal: create the biggest missing benchmark and provenance figures using only repaired local artifacts and logs.
- New shared helper added:
  - `scripts/figure_benchmark_utils.py`
- New figure scripts added:
  - `scripts/create_figure10_cross_benchmark_map.py`
  - `scripts/create_figure11_bootstrap_forest.py`
  - `scripts/create_figure12_codegen_ladder_benchmarks.py`
  - `scripts/create_figure13_strictness_cascade.py`
  - `scripts/create_figure14_coverage_audit.py`
- `scripts/create_all_figures.sh` was extended to include the new benchmark-figure scripts.
- New generated figures:
  - `outputs/figures/figure10_cross_benchmark_map.pdf`
  - `outputs/figures/figure11_bootstrap_forest.pdf`
  - `outputs/figures/figure12_codegen_ladder_benchmarks.pdf`
  - `outputs/figures/figure13_strictness_cascade.pdf`
  - `outputs/figures/figure14_coverage_audit.pdf`
- Evidence sources used for these new figures:
  - `outputs/tables/bootstrap_significance.json`
  - `outputs/tables/bootstrap_significance_mbpp_full.json`
  - `outputs/tables/bootstrap_significance_codegen_ladder.json`
  - `outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json`
  - `outputs/logs/evalplus_gpt2_humaneval.log`
  - `outputs/logs/evalplus_gpt2_medium_humaneval.log`
  - `outputs/logs/evalplus_codegen_humaneval.log`
  - `outputs/logs/evalplus_gpt2_mbppplus.log`
  - `outputs/logs/evalplus_gpt2_medium_mbppplus.log`
  - `outputs/logs/evalplus_codegen_mbppplus.log`
  - `outputs/tables/livecodebench_gpt2_summary.json`
  - `outputs/tables/livecodebench_gpt2_medium_summary.json`
  - `outputs/tables/livecodebench_codegen_summary.json`
  - `data/results_gpt2/evaluation_results.json`
  - `data/results_gpt2_medium/evaluation_results.json`
  - `data/results_codegen/evaluation_results.json`
  - `data/results_codegen_nl/evaluation_results.json`
  - `data/results_codegen_multi/evaluation_results.json`
  - `data/results_codegen_mono/evaluation_results.json`
  - `outputs/tables/codegen_main_repair_report.json`
  - `outputs/tables/codegen_nl_repair_report.json`
  - `outputs/tables/codegen_multi_repair_report.json`
  - `outputs/tables/codegen_mono_repair_report.json`
- New figure coverage:
  - flagship cross-benchmark comparison
  - bootstrap CI / pairwise-difference figure
  - CodeGen ladder performance figure
  - strictness cascade figure
  - HumanEval coverage / repair audit figure

## 2026-03-17 figure-integration pass

- Goal: place the strongest new benchmark figures into the paper itself instead of leaving them as standalone assets.
- Main-paper integrations into `bottleneck.tex`:
  - replaced the old HumanEval-only opening figure with the benchmark-wide map (`figure10_cross_benchmark_map.pdf`)
  - added the bootstrap forest figure to the statistical-validation subsection (`figure11_bootstrap_forest.pdf`)
  - added the CodeGen ladder benchmark figure to the within-family ladder subsection (`figure12_codegen_ladder_benchmarks.pdf`)
- Appendix integrations into `bottleneck.tex`:
  - added `\appendix`
  - added a supplementary section for benchmark/provenance support figures
  - inserted the strictness cascade figure (`figure13_strictness_cascade.pdf`)
  - inserted the coverage audit figure (`figure14_coverage_audit.pdf`)
- Textual wiring added:
  - main results now explicitly reference the benchmark map, bootstrap forest, ladder benchmark figure, strictness appendix figure, and coverage appendix figure
- Rebuild status after figure integration:
  - rebuilt sequentially with `pdflatex` twice
  - final log shows no `Overfull` entries and no `LaTeX Warning` entries
  - remaining noise is underfull-box output only
  - compiled PDF is now 32 pages
  - final PDF timestamp after this pass: `2026-03-17 21:44` local time

## Next likely moves

- Review the rendered PDF for visual placement quality, not just compile success.
- Do one final narrative polish pass on the abstract/introduction/conclusion so their phrasing fully matches the stronger figure set.
- Decide whether to keep all appendix figures in the submission PDF or move one to supplementary material depending on venue norms.
- Do a last pass on abstract/introduction if we want the same tightened style there too.
- If needed, make one more placement-only pass after viewing the PDF pages.

## 2026-03-21 TMLR submission pass

- Venue decision locked: target `TMLR` first.
- Verified from the official TMLR author guide that:
  - submissions must be anonymous
  - the official TMLR LaTeX style/template is required
  - appendices are allowed in the PDF
  - supplementary material may be submitted as anonymized PDF/ZIP
  - broader-impact statements are required when risk of harm is significant
- Downloaded the official style archive from the TMLR author-guide template link and extracted it under `vendor/tmlr-style-file/`.
- Copied `tmlr.sty` and `tmlr.bst` into the repo root for local compilation.
- Converted `bottleneck.tex` from a generic two-column article build to the official TMLR style:
  - `\documentclass[10pt]{article}`
  - `\usepackage{tmlr}` in blind mode by default
  - `\usepackage[preprint]{tmlr}` when `\blindsubmissionfalse`
  - switched bibliography style from `plainnat` to `tmlr`
  - removed extra caption packages to stay closer to venue style
- Submission-facing manuscript changes:
  - kept blind submission mode on by default
  - added `Reproducibility and Artifact Availability`
  - added `Broader Impact`
  - ensured the blind build does not print the public GitHub URL
- Submission support files added:
  - `TMLR_SUBMISSION_CHECKLIST.md`
  - `ARTIFACT_README.md`
- Build/verification status:
  - full `pdflatex -> bibtex -> pdflatex -> pdflatex` cycle succeeded
  - current blind PDF is `bottleneck.pdf`
  - current TMLR-style PDF is 29 pages
  - PDF text begins with `Under review as submission to TMLR`
  - PDF title page shows anonymous authors
  - search for `Ashish Pandey`, `Khwopa`, email, GitHub handle, and repo URL returned no self-identifying hits in the blind PDF
  - remaining compile noise is only underfull-box output, with no `LaTeX Warning`, `pdfTeX warning`, or `Overfull` matches in the final log scan
