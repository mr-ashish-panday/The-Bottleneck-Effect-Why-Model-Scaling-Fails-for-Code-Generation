# TMLR Submission Checklist

Target venue: [Transactions on Machine Learning Research (TMLR)](https://www.jmlr.org/tmlr/submissions.html)

## Current venue decision

- Submit this paper to TMLR.
- Keep the manuscript anonymized for review.
- Use the public-repo version only after review or for a camera-ready release.

## Manuscript closeout

- `bottleneck.tex` now uses the official `tmlr.sty` / `tmlr.bst` files and defaults to blind-review mode via `\blindsubmissiontrue`.
- Before a public or camera-ready release, switch to `\blindsubmissionfalse`.
- Keep the explicit `HumanEval/129` coverage note in the paper; do not hide it.
- Keep the `Reproducibility and Artifact Availability` and `Broader Impact` subsections in the submission draft.

## TMLR-specific packaging

- Upload the submission through OpenReview using the official TMLR workflow.
- Ensure the PDF is anonymous.
- Ensure any supplementary material is anonymous.
- Keep supplementary material under the TMLR size limit and in PDF or ZIP format.

## Recommended supplementary ZIP contents

- `outputs/tables/` files that back the main quantitative claims
- `outputs/logs/` files needed to audit the reported benchmark and repair numbers
- `data/results_*/` processed evaluation outputs needed for figure regeneration
- `scripts/create_figure*.py`
- `scripts/figure_benchmark_utils.py`
- `scripts/create_all_figures.sh`
- a short `ARTIFACT_README.md` explaining how figures and tables were generated

## Do not include in the anonymized supplement

- author names
- institution names
- personal email addresses
- public GitHub URLs
- commit history or metadata that reveals identity

## Final pre-submit checks

- compile the blind PDF successfully
- search the PDF for your name, email, institution, and GitHub handle
- verify citations and figures render correctly
- verify the supplement contains the files needed to reproduce the reported numbers
- confirm the uploaded PDF matches the blind version, not the camera-ready version

## After review / camera-ready

- switch `\blindsubmissionfalse`
- restore author metadata
- point the artifact statement to the public repository
- keep the same quantitative claims and coverage disclosures unless new evidence is added
