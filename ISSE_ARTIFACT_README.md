# ISSE Artifact README

This artifact package supports the manuscript:

The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation

Target journal: Innovations in Systems and Software Engineering.

## What Is Included

- `bottleneck.pdf`: compiled manuscript.
- `bottleneck.tex`: manuscript source.
- `references.bib`: bibliography source.
- `outputs/figures/`: figures referenced by the manuscript.
- `outputs/tables/`: processed tables and aggregate summaries used by the text and figures.
- `scripts/`: experiment, analysis, and figure-generation scripts.
- `src/`: project source code used by the scripts.
- `configs/` and `config*.yaml`: configuration files for benchmark and model runs.
- `data/results*/`: saved generated-result artifacts used for reported evaluations and figure reconstruction.
- `requirements.txt` and `requirements-lightning.txt`: environment notes.
- `ISSE_CLAIM_ARTIFACT_AUDIT.md`: the claim-to-artifact audit that maps reported manuscript claims to saved local evidence.

## What Is Not Included

- Pretrained model checkpoints are not redistributed.
- Raw Hugging Face model caches are not included.
- GPU runtime environments are not included.
- Raw local execution logs are not included in the reviewer ZIP; the local repository retains them, while the reviewer package includes structured result artifacts, processed tables, figures, scripts, configs, and audit reports used by the manuscript.

All model checkpoints used by the experiments are public pretrained models and can be reloaded by the scripts/configuration files in a compatible Python environment.

## Important Manuscript-Facing Files

- `outputs/tables/jss_targeted_robustness_summary.md`
- `outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json`
- `outputs/tables/jss_decoding_robustness_20s/aggregate_summary.json`
- `outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json`
- `outputs/figures/figure15_targeted_robustness_controls.pdf`
- `scripts/create_figure15_targeted_robustness_controls.py`

## Rebuild Notes

The manuscript was built locally with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error bottleneck.tex
bibtex bottleneck
pdflatex -interaction=nonstopmode -halt-on-error bottleneck.tex
pdflatex -interaction=nonstopmode -halt-on-error bottleneck.tex
```

The final checked build had no LaTeX errors, no unresolved citations, no unresolved references, and no serious overfull boxes.
