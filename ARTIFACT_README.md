# Artifact README

This repository contains the processed outputs and plotting scripts needed to audit the quantitative claims in the paper and regenerate the committed figures.

## Main evidence directories

- `outputs/tables/`: benchmark summaries, bootstrap significance files, ladder summaries, LiveCodeBench summaries, and repair reports
- `outputs/logs/`: generation logs, evaluation logs, steering logs, and repair logs
- `data/results_*/`: processed evaluation outputs for each model family and ablation setting
- `outputs/figures/`: rendered paper figures
- `scripts/`: plotting and figure-regeneration scripts

## Figure regeneration

The benchmark and provenance figures can be recreated from the saved processed outputs with:

```powershell
bash scripts/create_all_figures.sh
```

If a local environment does not support the shell wrapper, the individual `scripts/create_figure*.py` files can be run directly with Python.

## Notes on coverage

- The main synced CodeGen HumanEval artifact retains an empty `HumanEval/129` entry, so sample-level main results use 16,300 scored completions while the bootstrap summary keeps the empty task as a zero-success entry for 164-task problem-level intervals.
- The repaired CodeGen ladder artifacts restore full saved coverage for the ladder checkpoints.

## Anonymization guidance for review

For double-blind submission, package only the files needed to reproduce the reported claims and remove public-repository links, author names, institution names, and any other identifying metadata from the supplement bundle.
