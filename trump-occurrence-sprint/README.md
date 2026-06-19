# Trump Occurrence Forecasting Sprint

Offline benchmark for forecasting whether a target word or phrase will appear later in a Rev transcript.

The benchmark is API-free: local Rev scraping, local content features, calibrated baselines, validation diagnostics, and one locked test pass.

Final verdict: `STOP`. See `DECISION.md`.

LoRA was not trained. The challenger was feature extraction nested over the timing/survival baseline.

## Reproduction

Use `py -3` on Windows.

```powershell
cd trump-occurrence-sprint
py -3 -m pip install requests beautifulsoup4 pandas pyarrow numpy scikit-learn scipy pyyaml pytest tqdm transformers==4.46.3 tokenizers
py -3 -m pytest tests
py -3 -m src.run_phase0
py -3 -m src.run_phase1
py -3 -m src.run_phase2
py -3 -m src.run_phase3
py -3 -m src.run_phase4_locked_test
```

The contamination probe uses local `transformers` generation and therefore needs a working CPU `torch` install. If `transformers` imports fail because of a mismatched optional vision package, remove or fix that optional package before running the probe; the final local run used `transformers==4.46.3` and no paid API.

The contamination probe is a conservative local screen, not proof that public transcripts are absent from all larger-model pretraining corpora.

`src.run_phase4_locked_test` is the single locked test pass. Do not rerun it after inspecting test metrics unless declaring the prior test invalid.

## Folder Map

- `data/raw/`: saved Rev HTML pages.
- `data/corpus.jsonl`: cleaned transcript corpus.
- `data/data_card.md`: corpus counts, date range, format distribution, and cleaning notes.
- `data/splits.json`: chronological transcript-level split.
- `data/targets.json`: frozen train-only target list.
- `benchmark/`: train/validation/test checkpoint rows.
- `predictions/`: validation and test prediction parquet files.
- `metrics/`: validation, test, slice, bootstrap, reliability, contamination, ablation, duration, cost, and freeze artifacts.
- `models/`: frozen local model artifacts used by the locked test script.
- `src/`: scraper, cleaner, matcher, benchmark, baseline, challenger, diagnostic, and runner code.
- `tests/`: matcher, leakage, and harness tests.

## Final Test Summary

| Model | Test Brier | Test ECE |
|---|---:|---:|
| Constant baseline | 0.048416 | 0.022703 |
| Timing/survival baseline | 0.043319 | 0.018365 |
| Challenger full | 0.044434 | 0.018069 |
| Challenger blanked | 0.044460 | 0.014503 |

The challenger was worse than Baseline B on the locked test pass, so the sprint returns `STOP`.

## Known Limitations

- Corpus discovery starts from the Rev Donald Trump category page, then expands through Rev's own sitemap because the visible category page was below the 60-transcript floor.
- Primary-speaker extraction is heuristic; the data card records the cleaning choices.
- The final content challenger uses local train-only adjacency features, not a prompted paid API model.
- The chronological test split is temporally shifted toward newer, shorter 2026 press/event transcripts, which is the intended held-out protocol but makes the negative result conservative for train-only content features.
- The test set was touched once by `src.run_phase4_locked_test`.
