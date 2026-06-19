# Delivery Manifest

Package: `trump-occurrence-sprint`

Purpose: offline benchmark for forecasting whether a target word or phrase appears later in a Rev transcript, with a final `WIN`, `CONTINUE`, or `STOP` decision.

Final verdict: `STOP` under the specified sprint decision rules.

## Start Here

1. Read `DECISION.md` for the final result and reasoning.
2. Read `CLIENT_HANDOFF_AUDIT.md` for a requirement-by-requirement mapping against the sprint brief.
3. Read `README.md` for reproduction commands and folder structure.

## Included Evidence

- Corpus: `data/corpus.jsonl`, `data/data_card.md`, `data/raw/`
- Splits and targets: `data/splits.json`, `data/targets.json`
- Replay rows: `benchmark/checkpoints_train.parquet`, `benchmark/checkpoints_val.parquet`, `benchmark/checkpoints_test.parquet`
- Predictions: `predictions/`
- Metrics and diagnostics: `metrics/`
- Frozen model artifacts: `models/`
- Source code: `src/`
- Tests: `tests/`
- Dependency contract: `requirements.txt`

## Verification Performed

- `py -3 -m pytest tests -q`
- Result: `11 passed`

## Protocol Note

`src.run_phase4_locked_test` is the single locked test pass. The included metrics and predictions are the final scored artifacts. Do not rerun the locked test to tune the model after seeing the result.
