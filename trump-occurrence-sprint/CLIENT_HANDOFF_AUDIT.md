# Client Handoff Audit

This file maps the client's 48-hour sprint brief to the delivered benchmark artifacts.

## Bottom Line

Status: complete and ready for packaging.

The client asked for a reproducible offline benchmark that returns `WIN`, `CONTINUE`, or `STOP` with machine-readable predictions, metrics, diagnostics, and a short decision file. This repository returns `STOP` because the challenger did not clear the locked-test continuation bar against the timing/survival baseline.

This is a valid sprint outcome under the client's rules. The sprint's job was to decide whether the LLM/LoRA direction is worth further work under the specified protocol, and the result gives that decision with a reproducible audit trail.

## Client Requirement Match

| Client requirement | Delivered artifact | Status | Notes |
|---|---|---:|---|
| Offline retrospective benchmark | `README.md`, `config.yaml`, `src/` | Done | No paid or external API dependency. |
| Rev transcript corpus | `data/corpus.jsonl`, `data/data_card.md`, `data/raw/` | Done | 174 cleaned transcripts, every kept transcript has a real date. |
| Minimum 60 transcripts, target 100+ | `data/data_card.md` | Done | 174 transcripts exceeds target. |
| Shared occurrence matcher | `src/matching.py`, `tests/test_matching.py` | Done | Matcher tests cover words, phrases, punctuation, casing, spacing, hyphenation, possessives, start/end, and substring false positives. |
| Chronological train/val/test split | `data/splits.json`, `tests/test_leakage.py` | Done | Transcript-level split. |
| Train-only frozen targets | `data/targets.json`, `tests/test_leakage.py` | Done | 180 targets generated from train only. |
| Replay checkpoints | `benchmark/checkpoints_train.parquet`, `benchmark/checkpoints_val.parquet`, `benchmark/checkpoints_test.parquet` | Done | Rows use 0-90 percent checkpoint grid and drop already-appeared targets. |
| Evaluation harness | `src/evaluate.py`, `tests/test_harness.py` | Done | Brier, log loss, ECE, reliability, calibration diagnostics, AUC diagnostic, slices, and bootstrap support. |
| Baseline A and Baseline B | `src/baselines.py`, `predictions/val_constant.parquet`, `predictions/val_timing.parquet`, `metrics/val_metrics.json` | Done | Baseline B beat Baseline A on validation Brier. |
| P0 content challenger | `src/llm_features.py`, `src/challenger.py`, `predictions/val_challenger.parquet` | Done | Local TF-IDF/SVD and train-only adjacency features nested over Baseline B. |
| LoRA status documented | `README.md`, `DECISION.md` | Done | LoRA was not trained, which the brief allows for P0. |
| Contamination probe | `src/contamination.py`, `metrics/contamination_report.json`, `metrics/test_contamination_report.json` | Done | Local deterministic generation with `distilgpt2`; 0 validation and 0 test flags. |
| Content ablation | `metrics/ablation.json`, `predictions/test_challenger_blanked.parquet` | Done | Full and blanked challenger arms are reported. |
| Duration sensitivity | `src/duration_sensitivity.py`, `metrics/duration_sensitivity.json` | Done | Validation monotonic fraction `0.766`. |
| Cost and latency | `metrics/cost_latency.json` | Done | API spend is `0.00`. |
| Single locked test pass | `src/run_phase4_locked_test.py`, `metrics/freeze_manifest.json`, test predictions and metrics | Done | Test was scored after validation gates and freeze manifest. |
| Required final decision | `DECISION.md` | Done | Verdict is `STOP` with exact test numbers and bootstrap CI. |
| Required final folder structure | Project tree under `trump-occurrence-sprint/` | Done | Required files are present; extra generated/model files are included for reproducibility. |
| Reproducibility from README | `README.md`, `requirements.txt` | Done | README documents dependency expectations and `requirements.txt` captures the install contract. |

## Key Evidence

- Validation gate passed before test:
  - Baseline B validation Brier: `0.040780`
  - Challenger validation Brier: `0.040337`
  - Duration monotonic fraction: `0.766`
- Locked test result:
  - Baseline B test Brier: `0.043319`
  - Challenger full test Brier: `0.044434`
  - Absolute improvement: `-0.001115`
  - Relative improvement: `-2.58%`
  - Bootstrap CI for challenger-full minus Baseline B Brier: `[0.000259, 0.002074]`
- STOP rule triggered:
  - No improvement over Baseline B on all eval rows.
  - `t > 0` result is also worse.
  - Full and blanked arms do not show useful content signal on test.

## Final Sending Checklist

1. Send the packaged `trump-occurrence-sprint/` handoff archive together with the checksum file.
2. Keep the full local folder available, including `models/`, raw HTML, predictions, metrics, and the frozen manifest.
3. If the client wants to reproduce on a fresh machine, use `README.md` and `requirements.txt` as the install contract.

## What Should Not Be Done

- Do not tune the challenger after viewing locked test metrics.
- Do not rerun Phase 4 to chase a better result.
- Do not train LoRA inside this same evaluated sprint unless the current result is explicitly discarded and a new benchmark split/test protocol is started.
