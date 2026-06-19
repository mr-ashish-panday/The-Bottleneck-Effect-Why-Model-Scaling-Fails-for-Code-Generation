# Client Spec Line-by-Line Review

Source: client-provided `SPRINT_48H.md`

This review treats the client brief as the source of truth. Blank lines, separators, and table formatting lines are omitted only where they add no independent requirement.

## Core Intent

| Lines | Client says | Meaning |
|---:|---|---|
| 1 | 48-hour sprint: live word/phrase occurrence forecasting benchmark | The deliverable is a benchmark sprint, not a product build. |
| 3-7 | Build an offline benchmark and decide whether LLM/LoRA beats a simple non-neural baseline with better-calibrated forecasts | The sprint's single job is an evidence-backed `yes/no` decision. A negative answer is still a valid completed result. |
| 9-19 | Forecast whether a target appears after checkpoint `t`; Brier is primary; calibration matters; AUC diagnostic only | Success is judged by calibrated probabilities on held-out transcripts, not classification rank or narrative argument. |

## Scope Boundaries

| Lines | Client says | Meaning |
|---:|---|---|
| 21-28 | No production system, live ASR, UI, external integrations beyond Rev, prose-only report, per-target models, or fixed target vocabulary | Protect the benchmark. Do not spend time on app polish, services, dashboards, APIs, or target-specific hacks. |
| 30-34 | Starting input is one Rev Trump category URL; collect corpus from Rev | Corpus must be scraped from Rev, starting from the provided URL. |
| 36-44 | P0 required, P1 wanted, P2 bonus; LoRA is P1, P0 challenger is feature-extraction/prompted; document if no LoRA | LoRA is not required for completion. It is acceptable to skip if P0 is solid and documented. |
| 46-59 | Six gated phases over 48 hours; cut P1/P2 if behind and protect Phase 4/5 | The locked test and final decision matter more than ambitious model work. |

## Phase 0: Corpus and Matching

| Lines | Client says | Meaning |
|---:|---|---|
| 63-72 | Build scraper, cleaner, single matcher, corpus JSONL, data card, and matching tests | Must produce real data plus reusable occurrence logic and tests. |
| 66 | Scrape transcript text, title, date, source URL from Rev Trump category | These fields are mandatory corpus metadata. |
| 67 | Parse turns/timestamps, prefer primary speaker, strip boilerplate/annotations, dedupe, infer format, tokenize and record `n_words` | Cleaning choices must be systematic and documented. |
| 68 | One shared occurrence function; case-insensitive, normalized, word-boundary, phrase support, hyphen handling; no ad hoc matching elsewhere | Matching consistency is a hard correctness requirement. |
| 69-70 | `corpus.jsonl` fields and `data_card.md` content | Corpus and data card must be machine/audit friendly. |
| 74-78 | Real date on every transcript; floor 60, target 100+; matching tests pass | Gate 0 cannot pass without dated corpus and green matcher tests. |

## Phase 1: Splits, Targets, Replay Rows, Harness

| Lines | Client says | Meaning |
|---:|---|---|
| 82-91 | Build splits, targets, checkpoint parquets, evaluator, leakage tests | This is the benchmark backbone. |
| 85 | Chronological transcript-level 70/15/15 split; no transcript crosses splits; record date boundaries | Leakage control starts at split construction. |
| 86 | Targets generated from train split only, with frequency bands and cold set; frozen for rest of sprint | No val/test target mining. |
| 87 | Checkpoint rows at 0-90%; drop already-appeared targets; include required columns; `first_occurrence_index` label-side only | The replay dataset must simulate live prediction without leaking future words. |
| 88 | Evaluator outputs Brier, log loss, ECE, reliability, calibration slope/intercept, AUC diagnostic, slices, and 1000-rep transcript bootstrap CIs | Metrics must be machine-readable and statistically useful. |
| 89 | Leakage tests enforce no future words, no split overlap, train-only targets | Tests are part of the deliverable, not optional cleanup. |
| 93-98 | Parquets build, dummy predictor scores, leakage tests pass | Gate 1 proves the harness works before modeling. |

## Phase 2: Baselines and Calibration

| Lines | Client says | Meaning |
|---:|---|---|
| 102-113 | Build Baseline A, Baseline B, expected remaining words, calibration, validation predictions and metrics | Need a strong non-neural baseline before challenging it. |
| 104 | Same prediction interface for baselines | Comparable model APIs matter for clean evaluation. |
| 106 | Baseline A is smoothed train target-rate; no `t`, prefix, or remaining length | Baseline A is the simple floor. |
| 107 | Baseline B is timing/survival logistic using train-only target history, format, time/length features, no content | Baseline B is the real bar the challenger must beat. |
| 109 | Expected remaining words estimated from train length distribution by format, with fallback | Duration signal must be train-derived. |
| 111 | Calibration on validation only; freeze before test | Calibration cannot use test. |
| 115-119 | Both baselines calibrated; Baseline B must beat A on validation | Gate 2 blocks building on a broken baseline. |

## Phase 3: Challenger and Diagnostics

| Lines | Client says | Meaning |
|---:|---|---|
| 123-131 | Challenger must be additive over Baseline B in logit space with Baseline B as fixed offset; fit beta on train; calibrate on validation | The challenger must test whether content adds signal beyond timing, not relearn the baseline. |
| 133-139 | Pick a small set of LLM/content features; P0 needs at least one content feature; document computation | Feature count should stay small and auditable. |
| 141 | LoRA is P1; if skipped, P0 prompted/feature challenger stands | Skipping LoRA is allowed. |
| 143-150 | Four validation diagnostics: contamination, content ablation, duration sensitivity, cost/latency | Diagnostics are P0, not nice-to-have. |
| 145 | Contamination probe must generate deterministic continuation and compare overlap; flag transcripts; recompute pruned results in Phase 4 | Public transcripts may be memorized, so contamination has to be checked. |
| 146 | Full vs blanked challenger with same beta/calibrator | Measures whether transcript content carries the signal. |
| 147 | Duration sensitivity should be non-decreasing and non-flat | Model must understand remaining transcript length. |
| 148 | Report tokens/cost/latency and plausibility | Practicality is part of the decision. |
| 152-157 | Challenger must beat Baseline B on validation; all diagnostics exist; duration passes | Gate 3 justifies spending the single locked test pass. |

## Phase 4: Locked Test

| Lines | Client says | Meaning |
|---:|---|---|
| 161-163 | Freeze models/features/coefs/calibrators; test touched exactly once; no model modification after any test number | This is the protocol's highest-integrity requirement. |
| 165-169 | Score Baseline A, Baseline B, challenger-full, challenger-blanked; report metrics/slices/bootstrap/pruned test | The locked test must produce every final comparison in one pass. |
| 171-175 | Required test predictions and metrics artifacts; all written from a single pass | Gate 4 is artifact completeness plus one-pass integrity. |

## Phase 5: Decision and Finalization

| Lines | Client says | Meaning |
|---:|---|---|
| 179-186 | Build `DECISION.md`, `README.md`, `config.yaml`; include verdict, exact numbers, CIs, reasoning, LoRA status, reproduction, limitations, all knobs | Finalization is part of P0. |
| 190-199 | Evaluate STOP first; STOP if no improvement, pruned improvement disappears, only `t=0`, ablation weak, or cost implausible | STOP is a first-class required outcome. |
| 201-207 | WIN requires strong Brier improvement, acceptable ECE, pruned persistence, positive `t>0`, content signal, plausible cost | WIN bar is deliberately high. |
| 209 | CONTINUE only if improvement is real but below full WIN threshold | CONTINUE still requires positive evidence. |

## Cross-Cutting Controls

| Lines | Client says | Meaning |
|---:|---|---|
| 213-220 | Enforced leakage controls: chronological splits, train-only targets/stats/features, validation-only calibration, no future words, one test pass | These are correctness constraints across the whole repo. |
| 222-228 | Calibration requirements; all probabilities calibrated; ECE/reliability/slope/intercept; AUC diagnostic only | A model cannot win by AUC alone. |
| 230-283 | Required final folder structure | The client expects a reproducible repo with named artifacts. |
| 285-287 | Done means reproducible repo, machine-readable predictions/metrics for validation and single test pass, four diagnostics, `DECISION.md` with WIN/CONTINUE/STOP and LoRA status; benchmark, not prose, is deliverable | The final acceptance test is artifact-backed reproducibility, not a narrative report. |

## Critical Interpretation

The client did not ask for a model victory. The client asked for a clean benchmark that decides whether the LLM/LoRA direction is worth more work. Because the decision rules say to evaluate `STOP` first, a `STOP` verdict is complete if the repo contains the required artifacts, diagnostics, metrics, and reproducible decision trail.

The biggest client-risk areas are:

1. Locked-test integrity: no tuning or rerunning after seeing test numbers.
2. Challenger form: the content challenger must clearly be nested over Baseline B.
3. Contamination probe: if local generation is weak or unavailable, this must be documented honestly.
4. Reproducibility: README, requirements, config, raw data, metrics, predictions, and model artifacts must travel together.
5. Machine-readable artifacts: the benchmark files matter more than explanatory prose.
