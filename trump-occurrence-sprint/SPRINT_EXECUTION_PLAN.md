# 48-Hour Sprint Execution Plan

## Verdict

This sprint is buildable in 48 hours as an API-free P0 benchmark if we keep the scope disciplined.

The deliverable is not a guaranteed LLM win. The deliverable is a reproducible offline benchmark that returns `WIN`, `CONTINUE`, or `STOP` with calibrated Brier-score evidence. The main risk is not implementation volume; it is invalid evidence from leakage, test peeking, weak calibration, or wasting time on LoRA before the core benchmark is locked.

Primary bottleneck: `judgment`.

Primary rule: protect the P0 benchmark and cut P1/P2 immediately if any gate slips.

Client-alignment rating after review: 8.7/10 before implementation. The plan matches the core deliverables, gates, leakage controls, and decision rules. The main remaining risk is operational: executing the contamination probe and fixed-offset challenger exactly enough that the client cannot call them soft substitutes.

## Scope Decision

### In Scope

- Scrape Rev transcript pages starting from `https://www.rev.com/category/donald-trump`.
- Use category-page links first, then reproducibly expand through Rev's own sitemap if the visible category page is below the 60-transcript floor. Treat this as Rev-internal discovery from the provided starting domain, not as a second corpus source.
- Build a cleaned transcript corpus with real dates, source URLs, inferred formats, tokenized text, and word counts.
- Build one shared occurrence matcher used by all downstream code.
- Generate chronological train/validation/test splits at transcript level.
- Generate frozen targets from the train split only.
- Build replay checkpoint rows at the requested 0-90% word-position grid.
- Build constant and timing/survival baselines.
- Fit calibration only on validation.
- Build an API-free content challenger nested over Baseline B in logit space.
- Produce validation diagnostics before any test evaluation.
- Run a single locked test pass after all models/calibrators are frozen.
- Produce machine-readable predictions, metrics, diagnostics, and a short `DECISION.md`.

### Out of Scope Unless P0 Is Already Finished

- LoRA fine-tuning.
- Paid or unreliable free internet APIs.
- UI/dashboard/web app.
- Per-target models.
- Manual cherry-picking of targets or transcripts after validation/test results.
- Multiple test attempts.

## API-Free Strategy

No OpenAI, Claude, Gemini, or random free API is required.

The challenger will use local content features:

- trailing-window/target embedding cosine using local `sentence-transformers`
- prefix topical relevance from local embeddings
- train-only target-adjacent co-occurrence features
- optional lexical/context overlap features computed only from prefix words before `t`

This satisfies the P0 requirement for at least one content feature and keeps the benchmark reproducible. Cost/latency reporting will document local CPU/GPU wall-clock time and zero API spend.

## Non-Negotiable Evidence Rules

- The same `src/matching.py` functions must be used for every occurrence check.
- Splits are chronological and transcript-level.
- Targets are generated from train transcripts only.
- Target rates, first-occurrence statistics, co-occurrence tables, duration estimates, and all fitted feature tables are train-only.
- Calibration is validation-only.
- Test is evaluated exactly once after all coefficients, feature definitions, and calibrators are frozen.
- `first_occurrence_index` is label-side only and never enters model-facing feature matrices.
- AUC is diagnostic only; Brier and calibration drive the decision.
- If corpus count is below 60 after documented scraping, mark the benchmark underpowered in `data_card.md` and `DECISION.md` rather than hiding it.
- Aim for 100 or more transcripts; 60 is only the floor.
- Every predictions table must use the required columns: `transcript_id`, `target`, `t_pct`, `p_pred`, and `model_name`.

## Repository Layout

Target folder:

```text
trump-occurrence-sprint/
  README.md
  DECISION.md
  SPRINT_EXECUTION_PLAN.md
  config.yaml
  data/
    raw/
    corpus.jsonl
    data_card.md
    splits.json
    targets.json
  src/
    scrape.py
    clean.py
    matching.py
    targets.py
    checkpoints.py
    baselines.py
    calibrate.py
    llm_features.py
    challenger.py
    evaluate.py
    contamination.py
    duration_sensitivity.py
    run_phase0.py
    run_phase1.py
    run_phase2.py
    run_phase3.py
    run_phase4_locked_test.py
  benchmark/
    checkpoints_train.parquet
    checkpoints_val.parquet
    checkpoints_test.parquet
  predictions/
    val_constant.parquet
    val_timing.parquet
    val_challenger.parquet
    test_constant.parquet
    test_timing.parquet
    test_challenger_full.parquet
    test_challenger_blanked.parquet
  metrics/
    val_metrics.json
    test_metrics.json
    slice_metrics.json
    bootstrap_deltas.json
    reliability.json
    contamination_report.json
    contamination_pruned_metrics.json
    ablation.json
    duration_sensitivity.json
    cost_latency.json
  tests/
    test_matching.py
    test_leakage.py
    test_harness.py
```

## Setup Plan

Use `py -3`, not bare `python`, because the current bare `python` command resolves to a broken local trampoline.

Planned setup:

```powershell
cd C:\Users\Ashish\The-Bottleneck-Effect-Why-Model-Scaling-Fails-for-Code-Generation\trump-occurrence-sprint
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install requests beautifulsoup4 pandas pyarrow numpy scikit-learn pyyaml pytest tqdm sentence-transformers
```

If `sentence-transformers` is already available globally but slow to install in the sprint venv, use the global `py -3` environment and record that in `README.md`. Do not spend more than 45 minutes fighting environment setup.

## Phase Plan

### Phase -1: Sprint Boot and Repro Contract, H0-H1

Build:

- `config.yaml` with all knobs and seeds.
- empty output directories.
- `README.md` skeleton with planned reproduction commands.
- command runner scripts by phase.

Proof gate:

- `.\.venv\Scripts\python --version` works, or fallback runtime is documented.
- `pytest` can run an empty/smoke test.
- `config.yaml` exists.

Cut rule:

- If setup takes over 45 minutes, skip isolated venv and use working `py -3` directly.

### Phase 0: Corpus, Cleaning, Matching, H1-H10

Build:

- `src/scrape.py`
- `src/clean.py`
- `src/matching.py`
- `tests/test_matching.py`
- `data/raw/*.html`
- `data/corpus.jsonl`
- `data/data_card.md`

Scrape strategy:

1. Fetch the Rev Trump category page.
2. Extract transcript URLs present on the category page.
3. If fewer than 60 usable transcripts are found, fetch Rev sitemap and filter transcript URLs whose slug/title/category evidence ties them to Donald Trump.
4. For each candidate page, keep it only if it has a real date and transcript body.
5. Record in `data_card.md` exactly how URLs were discovered and filtered.

Cleaning strategy:

- Parse speaker turns and timestamps from Rev HTML.
- Prefer the primary speaker's words where there is a clear dominant Donald Trump speaker.
- If no reliable primary-speaker extraction is possible, keep full transcript text but record the cleaning limitation.
- Strip bracketed annotations such as applause/crosstalk/noise.
- Unicode-normalize text.
- Deduplicate near-identical transcripts by normalized title/date/text hash and approximate text hash.
- Infer format from title regex: `rally`, `speech`, `interview`, `debate`, `press_event`, `legal`, `unknown`.

Matching tests must cover:

- single words
- multi-word phrases
- punctuation boundaries
- casing
- repeated whitespace
- hyphenation
- possessives
- target at transcript start
- target at transcript end
- no substring false positives such as `war` in `toward` or `warmer`

Proof gate:

- corpus loads as JSONL
- every kept transcript has `date`
- `n_words` present and positive
- at least 60 transcripts, or underpowered status explicitly documented
- matching tests pass

Cut rule:

- If primary-speaker extraction becomes brittle, keep full transcript text and document it rather than burning the sprint.

### Phase 1: Splits, Targets, Replay Rows, Harness, H10-H20

Build:

- `src/targets.py`
- `src/checkpoints.py`
- `src/evaluate.py`
- `tests/test_leakage.py`
- `tests/test_harness.py`
- `data/splits.json`
- `data/targets.json`
- `benchmark/checkpoints_train.parquet`
- `benchmark/checkpoints_val.parquet`
- `benchmark/checkpoints_test.parquet`

Split plan:

- sort transcripts by date, then `transcript_id`
- oldest 70% train
- next 15% validation
- newest 15% test
- record date boundaries and transcript counts

Target plan:

- generate only from train split
- about 40 high-frequency unigrams
- about 40 mid-frequency unigrams
- about 40 rare unigrams
- about 30 two-to-three-word phrases
- about 30 cold targets that occur in at most a small number of train transcripts
- exclude stopword-only targets, obvious artifacts, and very short tokens
- freeze `data/targets.json` before validation modeling

Checkpoint plan:

- grid: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 percent of `n_words`
- drop rows where target already appeared by `t`
- labels are `label_occurs_after`
- include `first_occurrence_index` only for labels/tests; block it from model feature builders

Harness metrics:

- Brier
- log loss
- ECE with 10 bins
- reliability table
- calibration slope/intercept
- AUC diagnostic
- slices by checkpoint, target band, and `t=0` vs `t>0`
- bootstrap confidence intervals by transcript resampling

Proof gate:

- dummy constant predictor scores end to end
- metrics JSON has Brier, ECE, reliability, and bootstrap CIs
- leakage tests pass

Cut rule:

- If 1000 bootstrap reps are slow, use a config knob to run 200 during development and 1000 for final artifacts.
- During development, use a config knob to run 200 bootstrap reps. Final validation and test artifacts must use 1000 reps unless `DECISION.md` explicitly marks the run as timebox-limited and weaker than requested.

### Phase 2: Baselines and Calibration, H20-H28

Build:

- `src/baselines.py`
- `src/calibrate.py`
- `predictions/val_constant.parquet`
- `predictions/val_timing.parquet`
- `metrics/val_metrics.json`

Baseline A:

- smoothed target any-occurrence rate across train transcripts
- back off from target-format rate to target global rate to global rate
- no use of `t`, prefix, or remaining length

Baseline B:

- regularized logistic model trained on train checkpoint rows
- allowed features:
  - train-only target any-occurrence rate
  - train-only count per 1k words
  - train-only mean/median first occurrence position
  - train-only early-vs-late usage
  - format
  - `t_pct`
  - `elapsed_words`
  - `expected_remaining_words`
- no content features

Expected remaining words:

- train-only duration distribution by format
- conditional on transcript length exceeding `elapsed_words`
- use conditional median remaining
- back off format to global

Calibration:

- isotonic primary fit on validation only
- Platt alternative if isotonic overfits or fails on sparse bins
- save calibrator metadata and do not refit after test

Proof gate:

- calibrated Baseline B beats calibrated Baseline A on validation Brier
- if not, fix Baseline B before continuing

Cut rule:

- If complex duration features are unstable, simplify to robust conditional median plus regularized logistic features.

### Phase 3: API-Free Challenger and Diagnostics, H28-H40

Build:

- `src/llm_features.py`
- `src/challenger.py`
- `src/contamination.py`
- `src/duration_sensitivity.py`
- `predictions/val_challenger.parquet`
- `metrics/contamination_report.json`
- `metrics/ablation.json`
- `metrics/duration_sensitivity.json`
- `metrics/cost_latency.json`

Challenger form:

```text
logit(p_final) = logit(p_baseline_B) + beta * content_features
```

Implementation detail:

- compute Baseline B raw probabilities
- transform to clipped logits
- fit logistic regression with the baseline logit as a true fixed offset
- implement the fixed-offset fit with a small custom optimizer if needed: optimize only `beta` and intercept for `sigmoid(offset_logit + intercept + X_content @ beta)`
- do not treat Baseline B as an ordinary trainable feature unless the deviation is explicitly marked in `DECISION.md`
- calibrate challenger on validation only

Content features:

- embedding cosine between target text and trailing prefix window
- embedding cosine between target text and full prefix summary/window
- train-only co-occurrence adjacency score between prefix terms and target
- prefix lexical theme overlap with train documents where target later appears
- optional recency-weighted variant of the trailing-window cosine

Blanked ablation:

- same metadata/timing and target
- content-bearing prefix features set to blank/zero
- same fitted beta and same calibrator

Contamination probe:

- API-free default: local deterministic continuation generation, not an external API
- feed an early prefix to a small local model with deterministic decoding (`temperature=0` or greedy)
- compare generated continuation against the true continuation using longest common n-gram and exact-span recall
- freeze the threshold and method on validation before test
- apply the frozen contamination method to test inside the single locked test pass, then recompute headline metrics with flagged test transcripts removed
- if no local generative model can be made to run, use the lexical memorization probe only as a documented fallback and mark it as a P0 deviation from the client's requested contamination probe
- output flagged transcript IDs, thresholds, and method
- be explicit in `README.md` and `DECISION.md` about which contamination method actually ran

Duration sensitivity:

- sample validation rows
- hold prefix/content fixed
- recompute predictions at 0.5x, 1x, and 2x expected remaining words
- report monotonic fraction and mean sensitivity

Cost/latency:

- measure local feature-generation wall time
- measure prediction wall time
- estimate one full evaluation pass
- report tokens/window words processed per prediction, wall-clock latency per prediction, total local model calls or embedding batches, and zero API dollars

Proof gate:

- challenger beats Baseline B on validation Brier
- diagnostics files exist
- duration sensitivity monotonic fraction is clearly above half and non-flat

Cut rule:

- If challenger does not beat Baseline B on validation, do not touch test. Write the validation result and decide whether to stop or revise content features within the validation-only budget.

### Phase 4: Single Locked Test Evaluation, H40-H46

Build:

- `predictions/test_constant.parquet`
- `predictions/test_timing.parquet`
- `predictions/test_challenger_full.parquet`
- `predictions/test_challenger_blanked.parquet`
- `metrics/test_metrics.json`
- `metrics/slice_metrics.json`
- `metrics/bootstrap_deltas.json`
- `metrics/reliability.json`
- `metrics/contamination_pruned_metrics.json`
- test arm of `metrics/ablation.json`

Lock protocol:

1. Write `metrics/freeze_manifest.json` with:
   - config hash
   - corpus hash
   - split hash
   - targets hash
   - feature definition hash
   - model coefficients hash
   - calibrator hash
2. Run one command: `run_phase4_locked_test.py`.
3. The command writes all test predictions and metrics in one pass.
4. No model or feature code changes after viewing test numbers unless the final decision explicitly declares the test invalid.

Proof gate:

- every required test artifact exists
- contamination-pruned recomputation exists
- bootstrap deltas exist

Cut rule:

- If a file-write or packaging bug occurs after scoring but before writing all metrics, preserve raw predictions and declare exactly what happened. Do not silently rerun with changed models.

### Phase 5: Decision and Finalization, H46-H48

Build:

- `DECISION.md`
- final `README.md`
- final `config.yaml`
- final data card updates

Decision logic:

- evaluate STOP first
- then WIN
- otherwise CONTINUE

Required decision numbers:

- Baseline B test Brier
- challenger-full test Brier
- absolute improvement
- relative improvement
- bootstrap CI for challenger-full minus Baseline B
- `t>0` delta and CI
- contamination-pruned improvement
- content ablation share
- ECE comparison
- high-confidence reliability bucket note
- cost/latency note
- plain statement: LoRA was not trained unless P1 actually happened

Proof gate:

- `DECISION.md`, `README.md`, and `config.yaml` exist
- reproduction commands are accurate
- final tests pass

## Review Checkpoints During Execution

Use these as stop/go moments:

1. After Phase 0: review corpus count, date range, cleaning caveats.
2. After Phase 1: review split boundaries and target bands before modeling.
3. After Phase 2: review Baseline A vs B validation metrics.
4. After Phase 3: review challenger validation metrics and diagnostics before touching test.
5. Before Phase 4: explicit lock confirmation that test is about to be evaluated once.
6. After Phase 5: review final verdict.

## Risk Register

| Risk | Severity | Mitigation |
|---|---:|---|
| Visible category page has fewer than 60 transcripts | High | Expand through Rev sitemap, filter reproducibly, document method |
| Scraper breaks on inconsistent Rev page templates | High | Save raw HTML, parse multiple fallbacks, drop pages with documented reasons |
| Primary-speaker extraction is unreliable | Medium | Prefer primary speaker when clear; otherwise keep full transcript and document caveat |
| Leakage through target generation or feature stats | Critical | Enforce train-only builders and leakage tests |
| Baseline B fails to beat Baseline A | High | Fix survival/duration features before any challenger work |
| Challenger fails validation | Medium | Do not touch test; improve content features only on train/validation or report STOP/blocked |
| Embedding model install/download is slow | Medium | Use already installed local model if available; otherwise fallback to TF-IDF/SVD content features |
| Bootstrap is slow | Medium | Dev reps configurable; final requested artifacts use 1000 reps |
| Local generative contamination probe is slow or unavailable | High | Try a tiny local model first; lexical-only probe is allowed only as a documented P0 deviation |
| Test pass accidentally repeated | Critical | Freeze manifest and one locked script; do not tune after test |
| LoRA temptation consumes time | Critical | LoRA is P1 only and cut by default |

## Scoreboard

Leading metric:

- phase gate artifacts completed and tests passing by their timebox

Lagging validation metric:

- final `DECISION.md` contains a defensible WIN/CONTINUE/STOP verdict backed by single-pass test metrics

## Default Work Blocks in Nepal Time

Set exact NPT anchors when execution starts:

- Start:
- Check-in by:
- Late after:
- Proof due:

For each block, proof is a committed artifact, passing test output, or generated metrics file. Vague progress does not count.

## Final Recommendation Before Starting

Approve this plan only if we agree to these three constraints:

1. P0 benchmark first; LoRA is cut unless we are clearly ahead.
2. No external API dependency.
3. No test evaluation until validation gates and freeze manifest are done.

If approved, the first implementation move is Phase -1 setup plus Phase 0 scraper/matcher.
