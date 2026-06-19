# GPU Upgrade Protocol

Purpose: run a stronger follow-on benchmark after the completed 48-hour sprint, without mutating or weakening the original locked-test result.

The original `trump-occurrence-sprint/` package remains frozen as the client-requested P0 deliverable. This upgrade is a separate V2 experiment designed to answer a sharper question: if the LLM direction gets a fair GPU-backed LoRA/QLoRA attempt, does it beat the timing/survival baseline on a fresh sealed evaluation?

## Non-Negotiable Rules

1. Do not overwrite or edit the original `trump-occurrence-sprint/` metrics, predictions, model artifacts, or locked-test result.
2. Create separate V2 artifacts under `trump-occurrence-gpu-upgrade/`.
3. Use a fresh sealed test protocol before training or selecting models.
4. Do not tune on the V1 test set.
5. Keep Baseline B as the main control unless a stronger non-neural baseline is pre-registered before V2 test scoring.
6. Calibrate on validation only.
7. Report `WIN`, `CONTINUE`, or `STOP` for V2 using the same or stricter decision rules.

## V2 Upgrade Options

### Option A: Fresh-Corpus Holdout

Use if new Rev transcripts can be collected beyond the original corpus.

- Keep the V1 corpus and artifacts intact.
- Scrape additional Rev Trump-related transcript pages.
- Freeze a new chronological V2 split before training.
- Train and validate only on the allowed V2 train/validation partitions.
- Score once on the new V2 test partition.

This is the cleanest route if enough new/fresh transcripts exist.

### Option B: Nested Validation With Untouched V1 Test

Use only if fresh corpus is too small.

- Treat V1 test as already spent and not eligible for model selection.
- Create a new split from train+validation only, or use cross-validation inside train+validation.
- Use V1 test only for historical comparison, not as a new claim of improvement.

This is weaker and should be marked as exploratory.

## GPU Challenger Plan

The GPU challenger should be stronger than the P0 feature path:

1. Train a small supervised scorer:
   - input: title, date/format, target, checkpoint percentage, expected remaining words, and transcript prefix window
   - output: probability target appears after checkpoint
2. Use LoRA or QLoRA on a compact local instruction/base model.
3. Train only on train checkpoint rows.
4. Keep the model nested over Baseline B:
   - either train a residual/logit-offset model
   - or fit a calibrated combiner where Baseline B remains an explicit fixed control
5. Run full vs blanked ablation with the same weights and calibrator.
6. Run a stronger local contamination probe with a GPU-capable generation model.
7. Run at least 3 seeds if time/compute permits.

## Candidate Models

Choose based on available GPU memory:

- 24 GB class: 7B QLoRA, short context, aggressive batching.
- 40-80 GB class: 7B or 8B QLoRA comfortably; possible longer prefix windows.
- H100/H200 class: 8B QLoRA with larger context, more seeds, stronger contamination generation.

Model choice must be logged before training.

## Required V2 Artifacts

```text
trump-occurrence-gpu-upgrade/
  GPU_UPGRADE_PROTOCOL.md
  README.md
  DECISION_V2.md
  config_v2.yaml
  data/
  benchmark/
  predictions/
  metrics/
  models/
  src/
  tests/
```

At minimum:

- `DECISION_V2.md`
- `config_v2.yaml`
- frozen split/target files
- train/validation/test checkpoint parquets
- Baseline B predictions
- LoRA/QLoRA full predictions
- LoRA/QLoRA blanked predictions
- validation and test metrics
- bootstrap deltas
- reliability/ECE report
- contamination report
- contamination-pruned metrics
- duration sensitivity
- cost/latency with GPU type and wall-clock time

## V2 Win Bar

V2 only improves the client story if all are true:

1. Challenger beats Baseline B on Brier on the sealed V2 test.
2. Improvement is at least `2%` relative or `0.005` absolute, or the result is clearly marked `CONTINUE` rather than `WIN`.
3. `t > 0` improvement is positive.
4. Contamination-pruned improvement remains positive.
5. Full beats blanked by enough to show transcript content carries signal.
6. ECE is not materially worse than Baseline B.
7. The result survives more than one seed, or seed count is clearly documented as a limitation.

## Client-Safe Framing

If V2 is run, frame it as:

> The original 48-hour sprint was completed exactly under the requested protocol and returned STOP. We then ran a separate GPU-backed upgrade to test whether the conclusion changes when the LLM direction gets a stronger LoRA/QLoRA challenger and a fresh sealed evaluation.

Do not frame V2 as fixing a failed first result. Frame it as a higher-power follow-up that respects the original benchmark integrity.
