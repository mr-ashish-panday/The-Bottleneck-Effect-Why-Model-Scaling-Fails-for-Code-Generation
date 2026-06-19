# Decision

Verdict: STOP

LoRA was not trained. The challenger was an API-free feature-extraction method: train-only target-adjacent content features plus a fixed duration-sensitivity term, nested over the timing/survival baseline in logit space and calibrated on validation.

## Locked Test Result

The LLM/feature challenger does not clear the continuation bar in this sprint result.

| Model | Test Brier | Test ECE | Test Log Loss | AUC Diagnostic |
|---|---:|---:|---:|---:|
| Baseline A: constant | 0.048416 | 0.022703 | 0.170778 | 0.923560 |
| Baseline B: timing/survival | 0.043319 | 0.018365 | 0.150796 | 0.946593 |
| Challenger full | 0.044434 | 0.018069 | 0.154722 | 0.943082 |
| Challenger blanked | 0.044460 | 0.014503 | 0.155044 | 0.941189 |

Absolute improvement over Baseline B: `-0.001115`.

Relative improvement over Baseline B: `-2.58%`.

Bootstrap 95% CI for challenger-full minus Baseline B Brier: `[0.000259, 0.002074]`. Positive means the challenger is worse.

## Required STOP Checks

STOP condition met: no improvement over Baseline B on all eval rows.

Additional decision evidence:

- Contamination-pruned result is identical because the deterministic local contamination probe flagged 0 test transcripts.
- On `t > 0` rows, Baseline B Brier was `0.039284`; challenger-full Brier was `0.040592`, so the challenger has higher error after the transcript has begun.
- Content ablation did not show useful test content signal. Full Brier was `0.044434`; blanked Brier was `0.044460`; both had higher error than Baseline B.
- Test content share is not meaningful because the full improvement is negative: `-0.022860`.

## Diagnostics

Validation cleared the pre-test gate:

- Baseline B validation Brier: `0.040780`.
- Challenger validation Brier: `0.040337`.
- Duration sensitivity monotonic fraction: `0.766`.
- Mean high-minus-low duration sensitivity: `0.074876`.

Contamination probe:

- Method: local deterministic generation with `distilgpt2`.
- Validation flagged transcripts: 0.
- Test flagged transcripts: 0.
- The zero-flag result should be read conservatively: `distilgpt2` is a small local probe, so this is a benchmark-scale contamination screen rather than proof that no public transcript is present in any larger model's pretraining data.
- No P0 contamination-probe deviation was used in the final test pass.

Cost and latency:

- API dollars: `0.00`.
- Feature path: local TF-IDF/SVD plus train-only adjacency features.
- Validation prediction speed: about `0.000890` seconds per row after feature artifact construction.
- This is plausible to run at benchmark scale.

## Reasoning

The timing/survival baseline is strong and generalizes better than this content challenger. The challenger found validation signal, but that signal did not survive the locked chronological test. Because the first STOP rule is triggered, the protocol-conforming decision is to stop this specific LLM/feature path unless the problem is reframed with a larger corpus, stronger content features, or a materially different model class and then rerun from scratch with a new locked test.
