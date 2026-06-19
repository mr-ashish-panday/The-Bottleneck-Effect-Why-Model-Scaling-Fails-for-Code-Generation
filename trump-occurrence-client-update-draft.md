# Client Update Draft

Hi [Name],

We completed the 48-hour offline benchmark sprint and packaged the full reproducible artifact set.

The result is `STOP` under the sprint's decision rules. The content/LLM-style challenger cleared the validation gate, so it was fair to spend the single locked test pass. On that locked chronological test, it did not improve over the timing/survival baseline on Brier score.

Key numbers:

- Baseline B test Brier: `0.043319`
- Challenger-full test Brier: `0.044434`
- Absolute improvement: `-0.001115`
- Relative improvement: `-2.58%`
- Bootstrap CI for challenger-full minus Baseline B Brier: `[0.000259, 0.002074]`

LoRA was not trained; per the sprint brief, it was P1, and the P0 benchmark was already enough to make the required decision. The package includes the code, cleaned corpus, saved raw Rev pages, train/validation/test checkpoints, predictions, metrics, diagnostics, contamination probe outputs, frozen model artifacts, and `DECISION.md`.

Recommended read order:

1. `DECISION.md`
2. `CLIENT_HANDOFF_AUDIT.md`
3. `README.md`

The useful readout is that the timing/survival baseline is a strong control, and this first content challenger did not add reliable held-out signal beyond it. If we continue, I would treat the next step as a new upgraded protocol: larger/fresher corpus or a GPU-backed LoRA/QLoRA challenger with a fresh sealed holdout, rather than tuning against the already-scored test set.
