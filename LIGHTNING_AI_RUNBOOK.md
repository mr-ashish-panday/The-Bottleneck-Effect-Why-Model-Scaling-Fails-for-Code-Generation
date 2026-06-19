# Lightning AI Runbook

Use this instead of the old college-server commands. The runner is now portable:
it uses the directory containing `run_heavy_rebuttal_suite.sh` as `ROOT` unless
you explicitly set `ROOT`.

## 1. Open a Lightning Studio

Choose a GPU runtime with enough memory. Start with a smaller GPU for smoke
testing, but use the strongest available GPU for `gpt2-xl`, `pythia-1b`, and
DeepSeek-Coder-1.3B.

## 2. Put the Repo in the Studio

Either clone/pull the repo or upload this folder into Lightning. The project
root must contain:

- `scripts/`
- `src/`
- `configs/heavy_rebuttal/`
- `run_heavy_rebuttal_suite.sh`
- `requirements-lightning.txt`

## 3. Setup

From the project root:

```bash
bash scripts/setup_lightning_ai.sh
```

If you want to use Lightning's already-active environment instead of creating
`venv/`:

```bash
USE_EXISTING_ENV=0 bash scripts/setup_lightning_ai.sh
```

That creates `venv/` with `--system-site-packages` so it can still see
Lightning's CUDA-compatible PyTorch.

## 4. Smoke Test First

On the current Tesla T4 runtime, do this before any full run:

```bash
SMOKE=1 PHASE=t4_priority bash run_heavy_rebuttal_suite.sh
```

This runs only a tiny subset and skips expensive EvalPlus/extraction summaries
into separate smoke result directories so it does not poison final result files.

## 5. Main Run Order

Run one phase at a time:

```bash
CONFIRM_PAID_RUN=1 PHASE=t4_priority nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_t4_priority.log 2>&1 &
tail -f outputs/logs/heavy_t4_priority.log
```

Only move to the larger phases if the GPU/runtime is large enough:

```bash
CONFIRM_PAID_RUN=1 PHASE=core_scaling nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_core_scaling.log 2>&1 &
tail -f outputs/logs/heavy_core_scaling.log
```

Audit:

```bash
python scripts/audit_heavy_rebuttal_outputs.py --phase core_scaling
```

Then:

```bash
CONFIRM_PAID_RUN=1 PHASE=modern_code_validation nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_modern_code_validation.log 2>&1 &
CONFIRM_PAID_RUN=1 PHASE=decoding_robustness nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_decoding_robustness.log 2>&1 &
CONFIRM_PAID_RUN=1 PHASE=prompt_robustness nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_prompt_robustness.log 2>&1 &
CONFIRM_PAID_RUN=1 PHASE=livecodebench_stress nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_livecodebench_stress.log 2>&1 &
```

## 6. Optional Prompt Perplexity

Prompt perplexity is useful but extra. Enable it only after generation/evaluation
is stable:

```bash
CONFIRM_PAID_RUN=1 RUN_PROMPT_PPL=1 PHASE=core_scaling nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_core_scaling_ppl.log 2>&1 &
```

## 7. Do Not Do This First

Do not run:

```bash
PHASE=all bash run_heavy_rebuttal_suite.sh
```

If one model, package, or evaluator fails, debugging becomes messy. Run phase by
phase.
