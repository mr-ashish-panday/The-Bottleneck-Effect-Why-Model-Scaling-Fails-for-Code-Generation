# Heavy Rebuttal Runbook

This run is designed to fix the paper's main reviewer weakness: the current
claim needs stronger controlled evidence, not just more benchmark numbers.

## What to Run First

Run the phases in this order:

0. `t4_priority`
   - T4-safe subset for the current Lightning runtime
   - GPT-2 Small/Medium/Large
   - Pythia 70M/160M/410M
   - CodeGen NL/Multi/Mono 350M
   - HumanEval and MBPP only
   - Run this first on Tesla T4 so paid compute produces usable evidence.

1. `core_scaling`
   - GPT-2 Small/Medium/Large/XL
   - Pythia 70M/160M/410M/1B
   - CodeGen NL/Multi/Mono 350M
   - HumanEval, MBPP, MBPP+
   - This is the main rejection fix.

2. `modern_code_validation`
   - Qwen2.5-Coder-0.5B and DeepSeek-Coder-1.3B-base
   - HumanEval, MBPP, MBPP+
   - This checks whether the code-pretraining story survives newer code models.

3. `decoding_robustness`
   - Low, standard, and high temperature settings
   - HumanEval and MBPP
   - This controls for "your conclusion is just a decoding artifact."

4. `prompt_robustness`
   - HumanEval canonical prompt, signature-only prompt, comment-plus-signature prompt
   - This controls for prompt-format advantage/disadvantage.

5. `livecodebench_stress`
   - Hard contamination-aware calibration
   - Use this to bound the claim, not as the main positive result.

## Lightning AI Commands

From the Lightning AI project root:

```bash
bash scripts/setup_lightning_ai.sh
SMOKE=1 PHASE=t4_priority bash run_heavy_rebuttal_suite.sh
CONFIRM_PAID_RUN=1 PHASE=t4_priority nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_t4_priority.log 2>&1 &
```

Check progress:

```bash
tail -f outputs/logs/heavy_core_scaling.log
```

Audit completion:

```bash
python scripts/audit_heavy_rebuttal_outputs.py --phase core_scaling
```

Then continue:

```bash
PHASE=modern_code_validation nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_modern_code_validation.log 2>&1 &
PHASE=decoding_robustness nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_decoding_robustness.log 2>&1 &
PHASE=prompt_robustness nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_prompt_robustness.log 2>&1 &
PHASE=livecodebench_stress nohup bash run_heavy_rebuttal_suite.sh > outputs/logs/heavy_livecodebench_stress.log 2>&1 &
```

## Important Rule

Do not run `PHASE=all` first. If one model or evaluator fails, it becomes hard
to isolate. Run phase-by-phase and audit after each phase.

## What Counts as Success

The heavy run is successful if it gives:

- a GPT-2 scaling curve across 124M, 355M, 774M, and 1.5B;
- an independent Pythia scaling curve;
- a CodeGen NL/Multi/Mono within-family ladder;
- strict EvalPlus results for HumanEval/MBPP where available;
- prompt-format and decoding controls;
- extraction-sweep evidence showing the ranking is not a parser artifact;
- a clear boundary result on LiveCodeBench.

The paper should then be reframed as:

> In the small-model regime, scaling general-language LMs does not reliably
> induce executable-code ability; code-specialized pretraining and the internal
> distribution of code-relevant computation matter more than parameter count
> alone.
