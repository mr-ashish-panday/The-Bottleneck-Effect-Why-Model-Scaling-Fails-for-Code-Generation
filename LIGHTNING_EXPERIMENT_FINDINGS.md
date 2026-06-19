# Lightning Experiment Findings Ledger

Purpose: capture every GPU run, result, failed attempt, interpretation, and
paper-writing consequence for the Bottleneck Effect paper.

This file is intentionally local-first. Do not rely on chat history for results.

## Safety Rules

- Do not start paid Lightning compute without explicit approval in chat.
- Always run `SMOKE=1` before a full phase on a new Lightning studio/runtime.
- Never run `PHASE=all` first.
- On a small Lightning T4 runtime, prefer `t4_priority` first. It is the
  controlled, T4-safe subset of the larger `core_scaling` plan.
- Prefer full `core_scaling` only on a larger GPU/runtime because it fixes the
  main reviewer weakness: controlled general-LM scaling evidence.
- Record every result here before using it in the manuscript.

## Current Objective

Upgrade the rejected TMLR version into a credible lower-Q1 / strong-Q2 journal
submission by adding controlled experiments, not random extra benchmark volume.

## Venue Decision

- Primary target: `Journal of Systems and Software`.
- Reason: the upgraded paper is strongest as an empirical software-engineering
  study of LLM code-generation reliability, benchmark sensitivity, and failure
  modes, not as a pure ML-theory paper.
- Backup target: `Information and Software Technology`.
- Stretch target only if the mechanistic claims become much stronger:
  `Neurocomputing`.
- Paper-writing consequence: prioritize reproducible empirical controls,
  modern-code-model validation, benchmark robustness, decoding/prompt controls,
  and artifact traceability over broad mechanistic overclaiming.

## Primary Reviewer Weakness to Fix

The current paper's main vulnerability is causal control:

- GPT-2 vs CodeGen is too confounded by model family, tokenizer, data, and
  training recipe.
- The paper needs a real scaling curve for general-language LMs.
- It also needs a second same-family general-LM scaling curve.
- Prompt, decoding, and extraction controls are needed so reviewers cannot call
  the result an artifact.

## Planned Run Order

0. `t4_priority`
   - T4-safe priority phase for the current Lightning GPU
   - GPT-2 Small/Medium/Large
   - Pythia 70M/160M/410M
   - CodeGen NL/Multi/Mono 350M
   - HumanEval with 20 samples/task
   - MBPP with 10 samples/task

1. `core_scaling`
   - GPT-2 Small/Medium/Large/XL
   - Pythia 70M/160M/410M/1B
   - CodeGen NL/Multi/Mono 350M
   - HumanEval, MBPP, MBPP+

2. `modern_code_validation`
   - Qwen2.5-Coder-0.5B
   - DeepSeek-Coder-1.3B-base
   - HumanEval, MBPP, MBPP+

3. `decoding_robustness`
   - low, standard, and high temperature
   - HumanEval and MBPP

4. `prompt_robustness`
   - canonical HumanEval prompt
   - signature-only prompt
   - comment-plus-signature prompt

5. `livecodebench_stress`
   - contamination-aware boundary result
   - use as calibration, not as the main positive claim

## Next-Run Decision After Modern Validation

- Do not automatically launch the full built-in `prompt_robustness` phase on
  the T4: it contains 15 HumanEval jobs at 100 samples/task.
- Do not automatically launch the full built-in `decoding_robustness` phase on
  the T4: it contains 30 HumanEval/MBPP jobs.
- Preferred next JSS-aligned run if the GPU remains available:
  `jss_prompt_robustness_20s`, using separate output directories so official
  prompt-robustness directories are not under-sampled.
- Targeted subset:
  - GPT-2 Medium, CodeGen-350M-Mono, and Qwen2.5-Coder-0.5B;
  - HumanEval canonical, signature-only, and comment-plus-signature prompts;
  - 164 tasks, 20 samples/task;
  - local execution evaluation first, with strict EvalPlus only if the local
    result is worth spending extra CPU time.
- Reason: for JSS, this directly addresses a likely software-engineering
  reviewer objection: whether the bottleneck result is a prompt-format artifact.

## Run Log

### 2026-05-11

- Venue decision locked for the next rewrite: target `Journal of Systems and
  Software` first.
- Lightning SSH access restored for the active Lightning studio. The exact SSH
  host is intentionally not stored in this local findings ledger.
- Previous `modern_code_validation` attempt had completed Qwen2.5-Coder-0.5B
  HumanEval generation and local evaluation, but stopped before continuing
  because `evalplus.evaluate` was not available.
- Environment repair on the active Lightning T4:
  - installed EvalPlus into the project venv for the missing CLI check;
  - found that the project venv lacked `torch`;
  - identified the CUDA-capable Python as
    `/home/zeus/miniconda3/envs/cloudspace/bin/python`;
  - installed the project Lightning requirements into the cloudspace Python;
  - relaunched `modern_code_validation` with `VENV_PATH` disabled so the runner
    uses the CUDA-capable cloudspace Python.
- Started a tiny CUDA keepalive process while EvalPlus performs CPU-heavy
  scoring, because this Lightning studio may stop if the GPU is idle for too
  long.
- Current remote run:
  `outputs/logs/heavy_modern_code_validation_cloudspace_20260511.log`.
- Current remote keepalive log:
  `outputs/logs/gpu_keepalive_bottleneck_jss.log`.
- Current status when recorded: Qwen2.5-Coder-0.5B HumanEval reused all
  16,400 saved samples, re-evaluated successfully with 6,885 successful samples,
  and entered HumanEval+ EvalPlus scoring.
- Qwen2.5-Coder-0.5B HumanEval EvalPlus completed under the resumed
  `modern_code_validation` run:
  - local evaluator before EvalPlus: 16,400 samples, 6,885 successful samples
    (`42.0%` sample-level success), 320 runtime errors, 65 wrong outputs, 7
    timeouts, and 9,123 syntax errors;
  - EvalPlus HumanEval base tests: pass@1 `0.095`, pass@10 `0.349`,
    pass@100 `0.634`;
  - EvalPlus HumanEval+ base+extra tests: pass@1 `0.080`, pass@10 `0.314`,
    pass@100 `0.585`.
- After HumanEval+ scoring, the run advanced to
  `modern_code_validation__qwen25_coder_05b__mbpp__standard` and started GPU
  generation for 257 MBPP tasks with 20 samples per task.
- Qwen2.5-Coder-0.5B MBPP generated 257 tasks with exactly 20 samples per task
  and completed local evaluation:
  - 5,140 total samples;
  - 1,042 successful samples under the local evaluator.
- After MBPP local evaluation, the runner entered extraction-sweep checks for
  the MBPP outputs.
- Generic EvalPlus rescoring of the sanitized 257-task MBPP split failed with
  `AssertionError: Missing problems in samples` because the sanitized MBPP task
  IDs do not match the EvalPlus MBPP dataset. This is not counted as a model
  result. I wrote an explicit skipped summary at
  `outputs/tables/evalplus_modern_code_validation__qwen25_coder_05b__mbpp__standard__canonical_mbpp_summary.json`
  so the runner can continue to the dedicated MBPP+ job instead.
- Relaunched `modern_code_validation` as resume2 with the cloudspace CUDA Python:
  `outputs/logs/heavy_modern_code_validation_cloudspace_resume2_20260511.log`.
- Qwen2.5-Coder-0.5B MBPP+ completed dedicated EvalPlus scoring:
  - official MBPP+ task coverage: 378 tasks;
  - generated/evaluated samples: 7,560 samples (20 per task);
  - base-test pass statuses: 2,038 / 7,560;
  - base+extra pass statuses: 1,726 / 7,560;
  - EvalPlus wrapper metrics: pass@1 `0.222`, pass@10 `0.575`;
  - summary path:
    `outputs/tables/evalplus_modern_code_validation__qwen25_coder_05b__mbppplus__standard__canonical_mbppplus_summary.json`.
- Current resume2 status when recorded: DeepSeek-Coder-1.3B HumanEval
  generation active on the T4, with 133 / 164 HumanEval tasks generated and
  GPU utilization near full load.
- Pre-created the same explicit non-result skip summary for DeepSeek-Coder-1.3B
  sanitized MBPP EvalPlus:
  `outputs/tables/evalplus_modern_code_validation__deepseek_coder_13b__mbpp__standard__canonical_mbpp_summary.json`.
  Reason: the sanitized 257-task MBPP IDs do not match EvalPlus MBPP IDs, so
  generic EvalPlus rescoring is invalid; the valid strict MBPP evidence is the
  separate dedicated MBPP+ job.
- DeepSeek-Coder-1.3B HumanEval generation and local evaluation completed:
  - 164 HumanEval tasks;
  - 16,400 generated samples;
  - 129 successful samples under the local evaluator.
- The run then entered CPU-heavy EvalPlus HumanEval scoring. I manually sent a
  tiny CUDA pulse at `2026-05-11T09:52Z` because GPU memory had dropped to
  `0 MiB` during EvalPlus and the Lightning studio should not sit idle.
- DeepSeek-Coder-1.3B HumanEval EvalPlus completed:
  - HumanEval base tests: pass@1 `0.0`, pass@10 `0.0`, pass@100 `0.0`;
  - HumanEval+ base+extra tests: 0 / 16,400 samples passed;
  - summary path:
    `outputs/tables/evalplus_modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical_humaneval_summary.json`.
- The run then advanced to DeepSeek-Coder-1.3B sanitized MBPP generation. Status
  when recorded: 54 / 257 tasks generated with 20 samples/task and GPU
  utilization near full load.
- DeepSeek-Coder-1.3B sanitized MBPP completed local evaluation:
  - 257 tasks;
  - 5,140 samples;
  - 0 successful samples under the local evaluator.
- The pre-created sanitized-MBPP skip summary worked: the runner skipped the
  invalid generic EvalPlus MBPP rescore and advanced to the dedicated MBPP+
  job.
- Current status when recorded: DeepSeek-Coder-1.3B MBPP+ generation active,
  29 / 378 tasks generated with 20 samples/task.
- DeepSeek-Coder-1.3B MBPP+ generation completed:
  - 378 MBPP+ tasks;
  - 7,560 generated samples;
  - the run entered CPU-heavy EvalPlus MBPP+ scoring.
- I manually sent another tiny CUDA pulse at `2026-05-11T10:59Z` because GPU
  memory had dropped to `0 MiB` during MBPP+ EvalPlus scoring.
- DeepSeek-Coder-1.3B MBPP+ EvalPlus completed:
  - official MBPP+ task coverage: 378 tasks;
  - generated/evaluated samples: 7,560 samples (20 per task);
  - base-test pass statuses: 0 / 7,560;
  - base+extra pass statuses: 0 / 7,560;
  - EvalPlus wrapper metrics: pass@1 `0.0`, pass@10 `0.0`;
  - summary path:
    `outputs/tables/evalplus_modern_code_validation__deepseek_coder_13b__mbppplus__standard__canonical_mbppplus_summary.json`.
- `modern_code_validation` completion audit passed:
  - `Audited 6 jobs`;
  - `complete: 6`;
  - audit manifest written to `outputs/tables/heavy_rebuttal_coverage_audit.json`.
- Pulled completed `modern_code_validation` backup locally:
  `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\modern_code_validation_complete_20260511_170625.tar.gz`
  (`9.26 MB`).
- Added and launched the targeted JSS prompt-robustness subset:
  - local script:
    `scripts/run_jss_prompt_robustness_20s.sh`;
  - remote log:
    `outputs/logs/jss_prompt_robustness_20s_20260511.log`;
  - remote output root:
    `data/results_heavy_rebuttal/jss_prompt_robustness_20s`;
  - remote table root:
    `outputs/tables/jss_prompt_robustness_20s`;
  - first job started successfully:
    `prompt_robustness__gpt2_medium__humaneval__standard__canonical`.
- JSS prompt-robustness first job completed:
  - `prompt_robustness__gpt2_medium__humaneval__standard__canonical`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 69 successful samples under the local evaluator.
- Second targeted prompt job started:
  `prompt_robustness__gpt2_medium__humaneval__standard__signature_only`.
- JSS prompt-robustness second job completed:
  - `prompt_robustness__gpt2_medium__humaneval__standard__signature_only`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 63 successful samples under the local evaluator.
- Third targeted prompt job started:
  `prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature`.
- JSS prompt-robustness third job completed:
  - `prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 35 successful samples under the local evaluator.
- GPT-2 Medium prompt-control block result:
  - canonical: 69 / 3,280 successful samples;
  - signature-only: 63 / 3,280 successful samples;
  - comment-plus-signature: 35 / 3,280 successful samples.
- Fourth targeted prompt job started:
  `prompt_robustness__codegen_mono_350m__humaneval__standard__canonical`.
- JSS prompt-robustness fourth job completed:
  - `prompt_robustness__codegen_mono_350m__humaneval__standard__canonical`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 1,841 successful samples under the local evaluator.
- Fifth targeted prompt job started:
  `prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only`.
- JSS prompt-robustness fifth job completed:
  - `prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 1,398 successful samples under the local evaluator.
- Sixth targeted prompt job started:
  `prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature`.
- JSS prompt-robustness sixth job completed:
  - `prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 2,334 successful samples under the local evaluator.
- CodeGen-350M-Mono prompt-control block result:
  - canonical: 1,841 / 3,280 successful samples;
  - signature-only: 1,398 / 3,280 successful samples;
  - comment-plus-signature: 2,334 / 3,280 successful samples.
- Seventh targeted prompt job started:
  `prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical`.
- JSS prompt-robustness seventh job completed:
  - `prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 1,364 successful samples under the local evaluator.
- Eighth targeted prompt job started:
  `prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only`.
- JSS prompt-robustness eighth job completed:
  - `prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 1,614 successful samples under the local evaluator.
- Ninth targeted prompt job completed:
  - `prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature`;
  - 164 HumanEval tasks;
  - 3,280 samples;
  - 1,795 successful samples under the local evaluator.
- Qwen2.5-Coder-0.5B prompt-control block result:
  - canonical: 1,364 / 3,280 successful samples;
  - signature-only: 1,614 / 3,280 successful samples;
  - comment-plus-signature: 1,795 / 3,280 successful samples.
- Targeted JSS prompt-robustness block completed and backed up:
  - 9 / 9 job summaries;
  - aggregate summary:
    `outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json`;
  - all jobs cover 164 HumanEval tasks with exactly 20 samples per task;
  - local backup:
    `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_prompt_robustness_20s_complete_20260512_093854.tar.gz`;
  - compressed size: 47,215,781 bytes;
  - SHA-256:
    `010fa664738a2dca308105cc3fc2b075b84770fd306124485205630899acef91`.
- Added a cost-controlled targeted decoding-robustness script:
  `scripts/run_jss_decoding_robustness_20s.sh`.
  It reuses completed standard canonical baselines from
  `jss_prompt_robustness_20s` and spends GPU only on low/high-temperature
  HumanEval runs for GPT-2 Medium, CodeGen Mono 350M, and Qwen2.5-Coder-0.5B.
- Launched targeted JSS decoding-robustness subset on Lightning:
  - remote script:
    `scripts/run_jss_decoding_robustness_20s.sh`;
  - remote log:
    `outputs/logs/jss_decoding_robustness_20s_20260512.log`;
  - remote output root:
    `data/results_heavy_rebuttal/jss_decoding_robustness_20s`;
  - remote table root:
    `outputs/tables/jss_decoding_robustness_20s`;
  - remote PID: `164505`;
  - first standard canonical baseline reused successfully:
    `decoding_robustness__gpt2_medium__humaneval__standard__canonical`;
  - active first GPU job at remote time `2026-05-12 03:58:19 UTC`:
    `decoding_robustness__gpt2_medium__humaneval__low_temp__canonical`;
  - status at launch check: 14 / 164 HumanEval tasks generated, 20 samples
    per generated task, T4 using 877 MiB with 59% utilization.
- Targeted JSS HumanEval decoding-robustness subset completed:
  - GPT-2 Medium: low-temp 67, standard 69, high-temp 101 successful
    samples out of 3,280;
  - CodeGen Mono 350M: low-temp 1,563, standard 1,841, high-temp 1,763
    successful samples out of 3,280;
  - Qwen2.5-Coder 0.5B: low-temp 1,385, standard 1,364, high-temp 1,280
    successful samples out of 3,280;
  - all 9 jobs cover 164 HumanEval tasks with exactly 20 samples per task;
  - aggregate summary:
    `outputs/tables/jss_decoding_robustness_20s/aggregate_summary.json`;
  - local backup:
    `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_decoding_robustness_20s_complete_20260512_111426.tar.gz`;
  - compressed size: 11,986,417 bytes;
  - SHA-256:
    `6bf825fe56ca4efe180d6c18596af6abbb7021199d9eb270e710e97bdf891b17`.
- Added targeted MBPP decoding script:
  `scripts/run_jss_mbpp_decoding_10s.sh`.
  It reuses CodeGen Mono and Qwen standard MBPP baselines from completed T4
  runs and spends GPU only on low/high-temperature MBPP controls.
- Launched targeted MBPP decoding subset on Lightning:
  - remote script:
    `scripts/run_jss_mbpp_decoding_10s.sh`;
  - remote log:
    `outputs/logs/jss_mbpp_decoding_10s_20260512.log`;
  - remote output root:
    `data/results_heavy_rebuttal/jss_mbpp_decoding_10s`;
  - remote table root:
    `outputs/tables/jss_mbpp_decoding_10s`;
  - remote PID: `173982`;
  - first standard MBPP baseline reused successfully:
    `decoding_robustness__codegen_mono_350m__mbpp__standard`
    = 220 / 2,570 successful samples;
  - active first GPU job at remote time `2026-05-12 05:32:51 UTC`:
    `decoding_robustness__codegen_mono_350m__mbpp__low_temp`;
  - status at launch check: 10 / 257 MBPP tasks generated, 10 samples per
    generated task, T4 using 983 MiB with 31% utilization.
- Targeted MBPP decoding subset completed:
  - CodeGen Mono 350M: low-temp 371, standard 220, high-temp 128 successful
    samples out of 2,570;
  - CodeGen Mono 350M task coverage: low-temp 79, standard 88, high-temp
    66 tasks with at least one success;
  - Qwen2.5-Coder 0.5B: low-temp 741, standard 488, high-temp 336 successful
    samples out of 2,570;
  - Qwen2.5-Coder 0.5B task coverage: low-temp 128, standard 141, high-temp
    123 tasks with at least one success;
  - all 6 jobs cover 257 MBPP tasks with exactly 10 samples per task;
  - aggregate summary:
    `outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json`;
  - local backup:
    `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_mbpp_decoding_10s_complete_20260512_124037.tar.gz`;
  - compressed size: 16,944,291 bytes;
  - SHA-256:
    `3394fb3a5026cba2f2ca51cfc7934d6ecd0fbaf0444dfdbda47788a8aecb280e`.
- Copied JSS aggregate summaries into the local repo:
  - `outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json`;
  - `outputs/tables/jss_decoding_robustness_20s/aggregate_summary.json`;
  - `outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json`.
- Added paper-facing summary table:
  `outputs/tables/jss_targeted_robustness_summary.md`.
- Added and rendered Figure 15:
  - script:
    `scripts/create_figure15_jss_robustness_controls.py`;
  - PNG:
    `outputs/figures/figure15_jss_robustness_controls.png`;
  - PDF:
    `outputs/figures/figure15_jss_robustness_controls.pdf`.
- Integrated the targeted robustness evidence into the manuscript:
  - added one Key Findings bullet on prompt/decoding bottlenecks;
  - added Results subsection `Prompt and Decoding Robustness Controls`;
  - inserted Figure 15 into the main paper;
  - added a conclusion paragraph noting protocol-induced bottlenecks.
- Recompiled `bottleneck.tex` with direct `pdflatex` passes after `latexmk`
  failed due missing Perl in MiKTeX. Output:
  `bottleneck.pdf` (30 pages). Compile status: successful, no undefined
  references or overfull boxes in the final log scan; remaining warnings are
  underfull vbox layout warnings.

### 2026-05-09

- No paid Lightning experiment started in this resumed session.
- Reason: previous attempt was stopped by Ashish due compute-cost risk.
- Safe local action completed: created this findings ledger.
- Lightning access check later confirmed the available GPU is a Tesla T4 with
  15 GB VRAM. The full H100/H200-style suite is too aggressive for this runtime,
  so the plan was narrowed to `t4_priority`.
- Ashish explicitly approved using the active paid Lightning GPU after the
  safety reset.
- Started remote background supervisor at `/teamspace/studios/this_studio/bottleneck_t4_work`.
  Remote supervisor PID: `13990`.
- Supervisor behavior: run `SMOKE=1 PHASE=t4_priority` first, then automatically
  run `CONFIRM_PAID_RUN=1 PHASE=t4_priority` only if smoke passes.
- Remote logs:
  - `outputs/logs/t4_priority_supervisor.log`
  - `outputs/logs/smoke_t4_priority.log`
  - `outputs/logs/heavy_t4_priority.log`
- Smoke passed at remote time `2026-05-09 04:37:58`.
- Full paid T4-safe phase started immediately after smoke:
  `CONFIRM_PAID_RUN=1 PHASE=t4_priority bash run_heavy_rebuttal_suite.sh`.
- Correction: the first full run resumed smoke outputs, which would have
  under-sampled the first two tasks. Stopped it immediately, archived the partial
  directory, and restarted the full phase from a clean result directory.
- Archived contaminated partial outputs:
  `/teamspace/studios/this_studio/bottleneck_t4_work/data/results_heavy_rebuttal/_archived_smoke_resume/t4_priority_20260509_044003`.
- Clean full supervisor PID: `15971`.
- Clean full logs:
  - `outputs/logs/t4_priority_full_clean_supervisor.log`
  - `outputs/logs/heavy_t4_priority_clean.log`
- Clean full run confirmed `Need to generate: 164 problems` for the first job,
  meaning smoke files are no longer poisoning final sample counts.
- Continuation watcher armed after `t4_priority`: remote PID `31830`.
  It waits for clean supervisor PID `15971` to finish and only starts
  `t4_stretch` if `t4_priority_full_clean_supervisor.log` contains
  `Clean supervisor finished`.
- Planned follow-up phase: `t4_stretch` only, not the full heavy suite. It adds
  GPT-2 XL, Pythia 1B, and Qwen2.5-Coder-0.5B on HumanEval/MBPP T4 settings.
  This keeps paid GPU usage focused without jumping to `PHASE=all`.
- `t4_priority` completed and audited cleanly at remote time `2026-05-09
  07:51:39`.
- Audit result: `Audited 18 jobs`, `complete: 18`. Audit written to
  `outputs/tables/heavy_rebuttal_coverage_audit.json`.
- `t4_stretch` follow-up started at remote time `2026-05-09 07:51:52`.
- Completed `t4_priority` artifacts were compressed on Lightning and pulled
  locally. Local backup:
  `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\t4_priority_complete_20260509_082704.tar.gz`.
  Size: about 30.4 MB compressed. It contains the completed t4_priority result
  tree, extraction tables, coverage audit, and run logs.
- `t4_stretch` completed and audited cleanly at remote time `2026-05-09
  09:30:43`.
- Audit result: `Audited 6 jobs`, `complete: 6`.
- Completed `t4_stretch` artifacts were compressed on Lightning and pulled
  locally. Local backup:
  `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\t4_stretch_complete_20260509_093414.tar.gz`.
  Size: about 3.9 MB compressed.
- Started follow-up `modern_code_validation` at remote time `2026-05-09
  09:34:14`. Remote supervisor PID: `33433`.
- Current `modern_code_validation` first job:
  `modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical`
  with 100 samples per HumanEval task.

## Results Table

| Date | Phase | Model | Benchmark | Samples | Result | Interpretation | Paper Action |
|---|---|---|---:|---:|---|---|---|
| 2026-05-09 | smoke t4_priority | GPT-2/Pythia/CodeGen T4 subset | HumanEval + MBPP | 2 smoke samples/problem | passed | all planned T4 models loaded/evaluated in smoke | full t4_priority launched |
| 2026-05-09 | t4_priority | GPT-2/Pythia/CodeGen T4 subset | HumanEval + MBPP | 20 HumanEval, 10 MBPP | flawed resume stopped and archived | smoke outputs would under-sample first two tasks | replaced with clean full run |
| 2026-05-09 | t4_priority clean | GPT-2/Pythia/CodeGen T4 subset | HumanEval + MBPP | 20 HumanEval, 10 MBPP | running via clean supervisor PID 15971 | first controlled T4-safe evidence run | monitor and audit |
| 2026-05-09 | t4_priority clean | GPT-2 Small | HumanEval | 20 | complete: 164 tasks, 3280 samples, 172 successful samples, 98 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | use cautiously; confirm pass@k formula before manuscript |
| 2026-05-09 | t4_priority clean | GPT-2 Small | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | important contrast with HumanEval; investigate prompt/evaluator difference before claim |
| 2026-05-09 | t4_priority clean | GPT-2 Medium | HumanEval | 20 | complete: 164 tasks, 3280 samples, 82 successful samples, 59 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | early controlled evidence that scaling GPT-2 Small->Medium does not monotonically improve code execution |
| 2026-05-09 | t4_priority clean | GPT-2 Medium | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | MBPP zero-success pattern persists after GPT-2 Small->Medium scaling |
| 2026-05-09 | t4_priority clean | GPT-2 Large | HumanEval | 20 | complete: 164 tasks, 3280 samples, 55 successful samples, 49 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | GPT-2 scaling curve is non-monotonic/downward on HumanEval in this run |
| 2026-05-09 | t4_priority clean | GPT-2 Large | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | MBPP remains zero-success across GPT-2 Small/Medium/Large |
| 2026-05-09 | t4_priority clean | Pythia 70M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1194 successful samples, 163 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | surprisingly strong same-family control; inspect Pythia data/code exposure before framing |
| 2026-05-09 | t4_priority clean | Pythia 70M | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | strong HumanEval does not transfer to MBPP under current protocol |
| 2026-05-09 | t4_priority clean | Pythia 160M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 825 successful samples, 147 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | Pythia 160M underperforms Pythia 70M on HumanEval; another non-monotonic scaling signal |
| 2026-05-09 | t4_priority clean | Pythia 160M | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | MBPP remains zero-success even for strong HumanEval Pythia checkpoints |
| 2026-05-09 | t4_priority clean | Pythia 410M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1162 successful samples, 164 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | Pythia rebounds at 410M; same-family scaling remains non-smooth rather than monotonic |
| 2026-05-09 | t4_priority clean | Pythia 410M | MBPP | 10 | complete: 257 tasks, 2570 samples, 4 successful samples, 4 tasks with >=1 success | sample count verified min=max=10, extraction sweep running/written | first nonzero MBPP result in the general-model ladder |
| 2026-05-09 | t4_priority clean | CodeGen NL 350M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1371 successful samples, 163 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | strongest HumanEval result so far; code-family ladder now underway |
| 2026-05-09 | t4_priority clean | CodeGen NL 350M | MBPP | 10 | complete: 257 tasks, 2570 samples, 1 successful sample, 1 task with >=1 success | sample count verified min=max=10, extraction sweep written | strong HumanEval result barely transfers to MBPP |
| 2026-05-09 | t4_priority clean | CodeGen Multi 350M | MBPP | 10 | complete: 257 tasks, 2570 samples, 57 successful samples, 32 tasks with >=1 success | sample count verified min=max=10, extraction sweep written | multilingual code pretraining transfers better to MBPP than CodeGen NL |
| 2026-05-09 | t4_priority clean | CodeGen Mono 350M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1874 successful samples, 164 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | best HumanEval result in t4_priority; code-specialized pretraining ladder peaks at Mono |
| 2026-05-09 | t4_priority clean | CodeGen Mono 350M | MBPP | 10 | complete: 257 tasks, 2570 samples, 220 successful samples, 88 tasks with >=1 success | sample count verified min=max=10, extraction sweep written | best MBPP result in t4_priority; strongest evidence for data/domain specialization |
| 2026-05-09 | t4_priority audit | all planned T4 priority jobs | HumanEval + MBPP | mixed | complete: 18/18 jobs | audit wrote `outputs/tables/heavy_rebuttal_coverage_audit.json` | phase complete; follow-up t4_stretch started |
| 2026-05-09 | t4_stretch | GPT-2 XL | HumanEval | 20 | complete: 164 tasks, 3280 samples, 81 successful samples, 50 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | GPT-2 XL does not rescue GPT-2 scaling; similar to Medium and below Small |
| 2026-05-09 | t4_stretch | GPT-2 XL | MBPP | 10 | complete: 257 tasks, 2570 samples, 0 successful samples | sample count verified min=max=10, extraction sweep written | GPT-2 remains zero-success on MBPP even at XL |
| 2026-05-09 | t4_stretch | Pythia 1B | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1107 successful samples, 164 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | Pythia 1B remains strong but below 70M/410M; non-smooth scaling persists |
| 2026-05-09 | t4_stretch | Pythia 1B | MBPP | 10 | complete: 257 tasks, 2570 samples, 7 successful samples, 7 tasks with >=1 success | sample count verified min=max=10, extraction sweep written | Pythia 1B improves over smaller Pythia checkpoints on MBPP but remains very weak |
| 2026-05-09 | t4_stretch | Qwen2.5-Coder 0.5B | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1361 successful samples, 164 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | modern small code model is strong on HumanEval but below CodeGen Multi/Mono in this protocol |
| 2026-05-09 | t4_stretch | Qwen2.5-Coder 0.5B | MBPP | 10 | complete: 257 tasks, 2570 samples, 488 successful samples, 141 tasks with >=1 success | sample count verified min=max=10, extraction sweep running/written | strongest MBPP result so far; modern code model transfers better to MBPP |
| 2026-05-09 | t4_stretch audit | all planned T4 stretch jobs | HumanEval + MBPP | mixed | complete: 6/6 jobs | audit wrote `outputs/tables/heavy_rebuttal_coverage_audit.json` | phase complete; modern_code_validation started |
| 2026-05-09 | modern_code_validation | Qwen2.5-Coder 0.5B | HumanEval | 100 | running | first full-budget modern validation job started | monitor and record when complete |
| 2026-05-09 | t4_priority clean | CodeGen Multi 350M | HumanEval | 20 | complete: 164 tasks, 3280 samples, 1689 successful samples, 164 tasks with >=1 success | sample count verified min=max=20, extraction sweep written | strongest HumanEval result so far; code-pretraining ladder improves over CodeGen NL |
| 2026-05-11 | modern_code_validation audit | Qwen2.5-Coder 0.5B + DeepSeek-Coder 1.3B | HumanEval + MBPP/MBPP+ | mixed | complete: 6/6 jobs | audit wrote `outputs/tables/heavy_rebuttal_coverage_audit.json`; local backup pulled | use Qwen EvalPlus as strict modern-code validation; treat DeepSeek zero as an anomaly to inspect before manuscript use |
| 2026-05-12 | jss_prompt_robustness_20s | GPT-2 Medium | HumanEval prompt controls | 20 x 3 styles | complete: canonical 69, signature-only 63, comment-plus-signature 35 successful samples | 9-job aggregate confirms min=max=20 samples/task | GPT-2 remains weak and syntax-error dominated under prompt changes |
| 2026-05-12 | jss_prompt_robustness_20s | CodeGen Mono 350M | HumanEval prompt controls | 20 x 3 styles | complete: canonical 1841, signature-only 1398, comment-plus-signature 2334 successful samples | every prompt style solved at least one sample for all 164 tasks | strong evidence that code-specialized models are prompt-interface sensitive |
| 2026-05-12 | jss_prompt_robustness_20s | Qwen2.5-Coder 0.5B | HumanEval prompt controls | 20 x 3 styles | complete: canonical 1364, signature-only 1614, comment-plus-signature 1795 successful samples | every prompt style solved at least one sample for all 164 tasks | modern small code model also changes substantially with prompt format |
| 2026-05-12 | jss_prompt_robustness_20s backup | 9 targeted prompt jobs | HumanEval | 29,520 total samples | complete and pulled locally | backup hash `010fa664738a2dca308105cc3fc2b075b84770fd306124485205630899acef91` | artifact-safe; ready for figures/tables |
| 2026-05-12 | jss_decoding_robustness_20s | GPT-2 Medium | HumanEval decoding controls | 20 x 3 temperatures | complete: low 67, standard 69, high 101 successful samples | all jobs min=max=20 samples/task | high temperature helps slightly but does not escape syntax-error bottleneck |
| 2026-05-12 | jss_decoding_robustness_20s | CodeGen Mono 350M | HumanEval decoding controls | 20 x 3 temperatures | complete: low 1563, standard 1841, high 1763 successful samples | all jobs min=max=20 samples/task; every standard/high task has >=1 success | standard decoding is strongest; low temperature reduces coverage |
| 2026-05-12 | jss_decoding_robustness_20s | Qwen2.5-Coder 0.5B | HumanEval decoding controls | 20 x 3 temperatures | complete: low 1385, standard 1364, high 1280 successful samples | all jobs min=max=20 samples/task; all standard/high tasks have >=1 success | Qwen is mildly helped by low temperature but hurt by high temperature |
| 2026-05-12 | jss_decoding_robustness_20s backup | 9 targeted decoding jobs | HumanEval | 29,520 total samples | complete and pulled locally | backup hash `6bf825fe56ca4efe180d6c18596af6abbb7021199d9eb270e710e97bdf891b17` | artifact-safe; ready for figures/tables |
| 2026-05-12 | jss_mbpp_decoding_10s | CodeGen Mono 350M | MBPP decoding controls | 10 x 3 temperatures | complete: low 371, standard 220, high 128 successful samples | low has higher sample success but standard has broader known task coverage: 88 tasks vs 79 low | MBPP decoding changes the sample/coverage tradeoff |
| 2026-05-12 | jss_mbpp_decoding_10s | Qwen2.5-Coder 0.5B | MBPP decoding controls | 10 x 3 temperatures | complete: low 741, standard 488, high 336 successful samples | low has higher sample success but lower task coverage than standard: 128 vs 141 tasks | low temperature improves repeated success but narrows coverage |
| 2026-05-12 | jss_mbpp_decoding_10s backup | 6 targeted MBPP decoding jobs | MBPP | 15,420 total samples | complete and pulled locally | backup hash `3394fb3a5026cba2f2ca51cfc7934d6ecd0fbaf0444dfdbda47788a8aecb280e` | artifact-safe; ready for cross-benchmark decoding table |

## Findings to Carry Into Manuscript

- First valid clean result: GPT-2 Small on HumanEval generated exactly 20
  samples for each of 164 tasks. Raw evaluation found 172 successful samples
  out of 3280 total samples, with 98/164 tasks having at least one success.
- GPT-2 Small on MBPP generated exactly 10 samples for each of 257 tasks. Raw
  evaluation found 0 successful samples out of 2570 total samples.
- GPT-2 Medium on HumanEval generated exactly 20 samples for each of 164 tasks.
  Raw evaluation found 82 successful samples out of 3280 total samples, with
  59/164 tasks having at least one success.
- GPT-2 Medium on MBPP generated exactly 10 samples for each of 257 tasks. Raw
  evaluation found 0 successful samples out of 2570 total samples. Error mix:
  2546 syntax errors, 21 wrong outputs, and 3 runtime errors.
- Early controlled signal: GPT-2 Medium underperforms GPT-2 Small on HumanEval
  in this run. GPT-2 Small had 172 successful samples and 98 tasks with at
  least one success; GPT-2 Medium had 82 successful samples and 59 tasks with
  at least one success. This is directly relevant to the paper's thesis, but
  do not frame it strongly until GPT-2 Large and cross-family controls finish.
- MBPP remains at 0 successes for GPT-2 Small and GPT-2 Medium. This is useful
  as a stricter benchmark contrast, but it may also indicate that MBPP prompt
  formatting/evaluation is much harsher for general-language GPT-2 models.
- GPT-2 Large completed both HumanEval and MBPP. HumanEval dropped again:
  GPT-2 Small = 172 successful samples / 98 tasks with >=1 success; GPT-2
  Medium = 82 / 59; GPT-2 Large = 55 / 49. MBPP stayed at 0 successful samples
  across GPT-2 Small, Medium, and Large.
- Pythia 70M produced a much stronger HumanEval result: 1194 successful samples
  and 163/164 tasks with at least one success. Pythia 160M was lower: 825
  successful samples and 147/164 tasks with at least one success.
- Pythia 160M on MBPP generated exactly 10 samples for each of 257 tasks and
  still produced 0 successful samples. Error mix: 2319 syntax errors, 235 wrong
  outputs, and 16 runtime errors.
- Pythia 410M on HumanEval generated exactly 20 samples for each of 164 tasks.
  Raw evaluation found 1162 successful samples out of 3280, with every task
  having at least one successful sample. This is close to Pythia 70M's 1194
  successful samples but above Pythia 160M's 825 successful samples.
- Pythia 410M on MBPP generated exactly 10 samples for each of 257 tasks. Raw
  evaluation found 4 successful samples out of 2570, with 4/257 tasks having at
  least one success. Error mix: 1749 syntax errors, 253 wrong outputs, 562
  runtime errors, and 2 timeouts.
- MBPP zero-success pattern now spans GPT-2 Small/Medium/Large and Pythia
  70M/160M. This is a strong benchmark-sensitivity signal, not yet a final
  model-capability claim.
- Pythia 410M is the first general-model checkpoint in this T4 run to produce
  any MBPP successes, but the rate remains extremely low compared with
  HumanEval. This supports a benchmark-sensitivity / brittle code execution
  framing rather than a broad "can code" claim.
- CodeGen NL 350M on HumanEval generated exactly 20 samples for each of 164
  tasks. Raw evaluation found 1371 successful samples out of 3280, with
  163/164 tasks having at least one success. Error mix: 1829 syntax errors and
  80 runtime errors.
- CodeGen NL 350M is currently the strongest HumanEval result in the T4 run,
  above Pythia 70M/410M and all GPT-2 checkpoints. This is important for the
  paper because it supports a pretraining/distribution story over parameter
  count alone, but MBPP must finish before framing it strongly.
- CodeGen NL 350M on MBPP generated exactly 10 samples for each of 257 tasks.
  Raw evaluation found only 1 successful sample out of 2570, with 1/257 tasks
  having at least one success. Error mix: 1459 syntax errors, 857 runtime
  errors, 252 wrong outputs, and 1 timeout.
- CodeGen NL's HumanEval-to-MBPP gap is large: 1371 successful HumanEval
  samples versus 1 successful MBPP sample under the same model and T4 protocol.
  This is strong evidence that the observed capability is benchmark- and
  prompt-sensitive, not a robust general coding skill claim.
- CodeGen Multi 350M on HumanEval generated exactly 20 samples for each of 164
  tasks. Raw evaluation found 1689 successful samples out of 3280, and every
  task had at least one successful sample. Error mix: 1486 syntax errors, 103
  runtime errors, and 2 timeouts.
- CodeGen NL -> CodeGen Multi on HumanEval improves from 1371 to 1689
  successful samples at the same 350M scale. This is a high-value result for
  the paper because it isolates data/domain specialization more cleanly than
  parameter count.
- CodeGen Multi 350M on MBPP produced 57 successful samples out of 2570, with
  32/257 tasks having at least one success. This is a large improvement over
  CodeGen NL's 1 successful MBPP sample.
- CodeGen Mono 350M produced the strongest t4_priority results: 1874 successful
  HumanEval samples and 220 successful MBPP samples. It also solved at least
  one sample for all 164 HumanEval tasks and 88/257 MBPP tasks.
- CodeGen ladder at the same 350M scale shows a clean domain-specialization
  gradient: NL < Multi < Mono on both HumanEval and MBPP. This is one of the
  strongest paper-quality findings from the run.
- GPT-2 XL HumanEval in `t4_stretch` produced 81 successful samples and 50/164
  tasks with at least one success. That is far below GPT-2 Small's 172
  successful samples and 98/164 tasks, so GPT-2 scaling remains non-monotonic
  even after adding the 1.5B checkpoint.
- GPT-2 XL MBPP produced 0 successful samples out of 2570. The GPT-2 MBPP
  pattern is now consistently zero successes for Small, Medium, Large, and XL.
- Pythia 1B HumanEval produced 1107 successful samples out of 3280, with all
  164 tasks having at least one success. Pythia HumanEval trend is now:
  70M = 1194, 160M = 825, 410M = 1162, 1B = 1107 successful samples. This is
  strong evidence that same-family scaling is non-smooth rather than reliably
  monotonic in this setup.
- Pythia 1B MBPP produced 7 successful samples out of 2570, with 7/257 tasks
  having at least one success. This is higher than Pythia 410M's 4 successful
  samples and the smaller Pythia zero-success MBPP runs, but still tiny compared
  with HumanEval.
- Qwen2.5-Coder 0.5B HumanEval produced 1361 successful samples out of 3280,
  with all 164 tasks having at least one success. It is strong, but in this
  protocol it is below CodeGen Multi 350M (1689) and CodeGen Mono 350M (1874)
  on HumanEval.
- Qwen2.5-Coder 0.5B MBPP produced 488 successful samples out of 2570, with
  141/257 tasks having at least one success. This is the strongest MBPP result
  so far and is substantially above CodeGen Mono 350M's 220 successful MBPP
  samples.
- Qwen has a different profile from CodeGen: weaker than CodeGen Mono on
  HumanEval in this protocol, but stronger on MBPP. This is a valuable
  cross-benchmark generalization result.
- Targeted JSS prompt robustness shows that prompt format is not a cosmetic
  detail. CodeGen Mono 350M moved from 1,398 successful samples under
  signature-only to 2,334 under comment-plus-signature; Qwen2.5-Coder 0.5B
  moved from 1,364 canonical successes to 1,795 comment-plus-signature
  successes; GPT-2 Medium stayed weak across all three prompt styles.
- This supports the JSS version of the paper better than a pure scale story:
  execution success depends on model family, pretraining domain, benchmark, and
  interface protocol. Parameter count alone is an incomplete systems-level
  explanation.
- Prompt robustness should be framed as a controlled sensitivity analysis, not
  as final pass@k leaderboard evidence, because this targeted subset uses 20
  samples per task and local execution categories.
- Targeted decoding robustness adds a second sensitivity axis. GPT-2 Medium
  changes only from 67-101 successful samples and remains syntax-error
  dominated; CodeGen Mono is best at standard temperature; Qwen is best at low
  temperature by a small margin. This supports a model-specific systems story:
  decoding policy interacts with model family rather than uniformly fixing or
  breaking code generation.
- The prompt-control effect is larger than the HumanEval decoding-control
  effect for CodeGen Mono and Qwen in these targeted subsets. That is a useful
  JSS framing point because it shifts the paper from "scale failed" to
  "the code-generation pipeline has multiple bottlenecks, and scale is only one
  component."
- MBPP decoding shows a sample-success versus coverage tradeoff. Low
  temperature increases successful samples for both CodeGen Mono and Qwen, but
  it narrows distinct-task coverage relative to the standard baselines. This is
  a strong systems finding: optimizing one aggregate metric can make another
  reliability metric worse.
- Pythia HumanEval trend so far: 70M = 1194 successful samples, 160M = 825,
  410M = 1162. This is not simple monotonic scaling; it looks like a
  non-smooth capability curve and needs careful discussion.
- Early high-value paper angle: controlled scaling is not monotonic even within
  GPT-2 and Pythia in the T4-safe run. However, Pythia's pretraining data may
  include code, so this should be framed carefully as a same-family scaling
  control, not as a clean "general language only" control without qualification.
- Early pattern to verify: GPT-2 Small looks much better on HumanEval than MBPP
  under this local evaluator. Do not claim this as a scaling effect until
  GPT-2 Medium/Large and CodeGen comparisons finish, and inspect whether
  HumanEval prompt format makes success easier.
- This is not yet a manuscript claim by itself. It must be compared against
  GPT-2 Medium/Large, Pythia, and CodeGen under the same T4-safe protocol.

## 2026-05-12 JSS Formatting Closeout

- Venue lock: primary target is Journal of Systems and Software, with the
  manuscript framed as empirical software-engineering evidence about
  code-generation reliability, not as a TMLR-style ML theory submission.
- Converted `bottleneck.tex` from the TMLR style/header to Elsevier
  `elsarticle` preprint style with `\journal{Journal of Systems and Software}`.
- Removed double-blind/TMLR submission wording. JSS uses single-anonymized
  review, so the local submission build now contains the named author block.
- Added JSS-facing front-matter and submission statements: keywords, data
  availability, competing-interest declaration, funding declaration, and
  generative-AI-use declaration.
- Added `jss_highlights.txt` as a separate Elsevier highlights artifact.
- Re-rendered Figure 15 after moving the bottom-right legend; visually checked
  the standalone figure and the PDF page containing the figure.
- Verification: rebuilt `bottleneck.pdf` with `pdflatex`, `bibtex`, and two
  additional `pdflatex` passes. Final log has no LaTeX errors, no unresolved
  citations/references, and no overfull boxes; remaining warnings are underfull
  line-spacing warnings only.
- Visual verification: rendered and inspected page 1 of the JSS/Elsevier build
  and the Figure 15 page. Page 1 no longer contains the TMLR header or visible
  hyperlink boxes.
- Submission-support closeout: added `jss_cover_letter.txt`,
  `JSS_ARTIFACT_README.md`, `JSS_SUBMISSION_CHECKLIST.md`, and the verifier
  `scripts/audit_jss_submission_artifacts.py`.
- Claim-to-artifact audit: `JSS_CLAIM_ARTIFACT_AUDIT.md` reports 32 checks
  passed and 0 failed. The audit verifies all manuscript figure files, package
  readability, and key numeric claims for HumanEval, MBPP, CodeGen ladder,
  LiveCodeBench, JSS prompt/decoding controls, HumanEval+, and the syntax-error
  profile table.
- Final sanitized packages:
  `submission_jss_20260512_135646/jss_source_package.zip`
  SHA-256 `A321C0B0FC20B48C1366AF1ED870536E5EED10918B3B96D216A3E925E69F1271`;
  `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
  SHA-256 `31551AC28D359CE81367456053B46BBA8EC619AAD669E61F9F66B98C1E7BD376`.
  Both archives were validated with `tar -tf`; source has 23 entries and the
  cleaned supplement has 398 entries. A sensitive-token scan over the final staging
  folders found 0 hits for private keys, common API-token patterns, and
  Lightning SSH host strings; the reviewer-facing supplement excludes local
  planning notes with internal Lightning SSH host metadata.
- Data Availability was changed to point to the curated supplementary artifact
  archive instead of the full GitHub repository, because the working repository
  contains local run notes and should not be exposed wholesale at first
  submission.

## Open Questions

- Does decoding temperature change the same models as strongly as prompt
  format? This is the next highest-ROI JSS evidence gap.
- Should DeepSeek-Coder-1.3B's zero strict result be excluded from main claims
  until prompt compatibility is inspected? Current answer: yes, use it only as
  a validation anomaly unless inspected.

## Next Safe Command While Lightning GPU Is Active

```bash
# No new GPU run is currently justified for the JSS rewrite.
# Keep the Lightning keepalive running only if the instance must stay alive.
```

Do not launch the full built-in `decoding_robustness` phase on the T4 unless
there is a fresh reason to spend for the full 30-job, 100-sample/task sweep.
