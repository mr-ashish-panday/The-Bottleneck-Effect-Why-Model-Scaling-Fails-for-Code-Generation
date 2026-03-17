# Research Log

Last updated: 2026-03-13 16:55:01 +05:45

## Current paper state

- Paper title: `The Bottleneck Effect: Why Model Scaling Fails for Code Generation`
- Current overall readiness estimate: `92-93%`
- Current strongest framing:
  - small-model regime result
  - mechanism-first paper with benchmark evidence
  - scale alone does not induce robust code generation in small general LMs

## Completed benchmark results

### HumanEval main

- GPT-2 Small: `5.2%`
- GPT-2 Medium: `4.8%`
- CodeGen-350M: `37.4%`
- Problem-level significance:
  - GPT-2 Small vs GPT-2 Medium: `+0.37%`, `p=0.4907`
  - GPT-2 Small vs CodeGen: `-32.23%`, `p=0.0002`
  - GPT-2 Medium vs CodeGen: `-32.63%`, `p=0.0002`

### MBPP full

- GPT-2 Small: `0.00%`
- GPT-2 Medium: `0.00%`
- CodeGen-350M: `7.39%`
- CodeGen pass@5: `22.96%`
- Problem-level significance:
  - GPT-2 Small vs GPT-2 Medium: `0.00%`, `p=1.0000`
  - GPT-2 Small vs CodeGen: `-7.39%`, `p=0.0001`
  - GPT-2 Medium vs CodeGen: `-7.39%`, `p=0.0001`

### HumanEval+ strict

- GPT-2 Small pass@1: `0.00%`
- GPT-2 Medium pass@1: `0.00%`
- CodeGen pass@1: `2.10%`

### LiveCodeBench release_v2

- GPT-2 Small:
  - tasks scored: `510`
  - pass@1: `0.0`
  - pass@5: `0.0`
  - pass@10: `0.0`
- GPT-2 Medium:
  - tasks scored: `511`
  - pass@1: `0.0`
  - pass@5: `0.0`
  - pass@10: `0.0`
- CodeGen-350M:
  - tasks scored: `511`
  - pass@1: `0.0001956947`
  - pass@5: `0.0009784736`
  - pass@10: `0.0019569472`
- Interpretation:
  - useful as a contamination-aware stress test
  - all three small models are effectively near-zero
  - strengthens the negative claim more than the positive CodeGen claim

## Completed mechanism results

### Full ablation

- GPT-2 Medium:
  - every single-layer ablation reduces success to `0%`
  - layer `12` uniquely shifts failures to `76.1% syntax / 23.9% runtime`
- CodeGen:
  - residual nonzero layers at `5, 7, 13, 18`
  - layer `13` preserves `29.5%` success under hard ablation

### Scaled ablation

- GPT-2 Medium layer `12`, subset `10 problems x 5 samples`
  - baseline: `6.0%`
  - `0.75`: `8.0%`
  - `0.50`: `2.0%`
  - `0.25`: `0.0%`
  - `0.00`: `0.0%`, with `28.0%` runtime
- CodeGen layer `13`, subset `10 problems x 5 samples`
  - baseline: `40.0%`
  - `0.75`: `28.0%`
  - `0.50`: `30.0%`
  - `0.25`: `2.0%`
  - `0.00`: `24.0%`, with `24.0%` runtime

### Probe and steering

- GPT-2 Medium layer-12 top-5 probe accuracy: `72.5%`
- Full-layer probe accuracy: `75.0%`
- Steering best condition:
  - baseline: `6.0%`
  - target `+2.0`: `12.0%`
- Steering controls:
  - random mean: `6.2%`
  - learned vector beats `19 / 20`
  - best random control reaches `14.0%`
- Interpretation:
  - constructive effect is real
  - specificity evidence is suggestive, not decisive

## Manuscript state

- `bottleneck.tex` updated to include:
  - full MBPP section
  - HumanEval+ section
  - LiveCodeBench section
  - scaled-ablation and steering sections
  - conservative LiveCodeBench discussion
  - updated limitations and conclusion
- `bottleneck.pdf` builds successfully
- Remaining LaTeX issues are normal layout warnings only

## Deep-search conclusions

### Prompt 1: CodeGen NL -> Multi -> Mono

- Strong reviewer-safe framing:
  - a `within-family continued-pretraining ladder`
  - not a pure distribution-only causal isolation
- What the official sources support:
  - same 350M family and same core architecture
  - sequential checkpoint lineage:
    - `NL -> Multi -> Mono`
  - large monotonic benchmark gains across the ladder
- Required caveats:
  - extra steps/tokens are added at later stages
  - The Pile already contains code
  - Python-heavy benchmarks naturally favor `Mono`
- Best low-compute follow-ups:
  - HumanEval ladder
  - sanitized MBPP ladder
  - prompt perplexity

### Prompt 2: MBPP+ / EvalPlus protocol

- MBPP+ must be treated as a separate EvalPlus benchmark
- Do not describe it as “MBPP with more tests”
- Must report together:
  - EvalPlus version
  - task count
  - decoding regime
  - whether metric is `Base` or `Base+Extra`
- Clean paper wording:
  - “MBPP+ evaluated with the official EvalPlus evaluator”

### Prompt 3: prompt perplexity interpretation

- Prompt perplexity is useful only as an auxiliary diagnostic
- Safe interpretation:
  - prompt/checkpoint compatibility
  - prompt familiarity / expectedness under the model
- Unsafe interpretation:
  - direct capability metric
  - causal explanation for the `NL -> Multi -> Mono` ladder
  - replacement for functional metrics like `pass@k`
- Important finding from the CodeGen paper:
  - benchmark performance improves monotonically across `NL -> Multi -> Mono`
  - official prompt-perplexity tables do not provide an equally clean monotone ladder
- Paper decision:
  - keep prompt perplexity as appendix or secondary evidence
  - main claims must stay benchmark-first
  - if included, word it as a descriptive within-family diagnostic only

## Current running job

- Last confirmed active job:
  - clean `MBPP+` run on the server
  - runner: `run_mbppplus_clean_validation.sh`
  - current stage at last live check: `gpt2`
  - worker:
    - `python scripts/generate_mbppplus_evalplus.py --config config_mbppplus_gpt2.yaml --resume --num_problems 378 --num_samples 20`
  - log:
    - `/home/ashish/paper11_code_execution_failures/outputs/logs/mbppplus_clean_validation.log`
- Last confirmed progress snapshot:
  - `18 / 378` GPT-2 MBPP+ tasks generated
- Important note:
  - the server route has been intermittently unstable
  - inability to check live status does not imply the job stopped

### Latest live checkpoint

- Checked live after the run had matured:
  - `gpt2` MBPP+ generation is complete
  - saved tasks: `378 / 378`
  - exported samples: `7560`
  - current stage: `EvalPlus` scoring for `gpt2`
  - `gpt2-medium` and `codegen` have not started generation yet
  - no finished MBPP+ summary JSON exists yet
- Process state at this checkpoint:
  - parent runner still alive
  - active `evalplus.evaluate mbpp --samples samples.jsonl` workers are running

### Newer live checkpoint

- `gpt2` clean `MBPP+` strict summary is now written:
  - file: `outputs/tables/evalplus_gpt2_mbppplus_summary.json`
  - `pass@1 = 0.0`
  - `pass@10 = 0.0`
- Runner progress after that:
  - `gpt2-medium` generation completed
  - `gpt2-medium` samples exported: `7560`
  - current stage: `EvalPlus` scoring for `gpt2-medium`
  - `codegen` has not started its clean `MBPP+` generation yet
- Current confirmed server timeline:
  - `[2026-03-13 15:12:20]` started `gpt2` EvalPlus scoring
  - `Wrote EvalPlus summary to outputs/tables/evalplus_gpt2_mbppplus_summary.json`
  - `[2026-03-13 23:46:52]` started clean `MBPP+` generation for `gpt2-medium`
  - `[2026-03-14 04:32:31]` started `gpt2-medium` EvalPlus scoring

### Latest live checkpoint

- Parent runner is still alive:
  - `bash run_mbppplus_clean_validation.sh`
- Active workers:
  - four `evalplus.evaluate mbpp --samples samples.jsonl` processes
- Latest log progress for `gpt2-medium` scoring:
  - `2385 / 7560` samples tested
  - about `32%` complete
  - elapsed scoring time shown in log: `2:05:33`
- Practical interpretation:
  - `gpt2-medium` strict scoring is progressing normally
  - `codegen` clean `MBPP+` generation has not started yet

### Newest live checkpoint

- `gpt2-medium` clean `MBPP+` strict summary is now written:
  - file: `outputs/tables/evalplus_gpt2_medium_mbppplus_summary.json`
  - `pass@1 = 0.0`
  - `pass@10 = 0.0`
- Queue progress after that:
  - `codegen` clean `MBPP+` generation completed
  - `codegen` exported `7560` samples
  - `[2026-03-14 17:54:31]` started `codegen` EvalPlus scoring
- Latest codegen strict-scoring progress seen in the log:
  - `3573 / 7560`
  - about `47%` complete
  - elapsed scoring time shown in log: `3:46:18`
- Practical interpretation:
  - the run is in the final strict-evaluation leg
  - no `codegen` summary file exists yet

### Queued next run

- Synced the next ladder files to the server:
  - `config_codegen_nl.yaml`
  - `config_codegen_multi.yaml`
  - `config_codegen_mono.yaml`
  - `run_codegen_pretraining_ladder.sh`
  - `scripts/summarize_codegen_ladder.py`
- Added and synced queue wrapper:
  - `run_after_mbppplus_codegen_ladder.sh`
- Queue behavior:
  - waits for the clean `MBPP+` runner and its EvalPlus workers to exit
  - then launches `run_codegen_pretraining_ladder.sh` automatically
- Server queue log:
  - `/home/ashish/paper11_code_execution_failures/outputs/logs/codegen_ladder_queue.log`
- Current queue-log state:
  - `[2026-03-14 21:50:39] Waiting for clean MBPP+ queue to finish`

### Completed overnight handoff

- Clean `MBPP+` queue finished and the queued handoff worked:
  - `[2026-03-15 01:57:32] MBPP+ queue finished; launching CodeGen ladder`
  - `[2026-03-15 01:57:32] CodeGen ladder launch submitted`
- Final clean `MBPP+` strict results:
  - `gpt2`: `pass@1 = 0.0`, `pass@10 = 0.0`
  - `gpt2-medium`: `pass@1 = 0.0`, `pass@10 = 0.0`
  - `codegen`: `pass@1 = 0.011`, `pass@10 = 0.064`
- CodeGen ladder run finished:
  - completed at `[2026-03-15 08:35:07]`
  - summary file: `outputs/tables/codegen_ladder_summary.json`
  - significance file: `outputs/tables/bootstrap_significance_codegen_ladder.json`
- CodeGen ladder benchmark results:
  - `CodeGen-NL`: success `30.60%`, syntax `69.22%`, runtime `0.18%`
  - `CodeGen-Multi`: success `28.46%`, syntax `71.43%`, runtime `0.11%`
  - `CodeGen-Mono`: success `37.76%`, syntax `62.20%`, runtime `0.05%`
- Pairwise success-rate differences reported by the ladder run:
  - `CodeGen-NL vs CodeGen-Multi`: `+2.50%`, `p = 0.0443`
  - `CodeGen-NL vs CodeGen-Mono`: `-6.64%`, `p = 0.0001`
  - `CodeGen-Multi vs CodeGen-Mono`: `-9.29%`, `p = 0.0001`

### Later active run discovered

- After the HumanEval ladder finished, a separate remaining queue was still active on the server:
  - `bash run_codegen_ladder_mbpp.sh`
  - worker:
    - `python scripts/generate_samples_safe.py --config config_mbpp_full_codegen_multi.yaml --resume --num_problems 257 --num_samples 20`
- Confirmed process:
  - PID `340966`
  - elapsed about `2:44`
  - GPU memory in use: about `1054 MiB`
- Current stage from the log:
  - `CodeGen ladder MBPP validation` is in the `config_mbpp_full_codegen_multi.yaml` generation leg
  - progress seen: `226 / 257` problems generated for `CodeGen-Multi MBPP`
- Important implication:
  - the server is not idle
  - the ladder MBPP pipeline still has remaining work after the HumanEval ladder completion

### Later idle checkpoint

- Rechecked the server on `2026-03-16`.
- Current process state:
  - no active `run_codegen_ladder_mbpp.sh`
  - no active `run_codegen_remaining_queue.sh`
  - no active GPU compute process related to the paper runs
- Output state:
  - final MBPP ladder summary files do **not** exist:
    - `outputs/tables/codegen_ladder_mbpp_summary.json`
    - `outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json`
  - prompt-perplexity outputs also do not exist yet
- Last confirmed place where the queue stopped:
  - `CodeGen-Mono` MBPP generation
  - log ended around `188 / 257` problems
- Practical interpretation:
  - the remaining MBPP ladder queue did not finish
  - it needs a resume / restart from the partial `CodeGen-Mono` checkpoint

### Diagnosis of the stop

- What I could confirm:
  - the queue did **not** end with a Python traceback in the log
  - `outputs/logs/codegen_remaining_queue.log` and `data/results_mbpp_full_codegen_mono/generated_samples.json` both stop around `2026-03-15 22:00`
  - later server reboot/shutdown entries happened **after** that stop window
  - `CodeGen-NL` and `CodeGen-Multi` MBPP evaluation outputs already exist
  - `CodeGen-Mono` MBPP had only partial generations saved: `188 / 257`
- What I could not prove:
  - exact system-side cause of the stop
  - previous-boot `journalctl` access is restricted for this user, so no authoritative OOM / kill entry was visible
- Best diagnosis:
  - the queue was terminated abruptly from outside Python during the `CodeGen-Mono` MBPP generation leg
  - this was not a clean script-level exception with a traceback

### Resume action

- Added local resume wrapper:
  - `resume_codegen_remaining_queue.sh`
- Synced it to the server and relaunched from the partial checkpoint.
- Current resumed worker:
  - `python scripts/generate_samples_safe.py --config config_mbpp_full_codegen_mono.yaml --resume --num_problems 257 --num_samples 20`
- Resume log:
  - `/home/ashish/paper11_code_execution_failures/outputs/logs/resume_codegen_remaining_queue.log`
- Resume state at launch:
  - `Resuming: Found 188 existing problems`
  - `Need to generate: 69 problems`
  - observed resumed pace: about `5s/problem` for the remaining mono MBPP generation

### Resume queue completed

- Rechecked on `2026-03-17`.
- Current server state:
  - no active paper-related process
  - no active GPU compute process for this project
- Resume queue completion markers:
  - `[2026-03-16 21:07:32] CodeGen ladder mechanism rerun finished`
  - `[2026-03-16 21:07:32] Resume queue finished`
- Completed output files now present:
  - `outputs/tables/codegen_ladder_mbpp_summary.json`
  - `outputs/tables/bootstrap_significance_codegen_ladder_mbpp.json`
  - `outputs/tables/prompt_ppl_codegen_nl_humaneval.json`
  - `outputs/tables/prompt_ppl_codegen_multi_humaneval.json`
  - `outputs/tables/prompt_ppl_codegen_mono_humaneval.json`
  - `outputs/tables/prompt_ppl_codegen_nl_mbpp.json`
  - `outputs/tables/prompt_ppl_codegen_multi_mbpp.json`
  - `outputs/tables/prompt_ppl_codegen_mono_mbpp.json`
  - `data/results_codegen_mono/ablation/layer_ablation_results.json`
- CodeGen MBPP ladder summary:
  - `CodeGen-NL MBPP`: success `0.019%`, syntax `73.17%`, runtime `12.18%`
  - `CodeGen-Multi MBPP`: success `1.226%`, syntax `62.06%`, runtime `11.36%`
  - `CodeGen-Mono MBPP`: success `2.763%`, syntax `36.17%`, runtime `18.40%`
- MBPP ladder significance highlights:
  - `NL vs Multi` success difference: `-1.206%`, `p = 0.0001`
  - `Multi vs Mono` success difference: about `-1.537%`
  - `Mono` is the strongest checkpoint on MBPP in the ladder
- HumanEval prompt perplexity means:
  - `CodeGen-NL`: `4.9314`
  - `CodeGen-Multi`: `5.5429`
  - `CodeGen-Mono`: `4.8680`
- Mechanism rerun final visible highlight:
  - `CodeGen-Mono` layer `13` remained the standout ablation layer
  - layer `13`: `29.0%` success, `20.3%` runtime after ablation

### Remaining roadmap queue started

- Synced remaining ladder files to the server:
  - `config_mbpp_full_codegen_nl.yaml`
  - `config_mbpp_full_codegen_multi.yaml`
  - `config_mbpp_full_codegen_mono.yaml`
  - `run_codegen_ladder_mbpp.sh`
  - `run_codegen_ladder_mechanism.sh`
  - `run_codegen_ladder_prompt_ppl.sh`
  - `run_codegen_remaining_queue.sh`
  - `scripts/compute_prompt_perplexity.py`
- First launch attempt exposed a path bug:
  - `compute_prompt_perplexity.py` had landed at repo root instead of `scripts/`
  - fixed by copying it to `/home/ashish/paper11_code_execution_failures/scripts/`
- Current running queue:
  - `run_codegen_remaining_queue.sh`
  - log: `/home/ashish/paper11_code_execution_failures/outputs/logs/codegen_remaining_queue.log`
- Current live state:
  - `[2026-03-15 11:58:43] Starting remaining CodeGen ladder queue`
  - active step: `Step 1/3: sanitized MBPP ladder`
  - active worker:
    - `python scripts/generate_samples_safe.py --config config_mbpp_full_codegen_nl.yaml --resume --num_problems 257 --num_samples 20`

### MBPP ladder slowdown checkpoint

- Correction: the earlier read of the active-process elapsed time was wrong.
- `ELAPSED 07:20` was minutes, not hours.
- So this was not a multi-hour stall.
- Updated live interpretation below replaces the earlier slowdown concern.

### MBPP ladder corrected live checkpoint

- Process state:
  - active worker still alive
  - `python scripts/generate_samples_safe.py --config config_mbpp_full_codegen_nl.yaml --resume --num_problems 257 --num_samples 20`
  - elapsed process time at recheck: about `7 minutes`
- Saved-output state:
  - `data/results_mbpp_full_codegen_nl/generated_samples.json`
  - `7` saved problem entries
  - last saved task IDs: `MBPP/14`, `MBPP/16`, `MBPP/17`, `MBPP/18`, `MBPP/19`
- Interpretation:
  - the remaining queue is progressing normally
  - observed pace is roughly around one MBPP problem per minute for the first model
  - the long remaining time now comes from the total queued workload, not from a stall

## New pipeline files added

### Clean MBPP+

- `scripts/generate_mbppplus_evalplus.py`
- `config_mbppplus_gpt2.yaml`
- `config_mbppplus_gpt2_medium.yaml`
- `config_mbppplus_codegen.yaml`
- `run_mbppplus_clean_validation.sh`

### CodeGen ladder

- `config_codegen_nl.yaml`
- `config_codegen_multi.yaml`
- `config_codegen_mono.yaml`
- `run_codegen_pretraining_ladder.sh`
- `run_codegen_ladder_mechanism.sh`
- `scripts/summarize_codegen_ladder.py`

### Ladder extensions prepared locally

- `scripts/compute_prompt_perplexity.py`
- `config_mbpp_full_codegen_nl.yaml`
- `config_mbpp_full_codegen_multi.yaml`
- `config_mbpp_full_codegen_mono.yaml`
- `run_codegen_ladder_mbpp.sh`
- `run_codegen_ladder_prompt_ppl.sh`

## Next run order

1. Finish clean `MBPP+`
2. Run `CodeGen NL -> Multi -> Mono` on HumanEval
3. Run the same ladder on sanitized MBPP
4. Run ladder prompt-perplexity analysis
5. Run mechanism follow-up on the best ladder layers
6. Update manuscript with the ladder and strict MBPP+ results

## Logging rule going forward

- Every major run, result, framing change, and new script should be appended here.
- Do not rely only on chat history for research state.

## 2026-03-17 manuscript integration pass

- Updated `bottleneck.tex` to integrate the final strict-validation and within-family control results.
- Added MBPP+ as a separate EvalPlus section and kept it distinct from plain MBPP.
- Added the CodeGen-350M `NL -> Multi -> Mono` continued-pretraining ladder to the main manuscript.
- Added a conservative prompt-perplexity note as secondary evidence only.
- Added discussion/conclusion language that treats the ladder as a within-family continued-pretraining control, not a matched-compute causal isolation.
- Rebuilt `bottleneck.pdf` successfully. Remaining LaTeX issues are layout-only warnings (underfull boxes / float placement), not broken citations or failed compilation.

## 2026-03-17 final follow-up + GitHub snapshot

- Launched the remaining high-ROI experiment: targeted scaled follow-ups on the CodeGen ladder signature layers.
- Runner: `run_codegen_ladder_scaled_followups.sh`
- Signature layers:
  - `CodeGen-NL`: layer `11`
  - `CodeGen-Multi`: layer `7`
  - `CodeGen-Mono`: layer `13`
- Follow-up settings:
  - `20` problems
  - `10` samples per problem
  - scales `0.75, 0.5, 0.25, 0.0`
- Remote log:
  - `/home/ashish/paper11_code_execution_failures/outputs/logs/codegen_ladder_scaled_followups.log`
- Synced finished server tables/logs into the local repo and pushed the full paper/artifact snapshot to GitHub.
- Pushed commit:
  - `222a7cd` (`Add final benchmarks, manuscript updates, and research artifacts`)
