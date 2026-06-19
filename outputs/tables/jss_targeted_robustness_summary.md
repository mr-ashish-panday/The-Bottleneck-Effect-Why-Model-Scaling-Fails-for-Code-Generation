# JSS Targeted Robustness Summary

Generated from the local aggregate summaries copied from Lightning:

- `outputs/tables/jss_prompt_robustness_20s/aggregate_summary.json`
- `outputs/tables/jss_decoding_robustness_20s/aggregate_summary.json`
- `outputs/tables/jss_mbpp_decoding_10s/aggregate_summary.json`

All counts below are local execution results. Each cell is `successful samples / tasks with at least one success`.

## HumanEval Prompt Robustness

Each condition used 164 HumanEval tasks and 20 samples per task.

| Model | Canonical | Signature only | Comment + signature | Best sample count | Main signal |
|---|---:|---:|---:|---:|---|
| GPT-2 Medium | 69 / 48 | 63 / 53 | 35 / 33 | 69 | Prompt changes do not rescue the syntax-error bottleneck. |
| CodeGen Mono 350M | 1841 / 164 | 1398 / 164 | 2334 / 164 | 2334 | Prompt format changes success strongly even when task coverage stays saturated. |
| Qwen2.5-Coder 0.5B | 1364 / 164 | 1614 / 164 | 1795 / 164 | 1795 | Modern small code model is also prompt-interface sensitive. |

## HumanEval Decoding Robustness

Each condition used 164 HumanEval tasks and 20 samples per task.

| Model | Low temp | Standard | High temp | Best sample count | Main signal |
|---|---:|---:|---:|---:|---|
| GPT-2 Medium | 67 / 39 | 69 / 48 | 101 / 78 | 101 | Higher temperature helps slightly but remains far below code-specialized models. |
| CodeGen Mono 350M | 1563 / 156 | 1841 / 164 | 1763 / 164 | 1841 | Standard decoding is strongest; low temperature reduces distinct-task coverage. |
| Qwen2.5-Coder 0.5B | 1385 / 148 | 1364 / 164 | 1280 / 164 | 1385 | Low temperature slightly improves samples but lowers coverage. |

## MBPP Decoding Robustness

Each condition used 257 MBPP tasks and 10 samples per task.

| Model | Low temp | Standard | High temp | Best sample count | Best task coverage | Main signal |
|---|---:|---:|---:|---:|---:|---|
| CodeGen Mono 350M | 371 / 79 | 220 / 88 | 128 / 66 | 371 | 88 | Low temperature improves repeated sample success but narrows task coverage. |
| Qwen2.5-Coder 0.5B | 741 / 128 | 488 / 141 | 336 / 123 | 741 | 141 | Same sample-success vs coverage tradeoff appears on a stronger modern code model. |

## Paper-Ready Takeaways

- Prompt format is a large systems-level bottleneck: CodeGen Mono swings from 1398 to 2334 HumanEval successes without changing model size or benchmark.
- Decoding temperature is model- and benchmark-specific: there is no globally best setting across GPT-2, CodeGen, Qwen, HumanEval, and MBPP.
- MBPP exposes a reliability tradeoff: low temperature increases successful samples but reduces distinct-task coverage for both CodeGen Mono and Qwen.
- The JSS framing should not be a pure "scaling fails" claim. The stronger claim is that code-generation reliability is bottlenecked by model family, data specialization, benchmark distribution, prompt interface, decoding policy, and evaluation metric choice.

## Artifact Backups

| Run | Local backup | SHA-256 |
|---|---|---|
| HumanEval prompt robustness | `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_prompt_robustness_20s_complete_20260512_093854.tar.gz` | `010fa664738a2dca308105cc3fc2b075b84770fd306124485205630899acef91` |
| HumanEval decoding robustness | `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_decoding_robustness_20s_complete_20260512_111426.tar.gz` | `6bf825fe56ca4efe180d6c18596af6abbb7021199d9eb270e710e97bdf891b17` |
| MBPP decoding robustness | `C:\Users\Ashish\lightning_ai_codex\pulled_progress\backups\jss_mbpp_decoding_10s_complete_20260512_124037.tar.gz` | `3394fb3a5026cba2f2ca51cfc7934d6ecd0fbaf0444dfdbda47788a8aecb280e` |
