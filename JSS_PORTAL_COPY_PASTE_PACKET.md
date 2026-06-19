# JSS Portal Copy-Paste Packet

Date: 2026-05-16.

Use this during the Editorial Manager upload. This is a local submission aid,
not a reviewer-facing artifact.

## Article Type

Regular research article

## Title

The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation

## Author

Ashish Pandey

## Affiliation

Department of Computer and Electronics Engineering, Khwopa College of
Engineering, Nepal

## Corresponding Email

ashishpanday9818@gmail.com

## Keywords

Code generation; Empirical software engineering; Large language models; Model
evaluation; Mechanistic interpretability; Software reliability

## Plain-Text Abstract

Scaling language models from 124M to 355M parameters improves performance across
natural language tasks, yet we observe the opposite for code generation: GPT-2
Medium (355M) achieves only 4.8% success on HumanEval compared to 5.2% for
GPT-2 Small (124M), while the similarly sized CodeGen-350M achieves 37.4%
success. Problem-level bootstrap intervals show that the GPT-2 Small / Medium
gap is not statistically reliable (+0.37 percentage points, 95% CI [-0.73,
1.37], p=0.50), whereas both models trail CodeGen by more than 32 points with
p=0.0001. This pattern transfers to a second benchmark: on the full MBPP test
split (257 tasks, 20 samples per task), both GPT-2 variants achieve 0.0%
success, whereas CodeGen reaches 7.39% success and 22.96% pass@5. A stricter
HumanEval+ re-evaluation on the first five saved samples per task preserves the
same ordering: GPT-2 Small and GPT-2 Medium both remain at 0.0% pass@1, while
CodeGen retains 2.1% pass@1. On the official MBPP+ benchmark in EvalPlus (378
tasks, Base+Extra), GPT-2 Small and GPT-2 Medium again remain at 0.0% pass@1
and pass@10, while CodeGen retains 1.1% and 6.4%. On the contamination-aware
LiveCodeBench release v2 code-generation benchmark (511 tasks, 10 samples per
task), GPT-2 Small and GPT-2 Medium again remain at 0.0% on pass@1, pass@5, and
pass@10, while CodeGen retains only 0.02%, 0.10%, and 0.20%, indicating that
all three small models are near-zero on the harder benchmark even though
code-specialized pretraining still preserves the only nonzero signal. To
partially reduce the family-comparison objection, we also evaluate the
CodeGen-350M NL -> Multi -> Mono continued-pretraining ladder. Mono is
strongest on both benchmarks; HumanEval remains mixed at the intermediate Multi
checkpoint (31.60%, 30.66%, and 39.00%), while MBPP is cleanly monotone (0.02%,
1.23%, and 2.76%), showing that added code- and Python-specific continued
pretraining changes performance even when architecture and parameter count are
held fixed. Through systematic layer-wise ablation experiments across
16,300-16,400 HumanEval code generation attempts, we find that both GPT-2
variants are globally brittle under full-layer zeroing, whereas CodeGen retains
partial residual performance in a small subset of layers. GPT-2 Medium never
preserves successful generations under ablation, but ablation of layer index 12
uniquely shifts failures from pure syntax collapse to a mixed 76.1% syntax /
23.9% runtime regime, suggesting a distinctive mid-depth execution-relevant
checkpoint. A follow-up scaled-ablation pilot on layer 12 shows that
intermediate attenuation first worsens syntax (2.0% success at 0.5 scale; 0.0%
at 0.25), while only full zeroing reopens the runtime-error regime (28.0%),
reinforcing the view that layer 12 marks a thresholded execution-relevant stage
rather than a smooth redundancy basin. A corresponding CodeGen layer-13 pilot
retains 28-30% success under partial attenuation and still preserves 24%
success under full zeroing while shifting 24% of samples into runtime errors,
indicating broader residual support for executable code. A balanced linear
probe on 200 saved layer-12 activation vectors identifies five dimensions that
distinguish successful from syntactically invalid generations with 72.5%
held-out accuracy, compared with 75.0% for a full-layer probe. Constructive
steering along this learned five-dimensional direction doubles subset success
from 6.0% to 12.0%; against 20 matched sparse random controls, the learned
vector beats 19 while one control reaches 14.0% (control mean 6.2%, empirical
p <= 0.095). These results suggest that code-pretrained models distribute
useful computation across depth more effectively than similarly sized
general-language models, and that failure-mode shifts provide a more reliable
mechanistic signal than success alone.

## Highlights

- Small-model code generation does not improve reliably with parameter scaling
  alone.
- Code-specialized pretraining explains much more than GPT-2 scale in this
  regime.
- Layer ablations expose distinct syntax, runtime, and residual robustness
  profiles.
- Prompt format and decoding choices substantially change code-generation
  reliability.
- Reproducible artifacts link benchmark claims to saved outputs, logs, and
  scripts.

## Data Availability

Processed result tables, plotting scripts, benchmark summaries, saved result
artifacts, configuration files, source code, and the claim-to-artifact audit are
available in the curated supplementary artifact archive submitted with the
manuscript. Large model checkpoints and raw model caches are not redistributed;
all experiments use public pretrained checkpoints and regenerate derived outputs
from the provided scripts and configuration files.

## Competing Interest

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

## Funding

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

## Generative AI Declaration

During the preparation of this work, the author used OpenAI Codex to assist with
manuscript editing, code execution orchestration, artifact organization, and
formatting. After using this tool, the author reviewed and edited the content as
needed and takes full responsibility for the content of the submitted
manuscript.

## File Upload Hashes

- `bottleneck.pdf`:
  `8377146545171622D6C7D599734A4F99F0FA5094BD558B1FC6C4CBCB999E034C`
- `submission_jss_20260512_135646/jss_source_package.zip`:
  `A321C0B0FC20B48C1366AF1ED870536E5EED10918B3B96D216A3E925E69F1271`
- `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`:
  `31551AC28D359CE81367456053B46BBA8EC619AAD669E61F9F66B98C1E7BD376`

## Required Pre-Submit Commands

```powershell
python scripts/verify_jss_upload_manifest.py
python scripts/run_jss_preflight.py
```

Expected output:

```text
upload_manifest_checks=7 failed=0
jss_preflight_checks=6 failed=0
```
