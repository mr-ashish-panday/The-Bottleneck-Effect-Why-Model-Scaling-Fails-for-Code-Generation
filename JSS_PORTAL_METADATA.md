# JSS Portal Metadata

Target journal: Journal of Systems and Software.

Article type: Regular research article.

## Title

The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation

## Author

Ashish Pandey

Department of Computer and Electronics Engineering, Khwopa College of
Engineering, Nepal

Email: ashishpanday9818@gmail.com

## Keywords

- Code generation
- Empirical software engineering
- Large language models
- Model evaluation
- Mechanistic interpretability
- Software reliability

## Suggested Classifications

Use the closest matches available in the JSS / Elsevier submission portal. If
the portal wording differs, choose the nearest software-engineering category
rather than a generic machine-learning category.

Preferred order:

1. Artificial Intelligence applied in software engineering
2. Empirical software engineering
3. Software testing, verification, and validation
4. Software reliability
5. Mining software repositories / software analytics
6. Software engineering for AI systems

Rationale: JSS states that it covers all aspects of software engineering,
including AI/data analytics applied in software engineering, software
engineering for AI systems, methods and tools for empirical software
engineering research, testing/verification/validation, and metrics/evaluation.

## Highlights

- Small-model code generation does not improve reliably with parameter scaling alone.
- Code-specialized pretraining explains much more than GPT-2 scale in this regime.
- Layer ablations expose distinct syntax, runtime, and residual robustness profiles.
- Prompt format and decoding choices substantially change code-generation reliability.
- Reproducible artifacts link benchmark claims to saved outputs, logs, and scripts.

## Abstract

Use the abstract from `bottleneck.tex` / `bottleneck.pdf`. Do not paste a
shortened version unless the portal imposes a hard character limit.

## Upload Files

- Main manuscript PDF:
  `bottleneck.pdf`
- Manuscript source package:
  `submission_jss_20260512_135646/jss_source_package.zip`
- Highlights:
  `jss_highlights.txt`
- Cover letter:
  `jss_cover_letter.txt`
- Supplementary artifact package:
  `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
- Artifact README:
  `JSS_ARTIFACT_README.md`
- Claim-to-artifact audit:
  `JSS_CLAIM_ARTIFACT_AUDIT.md`

## File Hashes

- `jss_source_package.zip`
  SHA-256: `A321C0B0FC20B48C1366AF1ED870536E5EED10918B3B96D216A3E925E69F1271`
- `jss_supplement_artifact_full.zip`
  SHA-256: `31551AC28D359CE81367456053B46BBA8EC619AAD669E61F9F66B98C1E7BD376`

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

## Suggested Reviewers

Prepared in `JSS_SUGGESTED_REVIEWERS.md`.

Recommended portal order if reviewer suggestions are required:

1. David Lo
2. Michael Pradel
3. Baishakhi Ray
4. Lingming Zhang
5. Earl T. Barr
6. Martin Monperrus

Before entering names, confirm there are no personal, supervisory, employment,
funding, collaboration, institutional, or adversarial conflicts. Do not enter
Thomas Zimmermann as an author-suggested reviewer because the JSS editorial
board page currently lists T. Zimmermann on the editorial board.

## Final Manual Checks Before Submit

- Confirm author affiliation and email are correct.
- Confirm the portal accepts the supplement ZIP size.
- Confirm no generated package includes model checkpoints or private secrets.
- Confirm all portal declarations match the declarations in `bottleneck.tex`.
