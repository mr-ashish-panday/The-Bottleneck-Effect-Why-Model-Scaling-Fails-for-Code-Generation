# ISSE Portal Metadata

Target journal: Innovations in Systems and Software Engineering: A NASA Journal.

Submission link: https://submission.nature.com/

## Article Type

Original Article / Research Article, whichever the portal offers.

## Title

The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation

## Abstract

Code generation is an execution-constrained software-engineering task: generated programs must parse, run, and satisfy tests. This paper studies whether small-model code generation improves with parameter scaling, code-specialized pretraining, and more distributed layerwise computation. Across HumanEval, MBPP, EvalPlus HumanEval+/MBPP+, and LiveCodeBench release v2, GPT-2 Medium (355M) does not reliably improve over GPT-2 Small (124M): the HumanEval gap is not statistically significant, and both GPT-2 variants reach 0.0% success on full-coverage MBPP. In contrast, similarly sized CodeGen checkpoints retain substantially stronger executable-code behavior on HumanEval and MBPP. A fixed-scale CodeGen-350M NL->Multi->Mono ladder further shows that later code- and Python-focused continued pretraining improves the strongest checkpoint, especially on MBPP. To examine where these differences arise, we combine full-layer ablation, graded ablation, failure-mode analysis, and sparse activation steering. GPT-2 Medium is globally brittle under layer removal, while CodeGen retains residual success across several layers. A compact GPT-2 Medium layer-12 probe captures most of the success-vs.-syntax signal, but matched controls make the steering result suggestive rather than definitive. Overall, the results support a software-engineering reliability view: in this small-model regime, useful code behavior depends less on parameter count alone than on code-specific pretraining, generation protocol, and the distribution of execution-relevant computation across depth.

## Keywords

- Code generation
- Empirical software engineering
- Large language models
- Model evaluation
- Mechanistic interpretability
- Software reliability

## Author

- Ashish Pandey
- ORCID: 0009-0004-7085-7373
- Department of Computer and Electronics Engineering, Khwopa College of Engineering, Nepal
- Corresponding email: ashishpanday9818@gmail.com

## Declarations

Competing interests: The author declares no competing interests.

Funding: This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

Data availability: The processed result tables, plotting scripts, benchmark summaries, saved result artifacts, configuration files, source code, and claim-to-artifact audit used in this manuscript are available in the supplementary artifact archive submitted with this manuscript. Large model checkpoints and raw model caches are not redistributed; all experiments use public pretrained checkpoints and regenerate derived outputs from the provided scripts and configuration files.

AI-assisted technologies: During the preparation of this work, the author used OpenAI Codex to assist with manuscript editing, code execution orchestration, artifact organization, and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the submitted manuscript.

## Current Files To Upload

- Manuscript PDF: `bottleneck.pdf`
- LaTeX/source package: `submission_jss_20260512_135646/jss_source_package.zip` (rebuild and rename before final ISSE submission)
- Supplementary artifact: `submission_jss_20260512_135646/jss_supplement_artifact_full.zip` (rebuild and rename before final ISSE submission)
- Cover letter: `isse_cover_letter.txt`
- Highlights, if requested: `isse_highlights.txt`
- Declaration file, if requested: `jss_declaration_of_interest.txt`
