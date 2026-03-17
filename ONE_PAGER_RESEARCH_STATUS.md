# One-Page Research Summary

## The Bottleneck Effect: Why Model Scaling Fails for Code Generation

### Core claim
Increasing parameter count inside a general-language model family does not automatically improve code generation. In our study, scaling GPT-2 from 124M to 355M parameters does not improve executable-code performance, while a similarly sized code-pretrained model, CodeGen-350M, performs dramatically better. The central conclusion is that for code generation, pretraining distribution and internal computation structure matter more than raw scale alone.

### Main benchmark results
- **HumanEval**
  - GPT-2 Small: **5.2%**
  - GPT-2 Medium: **4.8%**
  - CodeGen-350M: **37.4%**
- **MBPP (full benchmark)**
  - GPT-2 Small: **0.0%**
  - GPT-2 Medium: **0.0%**
  - CodeGen-350M: **7.39%**
  - CodeGen-350M **pass@5**: **22.96%**

### Statistical result
Problem-level bootstrap testing shows that the GPT-2 Small vs GPT-2 Medium gap is not statistically meaningful, while both GPT-2 models are far behind CodeGen. This directly supports the paper's main negative result: simple scaling inside GPT-2 does not produce better code-generation capability.

### Mechanistic result
The paper does not stop at benchmark comparison. It identifies an internal execution bottleneck:
- GPT-2 Medium is globally brittle under layer ablation, but **layer 12** is distinctive.
- Graded intervention on GPT-2 Medium layer 12 shows a progressive collapse in success as the layer is weakened.
- Full suppression of the layer shifts failures toward runtime-error behavior, suggesting that execution-relevant computation is concentrated there.
- A compact layer-12 activation subspace separates successful and failed generations and can be used for intervention.
- In contrast, **CodeGen layer 13** shows better residual robustness under graded ablation, supporting the argument that code-specialized models distribute code-relevant computation more effectively.

### External validation status
We are currently strengthening the paper with stricter evaluation and contamination-aware testing.

**Completed so far under EvalPlus**
- GPT-2 HumanEval+: **0.0%**
- GPT-2 Medium HumanEval+: **0.0%**
- CodeGen HumanEval+: **2.1%**

These stricter-test results preserve the main story: GPT-2 scaling still fails, and CodeGen still survives better.

**In progress**
- EvalPlus on MBPP
- LiveCodeBench external validation

### Why this paper matters
This work argues that code generation does not follow the naive "bigger is better" rule. The paper contributes:
1. A clear empirical scaling failure result.
2. Cross-benchmark validation beyond a single test set.
3. A mechanistic explanation for why the failure occurs.
4. Evidence that code-specific pretraining changes how useful computation is organized internally.

### Current status
The paper is no longer just a benchmark comparison. It is now a benchmark-plus-mechanism paper with:
- multi-model comparison,
- significance testing,
- ablation evidence,
- activation probing,
- intervention analysis,
- and ongoing strict external validation.

At the current stage, it is a serious solo research paper and is close to submission quality once the remaining external-validation runs are fully integrated.
