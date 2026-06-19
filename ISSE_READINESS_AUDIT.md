# ISSE Readiness Audit

Target: Innovations in Systems and Software Engineering: A NASA Journal.

Status: active revision after JSS desk reject `JSSOFTWARE-D-26-01113`.

## Diagnosis

- Primary bottleneck: `judgment`.
- First broken link: target fit and framing, not raw effort.
- JSS signal: the paper was judged "not quite ready for publication in a leading journal" and advised to seek workshop/conference feedback before JSS.
- Current correction: retarget to a realistic software/systems engineering venue and lower rhetorical pressure while preserving the artifact-backed empirical contribution.

## Target Fit

- ISSE scope includes systems engineering, systems integration, software engineering, and software development.
- This paper is now framed as a software-engineering reliability study of neural code-generation systems.
- The retargeted cover letter explicitly emphasizes engineered pipelines, evaluation strictness, generation protocol, and failure-mode structure.

## Required Revision Gates

- [x] Remove JSS-specific manuscript wording.
- [x] Replace JSS cover letter with ISSE cover letter.
- [x] Shorten abstract toward Springer 150-250 word guidance.
- [x] Rebuild manuscript PDF after retargeting edits.
- [x] Run LaTeX log checks: no errors, unresolved citations, unresolved references, or serious overfull boxes.
- [x] Rebuild source ZIP and supplementary artifact package after final edits.
- [x] Prepare ISSE metadata packet: title, abstract, keywords, author info, declarations, data availability.
- [ ] Submit only after the package passes a fresh readiness audit.

## Current Submission Package Files

- Manuscript source: `bottleneck.tex`
- Retargeted cover letter: `isse_cover_letter.txt`
- Retargeted highlights: `isse_highlights.txt`
- Declaration of interest: `isse_declaration_of_interest.txt`
- ISSE source package: `submission_isse_20260527/isse_source_package.zip`
- ISSE supplementary artifact package: `submission_isse_20260527/isse_supplement_artifact_full.zip`
- ISSE artifact README: `ISSE_ARTIFACT_README.md`
- ISSE claim audit: `ISSE_CLAIM_ARTIFACT_AUDIT.md`

## Verification Log

- 2026-05-27: `bottleneck.pdf` rebuilt successfully at 47 pages.
- 2026-05-27: Abstract counted at 200 words, within Springer 150-250 word guidance.
- 2026-05-27: LaTeX log scan found no errors, unresolved citations, unresolved references, or overfull boxes.
- 2026-05-27: First page rendered to `outputs/pdf_checks/isse_revision_1/page-01.png` and visually checked.
- 2026-05-27: Neutralized the manuscript-facing robustness figure path and label from `jss_robustness_controls` to `targeted_robustness_controls`.
- 2026-05-27: Built `submission_isse_20260527/isse_source_package.zip` with 23 entries; `zipfile.testzip()` returned `None`.
- 2026-05-27: Built `submission_isse_20260527/isse_supplement_artifact_full.zip` with 407 entries; `zipfile.testzip()` returned `None`.
- 2026-05-27: Checked the ISSE source and supplement ZIPs for obvious old JSS package entries; none were found.
- 2026-05-27: Rendered first page and robustness-control page to `outputs/pdf_checks/isse_revision_2/` and visually checked both pages.
- 2026-05-27: Softened the contribution framing from broad architectural design-principle language to a narrower software-engineering reliability contribution.
- 2026-05-27: Rebuilt after the contribution-framing pass. Final hashes: `bottleneck.pdf` = `F35765A697D575570991D7845E6D124FB2DBEBF9C529AB58E125719579BBBD05`; `isse_source_package.zip` = `0966C451D6EAA36C152489DB164F463CFFC438973850D3A2BEDE46C97C822E3A`; `isse_supplement_artifact_full.zip` = `27E3A0F97A9C03A08D28F383C6D4076C32147F7AAC77A3B83C792662B1B8F979`.
- 2026-05-27: Final LaTeX log scan found no errors, unresolved citations, unresolved references, or overfull boxes.
