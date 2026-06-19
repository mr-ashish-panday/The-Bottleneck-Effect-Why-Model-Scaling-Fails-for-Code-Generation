# SCP Readiness Audit

Target: Science of Computer Programming.

Status: transfer confirmed from Elsevier Transfer Your Manuscript service on June 1, 2026.

Previous submission: `JSSOFTWARE-D-26-01113`, Journal of Systems and Software, desk rejected on May 27, 2026.

## Diagnosis

- Primary bottleneck: `judgment`.
- First broken link: target fit and claim pressure, not raw effort.
- Correction: choose an Elsevier transfer target that is lower-risk than JSS/IST, has a subscription/no-charge publication route, and fits programming, validation, testing, executable correctness, and software-development evaluation.

## Target Fit

- Elsevier transfer suggested Science of Computer Programming for this manuscript.
- Elsevier's journal page describes SCP as covering software systems development, use, maintenance, validation, verification, coding, testing, programming languages, and development tools.
- SCImago lists SCP as Software Q3 in 2024.
- The paper is now framed as a software-development reliability study of neural code-generation systems, rather than as a broad ML scaling-law paper.

## Required Revision Gates

- [x] Confirm transfer to Science of Computer Programming.
- [x] Change manuscript journal field to Science of Computer Programming.
- [x] Replace ISSE cover letter/highlights with SCP-facing files.
- [x] Prepare SCP metadata packet: title, abstract, keywords, author info, declarations, data availability.
- [x] Rebuild manuscript PDF after SCP retargeting edits.
- [x] Run LaTeX log checks: no errors, unresolved citations, unresolved references, or serious overfull boxes.
- [x] Rebuild SCP source ZIP and supplementary artifact package.
- [x] Verify SCP package ZIP integrity and upload-file list.
- [x] Open the Science of Computer Programming completion email/link.
- [ ] User accepts the Editorial Manager publisher/privacy/Aries policy registration question.
- [ ] Complete the Science of Computer Programming submission after the legal gate is cleared and the portal proof is reviewed.

## Current Submission Package Files

- Manuscript source: `bottleneck.tex`
- SCP cover letter: `scp_cover_letter.txt`
- SCP highlights: `scp_highlights.txt`
- Declaration of interest: `scp_declaration_of_interest.txt`
- SCP metadata packet: `SCP_PORTAL_METADATA.md`
- SCP artifact README: `SCP_ARTIFACT_README.md`
- SCP claim audit: `SCP_CLAIM_ARTIFACT_AUDIT.md`

## Verification Log

- 2026-06-01: Elsevier Transfer Your Manuscript portal confirmed transfer to Science of Computer Programming. Portal text: "Transfer confirmed" and "confirmed on 01 Jun 2026".
- 2026-06-01: SCP target files created locally.
- 2026-06-01: `bottleneck.pdf` rebuilt successfully at 47 pages with `\journal{Science of Computer Programming}`.
- 2026-06-01: Final LaTeX log scan found no errors, unresolved citations, unresolved references, or overfull boxes.
- 2026-06-01: Built `submission_scp_20260601/scp_source_package.zip` with 23 entries; `zipfile.testzip()` returned `None`.
- 2026-06-01: Built `submission_scp_20260601/scp_supplement_artifact_full.zip` with 402 entries; `zipfile.testzip()` returned `None`.
- 2026-06-01: Checked SCP source and supplement ZIPs for obvious old JSS/ISSE package entries; none were found.
- 2026-06-01: Rendered first page and contribution page to `outputs/pdf_checks/scp_transfer_20260601/` and visually checked both pages.
- 2026-06-01: Final hashes: `bottleneck.pdf` = `310E3D17D64AB9860AD2866A743034C75A46962AC27FE3EFA5CF59D9C901B71F`; `scp_source_package.zip` = `6E52A76BF88A5A4A38D04E09A3A47E10F6E5236F3D559A72E45B8B22E3D68A64`; `scp_supplement_artifact_full.zip` = `7265189CB0624F14FE7A8ED473F219F65FF471868C86E0E93D0895C9019614BF`.
- 2026-06-01: SCP completion emails arrived. The Elsevier email says the new submission expires on August 30, 2026 if no further action is taken, and instructs the author to find the manuscript under "Submissions Sent Back to Author", edit submission, replace outdated files if needed, build PDF for approval, accept Ethics in Publishing Policy, and approve submission.
- 2026-06-01: Editorial Manager login instructions email received. The password-creation link opened the SCICO registration/legal gate.
- 2026-06-01: Current blocker is a human/legal gate: SCICO asks the user to accept the Publisher Terms and Conditions, Privacy Policy, and Aries Privacy Policy before continuing.
