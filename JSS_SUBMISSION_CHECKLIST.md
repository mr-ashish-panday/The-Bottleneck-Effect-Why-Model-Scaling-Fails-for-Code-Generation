# JSS Submission Checklist

Target journal: Journal of Systems and Software.

Primary framing: empirical software-engineering evidence about code-generation
reliability, evaluation brittleness, prompt/decoding controls, and reproducible
artifact traces.

Official guide: https://www.sciencedirect.com/journal/journal-of-systems-and-software/publish/guide-for-authors

## Current Status

- Target venue: locked to Journal of Systems and Software.
- Backup venue: Information and Software Technology.
- Stretch venue: Neurocomputing only if the mechanistic contribution becomes
  substantially stronger.
- Manuscript source: `bottleneck.tex`.
- Compiled manuscript: `bottleneck.pdf`.
- Highlights file: `jss_highlights.txt`.
- Bibliography source: `references.bib`.
- Current LaTeX style: Elsevier `elsarticle` preprint with
  `\journal{Journal of Systems and Software}`.

## Completed

- Removed TMLR header and TMLR bibliography style from the active manuscript.
- Converted manuscript to Elsevier `elsarticle` front matter.
- Added named author metadata for JSS single-anonymized review.
- Added keywords.
- Added data availability statement.
- Added competing-interest declaration.
- Added funding declaration.
- Added generative-AI-use declaration.
- Added separate highlights file.
- Added the JSS-targeted robustness controls to the manuscript.
- Rebuilt Figure 15 after fixing the legend overlap.
- Created JSS cover letter: `jss_cover_letter.txt`.
- Created JSS source package:
  `submission_jss_20260512_135646/jss_source_package.zip`.
  SHA-256:
  `A321C0B0FC20B48C1366AF1ED870536E5EED10918B3B96D216A3E925E69F1271`.
- Added deterministic source packaging script:
  `scripts/build_jss_source_package.py`.
- Created JSS supplement artifact package:
  `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`.
  SHA-256:
  `31551AC28D359CE81367456053B46BBA8EC619AAD669E61F9F66B98C1E7BD376`.
- Updated `scripts/build_jss_supplement.py` to write normalized ZIP metadata
  so rebuilds keep stable hashes when file contents are unchanged.
- Validated both archives with ZIP readers and preflight gates. The source package has 23 entries;
  the cleaned supplement package has 398 entries.
- Added `scripts/audit_jss_submission_artifacts.py` and generated
  `JSS_CLAIM_ARTIFACT_AUDIT.md`.
- Ran the claim-to-artifact verifier: 32 checks passed, 0 failed.
- Corrected the CodeGen syntax-profile percentages in `bottleneck.tex`,
  regenerated Figure 3, and extended `scripts/audit_jss_submission_artifacts.py`
  to verify the GPT-2, GPT-2 Medium, and CodeGen syntax-profile tables against
  the tracked `syntax_analysis.json` artifacts.
- Prepared suggested reviewers in `JSS_SUGGESTED_REVIEWERS.md` and updated
  `JSS_PORTAL_METADATA.md` with the recommended portal order.
- Prepared JSS-facing portal classifications in `JSS_PORTAL_METADATA.md`.
- Added `JSS_FINAL_SUBMISSION_AUDIT.md` mapping requirements to current
  evidence and identifying the remaining manual blockers.
- Added `JSS_PORTAL_UPLOAD_RUNBOOK.md` with portal field values, upload order,
  declarations, reviewer order, stop gates, and post-submission proof to save.
- Added `LIGHTNING_COST_CONTROL_HANDOFF.md` documenting that no JSS experiment
  is currently running and the paid Lightning instance should be stopped after
  submission unless a specific new experiment is approved.
- Added `JSS_MANUAL_CONFIRMATION_FORM.md` so author metadata, reviewer
  conflicts, portal classifications, upload attachments, portal proof, and
  post-submit proof are confirmed explicitly.
- Added `JSS_UPLOAD_FILE_MANIFEST.csv` and
  `scripts/verify_jss_upload_manifest.py` to lock every portal upload file by
  byte count and SHA-256 before upload.
- Added a narrow `.gitignore` exception for `JSS_UPLOAD_FILE_MANIFEST.csv` so
  the upload manifest is not hidden by the repository's broad `*.csv` ignore
  rule.
- Added `scripts/run_jss_preflight.py` as the one-command pre-upload validation
  gate for claim audit, upload manifest, archive readability, LaTeX log health,
  Data Availability wording, and sensitive/stale text scanning.
- Added `scripts/build_jss_supplement.py` and rebuilt the supplement to remove
  reviewer-noise files such as `__pycache__`, `.pyc`, `.pyo`, transient logs,
  local PDF-check renders, and local packaging/preflight scripts.
- Added `JSS_PORTAL_COPY_PASTE_PACKET.md` with plain-text portal fields,
  abstract, highlights, declarations, upload hashes, and expected pre-submit
  command output.
- Added `JSS_POST_SUBMISSION_TRACKER.md` for manuscript ID, saved proof files,
  final uploaded-file checklist, and post-submit Lightning shutdown tracking.
- Added `submission_jss_20260512_135646/post_submission_proof/README.md` as the
  local destination for JSS confirmation screenshots, emails, portal proof, and
  final uploaded-file list.
- Added `scripts/jss_submission_status.py` for a lightweight status readout
  that does not rebuild or touch upload artifacts.
- Added `scripts/audit_jss_completion.py` as the strict non-rebuilding
  completion gate for manuscript ID, proof files, and tracker status.
- Added `scripts/record_jss_submission.py` for after final submit; it refuses
  to record completion unless a manuscript ID and all required proof files are
  provided, then updates `JSS_POST_SUBMISSION_TRACKER.md`,
  `JSS_FINAL_SUBMISSION_AUDIT.md`, and this checklist.
- Added `C:\Users\Ashish\all\Ashish\Bottleneck JSS Submit Now.md` in the active
  Obsidian vault with final files, hashes, validation commands, manual
  confirmations, and proof-save path.
- Added `JSS_NEXT_ACTION.md` to make the immediate next step explicit: submit
  the prepared JSS package, save proof, then stop Lightning unless a concrete
  new experiment is approved.
- Added the official JSS ScienceDirect page and `Submit your article` route to
  `JSS_PORTAL_UPLOAD_RUNBOOK.md`, `JSS_NEXT_ACTION.md`, and the active Obsidian
  submit-now note.
- Ran sensitive-token scan over the final source/supplement staging folders:
  0 hits for private keys, common API-token patterns, and Lightning SSH host
  strings. The reviewer-facing supplement excludes local planning notes with
  internal Lightning SSH host metadata.
- Extracted the rebuilt final source and supplement ZIPs and scanned their
  actual contents: 0 hits for private keys, common API-token patterns, Lightning
  SSH host strings, or the stale GitHub artifact URL.
- Recompiled with `pdflatex`, `bibtex`, and two final `pdflatex` passes.
- Verified the final log has no LaTeX errors, unresolved citations, unresolved
  references, or overfull boxes.
- Visually checked page 1 and the Figure 15 page.
- Decided not to expose the full GitHub repository at first submission because
  the local working tree contains local run notes and unreviewed untracked
  artifacts. Use the curated JSS supplementary archive for review instead.
- Rebuilt `bottleneck.pdf`, `jss_source_package.zip`, and
  `jss_supplement_artifact_full.zip` after changing Data Availability from a
  public GitHub repository link to the curated supplementary artifact archive.

## Submitted To JSS

- Submission date: `2026-05-24`
- Manuscript ID: `JSSOFTWARE-D-26-01113`
- Journal: `The Journal of Systems & Software`
- Confirmation source: email from `Journal of Systems and Software <em@editorialmanager.com>` reported by Ashish at 9:10 PM NPT.
- Remaining closeout: save remaining proof files, then run the guarded recorder/audit with `--force` if the tracker already contains the manuscript ID.

## Remaining Post-Submission Proof Tasks

- Save confirmation screen/PDF if not already saved.
- Save confirmation email export/screenshot if not already saved.
- Save final uploaded-file list if not already saved.
- Keep `04_portal_pdf_proof.pdf` in the post-submission proof folder.

## Do Not Reopen Without New Evidence

- Do not target TMLR again for this version.
- Do not switch to Neurocomputing unless the paper gains a clearly stronger
  neural/mechanistic contribution.
- Do not spend more GPU on broad sweeps unless the claim-to-artifact audit finds
  a specific missing control that affects the JSS argument.
