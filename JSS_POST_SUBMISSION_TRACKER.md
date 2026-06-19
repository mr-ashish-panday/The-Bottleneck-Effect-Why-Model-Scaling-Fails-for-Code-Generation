# JSS Post-Submission Tracker

Target journal: Journal of Systems and Software.

Status: submitted to JSS on 2026-05-24; manuscript ID `JSSOFTWARE-D-26-01113`.

Use this file immediately after final submission. The manuscript ID is now
known, but the full proof-file bundle is still incomplete until the confirmation
screen/email and final uploaded-file list are saved.

Preferred path: save the proof files first, then use
`scripts/record_jss_submission.py` so the tracker, final audit, and submission
checklist are updated together from real proof files rather than by hand.

## Submission Record

- Submission date: 2026-05-24
- Manuscript ID: JSSOFTWARE-D-26-01113
- Portal account used: ashishpanday9818@gmail.com
- Corresponding author email: ashishpanday9818@gmail.com
- Article type: Research Paper
- Final title: The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation

## Current Proof Captured

- Confirmation email text reported by Ashish from `Journal of Systems and Software <em@editorialmanager.com>` at 9:10 PM NPT on 2026-05-24.
- Email states that the submission was received by The Journal of Systems & Software and assigned manuscript number `JSSOFTWARE-D-26-01113`.
- Portal PDF proof already exists locally: `submission_jss_20260512_135646/post_submission_proof/04_portal_pdf_proof.pdf`.
- Remaining proof to save as files before running the guarded recorder: submission confirmation screen/PDF and final uploaded-file list. A reconstructed confirmation-email text proof is saved; replace it with raw `.eml` later if available.

## Recorded Proof Files

- confirmation_email: `submission_jss_20260512_135646/post_submission_proof/02_confirmation_email.txt`
- portal_pdf_proof: `submission_jss_20260512_135646/post_submission_proof/04_portal_pdf_proof.pdf`

## Proof Files To Save

Save each item inside the proof folder, then record the exact file path with
`scripts/record_jss_submission.py`:

- Submission confirmation PDF or screenshot:
  `submission_jss_20260512_135646/post_submission_proof/`
- Confirmation email:
  `submission_jss_20260512_135646/post_submission_proof/`
- Final portal uploaded-file list:
  `submission_jss_20260512_135646/post_submission_proof/`
- Portal-generated PDF proof:
  `submission_jss_20260512_135646/post_submission_proof/`
- Any editor/system acknowledgement:
  `submission_jss_20260512_135646/post_submission_proof/`

## Final Uploaded Files

Check against `JSS_UPLOAD_FILE_MANIFEST.csv`:

- [ ] `bottleneck.pdf`
- [ ] `submission_jss_20260512_135646/jss_source_package.zip`
- [ ] `jss_highlights.txt`
- [ ] `jss_cover_letter.txt`
- [ ] `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
- [ ] `JSS_ARTIFACT_README.md`, if accepted by the portal
- [ ] `JSS_CLAIM_ARTIFACT_AUDIT.md`, if accepted by the portal

## Final Gate After Submission

Only after the above proof is saved:

- [ ] Run `python -B scripts/record_jss_submission.py --manuscript-id <ID> --confirmation-proof <path> --email-proof <path> --uploaded-file-list-proof <path> --portal-pdf-proof <path>`.
- [ ] Run `python -B scripts/record_jss_manual_confirmations.py --confirm-author-metadata --confirm-no-reviewer-conflicts --confirm-portal-classifications --confirm-required-uploads --confirm-portal-proof --confirm-proof-saved`.
- [ ] Run `python -B scripts/audit_jss_completion.py --report JSS_COMPLETION_AUDIT_REPORT.md` and confirm it passes.
- [ ] Verify that `JSS_FINAL_SUBMISSION_AUDIT.md` now includes the manuscript ID.
- [ ] Verify that `JSS_SUBMISSION_CHECKLIST.md` now includes the submitted record.
- [ ] Stop the paid Lightning instance unless a specific new experiment is
  approved.
- [ ] Record the submission milestone in the global project map.

## Do Not Do This

- Do not submit the manuscript to another journal while the JSS submission is
  active.
- Do not start new GPU runs unless a concrete editor/reviewer-facing evidence
  gap is identified.
- Do not delete local run logs, pulled backups, or the final JSS upload package.
