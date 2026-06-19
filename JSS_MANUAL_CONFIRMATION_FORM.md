# JSS Manual Confirmation Form

Prepared: 2026-05-12.
Last revalidated: 2026-05-17 08:21 NPT.

Purpose: capture the human confirmations required across the final portal
submission and immediate closeout sequence.

Sequence:

1. Complete the pre-submit sections below before pressing final submit.
2. Submit in the Journal of Systems and Software portal.
3. Save the required post-submit proof files.
4. Complete the final post-submit proof section.

Do not mark the submission complete until every required box is confirmed.

## Required Author Metadata Confirmation

Confirm these are exactly correct for the portal:

- [ ] Author name: `Ashish Pandey`
- [ ] Affiliation: `Department of Computer and Electronics Engineering, Khwopa College of Engineering, Nepal`
- [ ] Corresponding email: `ashishpanday9818@gmail.com`
- [ ] No additional coauthors need to be added.
- [ ] No ORCID or institutional profile field needs correction before submit.

If any item is wrong, update `bottleneck.tex`, `JSS_PORTAL_METADATA.md`, and
`JSS_PORTAL_UPLOAD_RUNBOOK.md` before uploading.

## Required Suggested-Reviewer Conflict Confirmation

Confirm there is no personal, supervisory, employment, funding, collaboration,
institutional, or adversarial conflict with each reviewer before entering them:

- [ ] David Lo
- [ ] Michael Pradel
- [ ] Baishakhi Ray
- [ ] Lingming Zhang
- [ ] Earl T. Barr
- [ ] Martin Monperrus

Do not enter Thomas Zimmermann as an author-suggested reviewer because the JSS
editorial board page currently lists T. Zimmermann on the board.

## Portal Classification Confirmation

Use the closest live portal labels to these, prioritizing software-engineering
fit over generic machine-learning labels:

- [ ] Artificial Intelligence applied in software engineering
- [ ] Empirical software engineering
- [ ] Software testing, verification, and validation
- [ ] Software reliability
- [ ] Mining software repositories / software analytics
- [ ] Software engineering for AI systems

## Upload Confirmation

Confirm each file is attached in the portal:

- [ ] `bottleneck.pdf`
- [ ] `submission_jss_20260512_135646/jss_source_package.zip`
- [ ] `jss_highlights.txt`
- [ ] `jss_cover_letter.txt`
- [ ] `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
- [ ] `JSS_ARTIFACT_README.md`, if the portal allows it
- [ ] `JSS_CLAIM_ARTIFACT_AUDIT.md`, if the portal allows it

## Final Portal Proof Confirmation

Before final submit, inspect the portal-generated PDF proof:

- [ ] Title has no typo.
- [ ] Author and affiliation are correct.
- [ ] No TMLR header appears.
- [ ] Figure 15 renders correctly.
- [ ] Data Availability says `curated supplementary artifact archive`.
- [ ] Source ZIP and supplement ZIP are attached.
- [ ] Declarations match `bottleneck.tex`.

## Post-Submit Proof Confirmation

Save these inside `submission_jss_20260512_135646/post_submission_proof/`
immediately after pressing final submit:

- [ ] Manuscript ID.
- [ ] Submission confirmation PDF or screenshot.
- [ ] Confirmation email.
- [ ] Final uploaded file list shown by the portal.
- [ ] Portal-generated PDF proof.

Then run `scripts/record_jss_submission.py` with the manuscript ID and saved
proof locations so the tracker, final audit, and submission checklist update
together from the same evidence.

After all six sections are truly complete, the guarded recorder can update the
checkboxes in one pass:

```powershell
python -B scripts/record_jss_manual_confirmations.py `
  --confirm-author-metadata `
  --confirm-no-reviewer-conflicts `
  --confirm-portal-classifications `
  --confirm-required-uploads `
  --confirm-portal-proof `
  --confirm-proof-saved
```
