# JSS Final Portal Card

Status: submitted to The Journal of Systems & Software as `JSSOFTWARE-D-26-01113`.

Target: Journal of Systems and Software.

Use this card for status monitoring and proof retrieval. Do not reopen manuscript
editing unless an editor/reviewer-facing gate fails.

## Upload Files

1. `bottleneck.pdf`
2. `submission_jss_20260512_135646/jss_source_package.zip`
3. `jss_highlights.txt`
4. `jss_cover_letter.txt`
5. `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
6. Optional if accepted: `JSS_ARTIFACT_README.md`
7. Optional if accepted: `JSS_CLAIM_ARTIFACT_AUDIT.md`

## Submitted Metadata

- Author: `Ashish Pandey`
- Affiliation: `Department of Computer and Electronics Engineering, Khwopa College of Engineering, Nepal`
- Email: `ashishpanday9818@gmail.com`
- Article type: `Regular research article`
- Title: `The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation`
- Data availability says `curated supplementary artifact archive`.
- Portal-generated PDF proof has no TMLR header and Figure 15 renders.
- Suggested reviewers have no personal, supervisory, employment, funding,
  collaboration, institutional, or adversarial conflict.

## Save Proof With These Exact Names

Save all proof files in:

`submission_jss_20260512_135646/post_submission_proof/`

Required names:

- `01_submission_confirmation.pdf` or `.png`
- `02_confirmation_email.eml` or `.txt` or `.html`
- `03_uploaded_file_list.pdf` or `.png`
- `04_portal_pdf_proof.pdf`

If the portal gives a manuscript ID only as text, keep it ready for the command
below.

## One Command After Proof Exists

From the repository root:

```powershell
python -B scripts\record_jss_standard_proof.py --manuscript-id JSSOFTWARE-D-26-01113 --confirm-all-manual-gates --force
```

This command records the four standard proof files, records the manual
confirmations, and runs the strict completion audit.

Expected only after real submission proof exists:

```text
completion_status=complete
```

Then stop the paid Lightning instance unless a specific new experiment is
approved.
