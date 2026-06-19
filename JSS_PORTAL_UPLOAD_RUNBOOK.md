# JSS Portal Upload Runbook

Target journal: Journal of Systems and Software.

Use this file while submitting in Elsevier Editorial Manager. Do not improvise
metadata from memory; use the files and hashes below.

For copy-paste fields, use `JSS_PORTAL_COPY_PASTE_PACKET.md`.

Official entry point:

- JSS journal page: https://www.sciencedirect.com/journal/journal-of-systems-and-software
- Use the `Submit your article` link on the JSS page. Elsevier states that the
  relevant submission system is accessed through the journal homepage submit
  link.
- Direct Editorial Manager landing page observed on 2026-05-12:
  https://www.editorialmanager.com/jssoftware/default.aspx
- If the landing page shows `Important Message: Site under development. Do not
  use for live manuscript submission.`, do not stop there. Click `Submit a
  Manuscript`; the observed author login route is:
  https://www.editorialmanager.com/jssoftware/submit_manuscript.asp
- If the live browser still blocks submission after login or repeats the
  development warning inside the author workflow, do not submit blindly. Contact
  Elsevier/JSS support from the portal using
  `JSS_PORTAL_SUPPORT_ESCALATION.md`, then save the screenshot and support
  ticket proof in `submission_jss_20260512_135646/post_submission_proof/`.

## Pre-Portal Stop Gate

Do not submit if any of these fail:

- `python scripts/audit_jss_submission_artifacts.py` must report
  `checks=32 passed=32 failed=0`.
- `python scripts/verify_jss_upload_manifest.py` must report
  `upload_manifest_checks=7 failed=0`.
- `python scripts/run_jss_preflight.py` must report
  `jss_preflight_checks=6 failed=0`.
- `JSS_FINAL_SUBMISSION_AUDIT.md` must show the current package hashes.
- The uploaded source ZIP hash must be
  `A321C0B0FC20B48C1366AF1ED870536E5EED10918B3B96D216A3E925E69F1271`.
- The uploaded supplement ZIP hash must be
  `31551AC28D359CE81367456053B46BBA8EC619AAD669E61F9F66B98C1E7BD376`.
- Ashish must confirm author affiliation, corresponding email, and reviewer
  conflicts before pressing final submit.
- The pre-submit sections of `JSS_MANUAL_CONFIRMATION_FORM.md` must be
  completed or answered in chat before final submit. Its post-submit proof
  section is completed immediately after submission.

## Portal Fields

Article type:

`Regular research article`

Title:

`The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation`

Author:

`Ashish Pandey`

Affiliation:

`Department of Computer and Electronics Engineering, Khwopa College of Engineering, Nepal`

Corresponding email:

`ashishpanday9818@gmail.com`

Keywords:

- Code generation
- Empirical software engineering
- Large language models
- Model evaluation
- Mechanistic interpretability
- Software reliability

Classifications, in preferred order if matching labels are available:

1. Artificial Intelligence applied in software engineering
2. Empirical software engineering
3. Software testing, verification, and validation
4. Software reliability
5. Mining software repositories / software analytics
6. Software engineering for AI systems

## Upload Order

1. Main manuscript PDF:
   `bottleneck.pdf`
2. Manuscript source package:
   `submission_jss_20260512_135646/jss_source_package.zip`
3. Highlights:
   `jss_highlights.txt`
4. Cover letter:
   `jss_cover_letter.txt`
5. Supplementary artifact package:
   `submission_jss_20260512_135646/jss_supplement_artifact_full.zip`
6. Artifact README, if the portal allows another supplement/document:
   `JSS_ARTIFACT_README.md`
7. Claim-to-artifact audit, if the portal allows another supplement/document:
   `JSS_CLAIM_ARTIFACT_AUDIT.md`

Use `JSS_UPLOAD_FILE_MANIFEST.csv` to verify file sizes and hashes before
upload.

If the portal only permits one supplement, upload the supplement ZIP first and
include the artifact README/audit only if there is an "additional files" slot.

## Declarations

Data availability:

Processed result tables, plotting scripts, benchmark summaries, saved result
artifacts, configuration files, source code, and the claim-to-artifact audit are
available in the curated supplementary artifact archive submitted with the
manuscript. Large model checkpoints and raw model caches are not redistributed;
all experiments use public pretrained checkpoints and regenerate derived outputs
from the provided scripts and configuration files.

Competing interest:

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

Funding:

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

Generative AI declaration:

During the preparation of this work, the author used OpenAI Codex to assist with
manuscript editing, code execution orchestration, artifact organization, and
formatting. After using this tool, the author reviewed and edited the content as
needed and takes full responsibility for the content of the submitted
manuscript.

## Suggested Reviewers

Only enter these after Ashish confirms no conflict.

If the portal asks for three:

1. David Lo
2. Michael Pradel
3. Baishakhi Ray

If it asks for five, add:

4. Lingming Zhang
5. Earl T. Barr

If it asks for six or more, add:

6. Martin Monperrus

Do not enter Thomas Zimmermann as an author-suggested reviewer because he is
listed on the JSS editorial board.

Use `JSS_SUGGESTED_REVIEWERS.md` for affiliations, emails, and profile links.

## Final Review Before Submit

Read the generated PDF proof from the portal, not only the local PDF.

Check:

- Title has no typo.
- Author name and affiliation are correct.
- No TMLR header appears.
- Figures render, especially Figure 15.
- Data Availability says "curated supplementary artifact archive".
- Source and supplement ZIPs are attached.
- Supplement ZIP size is accepted by the portal.
- Suggested reviewers have no conflicts.
- Declarations match `bottleneck.tex`.

## Proof To Save After Submission

After final submit, save:

- Manuscript ID.
- Submission confirmation PDF or screenshot.
- Confirmation email.
- Final uploaded file list shown by the portal.
- Any portal-generated PDF proof.

Then update `JSS_FINAL_SUBMISSION_AUDIT.md` with the manuscript ID and mark the
manual blockers resolved.

Also complete the post-submit proof section of `JSS_MANUAL_CONFIRMATION_FORM.md`
and run:

```powershell
python -B scripts/record_jss_submission.py --manuscript-id <ID> --confirmation-proof <path> --email-proof <path> --uploaded-file-list-proof <path> --portal-pdf-proof <path>
python -B scripts/record_jss_manual_confirmations.py --confirm-author-metadata --confirm-no-reviewer-conflicts --confirm-portal-classifications --confirm-required-uploads --confirm-portal-proof --confirm-proof-saved
python -B scripts/audit_jss_completion.py --report JSS_COMPLETION_AUDIT_REPORT.md
```

The first command updates `JSS_POST_SUBMISSION_TRACKER.md`,
`JSS_FINAL_SUBMISSION_AUDIT.md`, and `JSS_SUBMISSION_CHECKLIST.md` from the saved
proof files. The second records the manual confirmations. The third is the
strict completion gate.
