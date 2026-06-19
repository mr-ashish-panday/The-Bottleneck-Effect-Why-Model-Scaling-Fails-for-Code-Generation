# Post-Submission Proof Folder

Save JSS portal proof files here after final submit.

Required proof:

- Manuscript ID / confirmation page screenshot.
- Confirmation email.
- Portal-generated PDF proof.
- Final uploaded-file list shown by the portal.
- Any editor/system acknowledgement.

Preferred exact filenames:

- `01_submission_confirmation.pdf` or `.png`
- `02_confirmation_email.eml` or `.txt` or `.html`
- `03_uploaded_file_list.pdf` or `.png`
- `04_portal_pdf_proof.pdf`

Use normal nonempty proof files such as `.pdf`, `.png`, `.jpg`, `.txt`, `.eml`,
or `.html`. `README.md` does not count as proof. The four required proof
categories must point to distinct files.

After saving proof files with the preferred names, record the submission from
the repository root:

```powershell
python -B scripts\record_jss_standard_proof.py --manuscript-id <ID> --confirm-all-manual-gates
```

If you used different proof filenames, run the longer recorder:

```powershell
python -B scripts/record_jss_submission.py --manuscript-id <ID> `
  --confirmation-proof submission_jss_20260512_135646/post_submission_proof/<confirmation-screenshot-or-pdf> `
  --email-proof submission_jss_20260512_135646/post_submission_proof/<confirmation-email> `
  --uploaded-file-list-proof submission_jss_20260512_135646/post_submission_proof/<uploaded-file-list> `
  --portal-pdf-proof submission_jss_20260512_135646/post_submission_proof/<portal-generated-proof>
```

Then run the completion audit:

```powershell
python -B scripts/audit_jss_completion.py
```

Expected after a real submit and proof record:

```text
completion_status=complete
```

The recorder updates:

- `..\..\JSS_POST_SUBMISSION_TRACKER.md`
- `..\..\JSS_FINAL_SUBMISSION_AUDIT.md`

Also update `..\..\JSS_SUBMISSION_CHECKLIST.md` and the global project map if
needed.

Then stop the Lightning instance unless a specific new experiment is approved.
