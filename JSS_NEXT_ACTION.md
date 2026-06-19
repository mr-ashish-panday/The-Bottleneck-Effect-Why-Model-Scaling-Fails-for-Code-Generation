# JSS Next Action

Status: submitted to The Journal of Systems & Software as `JSSOFTWARE-D-26-01113`.

Last updated: 2026-05-24 NPT. Submission confirmation email was received at
9:10 PM NPT.

The next action is not another experiment, another rebuild, or another venue
decision. It is proof closeout and status monitoring.

## Do This Now

1. Save the confirmation screen/PDF if the portal still shows it.
2. Save the final uploaded-file list if the portal still shows it.
3. Keep `02_confirmation_email.txt` and `04_portal_pdf_proof.pdf` in the proof folder.
4. When all required distinct proof files exist, run the guarded recorder with
   manuscript ID `JSSOFTWARE-D-26-01113`; because the tracker already records
   the ID, use `--force`.
5. Run `scripts/audit_jss_completion.py --report JSS_COMPLETION_AUDIT_REPORT.md`
   and require it to pass before calling the local proof bundle complete.
6. Stop/recheck the Lightning instance unless it has already been closed.

## Do Not Do More Local Work First

- Disk was usable for portal upload but tight: `C:` had about 4.94 GiB free at
  2026-05-17 08:21 NPT. Use `python scripts/jss_submission_status.py` for the
  live free-space readout and do not run another local rebuild unless a
  validation gate fails.
- The full preflight passed on the current package after deterministic archive
  rebuilds.
- The upload manifest and claim audit passed on the current package.
- The last recorded GPU check was 2026-05-12 11:41 UTC: idle except keepalive.
  Recheck only if you need to stop the instance after submission.
- Repo-local safe cleanup is not meaningful: bytecode, LaTeX auxiliaries, and
  root transient files total only about 0.33 MB, while research logs/results
  should be preserved.

## Commands Only If Evidence Is Challenged

```powershell
python scripts/verify_jss_upload_manifest.py
python scripts/run_jss_preflight.py
```

Expected:

```text
upload_manifest_checks=7 failed=0
jss_preflight_checks=6 failed=0
```
