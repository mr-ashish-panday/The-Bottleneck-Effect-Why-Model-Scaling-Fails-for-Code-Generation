# Lightning Cost-Control Handoff

Date: 2026-05-14.

Purpose: prevent paying for an idle Lightning GPU after the JSS package is
prepared.

## Current State

- The last recorded Lightning check was reachable.
- Last checked: 2026-05-12 11:41 UTC through `scripts/check_lightning_status.ps1`.
- GPU state at that recorded check: Tesla T4, 0 MiB used, 0% utilization.
- No experiment is currently running.
- A CUDA keepalive loop is still active so the instance does not shut down for
  GPU idleness; recent log pulses were visible at 11:35 and 11:40 UTC.
- Observed keepalive process: `gpu_keepalive_bottleneck_jss_safe`.

## Decision

Do not launch more GPU experiments for the JSS version unless the
claim-to-artifact audit identifies a specific missing control that affects the
paper's argument.

Do not run local rebuilds or downloads before portal submission. A disk cleanup
on 2026-05-12 cleared only local package caches, not research artifacts:
`C:\Users\Ashish\AppData\Local\pip\Cache` (~4015.7 MB) and
`C:\Users\Ashish\AppData\Local\uv\cache` (~683.1 MB). Use
`python scripts/jss_submission_status.py` for the live `C:` free-space reading
before upload rather than trusting an old static number. Preserving research
logs/results, pulled backups, Hugging Face model cache, browser data, and final
upload packages remains higher priority than cosmetic cleanup.

Current best use of time:

1. Submit the prepared JSS package.
2. Save the manuscript ID, confirmation email, uploaded file list, and portal
   PDF proof.
3. Stop the Lightning instance if no new experiment is approved.

## If You Need To Keep The Instance Alive

Keep the current keepalive running only while an immediate upload or new
approved experiment depends on the studio staying open.

Check status:

```powershell
$env:LIGHTNING_SSH_TARGET = '<current Lightning SSH target>'
.\scripts\check_lightning_status.ps1
```

The helper intentionally reads the SSH target from `LIGHTNING_SSH_TARGET` so the
repository does not hard-code the active Lightning endpoint.

## If You Are Done With Compute

Stop the Lightning studio from the Lightning UI.

If you only need to stop the keepalive inside the studio before shutting down,
use the guarded helper. It refuses to stop keepalive until
`scripts/audit_jss_completion.py` passes:

```powershell
$env:LIGHTNING_SSH_TARGET = '<current Lightning SSH target>'
.\scripts\stop_lightning_keepalive_after_jss_submit.ps1 -ConfirmStop
```

Do not delete remote run logs or local pulled backups during cost control. They
are part of the research record.
