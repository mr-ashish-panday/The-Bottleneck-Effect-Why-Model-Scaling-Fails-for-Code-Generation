# JSS Preflight Report

Checks passed: 6
Checks failed: 0

| Status | Check | Evidence |
|---|---|---|
| PASS | claim-to-artifact audit | `checks=32 passed=32 failed=0` |
| PASS | upload manifest verifier | `upload_manifest_checks=7 failed=0` |
| PASS | archive readability and counts | `jss_source_package.zip: entries=23, expected=23, bad=None, noise_hits=0<br>jss_supplement_artifact_full.zip: entries=398, expected=398, bad=None, noise_hits=0` |
| PASS | latex log health | `no unresolved refs/cites, rerun warnings, LaTeX warnings, or overfull boxes` |
| PASS | data availability wording | `local bottleneck.tex: curated=True, stale_repo_reference=False<br>source zip bottleneck.tex: curated=True, stale_repo_reference=False<br>supplement zip bottleneck.tex: curated=True, stale_repo_reference=False` |
| PASS | sensitive/stale text scan | `no sensitive strings or stale artifact-repository references found` |
