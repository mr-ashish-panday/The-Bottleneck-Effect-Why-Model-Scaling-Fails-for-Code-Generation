param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Drive = Get-PSDrive -Name C
$FreeGiB = [Math]::Round(($Drive.Free / 1GB), 2)
$PortalUrls = @(
    "https://www.sciencedirect.com/journal/journal-of-systems-and-software",
    "https://www.editorialmanager.com/jssoftware/submit_manuscript.asp"
)
$Paths = @(
    (Join-Path $Root "JSS_FINAL_PORTAL_CARD.md"),
    (Join-Path $Root "JSS_NEXT_ACTION.md"),
    (Join-Path $Root "JSS_PORTAL_COPY_PASTE_PACKET.md"),
    (Join-Path $Root "JSS_UPLOAD_FILE_MANIFEST.csv"),
    (Join-Path $Root "JSS_MANUAL_CONFIRMATION_FORM.md"),
    (Join-Path $Root "JSS_PORTAL_UPLOAD_RUNBOOK.md"),
    (Join-Path $Root "JSS_POST_SUBMISSION_TRACKER.md"),
    (Join-Path $Root "submission_jss_20260512_135646"),
    (Join-Path $Root "submission_jss_20260512_135646\post_submission_proof")
)
$OptionalPaths = @(
    "C:\Users\Ashish\all\Ashish\Bottleneck JSS Submit Now.md"
)

Write-Host "JSS submission workspace"
Write-Host "Root: $Root"
Write-Host "C: free space: $FreeGiB GiB"
Write-Host "Guard: do not rebuild packages unless verify_jss_upload_manifest.py or run_jss_preflight.py fails."
Write-Host ""
Write-Host "Portal URLs:"
foreach ($url in $PortalUrls) {
    Write-Host "- $url"
    if (-not $DryRun) {
        Start-Process $url
    }
}

Write-Host ""
Write-Host "Local files/folders:"
foreach ($path in $Paths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing expected path: $path"
    }
    Write-Host "- $path"
    if (-not $DryRun) {
        Start-Process -FilePath $path
    }
}

Write-Host ""
Write-Host "Optional notes:"
foreach ($path in $OptionalPaths) {
    if (Test-Path -LiteralPath $path) {
        Write-Host "- $path"
        if (-not $DryRun) {
            Start-Process -FilePath $path
        }
    }
}

Write-Host ""
Write-Host "After successful portal submit, save proof in post_submission_proof and run:"
Write-Host "python -B scripts\record_jss_standard_proof.py --manuscript-id <ID> --confirm-all-manual-gates"
Write-Host "python -B scripts/record_jss_manual_confirmations.py --confirm-author-metadata --confirm-no-reviewer-conflicts --confirm-portal-classifications --confirm-required-uploads --confirm-portal-proof --confirm-proof-saved"
Write-Host "python -B scripts/record_jss_submission.py --manuscript-id <ID> --confirmation-proof <path> --email-proof <path> --uploaded-file-list-proof <path> --portal-pdf-proof <path>"
Write-Host "python -B scripts/audit_jss_completion.py"
