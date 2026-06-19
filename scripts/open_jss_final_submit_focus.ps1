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
    (Join-Path $Root "submission_jss_20260512_135646\post_submission_proof")
)

Write-Host "JSS final-submit focus mode"
Write-Host "Root: $Root"
Write-Host "C: free space: $FreeGiB GiB"
Write-Host "Guard: submit and save proof; do not reopen manuscript work unless a gate fails."
Write-Host ""
Write-Host "Portal URLs:"
foreach ($url in $PortalUrls) {
    Write-Host "- $url"
    if (-not $DryRun) {
        Start-Process $url
    }
}

Write-Host ""
Write-Host "Focus files/folders:"
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
Write-Host "After successful portal submit, save the four standard proof files and run:"
Write-Host "python -B scripts\record_jss_standard_proof.py --manuscript-id <ID> --confirm-all-manual-gates"
