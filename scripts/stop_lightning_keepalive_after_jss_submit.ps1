param(
    [string]$SshTarget = $env:LIGHTNING_SSH_TARGET,
    [string]$KeyPath = "C:\Users\Ashish\.ssh\lightning_rsa",
    [string]$RemoteWorkDir = "/teamspace/studios/this_studio/bottleneck_t4_work",
    [switch]$ConfirmStop,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$auditScript = Join-Path $Root "scripts\audit_jss_completion.py"

if ([string]::IsNullOrWhiteSpace($SshTarget)) {
    Write-Error "Set LIGHTNING_SSH_TARGET or pass -SshTarget."
}

if (-not (Test-Path -LiteralPath $KeyPath)) {
    Write-Error "SSH key not found: $KeyPath"
}

if (-not (Test-Path -LiteralPath $auditScript)) {
    Write-Error "Completion audit script not found: $auditScript"
}

Push-Location $Root
try {
    python -B scripts\audit_jss_completion.py
    $auditCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($auditCode -ne 0) {
    Write-Host "Refusing to stop keepalive: JSS completion audit is not complete."
    Write-Host "Save portal proof, run record_jss_submission.py, then rerun this script."
    if ($DryRun) {
        Write-Host "dry_run=refusal_confirmed"
        exit 0
    }
    exit 1
}

if (-not $ConfirmStop) {
    Write-Host "Completion audit passed, but -ConfirmStop was not provided."
    Write-Host "Rerun with -ConfirmStop to stop the keepalive process."
    if ($DryRun) {
        Write-Host "dry_run=no_stop_without_confirm"
        exit 0
    }
    exit 1
}

$remote = @"
cd '$RemoteWorkDir' || exit 1
echo before:
pgrep -af gpu_keepalive_bottleneck_jss_safe || true
pkill -f gpu_keepalive_bottleneck_jss_safe || true
sleep 2
echo after:
pgrep -af gpu_keepalive_bottleneck_jss_safe || echo keepalive_not_running
"@

if ($DryRun) {
    Write-Host "dry_run=would_stop_keepalive"
    Write-Host "target=$SshTarget"
    Write-Host "remote_workdir=$RemoteWorkDir"
    exit 0
}

$remote = $remote -replace "`r", ""
$remote | ssh `
    -i $KeyPath `
    -o BatchMode=yes `
    -o ConnectTimeout=20 `
    -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=/dev/null `
    $SshTarget `
    "bash -s"
