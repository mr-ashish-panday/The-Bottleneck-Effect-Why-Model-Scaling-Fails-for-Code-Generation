param(
    [string]$SshTarget = $env:LIGHTNING_SSH_TARGET,
    [string]$KeyPath = "C:\Users\Ashish\.ssh\lightning_rsa",
    [string]$RemoteWorkDir = "/teamspace/studios/this_studio/bottleneck_t4_work"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SshTarget)) {
    Write-Error "Set LIGHTNING_SSH_TARGET or pass -SshTarget, for example: user@ssh.lightning.ai"
}

if (-not (Test-Path -LiteralPath $KeyPath)) {
    Write-Error "SSH key not found: $KeyPath"
}

$remote = @"
cd '$RemoteWorkDir' || exit 1
echo remote_utc=`$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo gpu_status:
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo relevant_processes:
ps -eo pid,etime,cmd | grep -E 'gpu_keepalive_bottleneck_jss_safe|python|accelerate|torchrun' | grep -v grep || true
echo keepalive_tail:
tail -n 10 outputs/logs/gpu_keepalive_bottleneck_jss.log 2>/dev/null || true
"@

$remote = $remote -replace "`r", ""
$remote | ssh `
    -i $KeyPath `
    -o BatchMode=yes `
    -o ConnectTimeout=20 `
    -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=/dev/null `
    $SshTarget `
    "bash -s"
