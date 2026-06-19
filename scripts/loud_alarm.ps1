param(
    [double]$DelayMinutes = 0,
    [int[]]$Frequencies = @(880, 1175, 1568, 1175),
    [int]$ToneMilliseconds = 450,
    [int]$GapMilliseconds = 80,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Test-AlarmArgs {
    if ($DelayMinutes -lt 0) {
        throw "DelayMinutes must be 0 or greater."
    }
    if ($ToneMilliseconds -lt 50) {
        throw "ToneMilliseconds must be at least 50."
    }
    if ($GapMilliseconds -lt 0) {
        throw "GapMilliseconds must be 0 or greater."
    }
    foreach ($frequency in $Frequencies) {
        if ($frequency -lt 37 -or $frequency -gt 32767) {
            throw "Frequency $frequency is outside the Windows beep range: 37-32767 Hz."
        }
    }
}

function Start-Countdown {
    param([double]$Minutes)

    $seconds = [int][Math]::Round($Minutes * 60)
    while ($seconds -gt 0) {
        Write-Host ("Alarm starts in {0:mm\:ss}. Close this window to cancel." -f [TimeSpan]::FromSeconds($seconds)) -NoNewline
        Start-Sleep -Seconds 1
        Write-Host "`r" -NoNewline
        $seconds--
    }
    Write-Host ""
}

Test-AlarmArgs

Write-Host "LOUD ALARM"
Write-Host "Stop it with Ctrl+C or by closing this terminal window."
Write-Host "Your speaker volume controls how loud it is."

if ($DryRun) {
    Write-Host "Dry run OK. No sound played."
    exit 0
}

if ($DelayMinutes -gt 0) {
    Start-Countdown -Minutes $DelayMinutes
}

try {
    while ($true) {
        foreach ($frequency in $Frequencies) {
            [Console]::Beep($frequency, $ToneMilliseconds)
            if ($GapMilliseconds -gt 0) {
                Start-Sleep -Milliseconds $GapMilliseconds
            }
        }
    }
}
finally {
    Write-Host ""
    Write-Host "Alarm stopped."
}
