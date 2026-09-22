# One scheduled cycle: pull, settle, scan, execute on paper, push.
#
# Install it with tools\install_schedule.ps1, or run it by hand to see what a
# scheduled cycle does before handing it to Task Scheduler.
#
#   powershell -ExecutionPolicy Bypass -File tools\scheduled_run.ps1
#   powershell -ExecutionPolicy Bypass -File tools\scheduled_run.ps1 -NoPush
#
# Everything here is paper. -Live is deliberately not a parameter: live
# execution is gated inside pmjev/live.py, not by a flag in a scheduled task.

param(
    [string]$Config = "",
    [switch]$Forecast,
    [switch]$NoPush,
    [switch]$NoExecute
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$startStamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$logDir = Join-Path $root "data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ((Get-Date).ToUniversalTime().ToString("yyyy-MM-dd") + ".log")

function Write-Log($message) {
    # Stamp per line, not per cycle: a scan takes minutes and a single
    # start-of-run timestamp makes a slow cycle look instantaneous.
    $now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$now] $message"
    Write-Output $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Write-Log "cycle start"

# Pull first so the judgment and position logs include whatever another device
# wrote. .gitattributes merges both with the union driver, so this is safe.
if (-not $NoPush) {
    try {
        git pull --quiet --no-rebase 2>&1 | Out-Null
        Write-Log "pulled"
    } catch {
        Write-Log "pull failed (continuing offline): $_"
    }
}

$scanArgs = @("run.py", "--scan")
if ($Config) { $scanArgs += @("--config", $Config) }
if ($Forecast) { $scanArgs += "--forecast" }
if (-not $NoExecute) { $scanArgs += "--execute" }

# Settle anything that resolved since the last cycle, before opening more.
try {
    $settle = & python run.py --positions 2>&1 | Out-String
    Write-Log "positions checked"
    Add-Content -Path $logFile -Value $settle -Encoding utf8
} catch {
    Write-Log "position check failed: $_"
}

try {
    $output = & python @scanArgs 2>&1 | Out-String
    Add-Content -Path $logFile -Value $output -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Log "scan exited $LASTEXITCODE"
    } else {
        $trades = ([regex]::Matches($output, "OPENED")).Count
        Write-Log "scan ok, $trades position(s) opened"
    }
} catch {
    Write-Log "scan failed: $_"
    exit 1
}

# The logs are the accumulated evidence; they have to leave this machine.
if (-not $NoPush) {
    try {
        git add data/judgments.jsonl data/positions.jsonl 2>&1 | Out-Null
        $dirty = git status --porcelain data/judgments.jsonl data/positions.jsonl
        if ($dirty) {
            git commit --quiet -m "scan $startStamp" 2>&1 | Out-Null
            git push --quiet 2>&1 | Out-Null
            Write-Log "pushed"
        } else {
            Write-Log "nothing new to push"
        }
    } catch {
        Write-Log "push failed (logs remain local): $_"
    }
}

Write-Log "cycle done"
