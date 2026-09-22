# Register (or remove) the scheduled scan in Windows Task Scheduler.
#
#   powershell -ExecutionPolicy Bypass -File tools\install_schedule.ps1
#   powershell -ExecutionPolicy Bypass -File tools\install_schedule.ps1 -Hours 2
#   powershell -ExecutionPolicy Bypass -File tools\install_schedule.ps1 -Remove
#
# Defaults to hourly. Resolution lag lives in hours, not days, and one cycle
# costs well under a cent of inference -- but each scanned market is also one
# Google News request, so going much below hourly at 40 markets will get you
# throttled by the news source long before TypeSafe notices.

param(
    [int]$Hours = 1,
    [string]$TaskName = "polymarket-jev",
    [string]$Config = "",
    [switch]$Forecast,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Remove) {
    schtasks /delete /tn $TaskName /f
    Write-Output "Removed scheduled task '$TaskName'."
    exit 0
}

$script = Join-Path $root "tools\scheduled_run.ps1"
if (-not (Test-Path $script)) { throw "missing $script" }

$inner = "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$script`""
if ($Config)   { $inner += " -Config `"$Config`"" }
if ($Forecast) { $inner += " -Forecast" }

$command = "powershell.exe $inner"

schtasks /create /tn $TaskName /tr $command /sc hourly /mo $Hours /f /rl LIMITED

Write-Output ""
Write-Output "Installed '$TaskName', every $Hours hour(s)."
Write-Output ""
Write-Output "  runs      : $command"
Write-Output "  logs      : data\logs\<date>.log"
Write-Output "  inspect   : schtasks /query /tn $TaskName /v /fo LIST"
Write-Output "  run now   : schtasks /run /tn $TaskName"
Write-Output "  remove    : powershell -File tools\install_schedule.ps1 -Remove"
Write-Output ""
Write-Output "Everything it does is paper. Live execution is gated in"
Write-Output "pmjev/live.py and cannot be switched on from the schedule."
