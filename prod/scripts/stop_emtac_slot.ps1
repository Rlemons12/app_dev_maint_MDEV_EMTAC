[CmdletBinding()]
param(
    [ValidateSet("blue", "green")]
    [string]$Slot = "green",

    [int]$Port = 0,

    [string]$ProdRoot = "E:\emtac\prod"
)

$ErrorActionPreference = "Stop"

if ($Port -le 0) {
    if ($Slot -eq "blue") {
        $Port = 8101
    }
    else {
        $Port = 8102
    }
}

$RunRoot = Join-Path $ProdRoot "run"
$PidPath = Join-Path $RunRoot "$Slot.pid"

function Get-ListeningPidForPort {
    param([int]$TargetPort)

    $Pattern = ":" + [regex]::Escape([string]$TargetPort) + "\s"

    $Lines = netstat -ano -p tcp |
        Select-String -Pattern "LISTENING" |
        Where-Object { $_.Line -match $Pattern }

    foreach ($Line in $Lines) {
        $Parts = $Line.Line.Trim() -split "\s+"
        if ($Parts.Count -ge 5) {
            $Candidate = 0
            if ([int]::TryParse($Parts[-1], [ref]$Candidate)) {
                return $Candidate
            }
        }
    }

    return $null
}

$StoppedSomething = $false

if (Test-Path $PidPath) {
    $RawPid = (Get-Content $PidPath -Raw -ErrorAction SilentlyContinue).Trim()

    if ($RawPid) {
        $ExistingPid = 0

        if ([int]::TryParse($RawPid, [ref]$ExistingPid)) {
            $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue

            if ($ExistingProcess) {
                Write-Host "Stopping $Slot process PID $ExistingPid"
                Stop-Process -Id $ExistingPid -Force -ErrorAction SilentlyContinue
                $StoppedSomething = $true
            }
            else {
                Write-Host "PID file existed but process was not running: $ExistingPid"
            }
        }
        else {
            Write-Host "PID file had invalid content: $RawPid"
        }
    }
    else {
        Write-Host "PID file was empty: $PidPath"
    }

    Remove-Item $PidPath -Force -ErrorAction SilentlyContinue
}
else {
    Write-Host "No PID file found for $Slot at $PidPath"
}

$ListeningPid = Get-ListeningPidForPort -TargetPort $Port

if ($ListeningPid) {
    Write-Host "Stopping process listening on port $Port PID $ListeningPid"
    Stop-Process -Id $ListeningPid -Force -ErrorAction SilentlyContinue
    $StoppedSomething = $true
}

if (-not $StoppedSomething) {
    Write-Host "No running $Slot process found on port $Port."
}
else {
    Start-Sleep -Seconds 2
    Write-Host "$Slot stopped."
}
