[CmdletBinding()]
param(
    [ValidateSet("blue", "green")]
    [string]$Slot = "green",

    [int]$Port = 0,

    [string]$HostAddress = "0.0.0.0",

    [string]$ProdRoot = "E:\emtac\prod",

    [int]$Threads = 12,

    [switch]$Force,

    [switch]$OpenFirewall
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

$SlotsRoot = Join-Path $ProdRoot "slots"
$SlotRoot = Join-Path $SlotsRoot $Slot
$RunRoot = Join-Path $ProdRoot "run"
$SharedRoot = Join-Path $ProdRoot "shared"
$LogRoot = Join-Path $SharedRoot "logs"

$Python = Join-Path $SlotRoot ".venv\Scripts\python.exe"
$WsgiPath = Join-Path $SlotRoot "wsgi.py"
$PidPath = Join-Path $RunRoot "$Slot.pid"

$OutLog = Join-Path $LogRoot "$Slot`_waitress.out.log"
$ErrLog = Join-Path $LogRoot "$Slot`_waitress.err.log"
$AppLog = Join-Path $LogRoot "$Slot`_app.log"

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

function Stop-ExistingProcess {
    param(
        [string]$PidFile,
        [int]$TargetPort,
        [switch]$ForceStop
    )

    if (-not $ForceStop) {
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($PidFile) -and (Test-Path $PidFile)) {
        $RawPidContent = Get-Content $PidFile -Raw -ErrorAction SilentlyContinue

        if ($null -eq $RawPidContent) {
            $RawPid = ""
        }
        else {
            $RawPid = $RawPidContent.ToString().Trim()
        }

        if (-not [string]::IsNullOrWhiteSpace($RawPid)) {
            $ExistingPid = 0

            if ([int]::TryParse($RawPid, [ref]$ExistingPid)) {
                $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue

                if ($ExistingProcess) {
                    Write-Host "Stopping existing $Slot process PID $ExistingPid"
                    Stop-Process -Id $ExistingPid -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Seconds 2
                }
                else {
                    Write-Host "Stale PID file found for $Slot. PID $ExistingPid is not running."
                }
            }
            else {
                Write-Host "PID file contained invalid PID text: '$RawPid'"
            }
        }
        else {
            Write-Host "PID file was empty: $PidFile"
        }

        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    $ListeningPid = Get-ListeningPidForPort -TargetPort $TargetPort

    if ($ListeningPid) {
        Write-Host "Stopping process listening on port $TargetPort PID $ListeningPid"
        Stop-Process -Id $ListeningPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

if (-not (Test-Path $SlotRoot)) {
    throw "Slot folder not found: $SlotRoot"
}

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Stop-ExistingProcess -PidFile $PidPath -TargetPort $Port -ForceStop:$Force

Write-Host "Verifying waitress is available..."
& $Python -c "import waitress; print('waitress OK')" | Out-Host

if ($LASTEXITCODE -ne 0) {
    throw "waitress is not available in $Python"
}

if (-not (Test-Path $WsgiPath)) {
@"
from ai_emtac import create_app

app = create_app(request_id="$Slot-wsgi")
application = app
"@ | Set-Content -Path $WsgiPath -Encoding UTF8
}

$env:APP_ENV = "production"
$env:FLASK_ENV = "production"
$env:FLASK_DEBUG = "0"
$env:EMTAC_SLOT = $Slot
$env:HOST = $HostAddress
$env:PORT = [string]$Port
$env:EMTAC_SHARED_DIR = $SharedRoot
$env:EMTAC_UPLOAD_FOLDER = Join-Path $SharedRoot "uploads"
$env:EMTAC_LOG_DIR = $LogRoot
$env:LOGS_DIR = $LogRoot
$env:LOG_FILE = $AppLog

$Listen = "$HostAddress`:$Port"
$ArgumentList = "-m waitress --listen=$Listen --threads=$Threads wsgi:app"

Write-Host ""
Write-Host "Starting EMTAC $Slot slot"
Write-Host "Slot root: $SlotRoot"
Write-Host "Listen: $Listen"
Write-Host "Threads: $Threads"
Write-Host "Python: $Python"
Write-Host "Logs:"
Write-Host " $OutLog"
Write-Host " $ErrLog"
Write-Host ""

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $SlotRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

if (-not $Process -or -not $Process.Id) {
    throw "Start-Process did not return a valid process ID."
}

$Process.Id | Set-Content -Path $PidPath -Encoding ASCII

if ($OpenFirewall) {
    try {
        New-NetFirewallRule `
            -DisplayName "EMTAC $Slot $Port" `
            -Direction Inbound `
            -Protocol TCP `
            -LocalPort $Port `
            -Action Allow `
            -ErrorAction SilentlyContinue | Out-Null
    }
    catch {
        Write-Warning "Firewall rule was not created. Run PowerShell as Administrator if needed."
    }
}

$HealthUrl = "http://127.0.0.1:$Port/health"
$Deadline = (Get-Date).AddSeconds(45)
$Healthy = $false

while ((Get-Date) -lt $Deadline) {
    $Process.Refresh()

    if ($Process.HasExited) {
        Write-Warning "$Slot process exited early."
        break
    }

    try {
        $Response = Invoke-WebRequest $HealthUrl -UseBasicParsing -TimeoutSec 3

        if ($Response.StatusCode -eq 200) {
            $Healthy = $true
            Write-Host "Health check OK: $HealthUrl -> 200"
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $Healthy) {
    Write-Host ""
    Write-Host "Startup failed or health check timed out."
    Write-Host ""
    Write-Host "STDERR tail:"
    Get-Content $ErrLog -Tail 120 -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "STDOUT tail:"
    Get-Content $OutLog -Tail 120 -ErrorAction SilentlyContinue

    throw "$Slot did not become healthy on port $Port."
}

Write-Host ""
Write-Host "$Slot started with PID $($Process.Id)"
Write-Host "PID file: $PidPath"
Write-Host ""
Write-Host "Local URL: http://127.0.0.1:$Port/"
Write-Host "Health URL: $HealthUrl"
Write-Host ""
Write-Host "For LAN access, use the machine IP with this port."
