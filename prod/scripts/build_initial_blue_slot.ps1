[CmdletBinding()]
param(
    [string]$SourceRoot = "E:\emtac\projects\llm\MDEV_EMTAC",
    [string]$ProdRoot = "E:\emtac\prod",
    [switch]$ReplaceExistingBlue
)

$ErrorActionPreference = "Stop"

function Read-SlotExclusionFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $Sections = @{
        exclude_dirs           = New-Object System.Collections.Generic.List[string]
        exclude_files          = New-Object System.Collections.Generic.List[string]
        exclude_specific       = New-Object System.Collections.Generic.List[string]
        post_copy_remove_files = New-Object System.Collections.Generic.List[string]
        post_copy_remove_dirs  = New-Object System.Collections.Generic.List[string]
    }

    if (-not (Test-Path $Path)) {
        throw "Slot exclusion file not found: $Path"
    }

    $CurrentSection = $null

    foreach ($RawLine in Get-Content $Path) {
        $Line = ($RawLine -as [string]).Trim()

        if ([string]::IsNullOrWhiteSpace($Line)) {
            continue
        }

        if ($Line.StartsWith("#")) {
            continue
        }

        if ($Line -match '^\[(.+?)\]$') {
            $SectionName = $Matches[1].Trim()

            if (-not $Sections.ContainsKey($SectionName)) {
                $Sections[$SectionName] = New-Object System.Collections.Generic.List[string]
            }

            $CurrentSection = $SectionName
            continue
        }

        if ($null -ne $CurrentSection) {
            $Sections[$CurrentSection].Add($Line)
        }
    }

    return $Sections
}

function Invoke-RobocopyChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    robocopy @Arguments

    $Code = $LASTEXITCODE

    if ($Code -gt 7) {
        throw "$StepName failed. Robocopy exit code: $Code"
    }

    Write-Host "$StepName completed. Robocopy exit code: $Code"
}

function Move-BadItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ItemPath,

        [Parameter(Mandatory = $true)]
        [string]$QuarantineRoot
    )

    if (-not (Test-Path $ItemPath)) {
        return
    }

    $Name = Split-Path $ItemPath -Leaf
    $Destination = Join-Path $QuarantineRoot $Name

    if (Test-Path $Destination) {
        $Destination = Join-Path $QuarantineRoot ("{0}_{1}" -f $Name, (Get-Date -Format "yyyyMMdd_HHmmss"))
    }

    Write-Host "Moving unwanted item to quarantine:"
    Write-Host "  From: $ItemPath"
    Write-Host "  To:   $Destination"

    Move-Item $ItemPath $Destination -Force
}

$SlotsRoot = Join-Path $ProdRoot "slots"
$Blue = Join-Path $SlotsRoot "blue"
$Green = Join-Path $SlotsRoot "green"

$Shared = Join-Path $ProdRoot "shared"
$BackupRoot = Join-Path $Shared "backups"
$LogRoot = Join-Path $Shared "logs"
$UploadsRoot = Join-Path $Shared "uploads"
$QuarantineRoot = Join-Path $Shared "cleanup_quarantine"

$ExclusionFile = Join-Path $ProdRoot "slot_exclusions.txt"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$GreenVenv = Join-Path $Green ".venv"
$BlueVenv = Join-Path $Blue ".venv"
$BluePython = Join-Path $BlueVenv "Scripts\python.exe"

if (-not (Test-Path $SourceRoot)) {
    throw "Source root not found: $SourceRoot"
}

if (-not (Test-Path $Green)) {
    throw "Green slot not found: $Green"
}

if (-not (Test-Path $GreenVenv)) {
    throw "Green venv not found: $GreenVenv"
}

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $UploadsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $QuarantineRoot | Out-Null

$Exclusions = Read-SlotExclusionFile -Path $ExclusionFile

$ExcludeDirs = @($Exclusions.exclude_dirs)
$ExcludeFilesFromFile = @($Exclusions.exclude_files)
$ExcludeSpecific = @($Exclusions.exclude_specific)
$PostCopyRemoveFiles = @($Exclusions.post_copy_remove_files)
$PostCopyRemoveDirs = @($Exclusions.post_copy_remove_dirs)

# Robocopy treats names beginning with "-" as options.
# Do not pass those through /XF. They are handled by post-copy cleanup.
$SafeExcludeFiles = @()
foreach ($FilePattern in $ExcludeFilesFromFile) {
    if ($FilePattern.StartsWith("-")) {
        if ($PostCopyRemoveFiles -notcontains $FilePattern) {
            $PostCopyRemoveFiles += $FilePattern
        }

        continue
    }

    $SafeExcludeFiles += $FilePattern
}

# Convert specific relative files into source-rooted paths for robocopy /XF.
$SpecificFilesForRobocopy = @()

foreach ($Specific in $ExcludeSpecific) {
    $SpecificPath = Join-Path $SourceRoot $Specific

    if ($Specific -match '\.[^\\\/]+$') {
        $SpecificFilesForRobocopy += $SpecificPath
    }
}

$ExcludeFiles = @($SafeExcludeFiles + $SpecificFilesForRobocopy)

Write-Host ""
Write-Host "Using exclusion file:"
Write-Host "  $ExclusionFile"
Write-Host ""
Write-Host "Excluded directories:"
$ExcludeDirs | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Excluded files:"
$ExcludeFiles | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Post-copy remove dirs:"
$PostCopyRemoveDirs | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Post-copy remove files:"
$PostCopyRemoveFiles | ForEach-Object { Write-Host "  $_" }
Write-Host ""

if (Test-Path $Blue) {
    if (-not $ReplaceExistingBlue) {
        throw "Blue slot already exists: $Blue. Re-run with -ReplaceExistingBlue to move it to backup."
    }

    $BlueBackup = Join-Path $BackupRoot "blue_before_rebuild_$Stamp"
    Write-Host "Moving existing blue slot to backup:"
    Write-Host $BlueBackup
    Move-Item $Blue $BlueBackup
}

New-Item -ItemType Directory -Force -Path $Blue | Out-Null

Write-Host ""
Write-Host "Copying app source to blue slot..."

$RoboArgs = @(
    $SourceRoot,
    $Blue,
    "/E",
    "/NFL",
    "/NDL",
    "/NP",
    "/R:2",
    "/W:2",
    "/MT:16",
    "/XJ"
)

if ($ExcludeDirs.Count -gt 0) {
    $RoboArgs += @("/XD")
    $RoboArgs += $ExcludeDirs
}

if ($ExcludeFiles.Count -gt 0) {
    $RoboArgs += @("/XF")
    $RoboArgs += $ExcludeFiles
}

Invoke-RobocopyChecked -Arguments $RoboArgs -StepName "Source copy"

Write-Host ""
Write-Host "Running post-copy cleanup..."

$PostCopyQuarantine = Join-Path $QuarantineRoot "blue_post_copy_cleanup_$Stamp"
New-Item -ItemType Directory -Force -Path $PostCopyQuarantine | Out-Null

foreach ($Dir in $PostCopyRemoveDirs) {
    $Path = Join-Path $Blue $Dir
    Move-BadItem -ItemPath $Path -QuarantineRoot $PostCopyQuarantine
}

foreach ($File in $PostCopyRemoveFiles) {
    $Path = Join-Path $Blue $File
    Move-BadItem -ItemPath $Path -QuarantineRoot $PostCopyQuarantine
}

Write-Host ""
Write-Host "Copying green venv to blue venv..."

$VenvRoboArgs = @(
    $GreenVenv,
    $BlueVenv,
    "/MIR",
    "/NFL",
    "/NDL",
    "/NP",
    "/R:2",
    "/W:2",
    "/MT:16",
    "/XJ"
)

Invoke-RobocopyChecked -Arguments $VenvRoboArgs -StepName "Venv copy"

if (-not (Test-Path $BluePython)) {
    throw "Blue Python not found after venv copy: $BluePython"
}

Write-Host ""
Write-Host "Creating blue .env from green .env..."

$GreenEnv = Join-Path $Green ".env"
$BlueEnv = Join-Path $Blue ".env"

if (-not (Test-Path $GreenEnv)) {
    throw "Green .env not found: $GreenEnv"
}

$EnvText = Get-Content $GreenEnv -Raw

$EnvText = $EnvText.Replace("E:\emtac\prod\slots\green", "E:\emtac\prod\slots\blue")
$EnvText = $EnvText -replace "(?m)^EMTAC_SLOT=.*$", "EMTAC_SLOT=blue"
$EnvText = $EnvText -replace "(?m)^PORT=.*$", "PORT=8101"
$EnvText = $EnvText -replace "(?m)^LOG_FILE=.*$", "LOG_FILE=E:\emtac\prod\shared\logs\blue_app.log"

Set-Content -Path $BlueEnv -Value $EnvText -Encoding UTF8

Write-Host ""
Write-Host "Creating Database junction..."

$BlueDb = Join-Path $Blue "Database"

if (Test-Path $BlueDb) {
    $Item = Get-Item $BlueDb -Force

    if ($Item.LinkType -eq "Junction") {
        cmd /c rmdir "$BlueDb"
    }
    else {
        $DbBackup = Join-Path $PostCopyQuarantine "blue_physical_database_removed_$Stamp"
        Move-Item $BlueDb $DbBackup -Force
    }
}

cmd /c mklink /J "$BlueDb" "E:\emtac\Database" | Out-Host

Write-Host ""
Write-Host "Creating blue wsgi.py..."

@"
from ai_emtac import create_app

app = create_app(request_id="blue-wsgi")
application = app
"@ | Set-Content -Path (Join-Path $Blue "wsgi.py") -Encoding UTF8

Write-Host ""
Write-Host "Writing blue lock file from copied venv..."

& $BluePython -m pip freeze | Set-Content (Join-Path $Blue "requirements.blue.lock.txt") -Encoding UTF8

Write-Host ""
Write-Host "Verifying blue imports from blue working directory..."

Push-Location $Blue

try {
    & $BluePython -c "import sys; print('sys.prefix:', sys.prefix)"
    & $BluePython -c "import waitress; print('waitress OK')"
    & $BluePython -c "from pgvector.sqlalchemy import Vector; import pgvector; print('pgvector:', pgvector.__file__)"
    & $BluePython -c "from ai_emtac import create_app; print('create_app OK')"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Blue root after build:"
Get-ChildItem $Blue -Force |
    Sort-Object Name |
    Select-Object Mode, Name, LinkType, Target |
    Format-Table -Auto

Write-Host ""
Write-Host "Initial blue slot build complete."
Write-Host "Blue slot: $Blue"
Write-Host "Blue port: 8101"
Write-Host ""
Write-Host "Next:"
Write-Host "cd E:\emtac\prod"
Write-Host ".\scripts\start_emtac_slot.ps1 -Slot blue -Port 8101 -HostAddress 0.0.0.0 -Force"
Write-Host ".\scripts\test_emtac_slot.ps1 -HostName 127.0.0.1 -Port 8101"
