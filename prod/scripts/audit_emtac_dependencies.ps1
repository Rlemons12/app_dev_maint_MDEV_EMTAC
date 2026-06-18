param(
    [ValidateSet("blue", "green")]
    [string]$Slot = "green",

    [string]$ProdRoot = "E:\emtac\prod",

    [string]$WheelRoot = "E:\wheels",

    [switch]$OpenReport
)

function Normalize-PackageName {
    param([string]$Name)

    if ([string]::IsNullOrWhiteSpace($Name)) {
        return ""
    }

    return (($Name -replace "[-_.]+", "").ToLowerInvariant())
}

function Get-FreezePackageName {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return $null
    }

    if ($Line -match "^\s*([^=<>!~\s]+)\s*@") {
        return $Matches[1]
    }

    if ($Line -match "^\s*([^=<>!~\s]+)==") {
        return $Matches[1]
    }

    if ($Line -match "^\s*([^=<>!~\s]+)") {
        return $Matches[1]
    }

    return $null
}

function Get-PipShowValue {
    param(
        [string[]]$ShowLines,
        [string]$Key
    )

    $Line = $ShowLines | Where-Object { $_ -like "$($Key):*" } | Select-Object -First 1

    if (-not $Line) {
        return ""
    }

    return ($Line -replace "^$([regex]::Escape($Key)):\s*", "").Trim()
}

$SlotRoot = Join-Path (Join-Path $ProdRoot "slots") $Slot
$Python = Join-Path $SlotRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $Python)) {
    throw "Python not found for slot '$Slot': $Python"
}

if (-not (Test-Path $WheelRoot)) {
    throw "Wheel root not found: $WheelRoot"
}

Write-Host "Auditing EMTAC dependencies"
Write-Host "Slot:      $Slot"
Write-Host "SlotRoot:  $SlotRoot"
Write-Host "Python:    $Python"
Write-Host "WheelRoot: $WheelRoot"
Write-Host ""

$ManifestCsv = Join-Path $ProdRoot "dependency_manifest.$Slot.csv"
$ManifestJson = Join-Path $ProdRoot "dependency_manifest.$Slot.json"
$ManifestMd = Join-Path $ProdRoot "dependency_manifest.$Slot.md"
$MissingWheelsTxt = Join-Path $ProdRoot "missing_wheels.$Slot.txt"
$WheelInventoryCsv = Join-Path $ProdRoot "wheel_inventory.csv"

$ReqLock = Join-Path $SlotRoot "requirements.$Slot.lock.txt"
$ReqLockStamped = Join-Path $SlotRoot "requirements.$Slot.lock_$Stamp.txt"
$PipCheckFile = Join-Path $SlotRoot "pip_check_$Stamp.txt"

# ------------------------------------------------------------
# Freeze current slot environment
# ------------------------------------------------------------
Write-Host "Freezing current slot requirements..."
$FreezeLines = & $Python -m pip freeze

$FreezeLines | Out-File -FilePath $ReqLock -Encoding utf8
Copy-Item $ReqLock $ReqLockStamped -Force

# Map freeze lines by normalized package name.
$FreezeMap = @{}

foreach ($Line in $FreezeLines) {
    $PkgName = Get-FreezePackageName $Line
    if ($PkgName) {
        $FreezeMap[(Normalize-PackageName $PkgName)] = $Line
    }
}

# ------------------------------------------------------------
# Get pip list
# ------------------------------------------------------------
Write-Host "Reading installed packages..."
$PipListRaw = & $Python -m pip list --format=json
$Packages = $PipListRaw | ConvertFrom-Json

# ------------------------------------------------------------
# Inventory all wheels
# ------------------------------------------------------------
Write-Host "Scanning wheelhouse. This can take a minute..."
$WheelFiles = Get-ChildItem $WheelRoot -Recurse -Filter "*.whl" -File

$WheelRows = foreach ($Wheel in $WheelFiles) {
    $Parts = $Wheel.BaseName -split "-"

    $DistName = ""
    $Version = ""

    if ($Parts.Count -ge 2) {
        $DistName = $Parts[0]
        $Version = $Parts[1]
    }
    else {
        $DistName = $Wheel.BaseName
        $Version = ""
    }

    [pscustomobject]@{
        FileName = $Wheel.Name
        DistName = $DistName
        NormalizedName = Normalize-PackageName $DistName
        Version = $Version
        Directory = $Wheel.DirectoryName
        FullName = $Wheel.FullName
    }
}

$WheelRows |
    Sort-Object DistName, Version, FullName |
    Export-Csv -NoTypeInformation -Encoding UTF8 $WheelInventoryCsv

# ------------------------------------------------------------
# Build dependency manifest
# ------------------------------------------------------------
Write-Host "Building dependency manifest..."

$ManifestRows = foreach ($Pkg in ($Packages | Sort-Object name)) {
    $PkgName = [string]$Pkg.name
    $PkgVersion = [string]$Pkg.version
    $Norm = Normalize-PackageName $PkgName

    $FreezeSpec = ""
    if ($FreezeMap.ContainsKey($Norm)) {
        $FreezeSpec = $FreezeMap[$Norm]
    }

    $ShowLines = & $Python -m pip show $PkgName 2>$null

    $Location = Get-PipShowValue -ShowLines $ShowLines -Key "Location"
    $Requires = Get-PipShowValue -ShowLines $ShowLines -Key "Requires"
    $RequiredBy = Get-PipShowValue -ShowLines $ShowLines -Key "Required-by"
    $Installer = Get-PipShowValue -ShowLines $ShowLines -Key "Installer"

    $MatchingWheels = @($WheelRows | Where-Object { $_.NormalizedName -eq $Norm })
    $ExactVersionWheels = @($MatchingWheels | Where-Object { $_.Version -eq $PkgVersion })

    $SelectedWheels = @()
    if ($ExactVersionWheels.Count -gt 0) {
        $SelectedWheels = $ExactVersionWheels
    }
    else {
        $SelectedWheels = $MatchingWheels
    }

    $WheelFound = $MatchingWheels.Count -gt 0
    $ExactVersionWheelFound = $ExactVersionWheels.Count -gt 0
    $WheelPaths = ($SelectedWheels | Select-Object -ExpandProperty FullName -Unique) -join "; "

    $Notes = @()

    if (-not $WheelFound) {
        $Notes += "No matching wheel found under $WheelRoot."
    }
    elseif (-not $ExactVersionWheelFound) {
        $Notes += "Wheel found for package, but not exact installed version."
    }

    if ($FreezeSpec -match "file:///") {
        $Notes += "Freeze spec references a local file path."
    }

    if ((Normalize-PackageName $PkgName) -eq "pgvector") {
        $Notes += "Known issue: pgvector was manually copied because no pgvector wheel was found earlier."
    }

    if ((Normalize-PackageName $PkgName) -eq "pymupdf") {
        $Notes += "PyMuPDF provides import name 'fitz'. Installed version may differ from original requirements if wheelhouse lacked newer pin."
    }

    if ((Normalize-PackageName $PkgName) -eq "pywin32") {
        $Notes += "Provides pythoncom/win32com. pywin32_postinstall.py may be required after install."
    }

    [pscustomobject]@{
        Package = $PkgName
        Version = $PkgVersion
        FreezeSpec = $FreezeSpec
        InstalledLocation = $Location
        Installer = $Installer
        Requires = $Requires
        RequiredBy = $RequiredBy
        WheelFound = $WheelFound
        ExactVersionWheelFound = $ExactVersionWheelFound
        WheelPaths = $WheelPaths
        Notes = ($Notes -join " ")
    }
}

$ManifestRows |
    Sort-Object Package |
    Export-Csv -NoTypeInformation -Encoding UTF8 $ManifestCsv

$ManifestRows |
    Sort-Object Package |
    ConvertTo-Json -Depth 5 |
    Out-File -FilePath $ManifestJson -Encoding utf8

# ------------------------------------------------------------
# Missing wheel report
# ------------------------------------------------------------
$IgnoreMissing = @(
    "pip",
    "setuptools",
    "wheel"
) | ForEach-Object { Normalize-PackageName $_ }

$MissingRows = @(
    $ManifestRows |
        Where-Object {
            (-not $_.WheelFound) -and
            ($IgnoreMissing -notcontains (Normalize-PackageName $_.Package))
        } |
        Sort-Object Package
)

$MissingRows |
    Format-Table Package, Version, InstalledLocation, Notes -AutoSize |
    Out-String -Width 240 |
    Out-File -FilePath $MissingWheelsTxt -Encoding utf8

# ------------------------------------------------------------
# pip check
# ------------------------------------------------------------
Write-Host "Running pip check..."
& $Python -m pip check | Tee-Object -FilePath $PipCheckFile

# ------------------------------------------------------------
# Markdown report
# ------------------------------------------------------------
$KeyPackageNames = @(
    "waitress",
    "Flask",
    "Werkzeug",
    "SQLAlchemy",
    "Flask-SQLAlchemy",
    "Flask-Migrate",
    "psycopg2-binary",
    "pandas",
    "numpy",
    "spacy",
    "en-core-web-sm",
    "torch",
    "transformers",
    "sentence-transformers",
    "openai",
    "pgvector",
    "pywin32",
    "docx2pdf",
    "PyMuPDF",
    "python-pptx",
    "comtypes",
    "fuzzywuzzy",
    "python-Levenshtein",
    "python-docx",
    "openpyxl",
    "pdfplumber",
    "PyPDF2"
)

$KeyNorms = $KeyPackageNames | ForEach-Object { Normalize-PackageName $_ }

$KeyRows = @(
    $ManifestRows |
        Where-Object { $KeyNorms -contains (Normalize-PackageName $_.Package) } |
        Sort-Object Package
)

$TotalPackages = @($ManifestRows).Count
$WheelFoundCount = @($ManifestRows | Where-Object { $_.WheelFound }).Count
$ExactWheelFoundCount = @($ManifestRows | Where-Object { $_.ExactVersionWheelFound }).Count
$MissingCount = @($MissingRows).Count

$Md = New-Object System.Collections.Generic.List[string]

$Md.Add("# EMTAC Dependency Manifest - $Slot")
$Md.Add("")
$Md.Add("Generated: $(Get-Date -Format s)")
$Md.Add("")
$Md.Add("## Paths")
$Md.Add("")
$Md.Add("| Item | Path |")
$Md.Add("|---|---|")
$Md.Add("| Production root | `$ProdRoot` |")
$Md.Add("| Slot | `$SlotRoot` |")
$Md.Add("| Python | `$Python` |")
$Md.Add("| Wheelhouse | `$WheelRoot` |")
$Md.Add("| Requirements lock | `$ReqLock` |")
$Md.Add("| Timestamped lock | `$ReqLockStamped` |")
$Md.Add("| CSV manifest | `$ManifestCsv` |")
$Md.Add("| JSON manifest | `$ManifestJson` |")
$Md.Add("| Missing wheels | `$MissingWheelsTxt` |")
$Md.Add("| Wheel inventory | `$WheelInventoryCsv` |")
$Md.Add("")
$Md.Add("## Summary")
$Md.Add("")
$Md.Add("| Metric | Count |")
$Md.Add("|---|---:|")
$Md.Add("| Installed packages | $TotalPackages |")
$Md.Add("| Packages with any matching wheel found | $WheelFoundCount |")
$Md.Add("| Packages with exact-version wheel found | $ExactWheelFoundCount |")
$Md.Add("| Packages missing matching wheel | $MissingCount |")
$Md.Add("")
$Md.Add("## Key Runtime Packages")
$Md.Add("")
$Md.Add("| Package | Version | Wheel Found | Exact Version Wheel | Installed Location | Notes |")
$Md.Add("|---|---:|---:|---:|---|---|")

foreach ($Row in $KeyRows) {
    $Md.Add("| $($Row.Package) | $($Row.Version) | $($Row.WheelFound) | $($Row.ExactVersionWheelFound) | `$($Row.InstalledLocation)` | $($Row.Notes) |")
}

$Md.Add("")
$Md.Add("## Packages Missing Wheels")
$Md.Add("")

if ($MissingRows.Count -eq 0) {
    $Md.Add("No missing wheels detected, excluding pip/setuptools/wheel.")
}
else {
    $Md.Add("| Package | Version | Installed Location | Notes |")
    $Md.Add("|---|---:|---|---|")

    foreach ($Row in $MissingRows) {
        $Md.Add("| $($Row.Package) | $($Row.Version) | `$($Row.InstalledLocation)` | $($Row.Notes) |")
    }
}

$Md.Add("")
$Md.Add("## Known Lessons From Green Setup")
$Md.Add("")
$Md.Add("- Do not rely on the original full requirements file without verifying the wheelhouse.")
$Md.Add("- The full requirements file may include dev/training packages that production does not need.")
$Md.Add("- `pgvector` must be added to `E:\wheels` for clean future rebuilds, or copied from a known-good venv.")
$Md.Add("- `PyMuPDF` provides the import name `fitz`.")
$Md.Add("- `pywin32` provides `pythoncom` and may need `pywin32_postinstall.py -install`.")
$Md.Add("- `python-pptx` provides the import name `pptx`.")
$Md.Add("- `sentence-transformers` provides the import name `sentence_transformers`.")
$Md.Add("- Freeze requirements only after `/health` and `/api-status` pass.")
$Md.Add("")
$Md.Add("## Rebuild Checklist")
$Md.Add("")
$Md.Add("1. Create or copy the slot app folder.")
$Md.Add("2. Create `.venv`.")
$Md.Add("3. Install from the frozen lock and wheelhouse.")
$Md.Add("4. Manually handle any package listed in the missing wheel report.")
$Md.Add("5. Ensure `wsgi.py` exists and calls `ai_emtac.create_app()`.")
$Md.Add("6. Start slot with Waitress on `0.0.0.0:<slot-port>`.")
$Md.Add("7. Test `/health` and `/api-status`.")
$Md.Add("8. Point tablet port `5000` to the active slot.")
$Md.Add("9. Freeze the final working environment.")
$Md.Add("")

$Md | Out-File -FilePath $ManifestMd -Encoding utf8

Write-Host ""
Write-Host "Dependency audit complete."
Write-Host ""
Write-Host "Created:"
Write-Host "  $ManifestMd"
Write-Host "  $ManifestCsv"
Write-Host "  $ManifestJson"
Write-Host "  $MissingWheelsTxt"
Write-Host "  $WheelInventoryCsv"
Write-Host "  $ReqLock"
Write-Host "  $ReqLockStamped"
Write-Host "  $PipCheckFile"

if ($OpenReport) {
    notepad $ManifestMd
    notepad $MissingWheelsTxt
}


