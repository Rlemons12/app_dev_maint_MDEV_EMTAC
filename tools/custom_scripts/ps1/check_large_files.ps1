<#
===============================================================================
check-large-files.ps1
EMTAC Project – Large File Sanity Check (pre-commit)
------------------------------------------------------------------------------
Finds all files >100 MB that are tracked or visible to Git.
Helps ensure Git LFS and .gitignore rules are working correctly.
Safe to run anytime before a commit.
===============================================================================
#>

# --- Configuration ---
$RepoRoot = "E:\emtac"
$Threshold = 100MB
$ThresholdMB = [math]::Round(($Threshold / 1MB), 0)

# --- Validate repo path ---
if (-not (Test-Path $RepoRoot)) {
    Write-Host "ERROR: Repository root not found at $RepoRoot" -ForegroundColor Red
    exit 1
}
Set-Location $RepoRoot

Write-Host "`n=== EMTAC Large-File Sanity Check (Pre-Commit) ===" -ForegroundColor Cyan
Write-Host ("Scanning for files > {0} MB in {1} ..." -f $ThresholdMB, $RepoRoot) -ForegroundColor Yellow
Write-Host ""

# --- Get all tracked + untracked (non-ignored) files ---
$gitFiles = git ls-files -o -c --exclude-standard 2>$null | ForEach-Object { Join-Path $RepoRoot $_ }

# --- Filter to those exceeding threshold ---
$largeFiles = @()
foreach ($f in $gitFiles) {
    if (Test-Path $f) {
        $fileSize = (Get-Item $f).Length
        if ($fileSize -gt $Threshold) {
            $largeFiles += $f
        }
    }
}

# --- Reporting ---
if ($largeFiles.Count -eq 0) {
    Write-Host "✔ No visible files larger than $ThresholdMB MB detected." -ForegroundColor Green
    Write-Host "Git LFS and .gitignore appear to be configured correctly.`n" -ForegroundColor DarkGray
}
else {
    Write-Host "⚠ Files exceeding $ThresholdMB MB and visible to Git:`n" -ForegroundColor Red
    foreach ($file in $largeFiles | Sort-Object { (Get-Item $_).Length } -Descending) {
        $sizeMB = [math]::Round((Get-Item $file).Length / 1MB, 2)
        Write-Host ("{0,8:N2} MB  {1}" -f $sizeMB, $file) -ForegroundColor Yellow
    }

    Write-Host "`nTip: For each file above, either:" -ForegroundColor Cyan
    Write-Host "  - Add an LFS pattern for its extension in .gitattributes" -ForegroundColor DarkGray
    Write-Host "  - Or add its path to .gitignore if it shouldn’t be versioned" -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "=== Scan Complete ===`n" -ForegroundColor Cyan
