<#
===============================================================================
 export-offline-repo.ps1
 EMTAC Project – Offline Repository Export Utility
 -------------------------------------------------------------------------------
 Creates a portable Git + Git LFS bundle for transfer from an offline environment
 to an online machine. Includes LFS binary data and logs all actions.
===============================================================================
#>

# --- Configuration ---
$RepoRoot      = "E:\emtac"
$BundleDir     = "E:\emtac_repo_bundle"
$LogDir        = "E:\emtac\logs"
$Timestamp     = Get-Date -Format "yyyyMMdd_HHmmss"
$BundleFile    = Join-Path $BundleDir "emtac_offline_$Timestamp.bundle"
$LfsExportDir  = Join-Path $BundleDir "lfs_export"
$LogFile       = Join-Path $LogDir "repo_bundle_$Timestamp.log"

# --- Ensure directories exist ---
New-Item -ItemType Directory -Path $BundleDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# --- Start logging ---
Start-Transcript -Path $LogFile -Force
Write-Host "=== EMTAC Offline Repo Export Utility ===" -ForegroundColor Cyan
Write-Host "Repository Root: $RepoRoot"
Write-Host "Bundle Destination: $BundleFile"
Write-Host "LFS Export Directory: $LfsExportDir"
Write-Host "`nStarting export process at $(Get-Date)...`n" -ForegroundColor Yellow

# --- Step 1: Verify repo path ---
if (-not (Test-Path $RepoRoot)) {
    Write-Host "ERROR: Repository root not found at $RepoRoot" -ForegroundColor Red
    Stop-Transcript
    exit 1
}

Set-Location $RepoRoot

# --- Step 2: Commit pending changes ---
Write-Host "`n[1/5] Committing pending changes..." -ForegroundColor Cyan
git add . | Out-Null
$commitMsg = "Offline sync point - exported $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
git commit -m "$commitMsg" 2>$null
git status

# --- Step 3: Fetch all LFS objects ---
Write-Host "`n[2/5] Fetching all Git LFS objects..." -ForegroundColor Cyan
git lfs fetch --all

# --- Step 4: Create Git bundle ---
Write-Host "`n[3/5] Creating Git bundle..." -ForegroundColor Cyan
git bundle create $BundleFile --all
if ($LASTEXITCODE -eq 0) {
    Write-Host "? Bundle created successfully at $BundleFile" -ForegroundColor Green
} else {
    Write-Host "? Failed to create bundle." -ForegroundColor Red
    Stop-Transcript
    exit 1
}

# --- Step 5: Export LFS binary data ---
Write-Host "`n[4/5] Exporting Git LFS objects..." -ForegroundColor Cyan
Remove-Item -Recurse -Force $LfsExportDir -ErrorAction SilentlyContinue
git lfs export $LfsExportDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "? LFS export completed successfully at $LfsExportDir" -ForegroundColor Green
} else {
    Write-Host "? LFS export failed (check logs)." -ForegroundColor Red
}

# --- Step 6: Verify bundle ---
Write-Host "`n[5/5] Verifying bundle integrity..." -ForegroundColor Cyan
git bundle verify $BundleFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "? Bundle verification passed." -ForegroundColor Green
} else {
    Write-Host "? Verification reported issues (check output)." -ForegroundColor Yellow
}

Write-Host "`n=== Export Complete ===" -ForegroundColor Cyan
Write-Host "Bundle saved to: $BundleFile"
Write-Host "LFS data exported to: $LfsExportDir"
Write-Host "Log file: $LogFile`n"

Stop-Transcript
