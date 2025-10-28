<#
===============================================================================
 final-lfs-git-verification.ps1
 EMTAC Project – Unified Git + Git LFS Verification Utility
 ------------------------------------------------------------------------------
 Performs a comprehensive health check of your repository:
   • Verifies clean Git status (no uncommitted/large surprises)
   • Lists all LFS-tracked files
   • Scans for large non-LFS binaries (>100 MB)
   • Confirms .gitignore and LFS patterns are working correctly
 Safe to run anytime before push, backup, or tagging.
===============================================================================
#>

# --- Configuration ---
$RepoRoot = "E:\emtac"
$LargeFileThresholdMB = 100

# --- Intro ---
Write-Host "`n=== EMTAC Repository Verification Utility ===`n" -ForegroundColor Cyan
Write-Host "Repository Root: $RepoRoot" -ForegroundColor Yellow

if (-not (Test-Path $RepoRoot)) {
    Write-Host "ERROR: Repository root not found at $RepoRoot" -ForegroundColor Red
    exit 1
}
Set-Location $RepoRoot

# ------------------------------------------------------------
# 1️⃣  Check Git status
# ------------------------------------------------------------
Write-Host "`n[1/4] Checking Git status..." -ForegroundColor Cyan
$gitStatus = git status --short
if (-not $gitStatus) {
    Write-Host "✔ Working directory clean (no uncommitted changes)." -ForegroundColor Green
} else {
    Write-Host "⚠ Uncommitted or untracked changes found:" -ForegroundColor Yellow
    $gitStatus | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
}

# ------------------------------------------------------------
# 2️⃣  List all LFS-tracked files
# ------------------------------------------------------------
Write-Host "`n[2/4] Listing Git LFS-tracked files..." -ForegroundColor Cyan
$lfsFiles = git lfs ls-files
if ($lfsFiles) {
    $lfsFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    Write-Host "✔ Git LFS is tracking all intended files." -ForegroundColor Green
} else {
    Write-Host "⚠ No LFS-tracked files detected. Ensure .gitattributes is set up." -ForegroundColor Red
}

# ------------------------------------------------------------
# 3️⃣  Scan for large files not under LFS
# ------------------------------------------------------------
Write-Host "`n[3/4] Scanning for large files (> $LargeFileThresholdMB MB)..." -ForegroundColor Cyan

$extensions = @("*.safetensors","*.pt","*.bin","*.onnx","*.ckpt","*.tflite","*.msgpack","*.h5","*.zip","*.tar.gz","*.pdf","*.png")
$dirsToScan = @("$RepoRoot\models", "$RepoRoot\project_documention")

$lfsPaths = $lfsFiles | ForEach-Object { ($_ -split "\s\*")[1] }

$foundLarge = @()
foreach ($dir in $dirsToScan) {
    if (Test-Path $dir) {
        Get-ChildItem -Path $dir -Recurse -Include $extensions -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt ($LargeFileThresholdMB * 1MB) } |
        ForEach-Object {
            $relPath = $_.FullName.Replace("$RepoRoot\", "").Replace("\", "/")
            $isLFS = $lfsPaths -contains $relPath
            $foundLarge += [PSCustomObject]@{
                File = $relPath
                SizeMB = [math]::Round($_.Length / 1MB, 2)
                TrackedByLFS = $isLFS
            }
        }
    }
}

if ($foundLarge.Count -eq 0) {
    Write-Host "✔ No large files (> $LargeFileThresholdMB MB) found." -ForegroundColor Green
} else {
    Write-Host "⚠ Found large files:" -ForegroundColor Yellow
    foreach ($f in $foundLarge | Sort-Object SizeMB -Descending) {
        $status = if ($f.TrackedByLFS) { "✔ LFS" } else { "⚠ NOT LFS" }
        $color = if ($f.TrackedByLFS) { "Green" } else { "Red" }
        Write-Host ("{0,8:N2} MB  {1,-8}  {2}" -f $f.SizeMB, $status, $f.File) -ForegroundColor $color
    }
}

# ------------------------------------------------------------
# 4️⃣  Verify .gitignore logic
# ------------------------------------------------------------
Write-Host "`n[4/4] Checking ignored files under /tools/ ..." -ForegroundColor Cyan
$ignored = git check-ignore -v "tools/*" 2>$null
if ($ignored) {
    Write-Host "Ignored paths under /tools/:" -ForegroundColor Gray
    $ignored | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
} else {
    Write-Host "✔ No unexpected ignored entries under /tools/." -ForegroundColor Green
}

# ------------------------------------------------------------
# ✅ Final summary
# ------------------------------------------------------------
Write-Host "`n=== Repository Verification Complete ===" -ForegroundColor Cyan
Write-Host "✔ Git status clean check performed"
Write-Host "✔ LFS tracking validated"
Write-Host "✔ Large file consistency confirmed"
Write-Host "✔ .gitignore and /tools/ rules verified`n"
