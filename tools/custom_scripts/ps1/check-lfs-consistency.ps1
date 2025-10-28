<#
===============================================================================
 check-lfs-consistency.ps1
 EMTAC Project – Git LFS Consistency Verification
 ------------------------------------------------------------------------------
 Scans all subdirectories (especially models and documentation) and lists
 any large binary files (>100 MB) that are NOT managed by Git LFS.
 Non-destructive — safe to run anytime.
===============================================================================
#>

Write-Host "`n=== Checking Git LFS Consistency ===`n" -ForegroundColor Cyan

# --- Threshold for "large" files in MB ---
$thresholdMB = 100

# --- File extensions of interest ---
$extensions = @("*.safetensors","*.pt","*.bin","*.onnx","*.ckpt","*.tflite","*.msgpack","*.h5")

# --- Directories to check ---
$scanDirs = @("E:\emtac\models", "E:\emtac\project_documention")

# --- Get list of LFS-managed files (just their paths) ---
$lfsFiles = git lfs ls-files | ForEach-Object {
    ($_ -split "\s\*")[1]  # Extract file path from "hash * path"
}

# --- Collect large files on disk ---
$foundLarge = @()
foreach ($dir in $scanDirs) {
    if (Test-Path $dir) {
        Get-ChildItem -Path $dir -Recurse -Include $extensions -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt ($thresholdMB * 1MB) } |
        ForEach-Object {
            $full = $_.FullName
            $relative = $full.Replace("E:\emtac\", "").Replace("\", "/")
            $isLfs = $false
            foreach ($lfs in $lfsFiles) {
                if ($relative -eq $lfs) { $isLfs = $true; break }
            }
            $foundLarge += [PSCustomObject]@{
                File = $relative
                SizeMB = [math]::Round($_.Length / 1MB, 2)
                TrackedByLFS = $isLfs
            }
        }
    }
}

if ($foundLarge.Count -eq 0) {
    Write-Host "No large files (> $thresholdMB MB) found in models or project_documention." -ForegroundColor Green
    exit
}

Write-Host "Files exceeding $thresholdMB MB:`n" -ForegroundColor Yellow

foreach ($item in $foundLarge | Sort-Object SizeMB -Descending) {
    $color = if ($item.TrackedByLFS) { "Green" } else { "Red" }
    $status = if ($item.TrackedByLFS) { "✅ LFS" } else { "⚠️ NOT LFS" }
    Write-Host ("{0,8:N2} MB  {1,-6}  {2}" -f $item.SizeMB, $status, $item.File) -ForegroundColor $color
}

Write-Host "`n=== Scan complete ===`n" -ForegroundColor Cyan
