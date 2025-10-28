<#
.SYNOPSIS
    Lists all ignored folders in the EMTAC Git repository and saves a report.

.DESCRIPTION
    This script detects ignored directories (not files) using .gitignore rules,
    prints them to the console, and saves a timestamped report in the repo root.
#>

# --- Repository root ---
$repoRoot = "E:\emtac"

if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
    Write-Host "Git repository not found at $repoRoot. Please update the path if needed." -ForegroundColor Red
    exit 1
}

Set-Location $repoRoot

# --- Generate timestamp and output file ---
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportFile = Join-Path $repoRoot "ignored_folders_report_$timestamp.txt"

# --- Run Git command to detect ignored folders ---
Write-Host "`nScanning for ignored folders in: $repoRoot ..." -ForegroundColor Cyan
$ignoredDirs = git ls-files --others --ignored --exclude-standard -d | Sort-Object

if ($ignoredDirs.Count -eq 0) {
    Write-Host "`nNo ignored folders found in this repository." -ForegroundColor Green
    exit 0
}

# --- Display results ---
Write-Host "`nIgnored folders found:`n" -ForegroundColor Yellow
$ignoredDirs | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }

# --- Save results ---
$ignoredDirs | Out-File -FilePath $reportFile -Encoding UTF8

# --- Summary ---
Write-Host "`nTotal ignored folders: $($ignoredDirs.Count)" -ForegroundColor Cyan
Write-Host "Report saved to: $reportFile`n" -ForegroundColor Green
