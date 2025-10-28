# ===========================================================
# verify_gitkeep_structure.ps1
# Checks every folder under E:\emtac for .gitkeep presence
# and reports tracked vs untracked structure folders.
# ===========================================================

$RepoRoot = "E:\emtac"
Set-Location $RepoRoot

Write-Host ""
Write-Host "🔍 Scanning folder structure under: $RepoRoot" -ForegroundColor Cyan
Write-Host "==========================================================="

# Get all directories recursively
$dirs = Get-ChildItem -Path $RepoRoot -Directory -Recurse | Where-Object {
    $_.FullName -notmatch '\\\.git($|\\)' -and $_.FullName -notmatch '\\\.venv'
}

$tracked = @()
$untracked = @()

foreach ($dir in $dirs) {
    $gitkeep = Join-Path $dir.FullName ".gitkeep"
    if (Test-Path $gitkeep) {
        $tracked += $dir.FullName.Replace("$RepoRoot\", "")
    } else {
        $untracked += $dir.FullName.Replace("$RepoRoot\", "")
    }
}

Write-Host ""
Write-Host "✅ Folders with .gitkeep (tracked):" -ForegroundColor Green
if ($tracked.Count -gt 0) {
    $tracked | Sort-Object | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   None found."
}

Write-Host ""
Write-Host "⚠️  Folders missing .gitkeep (untracked):" -ForegroundColor Yellow
if ($untracked.Count -gt 0) {
    $untracked | Sort-Object | ForEach-Object { Write-Host "   $_" }
} else {
    Write-Host "   All folders are tracked."
}

Write-Host ""
Write-Host "✨ Verification complete." -ForegroundColor Cyan
Write-Host "==========================================================="
