Write-Host "=== EMTAC Git Ignore Verification ===" -ForegroundColor Cyan

$RepoRoot = "E:\emtac"
$TargetFolder = "tools"

if (-not (Test-Path $RepoRoot)) {
    Write-Host "ERROR: Repository root not found at $RepoRoot" -ForegroundColor Red
    exit 1
}

Set-Location $RepoRoot
Write-Host ("Repository Root: {0}" -f $RepoRoot) -ForegroundColor Yellow
Write-Host ("Checking Git tracking status for /{0}/`n" -f $TargetFolder)

$tracked = git ls-files $TargetFolder
$ignored = git check-ignore -v "$TargetFolder\*" 2>$null

Write-Host "-> Tracked files and folders under /$TargetFolder/:" -ForegroundColor Green
if ($tracked) {
    $tracked | ForEach-Object { Write-Host ("  TRACKED: {0}" -f $_) -ForegroundColor Gray }
} else {
    Write-Host "  No tracked files under $TargetFolder/." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "-> Ignored files and folders under /$TargetFolder/:" -ForegroundColor Magenta
if ($ignored) {
    $ignored | ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor Gray }
} else {
    Write-Host "  No ignored items detected under $TargetFolder/." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan

$gitkeepPath = Join-Path $RepoRoot "$TargetFolder\.gitkeep"
if (Test-Path $gitkeepPath) {
    Write-Host "OK: The folder tools/ itself is tracked (contains .gitkeep)" -ForegroundColor Green
} else {
    Write-Host "WARN: tools/.gitkeep not found or not tracked ? ensure it exists for folder retention" -ForegroundColor Red
}

$shouldTrack = @("utilities", "custom_scripts")
$subfolders = Get-ChildItem -Path (Join-Path $RepoRoot $TargetFolder) -Directory -ErrorAction SilentlyContinue

foreach ($sub in $subfolders) {
    $subName = $sub.Name
    if ($shouldTrack -contains $subName) {
        Write-Host ("OK: Subfolder '{0}' is intended to be tracked." -f $subName) -ForegroundColor Green
    } else {
        Write-Host ("IGNORE: Subfolder '{0}' should be ignored per .gitignore rules." -f $subName) -ForegroundColor DarkRed
    }
}

Write-Host ""
Write-Host "Verification complete." -ForegroundColor Cyan
