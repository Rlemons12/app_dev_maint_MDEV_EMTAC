<#
.SYNOPSIS
    Interactive multi-commit Git bundle generator for EMTAC offline machine.
.DESCRIPTION
    Lets you review each changed file, choose whether to stage it,
    reuse a global commit message or provide a custom one,
    then bundles the repository for transfer to the online machine.
#>

# --- CONFIGURATION ---
$RepoPath  = "E:\emtac\projects\llm\MDEV_EMTAC"
$OutputDir = "E:\emtac"
$Branch    = "main"
$LogFile   = "E:\emtac\tools\custom_scripts\ps1\bundle_log.txt"

Write-Host "`n=== EMTAC Offline Commit & Bundle Creator ===`n" -ForegroundColor Cyan

# --- Verify repo ---
if (-not (Test-Path $RepoPath)) { Write-Host "❌ Repo not found at $RepoPath" -ForegroundColor Red; exit 1 }
Set-Location $RepoPath

# --- Detect changes ---
$Changes = git status --porcelain | ForEach-Object { $_.Trim() }
if (-not $Changes) {
    Write-Host "✅ No untracked or modified files." -ForegroundColor Green
} else {
    Write-Host "⚠️  Detected changed/untracked files:`n" -ForegroundColor Yellow
    $Changes | ForEach-Object { Write-Host " • $_" }
    Write-Host ""
}

if ($Changes) {
    $globalMsg = Read-Host "Enter a global commit message (leave blank if none)"
    foreach ($line in $Changes) {
        $file = $line.Substring(3)
        Write-Host "`n📄 File: $file" -ForegroundColor Cyan
        $stage = Read-Host "Stage this file? (y/n)"
        if ($stage -ne 'y') { continue }

        $useGlobal = $false
        if ($globalMsg -ne '') {
            $choice = Read-Host "Use global message for this file? (y/n)"
            if ($choice -eq 'y') { $useGlobal = $true }
        }

        git add "$file"

        if ($useGlobal) {
            git commit -m "$globalMsg" | Out-Null
        } else {
            $customMsg = Read-Host "Enter commit message for $file"
            if ([string]::IsNullOrWhiteSpace($customMsg)) {
                $customMsg = "Updated $file"
            }
            git commit -m "$customMsg" | Out-Null
        }
        Write-Host "✅ Committed $file" -ForegroundColor Green
    }
}

# --- Verify branch ---
$currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne $Branch) {
    Write-Host "⚠️  You are on branch '$currentBranch', not '$Branch'." -ForegroundColor Yellow
    $switch = Read-Host "Switch to '$Branch'? (y/n)"
    if ($switch -eq 'y') { git checkout $Branch | Out-Null }
}

# --- Create bundle ---
$LastCommit = (git log -1 --pretty=format:"%h - %s") 2>$null
$Timestamp  = Get-Date -Format "yyyyMMdd_HHmm"
$BundleName = "update_$Timestamp.bundle"
$BundlePath = Join-Path $OutputDir $BundleName

Write-Host "`n📦 Creating bundle $BundleName ..." -ForegroundColor Yellow
git bundle create $BundlePath $Branch | Out-Null

if ($LASTEXITCODE -ne 0 -or -not (Test-Path $BundlePath)) {
    Write-Host "❌ Failed to create bundle." -ForegroundColor Red
    exit 1
}

$sizeMB = [math]::Round((Get-Item $BundlePath).Length / 1MB,2)
Write-Host "✅ Bundle created: $BundlePath ($sizeMB MB)" -ForegroundColor Green

# --- Log entry ---
$TimestampLog = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$LogEntry = @"
[$TimestampLog]
Last Commit: $LastCommit
Bundle: $BundleName
Size: ${sizeMB}MB
Path: $BundlePath

"@
Add-Content -Path $LogFile -Value $LogEntry
Write-Host "`n🗒️  Bundle creation logged to: $LogFile" -ForegroundColor Yellow
Write-Host "`n=== Done ===`n" -ForegroundColor Cyan
