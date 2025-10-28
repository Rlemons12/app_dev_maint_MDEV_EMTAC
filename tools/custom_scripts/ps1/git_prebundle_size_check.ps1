# ==============================================================
# EMTAC - Git Pre-Bundle Size Check
# Lists the Top 10 Largest Tracked Files in the Repository
# ==============================================================

# --- CONFIGURATION ---
$RepoRoot = "E:\emtac"
$OutputDir = "$RepoRoot\git_repos\verify_logs"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportPath = "$OutputDir\prebundle_largefiles_$Timestamp.txt"

# --- HEADER ---
"=====================================================" | Out-File $ReportPath
" EMTAC GIT PRE-BUNDLE SIZE CHECK - $Timestamp" | Out-File $ReportPath -Append
"=====================================================" | Out-File $ReportPath -Append
"`nTop 10 largest tracked files:`n" | Out-File $ReportPath -Append

# --- MAIN LOGIC ---
Push-Location $RepoRoot

# Get all tracked files and their blob sizes
$files = git ls-files | ForEach-Object {
    $path = $_
    $size = git cat-file -s HEAD:$path 2>$null
    if ($size) {
        [PSCustomObject]@{
            Path = $path
            SizeBytes = [int64]$size
        }
    }
} | Sort-Object SizeBytes -Descending | Select-Object -First 10

# --- OUTPUT ---
$files | ForEach-Object {
    "{0,10:N0} bytes  {1}" -f $_.SizeBytes, $_.Path | Out-File $ReportPath -Append
}

Pop-Location

# --- SUMMARY ---
"`nReport saved to:`n$ReportPath" | Out-File $ReportPath -Append
Write-Host "✅ Pre-bundle report created at: $ReportPath"
Write-Host "------------------------------------------------------"
Write-Host ""
$files | Format-Table -AutoSize
