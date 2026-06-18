param(
[ValidateSet("blue", "green")]
[string]$Slot = "green"
)

$ProdRoot = Split-Path $PSScriptRoot -Parent
$SlotRoot = Join-Path (Join-Path $ProdRoot "slots") $Slot
$Python = Join-Path $SlotRoot ".venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path $Python)) {
throw "Slot Python not found: $Python"
}

$LockFile = Join-Path $SlotRoot "requirements.$Slot.lock.txt"
$TimestampedLock = Join-Path $SlotRoot "requirements.$Slot.lock_$Stamp.txt"
$PipCheckFile = Join-Path $SlotRoot "pip_check_$Stamp.txt"

Write-Host "Freezing $Slot requirements"
Write-Host "Python: $Python"
Write-Host "Lock: $LockFile"

& $Python -m pip freeze | Out-File $LockFile -Encoding utf8
Copy-Item $LockFile $TimestampedLock -Force

Write-Host "Running pip check"
& $Python -m pip check | Tee-Object -FilePath $PipCheckFile

Write-Host ""
Write-Host "Created:"
Write-Host " $LockFile"
Write-Host " $TimestampedLock"
Write-Host " $PipCheckFile"

Write-Host ""
Write-Host "Key packages:"
Select-String -Path $LockFile -Pattern "waitress|Flask|SQLAlchemy|psycopg2|pandas|docx2pdf|PyMuPDF|fuzzywuzzy|openai|sentence-transformers|python-pptx|comtypes|pgvector|pywin32|torch|transformers|spacy|en-core-web-sm"

