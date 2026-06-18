param(
[string]$HostName = "127.0.0.1",
[int]$Port = 8102
)

$Urls = @(
"http://$($HostName):$Port/health",
"http://$($HostName):$Port/api-status"
)

foreach ($Url in $Urls) {
Write-Host ""
Write-Host "Testing $Url"

try {
    $Response = Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 15
    Write-Host "OK $($Response.StatusCode)"
    Write-Host $Response.Content
}
catch {
    Write-Host "FAILED"
    Write-Host $_.Exception.Message
}

}
