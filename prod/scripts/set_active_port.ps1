param(
[int]$ActivePort = 5000,

[int]$TargetPort = 8102,

[string]$ListenAddress = "0.0.0.0",

[string]$TargetAddress = "127.0.0.1",

[switch]$OpenFirewall

)

function Test-IsAdmin {
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
Write-Warning "This script should be run as Administrator because netsh portproxy and firewall changes require admin rights."
}

Write-Host "Setting active port mapping:"
Write-Host " $ListenAddress:$ActivePort -> $TargetAddress:$TargetPort"

Delete previous mapping if present.

& netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$ActivePort *> $null

Add new mapping.

& netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$ActivePort connectaddress=$TargetAddress connectport=$TargetPort

if ($LASTEXITCODE -ne 0) {
throw "Failed to create portproxy mapping. Run PowerShell as Administrator."
}

if ($OpenFirewall) {
if (-not (Test-IsAdmin)) {
Write-Warning "OpenFirewall requested, but not running as Administrator."
}
else {
$RuleName = "EMTAC Active $ActivePort"
$ExistingRule = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue

    if (-not $ExistingRule) {
        New-NetFirewallRule `
            -DisplayName $RuleName `
            -Direction Inbound `
            -Protocol TCP `
            -LocalPort $ActivePort `
            -Action Allow | Out-Null

        Write-Host "Firewall rule added: $RuleName"
    }
    else {
        Write-Host "Firewall rule already exists: $RuleName"
    }
}

}

Write-Host ""
Write-Host "Current portproxy mappings:"
netsh interface portproxy show v4tov4

Write-Host ""
Write-Host "Test locally:"
Write-Host " Invoke-WebRequest http://127.0.0.1:$ActivePort/health -UseBasicParsing"
