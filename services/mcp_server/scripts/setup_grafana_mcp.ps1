param(
    [string]$GrafanaUrl = "http://localhost:3000",
    [string]$ServiceAccountName = "mcp-grafana-readwrite",
    [ValidateSet("Viewer", "Editor", "Admin")]
    [string]$ServiceAccountRole = "Editor",
    [string]$TokenName = "mcp-grafana-token",
    [int]$SecondsToLive = 604800,
    [string]$EnvPath = ".env",
    [string]$AdminUser = "admin",
    [string]$AdminPassword = "",
    [string]$AdminBearerToken = "",
    [switch]$NoEnvUpdate
)

$ErrorActionPreference = "Stop"

function New-AuthHeader {
    if (-not [string]::IsNullOrWhiteSpace($AdminBearerToken)) {
        return @{ Authorization = "Bearer $AdminBearerToken" }
    }

    if ([string]::IsNullOrWhiteSpace($AdminPassword)) {
        $secure = Read-Host "Grafana admin password" -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

        try {
            $script:AdminPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }

    $bytes = [System.Text.Encoding]::UTF8.GetBytes("${AdminUser}:${script:AdminPassword}")
    $basic = [Convert]::ToBase64String($bytes)
    return @{ Authorization = "Basic $basic" }
}

function Invoke-GrafanaJson {
    param(
        [ValidateSet("GET", "POST", "PATCH")]
        [string]$Method,
        [string]$Path,
        [hashtable]$Headers,
        [object]$Body = $null
    )

    $uri = "$($GrafanaUrl.TrimEnd('/'))$Path"
    $request = @{
        Method = $Method
        Uri = $uri
        Headers = $Headers
        ContentType = "application/json"
    }

    if ($null -ne $Body) {
        $request.Body = ($Body | ConvertTo-Json -Depth 10)
    }

    return Invoke-RestMethod @request
}

function Set-EnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $line = "$Name=$Value"

    if (Test-Path $Path) {
        $lines = @(Get-Content $Path)
        $updated = $false

        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match "^$([regex]::Escape($Name))=") {
                $lines[$i] = $line
                $updated = $true
                break
            }
        }

        if (-not $updated) {
            $lines += $line
        }

        [System.IO.File]::WriteAllText((Resolve-Path $Path), (($lines -join "`n") + "`n"), [System.Text.UTF8Encoding]::new($false))
        return
    }

    [System.IO.File]::WriteAllText((Join-Path (Get-Location) $Path), "$line`n", [System.Text.UTF8Encoding]::new($false))
}

$headers = New-AuthHeader

Write-Host "Searching for Grafana service account '$ServiceAccountName'."
$query = [uri]::EscapeDataString($ServiceAccountName)
$search = Invoke-GrafanaJson -Method "GET" -Path "/api/serviceaccounts/search?perpage=100&page=1&query=$query" -Headers $headers
$serviceAccount = $search.serviceAccounts | Where-Object { $_.name -eq $ServiceAccountName } | Select-Object -First 1

if (-not $serviceAccount) {
    Write-Host "Creating Grafana service account '$ServiceAccountName' with role '$ServiceAccountRole'."
    $serviceAccount = Invoke-GrafanaJson `
        -Method "POST" `
        -Path "/api/serviceaccounts" `
        -Headers $headers `
        -Body @{
            name = $ServiceAccountName
            role = $ServiceAccountRole
            isDisabled = $false
        }
}
elseif ($serviceAccount.role -ne $ServiceAccountRole) {
    Write-Host "Updating Grafana service account role to '$ServiceAccountRole'."
    $serviceAccount = Invoke-GrafanaJson `
        -Method "PATCH" `
        -Path "/api/serviceaccounts/$($serviceAccount.id)" `
        -Headers $headers `
        -Body @{
            name = $ServiceAccountName
            role = $ServiceAccountRole
        }
}
else {
    Write-Host "Grafana service account already has role '$ServiceAccountRole'."
}

Write-Host "Creating Grafana service account token '$TokenName'."
$token = Invoke-GrafanaJson `
    -Method "POST" `
    -Path "/api/serviceaccounts/$($serviceAccount.id)/tokens" `
    -Headers $headers `
    -Body @{
        name = $TokenName
        secondsToLive = $SecondsToLive
    }

if ([string]::IsNullOrWhiteSpace($token.key)) {
    throw "Grafana did not return a service account token key."
}

if (-not $NoEnvUpdate) {
    Set-EnvValue -Path $EnvPath -Name "GRAFANA_MCP_ENABLED" -Value "true"
    Set-EnvValue -Path $EnvPath -Name "GRAFANA_URL" -Value $GrafanaUrl
    Set-EnvValue -Path $EnvPath -Name "GRAFANA_SERVICE_ACCOUNT_TOKEN" -Value $token.key
    Set-EnvValue -Path $EnvPath -Name "GRAFANA_SERVICE_ACCOUNT_NAME" -Value $ServiceAccountName
    Set-EnvValue -Path $EnvPath -Name "GRAFANA_SERVICE_ACCOUNT_ROLE" -Value $ServiceAccountRole
}

Write-Host "Grafana MCP service account is ready."
Write-Host "Service account: $ServiceAccountName"
Write-Host "Role: $ServiceAccountRole"
Write-Host "Token: created and stored in $EnvPath"
