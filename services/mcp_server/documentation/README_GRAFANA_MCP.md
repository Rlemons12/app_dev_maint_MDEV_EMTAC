# Grafana MCP Integration

This document describes how Grafana integrates with the EMTAC MCP stack.

Grafana is used for service observability and dashboarding. Optional Grafana MCP support allows the MCP Coordinator and MCP AI Layer to route Grafana-related requests to a downstream Grafana MCP server.

## Current EMTAC Flow

```text
Service Dashboard
    ↓
MCP AI Layer
    ↓
MCP Coordinator
    ↓
Grafana MCP downstream server
    ↓
Grafana API
```

Main local services:

```text
Service Dashboard        http://127.0.0.1:5100
Grafana                  http://127.0.0.1:3000
MCP Coordinator          http://127.0.0.1:9100/mcp
MCP AI Layer             http://127.0.0.1:9200/api/ai/chat
```

Grafana should normally be started from the Service Dashboard so the dashboard can track PID, uptime, and output.

## Recommended Startup Order

```text
1. PostgreSQL Server
2. Grafana
3. GPU Service
4. EMTAC MCP Coordinator
5. AI Gateway
6. MCP AI Layer
```

Grafana should be running before Grafana MCP routing is tested.

## Requirements

* Local Grafana service available at `http://127.0.0.1:3000`
* `mcp-grafana` installed in the project virtual environment, or `uv` / `uvx` available on `PATH`
* Grafana service account token
* MCP Coordinator enabled
* MCP AI Layer enabled

Recommended token posture:

```text
Use read-only or Viewer-style access first.
Use Editor/write access only when dashboard creation or mutation tools are required.
Do not store real tokens in source control.
```

## Official Grafana MCP References

```text
https://grafana.com/docs/grafana/latest/developer-resources/mcp/
https://grafana.com/docs/grafana/latest/developer-resources/mcp/reference/mcp-tools-table/
```

## Environment Configuration

Primary environment file:

```text
E:\emtac\dev_env\.env
```

Recommended local Grafana values:

```env
GRAFANA_URL=http://127.0.0.1:3000

GRAFANA_MCP_ENABLED=true
GRAFANA_MCP_COMMAND=.venv\Scripts\mcp-grafana.exe
GRAFANA_MCP_ARGS=
GRAFANA_MCP_ENV_VARS=GRAFANA_URL,GRAFANA_SERVICE_ACCOUNT_TOKEN

GRAFANA_SERVICE_ACCOUNT_TOKEN=your_service_account_token
GRAFANA_SERVICE_ACCOUNT_NAME=mcp-grafana-readonly
GRAFANA_SERVICE_ACCOUNT_ROLE=Viewer
GRAFANA_SERVICE_ACCOUNT_TOKEN_SECONDS_TO_LIVE=604800
```

If using `uvx`:

```env
GRAFANA_MCP_COMMAND=uvx
GRAFANA_MCP_ARGS=mcp-grafana
```

For Grafana Cloud:

```env
GRAFANA_URL=https://your-instance.grafana.net
GRAFANA_SERVICE_ACCOUNT_TOKEN=your_cloud_service_account_token
```

## Service Account Guidance

Start with a read-focused service account.

Suggested name:

```text
mcp-grafana-readonly
```

Suggested role:

```text
Viewer
```

Use read/write access only when needed.

Suggested read/write name:

```text
mcp-grafana-readwrite
```

Suggested read/write role:

```text
Editor
```

Do not use Admin unless required for setup or service-account automation.

## Create or Update Local Grafana Service Account

For a local Grafana instance:

```powershell
Set-Location "E:\emtac\services\mcp_server"

.\scripts\setup_grafana_mcp.ps1 `
  -GrafanaUrl "http://127.0.0.1:3000" `
  -ServiceAccountName "mcp-grafana-readonly" `
  -ServiceAccountRole "Viewer"
```

For read/write access:

```powershell
Set-Location "E:\emtac\services\mcp_server"

.\scripts\setup_grafana_mcp.ps1 `
  -GrafanaUrl "http://127.0.0.1:3000" `
  -ServiceAccountName "mcp-grafana-readwrite" `
  -ServiceAccountRole "Editor"
```

The script should create or update:

```env
GRAFANA_SERVICE_ACCOUNT_TOKEN=...
GRAFANA_SERVICE_ACCOUNT_NAME=...
GRAFANA_SERVICE_ACCOUNT_ROLE=...
```

## Grafana Cloud Setup

For Grafana Cloud or token-based admin authentication:

```powershell
Set-Location "E:\emtac\services\mcp_server"

.\scripts\setup_grafana_mcp.ps1 `
  -GrafanaUrl "https://your-instance.grafana.net" `
  -AdminBearerToken "admin_token_with_service_account_permissions" `
  -ServiceAccountName "mcp-grafana-readonly" `
  -ServiceAccountRole "Viewer"
```

## Run Grafana From Dashboard

Start the Service Dashboard:

```powershell
Set-Location "E:\emtac\services\service_dashboard"

$env:PYTHONPATH = "E:\emtac\services\service_dashboard;$env:PYTHONPATH"
$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m app.main
```

Open:

```text
http://127.0.0.1:5100
```

Start Grafana from the dashboard.

Then verify Grafana health:

```powershell
Invoke-RestMethod "http://127.0.0.1:3000/api/health" |
ConvertTo-Json -Depth 10
```

Expected:

```text
database = ok
```

## Run Grafana Manually

Manual startup is useful for troubleshooting.

```powershell
Set-Location "E:\emtac\services\grafana-12.3.1"

& ".\bin\grafana.exe" server --homepath "E:\emtac\services\grafana-12.3.1"
```

Open:

```text
http://127.0.0.1:3000
```

## Run Grafana MCP Wrapper Directly

Manual run:

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m listed_server.mcp_grafana.server
```

The wrapper delegates to the official `mcp-grafana` command configured by:

```env
GRAFANA_MCP_COMMAND=...
GRAFANA_MCP_ARGS=...
```

## Run MCP Coordinator

Preferred method: start from the Service Dashboard.

Manual debug run:

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m listed_server.mcp_coordinator.server
```

Endpoint:

```text
http://127.0.0.1:9100/mcp
```

Note: `/mcp` is a streamable HTTP MCP endpoint. Plain GET requests may return HTTP `400`, `405`, or `406` and still indicate that the service is reachable.

## MCP AI Layer Grafana Tools

Status A.I. exposes Grafana-related tool routes through the MCP AI Layer.

Common AI-facing tool requests:

```text
get grafana health
list grafana dashboards
list grafana datasources
get grafana alert rules
```

Current MCP AI Layer tool names:

```text
get_grafana_health
list_grafana_dashboards
list_grafana_datasources
get_grafana_alert_rules
```

Example dashboard chat prompt:

```text
List Grafana dashboards.
```

Example dashboard chat prompt:

```text
List Grafana datasources.
```

Example dashboard chat prompt:

```text
Show Grafana alert rules.
```

## Test Through Dashboard Chat

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "List Grafana datasources."
        }
    )
} | ConvertTo-Json -Depth 10

Invoke-WebRequest `
    -Uri "http://127.0.0.1:5100/api/ai/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing `
    -TimeoutSec 120 |
Select-Object StatusCode, Content
```

Expected:

```text
StatusCode 200
```

## Useful Coordinator Requests

These should route toward Grafana capability:

```text
grafana health check
list grafana dashboards
list grafana datasources
list grafana alert rules
search grafana dashboards
```

For complex Grafana tools that require precise arguments, call the downstream Grafana MCP server directly from an MCP-aware client or extend the coordinator router with a specific argument mapper.

## RBAC Notes

Grafana tools require matching Grafana permissions.

Read-focused operations commonly require access to:

```text
dashboards:read
folders:read
datasources:read
alert rules read access
```

Datasource query tools may require datasource query permissions.

Write operations such as dashboard updates, folder creation, annotations, or alert changes require additional write permissions.

Use least privilege:

```text
Viewer for read-focused inspection
Editor only for write/update workflows
Admin only for setup when unavoidable
```

## Security Notes

Do not commit Grafana service account tokens.

Keep local Grafana bound to local/private interfaces unless remote access is intentionally configured.

Recommended local URL:

```text
http://127.0.0.1:3000
```

Avoid using write-capable Grafana tokens in production unless approval, auditing, and backups are in place.

## Troubleshooting

### Grafana health fails

Check:

```powershell
Invoke-RestMethod "http://127.0.0.1:3000/api/health" |
ConvertTo-Json -Depth 10
```

Check port owner:

```powershell
Get-NetTCPConnection -LocalPort 3000 -State Listen |
Select-Object LocalAddress, LocalPort, OwningProcess
```

### Dashboard shows Grafana running with unknown PID

Grafana may have been started outside the dashboard.

Stop the external Grafana process and start Grafana from the dashboard so the dashboard owns the PID and output.

### Grafana MCP cannot authenticate

Check:

```env
GRAFANA_URL=http://127.0.0.1:3000
GRAFANA_SERVICE_ACCOUNT_TOKEN=...
```

Verify the token has the needed Grafana permissions.

### Coordinator does not route Grafana requests

Check that Grafana MCP is enabled:

```env
GRAFANA_MCP_ENABLED=true
```

Check MCP Coordinator is running:

```powershell
Invoke-RestMethod "http://127.0.0.1:5100/api/system-health" |
ConvertTo-Json -Depth 10
```

Check MCP AI Layer available tools:

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/api/ai/tools" |
ConvertTo-Json -Depth 10
```
