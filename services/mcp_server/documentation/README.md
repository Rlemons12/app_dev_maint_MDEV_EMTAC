# EMTAC MCP Coordinator

The EMTAC MCP Coordinator is the local orchestration layer for routing AI/tool requests to downstream MCP-capable services.

The current project is organized under `listed_server/`:

```text
listed_server.mcp_coordinator
listed_server.mcp_postgres
listed_server.mcp_grafana
```

The coordinator is part of the local EMTAC service stack and is normally operated from the Service Dashboard.

## Current Service Stack

The local stack includes:

```text
Service Dashboard        http://127.0.0.1:5100
GPU Service              http://127.0.0.1:5051
AI Gateway               http://127.0.0.1:9000
MCP Coordinator          http://127.0.0.1:9100/mcp
MCP AI Layer             http://127.0.0.1:9200/api/ai/chat
PostgreSQL Server        127.0.0.1:5432
Grafana                  http://127.0.0.1:3000
```

## Recommended Operation

Start the Service Dashboard first.

The dashboard registers all known services and shows their status, but services should be started manually from the dashboard.

Preferred start order:

```text
1. PostgreSQL Server
2. Grafana
3. GPU Service
4. EMTAC MCP Coordinator
5. AI Gateway
6. MCP AI Layer
```

Preferred stop order is the reverse:

```text
1. MCP AI Layer
2. AI Gateway
3. EMTAC MCP Coordinator
4. GPU Service
5. Grafana
6. PostgreSQL Server
```

## Dashboard Ownership

For best PID, uptime, and log tracking, services should be started from the dashboard.

If a service is started manually outside the dashboard, the dashboard may detect that its port is reachable and show it as running, but it may not own the process handle. In that case, PID, uptime, and output may be unknown.

To allow the dashboard to fully manage a service:

1. Stop the externally started process.
2. Start the service from the dashboard.
3. Confirm the dashboard shows PID and uptime.

## Run the Service Dashboard

From PowerShell:

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

## Run the MCP Coordinator Manually

Manual runs are useful for debugging, but the dashboard should normally start the service.

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m listed_server.mcp_coordinator.server
```

Endpoint:

```text
http://127.0.0.1:9100/mcp
```

Note: `/mcp` is a streamable HTTP MCP endpoint, not a normal REST health route. A plain browser or health-check GET may return HTTP `400`, `405`, or `406` and still prove that the MCP endpoint is alive.

## Run the MCP AI Layer Manually

Manual runs are useful for debugging AI-to-tool routing.

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"
$env:AI_REST_DEBUG = "true"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m ai_layer.ai_rest_app
```

Endpoint:

```text
http://127.0.0.1:9200/api/ai/chat
```

Health:

```text
http://127.0.0.1:9200/api/ai/health
```

## Coordinator Tools

Coordinator tools include:

```text
coordinator_health_check
coordinator_list_capabilities
coordinator_explain_route
coordinator_route_request
```

Safety behavior:

```text
coordinator_explain_route never executes tools.
coordinator_route_request(..., execute=False) never executes tools.
dangerous requests require explicit confirmation.
unknown or unsupported downstream capabilities return not_implemented.
```

## PostgreSQL Tools

PostgreSQL tools are provided by:

```text
listed_server.mcp_postgres
```

Common tools include:

```text
postgres_health_check
postgres_whoami
postgres_list_databases
postgres_list_schemas
postgres_list_tables
postgres_describe_table
postgres_read_query
```

Write/admin tools may exist in the PostgreSQL MCP package, but EMTAC production usage should remain read-only against `emtac.public`.

Do not grant MCP write access to `emtac.public`.

If writable MCP testing is needed, use a separate schema or database such as:

```text
emtac.mcp_workspace
```

## MCP AI Layer Tool Safety

The MCP AI Layer currently uses deterministic pre-routing for common operational requests.

Examples:

```text
"what is running?"
"how many tables are in emtac?"
"list postgres tables"
"describe table document"
"SELECT COUNT(*) FROM information_schema.tables"
```

The AI Layer enforces:

```text
query_postgres only accepts SELECT statements.
OpenAI-style tool payloads are disabled by default for the local gateway.
Deterministic tool results are treated as the source of truth.
```

Recommended environment:

```env
AI_ENABLE_TOOLS=true
AI_SEND_OPENAI_TOOLS=false
```

This keeps deterministic MCP routing enabled while avoiding unreliable local-model tool-call payloads.

## Grafana MCP

Grafana support is available through:

```text
listed_server.mcp_grafana
```

Grafana can also be routed through the coordinator when configured.

See:

```text
README_GRAFANA_MCP.md
```

## Configuration

Primary environment file:

```text
E:\emtac\dev_env\.env
```

Important values:

```env
MCP_TRANSPORT=streamable-http
MCP_HTTP_HOST=127.0.0.1
MCP_HTTP_PORT=9100

SERVICE_MCP_COORDINATOR_BASE_URL=http://127.0.0.1:9100
SERVICE_MCP_AI_LAYER_BASE_URL=http://127.0.0.1:9200
SERVICE_AI_GATEWAY_BASE_URL=http://127.0.0.1:9000
SERVICE_GPU_BASE_URL=http://127.0.0.1:5051

AI_ENABLE_TOOLS=true
AI_SEND_OPENAI_TOOLS=false
```

Dashboard service ownership behavior:

```env
DASHBOARD_TREAT_EXTERNAL_REACHABLE_AS_RUNNING=false
```

When this is false, services should be started from the dashboard to be considered dashboard-owned.

## Compatibility Notes

Legacy entrypoints may still exist:

```text
python server.py
python -m listed_server.mcp_postgres.server
python -m listed_server.mcp_grafana.server
```

For the integrated EMTAC stack, prefer dashboard-managed service startup.

## Troubleshooting

Check dashboard health:

```powershell
Invoke-RestMethod "http://127.0.0.1:5100/api/system-health" |
ConvertTo-Json -Depth 10
```

Check MCP AI Layer health:

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/api/ai/health" |
ConvertTo-Json -Depth 10
```

Check AI Gateway health:

```powershell
Invoke-RestMethod "http://127.0.0.1:9000/health" |
ConvertTo-Json -Depth 10
```

Check GPU health:

```powershell
Invoke-RestMethod "http://127.0.0.1:5051/health" |
ConvertTo-Json -Depth 10
```

Check which process owns a service port:

```powershell
Get-NetTCPConnection -State Listen |
Where-Object { $_.LocalPort -in 3000,5051,9000,9100,9200 } |
Select-Object LocalAddress, LocalPort, OwningProcess
```

If the dashboard says a service is reachable but PID is unknown, stop the external process and start the service from the dashboard.
