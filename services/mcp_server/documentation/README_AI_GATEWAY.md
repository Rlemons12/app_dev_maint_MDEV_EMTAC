# EMTAC AI Gateway and MCP AI Layer

This document describes the current EMTAC AI-to-MCP flow.

The AI Gateway and MCP AI Layer are separate services:

```text
AI Gateway
    OpenAI-compatible local model gateway
    URL: http://127.0.0.1:9000

MCP AI Layer
    AI-to-MCP orchestration layer
    URL: http://127.0.0.1:9200

MCP Coordinator
    MCP routing and downstream tool coordination
    URL: http://127.0.0.1:9100/mcp
```

The AI Gateway does not directly own PostgreSQL MCP tool execution.

The MCP AI Layer owns tool routing, deterministic tool matching, PostgreSQL MCP calls, and final response formatting.

## Current Flow

Dashboard chat flow:

```text
Service Dashboard chat box
    ↓
POST http://127.0.0.1:5100/api/ai/chat
    ↓
Dashboard proxy forwards to MCP AI Layer
    ↓
POST http://127.0.0.1:9200/api/ai/chat
    ↓
MCP AI Layer checks for deterministic tool matches
    ↓
MCP Coordinator / downstream MCP tools execute when needed
    ↓
AI Gateway is called only when model summarization is needed
    ↓
GPU Service runs the local model
    ↓
Final answer returns to dashboard
```

Service ports:

```text
Service Dashboard        http://127.0.0.1:5100
GPU Service              http://127.0.0.1:5051
AI Gateway               http://127.0.0.1:9000
MCP Coordinator          http://127.0.0.1:9100/mcp
MCP AI Layer             http://127.0.0.1:9200/api/ai/chat
PostgreSQL Server        127.0.0.1:5432
Grafana                  http://127.0.0.1:3000
```

## Service Startup Policy

The Service Dashboard is the preferred owner for local services.

On dashboard startup, services are registered and monitored. They should not be launched automatically unless explicitly requested by the operator.

Recommended start order:

```text
1. PostgreSQL Server
2. Grafana
3. GPU Service
4. EMTAC MCP Coordinator
5. AI Gateway
6. MCP AI Layer
```

Recommended stop order:

```text
1. MCP AI Layer
2. AI Gateway
3. EMTAC MCP Coordinator
4. GPU Service
5. Grafana
6. PostgreSQL Server
```

If services are started manually outside the dashboard, the dashboard may detect that they are reachable, but it may not own their process handles. In that case, PID, uptime, and captured output may be unknown.

Recommended setting:

```env
DASHBOARD_TREAT_EXTERNAL_REACHABLE_AS_RUNNING=false
```

This keeps dashboard ownership clear.

## Important Services

### AI Gateway

The AI Gateway exposes an OpenAI-compatible endpoint.

Base URL:

```text
http://127.0.0.1:9000
```

Chat completions endpoint:

```text
http://127.0.0.1:9000/v1/chat/completions
```

Health endpoint:

```text
http://127.0.0.1:9000/health
```

PyCharm / JetBrains OpenAI-compatible provider URL:

```text
http://127.0.0.1:9000/v1
```

### MCP AI Layer

The MCP AI Layer is the dashboard-facing AI orchestration service.

Chat endpoint:

```text
http://127.0.0.1:9200/api/ai/chat
```

Health endpoint:

```text
http://127.0.0.1:9200/api/ai/health
```

Available tools endpoint:

```text
http://127.0.0.1:9200/api/ai/tools
```

### MCP Coordinator

The MCP Coordinator exposes streamable HTTP MCP at:

```text
http://127.0.0.1:9100/mcp
```

Note: `/mcp` is not a normal REST health endpoint. A plain GET may return HTTP `400`, `405`, or `406` and still prove the service is alive.

## Environment Settings

Primary environment file:

```text
E:\emtac\dev_env\.env
```

Recommended AI settings:

```env
AI_LOCAL_ENDPOINT=http://127.0.0.1:9000/v1/chat/completions
AI_LOCAL_MODEL=emtac-gpu-qwen

AI_ENABLE_TOOLS=true
AI_SEND_OPENAI_TOOLS=false

AI_MAX_TOKENS=1024
AI_TEMPERATURE=0.3
AI_TOP_P=0.95
AI_REQUEST_TIMEOUT_SECONDS=120
AI_MAX_HISTORY_MESSAGES=12
AI_MAX_TOOL_ROUNDS=5
```

Important distinction:

```text
AI_ENABLE_TOOLS=true
```

enables deterministic MCP routing inside the MCP AI Layer.

```text
AI_SEND_OPENAI_TOOLS=false
```

prevents the MCP AI Layer from sending OpenAI-style `tools` payloads to the local model gateway.

This is the recommended setup because the local model gateway may not reliably support OpenAI tool-call payloads.

## Deterministic Tool Routing

The MCP AI Layer performs deterministic pre-routing for common operational requests.

Examples:

```text
what is running?
how many tables are in the emtac database?
list postgres tables
show postgres tables
describe table document
SELECT COUNT(*) FROM information_schema.tables
```

When a deterministic tool result exists, the AI Layer treats it as the source of truth.

For simple deterministic results, such as PostgreSQL table counts, the AI Layer can return a direct answer without calling the model.

Example:

```text
ok PostgreSQL public schema contains 69 base tables.
```

## PostgreSQL MCP Safety

PostgreSQL MCP tools are provided by:

```text
listed_server.mcp_postgres
```

Normal inspection should use read-only tools:

```text
postgres_health_check
postgres_whoami
postgres_list_databases
postgres_list_schemas
postgres_list_tables
postgres_describe_table
postgres_read_query
```

The MCP AI Layer enforces:

```text
query_postgres only accepts SELECT statements.
```

Do not grant MCP write access to `emtac.public`.

If writable MCP testing is needed, use a separate schema or database, such as:

```text
emtac.mcp_workspace
```

Write/admin tools may exist in the PostgreSQL MCP package, but production EMTAC access should remain read-only against `emtac.public`.

## Run From Dashboard

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

Start services from the dashboard in this order:

```text
PostgreSQL Server
Grafana
GPU Service
EMTAC MCP Coordinator
AI Gateway
MCP AI Layer
```

## Run AI Gateway Manually

Manual startup is useful for debugging.

```powershell
Set-Location "E:\emtac\services\ai_gateway"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" run.py
```

Check health:

```powershell
Invoke-RestMethod "http://127.0.0.1:9000/health" |
ConvertTo-Json -Depth 10
```

Test chat completions directly:

```powershell
$body = @{
    model = "emtac-gpu-qwen"
    messages = @(
        @{
            role = "user"
            content = "Say OK only."
        }
    )
    max_tokens = 64
    temperature = 0.1
    stream = $false
} | ConvertTo-Json -Depth 10

Invoke-WebRequest `
    -Uri "http://127.0.0.1:9000/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing `
    -TimeoutSec 60 |
Select-Object StatusCode, Content
```

## Run MCP AI Layer Manually

Manual startup is useful for debugging AI-to-MCP routing.

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"
$env:AI_REST_DEBUG = "true"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m ai_layer.ai_rest_app
```

Check health:

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/api/ai/health" |
ConvertTo-Json -Depth 10
```

Expected important values:

```json
{
  "status": "ok",
  "model": "emtac-gpu-qwen",
  "tools_enabled": true,
  "openai_tool_payload_enabled": false
}
```

## Test Dashboard Chat

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "Say OK only."
        }
    )
} | ConvertTo-Json -Depth 10

Invoke-WebRequest `
    -Uri "http://127.0.0.1:5100/api/ai/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing `
    -TimeoutSec 60 |
Select-Object StatusCode, Content
```

Expected:

```text
StatusCode 200
```

## Test PostgreSQL Tool Routing

Table count:

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "How many tables are in the emtac database?"
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

Expected response includes:

```text
PostgreSQL public schema contains 69 base tables.
```

List tables:

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "Use the list_postgres_tables tool and show me the first 10 table names."
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

## Troubleshooting

### Dashboard returns HTTP 500

Check MCP AI Layer directly:

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/api/ai/health" |
ConvertTo-Json -Depth 10
```

If health fails, start MCP AI Layer from the dashboard or manually.

### AI Layer health works but chat fails

Test AI Gateway directly:

```powershell
Invoke-RestMethod "http://127.0.0.1:9000/health" |
ConvertTo-Json -Depth 10
```

Then test direct chat completions on:

```text
http://127.0.0.1:9000/v1/chat/completions
```

### Dashboard shows PID or uptime as unknown

The service may have been started outside the dashboard.

Stop the external process and start the service from the dashboard.

Check port ownership:

```powershell
Get-NetTCPConnection -State Listen |
Where-Object { $_.LocalPort -in 3000,5051,9000,9100,9200 } |
Select-Object LocalAddress, LocalPort, OwningProcess
```

### MCP Coordinator returns HTTP 400, 405, or 406

This can be normal for `/mcp`.

`/mcp` is a streamable HTTP MCP endpoint and expects MCP-style request flow. Plain browser or health-check GET requests may return HTTP `400`, `405`, or `406`.
