# How to Connect and Use EMTAC PostgreSQL MCP

This document explains how to connect to and use the PostgreSQL MCP tooling in the current EMTAC local service stack.

The current integrated project path is:

```text
E:\emtac\services\mcp_server
```

The PostgreSQL MCP tools are provided by:

```text
listed_server.mcp_postgres
```

They are normally accessed through:

```text
Service Dashboard
    ↓
MCP AI Layer
    ↓
MCP Coordinator
    ↓
PostgreSQL MCP tools
```

## Current Service URLs

```text
Service Dashboard        http://127.0.0.1:5100
GPU Service              http://127.0.0.1:5051
AI Gateway               http://127.0.0.1:9000
MCP Coordinator          http://127.0.0.1:9100/mcp
MCP AI Layer             http://127.0.0.1:9200/api/ai/chat
PostgreSQL Server        127.0.0.1:5432
Grafana                  http://127.0.0.1:3000
```

## Recommended Use

The recommended way to use PostgreSQL MCP is through the Service Dashboard chat.

Example prompts:

```text
How many tables are in the emtac database?
```

```text
Use the list_postgres_tables tool and show me the first 10 table names.
```

```text
Describe the document table.
```

```text
SELECT COUNT(*) AS table_count
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE';
```

The MCP AI Layer detects common PostgreSQL requests and routes them to the PostgreSQL MCP tools.

## Service Startup

Services should be started from the Service Dashboard so the dashboard owns the PID, uptime, and output.

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

The dashboard should register and monitor services on startup, but services should not be automatically launched unless the operator explicitly starts them.

Recommended setting:

```env
DASHBOARD_TREAT_EXTERNAL_REACHABLE_AS_RUNNING=false
```

## Safety Policy

The EMTAC PostgreSQL MCP setup should remain read-only against:

```text
emtac.public
```

Do not grant MCP write access to `emtac.public`.

Normal inspection should use read-only tools only.

The MCP AI Layer enforces:

```text
query_postgres only accepts SELECT statements.
```

If writable MCP testing is needed, use a separate test schema or database, such as:

```text
emtac.mcp_workspace
```

Do not test writes against production EMTAC tables.

## Primary Use Through Dashboard Chat

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

Start required services from the dashboard.

Then ask Status A.I.:

```text
How many tables are in the emtac database?
```

Expected response:

```text
ok PostgreSQL public schema contains 69 base tables.
```

## Direct Test Through Dashboard API

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

## Direct Test Through MCP AI Layer

The MCP AI Layer can be called directly without going through the dashboard proxy.

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
    -Uri "http://127.0.0.1:9200/api/ai/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing `
    -TimeoutSec 120 |
Select-Object StatusCode, Content
```

Health:

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/api/ai/health" |
ConvertTo-Json -Depth 10
```

Expected important values:

```json
{
  "status": "ok",
  "tools_enabled": true,
  "openai_tool_payload_enabled": false
}
```

## MCP AI Layer PostgreSQL Behavior

The MCP AI Layer includes deterministic routing for common PostgreSQL requests.

Examples:

```text
how many tables are in emtac?
number of tables in postgres
list postgres tables
show postgres tables
describe table document
SELECT current_database(), current_user;
```

Common routed tool names:

```text
query_postgres
list_postgres_tables
describe_postgres_table
get_postgres_insights
```

`query_postgres` only accepts SQL beginning with:

```sql
SELECT
```

Non-SELECT statements are rejected before execution.

## Direct PostgreSQL MCP Server Run

Manual runs are useful for debugging or for MCP-aware clients.

From the MCP server project:

```powershell
Set-Location "E:\emtac\services\mcp_server"

$env:EMTAC_ENV_PATH = "E:\emtac\dev_env\.env"

& "E:\emtac\services\.venv_services\Scripts\python.exe" -m listed_server.mcp_postgres.server
```

The terminal may appear idle if running in stdio MCP mode. That is normal. It is waiting for an MCP client.

## MCP Client Configuration

For MCP-aware clients, use:

```text
E:\emtac\services\.venv_services\Scripts\python.exe
```

Arguments:

```text
-m listed_server.mcp_postgres.server
```

Example shape:

```json
{
  "mcpServers": {
    "emtac_postgres": {
      "command": "E:\\emtac\\services\\.venv_services\\Scripts\\python.exe",
      "args": [
        "-m",
        "listed_server.mcp_postgres.server"
      ],
      "env": {
        "EMTAC_ENV_PATH": "E:\\emtac\\dev_env\\.env"
      }
    }
  }
}
```

## Direct Coordinator Run

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

Note: `/mcp` is a streamable HTTP MCP endpoint. Plain GET requests may return HTTP `400`, `405`, or `406` and still indicate the service is reachable.

## Common PostgreSQL MCP Tools

Read-focused tools:

```text
postgres_health_check
postgres_whoami
postgres_list_databases
postgres_list_schemas
postgres_list_tables
postgres_describe_table
postgres_read_query
```

Write/admin tools may exist in the package, but should not be used against `emtac.public`:

```text
postgres_create_database
postgres_drop_database
postgres_create_schema
postgres_drop_schema
postgres_create_table
postgres_drop_table
postgres_write_execute
postgres_admin_execute
postgres_insert_row
postgres_update_rows
postgres_delete_rows
```

Use write/admin tools only in approved test schemas or separate databases.

## Read-Only SQL Examples

Count public base tables:

```sql
SELECT COUNT(*) AS table_count
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE';
```

Show current database and user:

```sql
SELECT current_database(), current_user;
```

List public tables:

```sql
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name
LIMIT 25;
```

Check PostgreSQL version:

```sql
SELECT version();
```

## Legacy Standalone Browser UI

Older standalone documentation referenced a local browser/API UI at:

```text
http://127.0.0.1:8765
```

That was for the old standalone `postgres_mcp` project.

The current EMTAC integrated stack should use:

```text
Service Dashboard        http://127.0.0.1:5100
MCP AI Layer             http://127.0.0.1:9200
MCP Coordinator          http://127.0.0.1:9100/mcp
```

Do not use the old `8765` UI unless you intentionally run the legacy standalone project.

## Troubleshooting

### Dashboard chat fails

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

### PostgreSQL tool request fails

Check service health:

```powershell
Invoke-RestMethod "http://127.0.0.1:5100/api/system-health" |
ConvertTo-Json -Depth 10
```

Check that PostgreSQL is running:

```powershell
& "E:\emtac\databases\postgresql\pgsql\bin\pg_ctl.exe" `
  -D "E:\emtac\databases\postgresql\pgsql\data" `
  status
```

### Dashboard shows PID or uptime as unknown

The service may have been started outside the dashboard.

Stop the external service process and start it from the dashboard.

Check port ownership:

```powershell
Get-NetTCPConnection -State Listen |
Where-Object { $_.LocalPort -in 3000,5051,9000,9100,9200 } |
Select-Object LocalAddress, LocalPort, OwningProcess
```

### Write query is rejected

This is expected.

The dashboard AI path allows only read-only PostgreSQL queries through `query_postgres`.

Use only `SELECT` queries for normal inspection.

For writable tests, use a separate approved schema or database, not `emtac.public`.

## Security Notes

Keep local service endpoints bound to localhost unless network access is intentionally required.

Recommended local bind:

```text
127.0.0.1
```

Do not expose PostgreSQL MCP, MCP AI Layer, or the Service Dashboard publicly without authentication, authorization, logging, and backups.

Never commit real database passwords, Grafana service account tokens, or API keys.
