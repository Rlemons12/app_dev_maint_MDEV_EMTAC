# Grafana MCP Integration

This project uses the official Grafana MCP server as a downstream MCP process.
The coordinator can route Grafana-related requests to `mcp-grafana` when the
Grafana downstream server is enabled.

Grafana documentation:

- https://grafana.com/docs/grafana/latest/developer-resources/mcp/
- https://grafana.com/docs/grafana/latest/developer-resources/mcp/reference/mcp-tools-table/

## Requirements

- Grafana 9.0 or later.
- `mcp-grafana` installed in this project's virtual environment, or `uv` / `uvx`
  installed and available on `PATH`.
- A Grafana service account token with permissions for the tools you want.

The Grafana docs recommend this local MCP command:

```json
{
  "mcpServers": {
    "grafana": {
      "command": "uvx",
      "args": ["mcp-grafana"],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "<your service account token>"
      }
    }
  }
}
```

## Configure The Coordinator

Update `.env`:

```env
GRAFANA_MCP_ENABLED=true
GRAFANA_MCP_COMMAND=.venv\Scripts\mcp-grafana.exe
GRAFANA_MCP_ARGS=
GRAFANA_MCP_ENV_VARS=GRAFANA_URL,GRAFANA_SERVICE_ACCOUNT_TOKEN

GRAFANA_URL=http://localhost:3000
GRAFANA_SERVICE_ACCOUNT_TOKEN=your_service_account_token
GRAFANA_SERVICE_ACCOUNT_NAME=mcp-grafana-readwrite
GRAFANA_SERVICE_ACCOUNT_ROLE=Editor
GRAFANA_SERVICE_ACCOUNT_TOKEN_SECONDS_TO_LIVE=604800
```

If you prefer the official `uvx` form, use:

```env
GRAFANA_MCP_COMMAND=uvx
GRAFANA_MCP_ARGS=mcp-grafana
```

For Grafana Cloud, set `GRAFANA_URL` to your Cloud instance URL, for example:

```env
GRAFANA_URL=https://myinstance.grafana.net
```

## Run

Start the coordinator:

```powershell
python -m listed_server.mcp_coordinator.server
```

or:

```powershell
python server.py
```

Run Grafana MCP directly through the in-repo wrapper:

```powershell
python -m listed_server.mcp_grafana.server
```

This wrapper delegates to the official `mcp-grafana` executable configured by
`GRAFANA_MCP_COMMAND` and `GRAFANA_MCP_ARGS`.

## Useful Coordinator Requests

These route to Grafana:

```text
search grafana dashboards
list grafana datasources
search grafana folders
```

The coordinator currently maps simple Grafana requests to these downstream
tools:

- `search_dashboards`
- `search_folders`
- `list_datasources`
- `query_prometheus`
- `generate_deeplink`
- `get_annotations`

For complex Grafana tools that need precise arguments, call the downstream
Grafana MCP server directly from your MCP client, or extend the coordinator
router with a specific argument mapper.

## RBAC Notes

Grafana tools require the matching Grafana RBAC permissions and scopes. Common
read-focused grants include:

- `dashboards:read` with `dashboards:*`
- `folders:read` with `folders:*`
- `datasources:read` with `datasources:*`
- `datasources:query` with `datasources:uid:<uid>` for query tools

Write tools such as dashboard updates, folder creation, and annotations require
additional write permissions. Prefer least privilege for production service
accounts.

## Create Read/Write Access

For local Grafana, create or update a service account with the `Editor` role and
store a fresh token in `.env`:

```powershell
.\scripts\setup_grafana_mcp.ps1 `
  -GrafanaUrl "http://localhost:3000" `
  -ServiceAccountName "mcp-grafana-readwrite" `
  -ServiceAccountRole "Editor"
```

The script uses Grafana's service account API to create the service account,
create a token, and update:

```env
GRAFANA_SERVICE_ACCOUNT_TOKEN=...
GRAFANA_SERVICE_ACCOUNT_NAME=mcp-grafana-readwrite
GRAFANA_SERVICE_ACCOUNT_ROLE=Editor
```

For Grafana Cloud or token-based admin authentication, pass an admin bearer
token:

```powershell
.\scripts\setup_grafana_mcp.ps1 `
  -GrafanaUrl "https://myinstance.grafana.net" `
  -AdminBearerToken "admin_token_with_serviceaccount_permissions" `
  -ServiceAccountRole "Editor"
```
