# How to Connect and Use `postgres_mcp`

This project exposes PostgreSQL tools in two useful ways:

1. As an MCP server over stdio, for MCP-aware clients.
2. As a local HTTP web UI/API at `http://127.0.0.1:8765`, for browser use or another local Python project.

Use the HTTP API if another normal Python program needs to call this project.
Use the MCP server directly if the other program is an MCP client.

## Project Path

```text
C:\Users\cetax\PycharmProjects\postgres_mcp
```

## Start the Local Web UI/API

From this project directory:

```powershell
.\.venv\Scripts\python.exe -m app.web_ui --host 127.0.0.1 --port 8765
```

Open the UI in a browser:

```text
http://127.0.0.1:8765
```

Another program on the same computer should also use:

```text
http://127.0.0.1:8765
```

Do not use `0.0.0.0` unless you intentionally want other computers on the network to reach it.

## HTTP API Endpoints

```text
GET  /api/settings
GET  /api/tools
GET  /api/docs
GET  /api/docs/bundle
GET  /api/docs/{doc_id}
POST /api/ask
POST /api/read-query
POST /api/tool
```

Use `/api/docs/bundle` when an AI client needs one endpoint that explains how to
connect to and use this server.

Full local URLs:

```text
GET  http://127.0.0.1:8765/api/settings
GET  http://127.0.0.1:8765/api/tools
GET  http://127.0.0.1:8765/api/docs
GET  http://127.0.0.1:8765/api/docs/bundle
GET  http://127.0.0.1:8765/api/docs/postgres-mcp
GET  http://127.0.0.1:8765/api/docs/connect-and-use
GET  http://127.0.0.1:8765/api/docs/ai-gateway
POST http://127.0.0.1:8765/api/ask
POST http://127.0.0.1:8765/api/read-query
POST http://127.0.0.1:8765/api/tool
```

## Call `/api/ask`

This is the easiest endpoint for another program to use.

Request:

```json
{
  "question": "List my PostgreSQL databases."
}
```

Example Python:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8765/api/ask",
    json={"question": "List my PostgreSQL databases."},
    timeout=60,
)
response.raise_for_status()

print(response.json()["answer"])
```

Install `requests` in the other project if needed:

```powershell
pip install requests
```

## Call a Read-Only SQL Query

Use `/api/read-query` for read-only SQL.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8765/api/read-query",
    json={
        "database_name": "postgres",
        "sql": "SELECT current_database(), current_user;",
    },
    timeout=60,
)
response.raise_for_status()

print(response.json()["result"])
```

The server enforces read-only behavior for this endpoint through the MCP read query tool.

## Call a Specific MCP Tool

Use `/api/tool` when you know the exact MCP tool name and arguments.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8765/api/tool",
    json={
        "name": "postgres_list_databases",
        "arguments": {
            "include_templates": False
        },
    },
    timeout=60,
)
response.raise_for_status()

print(response.json()["result"])
```

List available tools:

```python
import requests

tools = requests.get("http://127.0.0.1:8765/api/tools", timeout=30)
tools.raise_for_status()

for tool in tools.json()["tools"]:
    print(tool["name"])
```

## Reusable Client for Another Python Project

Create a small helper in the other project:

```python
import os
from typing import Any

import requests


class PostgresMcpClient:
    def __init__(self, base_url: str | None = None, timeout: int = 60) -> None:
        self.base_url = (
            base_url
            or os.getenv("POSTGRES_MCP_UI_BASE_URL")
            or "http://127.0.0.1:8765"
        ).rstrip("/")
        self.timeout = timeout

    def settings(self) -> dict[str, Any]:
        return self._get("/api/settings")

    def tools(self) -> list[dict[str, Any]]:
        return self._get("/api/tools")["tools"]

    def ask(self, question: str, allow_destructive: bool = False) -> str:
        return self._post(
            "/api/ask",
            {
                "question": question,
                "allow_destructive": allow_destructive,
            },
        )["answer"]

    def read_query(self, sql: str, database_name: str | None = None) -> Any:
        payload: dict[str, Any] = {"sql": sql}

        if database_name:
            payload["database_name"] = database_name

        return self._post("/api/read-query", payload)["result"]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._post(
            "/api/tool",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )["result"]

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
```

Use it:

```python
client = PostgresMcpClient()

print(client.ask("List my PostgreSQL databases."))
print(client.read_query("SELECT current_database();", database_name="postgres"))
```

Optional environment variable in the other project:

```powershell
$env:POSTGRES_MCP_UI_BASE_URL = "http://127.0.0.1:8765"
```

## Connect as an MCP Server

For MCP-aware clients, use this command:

```text
C:\Users\cetax\PycharmProjects\postgres_mcp\.venv\Scripts\python.exe
```

Arguments:

```text
C:\Users\cetax\PycharmProjects\postgres_mcp\mcp_server\server.py
```

Example MCP client configuration shape:

```json
{
  "mcpServers": {
    "postgres_mcp": {
      "command": "C:\\Users\\cetax\\PycharmProjects\\postgres_mcp\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\cetax\\PycharmProjects\\postgres_mcp\\server.py"
      ],
      "env": {
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DEFAULT_DB": "postgres"
      }
    }
  }
}
```

Most local setup values should live in this project's `.env`.

## Common PostgreSQL Tools

Useful MCP tool names:

```text
postgres_health_check
postgres_whoami
postgres_list_databases
postgres_create_database
postgres_drop_database
postgres_list_schemas
postgres_create_schema
postgres_drop_schema
postgres_list_tables
postgres_describe_table
postgres_create_table
postgres_drop_table
postgres_read_query
postgres_write_execute
postgres_admin_execute
postgres_insert_row
postgres_update_rows
postgres_delete_rows
```

Prefer `postgres_read_query` or `/api/read-query` for normal inspection.
Use destructive tools only when you intend to change or delete data.

## Embedding Tools

The MCP server also exposes embedding tools:

```text
embedding_provider_settings
embedding_create
huggingface_inference_settings
huggingface_feature_extraction
```

Current local Hugging Face cache:

```text
C:\Users\cetax\.cache\huggingface\hub
```

Cached local text embedder:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Use these `.env` values for the local cached model:

```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_HUB_CACHE=C:\Users\cetax\.cache\huggingface\hub
```

Install local embedding dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Example embedding tool call over HTTP:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8765/api/tool",
    json={
        "name": "embedding_create",
        "arguments": {
            "texts": ["This is text to embed."],
            "provider": "sentence-transformers",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    timeout=120,
)
response.raise_for_status()

result = response.json()["result"]
print(result["dimensions"])
print(result["embeddings"][0][:5])
```

Use Hugging Face hosted inference with `HF_TOKEN`:

```env
EMBEDDING_PROVIDER=hf-inference
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=your_hugging_face_token_here
HF_INFERENCE_PROVIDER=hf-inference
```

Install the Hugging Face client:

```powershell
.\.venv\Scripts\pip.exe install huggingface_hub
```

Example hosted Hugging Face tool call:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8765/api/tool",
    json={
        "name": "huggingface_feature_extraction",
        "arguments": {
            "texts": ["This is text to embed."],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    timeout=120,
)
response.raise_for_status()

print(response.json()["result"]["dimensions"])
```

## Safety Notes

The local web UI/API has no built-in authentication. Keep it bound to:

```text
127.0.0.1
```

That keeps access limited to programs running on this computer.

Only bind to `0.0.0.0` if you understand that other machines on your network may be able to reach the API.

Destructive tools include:

```text
postgres_drop_database
postgres_drop_schema
postgres_drop_table
postgres_delete_rows
postgres_admin_execute
```

The AI gateway can require approval for destructive tools with:

```env
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=True
```

## Quick Troubleshooting

Check that the web UI/API is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/settings
```

Check available tools:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/tools
```

If another project cannot connect, confirm:

- `python -m app.web_ui` is still running.
- The URL is exactly `http://127.0.0.1:8765`.
- The other project is running on the same computer.
- No firewall or security tool is blocking local HTTP requests.

