# PostgreSQL MCP Server

This project exposes PostgreSQL cluster tools through a local MCP server.

It can be used in three ways:

1. **MCP stdio server** for MCP-aware clients.
2. **Local browser/API UI** at `http://127.0.0.1:8765`.
3. **AI gateway** that lets an OpenAI-compatible model call the MCP tools.

The server is designed for PostgreSQL instance / cluster-level work, not just one database.

## Project Path

```text
C:\Users\cetax\PycharmProjects\postgres_mcp
```

## Main Files

```text
mcp_server/server.py              MCP PostgreSQL tool server
mcp_server/settings.py            PostgreSQL settings
mcp_server/db.py                  PostgreSQL connection helper
mcp_server/sql_safety.py          SQL validation helpers
app/web_ui.py                     Local browser UI and HTTP API
ai_layer/mcp_client.py            Local MCP stdio client helper
ai_layer/gateway.py               AI layer that can call MCP tools
ai_layer/cli/chat_cli.py          Terminal AI chat client
ai_layer/settings.py              AI gateway settings
embeddings/embedding_settings.py  Embedding provider settings
embeddings/embedding_client.py    Embedding provider client
clients/                          HTTP client helper examples
docs/                             Markdown documentation served by /api/docs
sql/postgres_roles_setup.sql      PostgreSQL role setup SQL
scripts/setup_postgres_mcp.ps1    PostgreSQL MCP setup script
scripts/setup_ai_gateway.ps1      AI gateway setup script
huggingface_inferences/           Hugging Face Router examples/helpers
```

## Install

Create or use the local virtual environment:

```powershell
cd C:\Users\cetax\PycharmProjects\postgres_mcp
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional embedding and Hugging Face packages:

```powershell
pip install -r requirements.txt
```

## Environment

Settings are read from `.env`.

Core PostgreSQL settings:

```env
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432

POSTGRES_MAINTENANCE_DB=postgres
POSTGRES_DEFAULT_DB=postgres
POSTGRES_DEFAULT_SCHEMA=public

POSTGRES_ADMIN_USER=mcp_admin
POSTGRES_ADMIN_PASSWORD=change_this_admin_password

POSTGRES_READ_USER=mcp_read
POSTGRES_READ_PASSWORD=change_this_read_password

POSTGRES_WRITE_USER=mcp_write
POSTGRES_WRITE_PASSWORD=change_this_write_password

MCP_SERVER_NAME=PostgreSQL Cluster MCP Server
MCP_TRANSPORT=stdio
MCP_HTTP_HOST=127.0.0.1
MCP_HTTP_PORT=8000

MAX_READ_ROWS=500
```

AI and embedding settings:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-5.5

MCP_SERVER_PYTHON=.\.venv\Scripts\python.exe
MCP_SERVER_SCRIPT=.\mcp_server\server.py

AI_DEFAULT_DATABASE=postgres
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=True
AI_MAX_TOOL_ROUNDS=8

EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_TIMEOUT_SECONDS=60
HF_HUB_CACHE=C:\Users\cetax\.cache\huggingface\hub

HF_TOKEN=your_hugging_face_token_here
HF_INFERENCE_PROVIDER=hf-inference
```

Use `EMBEDDING_PROVIDER=hf-inference` when you want hosted Hugging Face inference as the default embedding provider.

## PostgreSQL Roles

This setup uses three login roles:

```text
mcp_admin
mcp_read
mcp_write
```

It also uses group roles:

```text
mcp_read_role
mcp_write_role
```

Role purpose:

- `mcp_admin`: cluster/admin operations such as listing, creating, and dropping databases.
- `mcp_read`: read-only SQL through `postgres_read_query`.
- `mcp_write`: write SQL and structured insert/update/delete tools.

Do not create PostgreSQL roles starting with `pg_`; PostgreSQL reserves that prefix.

## Apply PostgreSQL Role Setup

Run with `psql`:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" `
  -h 127.0.0.1 `
  -U postgres `
  -d postgres `
  -f .\sql\postgres_roles_setup.sql
```

Or use the setup script:

```powershell
.\scripts\setup_postgres_mcp.ps1 -ApplyDatabaseRoles
```

Install packages and apply roles:

```powershell
.\scripts\setup_postgres_mcp.ps1 -ForceEnv -InstallPythonPackages -ApplyDatabaseRoles
```

## Verify Settings

```powershell
python -m mcp_server.settings
```

## Verify PostgreSQL Connections

```powershell
python -c "from server import postgres_health_check; import pprint; pprint.pp(postgres_health_check())"
```

Expected output includes:

```text
admin_connection
read_connection
write_connection
```

## Run as MCP Server

Default `.env` transport:

```env
MCP_TRANSPORT=stdio
```

Run:

```powershell
python -m mcp_server.server
```

The terminal may look idle. That is normal for stdio MCP mode; the server is waiting for an MCP client.

## MCP Client Config

For MCP-aware clients, use:

```text
C:\Users\cetax\PycharmProjects\postgres_mcp\.venv\Scripts\python.exe
```

Argument:

```text
C:\Users\cetax\PycharmProjects\postgres_mcp\mcp_server\server.py
```

Example config:

```json
{
  "mcpServers": {
    "postgres_mcp": {
      "command": "C:\\Users\\cetax\\PycharmProjects\\postgres_mcp\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\cetax\\PycharmProjects\\postgres_mcp\\server.py"
      ]
    }
  }
}
```

## Streamable HTTP MCP Mode

To run the MCP server over HTTP instead of stdio:

```env
MCP_TRANSPORT=streamable-http
MCP_HTTP_HOST=127.0.0.1
MCP_HTTP_PORT=8000
```

Then:

```powershell
python -m mcp_server.server
```

Endpoint:

```text
http://127.0.0.1:8000/mcp
```

## Run the Browser UI / Local HTTP API

```powershell
python -m app.web_ui
```

Open:

```text
http://127.0.0.1:8765
```

Use another port if needed:

```powershell
python -m app.web_ui --port 8787
```

The API is intended for other programs running on the same computer.

## Local HTTP API

Endpoints:

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

Documentation endpoints are read-only and intended for AI clients that need to
learn how to connect to and use this server.

List documentation:

```text
http://127.0.0.1:8765/api/docs
```

Fetch all Markdown documentation as one bundle:

```text
http://127.0.0.1:8765/api/docs/bundle
```

Fetch specific documents:

```text
http://127.0.0.1:8765/api/docs/postgres-mcp
http://127.0.0.1:8765/api/docs/connect-and-use
http://127.0.0.1:8765/api/docs/ai-gateway
```

Ask the AI gateway:

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

Run read-only SQL:

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

Call a specific MCP tool:

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

For a fuller guide, see:

```text
HOW_TO_CONNECT_AND_USE_POSTGRES_MCP.md
```

## AI Gateway

Test the MCP client without AI:

```powershell
python -m ai_layer.cli.test_mcp_client
```

Start terminal chat:

```powershell
python chat_cli.py
```

Example prompts:

```text
List my PostgreSQL databases.
```

```text
Create a database called ai_demo_db, create a schema called demo, create a table called notes, and insert one row.
```

Destructive/admin tools require approval by default:

```env
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=True
```

## PostgreSQL Tools

Available PostgreSQL MCP tools include:

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

postgres_grant_standard_database_permissions

postgres_read_query
postgres_write_execute
postgres_admin_execute

postgres_insert_row
postgres_update_rows
postgres_delete_rows
```

Prefer `postgres_read_query` or `/api/read-query` for normal inspection.

## End-to-End PostgreSQL Test

Create a test database:

```powershell
python -c "from server import postgres_create_database; import pprint; pprint.pp(postgres_create_database(database_name='mcp_test_db'))"
```

Create a schema:

```powershell
python -c "from server import postgres_create_schema; import pprint; pprint.pp(postgres_create_schema(database_name='mcp_test_db', schema_name='demo'))"
```

Create a table:

```powershell
python -c "from server import postgres_create_table; import pprint; pprint.pp(postgres_create_table(database_name='mcp_test_db', schema_name='demo', table_name='test_notes', columns=[{'name':'id','type':'bigserial','primary_key':True},{'name':'note','type':'text','nullable':False},{'name':'created_at','type':'timestamp with time zone','default':'now()'}]))"
```

Insert a row:

```powershell
python -c "from server import postgres_insert_row; import pprint; pprint.pp(postgres_insert_row(database_name='mcp_test_db', schema_name='demo', table_name='test_notes', row={'note':'MCP server write test'}))"
```

Read the row:

```powershell
python -c "from server import postgres_read_query; import pprint; pprint.pp(postgres_read_query(database_name='mcp_test_db', sql='SELECT * FROM demo.test_notes'))"
```

Cleanup:

```powershell
python -c "from server import postgres_drop_database; import pprint; pprint.pp(postgres_drop_database(database_name='mcp_test_db', force=True))"
```

## Embedding Tools

Embedding tools:

```text
embedding_provider_settings
embedding_create
```

Supported providers:

```text
openai
ollama
hf-inference
sentence-transformers
local-huggingface
```

### Local Embeddings

Current local cached embedder:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Cache:

```text
C:\Users\cetax\.cache\huggingface\hub
```

Use:

```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_HUB_CACHE=C:\Users\cetax\.cache\huggingface\hub
```

Install local embedding dependencies:

```powershell
pip install -r requirements.txt
```

Tool call:

```json
{
  "texts": ["This is text to embed."],
  "provider": "sentence-transformers",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### Hugging Face Hosted Inference

Hosted Hugging Face inference uses `HF_TOKEN`.

Tools:

```text
huggingface_inference_settings
huggingface_feature_extraction
```

Environment:

```env
HF_TOKEN=your_hugging_face_token_here
HF_INFERENCE_PROVIDER=hf-inference
```

Example tool call:

```json
{
  "texts": ["This is text to embed."],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

Through HTTP API:

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

## Hugging Face Router Examples

The folder:

```text
huggingface_inferences/
```

contains examples for Hugging Face Router using the OpenAI-compatible API.

Run the image description example:

```powershell
python .\huggingface_inferences\describe_image_url.py
```

It uses:

```text
https://router.huggingface.co/v1
Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic
HF_TOKEN
```

## Security Notes

This project can create, modify, and delete PostgreSQL databases, schemas, tables, and data.

Keep the browser/API UI bound to local host unless you intentionally want network access:

```text
127.0.0.1
```

Do not expose this server publicly without authentication, logging, backups, and approval workflows.

Destructive tools include:

```text
postgres_drop_database
postgres_drop_schema
postgres_drop_table
postgres_delete_rows
postgres_admin_execute
```

For production use, consider:

- Separate development and production PostgreSQL instances
- Backups before write operations
- Audit logging
- Tool-level allowlists
- Confirmation workflows for destructive actions
- Avoiding superuser access unless absolutely required

