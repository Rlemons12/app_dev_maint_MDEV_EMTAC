# AI Gateway for PostgreSQL MCP

This adds an AI layer on top of the PostgreSQL MCP server.

The PostgreSQL MCP server remains the tool server. The AI Gateway is a separate client layer that:

1. Starts/connects to the local PostgreSQL MCP server over stdio.
2. Lists the MCP tools.
3. Sends the tool list to the AI model.
4. Lets the model request tool calls.
5. Executes MCP tool calls.
6. Sends tool results back to the model.
7. Returns the final answer.

## Files created

```text
ai_settings.py
mcp_tool_client.py
ai_gateway.py
chat_cli.py
test_mcp_client.py
requirements.txt
.env.ai.example
README_AI_GATEWAY.md
```

## Environment settings

Add these to `.env`:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.5

MCP_SERVER_PYTHON=.\.venv\Scripts\python.exe
MCP_SERVER_SCRIPT=.\mcp_server\server.py

AI_DEFAULT_DATABASE=postgres
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=True
AI_MAX_TOOL_ROUNDS=8

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=
EMBEDDING_TIMEOUT_SECONDS=60
OLLAMA_BASE_URL=http://127.0.0.1:11434
HF_HUB_CACHE=C:\Users\cetax\.cache\huggingface\hub
HF_TOKEN=
HF_INFERENCE_PROVIDER=hf-inference
```

## Install packages

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Test MCP client without AI

This confirms the AI gateway can start the MCP server and list/call tools.

```powershell
python -m ai_layer.cli.test_mcp_client
```

Expected behavior:

- It lists available MCP tools.
- It calls `postgres_list_databases`.

## Test AI chat

```powershell
python chat_cli.py
```

Example prompt:

```text
List my PostgreSQL databases.
```

Example prompt:

```text
Create a database called ai_demo_db, create a schema called demo, create a table called notes, and insert one row.
```

## Embedding tools

The MCP server includes provider-agnostic embedding tools:

```text
embedding_provider_settings
embedding_create
huggingface_inference_settings
huggingface_feature_extraction
```

OpenAI example:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_api_key_here
```

Ollama example:

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Local Hugging Face / sentence-transformers example:

```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_HUB_CACHE=C:\Users\cetax\.cache\huggingface\hub
```

This project currently has `sentence-transformers/all-MiniLM-L6-v2` cached locally
with 384-dimensional embeddings. Install the local embedding dependencies before
using this provider:

```powershell
pip install sentence-transformers safetensors
```

Hugging Face hosted inference example:

```env
EMBEDDING_PROVIDER=hf-inference
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=your_hugging_face_token_here
HF_INFERENCE_PROVIDER=hf-inference
```

Install the Hugging Face client:

```powershell
pip install huggingface_hub
```

You can also override `provider`, `model`, or `dimensions` on an individual
`embedding_create` tool call.

## Run the browser UI

```powershell
python -m app.web_ui
```

Open:

```text
http://127.0.0.1:8765
```

Use a different port if needed:

```powershell
python -m app.web_ui --port 8787
```

## Safety behavior

These tools require confirmation by default:

```text
postgres_drop_database
postgres_drop_schema
postgres_drop_table
postgres_delete_rows
postgres_admin_execute
```

To disable confirmation, set:

```env
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=False
```

For development, leaving confirmation enabled is recommended.

