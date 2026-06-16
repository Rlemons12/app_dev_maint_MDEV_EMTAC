param(
    [string]$ProjectRoot = (Get-Location).Path,

    [string]$McpServerPython = ".\.venv\Scripts\python.exe",
    [string]$McpServerScript = ".\mcp_server\server.py",

    [string]$AiProvider = "openai",
    [string]$OpenAiModel = "gpt-5.5",
    [string]$OpenAiApiKey = "",

    [string]$AiDefaultDatabase = "postgres",
    [bool]$RequireApprovalForDestructiveTools = $true,
    [int]$MaxToolRounds = 8,

    [switch]$InstallPythonPackages,
    [switch]$ForceEnv
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-FileUtf8NoBom {
    param(
        [string]$Path,
        [string]$Content
    )

    $Parent = Split-Path -Parent $Path

    if ($Parent -and -not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    [System.IO.File]::WriteAllText($FullPath, $Content, $Utf8NoBom)
}

function Add-LineIfMissing {
    param(
        [string]$Path,
        [string]$Line
    )

    if (-not (Test-Path $Path)) {
        Write-FileUtf8NoBom -Path $Path -Content ""
    }

    $existing = Get-Content $Path -Raw

    if ($existing -notmatch [regex]::Escape($Line)) {
        Add-Content -Path $Path -Value $Line
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PipExe = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Step "AI Gateway setup for PostgreSQL MCP"
Write-Host "Project root:       $ProjectRoot"
Write-Host "MCP Python:         $McpServerPython"
Write-Host "MCP server script:  $McpServerScript"
Write-Host "AI provider:        $AiProvider"
Write-Host "OpenAI model:       $OpenAiModel"

Set-Location $ProjectRoot

# .env.ai.example
# ------------------------------------------------------------

$envAiExample = @"
# ============================================================
# AI Gateway settings
# ============================================================

AI_PROVIDER=$AiProvider

# OpenAI provider settings.
# Put your real key in .env, not in source control.
OPENAI_API_KEY=$OpenAiApiKey
OPENAI_MODEL=$OpenAiModel

# MCP server launch command.
# These are relative to the project root by default.
MCP_SERVER_PYTHON=$McpServerPython
MCP_SERVER_SCRIPT=$McpServerScript

# AI behavior settings.
AI_DEFAULT_DATABASE=$AiDefaultDatabase
AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=$RequireApprovalForDestructiveTools
AI_MAX_TOOL_ROUNDS=$MaxToolRounds
"@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot ".env.ai.example") -Content $envAiExample

# ------------------------------------------------------------
# Optionally merge AI env values into .env
# ------------------------------------------------------------

$EnvPath = Join-Path $ProjectRoot ".env"

if ((-not (Test-Path $EnvPath)) -or $ForceEnv) {
    if (-not (Test-Path $EnvPath)) {
        Write-FileUtf8NoBom -Path $EnvPath -Content ""
    }

    Add-LineIfMissing -Path $EnvPath -Line ""
    Add-LineIfMissing -Path $EnvPath -Line "# ============================================================"
    Add-LineIfMissing -Path $EnvPath -Line "# AI Gateway settings"
    Add-LineIfMissing -Path $EnvPath -Line "# ============================================================"
    Add-LineIfMissing -Path $EnvPath -Line "AI_PROVIDER=$AiProvider"
    Add-LineIfMissing -Path $EnvPath -Line "OPENAI_API_KEY=$OpenAiApiKey"
    Add-LineIfMissing -Path $EnvPath -Line "OPENAI_MODEL=$OpenAiModel"
    Add-LineIfMissing -Path $EnvPath -Line "MCP_SERVER_PYTHON=$McpServerPython"
    Add-LineIfMissing -Path $EnvPath -Line "MCP_SERVER_SCRIPT=$McpServerScript"
    Add-LineIfMissing -Path $EnvPath -Line "AI_DEFAULT_DATABASE=$AiDefaultDatabase"
    Add-LineIfMissing -Path $EnvPath -Line "AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS=$RequireApprovalForDestructiveTools"
    Add-LineIfMissing -Path $EnvPath -Line "AI_MAX_TOOL_ROUNDS=$MaxToolRounds"

    Write-Host "Updated .env with AI Gateway settings." -ForegroundColor Green
}
else {
    Write-Host ".env exists. Leaving it unchanged. Use -ForceEnv to append AI settings." -ForegroundColor Yellow
}

# ------------------------------------------------------------
# ai_settings.py
# ------------------------------------------------------------

$aiSettingsPy = @'
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return int(value.strip())


@dataclass(frozen=True)
class AiSettings:
    project_root: Path

    ai_provider: str

    openai_api_key: str
    openai_model: str

    mcp_server_python: str
    mcp_server_script: str

    ai_default_database: str
    require_approval_for_destructive_tools: bool
    max_tool_rounds: int

    def resolved_mcp_python(self) -> str:
        value = self.mcp_server_python

        path = Path(value)

        if path.is_absolute():
            return str(path)

        return str((self.project_root / path).resolve())

    def resolved_mcp_script(self) -> str:
        value = self.mcp_server_script

        path = Path(value)

        if path.is_absolute():
            return str(path)

        return str((self.project_root / path).resolve())


def get_ai_settings() -> AiSettings:
    project_root = Path(__file__).resolve().parent

    return AiSettings(
        project_root=project_root,

        ai_provider=env_str("AI_PROVIDER", "openai"),

        openai_api_key=env_str("OPENAI_API_KEY", ""),
        openai_model=env_str("OPENAI_MODEL", "gpt-5.5"),

        mcp_server_python=env_str("MCP_SERVER_PYTHON", r".\.venv\Scripts\python.exe"),
        mcp_server_script=env_str("MCP_SERVER_SCRIPT", r".\server.py"),

        ai_default_database=env_str("AI_DEFAULT_DATABASE", "postgres"),
        require_approval_for_destructive_tools=env_bool(
            "AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS",
            True,
        ),
        max_tool_rounds=env_int("AI_MAX_TOOL_ROUNDS", 8),
    )


if __name__ == "__main__":
    settings = get_ai_settings()

    print("AI Gateway settings loaded:")
    print(f"  AI_PROVIDER={settings.ai_provider}")
    print(f"  OPENAI_MODEL={settings.openai_model}")
    print(f"  OPENAI_API_KEY_SET={bool(settings.openai_api_key)}")
    print(f"  MCP_SERVER_PYTHON={settings.resolved_mcp_python()}")
    print(f"  MCP_SERVER_SCRIPT={settings.resolved_mcp_script()}")
    print(f"  AI_DEFAULT_DATABASE={settings.ai_default_database}")
    print(
        "  AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS="
        f"{settings.require_approval_for_destructive_tools}"
    )
    print(f"  AI_MAX_TOOL_ROUNDS={settings.max_tool_rounds}")
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "ai_settings.py") -Content $aiSettingsPy

# ------------------------------------------------------------
# mcp_tool_client.py
# ------------------------------------------------------------

$mcpToolClientPy = @'
from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_settings import get_ai_settings


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if hasattr(value, "dict"):
        return _json_safe(value.dict())

    return str(value)


def _content_to_jsonable(content: Any) -> Any:
    if isinstance(content, list):
        values: list[Any] = []

        for item in content:
            if hasattr(item, "text"):
                text = item.text

                try:
                    values.append(json.loads(text))
                except Exception:
                    values.append(text)
            else:
                values.append(_json_safe(item))

        if len(values) == 1:
            return values[0]

        return values

    return _json_safe(content)


class McpToolClient:
    """
    Thin MCP stdio client for the local PostgreSQL MCP server.

    It starts server.py as a subprocess, lists tools, and can call tools by name.
    """

    def __init__(self) -> None:
        self.settings = get_ai_settings()

        self.server_params = StdioServerParameters(
            command=self.settings.resolved_mcp_python(),
            args=[self.settings.resolved_mcp_script()],
            env=None,
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                response = await session.list_tools()

                tools: list[dict[str, Any]] = []

                for tool in response.tools:
                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": _json_safe(tool.inputSchema),
                        }
                    )

                return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    name,
                    arguments=arguments or {},
                )

                if getattr(result, "isError", False):
                    raise RuntimeError(_content_to_jsonable(result.content))

                return _content_to_jsonable(result.content)
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "mcp_tool_client.py") -Content $mcpToolClientPy

# ------------------------------------------------------------
# ai_gateway.py
# ------------------------------------------------------------

$aiGatewayPy = @'
from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import OpenAI

from ai_settings import get_ai_settings
from mcp_tool_client import McpToolClient


READ_ONLY_TOOLS = {
    "postgres_health_check",
    "postgres_whoami",
    "postgres_list_databases",
    "postgres_list_schemas",
    "postgres_list_tables",
    "postgres_describe_table",
    "postgres_read_query",
}

WRITE_TOOLS = {
    "postgres_create_database",
    "postgres_create_schema",
    "postgres_create_table",
    "postgres_grant_standard_database_permissions",
    "postgres_write_execute",
    "postgres_insert_row",
    "postgres_update_rows",
}

DESTRUCTIVE_TOOLS = {
    "postgres_drop_database",
    "postgres_drop_schema",
    "postgres_drop_table",
    "postgres_delete_rows",
    "postgres_admin_execute",
}


def _as_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []

    for tool in mcp_tools:
        input_schema = tool.get("input_schema") or {}

        if not isinstance(input_schema, dict):
            input_schema = {
                "type": "object",
                "properties": {},
            }

        if input_schema.get("type") != "object":
            input_schema["type"] = "object"

        input_schema.setdefault("properties", {})

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": input_schema,
                },
            }
        )

    return openai_tools


def _approval_required(tool_name: str) -> bool:
    settings = get_ai_settings()

    if not settings.require_approval_for_destructive_tools:
        return False

    return tool_name in DESTRUCTIVE_TOOLS


def _print_tool_request(tool_name: str, arguments: dict[str, Any]) -> None:
    print("")
    print("Tool call requested:")
    print(f"  Tool: {tool_name}")
    print("  Arguments:")
    print(json.dumps(arguments, indent=2, default=str))
    print("")


def _confirm_tool_call(tool_name: str, arguments: dict[str, Any]) -> bool:
    _print_tool_request(tool_name, arguments)

    answer = input("Allow this destructive/admin tool call? Type YES to allow: ")

    return answer.strip() == "YES"


class AiMcpGateway:
    """
    OpenAI-powered AI gateway that can call the local PostgreSQL MCP server.

    Flow:
        user message
        -> model chooses MCP tool
        -> gateway calls MCP tool
        -> tool result sent back to model
        -> model returns final answer
    """

    def __init__(self) -> None:
        self.settings = get_ai_settings()

        if self.settings.ai_provider.lower() != "openai":
            raise ValueError(
                "Only AI_PROVIDER=openai is implemented in this starter gateway."
            )

        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env before using the AI gateway."
            )

        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        self.mcp_client = McpToolClient()

    async def ask(self, question: str) -> str:
        mcp_tools = await self.mcp_client.list_tools()
        openai_tools = _as_openai_tools(mcp_tools)

        system_prompt = (
            "You are an AI PostgreSQL assistant connected to a PostgreSQL MCP server. "
            "Use MCP tools for database inspection and database work. "
            "Prefer read-only tools for inspection. "
            "Do not call destructive tools unless the user clearly asks for a destructive action. "
            f"The default database is {self.settings.ai_default_database!r}. "
            "When creating objects, state exactly what database/schema/table you used."
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        for round_number in range(1, self.settings.max_tool_rounds + 1):
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Model returned invalid JSON arguments for {tool_name}: "
                        f"{tool_call.function.arguments}"
                    ) from exc

                if _approval_required(tool_name):
                    if not _confirm_tool_call(tool_name, arguments):
                        tool_result = {
                            "status": "blocked",
                            "reason": "User did not approve destructive/admin tool call.",
                            "tool_name": tool_name,
                            "arguments": arguments,
                        }
                    else:
                        tool_result = await self.mcp_client.call_tool(tool_name, arguments)
                else:
                    _print_tool_request(tool_name, arguments)
                    tool_result = await self.mcp_client.call_tool(tool_name, arguments)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, default=str),
                    }
                )

        return (
            "The AI gateway reached AI_MAX_TOOL_ROUNDS before producing a final answer. "
            "Try asking for fewer operations at once."
        )


async def ask_ai(question: str) -> str:
    gateway = AiMcpGateway()
    return await gateway.ask(question)


def ask_ai_sync(question: str) -> str:
    return asyncio.run(ask_ai(question))
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "ai_gateway.py") -Content $aiGatewayPy

# ------------------------------------------------------------
# chat_cli.py
# ------------------------------------------------------------

$chatCliPy = @'
from __future__ import annotations

from ai_gateway import ask_ai_sync


def main() -> None:
    print("PostgreSQL MCP AI Chat")
    print("Type 'exit' or 'quit' to stop.")
    print("")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            answer = ask_ai_sync(question)
            print("")
            print("AI:")
            print(answer)
            print("")
        except Exception as exc:
            print("")
            print("Error:")
            print(str(exc))
            print("")


if __name__ == "__main__":
    main()
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "chat_cli.py") -Content $chatCliPy

# ------------------------------------------------------------
# test_mcp_client.py
# ------------------------------------------------------------

$testMcpClientPy = @'
from __future__ import annotations

import asyncio
import pprint

from mcp_tool_client import McpToolClient


async def main() -> None:
    client = McpToolClient()

    print("Listing MCP tools...")
    tools = await client.list_tools()

    for tool in tools:
        print(f"- {tool['name']}")

    print("")
    print("Calling postgres_list_databases...")
    result = await client.call_tool("postgres_list_databases", {})
    pprint.pp(result)


if __name__ == "__main__":
    asyncio.run(main())
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "test_mcp_client.py") -Content $testMcpClientPy

# ------------------------------------------------------------
# README_AI_GATEWAY.md
# ------------------------------------------------------------

$readmeAi = @'
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
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "README_AI_GATEWAY.md") -Content $readmeAi

# ------------------------------------------------------------
# .gitignore additions
# ------------------------------------------------------------

$GitIgnorePath = Join-Path $ProjectRoot ".gitignore"

if (-not (Test-Path $GitIgnorePath)) {
    Write-FileUtf8NoBom -Path $GitIgnorePath -Content ""
}

Add-LineIfMissing -Path $GitIgnorePath -Line ""
Add-LineIfMissing -Path $GitIgnorePath -Line "# AI Gateway local secrets"
Add-LineIfMissing -Path $GitIgnorePath -Line ".env"
Add-LineIfMissing -Path $GitIgnorePath -Line "__pycache__/"
Add-LineIfMissing -Path $GitIgnorePath -Line "*.pyc"

# ------------------------------------------------------------
# Install packages
# ------------------------------------------------------------

if ($InstallPythonPackages) {
    Write-Step "Installing AI Gateway Python packages"

    if (-not (Test-Path $VenvPython)) {
        python -m venv .venv
    }

    & $VenvPython -m pip install --upgrade pip
    & $PipExe install -r requirements.txt
}
else {
    Write-Host ""
    Write-Host "Skipping package installation. To install packages, run:" -ForegroundColor Yellow
    Write-Host "  .\setup_ai_gateway.ps1 -InstallPythonPackages"
}

Write-Step "AI Gateway setup complete"

Write-Host "Files created or updated:"
Write-Host "  requirements.txt"
Write-Host "  .env.ai.example"
Write-Host "  ai_settings.py"
Write-Host "  mcp_tool_client.py"
Write-Host "  ai_gateway.py"
Write-Host "  chat_cli.py"
Write-Host "  test_mcp_client.py"
Write-Host "  README_AI_GATEWAY.md"

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Add OPENAI_API_KEY to .env"
Write-Host "  2. .\.venv\Scripts\activate"
Write-Host "  3. pip install -r requirements.txt"
Write-Host "  4. python -m ai_layer.cli.test_mcp_client"
Write-Host "  5. python chat_cli.py"

