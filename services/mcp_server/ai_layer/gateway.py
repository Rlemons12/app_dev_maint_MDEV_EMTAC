from __future__ import annotations

import asyncio
import json
from typing import Any

from ai_layer.mcp_client import McpToolClient
from ai_layer.providers.model_provider import create_provider_config
from ai_layer.settings import get_ai_settings


READ_ONLY_TOOLS = {
    "postgres_health_check",
    "postgres_whoami",
    "postgres_list_databases",
    "postgres_list_schemas",
    "postgres_list_tables",
    "postgres_describe_table",
    "postgres_read_query",
    "embedding_provider_settings",
    "embedding_create",
    "huggingface_inference_settings",
    "huggingface_feature_extraction",
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

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = get_ai_settings()
        provider_config = create_provider_config(
            self.settings,
            provider=provider,
            model=model,
        )
        self.provider = provider_config.provider_name
        self.model = provider_config.model
        self.chat_completions = provider_config.chat_completions

        self.mcp_client = McpToolClient()

    async def ask(
        self,
        question: str,
        approve_destructive_tools: bool | None = None,
        print_tool_calls: bool = True,
        use_tools: bool = True,
    ) -> str:
        """
        Ask the model a question and let it call MCP tools.

        approve_destructive_tools controls confirmation behavior:
            None  -> prompt in the terminal when a destructive/admin tool is requested
            False -> block destructive/admin tools without prompting
            True  -> allow destructive/admin tools for this request
        """
        openai_tools: list[dict[str, Any]] = []
        if use_tools:
            mcp_tools = await self.mcp_client.list_tools()
            openai_tools = _as_openai_tools(mcp_tools)

        system_prompt = (
            "You are an AI assistant connected to an MCP coordinator. "
            "Use MCP tools for PostgreSQL, Grafana, and other routed capabilities. "
            "Preserve the user's domain wording when routing; do not reinterpret Grafana dashboards as PostgreSQL database objects. "
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
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if use_tools:
                request_kwargs["tools"] = openai_tools
                request_kwargs["tool_choice"] = "auto"

            response = self.chat_completions.create(**request_kwargs)

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
                    if approve_destructive_tools is None:
                        approved = _confirm_tool_call(tool_name, arguments)
                    else:
                        approved = approve_destructive_tools
                        if print_tool_calls:
                            _print_tool_request(tool_name, arguments)

                    if not approved:
                        tool_result = {
                            "status": "blocked",
                            "reason": "User did not approve destructive/admin tool call.",
                            "tool_name": tool_name,
                            "arguments": arguments,
                        }
                    else:
                        tool_result = await self.mcp_client.call_tool(tool_name, arguments)
                else:
                    if print_tool_calls:
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


async def ask_ai(
    question: str,
    approve_destructive_tools: bool | None = None,
    print_tool_calls: bool = True,
    provider: str | None = None,
    model: str | None = None,
    use_tools: bool = True,
) -> str:
    gateway = AiMcpGateway(provider=provider, model=model)
    return await gateway.ask(
        question,
        approve_destructive_tools=approve_destructive_tools,
        print_tool_calls=print_tool_calls,
        use_tools=use_tools,
    )


def ask_ai_sync(
    question: str,
    approve_destructive_tools: bool | None = None,
    print_tool_calls: bool = True,
    provider: str | None = None,
    model: str | None = None,
    use_tools: bool = True,
) -> str:
    return asyncio.run(
        ask_ai(
            question,
            approve_destructive_tools=approve_destructive_tools,
            print_tool_calls=print_tool_calls,
            provider=provider,
            model=model,
            use_tools=use_tools,
        )
    )
