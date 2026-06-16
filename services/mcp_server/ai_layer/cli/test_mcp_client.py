from __future__ import annotations

import asyncio
import pprint

from ai_layer.mcp_client import McpToolClient


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
