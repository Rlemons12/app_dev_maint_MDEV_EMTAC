from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from listed_server.mcp_coordinator.logging_config import configure_logging
from listed_server.mcp_coordinator.settings import get_settings
from listed_server.mcp_postgres.tools.postgres_tools import register_postgres_tools


def build_mcp() -> FastMCP:
    settings = get_settings()
    mcp = FastMCP("EMTAC PostgreSQL MCP Server", json_response=True)
    register_postgres_tools(mcp)
    return mcp


def main() -> None:
    configure_logging()
    settings = get_settings()
    mcp = build_mcp()
    transport = settings.mcp_transport.lower().strip()

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if transport in {"http", "streamable-http", "streamable_http"}:
        mcp.settings.host = settings.mcp_http_host
        mcp.settings.port = settings.mcp_http_port
        mcp.run(transport="streamable-http")
        return

    raise ValueError("Invalid MCP_TRANSPORT. Use 'stdio' or 'streamable-http'.")


if __name__ == "__main__":
    main()
