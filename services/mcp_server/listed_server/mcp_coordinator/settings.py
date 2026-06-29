from __future__ import annotations

"""
Settings for the EMTAC MCP Coordinator.

This coordinator must load the shared EMTAC ecosystem environment file:

    E:\\emtac\\dev_env\\.env

Do not rely on the current working directory for .env loading. The MCP server
may be started from the service dashboard, PowerShell, PyCharm, Task Scheduler,
or another process, so the environment path must be explicit.
"""

from dataclasses import dataclass
import os
from pathlib import Path


# ---------------------------------------------------------------------
# Shared environment file
# ---------------------------------------------------------------------

ENV_PATH = Path(r"E:\emtac\dev_env\.env")


def _load_env_file(path: Path = ENV_PATH) -> None:
    """
    Lightweight .env loader that does not depend on python-dotenv behavior.

    Rules:
    - Ignores blank lines and comments.
    - Supports KEY=value.
    - Strips matching single or double quotes around values.
    - Always overrides existing process environment values because this file
      is the EMTAC source of truth.
    """

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2:
            if value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

        os.environ[key] = value


_load_env_file()


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    lowered = value.strip().lower()

    if lowered in {"1", "true", "yes", "y", "on"}:
        return True

    if lowered in {"0", "false", "no", "n", "off"}:
        return False

    return default


def env_list(name: str) -> list[str]:
    value = env_str(name, "")

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def first_env_str(names: list[str], default: str = "") -> str:
    """
    Return the first non-empty env value from a list of possible variable names.

    This allows the coordinator to use new MCP-specific names while still
    falling back to existing EMTAC environment values.
    """

    for name in names:
        value = env_str(name, "")
        if value:
            return value

    return default


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DownstreamServerConfig:
    enabled: bool
    command: str
    args: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class CoordinatorSettings:
    env_path: Path

    coordinator_mcp_server_name: str
    mcp_transport: str
    mcp_http_host: str
    mcp_http_port: int

    postgres_mcp_enabled: bool
    postgres_host: str
    postgres_port: int
    postgres_maintenance_db: str
    postgres_default_db: str
    postgres_db: str
    postgres_default_schema: str

    postgres_admin_user: str
    postgres_admin_password: str
    postgres_read_user: str
    postgres_read_password: str
    postgres_write_user: str
    postgres_write_password: str

    max_read_rows: int

    filesystem_mcp: DownstreamServerConfig
    git_mcp: DownstreamServerConfig
    github_mcp: DownstreamServerConfig
    grafana_mcp: DownstreamServerConfig
    browser_mcp: DownstreamServerConfig
    emtac_api_mcp: DownstreamServerConfig
    memory_mcp: DownstreamServerConfig


# ---------------------------------------------------------------------
# Downstream MCP helpers
# ---------------------------------------------------------------------

def _downstream_env(prefix: str) -> dict[str, str]:
    """
    Build environment values for a downstream MCP server.

    Example:

        GITHUB_MCP_ENV_VARS=GITHUB_TOKEN,GITHUB_OWNER
        GITHUB_TOKEN=...
        GITHUB_OWNER=Rlemons12

    Result:

        {
            "GITHUB_TOKEN": "...",
            "GITHUB_OWNER": "Rlemons12"
        }
    """

    names = env_list(f"{prefix}_ENV_VARS")
    resolved: dict[str, str] = {}

    for name in names:
        value = env_str(name, "")
        if value:
            resolved[name] = value

    return resolved


def _downstream(prefix: str) -> DownstreamServerConfig:
    return DownstreamServerConfig(
        enabled=env_bool(f"{prefix}_ENABLED", False),
        command=env_str(f"{prefix}_COMMAND", ""),
        args=env_list(f"{prefix}_ARGS"),
        env=_downstream_env(prefix),
    )


# ---------------------------------------------------------------------
# Public settings
# ---------------------------------------------------------------------

def get_settings() -> CoordinatorSettings:
    postgres_default_db = first_env_str(
        [
            "POSTGRES_DEFAULT_DB",
            "POSTGRES_DB",
        ],
        "emtac",
    )

    postgres_db = first_env_str(
        [
            "POSTGRES_DB",
            "POSTGRES_DEFAULT_DB",
        ],
        postgres_default_db,
    )

    postgres_admin_user = first_env_str(
        [
            "POSTGRES_ADMIN_USER",
            "POSTGRES_USER",
        ],
        "postgres",
    )

    postgres_admin_password = first_env_str(
        [
            "POSTGRES_ADMIN_PASSWORD",
            "POSTGRES_PASSWORD",
        ],
        "",
    )

    postgres_read_user = first_env_str(
        [
            "POSTGRES_READ_USER",
            "GRAFANA_POSTGRES_USER",
            "POSTGRES_USER_READ_ONLY",
            "POSTGRES_USER",
        ],
        postgres_admin_user,
    )

    postgres_read_password = first_env_str(
        [
            "POSTGRES_READ_PASSWORD",
            "GRAFANA_POSTGRES_PASSWORD",
            "POSTGRES_PASSWORD_READ_ONLY",
            "POSTGRES_PASSWORD",
        ],
        postgres_admin_password,
    )

    postgres_write_user = first_env_str(
        [
            "POSTGRES_WRITE_USER",
            "POSTGRES_USER",
        ],
        postgres_admin_user,
    )

    postgres_write_password = first_env_str(
        [
            "POSTGRES_WRITE_PASSWORD",
            "POSTGRES_PASSWORD",
        ],
        postgres_admin_password,
    )

    return CoordinatorSettings(
        env_path=ENV_PATH,

        coordinator_mcp_server_name=first_env_str(
            [
                "COORDINATOR_MCP_SERVER_NAME",
                "MCP_SERVER_NAME",
            ],
            "EMTAC MCP Coordinator",
        ),

        # Default to dashboard/service-friendly HTTP mode.
        # Use MCP_TRANSPORT=stdio only when launching directly from an MCP client
        # that expects stdio.
        mcp_transport=env_str("MCP_TRANSPORT", "streamable-http"),
        mcp_http_host=env_str("MCP_HTTP_HOST", "127.0.0.1"),
        mcp_http_port=env_int("MCP_HTTP_PORT", 9100),

        postgres_mcp_enabled=env_bool("POSTGRES_MCP_ENABLED", True),
        postgres_host=env_str("POSTGRES_HOST", "127.0.0.1"),
        postgres_port=env_int("POSTGRES_PORT", 5432),
        postgres_maintenance_db=env_str("POSTGRES_MAINTENANCE_DB", "postgres"),
        postgres_default_db=postgres_default_db,
        postgres_db=postgres_db,
        postgres_default_schema=env_str("POSTGRES_DEFAULT_SCHEMA", "public"),

        postgres_admin_user=postgres_admin_user,
        postgres_admin_password=postgres_admin_password,
        postgres_read_user=postgres_read_user,
        postgres_read_password=postgres_read_password,
        postgres_write_user=postgres_write_user,
        postgres_write_password=postgres_write_password,

        max_read_rows=env_int("MAX_READ_ROWS", 500),

        filesystem_mcp=_downstream("FILESYSTEM_MCP"),
        git_mcp=_downstream("GIT_MCP"),
        github_mcp=_downstream("GITHUB_MCP"),
        grafana_mcp=_downstream("GRAFANA_MCP"),
        browser_mcp=_downstream("BROWSER_MCP"),
        emtac_api_mcp=_downstream("EMTAC_API_MCP"),
        memory_mcp=_downstream("MEMORY_MCP"),
    )