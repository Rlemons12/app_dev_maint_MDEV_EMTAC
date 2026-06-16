from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


# IMPORTANT:
# override=True makes values in .env override existing Windows/system
# environment variables. This matters because your Python output was still
# showing localhost even though .env had 127.0.0.1.
load_dotenv(override=True)


def env_str(name: str, default: str = "") -> str:
    """
    Read a string environment variable.

    Empty strings are treated as missing values.
    Leading/trailing whitespace is stripped so values like:
        POSTGRES_WRITE_PASSWORD=McpWrite123!
    and accidental trailing spaces do not break authentication.
    """
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return value


def env_int(name: str, default: int) -> int:
    """
    Read an integer environment variable.

    Empty strings are treated as missing values.
    """
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return int(value)


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------
    # PostgreSQL server / cluster connection
    # ------------------------------------------------------------
    postgres_host: str
    postgres_port: int

    # Maintenance database is used for cluster-level actions:
    #   - list databases
    #   - create database
    #   - drop database
    postgres_maintenance_db: str

    # Default target database for tools when database_name is not provided.
    postgres_default_db: str

    # Backward-compatible alias used by older code.
    postgres_db: str

    postgres_default_schema: str

    # ------------------------------------------------------------
    # MCP admin role
    # ------------------------------------------------------------
    postgres_admin_user: str
    postgres_admin_password: str

    # ------------------------------------------------------------
    # MCP read-only role
    # ------------------------------------------------------------
    postgres_read_user: str
    postgres_read_password: str

    # ------------------------------------------------------------
    # MCP read/write role
    # ------------------------------------------------------------
    postgres_write_user: str
    postgres_write_password: str

    # ------------------------------------------------------------
    # MCP server settings
    # ------------------------------------------------------------
    mcp_server_name: str
    mcp_transport: str
    mcp_http_host: str
    mcp_http_port: int

    # ------------------------------------------------------------
    # Query limits
    # ------------------------------------------------------------
    max_read_rows: int

    def make_dsn(
        self,
        database_name: str,
        username: str,
        password: str,
    ) -> str:
        """
        Build a psycopg-compatible DSN string.
        """
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={database_name} "
            f"user={username} "
            f"password={password}"
        )

    @property
    def admin_dsn(self) -> str:
        """
        Admin DSN against the default target database.
        """
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_admin_user,
            password=self.postgres_admin_password,
        )

    @property
    def maintenance_admin_dsn(self) -> str:
        """
        Admin DSN against the maintenance database.

        Use this for cluster-level actions like CREATE DATABASE.
        """
        return self.make_dsn(
            database_name=self.postgres_maintenance_db,
            username=self.postgres_admin_user,
            password=self.postgres_admin_password,
        )

    @property
    def read_dsn(self) -> str:
        """
        Read-only DSN against the default target database.

        Kept for backward compatibility with the first version of server.py.
        """
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_read_user,
            password=self.postgres_read_password,
        )

    @property
    def write_dsn(self) -> str:
        """
        Read/write DSN against the default target database.

        Kept for backward compatibility with the first version of server.py.
        """
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_write_user,
            password=self.postgres_write_password,
        )


def get_settings() -> Settings:
    postgres_host = env_str("POSTGRES_HOST", "127.0.0.1")
    postgres_port = env_int("POSTGRES_PORT", 5432)

    maintenance_db = env_str("POSTGRES_MAINTENANCE_DB", "postgres")

    # Support both the new cluster-level variable and the old POSTGRES_DB.
    default_db = env_str(
        "POSTGRES_DEFAULT_DB",
        env_str("POSTGRES_DB", "postgres"),
    )

    default_schema = env_str("POSTGRES_DEFAULT_SCHEMA", "public")

    return Settings(
        postgres_host=postgres_host,
        postgres_port=postgres_port,

        postgres_maintenance_db=maintenance_db,
        postgres_default_db=default_db,

        # Backward-compatible alias.
        postgres_db=default_db,

        postgres_default_schema=default_schema,

        postgres_admin_user=env_str("POSTGRES_ADMIN_USER", "mcp_admin"),
        postgres_admin_password=env_str("POSTGRES_ADMIN_PASSWORD", ""),

        postgres_read_user=env_str("POSTGRES_READ_USER", "mcp_read"),
        postgres_read_password=env_str("POSTGRES_READ_PASSWORD", ""),

        postgres_write_user=env_str("POSTGRES_WRITE_USER", "mcp_write"),
        postgres_write_password=env_str("POSTGRES_WRITE_PASSWORD", ""),

        mcp_server_name=env_str(
            "MCP_SERVER_NAME",
            "PostgreSQL Cluster MCP Server",
        ),
        mcp_transport=env_str("MCP_TRANSPORT", "stdio"),
        mcp_http_host=env_str("MCP_HTTP_HOST", "127.0.0.1"),
        mcp_http_port=env_int("MCP_HTTP_PORT", 8000),

        max_read_rows=env_int("MAX_READ_ROWS", 500),
    )


if __name__ == "__main__":
    settings = get_settings()

    print("PostgreSQL MCP settings loaded:")
    print(f"  POSTGRES_HOST={settings.postgres_host}")
    print(f"  POSTGRES_PORT={settings.postgres_port}")
    print(f"  POSTGRES_MAINTENANCE_DB={settings.postgres_maintenance_db}")
    print(f"  POSTGRES_DEFAULT_DB={settings.postgres_default_db}")
    print(f"  POSTGRES_DEFAULT_SCHEMA={settings.postgres_default_schema}")
    print(f"  POSTGRES_ADMIN_USER={settings.postgres_admin_user}")
    print(f"  POSTGRES_READ_USER={settings.postgres_read_user}")
    print(f"  POSTGRES_WRITE_USER={settings.postgres_write_user}")
    print(f"  MCP_SERVER_NAME={settings.mcp_server_name}")
    print(f"  MCP_TRANSPORT={settings.mcp_transport}")
    print(f"  MCP_HTTP_HOST={settings.mcp_http_host}")
    print(f"  MCP_HTTP_PORT={settings.mcp_http_port}")
    print(f"  MAX_READ_ROWS={settings.max_read_rows}")