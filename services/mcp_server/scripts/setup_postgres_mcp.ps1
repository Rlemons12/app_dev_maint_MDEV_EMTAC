param(
    [string]$PostgresHost = "127.0.0.1",
    [int]$PostgresPort = 5432,

    [string]$MaintenanceDb = "postgres",
    [string]$DefaultDb = "postgres",
    [string]$DefaultSchema = "public",

    [string]$AdminUser = "mcp_admin",
    [string]$AdminPassword = "McpAdmin123!",

    [string]$ReadUser = "mcp_read",
    [string]$ReadPassword = "McpRead123!",

    [string]$WriteUser = "mcp_write",
    [string]$WritePassword = "McpWrite123!",

    [string]$McpServerName = "PostgreSQL Cluster MCP Server",
    [string]$McpTransport = "stdio",
    [string]$McpHttpHost = "127.0.0.1",
    [int]$McpHttpPort = 8000,

    [int]$MaxReadRows = 500,

    [string]$PostgresSuperUser = "postgres",
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\17\bin\psql.exe",

    [switch]$InstallPythonPackages,
    [switch]$ApplyDatabaseRoles,
    [switch]$ForceEnv,

    [switch]$Interactive,
    [switch]$NoInteractive
)

$ErrorActionPreference = "Stop"

# ============================================================
# PostgreSQL Cluster MCP setup script
#
# This script is designed to run from the project root.
#
# It generates or updates:
#   - requirements.txt
#   - .env.example
#   - .env, optionally
#   - settings.py
#   - db.py
#   - sql_safety.py
#   - server.py
#   - sql\postgres_roles_setup.sql
#   - README_POSTGRES_MCP.md
#   - .gitignore block
#
# It supports existing PostgreSQL installations by detecting services
# and psql.exe paths, then asking which instance to use.
# ============================================================

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Gray
}

function Write-Warn {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Good {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Read-WithDefault {
    param(
        [string]$Prompt,
        [string]$Default
    )

    $value = Read-Host "$Prompt [$Default]"

    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }

    return $value.Trim()
}

function Read-IntWithDefault {
    param(
        [string]$Prompt,
        [int]$Default
    )

    while ($true) {
        $raw = Read-WithDefault -Prompt $Prompt -Default "$Default"

        $parsed = 0

        if ([int]::TryParse($raw, [ref]$parsed)) {
            return $parsed
        }

        Write-Warn "Please enter a valid integer."
    }
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$Default = $false
    )

    $defaultText = if ($Default) { "Y" } else { "N" }

    while ($true) {
        $value = Read-Host "$Prompt [$defaultText]"

        if ([string]::IsNullOrWhiteSpace($value)) {
            return $Default
        }

        $clean = $value.Trim().ToLowerInvariant()

        if ($clean -in @("y", "yes", "true", "1")) {
            return $true
        }

        if ($clean -in @("n", "no", "false", "0")) {
            return $false
        }

        Write-Warn "Please answer Y or N."
    }
}

function Read-SecretWithDefault {
    param(
        [string]$Prompt,
        [string]$Default
    )

    $useDefault = Read-YesNo "$Prompt Use existing/default value?" $true

    if ($useDefault) {
        return $Default
    }

    $secure = Read-Host "$Prompt Enter new value" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
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

function Assert-RoleNameIsSafe {
    param(
        [string]$RoleName,
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($RoleName)) {
        throw "$Label cannot be blank."
    }

    if ($RoleName -match "^pg_") {
        throw "$Label '$RoleName' is invalid. PostgreSQL reserves role names starting with 'pg_'. Use names like mcp_admin, mcp_read, and mcp_write."
    }

    if ($RoleName -notmatch "^[a-z_][a-z0-9_]*$") {
        throw "$Label '$RoleName' is invalid. Use lowercase letters, numbers, and underscores only. The first character must be a lowercase letter or underscore."
    }
}

function Convert-ToSqlStringLiteral {
    param([string]$Value)

    return $Value -replace "'", "''"
}

function Find-PostgresServices {
    try {
        return @(Get-Service *postgres* -ErrorAction SilentlyContinue)
    }
    catch {
        return @()
    }
}

function Find-PsqlExecutables {
    $results = @()

    $commonRoot = "C:\Program Files\PostgreSQL"

    if (Test-Path $commonRoot) {
        $results += Get-ChildItem $commonRoot -Recurse -Filter psql.exe -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }

    $pathCommand = Get-Command psql -ErrorAction SilentlyContinue

    if ($pathCommand -and $pathCommand.Source) {
        $results += $pathCommand.Source
    }

    return @($results | Where-Object { $_ } | Sort-Object -Unique)
}

function Select-PsqlPathInteractively {
    param([string]$CurrentDefault)

    $found = Find-PsqlExecutables

    if ($found.Count -gt 0) {
        Write-Host ""
        Write-Host "Detected psql.exe candidates:" -ForegroundColor Cyan

        for ($i = 0; $i -lt $found.Count; $i++) {
            Write-Host "  [$($i + 1)] $($found[$i])"
        }

        $defaultIndex = 1
        $choiceRaw = Read-WithDefault -Prompt "Choose psql.exe number" -Default "$defaultIndex"

        $choice = 0

        if ([int]::TryParse($choiceRaw, [ref]$choice)) {
            if ($choice -ge 1 -and $choice -le $found.Count) {
                return $found[$choice - 1]
            }
        }

        Write-Warn "Invalid selection. Using current default."
    }

    return Read-WithDefault -Prompt "Path to psql.exe" -Default $CurrentDefault
}

function Test-PsqlCanRun {
    param([string]$CandidatePsqlPath)

    if (Test-Path $CandidatePsqlPath) {
        return $true
    }

    $cmd = Get-Command psql -ErrorAction SilentlyContinue

    return [bool]$cmd
}

# Convert switches to mutable booleans for interactive changes.
$DoInstallPythonPackages = [bool]$InstallPythonPackages
$DoApplyDatabaseRoles = [bool]$ApplyDatabaseRoles
$DoForceEnv = [bool]$ForceEnv

# ------------------------------------------------------------
# Optional interactive setup
# ------------------------------------------------------------

$UseInteractive = $false

if ($NoInteractive) {
    $UseInteractive = $false
}
elseif ($Interactive) {
    $UseInteractive = $true
}
else {
    $UseInteractive = Read-YesNo "Do you want interactive setup for an existing PostgreSQL install?" $true
}

if ($UseInteractive) {
    Write-Step "Detecting existing PostgreSQL installations"

    $services = Find-PostgresServices

    if ($services.Count -gt 0) {
        Write-Host "Detected PostgreSQL-related Windows services:" -ForegroundColor Cyan
        $services | Format-Table Status, Name, DisplayName -AutoSize
    }
    else {
        Write-Warn "No PostgreSQL Windows service was detected through Get-Service *postgres*."
    }

    $PsqlPath = Select-PsqlPathInteractively -CurrentDefault $PsqlPath

    $PostgresHost = Read-WithDefault -Prompt "PostgreSQL host" -Default $PostgresHost
    $PostgresPort = Read-IntWithDefault -Prompt "PostgreSQL port" -Default $PostgresPort

    $MaintenanceDb = Read-WithDefault -Prompt "Maintenance database for cluster-level actions" -Default $MaintenanceDb
    $DefaultDb = Read-WithDefault -Prompt "Default MCP target database" -Default $DefaultDb
    $DefaultSchema = Read-WithDefault -Prompt "Default schema" -Default $DefaultSchema

    $PostgresSuperUser = Read-WithDefault -Prompt "PostgreSQL admin/superuser used to apply roles" -Default $PostgresSuperUser

    $AdminUser = Read-WithDefault -Prompt "MCP admin role name" -Default $AdminUser
    $ReadUser = Read-WithDefault -Prompt "MCP read role name" -Default $ReadUser
    $WriteUser = Read-WithDefault -Prompt "MCP write role name" -Default $WriteUser

    $AdminPassword = Read-SecretWithDefault -Prompt "MCP admin password." -Default $AdminPassword
    $ReadPassword = Read-SecretWithDefault -Prompt "MCP read password." -Default $ReadPassword
    $WritePassword = Read-SecretWithDefault -Prompt "MCP write password." -Default $WritePassword

    $McpTransport = Read-WithDefault -Prompt "MCP transport: stdio or streamable-http" -Default $McpTransport
    $McpHttpHost = Read-WithDefault -Prompt "MCP HTTP host" -Default $McpHttpHost
    $McpHttpPort = Read-IntWithDefault -Prompt "MCP HTTP port" -Default $McpHttpPort
    $MaxReadRows = Read-IntWithDefault -Prompt "Max read rows" -Default $MaxReadRows

    $DoForceEnv = Read-YesNo "Overwrite .env with these values?" $true
    $DoInstallPythonPackages = Read-YesNo "Create/update .venv and install Python packages?" $true
    $DoApplyDatabaseRoles = Read-YesNo "Apply PostgreSQL MCP roles now?" $false
}

Assert-RoleNameIsSafe -RoleName $AdminUser -Label "AdminUser"
Assert-RoleNameIsSafe -RoleName $ReadUser -Label "ReadUser"
Assert-RoleNameIsSafe -RoleName $WriteUser -Label "WriteUser"

$ReadGroupRole = "${ReadUser}_role"
$WriteGroupRole = "${WriteUser}_role"

Assert-RoleNameIsSafe -RoleName $ReadGroupRole -Label "Read group role"
Assert-RoleNameIsSafe -RoleName $WriteGroupRole -Label "Write group role"

$AdminPasswordSql = Convert-ToSqlStringLiteral $AdminPassword
$ReadPasswordSql = Convert-ToSqlStringLiteral $ReadPassword
$WritePasswordSql = Convert-ToSqlStringLiteral $WritePassword

$ProjectRoot = (Get-Location).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

Write-Step "PostgreSQL Cluster MCP setup starting"
Write-Host "Project root:           $ProjectRoot"
Write-Host "PostgreSQL host:        $PostgresHost"
Write-Host "PostgreSQL port:        $PostgresPort"
Write-Host "Maintenance database:   $MaintenanceDb"
Write-Host "Default database:       $DefaultDb"
Write-Host "Default schema:         $DefaultSchema"
Write-Host "psql.exe:               $PsqlPath"
Write-Host "Postgres superuser:     $PostgresSuperUser"
Write-Host "Admin role:             $AdminUser"
Write-Host "Read role:              $ReadUser"
Write-Host "Write role:             $WriteUser"

# ------------------------------------------------------------
# requirements.txt
# ------------------------------------------------------------

$requirements = @'
mcp[cli]
psycopg[binary]
python-dotenv
sqlparse
pydantic
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "requirements.txt") -Content $requirements

# ------------------------------------------------------------
# .env and .env.example
# ------------------------------------------------------------

$envContent = @"
POSTGRES_HOST=$PostgresHost
POSTGRES_PORT=$PostgresPort

POSTGRES_MAINTENANCE_DB=$MaintenanceDb
POSTGRES_DEFAULT_DB=$DefaultDb
POSTGRES_DEFAULT_SCHEMA=$DefaultSchema

POSTGRES_ADMIN_USER=$AdminUser
POSTGRES_ADMIN_PASSWORD=$AdminPassword

POSTGRES_READ_USER=$ReadUser
POSTGRES_READ_PASSWORD=$ReadPassword

POSTGRES_WRITE_USER=$WriteUser
POSTGRES_WRITE_PASSWORD=$WritePassword

MCP_SERVER_NAME=$McpServerName
MCP_TRANSPORT=$McpTransport
MCP_HTTP_HOST=$McpHttpHost
MCP_HTTP_PORT=$McpHttpPort

MAX_READ_ROWS=$MaxReadRows
"@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot ".env.example") -Content $envContent

$EnvPath = Join-Path $ProjectRoot ".env"

if ((-not (Test-Path $EnvPath)) -or $DoForceEnv) {
    Write-FileUtf8NoBom -Path $EnvPath -Content $envContent
    Write-Good "Created/updated .env"
}
else {
    Write-Warn ".env already exists. Leaving it unchanged. Use -ForceEnv or answer Yes during interactive setup to overwrite it."
}

# ------------------------------------------------------------
# settings.py
# ------------------------------------------------------------

$settingsPy = @'
from __future__ import annotations

from dataclasses import dataclass
import os

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


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return int(value)


@dataclass(frozen=True)
class Settings:
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

    mcp_server_name: str
    mcp_transport: str
    mcp_http_host: str
    mcp_http_port: int

    max_read_rows: int

    def make_dsn(
        self,
        database_name: str,
        username: str,
        password: str,
    ) -> str:
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={database_name} "
            f"user={username} "
            f"password={password}"
        )

    @property
    def admin_dsn(self) -> str:
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_admin_user,
            password=self.postgres_admin_password,
        )

    @property
    def maintenance_admin_dsn(self) -> str:
        return self.make_dsn(
            database_name=self.postgres_maintenance_db,
            username=self.postgres_admin_user,
            password=self.postgres_admin_password,
        )

    @property
    def read_dsn(self) -> str:
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_read_user,
            password=self.postgres_read_password,
        )

    @property
    def write_dsn(self) -> str:
        return self.make_dsn(
            database_name=self.postgres_default_db,
            username=self.postgres_write_user,
            password=self.postgres_write_password,
        )


def get_settings() -> Settings:
    postgres_host = env_str("POSTGRES_HOST", "127.0.0.1")
    postgres_port = env_int("POSTGRES_PORT", 5432)

    maintenance_db = env_str("POSTGRES_MAINTENANCE_DB", "postgres")

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
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "settings.py") -Content $settingsPy

# ------------------------------------------------------------
# sql_safety.py
# ------------------------------------------------------------

$sqlSafetyPy = @'
from __future__ import annotations

import re
from typing import Iterable

import sqlparse


READ_ALLOWED_STARTERS = {
    "select",
    "with",
    "show",
    "explain",
}

READ_BLOCKED_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "do",
    "merge",
    "vacuum",
    "analyze",
    "refresh",
    "reindex",
    "cluster",
    "comment",
}


def split_sql_statements(sql: str) -> list[str]:
    return [
        statement.strip()
        for statement in sqlparse.split(sql)
        if statement.strip()
    ]


def first_token(sql: str) -> str:
    parsed = sqlparse.parse(sql)

    if not parsed:
        return ""

    for token in parsed[0].flatten():
        value = token.value.strip()

        if value:
            return value.lower()

    return ""


def contains_blocked_keyword(sql: str, blocked_keywords: Iterable[str]) -> bool:
    lowered = sql.lower()

    for keyword in blocked_keywords:
        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(pattern, lowered):
            return True

    return False


def validate_read_only_sql(sql: str) -> None:
    statements = split_sql_statements(sql)

    if not statements:
        raise ValueError("SQL is empty.")

    if len(statements) > 1:
        raise ValueError("Read-only tool accepts one SQL statement at a time.")

    statement = statements[0]
    starter = first_token(statement)

    if starter not in READ_ALLOWED_STARTERS:
        raise ValueError(
            "Read-only SQL must start with one of: "
            f"{', '.join(sorted(READ_ALLOWED_STARTERS))}."
        )

    if contains_blocked_keyword(statement, READ_BLOCKED_KEYWORDS):
        raise ValueError("Read-only SQL contains a blocked write/admin keyword.")


def ensure_limit(sql: str, max_rows: int) -> str:
    statement = sql.strip().rstrip(";")
    starter = first_token(statement)

    if starter not in {"select", "with"}:
        return statement

    if re.search(r"\blimit\b", statement, flags=re.IGNORECASE):
        return statement

    return f"{statement}\nLIMIT {int(max_rows)}"
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "sql_safety.py") -Content $sqlSafetyPy

# ------------------------------------------------------------
# db.py
# ------------------------------------------------------------

$dbPy = @'
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from psycopg.rows import dict_row
from psycopg import sql as pg_sql


class PostgresClient:
    def __init__(self, dsn: str, application_name: str) -> None:
        self.dsn = dsn
        self.application_name = application_name

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        conn = psycopg.connect(
            self.dsn,
            row_factory=dict_row,
            application_name=self.application_name,
        )

        try:
            yield conn
        finally:
            conn.close()

    def fetch_all(
        self,
        query: str | pg_sql.Composable,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                rows = cur.fetchall()
                return [dict(row) for row in rows]

    def execute(
        self,
        query: str | pg_sql.Composable,
        params: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())

                    result: dict[str, Any] = {
                        "status": "ok",
                        "rowcount": cur.rowcount,
                    }

                    if cur.description:
                        rows = cur.fetchall()
                        result["rows"] = [dict(row) for row in rows]
                        result["returned_rows"] = len(rows)

                    conn.commit()
                    return result

            except Exception:
                conn.rollback()
                raise

    def test_connection(self) -> dict[str, Any]:
        rows = self.fetch_all(
            """
            SELECT
                current_database() AS database_name,
                current_user AS current_user,
                inet_server_addr()::text AS server_address,
                inet_server_port() AS server_port,
                version() AS postgres_version
            """
        )

        return rows[0] if rows else {}
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "db.py") -Content $dbPy

# ------------------------------------------------------------
# server.py
# ------------------------------------------------------------

# The server.py generated here is intentionally the cluster-level MCP server.
# It uses dynamic database connections instead of binding to a single database.

$serverPy = @'
from __future__ import annotations

import re
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from mcp.server.fastmcp import FastMCP
from psycopg import sql as pg_sql
from psycopg.rows import dict_row

from db import PostgresClient
from settings import get_settings
from sql_safety import ensure_limit, validate_read_only_sql


settings = get_settings()

mcp = FastMCP(
    settings.mcp_server_name,
    json_response=True,
)


SYSTEM_DATABASES = {
    "postgres",
    "template0",
    "template1",
}

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
SAFE_SQL_FRAGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_\s\(\),\.\[\]'\":+-]+$")


def make_dsn(
    database_name: str,
    username: str,
    password: str,
) -> str:
    return settings.make_dsn(
        database_name=database_name,
        username=username,
        password=password,
    )


def resolve_database(database_name: str | None) -> str:
    database = database_name or settings.postgres_default_db

    if not database:
        raise ValueError(
            "No database name was provided and POSTGRES_DEFAULT_DB is not set."
        )

    return database


def resolve_schema(schema_name: str | None) -> str:
    schema = schema_name or settings.postgres_default_schema

    if not schema:
        raise ValueError(
            "No schema name was provided and POSTGRES_DEFAULT_SCHEMA is not set."
        )

    return schema


def validate_identifier_value(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} cannot be empty.")

    if not IDENTIFIER_PATTERN.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. "
            "Use letters, numbers, and underscores only. "
            "The first character must be a letter or underscore."
        )

    return value


def validate_sql_fragment(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} cannot be empty.")

    lowered = value.lower()

    blocked_tokens = [
        ";",
        "--",
        "/*",
        "*/",
        "\\",
    ]

    for token in blocked_tokens:
        if token in lowered:
            raise ValueError(
                f"Invalid {label}: blocked token {token!r} was found."
            )

    if not SAFE_SQL_FRAGMENT_PATTERN.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. "
            "Only simple SQL fragments are allowed here."
        )

    return value


def make_client(
    role: str,
    database_name: str | None = None,
) -> PostgresClient:
    database = resolve_database(database_name)

    if role == "read":
        return PostgresClient(
            dsn=make_dsn(
                database_name=database,
                username=settings.postgres_read_user,
                password=settings.postgres_read_password,
            ),
            application_name="postgres-mcp-read",
        )

    if role == "write":
        return PostgresClient(
            dsn=make_dsn(
                database_name=database,
                username=settings.postgres_write_user,
                password=settings.postgres_write_password,
            ),
            application_name="postgres-mcp-write",
        )

    if role == "admin":
        return PostgresClient(
            dsn=make_dsn(
                database_name=database,
                username=settings.postgres_admin_user,
                password=settings.postgres_admin_password,
            ),
            application_name="postgres-mcp-admin",
        )

    raise ValueError(f"Unknown database role: {role}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, memoryview):
        return value.tobytes().hex()

    if isinstance(value, bytes):
        return value.hex()

    return value


def fetch_all_json_safe(
    client: PostgresClient,
    query: str | pg_sql.Composable,
    params: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    return json_safe(client.fetch_all(query, params))


def execute_json_safe(
    client: PostgresClient,
    query: str | pg_sql.Composable,
    params: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    return json_safe(client.execute(query, params))


def execute_autocommit(
    query: str | pg_sql.Composable,
    params: tuple[Any, ...] | None = None,
    database_name: str | None = None,
    application_name: str = "postgres-mcp-admin-autocommit",
) -> dict[str, Any]:
    database = database_name or settings.postgres_maintenance_db

    dsn = make_dsn(
        database_name=database,
        username=settings.postgres_admin_user,
        password=settings.postgres_admin_password,
    )

    with psycopg.connect(
        dsn,
        row_factory=dict_row,
        application_name=application_name,
    ) as conn:
        conn.autocommit = True

        sql_text = (
            query.as_string(conn)
            if isinstance(query, pg_sql.Composable)
            else query
        )

        with conn.cursor() as cur:
            cur.execute(query, params or ())

            result: dict[str, Any] = {
                "status": "ok",
                "database_name": database,
                "rowcount": cur.rowcount,
                "sql_executed": sql_text,
            }

            if cur.description:
                rows = cur.fetchall()
                result["rows"] = [dict(row) for row in rows]
                result["returned_rows"] = len(rows)

            return json_safe(result)


def grant_standard_database_permissions_private(
    database_name: str,
    schema_name: str,
) -> dict[str, Any]:
    validate_identifier_value(database_name, "database_name")
    validate_identifier_value(schema_name, "schema_name")

    db_grant_query = pg_sql.SQL(
        "GRANT CONNECT ON DATABASE {} TO {}, {}"
    ).format(
        pg_sql.Identifier(database_name),
        pg_sql.Identifier(settings.postgres_read_user),
        pg_sql.Identifier(settings.postgres_write_user),
    )

    execute_autocommit(
        db_grant_query,
        database_name=settings.postgres_maintenance_db,
    )

    admin_client = make_client("admin", database_name)

    statements: list[pg_sql.Composable] = [
        pg_sql.SQL("GRANT USAGE ON SCHEMA {} TO {}, {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_read_user),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
        pg_sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
        pg_sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_read_user),
        ),
        pg_sql.SQL(
            "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
            "ON ALL TABLES IN SCHEMA {} TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
        pg_sql.SQL("GRANT SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_read_user),
        ),
        pg_sql.SQL(
            "GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA {} TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
        pg_sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} "
            "GRANT SELECT ON TABLES TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_read_user),
        ),
        pg_sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} "
            "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
            "ON TABLES TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
        pg_sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} "
            "GRANT SELECT ON SEQUENCES TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_read_user),
        ),
        pg_sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA {} "
            "GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {}"
        ).format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(settings.postgres_write_user),
        ),
    ]

    results: list[dict[str, Any]] = []

    for statement in statements:
        results.append(execute_json_safe(admin_client, statement))

    return {
        "status": "ok",
        "database_name": database_name,
        "schema_name": schema_name,
        "read_role": settings.postgres_read_user,
        "write_role": settings.postgres_write_user,
        "grant_results": results,
    }


@mcp.tool()
def postgres_health_check(
    database_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)

    admin_client = make_client("admin", settings.postgres_maintenance_db)
    read_client = make_client("read", target_database)
    write_client = make_client("write", target_database)

    return {
        "postgres_host": settings.postgres_host,
        "postgres_port": settings.postgres_port,
        "maintenance_database": settings.postgres_maintenance_db,
        "target_database": target_database,
        "admin_connection": json_safe(admin_client.test_connection()),
        "read_connection": json_safe(read_client.test_connection()),
        "write_connection": json_safe(write_client.test_connection()),
    }


@mcp.tool()
def postgres_whoami(
    database_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)

    query = """
        SELECT
            current_database() AS database_name,
            current_user AS current_user,
            session_user AS session_user,
            inet_server_addr()::text AS server_address,
            inet_server_port() AS server_port
    """

    return {
        "admin": fetch_all_json_safe(
            make_client("admin", target_database),
            query,
        ),
        "read": fetch_all_json_safe(
            make_client("read", target_database),
            query,
        ),
        "write": fetch_all_json_safe(
            make_client("write", target_database),
            query,
        ),
    }


@mcp.tool()
def postgres_list_databases(
    include_templates: bool = False,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            d.datname AS database_name,
            pg_get_userbyid(d.datdba) AS owner_name,
            pg_encoding_to_char(d.encoding) AS encoding,
            d.datcollate AS collation,
            d.datctype AS character_type,
            d.datistemplate AS is_template,
            d.datallowconn AS allows_connections,
            pg_size_pretty(pg_database_size(d.datname)) AS database_size
        FROM pg_database d
        WHERE (%s = TRUE OR d.datistemplate = FALSE)
        ORDER BY d.datistemplate, d.datname
    """

    admin_client = make_client("admin", settings.postgres_maintenance_db)

    return fetch_all_json_safe(
        admin_client,
        query,
        (include_templates,),
    )


@mcp.tool()
def postgres_create_database(
    database_name: str,
    owner_role: str | None = None,
    template_name: str = "template1",
    encoding: str = "UTF8",
    grant_standard_permissions: bool = True,
) -> dict[str, Any]:
    validate_identifier_value(database_name, "database_name")
    validate_identifier_value(template_name, "template_name")

    owner = owner_role or settings.postgres_admin_user
    validate_identifier_value(owner, "owner_role")

    encoding = validate_sql_fragment(encoding, "encoding")

    query = pg_sql.SQL(
        "CREATE DATABASE {} WITH OWNER {} TEMPLATE {} ENCODING {}"
    ).format(
        pg_sql.Identifier(database_name),
        pg_sql.Identifier(owner),
        pg_sql.Identifier(template_name),
        pg_sql.Literal(encoding),
    )

    create_result = execute_autocommit(
        query,
        database_name=settings.postgres_maintenance_db,
    )

    grant_result: dict[str, Any] | None = None

    if grant_standard_permissions:
        grant_result = grant_standard_database_permissions_private(
            database_name=database_name,
            schema_name="public",
        )

    return {
        "status": "ok",
        "operation": "create_database",
        "database_name": database_name,
        "owner_role": owner,
        "create_result": create_result,
        "grant_standard_permissions": grant_standard_permissions,
        "grant_result": grant_result,
    }


@mcp.tool()
def postgres_drop_database(
    database_name: str,
    force: bool = False,
    allow_system_database: bool = False,
) -> dict[str, Any]:
    validate_identifier_value(database_name, "database_name")

    if database_name in SYSTEM_DATABASES and not allow_system_database:
        raise ValueError(
            f"Refusing to drop system database {database_name!r}. "
            "Set allow_system_database=true if you really intend this."
        )

    if force:
        query = pg_sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
            pg_sql.Identifier(database_name)
        )
    else:
        query = pg_sql.SQL("DROP DATABASE {}").format(
            pg_sql.Identifier(database_name)
        )

    result = execute_autocommit(
        query,
        database_name=settings.postgres_maintenance_db,
    )

    return {
        "status": "ok",
        "operation": "drop_database",
        "database_name": database_name,
        "force": force,
        "result": result,
    }


@mcp.tool()
def postgres_list_schemas(
    database_name: str | None = None,
    include_system_schemas: bool = False,
) -> list[dict[str, Any]]:
    target_database = resolve_database(database_name)

    query = """
        SELECT
            schema_name,
            schema_owner
        FROM information_schema.schemata
        WHERE (
            %s = TRUE
            OR schema_name NOT IN ('pg_catalog', 'information_schema')
        )
        AND (
            %s = TRUE
            OR schema_name NOT LIKE 'pg_toast%%'
        )
        ORDER BY schema_name
    """

    admin_client = make_client("admin", target_database)

    return fetch_all_json_safe(
        admin_client,
        query,
        (include_system_schemas, include_system_schemas),
    )


@mcp.tool()
def postgres_create_schema(
    schema_name: str,
    database_name: str | None = None,
    owner_role: str | None = None,
    if_not_exists: bool = True,
    grant_standard_permissions: bool = True,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    validate_identifier_value(target_database, "database_name")
    validate_identifier_value(schema_name, "schema_name")

    owner = owner_role or settings.postgres_admin_user
    validate_identifier_value(owner, "owner_role")

    if if_not_exists:
        query = pg_sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(owner),
        )
    else:
        query = pg_sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
            pg_sql.Identifier(schema_name),
            pg_sql.Identifier(owner),
        )

    admin_client = make_client("admin", target_database)
    create_result = execute_json_safe(admin_client, query)

    grant_result: dict[str, Any] | None = None

    if grant_standard_permissions:
        grant_result = grant_standard_database_permissions_private(
            database_name=target_database,
            schema_name=schema_name,
        )

    return {
        "status": "ok",
        "operation": "create_schema",
        "database_name": target_database,
        "schema_name": schema_name,
        "owner_role": owner,
        "create_result": create_result,
        "grant_standard_permissions": grant_standard_permissions,
        "grant_result": grant_result,
    }


@mcp.tool()
def postgres_drop_schema(
    schema_name: str,
    database_name: str | None = None,
    cascade: bool = False,
    allow_public_schema: bool = False,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    validate_identifier_value(target_database, "database_name")
    validate_identifier_value(schema_name, "schema_name")

    if schema_name == "public" and not allow_public_schema:
        raise ValueError(
            "Refusing to drop the public schema by default. "
            "Set allow_public_schema=true if you really intend this."
        )

    if cascade:
        query = pg_sql.SQL("DROP SCHEMA {} CASCADE").format(
            pg_sql.Identifier(schema_name)
        )
    else:
        query = pg_sql.SQL("DROP SCHEMA {} RESTRICT").format(
            pg_sql.Identifier(schema_name)
        )

    admin_client = make_client("admin", target_database)
    result = execute_json_safe(admin_client, query)

    return {
        "status": "ok",
        "operation": "drop_schema",
        "database_name": target_database,
        "schema_name": schema_name,
        "cascade": cascade,
        "result": result,
    }


@mcp.tool()
def postgres_list_tables(
    database_name: str | None = None,
    schema_name: str | None = None,
    include_views: bool = True,
) -> list[dict[str, Any]]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    query = """
        SELECT
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = %s
          AND (
                table_type = 'BASE TABLE'
                OR (%s = TRUE AND table_type = 'VIEW')
          )
        ORDER BY table_schema, table_name
    """

    admin_client = make_client("admin", target_database)

    return fetch_all_json_safe(
        admin_client,
        query,
        (schema, include_views),
    )


@mcp.tool()
def postgres_describe_table(
    table_name: str,
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(table_name, "table_name")

    admin_client = make_client("admin", target_database)

    columns_query = """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            udt_name,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
    """

    indexes_query = """
        SELECT
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = %s
          AND tablename = %s
        ORDER BY indexname
    """

    constraints_query = """
        SELECT
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        LEFT JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema = kcu.table_schema
        LEFT JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
           AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = %s
          AND tc.table_name = %s
        ORDER BY tc.constraint_type, tc.constraint_name, kcu.ordinal_position
    """

    return {
        "database_name": target_database,
        "schema_name": schema,
        "table_name": table_name,
        "columns": fetch_all_json_safe(
            admin_client,
            columns_query,
            (schema, table_name),
        ),
        "indexes": fetch_all_json_safe(
            admin_client,
            indexes_query,
            (schema, table_name),
        ),
        "constraints": fetch_all_json_safe(
            admin_client,
            constraints_query,
            (schema, table_name),
        ),
    }


@mcp.tool()
def postgres_create_table(
    table_name: str,
    columns: list[dict[str, Any]],
    database_name: str | None = None,
    schema_name: str | None = None,
    if_not_exists: bool = True,
    grant_standard_permissions: bool = True,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(target_database, "database_name")
    validate_identifier_value(schema, "schema_name")
    validate_identifier_value(table_name, "table_name")

    if not columns:
        raise ValueError("columns cannot be empty.")

    column_definitions: list[pg_sql.Composable] = []
    primary_key_columns: list[str] = []

    for column in columns:
        column_name = validate_identifier_value(
            str(column.get("name", "")),
            "column name",
        )

        column_type = validate_sql_fragment(
            str(column.get("type") or column.get("data_type") or ""),
            f"data type for column {column_name}",
        )

        parts: list[pg_sql.Composable] = [
            pg_sql.Identifier(column_name),
            pg_sql.SQL(column_type),
        ]

        if bool(column.get("primary_key", False)):
            primary_key_columns.append(column_name)

        if column.get("nullable") is False or bool(column.get("not_null", False)):
            parts.append(pg_sql.SQL("NOT NULL"))

        if bool(column.get("unique", False)):
            parts.append(pg_sql.SQL("UNIQUE"))

        if "default" in column and column.get("default") is not None:
            default_expression = validate_sql_fragment(
                str(column["default"]),
                f"default expression for column {column_name}",
            )

            parts.append(
                pg_sql.SQL("DEFAULT ") + pg_sql.SQL(default_expression)
            )

        column_definitions.append(pg_sql.SQL(" ").join(parts))

    if primary_key_columns:
        primary_key_definition = pg_sql.SQL("PRIMARY KEY ({})").format(
            pg_sql.SQL(", ").join(
                pg_sql.Identifier(column_name)
                for column_name in primary_key_columns
            )
        )

        column_definitions.append(primary_key_definition)

    create_prefix = (
        pg_sql.SQL("CREATE TABLE IF NOT EXISTS")
        if if_not_exists
        else pg_sql.SQL("CREATE TABLE")
    )

    query = pg_sql.SQL("{} {}.{} ({})").format(
        create_prefix,
        pg_sql.Identifier(schema),
        pg_sql.Identifier(table_name),
        pg_sql.SQL(", ").join(column_definitions),
    )

    admin_client = make_client("admin", target_database)
    create_result = execute_json_safe(admin_client, query)

    grant_result: dict[str, Any] | None = None

    if grant_standard_permissions:
        grant_result = grant_standard_database_permissions_private(
            database_name=target_database,
            schema_name=schema,
        )

    return {
        "status": "ok",
        "operation": "create_table",
        "database_name": target_database,
        "schema_name": schema,
        "table_name": table_name,
        "create_result": create_result,
        "grant_standard_permissions": grant_standard_permissions,
        "grant_result": grant_result,
    }


@mcp.tool()
def postgres_drop_table(
    table_name: str,
    database_name: str | None = None,
    schema_name: str | None = None,
    if_exists: bool = True,
    cascade: bool = False,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(target_database, "database_name")
    validate_identifier_value(schema, "schema_name")
    validate_identifier_value(table_name, "table_name")

    query = pg_sql.SQL("DROP TABLE {}{}.{} {}").format(
        pg_sql.SQL("IF EXISTS ") if if_exists else pg_sql.SQL(""),
        pg_sql.Identifier(schema),
        pg_sql.Identifier(table_name),
        pg_sql.SQL("CASCADE") if cascade else pg_sql.SQL("RESTRICT"),
    )

    admin_client = make_client("admin", target_database)
    result = execute_json_safe(admin_client, query)

    return {
        "status": "ok",
        "operation": "drop_table",
        "database_name": target_database,
        "schema_name": schema,
        "table_name": table_name,
        "cascade": cascade,
        "result": result,
    }


@mcp.tool()
def postgres_grant_standard_database_permissions(
    database_name: str,
    schema_name: str | None = None,
) -> dict[str, Any]:
    schema = resolve_schema(schema_name)

    return grant_standard_database_permissions_private(
        database_name=database_name,
        schema_name=schema,
    )


@mcp.tool()
def postgres_read_query(
    sql: str,
    database_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)

    validate_read_only_sql(sql)

    limited_sql = ensure_limit(sql, settings.max_read_rows)

    read_client = make_client("read", target_database)
    rows = fetch_all_json_safe(read_client, limited_sql)

    return {
        "role": "read",
        "database_name": target_database,
        "max_rows": settings.max_read_rows,
        "sql_executed": limited_sql,
        "row_count": len(rows),
        "rows": rows,
    }


@mcp.tool()
def postgres_write_execute(
    sql: str,
    database_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)

    write_client = make_client("write", target_database)

    result = execute_json_safe(write_client, sql)

    result["role"] = "read_write"
    result["database_name"] = target_database
    result["sql_executed"] = sql

    return result


@mcp.tool()
def postgres_admin_execute(
    sql: str,
    database_name: str | None = None,
    autocommit: bool = True,
) -> dict[str, Any]:
    target_database = database_name or settings.postgres_maintenance_db

    if autocommit:
        result = execute_autocommit(
            sql,
            database_name=target_database,
        )
    else:
        admin_client = make_client("admin", target_database)
        result = execute_json_safe(admin_client, sql)
        result["sql_executed"] = sql

    result["role"] = "admin"
    result["database_name"] = target_database

    return result


@mcp.tool()
def postgres_insert_row(
    table_name: str,
    row: dict[str, Any],
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(table_name, "table_name")
    validate_identifier_value(schema, "schema_name")

    if not row:
        raise ValueError("row cannot be empty.")

    columns = [
        validate_identifier_value(str(column), "column name")
        for column in row.keys()
    ]

    values = list(row.values())

    query = pg_sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) RETURNING *").format(
        pg_sql.Identifier(schema),
        pg_sql.Identifier(table_name),
        pg_sql.SQL(", ").join(pg_sql.Identifier(column) for column in columns),
        pg_sql.SQL(", ").join(pg_sql.Placeholder() for _ in values),
    )

    write_client = make_client("write", target_database)
    result = execute_json_safe(write_client, query, tuple(values))

    result["role"] = "read_write"
    result["operation"] = "insert_row"
    result["database_name"] = target_database
    result["schema_name"] = schema
    result["table_name"] = table_name

    return result


@mcp.tool()
def postgres_update_rows(
    table_name: str,
    values: dict[str, Any],
    where_column: str,
    where_value: Any,
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(table_name, "table_name")
    validate_identifier_value(schema, "schema_name")
    validate_identifier_value(where_column, "where_column")

    if not values:
        raise ValueError("values cannot be empty.")

    set_columns = [
        validate_identifier_value(str(column), "column name")
        for column in values.keys()
    ]

    set_values = list(values.values())

    assignments = [
        pg_sql.SQL("{} = {}").format(
            pg_sql.Identifier(column),
            pg_sql.Placeholder(),
        )
        for column in set_columns
    ]

    query = pg_sql.SQL(
        "UPDATE {}.{} SET {} WHERE {} = {} RETURNING *"
    ).format(
        pg_sql.Identifier(schema),
        pg_sql.Identifier(table_name),
        pg_sql.SQL(", ").join(assignments),
        pg_sql.Identifier(where_column),
        pg_sql.Placeholder(),
    )

    params = tuple(set_values + [where_value])

    write_client = make_client("write", target_database)
    result = execute_json_safe(write_client, query, params)

    result["role"] = "read_write"
    result["operation"] = "update_rows"
    result["database_name"] = target_database
    result["schema_name"] = schema
    result["table_name"] = table_name
    result["where_column"] = where_column

    return result


@mcp.tool()
def postgres_delete_rows(
    table_name: str,
    where_column: str,
    where_value: Any,
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    target_database = resolve_database(database_name)
    schema = resolve_schema(schema_name)

    validate_identifier_value(table_name, "table_name")
    validate_identifier_value(schema, "schema_name")
    validate_identifier_value(where_column, "where_column")

    query = pg_sql.SQL("DELETE FROM {}.{} WHERE {} = {} RETURNING *").format(
        pg_sql.Identifier(schema),
        pg_sql.Identifier(table_name),
        pg_sql.Identifier(where_column),
        pg_sql.Placeholder(),
    )

    write_client = make_client("write", target_database)
    result = execute_json_safe(write_client, query, (where_value,))

    result["role"] = "read_write"
    result["operation"] = "delete_rows"
    result["database_name"] = target_database
    result["schema_name"] = schema
    result["table_name"] = table_name
    result["where_column"] = where_column

    return result


if __name__ == "__main__":
    transport = settings.mcp_transport.lower().strip()

    if transport == "stdio":
        mcp.run(transport="stdio")

    elif transport in {"http", "streamable-http", "streamable_http"}:
        mcp.settings.host = settings.mcp_http_host
        mcp.settings.port = settings.mcp_http_port
        mcp.run(transport="streamable-http")

    else:
        raise ValueError("Invalid MCP_TRANSPORT. Use 'stdio' or 'streamable-http'.")
'@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "server.py") -Content $serverPy

# ------------------------------------------------------------
# sql\postgres_roles_setup.sql
# ------------------------------------------------------------

$rolesSql = @"
-- ============================================================
-- PostgreSQL Cluster MCP Role Setup
--
-- This setup is for a PostgreSQL instance / cluster MCP server.
--
-- IMPORTANT:
--   Custom PostgreSQL role names must NOT start with "pg_".
--   PostgreSQL reserves role names beginning with "pg_".
--
-- Run as real PostgreSQL admin/superuser:
--
--   "$PsqlPath" -h $PostgresHost -U $PostgresSuperUser -d $MaintenanceDb -f sql\postgres_roles_setup.sql
--
-- ============================================================

\set ON_ERROR_STOP on

\connect $MaintenanceDb

DO `$`$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = '$ReadGroupRole'
    ) THEN
        CREATE ROLE $ReadGroupRole NOLOGIN;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = '$WriteGroupRole'
    ) THEN
        CREATE ROLE $WriteGroupRole NOLOGIN;
    END IF;
END
`$`$;

DO `$`$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = '$AdminUser'
    ) THEN
        CREATE ROLE $AdminUser
            LOGIN
            CREATEDB
            CREATEROLE
            PASSWORD '$AdminPasswordSql';
    ELSE
        ALTER ROLE $AdminUser
            WITH LOGIN
            CREATEDB
            CREATEROLE
            PASSWORD '$AdminPasswordSql';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = '$ReadUser'
    ) THEN
        CREATE ROLE $ReadUser
            LOGIN
            PASSWORD '$ReadPasswordSql';
    ELSE
        ALTER ROLE $ReadUser
            WITH LOGIN
            PASSWORD '$ReadPasswordSql';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = '$WriteUser'
    ) THEN
        CREATE ROLE $WriteUser
            LOGIN
            PASSWORD '$WritePasswordSql';
    ELSE
        ALTER ROLE $WriteUser
            WITH LOGIN
            PASSWORD '$WritePasswordSql';
    END IF;
END
`$`$;

GRANT $ReadGroupRole TO $ReadUser;
GRANT $WriteGroupRole TO $WriteUser;

GRANT $ReadGroupRole TO $AdminUser;
GRANT $WriteGroupRole TO $AdminUser;

GRANT $AdminUser TO $PostgresSuperUser;

GRANT pg_read_all_data TO $ReadGroupRole;

GRANT pg_read_all_data TO $WriteGroupRole;
GRANT pg_write_all_data TO $WriteGroupRole;

GRANT pg_read_all_data TO $AdminUser;
GRANT pg_write_all_data TO $AdminUser;

DO `$`$
DECLARE
    db_record record;
BEGIN
    FOR db_record IN
        SELECT datname
        FROM pg_database
        WHERE datallowconn = TRUE
          AND datistemplate = FALSE
    LOOP
        EXECUTE format(
            'GRANT CONNECT ON DATABASE %I TO $AdminUser, $ReadGroupRole, $WriteGroupRole',
            db_record.datname
        );

        EXECUTE format(
            'GRANT TEMPORARY ON DATABASE %I TO $AdminUser, $WriteGroupRole',
            db_record.datname
        );

        EXECUTE format(
            'GRANT CREATE ON DATABASE %I TO $AdminUser',
            db_record.datname
        );
    END LOOP;
END
`$`$;

GRANT CONNECT ON DATABASE $MaintenanceDb TO $AdminUser;
GRANT CONNECT ON DATABASE $MaintenanceDb TO $ReadGroupRole;
GRANT CONNECT ON DATABASE $MaintenanceDb TO $WriteGroupRole;

GRANT CREATE ON DATABASE $MaintenanceDb TO $AdminUser;
GRANT TEMPORARY ON DATABASE $MaintenanceDb TO $AdminUser;
GRANT TEMPORARY ON DATABASE $MaintenanceDb TO $WriteGroupRole;

\connect $MaintenanceDb

GRANT USAGE ON SCHEMA $DefaultSchema TO $AdminUser;
GRANT USAGE ON SCHEMA $DefaultSchema TO $ReadGroupRole;
GRANT USAGE ON SCHEMA $DefaultSchema TO $WriteGroupRole;

GRANT CREATE ON SCHEMA $DefaultSchema TO $AdminUser;
GRANT CREATE ON SCHEMA $DefaultSchema TO $WriteGroupRole;

GRANT SELECT
ON ALL TABLES IN SCHEMA $DefaultSchema
TO $ReadGroupRole;

GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
ON ALL TABLES IN SCHEMA $DefaultSchema
TO $WriteGroupRole;

GRANT ALL PRIVILEGES
ON ALL TABLES IN SCHEMA $DefaultSchema
TO $AdminUser;

GRANT SELECT
ON ALL SEQUENCES IN SCHEMA $DefaultSchema
TO $ReadGroupRole;

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA $DefaultSchema
TO $WriteGroupRole;

GRANT ALL PRIVILEGES
ON ALL SEQUENCES IN SCHEMA $DefaultSchema
TO $AdminUser;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT SELECT ON TABLES TO $ReadGroupRole;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
ON TABLES TO $WriteGroupRole;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT ALL PRIVILEGES ON TABLES TO $AdminUser;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT SELECT ON SEQUENCES TO $ReadGroupRole;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO $WriteGroupRole;

ALTER DEFAULT PRIVILEGES IN SCHEMA $DefaultSchema
GRANT ALL PRIVILEGES ON SEQUENCES TO $AdminUser;

ALTER DEFAULT PRIVILEGES FOR ROLE $AdminUser IN SCHEMA $DefaultSchema
GRANT SELECT ON TABLES TO $ReadGroupRole;

ALTER DEFAULT PRIVILEGES FOR ROLE $AdminUser IN SCHEMA $DefaultSchema
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
ON TABLES TO $WriteGroupRole;

ALTER DEFAULT PRIVILEGES FOR ROLE $AdminUser IN SCHEMA $DefaultSchema
GRANT SELECT ON SEQUENCES TO $ReadGroupRole;

ALTER DEFAULT PRIVILEGES FOR ROLE $AdminUser IN SCHEMA $DefaultSchema
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO $WriteGroupRole;

SELECT
    rolname,
    rolsuper,
    rolcreatedb,
    rolcreaterole,
    rolcanlogin
FROM pg_roles
WHERE rolname IN (
    '$AdminUser',
    '$ReadUser',
    '$WriteUser',
    '$ReadGroupRole',
    '$WriteGroupRole'
)
ORDER BY rolname;

SELECT
    datname AS database_name,
    has_database_privilege('$AdminUser', datname, 'CONNECT') AS admin_can_connect,
    has_database_privilege('$AdminUser', datname, 'CREATE') AS admin_can_create,
    has_database_privilege('$ReadUser', datname, 'CONNECT') AS read_can_connect,
    has_database_privilege('$WriteUser', datname, 'CONNECT') AS write_can_connect
FROM pg_database
WHERE datallowconn = TRUE
  AND datistemplate = FALSE
ORDER BY datname;

-- Optional extreme mode:
--
-- Do NOT enable this unless you intentionally want $AdminUser
-- to be a true PostgreSQL superuser.
--
--   ALTER ROLE $AdminUser WITH SUPERUSER;
--
-- To disable later:
--
--   ALTER ROLE $AdminUser WITH NOSUPERUSER;
"@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "sql\postgres_roles_setup.sql") -Content $rolesSql

# ------------------------------------------------------------
# README_POSTGRES_MCP.md
# ------------------------------------------------------------

$readme = @"
# PostgreSQL Cluster MCP Server

This project exposes PostgreSQL tools through a local MCP server.

This MCP server is designed for PostgreSQL instance / cluster-level control, not just one database.

## Existing PostgreSQL support

The setup script can detect existing PostgreSQL Windows services and psql.exe paths.

Interactive mode asks for:
- PostgreSQL host
- PostgreSQL port
- psql.exe path
- maintenance database
- default MCP target database
- PostgreSQL admin/superuser
- MCP role names and passwords
- whether to overwrite .env
- whether to install packages
- whether to apply database roles

Run interactive setup:

```powershell
.\setup_postgres_mcp.ps1
```

Run without prompts:

```powershell
.\setup_postgres_mcp.ps1 -NoInteractive -ForceEnv
```

## Roles

This setup uses three PostgreSQL login roles:

- $AdminUser
  - cluster/admin MCP role
  - can create databases, create schemas, create tables, and run admin tools
  - has CREATEDB and CREATEROLE
  - is not a superuser by default

- $ReadUser
  - read-only MCP role
  - used by postgres_read_query

- $WriteUser
  - read/write MCP role
  - used by postgres_write_execute, postgres_insert_row, postgres_update_rows, and postgres_delete_rows

## Important PostgreSQL role naming note

Custom PostgreSQL role names must not start with pg_.

PostgreSQL reserves role names beginning with pg_.

This setup uses:
- $AdminUser
- $ReadUser
- $WriteUser
- $ReadGroupRole
- $WriteGroupRole

## PyCharm setup

1. Open this folder in PyCharm.

2. Set the project interpreter to:

```text
.venv\Scripts\python.exe
```

3. Open the PyCharm Terminal.

4. Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

5. Test settings:

```powershell
python -m mcp_server.settings
```

6. Test database connections:

```powershell
python -c "from server import postgres_health_check; import pprint; pprint.pp(postgres_health_check())"
```

7. Run the server:

```powershell
python -m mcp_server.server
```

## Environment file

The setup script creates this .env.example:

```env
POSTGRES_HOST=$PostgresHost
POSTGRES_PORT=$PostgresPort

POSTGRES_MAINTENANCE_DB=$MaintenanceDb
POSTGRES_DEFAULT_DB=$DefaultDb
POSTGRES_DEFAULT_SCHEMA=$DefaultSchema

POSTGRES_ADMIN_USER=$AdminUser
POSTGRES_ADMIN_PASSWORD=$AdminPassword

POSTGRES_READ_USER=$ReadUser
POSTGRES_READ_PASSWORD=$ReadPassword

POSTGRES_WRITE_USER=$WriteUser
POSTGRES_WRITE_PASSWORD=$WritePassword

MCP_SERVER_NAME=$McpServerName
MCP_TRANSPORT=$McpTransport
MCP_HTTP_HOST=$McpHttpHost
MCP_HTTP_PORT=$McpHttpPort

MAX_READ_ROWS=$MaxReadRows
```

To force overwrite .env:

```powershell
.\setup_postgres_mcp.ps1 -ForceEnv
```

## Install Python packages

```powershell
.\setup_postgres_mcp.ps1 -InstallPythonPackages
```

## Apply PostgreSQL roles

```powershell
& "$PsqlPath" -h $PostgresHost -U $PostgresSuperUser -d $MaintenanceDb -f .\sql\postgres_roles_setup.sql
```

Or run:

```powershell
.\setup_postgres_mcp.ps1 -ApplyDatabaseRoles
```

## Verify admin role

```powershell
& "$PsqlPath" -h $PostgresHost -U $AdminUser -d $MaintenanceDb -c "SELECT current_user, current_database();"
```

Password:

```text
$AdminPassword
```

## End-to-end test

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

## MCP transport

Default transport is stdio.

For Streamable HTTP mode, update .env:

```env
MCP_TRANSPORT=streamable-http
MCP_HTTP_HOST=127.0.0.1
MCP_HTTP_PORT=8000
```

Then run:

```powershell
python -m mcp_server.server
```

HTTP MCP endpoint:

```text
http://127.0.0.1:8000/mcp
```

## Security warning

This MCP server can create and modify PostgreSQL databases, schemas, tables, and data.

Do not expose it publicly or connect it to an untrusted client without additional safeguards.
"@

Write-FileUtf8NoBom -Path (Join-Path $ProjectRoot "README_POSTGRES_MCP.md") -Content $readme

# ------------------------------------------------------------
# .gitignore
# ------------------------------------------------------------

$GitIgnorePath = Join-Path $ProjectRoot ".gitignore"

$gitIgnoreBlock = @"

# PostgreSQL MCP local files
.env
.venv/
__pycache__/
*.pyc
"@

if (Test-Path $GitIgnorePath) {
    $existingGitIgnore = Get-Content $GitIgnorePath -Raw

    if ($existingGitIgnore -notmatch "PostgreSQL MCP local files") {
        Add-Content -Path $GitIgnorePath -Value $gitIgnoreBlock
        Write-Host "Updated .gitignore"
    }
    else {
        Write-Host ".gitignore already contains PostgreSQL MCP block."
    }
}
else {
    Write-FileUtf8NoBom -Path $GitIgnorePath -Content $gitIgnoreBlock
    Write-Host "Created .gitignore"
}

# ------------------------------------------------------------
# Install packages
# ------------------------------------------------------------

if ($DoInstallPythonPackages) {
    Write-Step "Creating virtual environment and installing packages"

    if (-not (Test-Path $PythonExe)) {
        python -m venv .venv
    }
    else {
        Write-Host ".venv already exists."
    }

    & $PythonExe -m pip install --upgrade pip
    & $PipExe install -r requirements.txt
}
else {
    Write-Host ""
    Write-Warn "Skipping Python package installation."
    Write-Host "To install packages, run:"
    Write-Host "  .\setup_postgres_mcp.ps1 -InstallPythonPackages"
}

# ------------------------------------------------------------
# Apply database roles
# ------------------------------------------------------------

if ($DoApplyDatabaseRoles) {
    Write-Step "Applying PostgreSQL cluster role setup"

    if (Test-Path $PsqlPath) {
        $ResolvedPsql = $PsqlPath
    }
    else {
        $PsqlCommand = Get-Command psql -ErrorAction SilentlyContinue

        if (-not $PsqlCommand) {
            throw "psql was not found. Set -PsqlPath to your psql.exe path or add psql to PATH."
        }

        $ResolvedPsql = $PsqlCommand.Source
    }

    & $ResolvedPsql -h $PostgresHost -U $PostgresSuperUser -d $MaintenanceDb -f sql\postgres_roles_setup.sql
}
else {
    Write-Host ""
    Write-Warn "Skipping PostgreSQL role application."
    Write-Host "To apply roles manually, run:"
    Write-Host "  & `"$PsqlPath`" -h $PostgresHost -U $PostgresSuperUser -d $MaintenanceDb -f .\sql\postgres_roles_setup.sql"
}

Write-Step "Setup complete"

Write-Host "Files created or updated:"
Write-Host "  requirements.txt"
Write-Host "  .env.example"
Write-Host "  settings.py"
Write-Host "  db.py"
Write-Host "  sql_safety.py"
Write-Host "  server.py"
Write-Host "  sql\postgres_roles_setup.sql"
Write-Host "  README_POSTGRES_MCP.md"
Write-Host "  .gitignore"

if ((Test-Path $EnvPath) -and (-not $DoForceEnv)) {
    Write-Host ""
    Write-Warn ".env was preserved because it already existed."
    Write-Host "Use -ForceEnv if you want this script to overwrite .env."
}

Write-Host ""
Write-Host "Recommended next commands:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host "  python -m mcp_server.settings"
Write-Host "  python -c `"from server import postgres_health_check; import pprint; pprint.pp(postgres_health_check())`""
Write-Host ""
Write-Host "To run the MCP server:"
Write-Host "  python -m mcp_server.server"


