from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# Base paths
# ---------------------------------------------------------
SERVICE_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path(r"E:\emtac\dev_env\.env")
ENV_PATH = Path(os.getenv("EMTAC_ENV_PATH", str(DEFAULT_ENV_PATH)))


if not ENV_PATH.exists():
    raise FileNotFoundError(f"Environment file not found: {ENV_PATH}")


# ---------------------------------------------------------
# Load shared EMTAC environment once
# ---------------------------------------------------------
load_dotenv(ENV_PATH, override=False)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value == "":
        return default
    return value


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def split_command(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return value.split()


# ---------------------------------------------------------
# Dashboard host/port
# ---------------------------------------------------------
SERVICE_DASHBOARD_HOST: str = get_env("SERVICE_DASHBOARD_HOST", "127.0.0.1")
SERVICE_DASHBOARD_PORT: int = int(get_env("SERVICE_DASHBOARD_PORT", "5100"))


# ---------------------------------------------------------
# GPU service config
# ---------------------------------------------------------
SERVICE_GPU_PYTHON: Path = as_path(get_env("SERVICE_GPU_PYTHON"))
SERVICE_GPU_CWD: Path = as_path(get_env("SERVICE_GPU_CWD"))
SERVICE_GPU_ENTRY: str = get_env("SERVICE_GPU_ENTRY")
SERVICE_GPU_BASE_URL: str = get_env("SERVICE_GPU_BASE_URL", "http://127.0.0.1:5051")


# ---------------------------------------------------------
# AI Gateway service config
# ---------------------------------------------------------
SERVICE_AI_GATEWAY_PYTHON: Path = as_path(get_env("SERVICE_AI_GATEWAY_PYTHON"))
SERVICE_AI_GATEWAY_CWD: Path = as_path(get_env("SERVICE_AI_GATEWAY_CWD"))
SERVICE_AI_GATEWAY_ENTRY: str = get_env("SERVICE_AI_GATEWAY_ENTRY")
SERVICE_AI_GATEWAY_BASE_URL: str = get_env(
    "SERVICE_AI_GATEWAY_BASE_URL",
    "http://127.0.0.1:9000",
)


# ---------------------------------------------------------
# MCP Coordinator service config
# ---------------------------------------------------------
SERVICE_MCP_COORDINATOR_PYTHON: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_PYTHON",
        r"E:\emtac\services\.venv_services\Scripts\python.exe",
    )
)

SERVICE_MCP_COORDINATOR_CWD: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_CWD",
        r"E:\emtac\services\mcp_server",
    )
)

SERVICE_MCP_COORDINATOR_ENTRY: str = get_env(
    "SERVICE_MCP_COORDINATOR_ENTRY",
    "-m listed_server.mcp_coordinator.server",
)

SERVICE_MCP_COORDINATOR_BASE_URL: str = get_env(
    "SERVICE_MCP_COORDINATOR_BASE_URL",
    "http://127.0.0.1:9100",
)

SERVICE_MCP_COORDINATOR_HEALTH_URL: str = get_env(
    "SERVICE_MCP_COORDINATOR_HEALTH_URL",
    SERVICE_MCP_COORDINATOR_BASE_URL,
)

SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32",
        r"E:\emtac\services\.venv_services\Lib\site-packages\pywin32_system32",
    )
)

SERVICE_MCP_COORDINATOR_WIN32_DIR: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_WIN32_DIR",
        r"E:\emtac\services\.venv_services\Lib\site-packages\win32",
    )
)

SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR",
        r"E:\emtac\services\.venv_services\Lib\site-packages\win32\lib",
    )
)

SERVICE_MCP_COORDINATOR_SITE_PACKAGES: Path = as_path(
    get_env(
        "SERVICE_MCP_COORDINATOR_SITE_PACKAGES",
        r"E:\emtac\services\.venv_services\Lib\site-packages",
    )
)


# ---------------------------------------------------------
# MCP AI Layer REST API config
# ---------------------------------------------------------
SERVICE_MCP_AI_LAYER_PYTHON: Path = as_path(
    get_env(
        "SERVICE_MCP_AI_LAYER_PYTHON",
        r"E:\emtac\services\.venv_services\Scripts\python.exe",
    )
)

SERVICE_MCP_AI_LAYER_CWD: Path = as_path(
    get_env(
        "SERVICE_MCP_AI_LAYER_CWD",
        r"E:\emtac\services\mcp_server",
    )
)

SERVICE_MCP_AI_LAYER_ENTRY: str = get_env(
    "SERVICE_MCP_AI_LAYER_ENTRY",
    "-m ai_layer.ai_rest_app",
)

SERVICE_MCP_AI_LAYER_BASE_URL: str = get_env(
    "SERVICE_MCP_AI_LAYER_BASE_URL",
    "http://127.0.0.1:9200",
)

SERVICE_MCP_AI_LAYER_CHAT_URL: str = get_env(
    "SERVICE_MCP_AI_LAYER_CHAT_URL",
    f"{SERVICE_MCP_AI_LAYER_BASE_URL.rstrip('/')}/api/ai/chat",
)

SERVICE_MCP_AI_LAYER_HEALTH_URL: str = get_env(
    "SERVICE_MCP_AI_LAYER_HEALTH_URL",
    f"{SERVICE_MCP_AI_LAYER_BASE_URL.rstrip('/')}/api/ai/health",
)

SERVICE_MCP_AI_LAYER_TOOLS_URL: str = get_env(
    "SERVICE_MCP_AI_LAYER_TOOLS_URL",
    f"{SERVICE_MCP_AI_LAYER_BASE_URL.rstrip('/')}/api/ai/tools",
)


# ---------------------------------------------------------
# PostgreSQL service control config
# ---------------------------------------------------------
POSTGRES_BIN_DIR: Path = as_path(get_env("POSTGRES_BIN_DIR"))
POSTGRES_DATA_DIR: Path = as_path(get_env("POSTGRES_DATA_DIR"))
POSTGRES_LOG_FILE: Path = POSTGRES_DATA_DIR / "server.log"
PG_CTL_EXE: Path = POSTGRES_BIN_DIR / "pg_ctl.exe"


# ---------------------------------------------------------
# Grafana
# ---------------------------------------------------------
GRAFANA_URL: str = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_CWD: Path = Path(os.getenv("GRAFANA_CWD", r"E:\emtac\services\grafana-12.3.1"))
GRAFANA_EXE: Path = GRAFANA_CWD / "bin" / "grafana.exe"

GRAFANA_API_KEY: str | None = get_optional_env("GRAFANA_API_KEY")
GRAFANA_USER: str | None = get_optional_env("GRAFANA_USER", "admin")
GRAFANA_PASSWORD: str | None = get_optional_env("GRAFANA_PASSWORD")


# ---------------------------------------------------------
# Optional application DB connection values
# ---------------------------------------------------------
POSTGRES_USER: str | None = get_optional_env("POSTGRES_USER")
POSTGRES_PASSWORD: str | None = get_optional_env("POSTGRES_PASSWORD")
POSTGRES_HOST: str | None = get_optional_env("POSTGRES_HOST")
POSTGRES_PORT: str | None = get_optional_env("POSTGRES_PORT")
POSTGRES_DB: str | None = get_optional_env("POSTGRES_DB")
DATABASE_URL: str | None = get_optional_env("DATABASE_URL")
SQLITE_DB_PATH: str | None = get_optional_env("SQLITE_DB_PATH")


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
if not SERVICE_GPU_PYTHON.exists():
    raise FileNotFoundError(f"SERVICE_GPU_PYTHON not found: {SERVICE_GPU_PYTHON}")

if not SERVICE_GPU_CWD.exists():
    raise FileNotFoundError(f"SERVICE_GPU_CWD not found: {SERVICE_GPU_CWD}")

gpu_launcher_path = SERVICE_GPU_CWD / SERVICE_GPU_ENTRY
if not gpu_launcher_path.exists():
    raise FileNotFoundError(f"SERVICE_GPU_ENTRY not found: {gpu_launcher_path}")

if not SERVICE_AI_GATEWAY_PYTHON.exists():
    raise FileNotFoundError(
        f"SERVICE_AI_GATEWAY_PYTHON not found: {SERVICE_AI_GATEWAY_PYTHON}"
    )

if not SERVICE_AI_GATEWAY_CWD.exists():
    raise FileNotFoundError(
        f"SERVICE_AI_GATEWAY_CWD not found: {SERVICE_AI_GATEWAY_CWD}"
    )

ai_gateway_launcher_path = SERVICE_AI_GATEWAY_CWD / SERVICE_AI_GATEWAY_ENTRY
if not ai_gateway_launcher_path.exists():
    raise FileNotFoundError(
        f"SERVICE_AI_GATEWAY_ENTRY not found: {ai_gateway_launcher_path}"
    )

if not SERVICE_MCP_COORDINATOR_PYTHON.exists():
    raise FileNotFoundError(
        f"SERVICE_MCP_COORDINATOR_PYTHON not found: {SERVICE_MCP_COORDINATOR_PYTHON}"
    )

if not SERVICE_MCP_COORDINATOR_CWD.exists():
    raise FileNotFoundError(
        f"SERVICE_MCP_COORDINATOR_CWD not found: {SERVICE_MCP_COORDINATOR_CWD}"
    )

if not SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32.exists():
    raise FileNotFoundError(
        "SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32 not found: "
        f"{SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32}"
    )

if not SERVICE_MCP_COORDINATOR_WIN32_DIR.exists():
    raise FileNotFoundError(
        f"SERVICE_MCP_COORDINATOR_WIN32_DIR not found: "
        f"{SERVICE_MCP_COORDINATOR_WIN32_DIR}"
    )

if not SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR.exists():
    raise FileNotFoundError(
        "SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR not found: "
        f"{SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR}"
    )

if not SERVICE_MCP_COORDINATOR_SITE_PACKAGES.exists():
    raise FileNotFoundError(
        "SERVICE_MCP_COORDINATOR_SITE_PACKAGES not found: "
        f"{SERVICE_MCP_COORDINATOR_SITE_PACKAGES}"
    )

if not SERVICE_MCP_AI_LAYER_PYTHON.exists():
    raise FileNotFoundError(
        f"SERVICE_MCP_AI_LAYER_PYTHON not found: {SERVICE_MCP_AI_LAYER_PYTHON}"
    )

if not SERVICE_MCP_AI_LAYER_CWD.exists():
    raise FileNotFoundError(
        f"SERVICE_MCP_AI_LAYER_CWD not found: {SERVICE_MCP_AI_LAYER_CWD}"
    )

if not POSTGRES_BIN_DIR.exists():
    raise FileNotFoundError(f"POSTGRES_BIN_DIR not found: {POSTGRES_BIN_DIR}")

if not POSTGRES_DATA_DIR.exists():
    raise FileNotFoundError(f"POSTGRES_DATA_DIR not found: {POSTGRES_DATA_DIR}")

if not PG_CTL_EXE.exists():
    raise FileNotFoundError(f"pg_ctl.exe not found: {PG_CTL_EXE}")

if not GRAFANA_CWD.exists():
    raise FileNotFoundError(f"GRAFANA_CWD not found: {GRAFANA_CWD}")

if not GRAFANA_EXE.exists():
    raise FileNotFoundError(f"GRAFANA_EXE not found: {GRAFANA_EXE}")