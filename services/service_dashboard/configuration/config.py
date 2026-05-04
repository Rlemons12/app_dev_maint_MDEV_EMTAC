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


# ---------------------------------------------------------
# Dashboard host/port
# ---------------------------------------------------------
SERVICE_DASHBOARD_HOST: str = get_env("SERVICE_DASHBOARD_HOST", "127.0.0.1")
SERVICE_DASHBOARD_PORT: int = int(get_env("SERVICE_DASHBOARD_PORT", "5000"))


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
SERVICE_AI_GATEWAY_BASE_URL: str = get_env("SERVICE_AI_GATEWAY_BASE_URL", "http://127.0.0.1:9000")


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
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_CWD = Path(os.getenv("GRAFANA_CWD", r"E:\emtac\services\grafana-12.3.1"))
GRAFANA_EXE = GRAFANA_CWD / "bin" / "grafana.exe"

# Grafana API auth — only needed for dashboards / datasources / alert rules.
# Prefer a service-account API key; basic auth is a fallback.
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
    raise FileNotFoundError(f"SERVICE_AI_GATEWAY_PYTHON not found: {SERVICE_AI_GATEWAY_PYTHON}")

if not SERVICE_AI_GATEWAY_CWD.exists():
    raise FileNotFoundError(f"SERVICE_AI_GATEWAY_CWD not found: {SERVICE_AI_GATEWAY_CWD}")

ai_gateway_launcher_path = SERVICE_AI_GATEWAY_CWD / SERVICE_AI_GATEWAY_ENTRY
if not ai_gateway_launcher_path.exists():
    raise FileNotFoundError(f"SERVICE_AI_GATEWAY_ENTRY not found: {ai_gateway_launcher_path}")

if not POSTGRES_BIN_DIR.exists():
    raise FileNotFoundError(f"POSTGRES_BIN_DIR not found: {POSTGRES_BIN_DIR}")

if not POSTGRES_DATA_DIR.exists():
    raise FileNotFoundError(f"POSTGRES_DATA_DIR not found: {POSTGRES_DATA_DIR}")

if not PG_CTL_EXE.exists():
    raise FileNotFoundError(f"pg_ctl.exe not found: {PG_CTL_EXE}")