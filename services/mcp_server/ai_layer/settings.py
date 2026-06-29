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

    hf_token: str
    hf_router_base_url: str
    hf_router_model: str

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
    project_root = Path(__file__).resolve().parents[1]

    return AiSettings(
        project_root=project_root,

        ai_provider=env_str("AI_PROVIDER", "openai"),

        openai_api_key=env_str("OPENAI_API_KEY", ""),
        openai_model=env_str("OPENAI_MODEL", "gpt-5.5"),

        hf_token=env_str("HF_TOKEN", ""),
        hf_router_base_url=env_str("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1"),
        hf_router_model=env_str(
            "HF_ROUTER_MODEL",
            "Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
        ),

        mcp_server_python=env_str("MCP_SERVER_PYTHON", r".\.venv\Scripts\python.exe"),
        mcp_server_script=env_str("MCP_SERVER_SCRIPT", r".\mcp_server\server.py"),

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
    print(f"  HF_ROUTER_MODEL={settings.hf_router_model}")
    print(f"  HF_TOKEN_SET={bool(settings.hf_token)}")
    print(f"  MCP_SERVER_PYTHON={settings.resolved_mcp_python()}")
    print(f"  MCP_SERVER_SCRIPT={settings.resolved_mcp_script()}")
    print(f"  AI_DEFAULT_DATABASE={settings.ai_default_database}")
    print(
        "  AI_REQUIRE_APPROVAL_FOR_DESTRUCTIVE_TOOLS="
        f"{settings.require_approval_for_destructive_tools}"
    )
    print(f"  AI_MAX_TOOL_ROUNDS={settings.max_tool_rounds}")
