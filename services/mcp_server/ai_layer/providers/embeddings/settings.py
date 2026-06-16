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


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return int(value.strip())


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return None

    return int(value.strip())


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int | None
    timeout_seconds: int

    openai_api_key: str
    openai_base_url: str

    ollama_base_url: str

    huggingface_cache_dir: str
    huggingface_token: str
    huggingface_inference_provider: str


def get_embedding_settings() -> EmbeddingSettings:
    provider = env_str("EMBEDDING_PROVIDER", "openai").lower()

    return EmbeddingSettings(
        provider=provider,
        model=env_str("EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=env_optional_int("EMBEDDING_DIMENSIONS"),
        timeout_seconds=env_int("EMBEDDING_TIMEOUT_SECONDS", 60),
        openai_api_key=env_str("OPENAI_API_KEY", ""),
        openai_base_url=env_str("OPENAI_BASE_URL", ""),
        ollama_base_url=env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        huggingface_cache_dir=env_str(
            "HF_HUB_CACHE",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ),
        huggingface_token=env_str("HF_TOKEN", ""),
        huggingface_inference_provider=env_str(
            "HF_INFERENCE_PROVIDER",
            "hf-inference",
        ),
    )
