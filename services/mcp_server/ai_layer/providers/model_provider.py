from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from ai_layer.settings import AiSettings


class ChatCompletionsClient(Protocol):
    def create(self, **kwargs): ...


@dataclass(frozen=True)
class ProviderConfig:
    provider_name: str
    model: str
    chat_completions: ChatCompletionsClient


def create_provider_config(
    settings: AiSettings,
    provider: str | None = None,
    model: str | None = None,
) -> ProviderConfig:
    provider_name = (provider or settings.ai_provider).strip().lower()

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env before using the AI gateway."
            )
        client = OpenAI(api_key=settings.openai_api_key)
        return ProviderConfig(
            provider_name="openai",
            model=model or settings.openai_model,
            chat_completions=client.chat.completions,
        )

    if provider_name in {
        "huggingface-router",
        "huggingface_router",
        "hf-router",
        "hf_router",
    }:
        if not settings.hf_token:
            raise ValueError(
                "HF_TOKEN is not set. Add it to .env before using Hugging Face Router."
            )
        client = OpenAI(
            base_url=settings.hf_router_base_url,
            api_key=settings.hf_token,
        )
        return ProviderConfig(
            provider_name="huggingface-router",
            model=model or settings.hf_router_model,
            chat_completions=client.chat.completions,
        )

    raise ValueError(
        "AI_PROVIDER must be openai or huggingface-router."
    )
