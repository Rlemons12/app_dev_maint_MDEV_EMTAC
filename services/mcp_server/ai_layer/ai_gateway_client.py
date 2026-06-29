from __future__ import annotations

import os
from typing import Any

import requests


class AIGatewayClient:
    """
    Client for the local OpenAI-compatible AI Gateway.

    This does not execute tools. It only sends chat-completion payloads
    to the local model gateway.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.endpoint = (
            endpoint
            or os.getenv(
                "AI_LOCAL_ENDPOINT",
                "http://127.0.0.1:9000/v1/chat/completions",
            )
        )
        self.timeout_seconds = int(
            timeout_seconds
            or os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "120")
        )

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            self.endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def health_url(self) -> str:
        if "/v1/chat/completions" in self.endpoint:
            return self.endpoint.replace("/v1/chat/completions", "/health")
        return self.endpoint.rstrip("/") + "/health"