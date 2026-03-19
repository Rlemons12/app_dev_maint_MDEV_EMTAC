from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------
ENV_PATH = Path(r"E:\emtac\dev_env\.env")
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="EMTAC PyCharm MCP Gateway")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5050").rstrip("/")

# Gateway-facing model alias for OpenAI-compatible clients like PyCharm
MODEL_NAME = os.getenv("EMTAC_LOCAL_MODEL_NAME", "emtac-gpu-qwen")

# Actual GPU service generation model to use when a request does not specify one
DEFAULT_GPU_MODEL = os.getenv(
    "MCP_DEFAULT_GPU_MODEL",
    "qwen2.5-coder-7b-instruct",
).strip()

GPU_TIMEOUT = int(os.getenv("MCP_GPU_TIMEOUT", "300"))
GPU_MAX_RETRIES = int(os.getenv("MCP_GPU_MAX_RETRIES", "2"))

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "MCP_DEFAULT_SYSTEM_PROMPT",
    (
        "You are a careful coding assistant.\n"
        "Follow the user's requested task exactly.\n"
        "Return concise, task-specific output.\n"
        "Do not greet the user unless explicitly asked.\n"
        "For commit message generation, return only the commit message."
    ),
)

COMMIT_SYSTEM_PROMPT = os.getenv(
    "MCP_COMMIT_SYSTEM_PROMPT",
    (
        "Generate a git commit message from the provided diff.\n"
        "Return only the commit message text.\n"
        "Use imperative mood.\n"
        "Prefer a short subject line.\n"
        "Optionally include a blank line and 1-3 short body lines if needed.\n"
        "Do not add quotes.\n"
        "Do not add labels.\n"
        "Do not explain your reasoning.\n"
        "Do not greet the user.\n"
    ),
)

# Optional extra model aliases to advertise through /v1/models
ADVERTISED_MODELS = [
    MODEL_NAME,
    DEFAULT_GPU_MODEL,
    "qwen7b",
    "qwen-coder-7b",
    "qwen2.5-coder-7b-instruct",
]

# ---------------------------------------------------------
# GPU Service Client
# ---------------------------------------------------------
class GPUServiceClient:
    """
    Thin client used by the MCP gateway to communicate with the GPU service.

    Responsibilities:
    - health check
    - POST retry handling
    - generation forwarding
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        max_retries: int = 2,
        default_generation_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_generation_model = (
            (default_generation_model or "").strip()
            or DEFAULT_GPU_MODEL
        )

        logger.info(
            "GPUServiceClient initialized | base_url=%s | timeout=%s | max_retries=%s | default_generation_model=%s",
            self.base_url,
            self.timeout,
            self.max_retries,
            self.default_generation_model,
        )

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("GPU service health check failed: %s", exc)
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("POST %s | attempt=%s", endpoint, attempt)

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "POST failed | endpoint=%s | attempt=%s | error=%s",
                    endpoint,
                    attempt,
                    exc,
                )
                time.sleep(1)

        raise RuntimeError(f"GPU service request failed for {endpoint}: {last_error}")

    def generate_code(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ) -> str:
        resolved_model = (model or "").strip() or self.default_generation_model

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": resolved_model,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        logger.info(
            "Forwarding generate_code request | model=%s | max_tokens=%s | prompt_len=%s",
            resolved_model,
            max_new_tokens,
            len(prompt),
        )

        data = self._post("/mcp/tools/generate_code", payload)

        text = data.get("text") or data.get("output")
        if not text:
            raise RuntimeError("GPU service returned empty generation result")

        return str(text).strip()


gpu_client = GPUServiceClient(
    base_url=GPU_SERVICE_URL,
    timeout=GPU_TIMEOUT,
    max_retries=GPU_MAX_RETRIES,
    default_generation_model=DEFAULT_GPU_MODEL,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def content_to_text(content: Any) -> str:
    """
    Normalize OpenAI-style content into plain text.

    Supports:
    - plain strings
    - lists of content parts
    - dict-based content
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []

        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue

            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()

                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
                    continue

                for key in ("text", "content", "value"):
                    if key in item:
                        text = str(item.get(key, "")).strip()
                        if text:
                            parts.append(text)
                            break

        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return str(content.get(key, "")).strip()

    return str(content).strip()


def extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            text = content_to_text(message.get("content"))
            if text:
                return text
    return ""


def extract_system_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        if role == "system":
            text = content_to_text(message.get("content"))
            if text:
                return text
    return None


def build_forward_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Build a richer prompt from the full incoming message list
    instead of only forwarding the last user message.
    """
    parts: List[str] = []

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        text = content_to_text(message.get("content"))

        if not text:
            continue

        if role == "system":
            parts.append(f"SYSTEM:\n{text}")
        elif role == "user":
            parts.append(f"USER:\n{text}")
        elif role == "assistant":
            parts.append(f"ASSISTANT:\n{text}")
        else:
            parts.append(f"{role.upper()}:\n{text}")

    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts).strip()


def looks_like_commit_request(messages: List[Dict[str, Any]]) -> bool:
    combined_text_parts: List[str] = []

    for message in messages:
        text = content_to_text(message.get("content"))
        if text:
            combined_text_parts.append(text.lower())

    combined_text = "\n".join(combined_text_parts)

    return (
        "commit" in combined_text
        and (
            "message" in combined_text
            or "git" in combined_text
            or "diff" in combined_text
            or "changes" in combined_text
            or "vcs" in combined_text
        )
    )


def resolve_requested_model(requested_model: Optional[str]) -> str:
    """
    Resolve the incoming OpenAI-compatible model name into the actual GPU model name.

    Rules:
    - If missing, use DEFAULT_GPU_MODEL
    - If the request uses the gateway alias MODEL_NAME, map to DEFAULT_GPU_MODEL
    - Otherwise pass the requested model through directly
    """
    value = (requested_model or "").strip()

    if not value:
        return DEFAULT_GPU_MODEL

    if value == MODEL_NAME:
        return DEFAULT_GPU_MODEL

    return value


def build_chat_completion_response(content: str, model_name: str) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


async def build_streaming_response(content: str, model_name: str):
    chunk_1 = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(chunk_1)}\n\n"
    await asyncio.sleep(0.05)

    chunk_2 = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(chunk_2)}\n\n"
    yield "data: [DONE]\n\n"


def coerce_mcp_content_to_text(content: Any) -> str:
    """
    Normalize /mcp content field into text.
    """
    return content_to_text(content)


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Server starting")
    logger.info("Using .env file: %s", ENV_PATH)
    logger.info("GPU_SERVICE_URL: %s", GPU_SERVICE_URL)
    logger.info("Configured model name: %s", MODEL_NAME)
    logger.info("Default GPU model: %s", DEFAULT_GPU_MODEL)

    if gpu_client.health():
        logger.info("GPU service is available")
    else:
        logger.warning("GPU service is not reachable at startup")


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gateway": "emtac_pycharm_mcp",
        "gpu_service_url": GPU_SERVICE_URL,
        "gpu_service_available": gpu_client.health(),
        "model_name": MODEL_NAME,
        "default_gpu_model": DEFAULT_GPU_MODEL,
    }


@app.get("/sse")
async def sse():
    async def stream():
        logger.info("PyCharm connected to MCP SSE stream")
        while True:
            await asyncio.sleep(20)
            yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/mcp")
async def mcp_chat(request: Request):
    payload = await request.json()

    logger.info("MCP request received")
    logger.info("Payload keys: %s", list(payload.keys()))

    content = coerce_mcp_content_to_text(payload.get("content", ""))
    if not content:
        content = "Hello"

    requested_model = str(payload.get("model", "")).strip()
    resolved_model = resolve_requested_model(requested_model)

    try:
        reply = gpu_client.generate_code(
            prompt=content,
            model=resolved_model,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        return {"content": reply}

    except Exception as exc:
        logger.exception("MCP request failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"MCP request failed: {exc}"},
        )


@app.get("/v1/models")
async def list_models():
    seen = set()
    data = []

    for model_id in ADVERTISED_MODELS:
        model_id = str(model_id).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        data.append(
            {
                "id": model_id,
                "object": "model",
                "owned_by": "emtac-gpu-service",
            }
        )

    return {
        "object": "list",
        "data": data,
    }


def build_commit_prompt(messages: List[Dict[str, Any]]) -> str:
    diff_text = ""
    history_text = ""
    message_text = ""

    for message in messages:
        text = content_to_text(message.get("content"))
        if not text:
            continue

        lower = text.lower()

        if "[diff]" in lower:
            diff_text = text
        elif "[commits history]" in lower:
            history_text = text
        elif "[message]" in lower:
            message_text = text

    parts = []

    if diff_text:
        parts.append(diff_text[:8000])

    if history_text:
        parts.append(history_text[:2000])

    if message_text:
        parts.append(message_text[:500])

    return "\n\n".join(parts).strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()

    logger.info("Received /v1/chat/completions request")
    logger.info("Payload keys: %s", list(payload.keys()))

    messages = payload.get("messages", [])
    stream = bool(payload.get("stream", False))

    requested_model = str(payload.get("model", "")).strip() or MODEL_NAME
    resolved_model = resolve_requested_model(requested_model)

    max_tokens = int(payload.get("max_tokens", 512))
    temperature = float(payload.get("temperature", 0.2))
    top_p = float(payload.get("top_p", 0.95))

    if not isinstance(messages, list):
        logger.warning("Invalid messages payload type: %s", type(messages).__name__)
        messages = []

    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "")).strip().lower()
        text = content_to_text(msg.get("content"))
        logger.info(
            "Message[%s] | role=%s | chars=%s | preview=%r",
            idx,
            role,
            len(text),
            text[:300],
        )

    logger.info(
        "Model routing | requested_model=%s | resolved_gpu_model=%s",
        requested_model,
        resolved_model,
    )

    system_prompt = extract_system_message(messages) or DEFAULT_SYSTEM_PROMPT
    prompt = build_forward_prompt(messages)

    if looks_like_commit_request(messages):
        logger.info("Detected commit-message style request")
        system_prompt = COMMIT_SYSTEM_PROMPT
        prompt = build_commit_prompt(messages)
        temperature = 0.0
        top_p = 1.0
        max_tokens = min(max_tokens, 95)

        logger.info(
            "Commit prompt prepared | chars=%s | preview=%r",
            len(prompt),
            prompt[:500],
        )

    if not prompt.strip():
        reply_text = "No valid user message was provided."
    else:
        try:
            reply_text = gpu_client.generate_code(
                prompt=prompt,
                model=resolved_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            logger.exception("Chat completion failed: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": f"Chat completion failed: {exc}"},
            )

    reply_text = (reply_text or "").strip()

    logger.info(
        "Returning chat completion | requested_model=%s | reply_len=%s | preview=%r",
        requested_model,
        len(reply_text),
        reply_text[:300],
    )

    if stream:
        return StreamingResponse(
            build_streaming_response(reply_text, model_name=requested_model),
            media_type="text/event-stream",
        )

    return JSONResponse(
        content=build_chat_completion_response(reply_text, model_name=requested_model)
    )