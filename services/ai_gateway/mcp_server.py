from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------
# Load shared EMTAC .env
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
AI_GATEWAY_DIR = CURRENT_FILE.parent
SERVICES_DIR = AI_GATEWAY_DIR.parent
PROJECT_ROOT = SERVICES_DIR.parent

ENV_PATH = Path(os.getenv("EMTAC_ENV_PATH", r"E:\emtac\dev_env\.env"))
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("emtac.ai_gateway")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
SERVICE_NAME = "EMTAC AI Gateway"

# Prefer the newer service variable first.
GPU_SERVICE_URL = (
    os.getenv("SERVICE_GPU_BASE_URL", "").strip()
    or os.getenv("GPU_SERVICE_URL", "").strip()
    or "http://127.0.0.1:5051"
).rstrip("/")

# Name advertised to OpenAI-compatible clients.
MODEL_NAME = os.getenv("EMTAC_LOCAL_MODEL_NAME", "emtac-gpu-qwen").strip()

# Actual model name sent to the GPU service.
DEFAULT_GPU_MODEL = (
    os.getenv("MCP_DEFAULT_GPU_MODEL", "").strip()
    or os.getenv("PYCHARM_GPU_DEFAULT_MODEL", "").strip()
    or "qwen2.5-coder-7b-instruct"
)

GPU_TIMEOUT = int(os.getenv("MCP_GPU_TIMEOUT", "300"))
GPU_MAX_RETRIES = int(os.getenv("MCP_GPU_MAX_RETRIES", "2"))

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "MCP_DEFAULT_SYSTEM_PROMPT",
    (
        "You are a careful coding assistant.\n"
        "Follow the user's requested task exactly.\n"
        "Return concise, task-specific output.\n"
        "Do not greet the user unless explicitly asked.\n"
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
    ),
)

ADVERTISED_MODELS = [
    MODEL_NAME,
    DEFAULT_GPU_MODEL,
    "qwen7b",
    "qwen-coder-7b",
    "qwen2.5-coder-7b-instruct",
]

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title=SERVICE_NAME,
    version="1.0.0",
    description="OpenAI-compatible local AI gateway for EMTAC.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# GPU client
# ---------------------------------------------------------
class GPUServiceClient:
    """
    Thin client for the local EMTAC GPU service.

    Expected GPU endpoint:
        POST /mcp/tools/generate_code

    Expected response:
        {"text": "..."}
    or:
        {"output": "..."}
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: int = 300,
        max_retries: int = 2,
        default_generation_model: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_generation_model = default_generation_model

        logger.info(
            "GPUServiceClient initialized | base_url=%s | timeout=%s | retries=%s | default_model=%s",
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
            logger.warning("GPU health check failed: %s", exc)
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("POST %s | attempt=%s", url, attempt)

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code >= 400:
                    logger.warning(
                        "GPU service returned error | status=%s | body=%s",
                        response.status_code,
                        response.text[:1000],
                    )

                response.raise_for_status()
                return response.json()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "GPU request failed | endpoint=%s | attempt=%s | error=%s",
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
            "Forwarding generation request | model=%s | max_tokens=%s | temperature=%s | top_p=%s | prompt_chars=%s",
            resolved_model,
            max_new_tokens,
            temperature,
            top_p,
            len(prompt),
        )

        data = self._post("/mcp/tools/generate_code", payload)

        text = data.get("text") or data.get("output") or data.get("content")
        if not text:
            raise RuntimeError(f"GPU service returned empty generation result: {data}")

        return str(text).strip()


gpu_client = GPUServiceClient(
    base_url=GPU_SERVICE_URL,
    timeout=GPU_TIMEOUT,
    max_retries=GPU_MAX_RETRIES,
    default_generation_model=DEFAULT_GPU_MODEL,
)

# ---------------------------------------------------------
# OpenAI content helpers
# ---------------------------------------------------------
def content_to_text(content: Any) -> str:
    """
    Converts OpenAI-style message content into plain text.

    Supports:
    - "plain string"
    - [{"type": "text", "text": "..."}]
    - {"text": "..."}
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []

        for item in content:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    parts.append(value)
                continue

            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()

                if item_type == "text":
                    value = str(item.get("text", "")).strip()
                    if value:
                        parts.append(value)
                    continue

                for key in ("text", "content", "value"):
                    if key in item:
                        value = str(item.get(key, "")).strip()
                        if value:
                            parts.append(value)
                        break

        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return str(content.get(key, "")).strip()

    return str(content).strip()


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
    Builds a plain prompt from OpenAI-style messages.

    The GPU service currently expects a simple prompt string, not a full
    OpenAI messages array, so this preserves roles in text form.
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
        elif role == "tool":
            parts.append(f"TOOL:\n{text}")
        else:
            parts.append(f"{role.upper() or 'MESSAGE'}:\n{text}")

    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts).strip()


def looks_like_commit_request(messages: List[Dict[str, Any]]) -> bool:
    combined_parts: List[str] = []

    for message in messages:
        text = content_to_text(message.get("content"))
        if text:
            combined_parts.append(text.lower())

    combined = "\n".join(combined_parts)

    return (
        "commit" in combined
        and (
            "message" in combined
            or "git" in combined
            or "diff" in combined
            or "changes" in combined
            or "vcs" in combined
        )
    )


def build_commit_prompt(messages: List[Dict[str, Any]]) -> str:
    diff_text = ""
    history_text = ""
    message_text = ""
    fallback_parts: List[str] = []

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
        else:
            fallback_parts.append(text)

    parts: List[str] = []

    if diff_text:
        parts.append(diff_text[:12000])

    if history_text:
        parts.append(history_text[:3000])

    if message_text:
        parts.append(message_text[:1000])

    if not parts:
        parts.extend(fallback_parts)

    return "\n\n".join(parts).strip()


def resolve_requested_model(requested_model: Optional[str]) -> str:
    value = (requested_model or "").strip()

    if not value:
        return DEFAULT_GPU_MODEL

    if value == MODEL_NAME:
        return DEFAULT_GPU_MODEL

    return value


def build_chat_completion_response(
    *,
    content: str,
    requested_model: str,
) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-local-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
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
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def build_streaming_response(content: str, requested_model: str):
    chunk_id = f"chatcmpl-local-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": requested_model,
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

    yield f"data: {json.dumps(first_chunk)}\n\n"
    await asyncio.sleep(0.05)

    done_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting %s", SERVICE_NAME)
    logger.info("CURRENT_FILE=%s", CURRENT_FILE)
    logger.info("PROJECT_ROOT=%s", PROJECT_ROOT)
    logger.info("ENV_PATH=%s | exists=%s", ENV_PATH, ENV_PATH.exists())
    logger.info("GPU_SERVICE_URL=%s", GPU_SERVICE_URL)
    logger.info("MODEL_NAME=%s", MODEL_NAME)
    logger.info("DEFAULT_GPU_MODEL=%s", DEFAULT_GPU_MODEL)

    if gpu_client.health():
        logger.info("GPU service is available")
    else:
        logger.warning("GPU service is not reachable at startup")


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "gateway": "ai_gateway",
        "gpu_service_url": GPU_SERVICE_URL,
        "gpu_service_available": gpu_client.health(),
        "model_name": MODEL_NAME,
        "default_gpu_model": DEFAULT_GPU_MODEL,
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
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
                "created": 0,
                "owned_by": "emtac-gpu-service",
            }
        )

    return {
        "object": "list",
        "data": data,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON body: {exc}",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )

    logger.info("Received /v1/chat/completions request")
    logger.info("Payload keys: %s", list(payload.keys()))

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(
            status_code=400,
            detail="'messages' must be a list.",
        )

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="'messages' cannot be empty.",
        )

    requested_model = str(payload.get("model", "")).strip() or MODEL_NAME
    resolved_gpu_model = resolve_requested_model(requested_model)

    stream = bool(payload.get("stream", False))
    max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 512)))
    temperature = float(payload.get("temperature", 0.2))
    top_p = float(payload.get("top_p", 0.95))

    for idx, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        text = content_to_text(message.get("content"))
        logger.info(
            "Message[%s] | role=%s | chars=%s | preview=%r",
            idx,
            role,
            len(text),
            text[:300],
        )

    system_prompt = extract_system_message(messages) or DEFAULT_SYSTEM_PROMPT
    prompt = build_forward_prompt(messages)

    if looks_like_commit_request(messages):
        logger.info("Detected commit-message request")
        system_prompt = COMMIT_SYSTEM_PROMPT
        prompt = build_commit_prompt(messages)
        temperature = 0.0
        top_p = 1.0
        max_tokens = min(max_tokens, 120)

    if not prompt.strip():
        reply_text = "No valid user message was provided."
    else:
        try:
            reply_text = gpu_client.generate_code(
                prompt=prompt,
                model=resolved_gpu_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            logger.exception("Chat completion failed")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Chat completion failed: {exc}",
                        "type": "gpu_service_error",
                    }
                },
            )

    reply_text = (reply_text or "").strip()

    logger.info(
        "Returning chat completion | requested_model=%s | resolved_gpu_model=%s | reply_chars=%s | preview=%r",
        requested_model,
        resolved_gpu_model,
        len(reply_text),
        reply_text[:300],
    )

    if stream:
        return StreamingResponse(
            build_streaming_response(reply_text, requested_model=requested_model),
            media_type="text/event-stream",
        )

    return JSONResponse(
        content=build_chat_completion_response(
            content=reply_text,
            requested_model=requested_model,
        )
    )


@app.post("/chat/completions")
async def chat_completions_alias(request: Request):
    """
    Compatibility alias for clients that omit /v1.
    """
    return await chat_completions(request)