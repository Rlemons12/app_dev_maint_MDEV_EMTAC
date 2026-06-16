from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic"
DEFAULT_CHAT_MODEL = "deepseek-ai/DeepSeek-V4-Pro:novita"


class HuggingFaceRouterError(RuntimeError):
    """Raised when Hugging Face Router configuration is missing or invalid."""


def get_huggingface_router_client() -> OpenAI:
    """
    Create an OpenAI-compatible client for Hugging Face Router.

    Requires HF_TOKEN in the environment or project .env file.
    """
    load_dotenv(override=True)

    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not hf_token:
        raise HuggingFaceRouterError("HF_TOKEN is required for Hugging Face Router.")

    return OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=hf_token,
    )


def describe_image_url(
    image_url: str,
    prompt: str = "Describe this image in one sentence.",
    model: str = DEFAULT_VISION_MODEL,
) -> str:
    client = get_huggingface_router_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
    )

    message = completion.choices[0].message
    content: Any = message.content

    if isinstance(content, str):
        return content

    return str(message)


def ask_text_question(
    question: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    client = get_huggingface_router_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
    )

    message = completion.choices[0].message
    content: Any = message.content

    if isinstance(content, str):
        return content

    return str(message)
