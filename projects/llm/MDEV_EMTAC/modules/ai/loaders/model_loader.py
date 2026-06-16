from __future__ import annotations

from typing import Optional

from modules.ai.config.models_config import ModelsConfig


def load_ai_model(model_name: Optional[str] = None):
    """
    Public AI model loader entrypoint.

    Preferred for callers that should not know about ModelsConfig internals.
    """
    return ModelsConfig.load_ai_model(model_name)


def load_embedding_model(model_name: Optional[str] = None):
    """
    Public embedding model loader entrypoint.

    Preferred for callers that should not know about ModelsConfig internals.
    """
    return ModelsConfig.load_embedding_model(model_name)


def load_image_model(model_name: Optional[str] = None):
    """
    Public image model loader entrypoint.

    Preferred for callers that should not know about ModelsConfig internals.
    """
    return ModelsConfig.load_image_model(model_name)