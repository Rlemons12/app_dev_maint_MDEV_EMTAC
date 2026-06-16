"""
AI package.

Keep this file intentionally light to reduce the risk of circular imports.

Recommended external imports:
    from modules.ai.config import ModelsConfig
    from modules.ai.base import AIModel
    from modules.ai.models.text import TinyLlamaModel
    from modules.ai.models.embedding import TinyLlamaEmbeddingModel
    from modules.ai.image.models import CLIPModelHandler
    from modules.ai.image.services import ImageHandler
    from modules.ai.image.factories import get_image_model_handler

Internal package modules should prefer direct module imports, for example:
    from modules.ai.config.models_config import ModelsConfig
    from modules.ai.base.interfaces import AIModel
"""

__all__: list[str] = []
