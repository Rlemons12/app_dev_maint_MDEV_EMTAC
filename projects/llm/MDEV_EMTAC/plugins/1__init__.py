# plugins/__init__.py

# Expose ai_modules
from .ai_modules import (
    load_ai_model,
    load_embedding_model,
    generate_embedding,
    store_embedding,
    OpenAIModel,
    Llama3Model,
    OpenAIEmbeddingModel,
    NoAIModel,
    NoEmbeddingModel,
)

# Expose image_modules (THIS WAS MISSING)
from .image_modules.image_models import (
    CLIPModelHandler,
    NoImageModel,
    BaseImageModelHandler,
)

from .image_modules.image_handler import ImageHandler


__all__ = [
    # embedding + AI models
    'store_embedding',
    'load_ai_model',
    'load_embedding_model',
    'generate_embedding',
    'OpenAIModel',
    'Llama3Model',
    'OpenAIEmbeddingModel',
    'NoAIModel',
    'NoEmbeddingModel',

    # image models (critical)
    'CLIPModelHandler',
    'NoImageModel',
    'BaseImageModelHandler',
    'ImageHandler',
]
