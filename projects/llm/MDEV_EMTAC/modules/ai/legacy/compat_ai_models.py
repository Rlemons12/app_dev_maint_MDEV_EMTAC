"""
Compatibility exports for legacy ai_models imports.

This file provides stable import points while the legacy
plugins/ai_modules/ai_models/ai_models.py module is being split
into the modules/ai package structure.
"""

from __future__ import annotations

from modules.ai.base.interfaces import AIModel, EmbeddingModel, ImageModel
from modules.ai.bootstrap.initialize_models import (
from modules.ai.config.models_config import ModelsConfig
from modules.ai.image.models.clip_model_handler import CLIPModelHandler
from modules.ai.image.models.no_image_model import NoImageModel
from modules.ai.models.embedding.gpt4all_embedding_model import GPT4AllEmbeddingModel
from modules.ai.models.embedding.no_embedding_model import NoEmbeddingModel
from modules.ai.models.embedding.openai_embedding_model import OpenAIEmbeddingModel
from modules.ai.models.embedding.tinyllama_embedding_model import TinyLlamaEmbeddingModel
from modules.ai.models.text.anthropic_model import AnthropicModel
from modules.ai.models.text.gpt4all_model import GPT4AllModel
from modules.ai.models.text.llama3_model import Llama3Model
from modules.ai.models.text.no_ai_model import NoAIModel
from modules.ai.models.text.openai_model import OpenAIModel
from modules.ai.models.text.tinyllama_model import TinyLlamaModel

    initialize_models_config,
    register_default_models,
    register_default_models_with_tinyllama,
    register_default_models_with_tinyllama_updated,
)

from modules.ai.utils.embedding_storage import (
    generate_embedding,
    store_embedding_enhanced,
    store_embedding,
    generate_and_store_embedding,
    search_similar_embeddings,
    get_embedding_with_similarity,
    get_pgvector_statistics,
    example_completeDocument_integration,
)

from modules.ai.utils.model_diagnostics import (
    diagnose_models,
    diagnose_tinyllama,
    test_embedding_functionality,
    test_pgvector_functionality,
    test_tinyllama_framework_integration,
    test_tinyllama_functionality,
    test_tinyllama_embedding_functionality,
    example_model_configuration,
)

from modules.ai.utils.model_recommendations import (
    check_model_availability,
    get_recommended_model_setup,
    apply_recommended_models,
    configure_tinyllama_workflow,
    switch_to_local_models,
)

from modules.ai.utils.model_downloads import (
    download_recommended_models,
    get_available_local_models,
)


def load_ai_model(model_name=None):
    return ModelsConfig.load_ai_model(model_name)


def load_embedding_model(model_name=None):
    return ModelsConfig.load_embedding_model(model_name)


def load_image_model(model_name=None):
    return ModelsConfig.load_image_model(model_name)
