import requests
from .ai_models import (
    # Core loading functions
    load_ai_model,
    load_embedding_model,

    # Embedding functions
    generate_embedding,
    store_embedding,
    store_embedding_enhanced,
    generate_and_store_embedding,

    # Search and similarity functions
    search_similar_embeddings,
    get_embedding_with_similarity,

    # AI Model classes
    OpenAIModel,
    Llama3Model,
    AnthropicModel,
    GPT4AllModel,
    TinyLlamaModel,
    NoAIModel,

    # Embedding Model classes
    OpenAIEmbeddingModel,
    GPT4AllEmbeddingModel,
    TinyLlamaEmbeddingModel,
    NoEmbeddingModel,

    # Image Model classes
    CLIPModelHandler,
    NoImageModel,

    # Configuration and management
    ModelsConfig,

    # Testing and diagnostics
    test_embedding_functionality,
    test_pgvector_functionality,
    test_tinyllama_functionality,
    test_tinyllama_embedding_functionality,
    test_tinyllama_framework_integration,
    diagnose_models,
    diagnose_tinyllama,

    # Model availability and setup
    check_model_availability,
    get_recommended_model_setup,
    apply_recommended_models,
    get_available_local_models,
    download_recommended_models,
    switch_to_local_models,

    # Configuration functions
    get_current_models,
    initialize_models_config,
    register_default_models,
    get_tinyllama_config,
    update_tinyllama_config,
    get_tinyllama_embedding_config,
    configure_tinyllama_workflow,

    # Statistics and utilities
    get_pgvector_statistics,

    # Interface classes (for type checking)
    AIModel,
    EmbeddingModel,
    ImageModel
)

__all__ = [
    # Core loading functions
    'load_ai_model',
    'load_embedding_model',

    # Embedding functions
    'generate_embedding',
    'store_embedding',
    'store_embedding_enhanced',
    'generate_and_store_embedding',

    # Search and similarity functions
    'search_similar_embeddings',
    'get_embedding_with_similarity',

    # AI Model classes
    'OpenAIModel',
    'Llama3Model',
    'AnthropicModel',
    'GPT4AllModel',
    'TinyLlamaModel',
    'NoAIModel',

    # Embedding Model classes
    'OpenAIEmbeddingModel',
    'GPT4AllEmbeddingModel',
    'TinyLlamaEmbeddingModel',
    'NoEmbeddingModel',

    # Image Model classes
    'CLIPModelHandler',
    'NoImageModel',

    # Configuration and management
    'ModelsConfig',

    # Testing and diagnostics
    'test_embedding_functionality',
    'test_pgvector_functionality',
    'test_tinyllama_functionality',
    'test_tinyllama_embedding_functionality',
    'test_tinyllama_framework_integration',
    'diagnose_models',
    'diagnose_tinyllama',

    # Model availability and setup
    'check_model_availability',
    'get_recommended_model_setup',
    'apply_recommended_models',
    'get_available_local_models',
    'download_recommended_models',
    'switch_to_local_models',

    # Configuration functions
    'get_current_models',
    'initialize_models_config',
    'register_default_models',
    'get_tinyllama_config',
    'update_tinyllama_config',
    'get_tinyllama_embedding_config',
    'configure_tinyllama_workflow',

    # Statistics and utilities
    'get_pgvector_statistics',

    # Interface classes (for type checking)
    'AIModel',
    'EmbeddingModel',
    'ImageModel'
]