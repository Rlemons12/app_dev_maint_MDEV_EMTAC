from __future__ import annotations

from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")
OVERWRITE_EXISTING = True

INIT_FILE_CONTENTS: Dict[str, str] = {
    "modules/__init__.py": '''"""Top-level modules package."""\n''',

    "modules/ai/__init__.py": '''"""
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
''',

    "modules/ai/base/__init__.py": '''"""Base interfaces for AI-related types."""

from .interfaces import AIModel, EmbeddingModel, ImageModel

__all__ = [
    "AIModel",
    "EmbeddingModel",
    "ImageModel",
]
''',

    "modules/ai/bootstrap/__init__.py": '''"""
Bootstrap helpers for AI package setup.

Keep this package export small. Import concrete modules directly inside
the package implementation if circular imports appear.
"""

from .initialize_models import (
    initialize_models_config,
    register_default_models,
    register_default_models_with_tinyllama,
    register_default_models_with_tinyllama_updated,
)

__all__ = [
    "initialize_models_config",
    "register_default_models",
    "register_default_models_with_tinyllama",
    "register_default_models_with_tinyllama_updated",
]
''',

    "modules/ai/config/__init__.py": '''"""Configuration exports for AI package."""

from .models_config import (
    ModelsConfig,
    get_current_models,
    get_tinyllama_config,
    update_tinyllama_config,
    get_tinyllama_embedding_config,
)

__all__ = [
    "ModelsConfig",
    "get_current_models",
    "get_tinyllama_config",
    "update_tinyllama_config",
    "get_tinyllama_embedding_config",
]
''',

    "modules/ai/legacy/__init__.py": '''"""Legacy compatibility package for AI imports."""\n''',

    "modules/ai/models/__init__.py": '''"""
Model package for text and embedding model implementations.

Prefer importing from subpackages:
    from modules.ai.models.text import TinyLlamaModel
    from modules.ai.models.embedding import TinyLlamaEmbeddingModel
"""

__all__: list[str] = []
''',

    "modules/ai/models/text/__init__.py": '''"""Text model exports."""

from .no_ai_model import NoAIModel
from .anthropic_model import AnthropicModel
from .openai_model import OpenAIModel
from .llama3_model import Llama3Model
from .gpt4all_model import GPT4AllModel
from .tinyllama_model import TinyLlamaModel

__all__ = [
    "NoAIModel",
    "AnthropicModel",
    "OpenAIModel",
    "Llama3Model",
    "GPT4AllModel",
    "TinyLlamaModel",
]
''',

    "modules/ai/models/embedding/__init__.py": '''"""Embedding model exports."""

from .no_embedding_model import NoEmbeddingModel
from .openai_embedding_model import OpenAIEmbeddingModel
from .gpt4all_embedding_model import GPT4AllEmbeddingModel
from .tinyllama_embedding_model import TinyLlamaEmbeddingModel

__all__ = [
    "NoEmbeddingModel",
    "OpenAIEmbeddingModel",
    "GPT4AllEmbeddingModel",
    "TinyLlamaEmbeddingModel",
]
''',

    "modules/ai/utils/__init__.py": '''"""
Utility package for AI helpers.

Keep this file intentionally light. Utility modules often depend on many
other modules, so broad re-exports here can increase circular-import risk.

Preferred imports:
    from modules.ai.utils.embedding_storage import generate_embedding
    from modules.ai.utils.model_diagnostics import diagnose_models
    from modules.ai.utils.model_downloads import download_recommended_models
    from modules.ai.utils.model_recommendations import get_recommended_model_setup
"""

__all__: list[str] = []
''',

    "modules/ai/image/__init__.py": '''"""
Image-related package exports.

Prefer importing from subpackages:
    from modules.ai.image.models import CLIPModelHandler
    from modules.ai.image.services import ImageHandler
"""

__all__: list[str] = []
''',

    "modules/ai/image/base/__init__.py": '''"""Base image model handler exports."""

from .base_image_model_handler import BaseImageModelHandler

__all__ = [
    "BaseImageModelHandler",
]
''',

    "modules/ai/image/bootstrap/__init__.py": '''"""Image bootstrap package."""\n''',

    "modules/ai/image/config/__init__.py": '''"""Image runtime configuration package."""\n''',

    "modules/ai/image/factories/__init__.py": '''"""Image factory exports."""

from .model_handler_factory import get_image_model_handler

__all__ = [
    "get_image_model_handler",
]
''',

    "modules/ai/image/legacy/__init__.py": '''"""Legacy compatibility package for image imports."""\n''',

    "modules/ai/image/models/__init__.py": '''"""Image model exports."""

from .no_image_model import NoImageModel
from .clip_model_handler import CLIPModelHandler

__all__ = [
    "NoImageModel",
    "CLIPModelHandler",
]
''',

    "modules/ai/image/services/__init__.py": '''"""Image service exports."""

from .image_handler_service import ImageHandler

__all__ = [
    "ImageHandler",
]
''',
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, overwrite: bool) -> None:
    ensure_parent(path)

    if path.exists() and not overwrite:
        print(f"[SKIP] {path}")
        return

    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"[WRITE] {path}")


def main() -> None:
    print("=" * 100)
    print("BUILDING SAFE AI PACKAGE EXPORTS")
    print("=" * 100)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OVERWRITE_EXISTING: {OVERWRITE_EXISTING}")
    print("=" * 100)

    for relative_path, content in INIT_FILE_CONTENTS.items():
        full_path = PROJECT_ROOT / relative_path
        write_file(full_path, content, overwrite=OVERWRITE_EXISTING)

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Recommended import style after this:")
    print("  from modules.ai.config import ModelsConfig")
    print("  from modules.ai.base import AIModel")
    print("  from modules.ai.models.text import TinyLlamaModel")
    print("  from modules.ai.models.embedding import TinyLlamaEmbeddingModel")
    print("  from modules.ai.image.models import CLIPModelHandler")
    print("  from modules.ai.image.services import ImageHandler")
    print("  from modules.ai.image.factories import get_image_model_handler")
    print("  from modules.ai.utils.embedding_storage import generate_embedding")
    print("=" * 100)
    print("Important:")
    print("  - modules/ai/__init__.py is intentionally minimal")
    print("  - modules/ai/utils/__init__.py is intentionally minimal")
    print("  - internal package modules should prefer direct module imports")
    print("=" * 100)


if __name__ == "__main__":
    main()