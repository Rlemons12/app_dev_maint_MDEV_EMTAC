from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")

SOURCE_FILES = {
    "ai_models": PROJECT_ROOT / r"plugins\ai_modules\ai_models\ai_models.py",
    "image_handler": PROJECT_ROOT / r"plugins\image_modules\image_handler.py",
    "image_models": PROJECT_ROOT / r"plugins\image_modules\image_models.py",
}

OVERWRITE_EXISTING = False
WRITE_UNMAPPED_REVIEW_FILES = True
WRITE_MANIFEST_FILE = True

OUTPUT_DIR = PROJECT_ROOT / r"scripts\folder_outputs"
REVIEW_DIR = OUTPUT_DIR / "migration_review"
MANIFEST_FILE = REVIEW_DIR / "combined_migration_manifest.json"


@dataclass
class ExtractedNode:
    source_file_key: str
    name: str
    node_type: str  # "class" | "function" | "assign" | "import_block"
    source: str
    lineno: int
    end_lineno: int


@dataclass
class TargetFileSpec:
    path: Path
    header: str = ""
    imports: List[str] = field(default_factory=list)
    body_blocks: List[str] = field(default_factory=list)

    def render(self) -> str:
        parts: List[str] = []

        if self.header.strip():
            parts.append(self.header.rstrip())

        if self.imports:
            unique_imports: List[str] = []
            seen = set()
            for item in self.imports:
                normalized = item.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_imports.append(normalized)
            if unique_imports:
                parts.append("\n".join(unique_imports))

        if self.body_blocks:
            parts.append(
                "\n\n\n".join(
                    block.rstrip() for block in self.body_blocks if block.strip()
                )
            )

        rendered = "\n\n\n".join(part for part in parts if part.strip()).rstrip()
        return rendered + "\n"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str, overwrite: bool = False) -> None:
    ensure_parent(path)
    if path.exists() and not overwrite:
        print(f"[SKIP] {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"[WRITE] {path}")


def extract_module_docstring(module: ast.Module, source: str) -> Optional[str]:
    doc = ast.get_docstring(module)
    if not doc:
        return None

    if module.body and isinstance(module.body[0], ast.Expr):
        first = module.body[0]
        if hasattr(first, "lineno") and hasattr(first, "end_lineno"):
            lines = source.splitlines()
            return "\n".join(lines[first.lineno - 1:first.end_lineno])
    return None


def get_source_segment(lines: List[str], node: ast.AST) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def get_assign_name(node: ast.AST) -> str:
    if isinstance(node, ast.Assign) and node.targets:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return "<assign>"

    if isinstance(target, ast.Name):
        return target.id

    if isinstance(target, ast.Attribute):
        try:
            return ast.unparse(target)
        except Exception:
            return "<assign>"

    return "<assign>"


def collect_nodes(
    source_file_key: str,
    tree: ast.Module,
    source: str,
) -> Tuple[
    Optional[str],
    List[ExtractedNode],
    List[ExtractedNode],
    List[ExtractedNode],
]:
    lines = source.splitlines()
    docstring_src = extract_module_docstring(tree, source)

    imports: List[ExtractedNode] = []
    assigns: List[ExtractedNode] = []
    defs: List[ExtractedNode] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(
                ExtractedNode(
                    source_file_key=source_file_key,
                    name="<import>",
                    node_type="import_block",
                    source=get_source_segment(lines, node),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assigns.append(
                ExtractedNode(
                    source_file_key=source_file_key,
                    name=get_assign_name(node),
                    node_type="assign",
                    source=get_source_segment(lines, node),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node_type = "class" if isinstance(node, ast.ClassDef) else "function"
            defs.append(
                ExtractedNode(
                    source_file_key=source_file_key,
                    name=node.name,
                    node_type=node_type,
                    source=get_source_segment(lines, node),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

    return docstring_src, imports, assigns, defs


def build_target_specs() -> Dict[str, TargetFileSpec]:
    return {
        # ------------------------------------------------------------------
        # AI TARGETS
        # ------------------------------------------------------------------
        "modules/ai/config/models_config.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/config/models_config.py",
            header=textwrap.dedent(
                """
                \"\"\"
                Extracted ModelsConfig and related configuration helpers
                from legacy plugins/ai_modules/ai_models/ai_models.py.
                \"\"\"
                """
            ).strip(),
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/base/interfaces.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/base/interfaces.py",
            header='"""Abstract interfaces for AI, embedding, and image model types."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/openai_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/openai_model.py",
            header='"""OpenAI text and vision-assisted description model wrapper."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/anthropic_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/anthropic_model.py",
            header='"""Anthropic model wrapper extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/llama3_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/llama3_model.py",
            header='"""Llama 3 model wrapper extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/gpt4all_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/gpt4all_model.py",
            header='"""GPT4All model wrapper extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/tinyllama_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/tinyllama_model.py",
            header='"""TinyLlama model wrapper extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/text/no_ai_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/no_ai_model.py",
            header='"""Disabled AI model implementation."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/embedding/openai_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/openai_embedding_model.py",
            header='"""OpenAI embedding model wrapper."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/embedding/gpt4all_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/gpt4all_embedding_model.py",
            header='"""GPT4All/SentenceTransformer embedding model wrapper."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/embedding/tinyllama_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/tinyllama_embedding_model.py",
            header='"""TinyLlama embedding model wrapper."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/embedding/no_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/no_embedding_model.py",
            header='"""Disabled embedding model implementation."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/image/no_image_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/image/no_image_model.py",
            header='"""Disabled image model implementation extracted from ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/models/image/clip_model_handler.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/image/clip_model_handler.py",
            header='"""Legacy CLIP model handler extracted from ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/utils/model_diagnostics.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_diagnostics.py",
            header='"""Model diagnostics and test helpers extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/utils/model_recommendations.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_recommendations.py",
            header='"""Model recommendation and availability helpers."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/utils/model_downloads.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_downloads.py",
            header='"""Model download and local model discovery helpers."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/utils/embedding_storage.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/embedding_storage.py",
            header='"""Embedding generation, storage, and similarity helpers."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/bootstrap/initialize_models.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/bootstrap/initialize_models.py",
            header='"""Bootstrap helpers for model config initialization and defaults."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/bootstrap/env_bootstrap.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/bootstrap/env_bootstrap.py",
            header='"""Environment bootstrap helpers extracted from legacy ai_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/legacy/compat_ai_models.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/legacy/compat_ai_models.py",
            header='"""Compatibility wrapper area for legacy imports during migration."""',
            imports=["from __future__ import annotations"],
        ),

        # ------------------------------------------------------------------
        # IMAGE TARGETS
        # ------------------------------------------------------------------
        "modules/ai/image/services/image_handler_service.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/services/image_handler_service.py",
            header='"""Image handler service extracted from legacy plugins/image_modules/image_handler.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/base/base_image_model_handler.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/base/base_image_model_handler.py",
            header='"""Base abstract interface for image model handlers."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/models/clip_model_handler.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/models/clip_model_handler.py",
            header='"""CLIP image model handler extracted from legacy image_models.py."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/models/no_image_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/models/no_image_model.py",
            header='"""Disabled image model implementation."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/factories/model_handler_factory.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/factories/model_handler_factory.py",
            header='"""Factory helpers for resolving image model handler instances."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/config/image_runtime_config.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/config/image_runtime_config.py",
            header='"""Runtime image module configuration and preserved globals."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/bootstrap/image_env_bootstrap.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/bootstrap/image_env_bootstrap.py",
            header='"""Environment/bootstrap helpers for the image module migration."""',
            imports=["from __future__ import annotations"],
        ),
        "modules/ai/image/legacy/compat_image_modules.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/legacy/compat_image_modules.py",
            header='"""Compatibility exports for legacy image module imports."""',
            imports=["from __future__ import annotations"],
        ),
    }


AI_CLASS_TARGET_MAP = {
    "ModelsConfig": "modules/ai/config/models_config.py",
    "AIModel": "modules/ai/base/interfaces.py",
    "EmbeddingModel": "modules/ai/base/interfaces.py",
    "ImageModel": "modules/ai/base/interfaces.py",
    "NoAIModel": "modules/ai/models/text/no_ai_model.py",
    "AnthropicModel": "modules/ai/models/text/anthropic_model.py",
    "OpenAIModel": "modules/ai/models/text/openai_model.py",
    "Llama3Model": "modules/ai/models/text/llama3_model.py",
    "GPT4AllModel": "modules/ai/models/text/gpt4all_model.py",
    "TinyLlamaModel": "modules/ai/models/text/tinyllama_model.py",
    "NoEmbeddingModel": "modules/ai/models/embedding/no_embedding_model.py",
    "OpenAIEmbeddingModel": "modules/ai/models/embedding/openai_embedding_model.py",
    "GPT4AllEmbeddingModel": "modules/ai/models/embedding/gpt4all_embedding_model.py",
    "TinyLlamaEmbeddingModel": "modules/ai/models/embedding/tinyllama_embedding_model.py",
    "NoImageModel": "modules/ai/models/image/no_image_model.py",
    "CLIPModelHandler": "modules/ai/models/image/clip_model_handler.py",
}

AI_FUNCTION_TARGET_MAP = {
    "register_default_models_with_tinyllama_updated": "modules/ai/bootstrap/initialize_models.py",
    "test_tinyllama_framework_integration": "modules/ai/utils/model_diagnostics.py",
    "test_tinyllama_embedding_functionality": "modules/ai/utils/model_diagnostics.py",
    "register_default_models_with_tinyllama": "modules/ai/bootstrap/initialize_models.py",
    "diagnose_tinyllama": "modules/ai/utils/model_diagnostics.py",
    "test_tinyllama_functionality": "modules/ai/utils/model_diagnostics.py",
    "diagnose_models": "modules/ai/utils/model_diagnostics.py",
    "download_recommended_models": "modules/ai/utils/model_downloads.py",
    "get_available_local_models": "modules/ai/utils/model_downloads.py",
    "switch_to_local_models": "modules/ai/utils/model_recommendations.py",
    "initialize_models_config": "modules/ai/bootstrap/initialize_models.py",
    "register_default_models": "modules/ai/bootstrap/initialize_models.py",
    "get_tinyllama_config": "modules/ai/config/models_config.py",
    "update_tinyllama_config": "modules/ai/config/models_config.py",
    "check_model_availability": "modules/ai/utils/model_recommendations.py",
    "get_recommended_model_setup": "modules/ai/utils/model_recommendations.py",
    "apply_recommended_models": "modules/ai/utils/model_recommendations.py",
    "generate_embedding": "modules/ai/utils/embedding_storage.py",
    "store_embedding_enhanced": "modules/ai/utils/embedding_storage.py",
    "store_embedding": "modules/ai/utils/embedding_storage.py",
    "generate_and_store_embedding": "modules/ai/utils/embedding_storage.py",
    "get_current_models": "modules/ai/config/models_config.py",
    "test_embedding_functionality": "modules/ai/utils/model_diagnostics.py",
    "load_ai_model": "modules/ai/legacy/compat_ai_models.py",
    "load_embedding_model": "modules/ai/legacy/compat_ai_models.py",
    "load_image_model": "modules/ai/legacy/compat_ai_models.py",
    "example_completeDocument_integration": "modules/ai/utils/embedding_storage.py",
    "example_model_configuration": "modules/ai/utils/model_diagnostics.py",
    "search_similar_embeddings": "modules/ai/utils/embedding_storage.py",
    "get_embedding_with_similarity": "modules/ai/utils/embedding_storage.py",
    "get_pgvector_statistics": "modules/ai/utils/embedding_storage.py",
    "test_pgvector_functionality": "modules/ai/utils/model_diagnostics.py",
    "get_tinyllama_embedding_config": "modules/ai/config/models_config.py",
    "configure_tinyllama_workflow": "modules/ai/utils/model_recommendations.py",
}

AI_ASSIGN_TARGET_MAP = {
    "DEV_ENV": "modules/ai/bootstrap/env_bootstrap.py",
    "PROJECT_ENV": "modules/ai/bootstrap/env_bootstrap.py",
    "engine": "modules/ai/config/models_config.py",
    "Session": "modules/ai/config/models_config.py",
    "QUANTIZATION_AVAILABLE": "modules/ai/config/models_config.py",
    "TORCH_COMPILE_AVAILABLE": "modules/ai/config/models_config.py",
    "TRANSFORMERS_AVAILABLE": "modules/ai/config/models_config.py",
    "MODEL_MINILM_DIR_ENV": "modules/ai/config/models_config.py",
    "LOGGING_AVAILABLE": "modules/ai/config/models_config.py",
    "logger": "modules/ai/config/models_config.py",
}

IMAGE_CLASS_TARGET_MAP = {
    "ImageHandler": "modules/ai/image/services/image_handler_service.py",
    "BaseImageModelHandler": "modules/ai/image/base/base_image_model_handler.py",
    "NoImageModel": "modules/ai/image/models/no_image_model.py",
    "CLIPModelHandler": "modules/ai/image/models/clip_model_handler.py",
}

IMAGE_FUNCTION_TARGET_MAP = {
    "get_image_model_handler": "modules/ai/image/factories/model_handler_factory.py",
}

IMAGE_ASSIGN_TARGET_MAP = {
    "logger": "modules/ai/image/config/image_runtime_config.py",
    "engine": "modules/ai/image/config/image_runtime_config.py",
    "Session": "modules/ai/image/config/image_runtime_config.py",
    "ImageFile.LOAD_TRUNCATED_IMAGES": "modules/ai/image/bootstrap/image_env_bootstrap.py",
}

COMMON_IMPORT_BLOCK = "from __future__ import annotations"


def build_file_initializers(specs: Dict[str, TargetFileSpec]) -> None:
    for spec in specs.values():
        ensure_parent(spec.path)


def add_block(specs: Dict[str, TargetFileSpec], rel_path: str, block: str) -> None:
    specs[rel_path].body_blocks.append(block)


def normalize_block(block: str) -> str:
    return block.rstrip() + "\n"


def generate_unmapped_review_file(source_name: str, unmapped: List[ExtractedNode]) -> str:
    sections = [
        f'"""Unmapped content extracted from legacy {source_name} for manual review."""',
        "from __future__ import annotations",
    ]

    for item in unmapped:
        sections.append(
            f"# ---- {item.node_type.upper()}: {item.name} "
            f"(lines {item.lineno}-{item.end_lineno}) ----"
        )
        sections.append(item.source.rstrip())

    return "\n\n\n".join(sections) + "\n"


def generate_manifest(
    specs: Dict[str, TargetFileSpec],
    counts_by_source: Dict[str, Dict[str, int]],
    unmapped_by_source: Dict[str, List[ExtractedNode]],
) -> str:
    payload = {
        "project_root": str(PROJECT_ROOT),
        "source_files": {k: str(v) for k, v in SOURCE_FILES.items()},
        "overwrite_existing": OVERWRITE_EXISTING,
        "output_dir": str(OUTPUT_DIR),
        "review_dir": str(REVIEW_DIR),
        "summary": {
            "files_with_content": sum(1 for spec in specs.values() if spec.body_blocks),
            "total_unmapped_items": sum(len(v) for v in unmapped_by_source.values()),
        },
        "counts_by_source": counts_by_source,
        "written_files": {
            rel_path: {
                "path": str(spec.path),
                "blocks": len(spec.body_blocks),
            }
            for rel_path, spec in specs.items()
            if spec.body_blocks
        },
        "unmapped_by_source": {
            source_name: [
                {
                    "name": item.name,
                    "node_type": item.node_type,
                    "lineno": item.lineno,
                    "end_lineno": item.end_lineno,
                }
                for item in items
            ]
            for source_name, items in unmapped_by_source.items()
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def create_package_files() -> None:
    packages = [
        "modules/__init__.py",
        "modules/ai/__init__.py",
        "modules/ai/base/__init__.py",
        "modules/ai/bootstrap/__init__.py",
        "modules/ai/config/__init__.py",
        "modules/ai/legacy/__init__.py",
        "modules/ai/models/__init__.py",
        "modules/ai/models/text/__init__.py",
        "modules/ai/models/embedding/__init__.py",
        "modules/ai/models/image/__init__.py",
        "modules/ai/utils/__init__.py",
        "modules/ai/image/__init__.py",
        "modules/ai/image/base/__init__.py",
        "modules/ai/image/bootstrap/__init__.py",
        "modules/ai/image/config/__init__.py",
        "modules/ai/image/factories/__init__.py",
        "modules/ai/image/legacy/__init__.py",
        "modules/ai/image/models/__init__.py",
        "modules/ai/image/services/__init__.py",
    ]

    for rel in packages:
        path = PROJECT_ROOT / rel
        if not path.exists():
            write_text(path, '"""Package marker."""\n', overwrite=False)


def create_ai_compat_wrapper_file() -> str:
    return textwrap.dedent(
        '''
        """
        Compatibility exports for legacy ai_models imports.

        This file provides stable import points while the legacy
        plugins/ai_modules/ai_models/ai_models.py module is being split
        into the modules/ai package structure.
        """

        from __future__ import annotations

        from modules.ai.config.models_config import ModelsConfig
        from modules.ai.base.interfaces import AIModel, EmbeddingModel, ImageModel

        from modules.ai.models.text.no_ai_model import NoAIModel
        from modules.ai.models.text.openai_model import OpenAIModel
        from modules.ai.models.text.anthropic_model import AnthropicModel
        from modules.ai.models.text.llama3_model import Llama3Model
        from modules.ai.models.text.gpt4all_model import GPT4AllModel
        from modules.ai.models.text.tinyllama_model import TinyLlamaModel

        from modules.ai.models.embedding.no_embedding_model import NoEmbeddingModel
        from modules.ai.models.embedding.openai_embedding_model import OpenAIEmbeddingModel
        from modules.ai.models.embedding.gpt4all_embedding_model import GPT4AllEmbeddingModel
        from modules.ai.models.embedding.tinyllama_embedding_model import TinyLlamaEmbeddingModel

        from modules.ai.models.image.no_image_model import NoImageModel
        from modules.ai.models.image.clip_model_handler import CLIPModelHandler

        from modules.ai.bootstrap.initialize_models import (
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
        '''
    ).strip() + "\n"


def create_image_compat_wrapper_file() -> str:
    return textwrap.dedent(
        '''
        """
        Compatibility exports for legacy image module imports.

        This file provides stable import points while the legacy
        plugins/image_modules/image_handler.py and image_models.py modules
        are being split into the modules/ai/image package structure.
        """

        from __future__ import annotations

        from modules.ai.image.services.image_handler_service import ImageHandler
        from modules.ai.image.base.base_image_model_handler import BaseImageModelHandler
        from modules.ai.image.models.clip_model_handler import CLIPModelHandler
        from modules.ai.image.models.no_image_model import NoImageModel
        from modules.ai.image.factories.model_handler_factory import get_image_model_handler
        '''
    ).strip() + "\n"


def process_ai_source(
    source_key: str,
    source_path: Path,
    specs: Dict[str, TargetFileSpec],
) -> Tuple[Dict[str, int], List[ExtractedNode]]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstring_src, imports, assigns, defs = collect_nodes(source_key, tree, source)
    unmapped: List[ExtractedNode] = []

    compat_target = "modules/ai/legacy/compat_ai_models.py"

    if docstring_src:
        add_block(specs, compat_target, normalize_block(docstring_src))

    add_block(
        specs,
        compat_target,
        normalize_block(
            textwrap.dedent(
                f"""
                # Legacy source reference:
                # Original source:
                # {source_path}
                """
            ).strip()
        ),
    )

    for item in assigns:
        target = AI_ASSIGN_TARGET_MAP.get(item.name)
        if target:
            add_block(specs, target, normalize_block(item.source))
        else:
            unmapped.append(item)

    for item in defs:
        if item.node_type == "class":
            target = AI_CLASS_TARGET_MAP.get(item.name)
        else:
            target = AI_FUNCTION_TARGET_MAP.get(item.name)

        if target:
            add_block(specs, target, normalize_block(item.source))
        else:
            unmapped.append(item)

    counts = {
        "imports_found": len(imports),
        "assignments_found": len(assigns),
        "definitions_found": len(defs),
        "unmapped_items": len(unmapped),
    }
    return counts, unmapped


def process_image_source(
    source_key: str,
    source_path: Path,
    specs: Dict[str, TargetFileSpec],
) -> Tuple[Dict[str, int], List[ExtractedNode]]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstring_src, imports, assigns, defs = collect_nodes(source_key, tree, source)
    unmapped: List[ExtractedNode] = []

    compat_target = "modules/ai/image/legacy/compat_image_modules.py"

    if docstring_src:
        add_block(
            specs,
            compat_target,
            normalize_block(
                f"# Source module docstring from: {source_path.name}\n{docstring_src}"
            ),
        )

    add_block(
        specs,
        compat_target,
        normalize_block(
            textwrap.dedent(
                f"""
                # Legacy source reference:
                # Original source:
                # {source_path}
                """
            ).strip()
        ),
    )

    for item in assigns:
        target = IMAGE_ASSIGN_TARGET_MAP.get(item.name)
        if target:
            add_block(specs, target, normalize_block(item.source))
        else:
            unmapped.append(item)

    for item in defs:
        if item.node_type == "class":
            target = IMAGE_CLASS_TARGET_MAP.get(item.name)
        else:
            target = IMAGE_FUNCTION_TARGET_MAP.get(item.name)

        if target:
            add_block(specs, target, normalize_block(item.source))
        else:
            unmapped.append(item)

    counts = {
        "imports_found": len(imports),
        "assignments_found": len(assigns),
        "definitions_found": len(defs),
        "unmapped_items": len(unmapped),
    }
    return counts, unmapped


def main() -> None:
    print("=" * 100)
    print("MIGRATING LEGACY AI + IMAGE MODULES INTO NEW MODULE STRUCTURE")
    print("=" * 100)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OVERWRITE_EXISTING: {OVERWRITE_EXISTING}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print("=" * 100)

    create_package_files()

    specs = build_target_specs()
    build_file_initializers(specs)

    counts_by_source: Dict[str, Dict[str, int]] = {}
    unmapped_by_source: Dict[str, List[ExtractedNode]] = {}

    for source_key, source_path in SOURCE_FILES.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        print("-" * 100)
        print(f"PROCESSING SOURCE: {source_key}")
        print(f"PATH: {source_path}")
        print("-" * 100)

        if source_key == "ai_models":
            counts, unmapped = process_ai_source(source_key, source_path, specs)
        else:
            counts, unmapped = process_image_source(source_key, source_path, specs)

        counts_by_source[source_key] = counts
        unmapped_by_source[source_key] = unmapped

    for spec in specs.values():
        if not spec.imports:
            spec.imports.append(COMMON_IMPORT_BLOCK)

    for rel_path, spec in specs.items():
        if rel_path == "modules/ai/legacy/compat_ai_models.py":
            write_text(spec.path, create_ai_compat_wrapper_file(), overwrite=True)
            continue

        if rel_path == "modules/ai/image/legacy/compat_image_modules.py":
            write_text(spec.path, create_image_compat_wrapper_file(), overwrite=True)
            continue

        if spec.body_blocks:
            rendered = spec.render()
            write_text(spec.path, rendered, overwrite=OVERWRITE_EXISTING)

    if WRITE_UNMAPPED_REVIEW_FILES:
        for source_key, unmapped in unmapped_by_source.items():
            if not unmapped:
                continue

            review_path = REVIEW_DIR / f"unmapped_from_{source_key}.py"
            review_text = generate_unmapped_review_file(source_key, unmapped)
            write_text(review_path, review_text, overwrite=True)

    if WRITE_MANIFEST_FILE:
        manifest_text = generate_manifest(
            specs=specs,
            counts_by_source=counts_by_source,
            unmapped_by_source=unmapped_by_source,
        )
        write_text(MANIFEST_FILE, manifest_text, overwrite=True)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    written_count = sum(1 for spec in specs.values() if spec.body_blocks)
    total_unmapped = sum(len(v) for v in unmapped_by_source.values())
    print(f"Files written with extracted content: {written_count}")
    print(f"Total unmapped items: {total_unmapped}")

    for source_key, unmapped in unmapped_by_source.items():
        review_path = REVIEW_DIR / f"unmapped_from_{source_key}.py"
        print("-" * 100)
        print(f"SOURCE: {source_key}")
        print(f"Unmapped items: {len(unmapped)}")
        if unmapped:
            print(f"Review here: {review_path}")
            print("Unmapped names:")
            for item in unmapped:
                print(f"  - {item.node_type}: {item.name} ({item.lineno}-{item.end_lineno})")

    if WRITE_MANIFEST_FILE:
        print("-" * 100)
        print(f"Combined migration manifest written here: {MANIFEST_FILE}")

    print("=" * 100)
    print("IMPORTANT")
    print("=" * 100)
    print("1. This script extracts by top-level class/function/assignment name.")
    print("2. It handles ai_models.py, image_handler.py, and image_models.py in one run.")
    print("3. It writes review/manifest output under scripts/folder_outputs, not modules/ai.")
    print("4. It writes separate compatibility wrappers for ai and image.")
    print("5. It does not rewrite imports inside extracted blocks.")
    print("6. After extraction, do a second pass to clean imports and cross-file references.")
    print("=" * 100)


if __name__ == "__main__":
    main()