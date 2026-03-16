from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")

SOURCE_FILES = {
    "ai_models": PROJECT_ROOT / r"plugins\ai_modules\ai_models\ai_models.py",
    "image_handler": PROJECT_ROOT / r"plugins\image_modules\image_handler.py",
    "image_models": PROJECT_ROOT / r"plugins\image_modules\image_models.py",
}

OVERWRITE_EXISTING = False
WRITE_UNMAPPED_REVIEW_FILES = True
WRITE_MANIFEST_FILE = True
WRITE_OVERLAP_REPORT = True
WRITE_DUPLICATE_REVIEW_FILES = True

OUTPUT_DIR = PROJECT_ROOT / r"scripts\folder_outputs"
REVIEW_DIR = OUTPUT_DIR / "migration_review"
MANIFEST_FILE = REVIEW_DIR / "combined_migration_manifest.json"
OVERLAP_REPORT_FILE = REVIEW_DIR / "combined_overlap_report.json"

COMMON_IMPORT_BLOCK = "from __future__ import annotations"

AI_CLASS_SKIP_IF_CANONICAL_IN_IMAGE = {
    "NoImageModel",
    "CLIPModelHandler",
}
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

        unique_imports = dedupe_preserve_order(self.imports)
        if unique_imports:
            parts.append("\n".join(item.strip() for item in unique_imports if item.strip()))

        body_blocks = [block.rstrip() for block in self.body_blocks if block.strip()]
        if body_blocks:
            parts.append("\n\n\n".join(body_blocks))

        rendered = "\n\n\n".join(part for part in parts if part.strip()).rstrip()
        return rendered + "\n"


@dataclass
class MappingRecord:
    source_file_key: str
    source_path: str
    node_type: str
    name: str
    lineno: int
    end_lineno: int
    target_rel_path: Optional[str]
    status: str  # mapped | unmapped | skipped_duplicate
    note: str = ""


@dataclass
class ResolvedTarget:
    rel_path: str
    expected_line: Optional[int] = None
    note: str = ""


FunctionTargetValue = Union[str, Dict[str, Any]]


AI_CLASS_TARGET_MAP: Dict[str, str] = {
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
    "NoImageModel": "modules/ai/image/models/no_image_model.py",
    "CLIPModelHandler": "modules/ai/image/models/clip_model_handler.py",
}

AI_FUNCTION_TARGET_MAP: Dict[str, FunctionTargetValue] = {
    "register_default_models_with_tinyllama_updated": {
        "path": "modules/ai/bootstrap/initialize_models.py",
        "line": 1932,
    },
    "test_tinyllama_framework_integration": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 2209,
    },
    "test_tinyllama_embedding_functionality": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 2237,
    },
    "register_default_models_with_tinyllama": {
        "path": "modules/ai/bootstrap/initialize_models.py",
        "line": 2316,
    },
    "diagnose_tinyllama": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 2405,
    },
    "test_tinyllama_functionality": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 2468,
    },
    "diagnose_models": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 2498,
    },
    "download_recommended_models": {
        "path": "modules/ai/utils/model_downloads.py",
        "line": 2542,
    },
    "get_available_local_models": {
        "path": "modules/ai/utils/model_downloads.py",
        "line": 2606,
    },
    "switch_to_local_models": {
        "path": "modules/ai/utils/model_recommendations.py",
        "line": 2641,
    },
    "initialize_models_config": {
        "path": "modules/ai/bootstrap/initialize_models.py",
        "line": 2665,
    },
    "register_default_models": {
        "path": "modules/ai/bootstrap/initialize_models.py",
        "line": 2702,
    },
    "get_tinyllama_config": {
        "path": "modules/ai/config/models_config.py",
        "line": 2806,
    },
    "update_tinyllama_config": {
        "path": "modules/ai/config/models_config.py",
        "line": 2828,
    },
    "check_model_availability": {
        "path": "modules/ai/utils/model_recommendations.py",
        "line": 2859,
    },
    "get_recommended_model_setup": {
        "path": "modules/ai/utils/model_recommendations.py",
        "line": 3011,
    },
    "apply_recommended_models": {
        "path": "modules/ai/utils/model_recommendations.py",
        "line": 3098,
    },
    "generate_embedding": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3129,
    },
    "store_embedding_enhanced": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3150,
    },
    "store_embedding": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3215,
    },
    "generate_and_store_embedding": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3243,
    },
    "get_current_models": {
        "path": "modules/ai/config/models_config.py",
        "line": 3284,
    },
    "test_embedding_functionality": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 3305,
    },
    "load_ai_model": {
        "path": "modules/ai/legacy/compat_ai_models.py",
        "line": 3332,
    },
    "load_embedding_model": {
        "path": "modules/ai/legacy/compat_ai_models.py",
        "line": 3337,
    },
    "load_image_model": {
        "path": "modules/ai/legacy/compat_ai_models.py",
        "line": 3372,
    },
    "example_completeDocument_integration": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3388,
    },
    "example_model_configuration": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 3424,
    },
    "search_similar_embeddings": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3456,
    },
    "get_embedding_with_similarity": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3525,
    },
    "get_pgvector_statistics": {
        "path": "modules/ai/utils/embedding_storage.py",
        "line": 3578,
    },
    "test_pgvector_functionality": {
        "path": "modules/ai/utils/model_diagnostics.py",
        "line": 3630,
    },
    "get_tinyllama_embedding_config": {
        "path": "modules/ai/config/models_config.py",
        "line": 3732,
    },
    "configure_tinyllama_workflow": {
        "path": "modules/ai/utils/model_recommendations.py",
        "line": 3751,
    },
}

AI_ASSIGN_TARGET_MAP: Dict[str, str] = {
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

IMAGE_CLASS_TARGET_MAP: Dict[str, str] = {
    "ImageHandler": "modules/ai/image/services/image_handler_service.py",
    "BaseImageModelHandler": "modules/ai/image/base/base_image_model_handler.py",
    "NoImageModel": "modules/ai/image/models/no_image_model.py",
    "CLIPModelHandler": "modules/ai/image/models/clip_model_handler.py",
}

IMAGE_FUNCTION_TARGET_MAP: Dict[str, str] = {
    "get_image_model_handler": "modules/ai/image/factories/model_handler_factory.py",
}

IMAGE_ASSIGN_TARGET_MAP: Dict[str, str] = {
    "logger": "modules/ai/image/config/image_runtime_config.py",
    "engine": "modules/ai/image/config/image_runtime_config.py",
    "Session": "modules/ai/image/config/image_runtime_config.py",
    "ImageFile.LOAD_TRUNCATED_IMAGES": "modules/ai/image/bootstrap/image_env_bootstrap.py",
}


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str, overwrite: bool = False) -> None:
    ensure_parent(path)
    if path.exists() and not overwrite:
        print(f"[SKIP] {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"[WRITE] {path}")


def normalize_block(block: str) -> str:
    return block.rstrip() + "\n"


def add_block(specs: Dict[str, TargetFileSpec], rel_path: str, block: str) -> None:
    if rel_path not in specs:
        raise KeyError(f"Target spec not found for path: {rel_path}")
    specs[rel_path].body_blocks.append(block)


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


def build_symbol_ordered_defs(
    source_file_key: str,
    nodes: List[ast.AST],
    lines: List[str],
) -> Tuple[List[ExtractedNode], List[ExtractedNode]]:
    """
    Returns:
      - canonical_defs: last-definition-wins list for class/function names
      - duplicate_defs: shadowed earlier definitions kept for review/reporting
    """
    all_defs: List[ExtractedNode] = []

    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node_type = "class" if isinstance(node, ast.ClassDef) else "function"
            all_defs.append(
                ExtractedNode(
                    source_file_key=source_file_key,
                    name=node.name,
                    node_type=node_type,
                    source=get_source_segment(lines, node),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

    last_seen_index: Dict[Tuple[str, str], int] = {}
    for idx, item in enumerate(all_defs):
        last_seen_index[(item.node_type, item.name)] = idx

    canonical_defs: List[ExtractedNode] = []
    duplicate_defs: List[ExtractedNode] = []

    for idx, item in enumerate(all_defs):
        key = (item.node_type, item.name)
        if last_seen_index[key] == idx:
            canonical_defs.append(item)
        else:
            duplicate_defs.append(item)

    return canonical_defs, duplicate_defs


def collect_nodes(
    source_file_key: str,
    tree: ast.Module,
    source: str,
) -> Tuple[
    Optional[str],
    List[ExtractedNode],
    List[ExtractedNode],
    List[ExtractedNode],
    List[ExtractedNode],
]:
    lines = source.splitlines()
    docstring_src = extract_module_docstring(tree, source)

    imports: List[ExtractedNode] = []
    assigns: List[ExtractedNode] = []

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

    defs, duplicate_defs = build_symbol_ordered_defs(source_file_key, tree.body, lines)

    return docstring_src, imports, assigns, defs, duplicate_defs


def resolve_function_target(
    mapping: Dict[str, FunctionTargetValue],
    item: ExtractedNode,
) -> Optional[ResolvedTarget]:
    raw = mapping.get(item.name)
    if raw is None:
        return None

    if isinstance(raw, str):
        return ResolvedTarget(rel_path=raw)

    if not isinstance(raw, dict):
        raise TypeError(
            f"Invalid function target mapping for '{item.name}': expected str|dict, got {type(raw)!r}"
        )

    rel_path = raw.get("path")
    if not rel_path or not isinstance(rel_path, str):
        raise ValueError(f"Function target for '{item.name}' is missing a valid 'path'")

    expected_line = raw.get("line")
    note = ""

    if expected_line is not None:
        try:
            expected_line = int(expected_line)
        except Exception as exc:
            raise ValueError(
                f"Function target for '{item.name}' has invalid line value: {expected_line!r}"
            ) from exc

        if expected_line != item.lineno:
            note = (
                f"line_mismatch: expected {expected_line}, extracted {item.lineno}; "
                f"mapped by function name to {rel_path}"
            )

    return ResolvedTarget(
        rel_path=rel_path,
        expected_line=expected_line,
        note=note,
    )


def append_mapping_record(
    mapping_records: List[MappingRecord],
    *,
    node: ExtractedNode,
    source_path: Path,
    target_rel_path: Optional[str],
    status: str,
    note: str = "",
) -> None:
    mapping_records.append(
        MappingRecord(
            source_file_key=node.source_file_key,
            source_path=str(source_path),
            node_type=node.node_type,
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            target_rel_path=target_rel_path,
            status=status,
            note=note,
        )
    )


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


def build_target_specs() -> Dict[str, TargetFileSpec]:
    return {
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
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/base/interfaces.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/base/interfaces.py",
            header='"""Abstract interfaces for AI, embedding, and image model types."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/openai_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/openai_model.py",
            header='"""OpenAI text and vision-assisted description model wrapper."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/anthropic_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/anthropic_model.py",
            header='"""Anthropic model wrapper extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/llama3_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/llama3_model.py",
            header='"""Llama 3 model wrapper extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/gpt4all_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/gpt4all_model.py",
            header='"""GPT4All model wrapper extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/tinyllama_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/tinyllama_model.py",
            header='"""TinyLlama model wrapper extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/text/no_ai_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/text/no_ai_model.py",
            header='"""Disabled AI model implementation."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/embedding/openai_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/openai_embedding_model.py",
            header='"""OpenAI embedding model wrapper."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/embedding/gpt4all_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/gpt4all_embedding_model.py",
            header='"""GPT4All/SentenceTransformer embedding model wrapper."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/embedding/tinyllama_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/tinyllama_embedding_model.py",
            header='"""TinyLlama embedding model wrapper."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/models/embedding/no_embedding_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/models/embedding/no_embedding_model.py",
            header='"""Disabled embedding model implementation."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/utils/model_diagnostics.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_diagnostics.py",
            header='"""Model diagnostics and test helpers extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/utils/model_recommendations.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_recommendations.py",
            header='"""Model recommendation and availability helpers."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/utils/model_downloads.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/model_downloads.py",
            header='"""Model download and local model discovery helpers."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/utils/embedding_storage.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/utils/embedding_storage.py",
            header='"""Embedding generation, storage, and similarity helpers."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/bootstrap/initialize_models.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/bootstrap/initialize_models.py",
            header='"""Bootstrap helpers for model config initialization and defaults."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/bootstrap/env_bootstrap.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/bootstrap/env_bootstrap.py",
            header='"""Environment bootstrap helpers extracted from legacy ai_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/legacy/compat_ai_models.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/legacy/compat_ai_models.py",
            header='"""Compatibility wrapper area for legacy imports during migration."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/services/image_handler_service.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/services/image_handler_service.py",
            header='"""Image handler service extracted from legacy plugins/image_modules/image_handler.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/base/base_image_model_handler.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/base/base_image_model_handler.py",
            header='"""Base abstract interface for image model handlers."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/models/clip_model_handler.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/models/clip_model_handler.py",
            header='"""CLIP image model handler extracted from legacy image_models.py."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/models/no_image_model.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/models/no_image_model.py",
            header='"""Disabled image model implementation."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/factories/model_handler_factory.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/factories/model_handler_factory.py",
            header='"""Factory helpers for resolving image model handler instances."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/config/image_runtime_config.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/config/image_runtime_config.py",
            header='"""Runtime image module configuration and preserved globals."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/bootstrap/image_env_bootstrap.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/bootstrap/image_env_bootstrap.py",
            header='"""Environment/bootstrap helpers for the image module migration."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
        "modules/ai/image/legacy/compat_image_modules.py": TargetFileSpec(
            path=PROJECT_ROOT / "modules/ai/image/legacy/compat_image_modules.py",
            header='"""Compatibility exports for legacy image module imports."""',
            imports=[COMMON_IMPORT_BLOCK],
        ),
    }


def build_file_initializers(specs: Dict[str, TargetFileSpec]) -> None:
    for spec in specs.values():
        ensure_parent(spec.path)


def generate_unmapped_review_file(source_name: str, unmapped: List[ExtractedNode]) -> str:
    sections = [
        f'"""Unmapped content extracted from legacy {source_name} for manual review."""',
        COMMON_IMPORT_BLOCK,
    ]

    for item in unmapped:
        sections.append(
            f"# ---- {item.node_type.upper()}: {item.name} "
            f"(lines {item.lineno}-{item.end_lineno}) ----"
        )
        sections.append(item.source.rstrip())

    return "\n\n\n".join(sections) + "\n"


def generate_duplicate_review_file(source_name: str, duplicates: List[ExtractedNode]) -> str:
    sections = [
        f'"""Shadowed duplicate top-level definitions from legacy {source_name}."""',
        COMMON_IMPORT_BLOCK,
        "# These earlier definitions were skipped because a later definition with the same",
        "# top-level symbol name wins at runtime in Python modules.",
    ]

    for item in duplicates:
        sections.append(
            f"# ---- SHADOWED {item.node_type.upper()}: {item.name} "
            f"(lines {item.lineno}-{item.end_lineno}) ----"
        )
        sections.append(item.source.rstrip())

    return "\n\n\n".join(sections) + "\n"


def generate_manifest(
    specs: Dict[str, TargetFileSpec],
    counts_by_source: Dict[str, Dict[str, int]],
    unmapped_by_source: Dict[str, List[ExtractedNode]],
    duplicate_defs_by_source: Dict[str, List[ExtractedNode]],
    mapping_records: List[MappingRecord],
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
            "total_shadowed_duplicate_defs": sum(len(v) for v in duplicate_defs_by_source.values()),
            "total_mapping_records": len(mapping_records),
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
        "shadowed_duplicate_defs_by_source": {
            source_name: [
                {
                    "name": item.name,
                    "node_type": item.node_type,
                    "lineno": item.lineno,
                    "end_lineno": item.end_lineno,
                }
                for item in items
            ]
            for source_name, items in duplicate_defs_by_source.items()
        },
        "mapping_records": [asdict(record) for record in mapping_records],
    }
    return json.dumps(payload, indent=2) + "\n"


def generate_overlap_report(mapping_records: List[MappingRecord]) -> str:
    by_symbol: Dict[Tuple[str, str], List[MappingRecord]] = {}
    by_target: Dict[str, List[MappingRecord]] = {}

    for record in mapping_records:
        symbol_key = (record.node_type, record.name)
        by_symbol.setdefault(symbol_key, []).append(record)

        if record.target_rel_path:
            by_target.setdefault(record.target_rel_path, []).append(record)

    symbol_collisions = []
    for (node_type, name), records in sorted(by_symbol.items()):
        distinct_targets = sorted(
            {r.target_rel_path for r in records if r.target_rel_path is not None}
        )
        distinct_sources = sorted({r.source_file_key for r in records})

        if len(records) > 1:
            symbol_collisions.append(
                {
                    "node_type": node_type,
                    "name": name,
                    "occurrences": len(records),
                    "distinct_sources": distinct_sources,
                    "distinct_targets": distinct_targets,
                    "records": [asdict(r) for r in records],
                }
            )

    target_overlaps = []
    for target_rel_path, records in sorted(by_target.items()):
        if len(records) > 1:
            target_overlaps.append(
                {
                    "target_rel_path": target_rel_path,
                    "count": len(records),
                    "records": [asdict(r) for r in records],
                }
            )

    payload = {
        "project_root": str(PROJECT_ROOT),
        "summary": {
            "mapping_records": len(mapping_records),
            "symbol_collision_groups": len(symbol_collisions),
            "target_overlap_groups": len(target_overlaps),
        },
        "symbol_collisions": symbol_collisions,
        "target_overlaps": target_overlaps,
    }
    return json.dumps(payload, indent=2) + "\n"


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

        from modules.ai.image.models.no_image_model import NoImageModel
        from modules.ai.image.models.clip_model_handler import CLIPModelHandler

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


        def load_ai_model(model_name=None):
            return ModelsConfig.load_ai_model(model_name)


        def load_embedding_model(model_name=None):
            return ModelsConfig.load_embedding_model(model_name)


        def load_image_model(model_name=None):
            return ModelsConfig.load_image_model(model_name)
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
    mapping_records: List[MappingRecord],
) -> Tuple[Dict[str, int], List[ExtractedNode], List[ExtractedNode]]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstring_src, imports, assigns, defs, duplicate_defs = collect_nodes(source_key, tree, source)
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

    for item in duplicate_defs:
        append_mapping_record(
            mapping_records,
            node=item,
            source_path=source_path,
            target_rel_path=None,
            status="skipped_duplicate",
            note="shadowed by later top-level definition with same symbol name",
        )

    for item in assigns:
        target_rel_path = AI_ASSIGN_TARGET_MAP.get(item.name)
        if target_rel_path:
            add_block(specs, target_rel_path, normalize_block(item.source))
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=target_rel_path,
                status="mapped",
            )
        else:
            unmapped.append(item)
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=None,
                status="unmapped",
            )

    for item in defs:
        if item.node_type == "class":
            if item.name in AI_CLASS_SKIP_IF_CANONICAL_IN_IMAGE:
                append_mapping_record(
                    mapping_records,
                    node=item,
                    source_path=source_path,
                    target_rel_path=None,
                    status="skipped_duplicate",
                    note="canonical image class is extracted from image_models.py",
                )
                continue

            target_rel_path = AI_CLASS_TARGET_MAP.get(item.name)
            if target_rel_path:
                add_block(specs, target_rel_path, normalize_block(item.source))
                append_mapping_record(
                    mapping_records,
                    node=item,
                    source_path=source_path,
                    target_rel_path=target_rel_path,
                    status="mapped",
                )
            else:
                unmapped.append(item)
                append_mapping_record(
                    mapping_records,
                    node=item,
                    source_path=source_path,
                    target_rel_path=None,
                    status="unmapped",
                )
            continue

        resolved = resolve_function_target(AI_FUNCTION_TARGET_MAP, item)
        if resolved:
            # Compat wrappers are generated explicitly, so do not inject extracted legacy versions.
            if resolved.rel_path == "modules/ai/legacy/compat_ai_models.py":
                append_mapping_record(
                    mapping_records,
                    node=item,
                    source_path=source_path,
                    target_rel_path=resolved.rel_path,
                    status="skipped_duplicate",
                    note="compat wrapper function generated explicitly; extracted legacy definition skipped",
                )
                continue

            add_block(specs, resolved.rel_path, normalize_block(item.source))
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=resolved.rel_path,
                status="mapped",
                note=resolved.note,
            )
        else:
            unmapped.append(item)
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=None,
                status="unmapped",
            )

    counts = {
        "imports_found": len(imports),
        "assignments_found": len(assigns),
        "definitions_found": len(defs),
        "shadowed_duplicate_defs": len(duplicate_defs),
        "unmapped_items": len(unmapped),
    }
    return counts, unmapped, duplicate_defs


def process_image_source(
    source_key: str,
    source_path: Path,
    specs: Dict[str, TargetFileSpec],
    mapping_records: List[MappingRecord],
) -> Tuple[Dict[str, int], List[ExtractedNode], List[ExtractedNode]]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstring_src, imports, assigns, defs, duplicate_defs = collect_nodes(source_key, tree, source)
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

    for item in duplicate_defs:
        append_mapping_record(
            mapping_records,
            node=item,
            source_path=source_path,
            target_rel_path=None,
            status="skipped_duplicate",
            note="shadowed by later top-level definition with same symbol name",
        )

    for item in assigns:
        target_rel_path = IMAGE_ASSIGN_TARGET_MAP.get(item.name)
        if target_rel_path:
            add_block(specs, target_rel_path, normalize_block(item.source))
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=target_rel_path,
                status="mapped",
            )
        else:
            unmapped.append(item)
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=None,
                status="unmapped",
            )

    for item in defs:
        target_rel_path = (
            IMAGE_CLASS_TARGET_MAP.get(item.name)
            if item.node_type == "class"
            else IMAGE_FUNCTION_TARGET_MAP.get(item.name)
        )

        if target_rel_path:
            add_block(specs, target_rel_path, normalize_block(item.source))
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=target_rel_path,
                status="mapped",
            )
        else:
            unmapped.append(item)
            append_mapping_record(
                mapping_records,
                node=item,
                source_path=source_path,
                target_rel_path=None,
                status="unmapped",
            )

    counts = {
        "imports_found": len(imports),
        "assignments_found": len(assigns),
        "definitions_found": len(defs),
        "shadowed_duplicate_defs": len(duplicate_defs),
        "unmapped_items": len(unmapped),
    }
    return counts, unmapped, duplicate_defs


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
    duplicate_defs_by_source: Dict[str, List[ExtractedNode]] = {}
    mapping_records: List[MappingRecord] = []

    for source_key, source_path in SOURCE_FILES.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        print("-" * 100)
        print(f"PROCESSING SOURCE: {source_key}")
        print(f"PATH: {source_path}")
        print("-" * 100)

        if source_key == "ai_models":
            counts, unmapped, duplicate_defs = process_ai_source(
                source_key=source_key,
                source_path=source_path,
                specs=specs,
                mapping_records=mapping_records,
            )
        else:
            counts, unmapped, duplicate_defs = process_image_source(
                source_key=source_key,
                source_path=source_path,
                specs=specs,
                mapping_records=mapping_records,
            )

        counts_by_source[source_key] = counts
        unmapped_by_source[source_key] = unmapped
        duplicate_defs_by_source[source_key] = duplicate_defs

    for spec in specs.values():
        if not spec.imports:
            spec.imports.append(COMMON_IMPORT_BLOCK)
        spec.imports[:] = dedupe_preserve_order(spec.imports)

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

    if WRITE_DUPLICATE_REVIEW_FILES:
        for source_key, duplicates in duplicate_defs_by_source.items():
            if not duplicates:
                continue

            review_path = REVIEW_DIR / f"shadowed_duplicates_from_{source_key}.py"
            review_text = generate_duplicate_review_file(source_key, duplicates)
            write_text(review_path, review_text, overwrite=True)

    if WRITE_MANIFEST_FILE:
        manifest_text = generate_manifest(
            specs=specs,
            counts_by_source=counts_by_source,
            unmapped_by_source=unmapped_by_source,
            duplicate_defs_by_source=duplicate_defs_by_source,
            mapping_records=mapping_records,
        )
        write_text(MANIFEST_FILE, manifest_text, overwrite=True)

    if WRITE_OVERLAP_REPORT:
        overlap_text = generate_overlap_report(mapping_records)
        write_text(OVERLAP_REPORT_FILE, overlap_text, overwrite=True)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    written_count = sum(1 for spec in specs.values() if spec.body_blocks)
    total_unmapped = sum(len(v) for v in unmapped_by_source.values())
    total_shadowed_duplicates = sum(len(v) for v in duplicate_defs_by_source.values())
    print(f"Files written with extracted content: {written_count}")
    print(f"Total unmapped items: {total_unmapped}")
    print(f"Total shadowed duplicate defs skipped: {total_shadowed_duplicates}")

    for source_key, unmapped in unmapped_by_source.items():
        review_path = REVIEW_DIR / f"unmapped_from_{source_key}.py"
        duplicate_review_path = REVIEW_DIR / f"shadowed_duplicates_from_{source_key}.py"

        print("-" * 100)
        print(f"SOURCE: {source_key}")
        print(f"Unmapped items: {len(unmapped)}")
        print(f"Shadowed duplicate defs skipped: {len(duplicate_defs_by_source.get(source_key, []))}")

        if unmapped:
            print(f"Unmapped review here: {review_path}")
            print("Unmapped names:")
            for item in unmapped:
                print(f"  - {item.node_type}: {item.name} ({item.lineno}-{item.end_lineno})")

        if duplicate_defs_by_source.get(source_key):
            print(f"Shadowed duplicate review here: {duplicate_review_path}")
            print("Shadowed names:")
            for item in duplicate_defs_by_source[source_key]:
                print(f"  - {item.node_type}: {item.name} ({item.lineno}-{item.end_lineno})")

    if WRITE_MANIFEST_FILE:
        print("-" * 100)
        print(f"Combined migration manifest written here: {MANIFEST_FILE}")

    if WRITE_OVERLAP_REPORT:
        print("-" * 100)
        print(f"Combined overlap report written here: {OVERLAP_REPORT_FILE}")

    print("=" * 100)
    print("IMPORTANT")
    print("=" * 100)
    print("1. This script extracts by top-level class/function/assignment name.")
    print("2. It handles ai_models.py, image_handler.py, and image_models.py in one run.")
    print("3. It writes review/manifest output under scripts/folder_outputs, not modules/ai.")
    print("4. It writes separate compatibility wrappers for ai and image.")
    print("5. It writes an overlap report so duplicate symbol moves are visible.")
    print("6. It does not rewrite imports inside extracted blocks.")
    print("7. Function target mappings support {'path', 'line'} metadata.")
    print("8. If a mapped function line number drifts, extraction still proceeds and logs the mismatch.")
    print("9. Earlier duplicate top-level defs are skipped; last-definition-wins matches Python module behavior.")
    print("10. The canonical image namespace is modules/ai/image/* only.")
    print("11. Compat loader functions are generated explicitly in modules/ai/legacy/compat_ai_models.py.")
    print("12. After extraction, do a second pass to clean imports and cross-file references.")
    print("=" * 100)


if __name__ == "__main__":
    main()