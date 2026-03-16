from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")

SOURCE_FILES = {
    "image_handler": PROJECT_ROOT / r"plugins\image_modules\image_handler.py",
    "image_models": PROJECT_ROOT / r"plugins\image_modules\image_models.py",
}

OUTPUT_BASE = PROJECT_ROOT

OVERWRITE_EXISTING = False
WRITE_UNMAPPED_REVIEW_FILES = True
WRITE_MANIFEST_FILE = True

REVIEW_DIR = PROJECT_ROOT / r"modules\ai\migration_review"
MANIFEST_FILE = REVIEW_DIR / "image_migration_manifest.json"


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
            assign_name = "<assign>"

            if isinstance(node, ast.Assign) and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    assign_name = target.id
                elif isinstance(target, ast.Attribute):
                    assign_name = ast.unparse(target)
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name):
                    assign_name = target.id
                elif isinstance(target, ast.Attribute):
                    assign_name = ast.unparse(target)

            assigns.append(
                ExtractedNode(
                    source_file_key=source_file_key,
                    name=assign_name,
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


CLASS_TARGET_MAP = {
    "ImageHandler": "modules/ai/image/services/image_handler_service.py",
    "BaseImageModelHandler": "modules/ai/image/base/base_image_model_handler.py",
    "NoImageModel": "modules/ai/image/models/no_image_model.py",
    "CLIPModelHandler": "modules/ai/image/models/clip_model_handler.py",
}

FUNCTION_TARGET_MAP = {
    "get_image_model_handler": "modules/ai/image/factories/model_handler_factory.py",
}

ASSIGN_NAMES_TO_PRESERVE = {
    "logger",
    "engine",
    "Session",
    "ImageFile.LOAD_TRUNCATED_IMAGES",
}

ASSIGN_TARGET_MAP = {
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
        "modules/ai/image/__init__.py",
        "modules/ai/image/base/__init__.py",
        "modules/ai/image/bootstrap/__init__.py",
        "modules/ai/image/config/__init__.py",
        "modules/ai/image/factories/__init__.py",
        "modules/ai/image/legacy/__init__.py",
        "modules/ai/image/models/__init__.py",
        "modules/ai/image/services/__init__.py",
        "modules/ai/migration_review/__init__.py",
    ]

    for rel in packages:
        path = PROJECT_ROOT / rel
        if not path.exists():
            write_text(path, '"""Package marker."""\n', overwrite=False)


def create_compat_wrapper_file() -> str:
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


def process_source(
    source_key: str,
    source_path: Path,
    specs: Dict[str, TargetFileSpec],
) -> Tuple[Dict[str, int], List[ExtractedNode]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    print("-" * 100)
    print(f"PROCESSING SOURCE: {source_key}")
    print(f"PATH: {source_path}")
    print("-" * 100)

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
        if item.name in ASSIGN_NAMES_TO_PRESERVE and item.name in ASSIGN_TARGET_MAP:
            add_block(specs, ASSIGN_TARGET_MAP[item.name], normalize_block(item.source))
        else:
            unmapped.append(item)

    for item in defs:
        if item.node_type == "class":
            target = CLASS_TARGET_MAP.get(item.name)
            if target:
                add_block(specs, target, normalize_block(item.source))
            else:
                unmapped.append(item)
        elif item.node_type == "function":
            target = FUNCTION_TARGET_MAP.get(item.name)
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
    print("MIGRATING LEGACY IMAGE MODULES INTO NEW MODULE STRUCTURE")
    print("=" * 100)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OVERWRITE_EXISTING: {OVERWRITE_EXISTING}")
    print("=" * 100)

    create_package_files()

    specs = build_target_specs()
    build_file_initializers(specs)

    counts_by_source: Dict[str, Dict[str, int]] = {}
    unmapped_by_source: Dict[str, List[ExtractedNode]] = {}

    for source_key, source_path in SOURCE_FILES.items():
        counts, unmapped = process_source(source_key, source_path, specs)
        counts_by_source[source_key] = counts
        unmapped_by_source[source_key] = unmapped

    for spec in specs.values():
        if not spec.imports:
            spec.imports.append(COMMON_IMPORT_BLOCK)

    for rel_path, spec in specs.items():
        if rel_path == "modules/ai/image/legacy/compat_image_modules.py":
            rendered = create_compat_wrapper_file()
            write_text(spec.path, rendered, overwrite=True)
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
        print(f"Migration manifest written here: {MANIFEST_FILE}")

    print("=" * 100)
    print("IMPORTANT")
    print("=" * 100)
    print("1. This script extracts by top-level class/function/assignment name.")
    print("2. It handles BOTH image_handler.py and image_models.py in one run.")
    print("3. It preserves important runtime assignments like Session, engine, logger.")
    print("4. It writes separate unmapped review files so nothing silently disappears.")
    print("5. It writes a compatibility wrapper for the new image module layout.")
    print("6. After extraction, do a second pass to clean imports and cross-file references.")
    print("=" * 100)


if __name__ == "__main__":
    main()