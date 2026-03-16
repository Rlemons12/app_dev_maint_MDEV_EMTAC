from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")

LEGACY_FILES = {
    "ai_models": PROJECT_ROOT / r"plugins\ai_modules\ai_models\ai_models.py",
    "image_handler": PROJECT_ROOT / r"plugins\image_modules\image_handler.py",
    "image_models": PROJECT_ROOT / r"plugins\image_modules\image_models.py",
}

NEW_MODULE_ROOT = PROJECT_ROOT / r"modules\ai"

OUTPUT_DIR = PROJECT_ROOT / r"scripts\folder_outputs"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "verification_report.json"
DEFAULT_REFACTOR_MAP_PATH = OUTPUT_DIR / "refactor_map.md"

DEFAULT_INCLUDE_ASSIGNMENTS = True
DEFAULT_FAIL_ON_MISSING = False
DEFAULT_FAIL_ON_DUPLICATES = False
DEFAULT_FAIL_ON_WRONG_TARGET = False
DEFAULT_WRITE_JSON = True
DEFAULT_WRITE_REFACTOR_MAP = True
DEFAULT_PRINT_FOUND = True
DEFAULT_IGNORE_COMPAT = True
DEFAULT_IGNORE_LEGACY_DIRS = True

ASSIGNMENTS_TO_TRACK = {
    "DEV_ENV",
    "PROJECT_ENV",
    "logger",
    "engine",
    "Session",
    "QUANTIZATION_AVAILABLE",
    "TORCH_COMPILE_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
    "MODEL_MINILM_DIR_ENV",
    "LOGGING_AVAILABLE",
    "ImageFile.LOAD_TRUNCATED_IMAGES",
}

# Expected destinations based on source file + symbol type + symbol name.
EXPECTED_TARGETS: Dict[Tuple[str, str, str], str] = {
    # ------------------------------------------------------------------
    # ai_models.py
    # ------------------------------------------------------------------
    ("ai_models", "assign", "DEV_ENV"): "modules/ai/bootstrap/env_bootstrap.py",
    ("ai_models", "assign", "PROJECT_ENV"): "modules/ai/bootstrap/env_bootstrap.py",
    ("ai_models", "assign", "engine"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "Session"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "QUANTIZATION_AVAILABLE"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "TORCH_COMPILE_AVAILABLE"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "TRANSFORMERS_AVAILABLE"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "MODEL_MINILM_DIR_ENV"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "LOGGING_AVAILABLE"): "modules/ai/config/models_config.py",
    ("ai_models", "assign", "logger"): "modules/ai/config/models_config.py",

    ("ai_models", "class", "ModelsConfig"): "modules/ai/config/models_config.py",
    ("ai_models", "class", "AIModel"): "modules/ai/base/interfaces.py",
    ("ai_models", "class", "EmbeddingModel"): "modules/ai/base/interfaces.py",
    ("ai_models", "class", "ImageModel"): "modules/ai/base/interfaces.py",
    ("ai_models", "class", "NoAIModel"): "modules/ai/models/text/no_ai_model.py",
    ("ai_models", "class", "AnthropicModel"): "modules/ai/models/text/anthropic_model.py",
    ("ai_models", "class", "OpenAIModel"): "modules/ai/models/text/openai_model.py",
    ("ai_models", "class", "Llama3Model"): "modules/ai/models/text/llama3_model.py",
    ("ai_models", "class", "GPT4AllModel"): "modules/ai/models/text/gpt4all_model.py",
    ("ai_models", "class", "TinyLlamaModel"): "modules/ai/models/text/tinyllama_model.py",
    ("ai_models", "class", "NoEmbeddingModel"): "modules/ai/models/embedding/no_embedding_model.py",
    ("ai_models", "class", "OpenAIEmbeddingModel"): "modules/ai/models/embedding/openai_embedding_model.py",
    ("ai_models", "class", "GPT4AllEmbeddingModel"): "modules/ai/models/embedding/gpt4all_embedding_model.py",
    ("ai_models", "class", "TinyLlamaEmbeddingModel"): "modules/ai/models/embedding/tinyllama_embedding_model.py",

    # Canonical image namespace is modules/ai/image/*
    ("ai_models", "class", "NoImageModel"): "modules/ai/image/models/no_image_model.py",
    ("ai_models", "class", "CLIPModelHandler"): "modules/ai/image/models/clip_model_handler.py",

    ("ai_models", "function", "register_default_models_with_tinyllama_updated"): "modules/ai/bootstrap/initialize_models.py",
    ("ai_models", "function", "test_tinyllama_framework_integration"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "test_tinyllama_embedding_functionality"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "register_default_models_with_tinyllama"): "modules/ai/bootstrap/initialize_models.py",
    ("ai_models", "function", "diagnose_tinyllama"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "test_tinyllama_functionality"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "diagnose_models"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "download_recommended_models"): "modules/ai/utils/model_downloads.py",
    ("ai_models", "function", "get_available_local_models"): "modules/ai/utils/model_downloads.py",
    ("ai_models", "function", "switch_to_local_models"): "modules/ai/utils/model_recommendations.py",
    ("ai_models", "function", "initialize_models_config"): "modules/ai/bootstrap/initialize_models.py",
    ("ai_models", "function", "register_default_models"): "modules/ai/bootstrap/initialize_models.py",
    ("ai_models", "function", "get_tinyllama_config"): "modules/ai/config/models_config.py",
    ("ai_models", "function", "update_tinyllama_config"): "modules/ai/config/models_config.py",
    ("ai_models", "function", "check_model_availability"): "modules/ai/utils/model_recommendations.py",
    ("ai_models", "function", "get_recommended_model_setup"): "modules/ai/utils/model_recommendations.py",
    ("ai_models", "function", "apply_recommended_models"): "modules/ai/utils/model_recommendations.py",
    ("ai_models", "function", "generate_embedding"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "store_embedding_enhanced"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "store_embedding"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "generate_and_store_embedding"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "get_current_models"): "modules/ai/config/models_config.py",
    ("ai_models", "function", "test_embedding_functionality"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "load_ai_model"): "modules/ai/legacy/compat_ai_models.py",
    ("ai_models", "function", "load_embedding_model"): "modules/ai/legacy/compat_ai_models.py",
    ("ai_models", "function", "load_image_model"): "modules/ai/legacy/compat_ai_models.py",
    ("ai_models", "function", "example_completeDocument_integration"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "example_model_configuration"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "search_similar_embeddings"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "get_embedding_with_similarity"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "get_pgvector_statistics"): "modules/ai/utils/embedding_storage.py",
    ("ai_models", "function", "test_pgvector_functionality"): "modules/ai/utils/model_diagnostics.py",
    ("ai_models", "function", "get_tinyllama_embedding_config"): "modules/ai/config/models_config.py",
    ("ai_models", "function", "configure_tinyllama_workflow"): "modules/ai/utils/model_recommendations.py",

    # ------------------------------------------------------------------
    # image_handler.py
    # ------------------------------------------------------------------
    ("image_handler", "class", "ImageHandler"): "modules/ai/image/services/image_handler_service.py",

    # ------------------------------------------------------------------
    # image_models.py
    # ------------------------------------------------------------------
    ("image_models", "assign", "logger"): "modules/ai/image/config/image_runtime_config.py",
    ("image_models", "assign", "engine"): "modules/ai/image/config/image_runtime_config.py",
    ("image_models", "assign", "Session"): "modules/ai/image/config/image_runtime_config.py",
    ("image_models", "assign", "ImageFile.LOAD_TRUNCATED_IMAGES"): "modules/ai/image/bootstrap/image_env_bootstrap.py",

    ("image_models", "class", "BaseImageModelHandler"): "modules/ai/image/base/base_image_model_handler.py",
    ("image_models", "class", "NoImageModel"): "modules/ai/image/models/no_image_model.py",
    ("image_models", "class", "CLIPModelHandler"): "modules/ai/image/models/clip_model_handler.py",

    ("image_models", "function", "get_image_model_handler"): "modules/ai/image/factories/model_handler_factory.py",
}

IGNORED_RELATIVE_PATH_PARTS = {
    "migration_review",
}

IGNORED_EXACT_RELATIVE_PATHS = {
    "modules/ai/legacy/compat_ai_models.py",
    "modules/ai/image/legacy/compat_image_modules.py",
}


@dataclass
class SymbolRecord:
    source_file_key: str
    source_file: str
    node_type: str  # class | function | assign
    name: str
    lineno: int
    end_lineno: int


@dataclass
class NewLocation:
    file_path: str
    node_type: str
    lineno: int
    end_lineno: int


@dataclass
class VerificationEntry:
    source_file_key: str
    source_file: str
    node_type: str
    name: str
    lineno: int
    end_lineno: int
    expected_target: str | None
    found: bool
    found_in_expected_target: bool
    matches: List[NewLocation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    legacy_files: Dict[str, str]
    new_module_root: str
    output_dir: str
    include_assignments: bool
    ignore_compat: bool
    ignore_legacy_dirs: bool
    tracked_assignments: List[str]
    selected_keys: List[str]
    total_legacy_symbols: int
    found_symbols: int
    missing_symbols: int
    duplicate_symbols: int
    wrong_target_symbols: int
    entries: List[VerificationEntry]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_python_file(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


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


def _relative_to_project(path_str: str) -> str:
    path = Path(path_str)
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _should_ignore_path(
    py_file: Path,
    *,
    ignore_compat: bool,
    ignore_legacy_dirs: bool,
) -> bool:
    rel = _relative_to_project(str(py_file))

    if ignore_compat and rel in {p.replace("\\", "/") for p in IGNORED_EXACT_RELATIVE_PATHS}:
        return True

    if ignore_legacy_dirs:
        parts = set(Path(rel).parts)
        if parts & IGNORED_RELATIVE_PATH_PARTS:
            return True

    return False


def collect_legacy_symbols(
    file_key: str,
    path: Path,
    *,
    include_assignments: bool,
    assignments_to_track: set[str],
) -> List[SymbolRecord]:
    tree = parse_python_file(path)
    symbols: List[SymbolRecord] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(
                SymbolRecord(
                    source_file_key=file_key,
                    source_file=str(path),
                    node_type="class",
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                SymbolRecord(
                    source_file_key=file_key,
                    source_file=str(path),
                    node_type="function",
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            )

        elif include_assignments and isinstance(node, (ast.Assign, ast.AnnAssign)):
            assign_name = get_assign_name(node)
            if assign_name in assignments_to_track:
                symbols.append(
                    SymbolRecord(
                        source_file_key=file_key,
                        source_file=str(path),
                        node_type="assign",
                        name=assign_name,
                        lineno=node.lineno,
                        end_lineno=node.end_lineno,
                    )
                )

    return symbols


def collect_new_symbols(
    root: Path,
    *,
    include_assignments: bool,
    assignments_to_track: set[str],
    ignore_compat: bool,
    ignore_legacy_dirs: bool,
) -> Dict[Tuple[str, str], List[NewLocation]]:
    index: Dict[Tuple[str, str], List[NewLocation]] = {}

    for py_file in root.rglob("*.py"):
        if _should_ignore_path(
            py_file,
            ignore_compat=ignore_compat,
            ignore_legacy_dirs=ignore_legacy_dirs,
        ):
            continue

        try:
            tree = parse_python_file(py_file)
        except SyntaxError as exc:
            print(f"[WARN] Skipping parse error in {py_file}: {exc}")
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                key = ("class", node.name)
                index.setdefault(key, []).append(
                    NewLocation(
                        file_path=str(py_file),
                        node_type="class",
                        lineno=node.lineno,
                        end_lineno=node.end_lineno,
                    )
                )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = ("function", node.name)
                index.setdefault(key, []).append(
                    NewLocation(
                        file_path=str(py_file),
                        node_type="function",
                        lineno=node.lineno,
                        end_lineno=node.end_lineno,
                    )
                )

            elif include_assignments and isinstance(node, (ast.Assign, ast.AnnAssign)):
                assign_name = get_assign_name(node)
                if assign_name in assignments_to_track:
                    key = ("assign", assign_name)
                    index.setdefault(key, []).append(
                        NewLocation(
                            file_path=str(py_file),
                            node_type="assign",
                            lineno=node.lineno,
                            end_lineno=node.end_lineno,
                        )
                    )

    return index


def verify_migration(
    legacy_files: Dict[str, Path],
    new_module_root: Path,
    *,
    include_assignments: bool,
    assignments_to_track: set[str],
    ignore_compat: bool,
    ignore_legacy_dirs: bool,
) -> VerificationReport:
    legacy_symbols: List[SymbolRecord] = []

    for file_key, file_path in legacy_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Legacy file not found: {file_path}")

        legacy_symbols.extend(
            collect_legacy_symbols(
                file_key,
                file_path,
                include_assignments=include_assignments,
                assignments_to_track=assignments_to_track,
            )
        )

    if not new_module_root.exists():
        raise FileNotFoundError(f"New module root not found: {new_module_root}")

    new_index = collect_new_symbols(
        new_module_root,
        include_assignments=include_assignments,
        assignments_to_track=assignments_to_track,
        ignore_compat=ignore_compat,
        ignore_legacy_dirs=ignore_legacy_dirs,
    )

    entries: List[VerificationEntry] = []
    found_count = 0
    missing_count = 0
    duplicate_count = 0
    wrong_target_count = 0

    for symbol in legacy_symbols:
        symbol_key = (symbol.source_file_key, symbol.node_type, symbol.name)
        matches = new_index.get((symbol.node_type, symbol.name), [])
        expected_target = EXPECTED_TARGETS.get(symbol_key)

        found_in_expected_target = False
        if expected_target and matches:
            expected_norm = expected_target.replace("\\", "/")
            for match in matches:
                rel = _relative_to_project(match.file_path)
                if rel == expected_norm:
                    found_in_expected_target = True
                    break
        elif matches and expected_target is None:
            # If no expected target is defined, treat any match as acceptable.
            found_in_expected_target = True

        entry = VerificationEntry(
            source_file_key=symbol.source_file_key,
            source_file=symbol.source_file,
            node_type=symbol.node_type,
            name=symbol.name,
            lineno=symbol.lineno,
            end_lineno=symbol.end_lineno,
            expected_target=expected_target,
            found=bool(matches),
            found_in_expected_target=found_in_expected_target,
            matches=matches,
            notes=[],
        )

        if matches:
            found_count += 1

            if len(matches) > 1:
                duplicate_count += 1
                entry.notes.append(
                    f"Symbol appears multiple times in modules/ai ({len(matches)} matches)"
                )

            if expected_target and not found_in_expected_target:
                wrong_target_count += 1
                entry.notes.append(
                    f"Symbol was found, but not in expected target: {expected_target}"
                )
        else:
            missing_count += 1
            entry.notes.append("No matching top-level symbol found in modules/ai")

        entries.append(entry)

    return VerificationReport(
        legacy_files={k: str(v) for k, v in legacy_files.items()},
        new_module_root=str(new_module_root),
        output_dir=str(OUTPUT_DIR),
        include_assignments=include_assignments,
        ignore_compat=ignore_compat,
        ignore_legacy_dirs=ignore_legacy_dirs,
        tracked_assignments=sorted(assignments_to_track),
        selected_keys=list(legacy_files.keys()),
        total_legacy_symbols=len(legacy_symbols),
        found_symbols=found_count,
        missing_symbols=missing_count,
        duplicate_symbols=duplicate_count,
        wrong_target_symbols=wrong_target_count,
        entries=entries,
    )


def write_report(report: VerificationReport, output_path: Path) -> None:
    ensure_parent(output_path)
    payload = asdict(report)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[WRITE] {output_path}")


def build_refactor_map_markdown(report: VerificationReport) -> str:
    lines: List[str] = []

    lines.append("# Refactor Map")
    lines.append("")
    lines.append("This report shows where each tracked top-level symbol was before and where it is now.")
    lines.append("")
    lines.append(f"- Project root: `{PROJECT_ROOT}`")
    lines.append(f"- New module root: `{report.new_module_root}`")
    lines.append(f"- Selected legacy files: `{', '.join(report.selected_keys)}`")
    lines.append(f"- Assignments included: `{report.include_assignments}`")
    lines.append(f"- Ignore compat files: `{report.ignore_compat}`")
    lines.append(f"- Ignore legacy/review dirs: `{report.ignore_legacy_dirs}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total legacy symbols: **{report.total_legacy_symbols}**")
    lines.append(f"- Found: **{report.found_symbols}**")
    lines.append(f"- Missing: **{report.missing_symbols}**")
    lines.append(f"- Duplicates: **{report.duplicate_symbols}**")
    lines.append(f"- Wrong target: **{report.wrong_target_symbols}**")
    lines.append("")

    grouped: Dict[str, List[VerificationEntry]] = {}
    for entry in report.entries:
        grouped.setdefault(entry.source_file_key, []).append(entry)

    for source_key in sorted(grouped):
        source_entries = sorted(
            grouped[source_key],
            key=lambda e: (e.node_type, e.name.lower(), e.lineno),
        )

        lines.append(f"## Source: `{source_key}`")
        lines.append("")
        if source_entries:
            source_file = _relative_to_project(source_entries[0].source_file)
            lines.append(f"Legacy file: `{source_file}`")
            lines.append("")

        lines.append("| Type | Name | Was Here | Expected | Now Here | Status |")
        lines.append("|---|---|---|---|---|---|")

        for entry in source_entries:
            was_here = f"`{_relative_to_project(entry.source_file)}:{entry.lineno}-{entry.end_lineno}`"
            expected = f"`{entry.expected_target}`" if entry.expected_target else "_none_"

            if entry.matches:
                now_here_parts = []
                for match in entry.matches:
                    now_here_parts.append(
                        f"`{_relative_to_project(match.file_path)}:{match.lineno}-{match.end_lineno}`"
                    )
                now_here = "<br>".join(now_here_parts)

                if len(entry.matches) > 1:
                    status = "duplicate"
                elif entry.expected_target and not entry.found_in_expected_target:
                    status = "wrong-target"
                else:
                    status = "found"
            else:
                now_here = "_missing_"
                status = "missing"

            lines.append(
                f"| {entry.node_type} | {entry.name} | {was_here} | {expected} | {now_here} | {status} |"
            )

        lines.append("")

    missing_entries = [e for e in report.entries if not e.found]
    duplicate_entries = [e for e in report.entries if len(e.matches) > 1]
    wrong_target_entries = [e for e in report.entries if e.found and e.expected_target and not e.found_in_expected_target]

    if missing_entries:
        lines.append("## Missing Symbols")
        lines.append("")
        for entry in sorted(missing_entries, key=lambda e: (e.source_file_key, e.node_type, e.name.lower())):
            lines.append(
                f"- `{entry.node_type}` `{entry.name}` from "
                f"`{_relative_to_project(entry.source_file)}:{entry.lineno}-{entry.end_lineno}`"
            )
        lines.append("")

    if wrong_target_entries:
        lines.append("## Wrong Target Symbols")
        lines.append("")
        for entry in sorted(wrong_target_entries, key=lambda e: (e.source_file_key, e.node_type, e.name.lower())):
            lines.append(
                f"- `{entry.node_type}` `{entry.name}` expected `{entry.expected_target}`"
            )
            for match in entry.matches:
                lines.append(
                    f"  - found at `{_relative_to_project(match.file_path)}:{match.lineno}-{match.end_lineno}`"
                )
        lines.append("")

    if duplicate_entries:
        lines.append("## Duplicate Matches")
        lines.append("")
        for entry in sorted(duplicate_entries, key=lambda e: (e.node_type, e.name.lower())):
            lines.append(
                f"- `{entry.node_type}` `{entry.name}` has **{len(entry.matches)}** matches:"
            )
            for match in entry.matches:
                lines.append(
                    f"  - `{_relative_to_project(match.file_path)}:{match.lineno}-{match.end_lineno}`"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_refactor_map(report: VerificationReport, output_path: Path) -> None:
    ensure_parent(output_path)
    text = build_refactor_map_markdown(report)
    output_path.write_text(text, encoding="utf-8")
    print(f"[WRITE] {output_path}")


def print_report(report: VerificationReport, *, print_found: bool = True) -> None:
    print("=" * 120)
    print("LEGACY -> modules/ai VERIFICATION REPORT")
    print("=" * 120)
    print(f"Selected files   : {', '.join(report.selected_keys)}")
    print(f"New module root  : {report.new_module_root}")
    print(f"Output dir       : {report.output_dir}")
    print(f"Assignments      : {'enabled' if report.include_assignments else 'disabled'}")
    print(f"Ignore compat    : {report.ignore_compat}")
    print(f"Ignore legacy    : {report.ignore_legacy_dirs}")
    print(f"Total symbols    : {report.total_legacy_symbols}")
    print(f"Found            : {report.found_symbols}")
    print(f"Missing          : {report.missing_symbols}")
    print(f"Duplicates       : {report.duplicate_symbols}")
    print(f"Wrong target     : {report.wrong_target_symbols}")
    print("=" * 120)

    missing_entries = [e for e in report.entries if not e.found]
    duplicate_entries = [e for e in report.entries if len(e.matches) > 1]
    wrong_target_entries = [e for e in report.entries if e.found and e.expected_target and not e.found_in_expected_target]
    found_entries = [e for e in report.entries if e.found]

    if missing_entries:
        print("MISSING SYMBOLS")
        print("-" * 120)
        for entry in sorted(missing_entries, key=lambda x: (x.source_file_key, x.node_type, x.name)):
            print(
                f"[MISSING] {entry.node_type:<12} {entry.name:<42} "
                f"from {entry.source_file_key} ({entry.lineno}-{entry.end_lineno})"
            )
        print("-" * 120)

    if wrong_target_entries:
        print("WRONG TARGET SYMBOLS")
        print("-" * 120)
        for entry in sorted(wrong_target_entries, key=lambda x: (x.source_file_key, x.node_type, x.name)):
            print(
                f"[WRONG-TARGET] {entry.node_type:<12} {entry.name:<42} "
                f"expected={entry.expected_target}"
            )
            for match in entry.matches:
                print(f"    -> {match.file_path} ({match.lineno}-{match.end_lineno})")
        print("-" * 120)

    if duplicate_entries:
        print("DUPLICATE TARGET SYMBOLS")
        print("-" * 120)
        for entry in sorted(duplicate_entries, key=lambda x: (x.node_type, x.name)):
            print(
                f"[DUPLICATE] {entry.node_type:<12} {entry.name:<42} "
                f"matches={len(entry.matches)}"
            )
            for match in entry.matches:
                print(f"    -> {match.file_path} ({match.lineno}-{match.end_lineno})")
        print("-" * 120)

    if print_found:
        print("FOUND SYMBOLS")
        print("-" * 120)
        for entry in sorted(found_entries, key=lambda x: (x.source_file_key, x.node_type, x.name)):
            marker = "[FOUND]"
            if entry.expected_target and not entry.found_in_expected_target:
                marker = "[FOUND-WRONG-TARGET]"
            elif len(entry.matches) > 1:
                marker = "[FOUND-DUPLICATE]"

            print(
                f"{marker} {entry.node_type:<12} {entry.name:<42} "
                f"from {entry.source_file_key}"
            )
            for match in entry.matches:
                print(f"    -> {match.file_path} ({match.lineno}-{match.end_lineno})")
        print("-" * 120)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that legacy symbols from selected legacy files now exist under modules/ai, "
            "and optionally write a JSON report and Markdown refactor map."
        )
    )

    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        choices=sorted(LEGACY_FILES.keys()),
        help=(
            "Select one or more legacy file keys to verify. "
            "Valid values: ai_models, image_handler, image_models. "
            "Can be passed multiple times."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all legacy files listed in LEGACY_FILES.",
    )

    parser.add_argument(
        "--new-root",
        type=Path,
        default=NEW_MODULE_ROOT,
        help=f"Override modules/ai root. Default: {NEW_MODULE_ROOT}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Base output directory for generated files. Default: {OUTPUT_DIR}",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to write JSON report. "
            f"Default: <output-dir>\\{DEFAULT_REPORT_PATH.name}"
        ),
    )

    parser.add_argument(
        "--refactor-map",
        type=Path,
        default=None,
        help=(
            "Path to write Markdown refactor map. "
            f"Default: <output-dir>\\{DEFAULT_REFACTOR_MAP_PATH.name}"
        ),
    )

    parser.add_argument(
        "--write-json",
        dest="write_json",
        action="store_true",
        default=DEFAULT_WRITE_JSON,
        help="Write JSON verification report.",
    )

    parser.add_argument(
        "--no-write-json",
        dest="write_json",
        action="store_false",
        help="Do not write JSON verification report.",
    )

    parser.add_argument(
        "--write-refactor-map",
        dest="write_refactor_map",
        action="store_true",
        default=DEFAULT_WRITE_REFACTOR_MAP,
        help="Write Markdown refactor map.",
    )

    parser.add_argument(
        "--no-write-refactor-map",
        dest="write_refactor_map",
        action="store_false",
        help="Do not write Markdown refactor map.",
    )

    parser.add_argument(
        "--include-assignments",
        dest="include_assignments",
        action="store_true",
        default=DEFAULT_INCLUDE_ASSIGNMENTS,
        help="Include tracked top-level assignments in verification.",
    )

    parser.add_argument(
        "--no-assignments",
        dest="include_assignments",
        action="store_false",
        help="Disable assignment verification and only verify classes/functions.",
    )

    parser.add_argument(
        "--ignore-compat",
        dest="ignore_compat",
        action="store_true",
        default=DEFAULT_IGNORE_COMPAT,
        help="Ignore compatibility wrapper files during verification.",
    )

    parser.add_argument(
        "--include-compat",
        dest="ignore_compat",
        action="store_false",
        help="Include compatibility wrapper files during verification.",
    )

    parser.add_argument(
        "--ignore-legacy-dirs",
        dest="ignore_legacy_dirs",
        action="store_true",
        default=DEFAULT_IGNORE_LEGACY_DIRS,
        help="Ignore legacy/review-like directories such as migration_review.",
    )

    parser.add_argument(
        "--include-legacy-dirs",
        dest="ignore_legacy_dirs",
        action="store_false",
        help="Include legacy/review-like directories during verification.",
    )

    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        default=DEFAULT_FAIL_ON_MISSING,
        help="Exit with code 1 if any symbols are missing.",
    )

    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        default=DEFAULT_FAIL_ON_DUPLICATES,
        help="Exit with code 1 if duplicate symbols are found.",
    )

    parser.add_argument(
        "--fail-on-wrong-target",
        action="store_true",
        default=DEFAULT_FAIL_ON_WRONG_TARGET,
        help="Exit with code 1 if a symbol is found outside its expected target.",
    )

    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print available legacy file keys and exit.",
    )

    parser.add_argument(
        "--print-found",
        dest="print_found",
        action="store_true",
        default=DEFAULT_PRINT_FOUND,
        help="Print found symbols to console.",
    )

    parser.add_argument(
        "--no-print-found",
        dest="print_found",
        action="store_false",
        help="Do not print found symbols to console. Missing, wrong-target, and duplicates still print.",
    )

    return parser


def resolve_selected_legacy_files(args: argparse.Namespace) -> Dict[str, Path]:
    if args.list_files:
        print("Available legacy file keys:")
        for key, path in LEGACY_FILES.items():
            print(f"  - {key}: {path}")
        raise SystemExit(0)

    if args.all:
        return dict(LEGACY_FILES)

    selected = args.files or []

    if not selected:
        raise SystemExit(
            "No legacy files selected. Use --all or one/more --file values "
            "(for example: --file ai_models --file image_models)."
        )

    resolved: Dict[str, Path] = {}
    for key in selected:
        resolved[key] = LEGACY_FILES[key]

    return resolved


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir
    report_path = args.report or (output_dir / DEFAULT_REPORT_PATH.name)
    refactor_map_path = args.refactor_map or (output_dir / DEFAULT_REFACTOR_MAP_PATH.name)
    return report_path, refactor_map_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    legacy_files = resolve_selected_legacy_files(args)
    report_path, refactor_map_path = resolve_output_paths(args)

    report = verify_migration(
        legacy_files=legacy_files,
        new_module_root=args.new_root,
        include_assignments=args.include_assignments,
        assignments_to_track=set(ASSIGNMENTS_TO_TRACK),
        ignore_compat=args.ignore_compat,
        ignore_legacy_dirs=args.ignore_legacy_dirs,
    )

    print_report(report, print_found=args.print_found)

    if args.write_json:
        write_report(report, report_path)

    if args.write_refactor_map:
        write_refactor_map(report, refactor_map_path)

    if args.fail_on_missing and report.missing_symbols > 0:
        return 1

    if args.fail_on_duplicates and report.duplicate_symbols > 0:
        return 1

    if args.fail_on_wrong_target and report.wrong_target_symbols > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())