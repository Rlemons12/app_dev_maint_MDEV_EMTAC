"""
modules/database_manager/services/drawing_file_sync_service.py

Folder-first Drawing file sync service.

Purpose:
    1. Scan the configured drawing folder first.
    2. For each physical drawing file, look for a matching Drawing row in the database.
    3. Update Drawing.file_path when exactly one safe database match is found.
    4. Report files that have no matching DB record.
    5. Report ambiguous DB matches without updating them.
    6. Optionally create missing DB records, but this is disabled by default.

Expected .env/config value:
    DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWINNG

Recommended command examples:

    Dry run only, no database updates:
        python -m modules.database_manager.services.drawing_file_sync_service --show-results

    Apply safe updates:
        python -m modules.database_manager.services.drawing_file_sync_service --apply --show-results

    Apply safe updates and save JSON report:
        python -m modules.database_manager.services.drawing_file_sync_service --apply --report-json "E:\\emtac\\logs\\drawing_file_sync_report.json"

    Create missing Drawing records from unmatched files:
        python -m modules.database_manager.services.drawing_file_sync_service --apply --create-missing

Notes:
    - Dry run is the default.
    - Folder-first means the physical drawing files drive the sync.
    - The service updates only when one safe database match is found.
    - Ambiguous matches are reported but not updated unless --prefer-first-on-ambiguous is used.
    - Creating missing DB records is optional and disabled by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Drawing

try:
    from modules.configuration.config import DATABASE_DRAWING as CONFIG_DATABASE_DRAWING
except Exception:
    CONFIG_DATABASE_DRAWING = None

try:
    from modules.configuration.log_config import error_id, info_id, with_request_id
except Exception:
    def info_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[INFO] {message}")

    def error_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)

    def with_request_id(func):
        return func


@dataclass
class DrawingFileSyncResult:
    """
    One result per physical file found in the drawing folder.
    """

    file_path: str
    file_name: str
    file_stem: str
    file_extension: str

    drawing_id: Optional[int]
    drw_number: Optional[str]
    drw_name: Optional[str]
    old_file_path: Optional[str]
    matched_file_path: Optional[str]

    status: str
    match_reason: str
    message: str

    candidate_count: int = 0
    ambiguous_drawing_ids: List[int] = field(default_factory=list)
    ambiguous_drawings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DrawingFileSyncSummary:
    """
    Summary for the folder-first sync run.
    """

    dry_run: bool
    drawing_root: str
    recursive: bool

    total_files_scanned: int = 0
    total_db_drawings: int = 0

    current_exists: int = 0
    updated: int = 0
    would_update: int = 0

    no_db_match: int = 0
    ambiguous: int = 0

    created: int = 0
    would_create: int = 0

    skipped: int = 0
    errors: int = 0

    results: List[DrawingFileSyncResult] = field(default_factory=list)

    def add_result(self, result: DrawingFileSyncResult) -> None:
        self.results.append(result)

        if result.status == "current_exists":
            self.current_exists += 1
        elif result.status == "updated":
            self.updated += 1
        elif result.status == "would_update":
            self.would_update += 1
        elif result.status == "no_db_match":
            self.no_db_match += 1
        elif result.status == "ambiguous":
            self.ambiguous += 1
        elif result.status == "created":
            self.created += 1
        elif result.status == "would_create":
            self.would_create += 1
        elif result.status == "skipped":
            self.skipped += 1
        elif result.status == "error":
            self.errors += 1

    # Backward-compatible aliases for older orchestrator/report expectations.
    @property
    def missing(self) -> int:
        return self.no_db_match

    def to_dict(self, include_results: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        data["missing"] = self.no_db_match

        if not include_results:
            data.pop("results", None)

        return data


@dataclass
class DrawingDbIndex:
    """
    Database lookup index for matching files to Drawing rows.
    """

    drawings: List[Drawing] = field(default_factory=list)

    by_normalized_file_path: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_file_name: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_file_stem: Dict[str, List[Drawing]] = field(default_factory=dict)

    by_drw_number: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_drw_name: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_spare_part_number: Dict[str, List[Drawing]] = field(default_factory=dict)

    by_compact_file_stem: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_compact_drw_number: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_compact_drw_name: Dict[str, List[Drawing]] = field(default_factory=dict)
    by_compact_spare_part_number: Dict[str, List[Drawing]] = field(default_factory=dict)


@dataclass
class DrawingDbMatch:
    drawing: Optional[Drawing]
    reason: str
    candidate_count: int
    ambiguous_drawings: List[Drawing] = field(default_factory=list)


class DrawingFileSyncService:
    """
    Folder-first service for reconciling physical drawing files with Drawing records.

    Main flow:
        physical file -> database match -> update Drawing.file_path
    """

    DEFAULT_EXTENSIONS: Tuple[str, ...] = (
        ".dwg",
        ".dxf",
        ".slddrw",
        ".pdf",
    )

    def __init__(
            self,
            db_config: Optional[DatabaseConfig] = None,
            drawing_root: Optional[str] = None,
            allowed_extensions: Optional[Sequence[str]] = None,
            recursive: bool = True,
            all_files: bool = False,
            request_id: Optional[str] = None,
    ) -> None:
        self.db_config = db_config or DatabaseConfig()
        self.request_id = request_id
        self.recursive = recursive
        self.all_files = all_files

        if all_files:
            self.allowed_extensions = None
        else:
            self.allowed_extensions = self._normalize_extensions(
                allowed_extensions or self.DEFAULT_EXTENSIONS
            )

        self.drawing_root = self._resolve_drawing_root(drawing_root)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_text(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _key(value: str) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _compact_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    @staticmethod
    def _path_key(path: Path) -> str:
        return os.path.normcase(str(path.resolve()))

    @staticmethod
    def _dedupe_drawings(drawings: Iterable[Drawing]) -> List[Drawing]:
        seen: Set[int] = set()
        output: List[Drawing] = []

        for drawing in drawings:
            drawing_id = int(drawing.id)

            if drawing_id in seen:
                continue

            seen.add(drawing_id)
            output.append(drawing)

        return output

    @staticmethod
    def _normalize_extensions(extensions: Sequence[str]) -> Tuple[str, ...]:
        normalized: List[str] = []

        for ext in extensions:
            if ext is None:
                continue

            value = str(ext).strip().lower()

            if not value:
                continue

            if not value.startswith("."):
                value = f".{value}"

            normalized.append(value)

        return tuple(dict.fromkeys(normalized))

    def _resolve_drawing_root(self, drawing_root: Optional[str]) -> Path:
        configured_root = (
            drawing_root
            or CONFIG_DATABASE_DRAWING
            or os.getenv("DATABASE_DRAWING")
        )

        if not configured_root or not str(configured_root).strip():
            raise ValueError(
                "DATABASE_DRAWING is not configured. "
                "Add this to your .env/config: "
                "DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWINNG"
            )

        root_path = Path(str(configured_root)).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)

        return root_path

    def resolve_database_file_path(self, file_path: Optional[str]) -> Optional[Path]:
        """
        Resolve a Drawing.file_path value.

        If the DB path is absolute, normalize it.
        If the DB path is relative, resolve it under DATABASE_DRAWING.
        """
        if file_path is None or not str(file_path).strip():
            return None

        raw_path = Path(str(file_path).strip()).expanduser()

        if raw_path.is_absolute():
            return raw_path.resolve()

        return (self.drawing_root / raw_path).resolve()

    # ------------------------------------------------------------------
    # Folder scan
    # ------------------------------------------------------------------

    def scan_drawing_folder(self, limit: Optional[int] = None) -> List[Path]:
        """
        Walk DATABASE_DRAWING and return physical drawing files.

        When self.recursive=True:
            Walks the drawing root and all subfolders.

        When self.recursive=False:
            Only scans the top level of the drawing root.

        This is folder-first:
            physical files -> match to database Drawing rows
        """
        if not self.drawing_root.exists():
            raise FileNotFoundError(f"Drawing root does not exist: {self.drawing_root}")

        if not self.drawing_root.is_dir():
            raise NotADirectoryError(f"Drawing root is not a folder: {self.drawing_root}")

        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1 when provided.")

        files: List[Path] = []

        if self.recursive:
            for root, dirs, filenames in os.walk(self.drawing_root):
                # Sort for repeatable results
                dirs.sort()
                filenames.sort()

                root_path = Path(root)

                for filename in filenames:
                    if filename.startswith("~$"):
                        continue

                    file_path = root_path / filename

                    if not file_path.is_file():
                        continue

                    if self.allowed_extensions is not None and file_path.suffix.lower() not in self.allowed_extensions:
                        continue

                    files.append(file_path.resolve())

                    if limit is not None and len(files) >= limit:
                        info_id(
                            (
                                f"Walked drawing root '{self.drawing_root}' recursively "
                                f"and stopped at limit={limit}. "
                                f"Files found={len(files)}."
                            ),
                            self.request_id,
                        )
                        return files

        else:
            for file_path in sorted(self.drawing_root.iterdir()):
                if not file_path.is_file():
                    continue

                if file_path.name.startswith("~$"):
                    continue

                if self.allowed_extensions is not None and file_path.suffix.lower() not in self.allowed_extensions:
                    continue

                files.append(file_path.resolve())

                if limit is not None and len(files) >= limit:
                    info_id(
                        (
                            f"Scanned top level of drawing root '{self.drawing_root}' "
                            f"and stopped at limit={limit}. "
                            f"Files found={len(files)}."
                        ),
                        self.request_id,
                    )
                    return files

        info_id(
            (
                f"Walked drawing root '{self.drawing_root}'. "
                f"recursive={self.recursive}. "
                f"Files found={len(files)}."
            ),
            self.request_id,
        )

        return files

    # ------------------------------------------------------------------
    # Database index
    # ------------------------------------------------------------------

    def build_db_index(self, session) -> DrawingDbIndex:
        """
        Load all Drawing records and build lookup indexes.
        """
        index = DrawingDbIndex()
        index.drawings = session.query(Drawing).order_by(Drawing.id).all()

        for drawing in index.drawings:
            self._add_drawing_to_index(index, drawing)

        info_id(
            f"Loaded {len(index.drawings)} Drawing records into DB match index.",
            self.request_id,
        )

        return index

    def _add_to_index(
        self,
        dictionary: Dict[str, List[Drawing]],
        key_value: Optional[str],
        drawing: Drawing,
    ) -> None:
        text = self._safe_text(key_value)

        if not text:
            return

        key = self._key(text)

        if not key:
            return

        dictionary.setdefault(key, []).append(drawing)

    def _add_compact_to_index(
        self,
        dictionary: Dict[str, List[Drawing]],
        key_value: Optional[str],
        drawing: Drawing,
    ) -> None:
        text = self._safe_text(key_value)

        if not text:
            return

        key = self._compact_key(text)

        if not key:
            return

        dictionary.setdefault(key, []).append(drawing)

    def _add_drawing_to_index(self, index: DrawingDbIndex, drawing: Drawing) -> None:
        """
        Add one Drawing row to the in-memory database matching index.

        Important:
            The file_path column is only indexed when it looks like a real file path.

            This prevents placeholder values like:
                active_drawing_list_import

            from being treated as actual drawing file paths.

        The Drawing row can still be matched by:
            - drw_number
            - drw_name
            - drw_spare_part_number
            - compact versions of those fields
        """
        old_file_path = self._safe_text(getattr(drawing, "file_path", None))
        drw_number = self._safe_text(getattr(drawing, "drw_number", None))
        drw_name = self._safe_text(getattr(drawing, "drw_name", None))
        spare_part_number = self._safe_text(
            getattr(drawing, "drw_spare_part_number", None)
        )

        # ------------------------------------------------------------
        # Only index file_path when it looks like a real file path.
        # This prevents placeholder values like:
        # active_drawing_list_import
        # from being used as a file path match.
        # ------------------------------------------------------------
        if self._file_path_looks_real(old_file_path):
            resolved_db_path = self.resolve_database_file_path(old_file_path)

            if resolved_db_path is not None:
                index.by_normalized_file_path.setdefault(
                    self._path_key(resolved_db_path),
                    [],
                ).append(drawing)

                self._add_to_index(
                    index.by_file_name,
                    resolved_db_path.name,
                    drawing,
                )

                self._add_to_index(
                    index.by_file_stem,
                    resolved_db_path.stem,
                    drawing,
                )

                self._add_compact_to_index(
                    index.by_compact_file_stem,
                    resolved_db_path.stem,
                    drawing,
                )

        # ------------------------------------------------------------
        # Always index drawing metadata.
        # This allows a physical file name to match the DB row even when
        # file_path currently contains a placeholder.
        # ------------------------------------------------------------
        self._add_to_index(index.by_drw_number, drw_number, drawing)
        self._add_to_index(index.by_drw_name, drw_name, drawing)
        self._add_to_index(index.by_spare_part_number, spare_part_number, drawing)

        self._add_compact_to_index(
            index.by_compact_drw_number,
            drw_number,
            drawing,
        )

        self._add_compact_to_index(
            index.by_compact_drw_name,
            drw_name,
            drawing,
        )

        self._add_compact_to_index(
            index.by_compact_spare_part_number,
            spare_part_number,
            drawing,
        )

    PLACEHOLDER_FILE_PATH_VALUES = {
        "active_drawing_list_import",
        "none",
        "null",
        "n/a",
        "na",
        "",
    }

    def _file_path_looks_real(self, file_path: Optional[str]) -> bool:
        """
        Return True only when Drawing.file_path looks like an actual file path.

        This prevents placeholder values from being used as path matches.
        """
        text = self._safe_text(file_path)

        if not text:
            return False

        lowered = text.lower().strip()

        if lowered in self.PLACEHOLDER_FILE_PATH_VALUES:
            return False

        path = Path(text)

        # Absolute paths are real path candidates.
        if path.is_absolute():
            return True

        # Relative paths with folder separators are real path candidates.
        if "\\" in text or "/" in text:
            return True

        # File names with supported drawing extensions are real path candidates.
        if path.suffix.lower() in self.allowed_extensions:
            return True

        return False

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _select_match(
        self,
        candidates: Sequence[Drawing],
        reason: str,
        prefer_first_on_ambiguous: bool,
    ) -> DrawingDbMatch:
        unique = self._dedupe_drawings(candidates)

        if not unique:
            return DrawingDbMatch(
                drawing=None,
                reason="no_match",
                candidate_count=0,
                ambiguous_drawings=[],
            )

        if len(unique) == 1:
            return DrawingDbMatch(
                drawing=unique[0],
                reason=reason,
                candidate_count=1,
                ambiguous_drawings=[],
            )

        if prefer_first_on_ambiguous:
            return DrawingDbMatch(
                drawing=unique[0],
                reason=f"{reason}:preferred_first",
                candidate_count=len(unique),
                ambiguous_drawings=[],
            )

        return DrawingDbMatch(
            drawing=None,
            reason=reason,
            candidate_count=len(unique),
            ambiguous_drawings=unique,
        )

    def match_file_to_db(
        self,
        file_path: Path,
        db_index: DrawingDbIndex,
        use_compact_match: bool = True,
        prefer_first_on_ambiguous: bool = False,
    ) -> DrawingDbMatch:
        """
        Match one physical file to a Drawing row.

        Match priority:
            1. Exact normalized file_path match
            2. Existing DB file_path filename match
            3. Existing DB file_path stem match
            4. drw_number exact match to file stem
            5. drw_name exact match to file stem
            6. spare part exact match to file stem
            7. Compact/loose versions of above
        """
        resolved_file = file_path.resolve()

        exact_path_candidates = db_index.by_normalized_file_path.get(
            self._path_key(resolved_file),
            [],
        )
        if exact_path_candidates:
            return self._select_match(
                exact_path_candidates,
                "exact_normalized_file_path",
                prefer_first_on_ambiguous,
            )

        file_name_key = self._key(resolved_file.name)
        file_stem_key = self._key(resolved_file.stem)
        compact_stem_key = self._compact_key(resolved_file.stem)

        lookup_steps: List[Tuple[str, List[Drawing]]] = [
            (
                "database_file_name_equals_physical_file_name",
                db_index.by_file_name.get(file_name_key, []),
            ),
            (
                "database_file_stem_equals_physical_file_stem",
                db_index.by_file_stem.get(file_stem_key, []),
            ),
            (
                "drw_number_equals_physical_file_stem",
                db_index.by_drw_number.get(file_stem_key, []),
            ),
            (
                "drw_name_equals_physical_file_stem",
                db_index.by_drw_name.get(file_stem_key, []),
            ),
            (
                "spare_part_number_equals_physical_file_stem",
                db_index.by_spare_part_number.get(file_stem_key, []),
            ),
        ]

        if use_compact_match:
            lookup_steps.extend(
                [
                    (
                        "compact_database_file_stem_equals_compact_physical_file_stem",
                        db_index.by_compact_file_stem.get(compact_stem_key, []),
                    ),
                    (
                        "compact_drw_number_equals_compact_physical_file_stem",
                        db_index.by_compact_drw_number.get(compact_stem_key, []),
                    ),
                    (
                        "compact_drw_name_equals_compact_physical_file_stem",
                        db_index.by_compact_drw_name.get(compact_stem_key, []),
                    ),
                    (
                        "compact_spare_part_number_equals_compact_physical_file_stem",
                        db_index.by_compact_spare_part_number.get(compact_stem_key, []),
                    ),
                ]
            )

        for reason, candidates in lookup_steps:
            if not candidates:
                continue

            return self._select_match(
                candidates,
                reason,
                prefer_first_on_ambiguous,
            )

        return DrawingDbMatch(
            drawing=None,
            reason="no_db_match",
            candidate_count=0,
            ambiguous_drawings=[],
        )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def drawing_to_dict(drawing: Drawing) -> Dict[str, Any]:
        return {
            "id": drawing.id,
            "drw_number": getattr(drawing, "drw_number", None),
            "drw_name": getattr(drawing, "drw_name", None),
            "drw_revision": getattr(drawing, "drw_revision", None),
            "drw_spare_part_number": getattr(drawing, "drw_spare_part_number", None),
            "drw_type": getattr(drawing, "drw_type", None),
            "file_path": getattr(drawing, "file_path", None),
        }

    def _build_base_result(
        self,
        file_path: Path,
        drawing: Optional[Drawing],
        status: str,
        match_reason: str,
        message: str,
        matched_file_path: Optional[str] = None,
        candidate_count: int = 0,
        ambiguous_drawings: Optional[List[Drawing]] = None,
    ) -> DrawingFileSyncResult:
        ambiguous_drawings = ambiguous_drawings or []

        return DrawingFileSyncResult(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_stem=file_path.stem,
            file_extension=file_path.suffix.lower(),
            drawing_id=int(drawing.id) if drawing is not None else None,
            drw_number=getattr(drawing, "drw_number", None) if drawing is not None else None,
            drw_name=getattr(drawing, "drw_name", None) if drawing is not None else None,
            old_file_path=getattr(drawing, "file_path", None) if drawing is not None else None,
            matched_file_path=matched_file_path,
            status=status,
            match_reason=match_reason,
            message=message,
            candidate_count=candidate_count,
            ambiguous_drawing_ids=[int(item.id) for item in ambiguous_drawings],
            ambiguous_drawings=[
                self.drawing_to_dict(item)
                for item in ambiguous_drawings
            ],
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _create_missing_drawing_from_file(
        self,
        session,
        file_path: Path,
        default_drw_type: str = "Other",
    ) -> Drawing:
        """
        Create a basic Drawing record from an unmatched file.

        This is intentionally conservative:
            drw_number = file stem
            drw_name = file stem
            file_path = full file path
        """
        drawing = Drawing(
            drw_equipment_name=None,
            drw_number=file_path.stem,
            drw_name=file_path.stem,
            drw_revision=None,
            drw_spare_part_number=None,
            drw_type=default_drw_type,
            file_path=str(file_path.resolve()),
        )

        session.add(drawing)
        session.flush()

        return drawing

    def _process_one_file(
        self,
        session,
        file_path: Path,
        db_index: DrawingDbIndex,
        dry_run: bool,
        use_compact_match: bool,
        prefer_first_on_ambiguous: bool,
        create_missing: bool,
        default_drw_type: str,
    ) -> DrawingFileSyncResult:
        try:
            match = self.match_file_to_db(
                file_path=file_path,
                db_index=db_index,
                use_compact_match=use_compact_match,
                prefer_first_on_ambiguous=prefer_first_on_ambiguous,
            )

            resolved_file_str = str(file_path.resolve())

            if match.ambiguous_drawings:
                return self._build_base_result(
                    file_path=file_path,
                    drawing=None,
                    status="ambiguous",
                    match_reason=match.reason,
                    message="Multiple database Drawing rows matched this physical file. No update was made.",
                    matched_file_path=None,
                    candidate_count=match.candidate_count,
                    ambiguous_drawings=match.ambiguous_drawings,
                )

            if match.drawing is None:
                if not create_missing:
                    return self._build_base_result(
                        file_path=file_path,
                        drawing=None,
                        status="no_db_match",
                        match_reason=match.reason,
                        message="Physical drawing file has no matching Drawing row.",
                        matched_file_path=None,
                        candidate_count=0,
                    )

                if dry_run:
                    return self._build_base_result(
                        file_path=file_path,
                        drawing=None,
                        status="would_create",
                        match_reason="create_missing_enabled",
                        message="Would create a new Drawing row for this unmatched physical file.",
                        matched_file_path=resolved_file_str,
                        candidate_count=0,
                    )

                drawing = self._create_missing_drawing_from_file(
                    session=session,
                    file_path=file_path,
                    default_drw_type=default_drw_type,
                )

                self._add_drawing_to_index(db_index, drawing)

                return self._build_base_result(
                    file_path=file_path,
                    drawing=drawing,
                    status="created",
                    match_reason="create_missing_enabled",
                    message="Created new Drawing row for unmatched physical file.",
                    matched_file_path=resolved_file_str,
                    candidate_count=1,
                )

            drawing = match.drawing
            old_file_path = self._safe_text(getattr(drawing, "file_path", None))

            if old_file_path == resolved_file_str:
                return self._build_base_result(
                    file_path=file_path,
                    drawing=drawing,
                    status="current_exists",
                    match_reason=match.reason,
                    message="Database Drawing.file_path already matches the physical file.",
                    matched_file_path=resolved_file_str,
                    candidate_count=match.candidate_count,
                )

            if not dry_run:
                drawing.file_path = resolved_file_str

            return self._build_base_result(
                file_path=file_path,
                drawing=drawing,
                status="would_update" if dry_run else "updated",
                match_reason=match.reason,
                message="Matched physical file to Drawing row and updated file_path."
                if not dry_run
                else "Matched physical file to Drawing row and would update file_path.",
                matched_file_path=resolved_file_str,
                candidate_count=match.candidate_count,
            )

        except Exception as exc:
            return self._build_base_result(
                file_path=file_path,
                drawing=None,
                status="error",
                match_reason="exception",
                message=str(exc),
                matched_file_path=None,
                candidate_count=0,
            )

    # ------------------------------------------------------------------
    # Main folder-first sync
    # ------------------------------------------------------------------

    @with_request_id
    def sync_all_drawings(
        self,
        dry_run: bool = True,
        limit: Optional[int] = None,
        commit_every: int = 100,
        use_compact_match: bool = True,
        prefer_newest_on_ambiguous: bool = False,
        force_database_drawing_root: bool = False,
        create_missing: bool = False,
        prefer_first_on_ambiguous: bool = False,
        default_drw_type: str = "Other",
        request_id: Optional[str] = None,
    ) -> DrawingFileSyncSummary:
        """
        Folder-first sync.

        Args:
            dry_run:
                True means no database writes.

            limit:
                Optional max number of physical files to process.

            commit_every:
                Commit interval when dry_run=False.

            use_compact_match:
                Enables loose matching.

            prefer_newest_on_ambiguous:
                Kept for compatibility with the older DB-first service.
                Not used in folder-first mode.

            force_database_drawing_root:
                Kept for compatibility with the older DB-first service.
                Not needed in folder-first mode.

            create_missing:
                If True, unmatched files can create new Drawing rows.
                Default False.

            prefer_first_on_ambiguous:
                If True, the first ambiguous DB match is used.
                Default False is safer.

            default_drw_type:
                Drawing type used when create_missing=True.

        Returns:
            DrawingFileSyncSummary
        """
        active_request_id = request_id or self.request_id

        if commit_every < 1:
            raise ValueError("commit_every must be at least 1")

        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1 when provided")

        files = self.scan_drawing_folder(limit=limit)

        summary = DrawingFileSyncSummary(
            dry_run=dry_run,
            drawing_root=str(self.drawing_root),
            recursive=self.recursive,
            total_files_scanned=len(files),
        )

        info_id(
            (
                "Starting folder-first drawing file sync. "
                f"dry_run={dry_run}, "
                f"root={self.drawing_root}, "
                f"files={len(files)}, "
                f"create_missing={create_missing}"
            ),
            active_request_id,
        )

        with self.db_config.main_session() as session:
            try:
                db_index = self.build_db_index(session)
                summary.total_db_drawings = len(db_index.drawings)

                write_counter = 0

                for file_path in files:
                    result = self._process_one_file(
                        session=session,
                        file_path=file_path,
                        db_index=db_index,
                        dry_run=dry_run,
                        use_compact_match=use_compact_match,
                        prefer_first_on_ambiguous=prefer_first_on_ambiguous,
                        create_missing=create_missing,
                        default_drw_type=default_drw_type,
                    )

                    summary.add_result(result)

                    if result.status in {"updated", "created"}:
                        write_counter += 1

                        if write_counter % commit_every == 0:
                            session.commit()
                            info_id(
                                f"Committed {write_counter} drawing file sync writes so far.",
                                active_request_id,
                            )

                if dry_run:
                    session.rollback()
                else:
                    session.commit()

                info_id(
                    (
                        "Folder-first drawing file sync complete. "
                        f"files_scanned={summary.total_files_scanned}, "
                        f"db_drawings={summary.total_db_drawings}, "
                        f"updated={summary.updated}, "
                        f"would_update={summary.would_update}, "
                        f"created={summary.created}, "
                        f"would_create={summary.would_create}, "
                        f"no_db_match={summary.no_db_match}, "
                        f"ambiguous={summary.ambiguous}, "
                        f"errors={summary.errors}"
                    ),
                    active_request_id,
                )

                return summary

            except SQLAlchemyError as exc:
                session.rollback()
                error_id(f"Folder-first drawing file sync failed: {exc}", active_request_id)
                raise

    # Clearer alias for new folder-first behavior.
    def sync_folder_to_database(
        self,
        dry_run: bool = True,
        limit: Optional[int] = None,
        commit_every: int = 100,
        use_compact_match: bool = True,
        create_missing: bool = False,
        prefer_first_on_ambiguous: bool = False,
        default_drw_type: str = "Other",
        request_id: Optional[str] = None,
    ) -> DrawingFileSyncSummary:
        return self.sync_all_drawings(
            dry_run=dry_run,
            limit=limit,
            commit_every=commit_every,
            use_compact_match=use_compact_match,
            create_missing=create_missing,
            prefer_first_on_ambiguous=prefer_first_on_ambiguous,
            default_drw_type=default_drw_type,
            request_id=request_id,
        )


def write_json_report(
    summary: DrawingFileSyncSummary,
    report_path: str,
    include_results: bool = True,
) -> None:
    output_path = Path(report_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary.to_dict(include_results=include_results),
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_console_summary(summary: DrawingFileSyncSummary, show_results: bool = False) -> None:
    print()
    print("Folder-First Drawing File Sync Summary")
    print("--------------------------------------")
    print(f"Dry run:              {summary.dry_run}")
    print(f"Drawing root:         {summary.drawing_root}")
    print(f"Recursive scan:       {summary.recursive}")
    print(f"Files scanned:        {summary.total_files_scanned}")
    print(f"DB drawings loaded:   {summary.total_db_drawings}")
    print(f"Current exists:       {summary.current_exists}")
    print(f"Would update:         {summary.would_update}")
    print(f"Updated:              {summary.updated}")
    print(f"No DB match:          {summary.no_db_match}")
    print(f"Ambiguous:            {summary.ambiguous}")
    print(f"Would create:         {summary.would_create}")
    print(f"Created:              {summary.created}")
    print(f"Skipped:              {summary.skipped}")
    print(f"Errors:               {summary.errors}")

    if not show_results:
        return

    print()
    print("Detailed Results")
    print("----------------")

    for result in summary.results:
        print(
            f"[{result.status}] "
            f"file={result.file_name!r} "
            f"drawing_id={result.drawing_id} "
            f"number={result.drw_number!r} "
            f"name={result.drw_name!r}"
        )
        print(f"  file:    {result.file_path}")
        print(f"  old db:  {result.old_file_path}")
        print(f"  matched: {result.matched_file_path}")
        print(f"  reason:  {result.match_reason}")
        print(f"  message: {result.message}")

        if result.ambiguous_drawings:
            print("  ambiguous DB matches:")
            for item in result.ambiguous_drawings:
                print(
                    f"    - id={item.get('id')} "
                    f"number={item.get('drw_number')!r} "
                    f"name={item.get('drw_name')!r} "
                    f"file_path={item.get('file_path')!r}"
                )

        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Folder-first sync: scan DATABASE_DRAWING, match files to Drawing rows, update file_path."
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply database updates. Without this flag, the script runs as dry-run.",
    )

    parser.add_argument(
        "--drawing-root",
        default=None,
        help="Optional drawing folder override. Defaults to DATABASE_DRAWING from config/.env.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of the drawing folder.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of physical files to process.",
    )

    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit after this many writes when --apply is used.",
    )

    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Allowed file extensions. Example: --extensions .dwg .dxf .slddrw .pdf",
    )

    parser.add_argument(
        "--no-compact-match",
        action="store_true",
        help="Disable loose matching that ignores spaces, dashes, underscores, and punctuation.",
    )

    parser.add_argument(
        "--prefer-first-on-ambiguous",
        action="store_true",
        help="Use the first DB match when multiple rows match a file. Default is safer: mark ambiguous.",
    )

    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create Drawing rows for files that do not match any DB record. Only works with --apply.",
    )

    parser.add_argument(
        "--default-drw-type",
        default="Other",
        help="Default drw_type used when --create-missing is enabled.",
    )

    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save a JSON sync report.",
    )

    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Print detailed per-file results to the console.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    if args.commit_every < 1:
        raise ValueError("--commit-every must be at least 1.")

    if args.create_missing and not args.apply:
        print(
            "[WARNING] --create-missing was provided without --apply. "
            "This will only report would_create records."
        )

    service = DrawingFileSyncService(
        drawing_root=args.drawing_root,
        allowed_extensions=args.extensions,
        recursive=not args.no_recursive,
    )

    summary = service.sync_folder_to_database(
        dry_run=not args.apply,
        limit=args.limit,
        commit_every=args.commit_every,
        use_compact_match=not args.no_compact_match,
        create_missing=args.create_missing,
        prefer_first_on_ambiguous=args.prefer_first_on_ambiguous,
        default_drw_type=args.default_drw_type,
    )

    print_console_summary(summary, show_results=args.show_results)

    if args.report_json:
        write_json_report(
            summary=summary,
            report_path=args.report_json,
            include_results=True,
        )
        print()
        print(f"JSON report saved to: {Path(args.report_json).expanduser().resolve()}")

    if summary.errors:
        return 2

    if summary.ambiguous:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())