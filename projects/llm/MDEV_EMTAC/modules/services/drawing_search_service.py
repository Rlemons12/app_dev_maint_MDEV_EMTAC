"""
modules/services/drawing_search_service.py

Service layer for drawing search.

Business/search responsibilities:
    - drawing search
    - path-aware file_path search support
    - spare part search support
    - drawing type search
    - optional part image enrichment
    - serialization helpers
    - drawing file resolution for /drawings/view/<drawing_id>

Rules:
    - This service does NOT open sessions.
    - This service does NOT close sessions.
    - This service does NOT commit or rollback.
    - The orchestrator owns transaction/session lifecycle.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from modules.configuration.config import DATABASE_DIR, DATABASE_DRAWING
from modules.configuration.log_config import (
    debug_id,
    error_id,
    info_id,
    log_timed_operation,
    with_request_id,
)
from modules.emtacdb.emtacdb_fts import Drawing


class DrawingSearchService:
    """
    Pure service layer for Drawing search logic.

    This service is used by:
        modules/orchestrators/drawing_search_orchestrator.py

    Main route consumers:
        /drawings/search
        /drawings/types
        /drawings/search/by-type/<drawing_type>
        /drawings/view/<drawing_id>
    """

    # ---------------------------------------------------------
    # BASIC HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _require_session(session: Session, method_name: str) -> None:
        if session is None:
            raise RuntimeError(f"Session required for DrawingSearchService.{method_name}")

    @staticmethod
    def _clean_text(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        return max(1, min(int(limit or 100), 1000))

    # ---------------------------------------------------------
    # SERIALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def drawing_to_payload(drawing: Drawing) -> Dict[str, Any]:
        """
        Convert a Drawing ORM object into a JSON-safe dictionary.

        Important:
            The url/view_url/download_url values point to the serving route:

                /drawings/view/<drawing_id>
        """
        return {
            "id": drawing.id,
            "drw_equipment_name": drawing.drw_equipment_name,
            "drw_number": drawing.drw_number,
            "drw_name": drawing.drw_name,
            "drw_revision": drawing.drw_revision,
            "drw_spare_part_number": drawing.drw_spare_part_number,
            "drw_type": drawing.drw_type,
            "file_path": drawing.file_path,
            "url": f"/drawings/view/{drawing.id}",
            "view_url": f"/drawings/view/{drawing.id}",
            "download_url": f"/drawings/view/{drawing.id}?download=1",
        }

    # ---------------------------------------------------------
    # PATH HELPERS FOR SERVING DRAWING FILES
    # ---------------------------------------------------------

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        """
        Detect absolute paths, including Windows paths like:

            E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        if value is None or not str(value).strip():
            return False

        text = str(value).strip()

        return (
            Path(text).expanduser().is_absolute()
            or PureWindowsPath(text).is_absolute()
        )

    @staticmethod
    def _normalize_path_text(value: str) -> str:
        """
        Normalize a stored path string without forcing it absolute.
        """
        return os.path.normpath(str(value).strip())

    @staticmethod
    def _is_under_path(child_path: Path, parent_path: Path) -> bool:
        """
        Return True when child_path is inside parent_path.

        This prevents accidental serving outside DATABASE_DRAWING.
        """
        try:
            child_path.resolve().relative_to(parent_path.resolve())
            return True
        except ValueError:
            return False

    def _get_database_root(self) -> Path:
        """
        Resolve DATABASE_DIR.

        Example:
            E:\\emtac\\Database
        """
        if DATABASE_DIR and str(DATABASE_DIR).strip():
            return Path(str(DATABASE_DIR)).expanduser().resolve()

        drawing_root = self._get_drawing_root()
        return drawing_root.parent.resolve()

    def _get_drawing_root(self) -> Path:
        """
        Resolve DATABASE_DRAWING.

        Example:
            E:\\emtac\\Database\\DB_DRAWING
        """
        if not DATABASE_DRAWING or not str(DATABASE_DRAWING).strip():
            raise ValueError("DATABASE_DRAWING is not configured.")

        return Path(str(DATABASE_DRAWING)).expanduser().resolve()

    def _resolve_drawing_physical_path(self, stored_file_path: str) -> Path:
        """
        Resolve Drawing.file_path from the database into a physical file path.

        Supported stored formats:

            Preferred:
                DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

            Legacy drawing-root-relative:
                AFL31600\\E-5347-2-2.dwg

            Legacy absolute:
                E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        The resolved file must be inside DATABASE_DRAWING.
        """
        if stored_file_path is None or not str(stored_file_path).strip():
            raise ValueError("Drawing has no file_path value.")

        stored_text = self._normalize_path_text(stored_file_path)

        database_root = self._get_database_root()
        drawing_root = self._get_drawing_root()

        if self._is_absolute_path(stored_text):
            physical_path = Path(stored_text).expanduser().resolve()
        else:
            parts = [
                part
                for part in stored_text.replace("/", "\\").split("\\")
                if part
            ]

            if parts and parts[0].lower() == drawing_root.name.lower():
                # DB_DRAWING\AFL31600\E-5347-2-2.dwg
                physical_path = (database_root / stored_text).resolve()
            else:
                # AFL31600\E-5347-2-2.dwg or filename only
                physical_path = (drawing_root / stored_text).resolve()

        if not self._is_under_path(physical_path, drawing_root):
            raise ValueError(
                "Resolved drawing file path is outside DATABASE_DRAWING. "
                f"stored_file_path={stored_file_path}, "
                f"resolved_path={physical_path}, "
                f"drawing_root={drawing_root}"
            )

        return physical_path

    @staticmethod
    def _guess_mime_type(file_path: Path) -> str:
        """
        Guess a MIME type for Flask send_file.
        """
        guessed_type, _ = mimetypes.guess_type(str(file_path))

        if guessed_type:
            return guessed_type

        suffix = file_path.suffix.lower()

        if suffix == ".dwg":
            return "application/acad"

        if suffix == ".dxf":
            return "application/dxf"

        if suffix == ".slddrw":
            return "application/octet-stream"

        return "application/octet-stream"

    # ---------------------------------------------------------
    # FILE PATH SEARCH HELPERS
    # ---------------------------------------------------------

    def _build_file_path_search_variants(self, file_path: Optional[str]) -> List[str]:
        """
        Build search variants for Drawing.file_path.

        This lets users search by:
            - DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
            - AFL31600\\E-5347-2-2.dwg
            - E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
            - E-5347-2-2.dwg
        """
        text = self._clean_text(file_path)

        if not text:
            return []

        variants: List[str] = []

        def add(value: Optional[str]) -> None:
            cleaned = self._clean_text(value)
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(text)
        add(text.replace("/", "\\"))
        add(text.replace("\\", "/"))

        try:
            path = Path(text)
            add(path.name)
            add(path.stem)
        except Exception:
            pass

        try:
            if self._is_absolute_path(text):
                physical_path = Path(text).expanduser().resolve()
                drawing_root = self._get_drawing_root()
                database_root = self._get_database_root()

                try:
                    add(str(physical_path.relative_to(database_root)))
                except ValueError:
                    pass

                try:
                    add(str(physical_path.relative_to(drawing_root)))
                except ValueError:
                    pass

                add(physical_path.name)
                add(physical_path.stem)
        except Exception:
            pass

        return variants

    def _apply_file_path_filter(
        self,
        query,
        file_path: Optional[str],
        exact_match: bool,
    ):
        """
        Apply path-aware file_path filtering to a SQLAlchemy query.
        """
        variants = self._build_file_path_search_variants(file_path)

        if not variants:
            return query

        conditions = []

        for variant in variants:
            if exact_match:
                conditions.append(Drawing.file_path == variant)
            else:
                conditions.append(Drawing.file_path.ilike(f"%{variant}%"))

        return query.filter(or_(*conditions))

    # ---------------------------------------------------------
    # SEARCH HELPERS
    # ---------------------------------------------------------

    def _search_spare_part_number(
        self,
        session: Session,
        *,
        drw_spare_part_number: str,
        drw_type: Optional[str],
        file_path: Optional[str],
        exact_match: bool,
        limit: int,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Flexible spare part search.

        This preserves the older route behavior where separators are removed
        for better spare part matching.
        """
        self._require_session(session, "_search_spare_part_number")

        query = session.query(Drawing)

        if drw_type:
            query = query.filter(Drawing.drw_type == drw_type)

        normalized_spare = (
            drw_spare_part_number
            .replace("-", "")
            .replace(" ", "")
            .replace("_", "")
        )

        patterns = [
            f"%{normalized_spare}%",
            f"{normalized_spare}%",
            f"%{normalized_spare}",
        ]

        if len(normalized_spare) > 5:
            patterns.append(f"%{normalized_spare[-5:]}%")

        conditions = []

        for pattern in patterns:
            conditions.append(
                func.lower(Drawing.drw_spare_part_number).like(
                    func.lower(pattern)
                )
            )

            conditions.append(
                func.lower(
                    func.replace(
                        func.replace(
                            func.replace(Drawing.drw_spare_part_number, "-", ""),
                            " ",
                            "",
                        ),
                        "_",
                        "",
                    )
                ).like(func.lower(pattern))
            )

        query = query.filter(or_(*conditions))
        query = self._apply_file_path_filter(
            query=query,
            file_path=file_path,
            exact_match=exact_match,
        )

        results = query.limit(limit).all()

        debug_id(
            (
                "[DrawingSearchService] spare part search returned "
                f"{len(results)} result(s) for spare='{drw_spare_part_number}'"
            ),
            request_id,
        )

        return results

    def _search_file_path_only(
        self,
        session: Session,
        *,
        file_path: str,
        exact_match: bool,
        drw_type: Optional[str],
        limit: int,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Path-aware search for file_path-only queries.
        """
        self._require_session(session, "_search_file_path_only")

        query = session.query(Drawing)

        if drw_type:
            query = query.filter(Drawing.drw_type == drw_type)

        query = self._apply_file_path_filter(
            query=query,
            file_path=file_path,
            exact_match=exact_match,
        )

        results = query.limit(limit).all()

        debug_id(
            (
                "[DrawingSearchService] file_path search returned "
                f"{len(results)} result(s) for file_path='{file_path}'"
            ),
            request_id,
        )

        return results

    def _add_part_images_to_payloads(
        self,
        session: Session,
        drawings_data: List[Dict[str, Any]],
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Add associated part images to drawing payloads.

        This preserves the behavior from the old Flask route.
        """
        self._require_session(session, "_add_part_images_to_payloads")

        try:
            from modules.emtacdb.emtacdb_fts import (
                DrawingPartAssociation,
                Image,
                PartsPositionImageAssociation,
            )
        except Exception as exc:
            error_id(
                f"[DrawingSearchService] Could not import part/image associations: {exc}",
                request_id,
                exc_info=True,
            )
            return drawings_data

        for drawing_data in drawings_data:
            drawing_id = drawing_data.get("id")

            if not drawing_id:
                drawing_data["part_images"] = []
                continue

            debug_id(
                f"[DrawingSearchService] Fetching part images for drawing id={drawing_id}",
                request_id,
            )

            part_images: List[Dict[str, Any]] = []

            try:
                with log_timed_operation(
                    f"DrawingSearchService.get_parts_by_drawing.{drawing_id}",
                    request_id,
                ):
                    parts = DrawingPartAssociation.get_parts_by_drawing(
                        drawing_id=drawing_id,
                        request_id=request_id,
                        session=session,
                    )

                for part in parts:
                    with log_timed_operation(
                        f"DrawingSearchService.part_image_search.{getattr(part, 'id', 'unknown')}",
                        request_id,
                    ):
                        associations = PartsPositionImageAssociation.search(
                            session=session,
                            part_id=part.id,
                        )

                    for assoc in associations:
                        image_id = getattr(assoc, "image_id", None)

                        if not image_id:
                            continue

                        with log_timed_operation(
                            f"DrawingSearchService.serve_image.{image_id}",
                            request_id,
                        ):
                            image_data = Image.serve_image(
                                image_id=image_id,
                                request_id=request_id,
                                session=session,
                            )

                        if image_data:
                            part_images.append(
                                {
                                    "part_id": part.id,
                                    "part_number": getattr(part, "part_number", None),
                                    "part_name": getattr(part, "name", None),
                                    "image_id": image_data.get("id"),
                                    "image_title": image_data.get("title"),
                                    "image_path": image_data.get("file_path"),
                                    "image_url": f"/images/{image_data.get('id')}",
                                }
                            )

            except Exception as exc:
                error_id(
                    (
                        "[DrawingSearchService] Failed to add part images for "
                        f"drawing_id={drawing_id}: {exc}"
                    ),
                    request_id,
                    exc_info=True,
                )

            drawing_data["part_images"] = part_images

        return drawings_data

    # ---------------------------------------------------------
    # PUBLIC SEARCH METHODS
    # ---------------------------------------------------------

    @with_request_id
    def search_drawings(
        self,
        session: Session,
        *,
        search_text: Optional[str] = None,
        fields: Optional[List[str]] = None,
        exact_match: bool = False,
        drawing_id: Optional[int] = None,
        drw_equipment_name: Optional[str] = None,
        drw_number: Optional[str] = None,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        drw_type: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 100,
        include_part_images: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search drawings and return a JSON-safe payload.
        """
        self._require_session(session, "search_drawings")

        limit = self._normalize_limit(limit)

        # Special case:
        # If the user picked spare part number as the search_text field, treat
        # search_text as the spare part search value.
        if search_text and fields and "drw_spare_part_number" in fields:
            if not drw_spare_part_number and search_text.strip():
                drw_spare_part_number = search_text.strip()
                search_text = None
                exact_match = False

        if drw_spare_part_number:
            results = self._search_spare_part_number(
                session=session,
                drw_spare_part_number=drw_spare_part_number,
                drw_type=drw_type,
                file_path=file_path,
                exact_match=exact_match,
                limit=limit,
                request_id=request_id,
            )

        elif file_path and not any(
            [
                search_text,
                drawing_id,
                drw_equipment_name,
                drw_number,
                drw_name,
                drw_revision,
                drw_type,
            ]
        ):
            results = self._search_file_path_only(
                session=session,
                file_path=file_path,
                exact_match=exact_match,
                drw_type=drw_type,
                limit=limit,
                request_id=request_id,
            )

        else:
            results = Drawing.search(
                search_text=search_text,
                fields=fields,
                exact_match=exact_match,
                drawing_id=drawing_id,
                drw_equipment_name=drw_equipment_name,
                drw_number=drw_number,
                drw_name=drw_name,
                drw_revision=drw_revision,
                drw_spare_part_number=drw_spare_part_number,
                drw_type=drw_type,
                file_path=file_path,
                limit=limit,
                request_id=request_id,
                session=session,
            )

        drawings_data = [
            self.drawing_to_payload(drawing)
            for drawing in results
        ]

        if include_part_images:
            drawings_data = self._add_part_images_to_payloads(
                session=session,
                drawings_data=drawings_data,
                request_id=request_id,
            )

        debug_id(
            f"[DrawingSearchService] search_drawings returned {len(drawings_data)} result(s)",
            request_id,
        )

        return {
            "count": len(drawings_data),
            "results": drawings_data,
        }

    @with_request_id
    def get_drawing_types(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return available drawing types.
        """
        available_types = Drawing.get_available_types()

        info_id(
            f"[DrawingSearchService] Retrieved {len(available_types)} drawing type(s)",
            request_id,
        )

        return {
            "available_types": available_types,
            "count": len(available_types),
        }

    @with_request_id
    def search_by_type(
        self,
        session: Session,
        *,
        drawing_type: str,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search drawings by drawing type.
        """
        self._require_session(session, "search_by_type")

        limit = self._normalize_limit(limit)

        results = Drawing.search_by_type(
            drawing_type=drawing_type,
            request_id=request_id,
            session=session,
        )

        results = results[:limit]

        drawings_data = [
            self.drawing_to_payload(drawing)
            for drawing in results
        ]

        return {
            "drawing_type": drawing_type,
            "count": len(drawings_data),
            "results": drawings_data,
        }

    # ---------------------------------------------------------
    # PUBLIC FILE SERVING METHOD
    # ---------------------------------------------------------

    @with_request_id
    def get_drawing_file_payload(
        self,
        session: Session,
        *,
        drawing_id: int,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Get the physical file payload for /drawings/view/<drawing_id>.

        The route layer calls Flask send_file using this payload.
        """
        self._require_session(session, "get_drawing_file_payload")

        if drawing_id is None:
            return {
                "error": "Invalid drawing_id",
                "message": "drawing_id is required",
            }, 400

        drawing = session.get(Drawing, drawing_id)

        if drawing is None:
            return {
                "error": "Drawing not found",
                "message": f"No drawing found for id={drawing_id}",
            }, 404

        stored_file_path = getattr(drawing, "file_path", None)

        if not stored_file_path or not str(stored_file_path).strip():
            return {
                "error": "Drawing file path missing",
                "message": f"Drawing id={drawing_id} does not have a file_path value.",
                "drawing": self.drawing_to_payload(drawing),
            }, 404

        try:
            physical_path = self._resolve_drawing_physical_path(stored_file_path)
        except Exception as exc:
            return {
                "error": "Drawing file path could not be resolved",
                "message": str(exc),
                "drawing": self.drawing_to_payload(drawing),
                "stored_file_path": stored_file_path,
            }, 500

        if not physical_path.exists() or not physical_path.is_file():
            return {
                "error": "Drawing file not found",
                "message": f"Drawing file does not exist on disk: {physical_path}",
                "stored_file_path": stored_file_path,
                "resolved_file_path": str(physical_path),
                "drawing": self.drawing_to_payload(drawing),
            }, 404

        mime_type = self._guess_mime_type(physical_path)

        return {
            "drawing": self.drawing_to_payload(drawing),
            "stored_file_path": stored_file_path,
            "physical_file_path": str(physical_path),
            "download_name": physical_path.name,
            "mime_type": mime_type,
        }, 200