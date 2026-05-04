# modules/services/chat_service.py

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from modules.configuration.log_config import debug_id


class ChatService:
    """
    Pure response formatting service.

    Responsibilities:
        - Convert raw AI result into stable UI contract
        - Guarantee block structure
        - Preserve legacy fields: method
        - Preserve new architecture fields: strategy, model_name
        - Normalize image URLs to /serve_image/<id>
        - Prevent raw DB_IMAGES paths from reaching frontend link/src fields
        - Promote nested images/drawings/parts when needed
        - Cap large UI payloads so forced-chunk tests do not flood the browser
        - Stateless

    Important:
        This service does not own DB sessions.
        This service does not query the database.
        This service only formats already-resolved domain results.
    """

    # ---------------------------------------------------------
    # UI safety limits
    # ---------------------------------------------------------
    MAX_UI_IMAGES = 25
    MAX_UI_PARTS = 100
    MAX_UI_DRAWINGS = 100

    MAX_DOCUMENT_IMAGES = 25
    MAX_DOCUMENT_PART_IMAGES = 25
    MAX_DOCUMENT_PARTS = 100
    MAX_DOCUMENT_DRAWINGS = 100
    MAX_DOCUMENT_NAV_DRAWINGS = 100

    # ---------------------------------------------------------
    # Response Formatting
    # ---------------------------------------------------------

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts raw AI domain result into UI contract.

        Guarantees:
            - Stable block structure
            - method field for legacy compatibility
            - strategy field for current architecture
            - model_name passthrough
            - image links use /serve_image/<id>
            - raw file paths are preserved only as raw_file_path
            - UI-facing file_path/src/url/href all point to web-safe route
            - large images/parts/drawings lists are capped
        """

        if not isinstance(result, dict):
            return self._error_response("Invalid AI response format.")

        documents = self._dicts_only(self._as_list(result.get("documents")))
        parts = self._dicts_only(self._as_list(result.get("parts")))
        images = self._dicts_only(self._as_list(result.get("images")))
        drawings = self._dicts_only(self._as_list(result.get("drawings")))

        # ---------------------------------------------------------
        # Normalize image URLs everywhere before response leaves backend
        # ---------------------------------------------------------
        images = self._normalize_images(images)

        for doc in documents:
            self._normalize_document_images_in_place(doc)

        for part in parts:
            self._normalize_part_images_in_place(part)

        # ---------------------------------------------------------
        # Promote nested images into top-level images container
        # ---------------------------------------------------------
        promoted_images: List[Dict[str, Any]] = []

        for doc in documents:
            self._extend_unique(
                promoted_images,
                doc.get("images"),
                identity_fields=("id", "src", "url", "href", "title"),
            )
            self._extend_unique(
                promoted_images,
                doc.get("part_images"),
                identity_fields=("id", "src", "url", "href", "title"),
            )

            parts_panel = doc.get("parts_panel")
            if isinstance(parts_panel, dict):
                self._extend_unique(
                    promoted_images,
                    parts_panel.get("images"),
                    identity_fields=("id", "src", "url", "href", "title"),
                )

        self._extend_unique(
            images,
            promoted_images,
            identity_fields=("id", "src", "url", "href", "title"),
        )

        images = self._normalize_images(images)

        # ---------------------------------------------------------
        # Promote nested parts into top-level parts container
        # ---------------------------------------------------------
        promoted_parts: List[Dict[str, Any]] = []

        for doc in documents:
            self._extend_unique(
                promoted_parts,
                doc.get("parts"),
                identity_fields=("id", "part_number", "name", "title"),
            )

            parts_panel = doc.get("parts_panel")
            if isinstance(parts_panel, dict):
                self._extend_unique(
                    promoted_parts,
                    parts_panel.get("parts"),
                    identity_fields=("id", "part_number", "name", "title"),
                )

        self._extend_unique(
            parts,
            promoted_parts,
            identity_fields=("id", "part_number", "name", "title"),
        )

        # ---------------------------------------------------------
        # Promote nested drawings into top-level drawings container
        # ---------------------------------------------------------
        promoted_drawings: List[Dict[str, Any]] = []

        for doc in documents:
            for key in (
                "drawings",
                "drawing",
                "related_drawings",
                "position_drawings",
                "task_drawings",
            ):
                self._extend_unique(
                    promoted_drawings,
                    doc.get(key),
                    identity_fields=(
                        "id",
                        "drw_number",
                        "drawing_number",
                        "file_path",
                        "title",
                    ),
                )

            relationship_map = doc.get("relationship_map") or doc.get("relationships") or {}
            if isinstance(relationship_map, dict):
                for tier_key in ("forward", "reverse", "1st_tier", "2nd_tier", "summary"):
                    tier = relationship_map.get(tier_key)
                    if isinstance(tier, dict):
                        self._extend_unique(
                            promoted_drawings,
                            tier.get("drawings"),
                            identity_fields=(
                                "id",
                                "drw_number",
                                "drawing_number",
                                "file_path",
                                "title",
                            ),
                        )

                forward = relationship_map.get("forward")
                if isinstance(forward, dict):
                    for nested_key in ("1st_tier", "2nd_tier", "3rd_tier"):
                        nested = forward.get(nested_key)
                        if isinstance(nested, dict):
                            self._extend_unique(
                                promoted_drawings,
                                nested.get("drawings"),
                                identity_fields=(
                                    "id",
                                    "drw_number",
                                    "drawing_number",
                                    "file_path",
                                    "title",
                                ),
                            )

        used_chunks = result.get("used_chunks") or []
        if isinstance(used_chunks, list):
            for chunk in used_chunks:
                if not isinstance(chunk, dict):
                    continue

                for key in (
                    "drawings",
                    "drawing",
                    "related_drawings",
                    "position_drawings",
                    "task_drawings",
                ):
                    self._extend_unique(
                        promoted_drawings,
                        chunk.get(key),
                        identity_fields=(
                            "id",
                            "drw_number",
                            "drawing_number",
                            "file_path",
                            "title",
                        ),
                    )

                relationship_map = chunk.get("relationship_map") or chunk.get("relationships") or {}
                if isinstance(relationship_map, dict):
                    forward = relationship_map.get("forward")
                    if isinstance(forward, dict):
                        for nested_key in ("1st_tier", "2nd_tier", "3rd_tier"):
                            nested = forward.get(nested_key)
                            if isinstance(nested, dict):
                                self._extend_unique(
                                    promoted_drawings,
                                    nested.get("drawings"),
                                    identity_fields=(
                                        "id",
                                        "drw_number",
                                        "drawing_number",
                                        "file_path",
                                        "title",
                                    ),
                                )

        for part in parts:
            self._extend_unique(
                promoted_drawings,
                part.get("drawings"),
                identity_fields=("id", "drw_number", "drawing_number", "file_path", "title"),
            )
            self._extend_unique(
                promoted_drawings,
                part.get("related_drawings"),
                identity_fields=("id", "drw_number", "drawing_number", "file_path", "title"),
            )

        self._extend_unique(
            drawings,
            promoted_drawings,
            identity_fields=("id", "drw_number", "drawing_number", "file_path", "title"),
        )

        # ---------------------------------------------------------
        # Count before UI caps
        # ---------------------------------------------------------
        total_documents = len(documents)
        total_images = len(images)
        total_parts = len(parts)
        total_drawings = len(drawings)

        # ---------------------------------------------------------
        # Cap top-level UI containers
        # ---------------------------------------------------------
        images, images_truncated = self._cap_list(images, self.MAX_UI_IMAGES)
        parts, parts_truncated = self._cap_list(parts, self.MAX_UI_PARTS)
        drawings, drawings_truncated = self._cap_list(drawings, self.MAX_UI_DRAWINGS)

        # ---------------------------------------------------------
        # Cap nested document payloads too.
        #
        # This is important because some frontend code renders from:
        #     blocks["documents-container"][i]["images"]
        # instead of:
        #     blocks["images-container"]
        # ---------------------------------------------------------
        nested_cap_summary = {
            "document_images_truncated": 0,
            "document_part_images_truncated": 0,
            "document_parts_truncated": 0,
            "document_drawings_truncated": 0,
            "drawing_navigation_truncated": 0,
        }

        for doc in documents:
            self._cap_document_payload_in_place(
                doc,
                nested_cap_summary=nested_cap_summary,
            )

        strategy = result.get("strategy", "rag")

        response = {
            "status": result.get("status", "success"),
            "answer": result.get("answer", ""),
            "method": strategy,
            "strategy": strategy,
            "model_name": result.get("model_name"),
            "blocks": {
                "documents-container": documents,
                "parts-container": parts,
                "images-container": images,
                "drawings-container": drawings,
            },
            "counts": {
                "documents": {
                    "total": total_documents,
                    "shown": len(documents),
                    "truncated": False,
                },
                "images": {
                    "total": total_images,
                    "shown": len(images),
                    "truncated": images_truncated,
                },
                "parts": {
                    "total": total_parts,
                    "shown": len(parts),
                    "truncated": parts_truncated,
                },
                "drawings": {
                    "total": total_drawings,
                    "shown": len(drawings),
                    "truncated": drawings_truncated,
                },
                "nested": nested_cap_summary,
            },
            "ui_limits": {
                "max_images": self.MAX_UI_IMAGES,
                "max_parts": self.MAX_UI_PARTS,
                "max_drawings": self.MAX_UI_DRAWINGS,
                "max_document_images": self.MAX_DOCUMENT_IMAGES,
                "max_document_part_images": self.MAX_DOCUMENT_PART_IMAGES,
                "max_document_parts": self.MAX_DOCUMENT_PARTS,
                "max_document_drawings": self.MAX_DOCUMENT_DRAWINGS,
                "max_document_nav_drawings": self.MAX_DOCUMENT_NAV_DRAWINGS,
            },
            "used_chunks": result.get("used_chunks", []),
            "retriever_top_k": result.get("retriever_top_k"),
        }

        debug_id(
            f"ChatService formatted response "
            f"(docs={len(documents)}, "
            f"parts={len(parts)}/{total_parts}, "
            f"images={len(images)}/{total_images}, "
            f"drawings={len(drawings)}/{total_drawings}, "
            f"promoted_images={len(promoted_images)}, "
            f"promoted_parts={len(promoted_parts)}, "
            f"promoted_drawings={len(promoted_drawings)}, "
            f"image_truncated={images_truncated}, "
            f"part_truncated={parts_truncated}, "
            f"drawing_truncated={drawings_truncated})",
            None,
        )

        debug_id(
            f"ChatService result keys={list(result.keys())}",
            None,
        )

        return response

    # ---------------------------------------------------------
    # Image Normalization
    # ---------------------------------------------------------

    def _normalize_document_images_in_place(self, doc: Dict[str, Any]) -> None:
        for key in ("images", "part_images"):
            value = doc.get(key)
            if isinstance(value, list):
                doc[key] = self._normalize_images(value)

        parts_panel = doc.get("parts_panel")
        if isinstance(parts_panel, dict):
            panel_images = parts_panel.get("images")
            if isinstance(panel_images, list):
                parts_panel["images"] = self._normalize_images(panel_images)

    def _normalize_part_images_in_place(self, part: Dict[str, Any]) -> None:
        for key in ("images", "part_images"):
            value = part.get(key)
            if isinstance(value, list):
                part[key] = self._normalize_images(value)

    def _normalize_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for image in images or []:
            if not isinstance(image, dict):
                continue

            item = dict(image)
            image_id = item.get("id")

            image_url = self._build_image_url(image_id=image_id, item=item)

            if image_url:
                item["src"] = image_url
                item["url"] = image_url
                item["href"] = image_url

                raw_file_path = item.get("file_path")
                if raw_file_path and raw_file_path != image_url:
                    item.setdefault("raw_file_path", raw_file_path)

                # Compatibility guard:
                # Some old frontend code incorrectly uses image.file_path
                # as the href/src. Make it web-safe.
                item["file_path"] = image_url

            normalized.append(item)

        return normalized

    @staticmethod
    def _build_image_url(*, image_id: Any, item: Dict[str, Any]) -> str:
        if image_id is not None:
            try:
                return f"/serve_image/{int(image_id)}"
            except Exception:
                pass

        for key in ("src", "url", "href", "file_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()

                if value.startswith("/serve_image/"):
                    return value

                if value.startswith("/images/"):
                    maybe_id = value.rsplit("/", 1)[-1]
                    if maybe_id.isdigit():
                        return f"/serve_image/{int(maybe_id)}"

        return ""

    # ---------------------------------------------------------
    # Document Payload Capping
    # ---------------------------------------------------------

    def _cap_document_payload_in_place(
        self,
        doc: Dict[str, Any],
        *,
        nested_cap_summary: Dict[str, int],
    ) -> None:
        if not isinstance(doc, dict):
            return

        self._cap_nested_list_field(
            doc,
            field_name="images",
            limit=self.MAX_DOCUMENT_IMAGES,
            summary=nested_cap_summary,
            summary_key="document_images_truncated",
        )

        self._cap_nested_list_field(
            doc,
            field_name="part_images",
            limit=self.MAX_DOCUMENT_PART_IMAGES,
            summary=nested_cap_summary,
            summary_key="document_part_images_truncated",
        )

        self._cap_nested_list_field(
            doc,
            field_name="parts",
            limit=self.MAX_DOCUMENT_PARTS,
            summary=nested_cap_summary,
            summary_key="document_parts_truncated",
        )

        self._cap_nested_list_field(
            doc,
            field_name="drawings",
            limit=self.MAX_DOCUMENT_DRAWINGS,
            summary=nested_cap_summary,
            summary_key="document_drawings_truncated",
        )

        parts_panel = doc.get("parts_panel")
        if isinstance(parts_panel, dict):
            self._cap_nested_list_field(
                parts_panel,
                field_name="images",
                limit=self.MAX_DOCUMENT_PART_IMAGES,
                summary=nested_cap_summary,
                summary_key="document_part_images_truncated",
            )
            self._cap_nested_list_field(
                parts_panel,
                field_name="parts",
                limit=self.MAX_DOCUMENT_PARTS,
                summary=nested_cap_summary,
                summary_key="document_parts_truncated",
            )

        self._cap_drawing_navigation_in_place(
            doc,
            nested_cap_summary=nested_cap_summary,
        )

    def _cap_drawing_navigation_in_place(
        self,
        doc: Dict[str, Any],
        *,
        nested_cap_summary: Dict[str, int],
    ) -> None:
        nav = doc.get("drawing_navigation")
        if not isinstance(nav, dict):
            return

        total_before = 0
        total_after = 0
        truncated_any = False

        for area in nav.get("areas", []) or []:
            if not isinstance(area, dict):
                continue

            for model in area.get("models", []) or []:
                if not isinstance(model, dict):
                    continue

                for asset in model.get("assets", []) or []:
                    if not isinstance(asset, dict):
                        continue

                    drawings = asset.get("drawings")
                    if not isinstance(drawings, list):
                        continue

                    before = len(drawings)
                    capped, truncated = self._cap_list(drawings, self.MAX_DOCUMENT_NAV_DRAWINGS)
                    asset["drawings"] = capped

                    total_before += before
                    total_after += len(capped)

                    if truncated:
                        truncated_any = True
                        asset.setdefault("ui_truncation", {})
                        asset["ui_truncation"]["drawings"] = {
                            "total": before,
                            "shown": len(capped),
                            "truncated": True,
                            "limit": self.MAX_DOCUMENT_NAV_DRAWINGS,
                        }

        if truncated_any:
            nested_cap_summary["drawing_navigation_truncated"] += 1

            nav.setdefault("meta", {})
            nav["meta"]["ui_truncated"] = True
            nav["meta"]["drawing_count_before_ui_cap"] = total_before
            nav["meta"]["drawing_count_after_ui_cap"] = total_after
            nav["meta"]["drawing_limit_per_asset"] = self.MAX_DOCUMENT_NAV_DRAWINGS

    def _cap_nested_list_field(
        self,
        container: Dict[str, Any],
        *,
        field_name: str,
        limit: int,
        summary: Dict[str, int],
        summary_key: str,
    ) -> None:
        value = container.get(field_name)
        if not isinstance(value, list):
            return

        before = len(value)
        capped, truncated = self._cap_list(value, limit)

        container[field_name] = capped

        if truncated:
            summary[summary_key] += 1
            container.setdefault("ui_truncation", {})
            container["ui_truncation"][field_name] = {
                "total": before,
                "shown": len(capped),
                "truncated": True,
                "limit": limit,
            }

    # ---------------------------------------------------------
    # Generic Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    @staticmethod
    def _dicts_only(items: List[Any]) -> List[Dict[str, Any]]:
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _cap_list(items: List[Any], limit: int) -> Tuple[List[Any], bool]:
        if not isinstance(items, list):
            return [], False

        limit = max(0, int(limit))

        if len(items) <= limit:
            return items, False

        return items[:limit], True

    def _extend_unique(
        self,
        target: List[Any],
        items: Any,
        *,
        identity_fields: Tuple[str, ...],
    ) -> None:
        seen = set()

        for existing in target:
            seen.add(self._identity_key(existing, identity_fields=identity_fields))

        for item in self._as_list(items):
            if item is None:
                continue

            key = self._identity_key(item, identity_fields=identity_fields)

            if key not in seen:
                target.append(item)
                seen.add(key)

    @staticmethod
    def _identity_key(item: Any, *, identity_fields: Tuple[str, ...]) -> Any:
        if isinstance(item, dict):
            for field in identity_fields:
                value = item.get(field)
                if value:
                    return field, value
            return "dict", str(item)

        return "value", str(item)

    # ---------------------------------------------------------
    # Error Helper
    # ---------------------------------------------------------

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "answer": message,
            "method": "error",
            "strategy": "error",
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
            "counts": {
                "documents": {"total": 0, "shown": 0, "truncated": False},
                "images": {"total": 0, "shown": 0, "truncated": False},
                "parts": {"total": 0, "shown": 0, "truncated": False},
                "drawings": {"total": 0, "shown": 0, "truncated": False},
                "nested": {},
            },
            "used_chunks": [],
            "retriever_top_k": None,
        }