# modules/services/drawing_navigation_projection.py
#
# Purpose:
# Build a UI-friendly navigation tree for the Drawings panel when a CompleteDocument
# can be associated with multiple Positions.
#
# Output shape (contract):
# {
#   "complete_document_id": int | None,
#   "areas": [
#     {
#       "area_id": int | None,
#       "area_name": str,
#       "models": [
#         {
#           "model_id": int | None,
#           "model_name": str,
#           "assets": [
#             {
#               "asset_number_id": int | None,
#               "asset_name": str,
#               "drawings": [ <drawing_payload>, ... ]
#             }
#           ]
#         }
#       ]
#     }
#   ],
#   "meta": {
#       "area_count": int,
#       "model_count": int,
#       "asset_count": int,
#       "drawing_count": int
#   }
# }
#
# Notes:
# - Names are resolved immediately so the frontend can render hover menus right away.
# - Empty nodes are pruned.
# - Filesystem paths are not exposed. Drawings emit a route URL instead.
# - Spare parts are resolved in one batch query instead of one query per drawing.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Type

from modules.configuration.log_config import (
    debug_id,
    error_id,
    with_request_id,
    get_request_id,
    log_timed_operation,
)

from modules.emtacdb.emtacdb_fts import (
    Position,
    Drawing,
    DrawingPositionAssociation,
    Area,
    Model,
    AssetNumber,
)


@dataclass(frozen=True)
class _PositionKey:
    area_id: Optional[int]
    model_id: Optional[int]
    asset_number_id: Optional[int]


class DrawingNavigationProjection:
    """
    Builds the Area → Model → Asset navigation structure for drawings associated
    with a CompleteDocument via its Positions.

    Performance notes:
    - Uses the session passed into the projection.
    - Does not create a new session per drawing.
    - Resolves spare parts for all drawings in one batch query.
    - Caches repeated navigation builds within the same projection instance.
    """

    def __init__(self, session):
        self.session = session
        self._navigation_cache: Dict[Tuple[Optional[int], Tuple[int, ...]], Dict[str, Any]] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    @with_request_id
    def build_navigation(
        self,
        complete_document_id: Optional[int],
        position_ids: List[int],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        rid = request_id or get_request_id()

        position_ids = self._normalize_ids(position_ids)

        if not position_ids:
            debug_id(
                "[DrawingNavigationProjection] No position_ids provided",
                rid,
            )
            return self._empty_navigation(complete_document_id=complete_document_id)

        cache_key = (
            int(complete_document_id) if complete_document_id is not None else None,
            tuple(position_ids),
        )

        cached = self._navigation_cache.get(cache_key)
        if cached is not None:
            debug_id(
                "[DrawingNavigationProjection] Returning cached navigation "
                f"complete_document_id={complete_document_id} "
                f"positions={len(position_ids)}",
                rid,
            )
            return deepcopy(cached)

        try:
            with log_timed_operation("DrawingNavigationProjection.build_navigation", rid):

                # --------------------------------------------------
                # 1) Load lightweight position rows
                # --------------------------------------------------
                positions = self._load_positions(position_ids=position_ids, rid=rid)

                if not positions:
                    debug_id(
                        "[DrawingNavigationProjection] No Position rows loaded",
                        rid,
                    )
                    return self._empty_navigation(complete_document_id=complete_document_id)

                pos_key_by_id: Dict[int, _PositionKey] = {}
                area_ids: Set[int] = set()
                model_ids: Set[int] = set()
                asset_ids: Set[int] = set()

                for p in positions:
                    position_id = getattr(p, "id", None)

                    if position_id is None:
                        continue

                    key = _PositionKey(
                        area_id=getattr(p, "area_id", None),
                        model_id=getattr(p, "model_id", None),
                        asset_number_id=getattr(p, "asset_number_id", None),
                    )

                    pos_key_by_id[int(position_id)] = key

                    if key.area_id is not None:
                        area_ids.add(int(key.area_id))

                    if key.model_id is not None:
                        model_ids.add(int(key.model_id))

                    if key.asset_number_id is not None:
                        asset_ids.add(int(key.asset_number_id))

                # --------------------------------------------------
                # 2) Resolve display names
                # --------------------------------------------------
                area_name_by_id = self._resolve_names(
                    Area,
                    area_ids,
                    fallback_prefix="Area",
                    rid=rid,
                )

                model_name_by_id = self._resolve_names(
                    Model,
                    model_ids,
                    fallback_prefix="Model",
                    rid=rid,
                )

                asset_name_by_id = self._resolve_names(
                    AssetNumber,
                    asset_ids,
                    fallback_prefix="Asset",
                    rid=rid,
                )

                # --------------------------------------------------
                # 3) Fetch drawings for positions
                # --------------------------------------------------
                drawings, drawing_ids, drawing_to_positions = self._fetch_drawings_for_positions(
                    position_ids=position_ids,
                    rid=rid,
                )

                debug_id(
                    f"[DrawingNavigationProjection] drawings_found={len(drawings)}",
                    rid,
                )

                if not drawings:
                    return self._empty_navigation(complete_document_id=complete_document_id)

                # --------------------------------------------------
                # 4) Optional complete_document filter
                # --------------------------------------------------
                if complete_document_id is not None:
                    doc_filtered = [
                        d for d in drawings
                        if getattr(d, "complete_document_id", None) == complete_document_id
                    ]

                    if doc_filtered:
                        drawings = doc_filtered
                        drawing_ids = {int(getattr(d, "id")) for d in drawings if getattr(d, "id", None) is not None}
                        drawing_to_positions = {
                            did: pids
                            for did, pids in drawing_to_positions.items()
                            if did in drawing_ids
                        }

                        debug_id(
                            "[DrawingNavigationProjection] "
                            f"document_filter_applied drawings={len(drawings)}",
                            rid,
                        )
                    else:
                        debug_id(
                            "[DrawingNavigationProjection] "
                            "document filter produced 0 results; using position drawings",
                            rid,
                        )

                # --------------------------------------------------
                # 5) Serialize drawings
                # --------------------------------------------------
                drawing_payload_by_id: Dict[int, Dict[str, Any]] = {}

                for d in drawings:
                    drawing_id = getattr(d, "id", None)

                    if drawing_id is None:
                        continue

                    drawing_payload_by_id[int(drawing_id)] = self._serialize_drawing(d)

                if not drawing_payload_by_id:
                    return self._empty_navigation(complete_document_id=complete_document_id)

                # --------------------------------------------------
                # 5.5) Resolve spare parts in one batch query
                # --------------------------------------------------
                spare_parts_by_drawing = self._fetch_spare_parts_for_drawings(
                    drawing_ids=list(drawing_payload_by_id.keys()),
                    rid=rid,
                )

                for drw_id, payload in drawing_payload_by_id.items():
                    payload["spare_parts"] = spare_parts_by_drawing.get(drw_id, [])

                # --------------------------------------------------
                # 6) Bucket by Area → Model → Asset
                # --------------------------------------------------
                area_drawings: DefaultDict[Optional[int], Set[int]] = defaultdict(set)
                model_drawings: DefaultDict[Tuple[Optional[int], Optional[int]], Set[int]] = defaultdict(set)
                asset_drawings: DefaultDict[Tuple[Optional[int], Optional[int], Optional[int]], Set[int]] = defaultdict(set)

                models_in_area: DefaultDict[Optional[int], Set[Optional[int]]] = defaultdict(set)
                assets_in_model: DefaultDict[Tuple[Optional[int], Optional[int]], Set[Optional[int]]] = defaultdict(set)

                for drw_id, pos_ids_for_drw in drawing_to_positions.items():
                    if drw_id not in drawing_payload_by_id:
                        continue

                    for pid in pos_ids_for_drw:
                        key = pos_key_by_id.get(pid)

                        if not key:
                            continue

                        area_id = key.area_id
                        model_id = key.model_id
                        asset_number_id = key.asset_number_id

                        area_drawings[area_id].add(drw_id)
                        model_drawings[(area_id, model_id)].add(drw_id)
                        asset_drawings[(area_id, model_id, asset_number_id)].add(drw_id)

                        models_in_area[area_id].add(model_id)
                        assets_in_model[(area_id, model_id)].add(asset_number_id)

                # --------------------------------------------------
                # 7) Build navigation payload
                # --------------------------------------------------
                areas_payload: List[Dict[str, Any]] = []

                for area_id in sorted(
                    area_drawings.keys(),
                    key=lambda x: area_name_by_id.get(x, self._fallback_name("Area", x)),
                ):
                    models_payload: List[Dict[str, Any]] = []

                    for model_id in sorted(
                        models_in_area[area_id],
                        key=lambda x: model_name_by_id.get(x, self._fallback_name("Model", x)),
                    ):
                        assets_payload: List[Dict[str, Any]] = []

                        for asset_number_id in sorted(
                            assets_in_model[(area_id, model_id)],
                            key=lambda x: asset_name_by_id.get(x, self._fallback_name("Asset", x)),
                        ):
                            drw_ids = asset_drawings[(area_id, model_id, asset_number_id)]

                            if not drw_ids:
                                continue

                            assets_payload.append(
                                {
                                    "asset_number_id": asset_number_id,
                                    "asset_name": asset_name_by_id.get(asset_number_id)
                                    or self._fallback_name("Asset", asset_number_id),
                                    "drawings": [
                                        drawing_payload_by_id[i]
                                        for i in self._sort_drawing_ids(
                                            drw_ids,
                                            drawing_payload_by_id,
                                        )
                                    ],
                                }
                            )

                        if not assets_payload:
                            continue

                        models_payload.append(
                            {
                                "model_id": model_id,
                                "model_name": model_name_by_id.get(model_id)
                                or self._fallback_name("Model", model_id),
                                "assets": assets_payload,
                            }
                        )

                    if not models_payload:
                        continue

                    areas_payload.append(
                        {
                            "area_id": area_id,
                            "area_name": area_name_by_id.get(area_id)
                            or self._fallback_name("Area", area_id),
                            "models": models_payload,
                        }
                    )

                # --------------------------------------------------
                # 8) Meta
                # --------------------------------------------------
                meta = {
                    "area_count": len(areas_payload),
                    "model_count": sum(len(a["models"]) for a in areas_payload),
                    "asset_count": sum(
                        len(m["assets"])
                        for a in areas_payload
                        for m in a["models"]
                    ),
                    "drawing_count": len(drawing_payload_by_id),
                }

                debug_id(
                    "[DrawingNavigationProjection] FINAL "
                    f"areas={meta['area_count']} "
                    f"models={meta['model_count']} "
                    f"assets={meta['asset_count']} "
                    f"drawings={meta['drawing_count']}",
                    rid,
                )

                result = {
                    "complete_document_id": complete_document_id,
                    "areas": areas_payload,
                    "meta": meta,
                }

                self._navigation_cache[cache_key] = deepcopy(result)

                return result

        except Exception as exc:
            error_id(
                f"[DrawingNavigationProjection] build_navigation failed: {exc}",
                rid,
                exc_info=True,
            )
            return {
                "complete_document_id": complete_document_id,
                "areas": [],
                "meta": {
                    "area_count": 0,
                    "model_count": 0,
                    "asset_count": 0,
                    "drawing_count": 0,
                },
                "error": "drawing_navigation_failed",
            }

    # ---------------------------------------------------------
    # Internals
    # ---------------------------------------------------------

    def _load_positions(self, position_ids: List[int], rid: str) -> List[Any]:
        """
        Load only Position columns needed for navigation.
        """

        try:
            rows = (
                self.session.query(
                    Position.id.label("id"),
                    Position.area_id.label("area_id"),
                    Position.model_id.label("model_id"),
                    Position.asset_number_id.label("asset_number_id"),
                )
                .filter(Position.id.in_(position_ids))
                .all()
            )

            debug_id(
                f"[DrawingNavigationProjection] Loaded positions={len(rows)}",
                rid,
            )

            return rows

        except Exception as exc:
            error_id(
                f"[DrawingNavigationProjection] Failed loading positions: {exc}",
                rid,
                exc_info=True,
            )
            return []

    def _fetch_drawings_for_positions(
        self,
        position_ids: List[int],
        rid: str,
    ) -> Tuple[List[Any], Set[int], Dict[int, Set[int]]]:
        """
        Returns:
            drawings: lightweight drawing rows deduped by drawing id
            drawing_ids: set[int]
            drawing_to_positions: dict[drawing_id -> set[position_id)]
        """

        drawing_rows_by_id: Dict[int, Any] = {}
        drawing_to_positions: DefaultDict[int, Set[int]] = defaultdict(set)

        try:
            drawing_columns = self._drawing_columns()

            rows = (
                self.session.query(*drawing_columns)
                .join(
                    DrawingPositionAssociation,
                    DrawingPositionAssociation.drawing_id == Drawing.id,
                )
                .filter(DrawingPositionAssociation.position_id.in_(position_ids))
                .all()
            )

            if not rows:
                return [], set(), {}

            for row in rows:
                drawing_id = getattr(row, "id", None)
                position_id = getattr(row, "position_id", None)

                if drawing_id is None:
                    continue

                drawing_id = int(drawing_id)

                if drawing_id not in drawing_rows_by_id:
                    drawing_rows_by_id[drawing_id] = row

                if position_id is not None:
                    drawing_to_positions[drawing_id].add(int(position_id))

            debug_id(
                "[DrawingNavigationProjection] Loaded drawings="
                f"{len(drawing_rows_by_id)} assoc_rows={len(rows)}",
                rid,
            )

            return (
                list(drawing_rows_by_id.values()),
                set(drawing_rows_by_id.keys()),
                dict(drawing_to_positions),
            )

        except Exception as exc:
            error_id(
                f"[DrawingNavigationProjection] Failed fetching drawings: {exc}",
                rid,
                exc_info=True,
            )
            return [], set(), {}

    def _fetch_spare_parts_for_drawings(
        self,
        *,
        drawing_ids: List[int],
        rid: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Batch-resolve spare parts for all drawings.

        This replaces the old N+1 pattern:

            for drawing_id in drawing_ids:
                get_parts_by_drawing(drawing_id)

        with one database query.
        """

        drawing_ids = self._normalize_ids(drawing_ids)

        if not drawing_ids:
            return {}

        try:
            from modules.emtacdb.emtacdb_fts import DrawingPartAssociation, Part

            rows = (
                self.session.query(
                    DrawingPartAssociation.drawing_id.label("drawing_id"),
                    Part.id.label("part_id"),
                    Part.part_number.label("part_number"),
                    Part.name.label("name"),
                    Part.oem_mfg.label("oem_mfg"),
                    Part.model.label("model"),
                )
                .join(
                    Part,
                    DrawingPartAssociation.part_id == Part.id,
                )
                .filter(DrawingPartAssociation.drawing_id.in_(drawing_ids))
                .all()
            )

            spare_parts_by_drawing: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
            seen: Set[Tuple[int, int]] = set()

            for row in rows:
                drawing_id = getattr(row, "drawing_id", None)
                part_id = getattr(row, "part_id", None)

                if drawing_id is None or part_id is None:
                    continue

                drawing_id = int(drawing_id)
                part_id = int(part_id)

                key = (drawing_id, part_id)

                if key in seen:
                    continue

                seen.add(key)

                spare_parts_by_drawing[drawing_id].append(
                    {
                        "part_id": part_id,
                        "part_number": getattr(row, "part_number", None),
                        "name": getattr(row, "name", None),
                        "oem_mfg": getattr(row, "oem_mfg", None),
                        "model": getattr(row, "model", None),
                    }
                )

            debug_id(
                "[DrawingNavigationProjection] Batch spare parts resolved "
                f"drawings={len(drawing_ids)} "
                f"rows={len(rows)} "
                f"drawings_with_parts={len(spare_parts_by_drawing)}",
                rid,
            )

            return dict(spare_parts_by_drawing)

        except Exception as exc:
            error_id(
                "[DrawingNavigationProjection] Failed batch loading spare parts "
                f"for drawings: {exc}",
                rid,
                exc_info=True,
            )
            return {}

    def _serialize_drawing(self, drawing: Any) -> Dict[str, Any]:
        """
        UI-safe drawing serializer.
        """

        drawing_id = getattr(drawing, "id", None)

        return {
            "id": drawing_id,
            "drw_equipment_name": getattr(drawing, "drw_equipment_name", None),
            "drw_number": getattr(drawing, "drw_number", None),
            "drw_name": getattr(drawing, "drw_name", None),
            "drw_revision": getattr(drawing, "drw_revision", None),
            "drw_spare_part_number": getattr(drawing, "drw_spare_part_number", None),
            "url": f"/drawings/{drawing_id}" if drawing_id is not None else None,
        }

    def _resolve_names(
        self,
        model: Type[Any],
        ids: Iterable[int],
        *,
        fallback_prefix: str,
        rid: Optional[str] = None,
    ) -> Dict[Optional[int], str]:
        """
        Resolve IDs to display names safely.

        Supports:
        - Area.name
        - Model.name
        - AssetNumber.number
        - description fallback
        """

        clean_ids = self._normalize_ids(list(ids))

        if not clean_ids:
            return {}

        try:
            display_column = self._display_column_for_model(model)

            rows = (
                self.session.query(
                    model.id.label("id"),
                    display_column.label("display_name"),
                )
                .filter(model.id.in_(clean_ids))
                .all()
            )

            name_by_id: Dict[Optional[int], str] = {}

            for row in rows:
                row_id = getattr(row, "id", None)
                display_name = getattr(row, "display_name", None)

                if row_id is None:
                    continue

                name_by_id[int(row_id)] = (
                    display_name
                    or self._fallback_name(fallback_prefix, int(row_id))
                )

            debug_id(
                f"[DrawingNavigationProjection] Resolved {model.__name__} names: {len(name_by_id)}",
                rid,
            )

            return name_by_id

        except Exception as exc:
            error_id(
                f"[DrawingNavigationProjection] Failed resolving {model.__name__} names: {exc}",
                rid,
                exc_info=True,
            )
            return {}

    @staticmethod
    def _display_column_for_model(model: Type[Any]) -> Any:
        """
        Pick the best display column for a model.
        """

        for attr_name in ("name", "number", "description", "title"):
            column = getattr(model, attr_name, None)

            if column is not None:
                return column

        return model.id

    @staticmethod
    def _drawing_columns() -> List[Any]:
        """
        Return only drawing columns needed by the frontend.
        """

        columns = [
            Drawing.id.label("id"),
            Drawing.drw_number.label("drw_number"),
            Drawing.drw_name.label("drw_name"),
            Drawing.drw_revision.label("drw_revision"),
            DrawingPositionAssociation.position_id.label("position_id"),
        ]

        for optional_name in (
            "drw_equipment_name",
            "drw_spare_part_number",
            "complete_document_id",
        ):
            column = getattr(Drawing, optional_name, None)

            if column is not None:
                columns.append(column.label(optional_name))

        return columns

    def _sort_drawing_ids(
        self,
        drawing_ids: Iterable[int],
        drawing_payload_by_id: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        Stable drawing ordering for UI navigation.
        """

        def sort_key(drawing_id: int) -> Tuple[str, str, int]:
            payload = drawing_payload_by_id.get(drawing_id, {})

            return (
                str(payload.get("drw_number") or ""),
                str(payload.get("drw_name") or ""),
                int(drawing_id),
            )

        return sorted(
            [int(drawing_id) for drawing_id in drawing_ids if drawing_id in drawing_payload_by_id],
            key=sort_key,
        )

    @staticmethod
    def _normalize_ids(values: Optional[Iterable[Any]]) -> List[int]:
        cleaned: List[int] = []

        if not values:
            return cleaned

        for value in values:
            if value in (None, "", "None"):
                continue

            try:
                cleaned.append(int(value))
            except (TypeError, ValueError):
                continue

        return sorted(dict.fromkeys(cleaned))

    @staticmethod
    def _fallback_name(prefix: str, id_val: Optional[int]) -> str:
        if id_val is None:
            return f"{prefix} (Unknown)"

        return f"{prefix} #{id_val}"

    @staticmethod
    def _empty_navigation(
        *,
        complete_document_id: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "complete_document_id": complete_document_id,
            "areas": [],
            "meta": {
                "area_count": 0,
                "model_count": 0,
                "asset_count": 0,
                "drawing_count": 0,
            },
        }