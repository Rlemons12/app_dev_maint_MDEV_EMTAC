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
#       "drawings": [ <drawing_payload>, ... ],            # all drawings in this area
#       "models": [
#         {
#           "model_id": int | None,
#           "model_name": str,
#           "drawings": [ <drawing_payload>, ... ],        # drawings for this model in this area
#           "assets": [
#             {
#               "asset_number_id": int | None,
#               "asset_name": str,
#               "drawings": [ <drawing_payload>, ... ]     # drawings for this asset in this model+area
#             }
#           ]
#         }
#       ]
#     }
#   ],
#   "meta": { "area_count": int, "model_count": int, "asset_count": int, "drawing_count": int }
# }
#
# Notes:
# - Names are resolved immediately (Area/Model/Asset) so frontend can render hover menus right away.
# - We prune empty nodes:
#   - only Areas that have at least one drawing are returned
#   - only Models that have at least one drawing are returned
#   - only Assets that have at least one drawing are returned
# - We DO NOT expose filesystem paths. For drawings we emit a URL placeholder:
#   url: f"/drawings/{drawing_id}"
#   Adjust this to your real route if different.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set, DefaultDict
from collections import defaultdict
from modules.services.drawing_part_association_service import (
    DrawingPartAssociationService
)

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
    log_timed_operation,
)

# ORM models (adjust imports if your modules path differs)
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
    Builds the "Area → Model → Asset" navigation structure for drawings associated
    with a CompleteDocument via its Positions.
    """

    def __init__(self, session):
        self.session = session

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
        drawing_part_service = DrawingPartAssociationService()
        # --------------------------------------------------
        # Normalize + guard
        # --------------------------------------------------
        position_ids = sorted({int(pid) for pid in (position_ids or []) if pid is not None})
        if not position_ids:
            debug_id(
                "[DrawingNavigationProjection] No position_ids provided",
                rid,
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
            }

        try:
            with log_timed_operation("DrawingNavigationProjection.build_navigation", rid):

                # --------------------------------------------------
                # 1) Load positions
                # --------------------------------------------------
                positions = self._load_positions(position_ids=position_ids, rid=rid)
                if not positions:
                    debug_id(
                        "[DrawingNavigationProjection] No Position rows loaded",
                        rid,
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
                    }

                pos_key_by_id: Dict[int, _PositionKey] = {}
                area_ids, model_ids, asset_ids = set(), set(), set()

                for p in positions:
                    key = _PositionKey(
                        area_id=getattr(p, "area_id", None),
                        model_id=getattr(p, "model_id", None),
                        asset_number_id=getattr(p, "asset_number_id", None),
                    )
                    pos_key_by_id[p.id] = key

                    if key.area_id:
                        area_ids.add(key.area_id)
                    if key.model_id:
                        model_ids.add(key.model_id)
                    if key.asset_number_id:
                        asset_ids.add(key.asset_number_id)

                # --------------------------------------------------
                # 2) Resolve names
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
                drawings, drawing_ids, drawing_to_positions = (
                    self._fetch_drawings_for_positions(
                        position_ids=position_ids,
                        rid=rid,
                    )
                )

                debug_id(
                    f"[DrawingNavigationProjection] drawings_found={len(drawings)}",
                    rid,
                )

                if not drawings:
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

                # --------------------------------------------------
                # 4) OPTIONAL document filter (SAFE)
                # --------------------------------------------------
                if complete_document_id is not None:
                    doc_filtered = [
                        d for d in drawings
                        if getattr(d, "complete_document_id", None) == complete_document_id
                    ]

                    if doc_filtered:
                        drawings = doc_filtered
                        drawing_ids = {d.id for d in drawings}
                        drawing_to_positions = {
                            did: pids
                            for did, pids in drawing_to_positions.items()
                            if did in drawing_ids
                        }

                        debug_id(
                            f"[DrawingNavigationProjection] "
                            f"document_filter_applied drawings={len(drawings)}",
                            rid,
                        )
                    else:
                        debug_id(
                            "[DrawingNavigationProjection] "
                            "document filter produced 0 results — using position drawings",
                            rid,
                        )

                # --------------------------------------------------
                # 5) Serialize drawings
                # --------------------------------------------------
                drawing_payload_by_id = {
                    d.id: self._serialize_drawing(d) for d in drawings
                }

                # --------------------------------------------------
                # 6) Bucket by Area → Model → Asset
                # --------------------------------------------------
                area_drawings = defaultdict(set)
                model_drawings = defaultdict(set)
                asset_drawings = defaultdict(set)

                models_in_area = defaultdict(set)
                assets_in_model = defaultdict(set)

                for drw_id, pos_ids_for_drw in drawing_to_positions.items():
                    if drw_id not in drawing_payload_by_id:
                        continue

                    for pid in pos_ids_for_drw:
                        key = pos_key_by_id.get(pid)
                        if not key:
                            continue

                        a, m, s = key.area_id, key.model_id, key.asset_number_id

                        area_drawings[a].add(drw_id)
                        model_drawings[(a, m)].add(drw_id)
                        asset_drawings[(a, m, s)].add(drw_id)

                        models_in_area[a].add(m)
                        assets_in_model[(a, m)].add(s)

                # --------------------------------------------------
                # 5.5) Resolve spare parts for drawings
                # --------------------------------------------------
                drawing_ids_list = list(drawing_payload_by_id.keys())
                spare_parts_by_drawing: Dict[int, List[Dict[str, Any]]] = {}

                if drawing_ids_list:
                    for drw_id in drawing_ids_list:
                        parts = drawing_part_service.get_parts_by_drawing(
                            drawing_id=drw_id
                        )

                        spare_parts_by_drawing[drw_id] = [
                            {
                                "part_id": p.id,
                                "part_number": p.part_number,
                                "name": p.name,
                                "oem_mfg": p.oem_mfg,
                                "model": p.model,
                            }
                            for p in parts
                        ]
                # --------------------------------------------------
                # 5.6) Attach spare parts to drawing payloads
                # --------------------------------------------------
                for drw_id, payload in drawing_payload_by_id.items():
                    payload["spare_parts"] = spare_parts_by_drawing.get(drw_id, [])


                # --------------------------------------------------
                # 7) Build navigation payload
                # --------------------------------------------------
                areas_payload: List[Dict[str, Any]] = []

                for a_id in sorted(area_drawings.keys(), key=lambda x: area_name_by_id.get(x, "")):
                    models_payload = []

                    for m_id in sorted(models_in_area[a_id], key=lambda x: model_name_by_id.get(x, "")):
                        assets_payload = []

                        for s_id in sorted(
                                assets_in_model[(a_id, m_id)],
                                key=lambda x: asset_name_by_id.get(x, ""),
                        ):
                            drw_ids = asset_drawings[(a_id, m_id, s_id)]
                            if not drw_ids:
                                continue

                            assets_payload.append({
                                "asset_number_id": s_id,
                                "asset_name": asset_name_by_id.get(s_id)
                                              or self._fallback_name("Asset", s_id),
                                "drawings": [
                                    drawing_payload_by_id[i] for i in drw_ids
                                ],
                            })

                        if not assets_payload:
                            continue

                        models_payload.append({
                            "model_id": m_id,
                            "model_name": model_name_by_id.get(m_id)
                                          or self._fallback_name("Model", m_id),
                            "assets": assets_payload,
                        })

                    if not models_payload:
                        continue

                    areas_payload.append({
                        "area_id": a_id,
                        "area_name": area_name_by_id.get(a_id)
                                     or self._fallback_name("Area", a_id),
                        "models": models_payload,
                    })

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
                    f"[DrawingNavigationProjection] FINAL "
                    f"areas={meta['area_count']} "
                    f"models={meta['model_count']} "
                    f"assets={meta['asset_count']} "
                    f"drawings={meta['drawing_count']}",
                    rid,
                )

                return {
                    "complete_document_id": complete_document_id,
                    "areas": areas_payload,
                    "meta": meta,
                }

        except Exception as e:
            error_id(
                f"[DrawingNavigationProjection] build_navigation failed: {e}",
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
    def _load_positions(self, position_ids: List[int], rid: str) -> List[Position]:
        try:
            # Keep it lightweight: only need FK columns on Position
            q = self.session.query(Position).filter(Position.id.in_(position_ids))
            positions = q.all()
            debug_id(f"[DrawingNavigationProjection] Loaded positions={len(positions)}", rid)
            return positions
        except Exception as e:
            error_id(f"[DrawingNavigationProjection] Failed loading positions: {e}", rid, exc_info=True)
            return []

    def _resolve_area_names(self, area_ids: Set[int], rid: str) -> Dict[int, str]:
        if not area_ids:
            return {}

        rows = (
            self.session.query(Area.id, Area.name)
            .filter(Area.id.in_(area_ids))
            .all()
        )

        result = {row.id: row.name for row in rows}
        debug_id(f"[DrawingNavigationProjection] Resolved Area names: {len(result)}", rid)
        return result

    def _resolve_model_names(self, model_ids: Set[int], rid: str) -> Dict[int, str]:
        if not model_ids:
            return {}

        rows = (
            self.session.query(Model.id, Model.name)
            .filter(Model.id.in_(model_ids))
            .all()
        )

        result = {row.id: row.name for row in rows}
        debug_id(f"[DrawingNavigationProjection] Resolved Model names: {len(result)}", rid)
        return result

    def _resolve_asset_numbers(self, asset_ids: Set[int], rid: str) -> Dict[int, str]:
        if not asset_ids:
            return {}

        rows = (
            self.session.query(AssetNumber.id, AssetNumber.number)
            .filter(AssetNumber.id.in_(asset_ids))
            .all()
        )

        result = {row.id: row.number for row in rows}
        debug_id(f"[DrawingNavigationProjection] Resolved Asset numbers: {len(result)}", rid)
        return result

    def _fallback_name(self, prefix: str, id_val: Optional[int]) -> str:
        if id_val is None:
            return f"{prefix} (Unknown)"
        return f"{prefix} #{id_val}"

    def _fetch_drawings_for_positions(
        self,
        position_ids: List[int],
        rid: str,
    ) -> Tuple[List[Drawing], Set[int], Dict[int, Set[int]]]:
        """
        Returns:
          drawings: list[Drawing] (deduped)
          drawing_ids: set[int]
          drawing_to_positions: dict[drawing_id -> set[position_id]]
        """
        drawings_map: Dict[int, Drawing] = {}
        drawing_to_positions: DefaultDict[int, Set[int]] = defaultdict(set)

        try:
            # Query associations once (avoid N+1)
            rows = (
                self.session.query(DrawingPositionAssociation.drawing_id, DrawingPositionAssociation.position_id)
                .filter(DrawingPositionAssociation.position_id.in_(position_ids))
                .all()
            )

            if not rows:
                return [], set(), {}

            drawing_ids = sorted({int(r[0]) for r in rows if r[0] is not None})
            for drawing_id, position_id in rows:
                if drawing_id is None or position_id is None:
                    continue
                drawing_to_positions[int(drawing_id)].add(int(position_id))

            # Load drawing objects
            drawings = (
                self.session.query(Drawing)
                .filter(Drawing.id.in_(drawing_ids))
                .all()
            )

            for d in drawings:
                drawings_map[d.id] = d

            debug_id(
                f"[DrawingNavigationProjection] Loaded drawings={len(drawings_map)} assoc_rows={len(rows)}",
                rid,
            )

            return list(drawings_map.values()), set(drawings_map.keys()), dict(drawing_to_positions)

        except Exception as e:
            error_id(f"[DrawingNavigationProjection] Failed fetching drawings: {e}", rid, exc_info=True)
            return [], set(), {}

    def _serialize_drawing(self, d: Drawing) -> Dict[str, Any]:
        """
        UI-safe drawing serializer.
        Adjust URL routing here to match your Flask route.
        """
        return {
            "id": d.id,
            "drw_equipment_name": getattr(d, "drw_equipment_name", None),
            "drw_number": getattr(d, "drw_number", None),
            "drw_name": getattr(d, "drw_name", None),
            "drw_revision": getattr(d, "drw_revision", None),
            "drw_spare_part_number": getattr(d, "drw_spare_part_number", None),
            # IMPORTANT: do not expose file_path; provide a route instead
            "url": f"/drawings/{d.id}",
        }

    def _resolve_names(self,model: Type,ids: Iterable[int],*,fallback_prefix: str,rid: str | None = None,)\
            -> Dict[int, str]:
        """
        Resolve a set of model IDs to {id: name} safely.

        - Missing rows are tolerated
        - Returns fallback names if needed
        """
        ids = {int(i) for i in ids if i is not None}
        if not ids:
            return {}

        rows = (
            self.session.query(model)
            .filter(model.id.in_(ids))
            .all()
        )

        name_by_id: Dict[int, str] = {}

        for row in rows:
            name = getattr(row, "name", None) or getattr(row, "description", None)
            if not name:
                name = f"{fallback_prefix} {row.id}"
            name_by_id[row.id] = name

        debug_id(
            f"[DrawingNavigationProjection] Resolved {model.__name__} names: {len(name_by_id)}",
            rid,
        )

        return name_by_id
