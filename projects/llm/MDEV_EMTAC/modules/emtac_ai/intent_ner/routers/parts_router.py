from typing import Any, Dict, Callable, List, Optional

from modules.services import DBServices
from .base_router import BaseRouter

DB = DBServices()


# ============================================================
# SERIALIZER
# ============================================================
def serialize_part(p):
    return {
        "id": p.id,
        "part_number": p.part_number,
        "name": p.name,
        "oem_mfg": p.oem_mfg,
        "model": p.model,
        "class_flag": p.class_flag,
        "notes": p.notes,
    }


# Bind BaseRouter to PartService
router = BaseRouter(DB.parts, serialize_part)


# ============================================================
# PRIORITY HANDLERS
# ============================================================

def by_part_id(entities):
    pid = entities.get("part_id")
    if not pid:
        return None

    part = DB.parts.get(pid)
    if not part:
        return None

    return [part], "by_part_id"


def by_part_number(entities):
    pn = entities.get("part_number")
    if not pn:
        return None

    hits = DB.parts.find(part_number=pn)
    if hits:
        return hits, "by_part_number"

    return None


def by_part_name(entities):
    name = entities.get("part_name")
    if not name:
        return None

    hits = DB.parts.find(name=name)
    if hits:
        return hits, "by_part_name"

    return None


def by_oem(entities):
    oem = entities.get("oem_mfg")
    if not oem:
        return None

    hits = DB.parts.find(oem_mfg=oem)
    if hits:
        return hits, "by_oem_mfg"

    return None


def by_position(entities):
    pos = entities.get("position")
    if not pos:
        return None

    hits = DB.parts.find_by_position(pos)
    if hits:
        return hits, "by_position"

    return None


def by_drawing(entities):
    drw = entities.get("drawing_id")
    if not drw:
        return None

    hits = DB.drawing_part_associations.get_parts_by_drawing(drawing_id=drw)
    if hits:
        return hits, "by_drawing"

    return None


def by_image(entities):
    image_id = entities.get("image_id")
    if not image_id:
        return None

    hits = DB.parts.find_by_image(image_id)
    if hits:
        return hits, "by_image"

    return None


# ============================================================
# FALLBACK SEARCH
# ============================================================
def fallback_search(text: str):
    """
    Fallback order:
        1) FTS via search_text → returns dict rows (needs special serializer)
        2) metadata search → returns Part model rows
    """

    # ---------- Full-text search ----------
    fts_hits = DB.parts.search_text(text)
    if fts_hits:
        return (
            fts_hits,
            "fts",
            serialize_fts_row,       # custom serializer for dict rows
        )

    # ---------- Metadata search ----------
    meta = DB.parts.search(text)
    if meta:
        return meta, "metadata"

    return None


def serialize_fts_row(row):
    """Serializer for FTS fallback rows."""
    return {
        "id": row.get("part_id"),
        "part_number": row.get("part_number"),
        "name": row.get("name"),
        "rank": float(row.get("rank", 0)),
    }


# ============================================================
# MAIN ROUTER ENTRYPOINT
# ============================================================
def parts_router(
    *,
    text: str,
    intent: str,
    confidence: float,
    entities: Dict[str, Any]
):
    return router.route(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        priority_handlers=[
            by_part_id,
            by_part_number,
            by_part_name,
            by_oem,
            by_position,
            by_drawing,
            by_image,
        ],
        fallback=fallback_search,
    )
