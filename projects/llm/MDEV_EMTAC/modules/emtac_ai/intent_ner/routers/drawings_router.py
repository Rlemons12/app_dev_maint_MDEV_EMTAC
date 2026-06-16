from typing import Any, Dict

from modules.services import DBServices
from .base_router import BaseRouter


DB = DBServices()


# ----------------------------------------------------------------------
# SERIALIZER
# ----------------------------------------------------------------------
def serialize_drawing(d):
    return {
        "id": d.id,
        "number": d.drw_number,
        "name": d.drw_name,
        "equipment_name": d.drw_equipment_name,
        "revision": d.drw_revision,
        "spare_part_number": d.drw_spare_part_number,
        "type": d.drw_type,
        "file_path": d.file_path,
    }


router = BaseRouter(DB.drawings, serialize_drawing)


# ----------------------------------------------------------------------
# PRIORITY HANDLERS
# ----------------------------------------------------------------------
def search_by_drawing_number(entities):
    num = entities.get("drawing_number")
    if not num:
        return None

    hits = DB.drawings.find(drw_number=num)
    if hits:
        return (hits, search_by_drawing_number.__name__)
    return None


def search_by_drawing_name(entities):
    name = entities.get("drawing_name")
    if not name:
        return None

    hits = DB.drawings.find(drw_name=name)
    if hits:
        return (hits, search_by_drawing_name.__name__)
    return None


def search_by_equipment(entities):
    equip = entities.get("equipment_name")
    if not equip:
        return None

    hits = DB.drawings.find(drw_equipment_name=equip)
    if hits:
        return (hits, search_by_equipment.__name__)
    return None


def search_by_metadata(entities):
    rev = entities.get("revision")
    spare = entities.get("spare_part_number")
    dtype = entities.get("drawing_type")

    if not (rev or spare or dtype):
        return None

    hits = DB.drawings.find(
        drw_revision=rev,
        drw_spare_part_number=spare,
        drw_type=dtype
    )
    if hits:
        return (hits, search_by_metadata.__name__)
    return None


def search_by_file_fragment(entities):
    file_fragment = entities.get("file_name")
    if not file_fragment:
        return None

    hits = DB.drawings.find(file_path=file_fragment)
    if hits:
        return (hits, search_by_file_fragment.__name__)
    return None


# ----------------------------------------------------------------------
# FALLBACK HANDLER — tests expect matched_on="search_text"
# ----------------------------------------------------------------------
def fallback_search(text: str):
    hits = DB.drawings.find(search_text=text)
    if hits:
        return (hits, "search_text")
    return None


# ----------------------------------------------------------------------
# MAIN ROUTER ENTRY
# ----------------------------------------------------------------------
def drawings_router(
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
            search_by_drawing_number,
            search_by_drawing_name,
            search_by_equipment,
            search_by_metadata,
            search_by_file_fragment,
        ],
        fallback=fallback_search,
    )
