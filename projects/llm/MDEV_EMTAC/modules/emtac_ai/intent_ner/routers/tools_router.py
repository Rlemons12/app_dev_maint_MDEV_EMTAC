"""
Tools intent router.

Priority:
    1) tool_id
    2) name
    3) metadata filters (type, size, material)
    4) position → tools
    5) fallback fuzzy search
"""

from typing import Any, Dict, Callable, Optional

from modules.services import DBServices
from .base_router import BaseRouter

DB = DBServices()


# ============================================================
# SERIALIZER
# ============================================================
def serialize_tool(t):
    return {
        "id": t.id,
        "name": t.name,
        "type": t.type,
        "size": t.size,
        "material": t.material,
        "description": t.description,
    }


# BaseRouter wrapper
router = BaseRouter(DB.tools, serialize_tool)


# ============================================================
# PRIORITY HANDLERS
# ============================================================
def by_tool_id(entities):
    tid = entities.get("tool_id")
    if not tid:
        return None

    tool = DB.tools.get(tid)
    if not tool:
        return None

    # Third slot = related context object
    related = build_related(tid)

    # FORMAT: (models, matched_on, related_payload OR custom_serializer)
    return ([tool], "by_tool_id", related)


def by_name(entities):
    name = entities.get("name")
    if not name:
        return None

    hits = DB.tools.find(name=name)
    if hits:
        return (hits, "by_name")

    return None


def by_metadata(entities):
    ttype = entities.get("type")
    size = entities.get("size")
    material = entities.get("material")

    if not (ttype or size or material):
        return None

    hits = DB.tools.find(type=ttype, size=size, material=material)
    if hits:
        return (hits, "by_metadata")

    return None


def by_position(entities):
    pos = entities.get("position")
    if not pos:
        return None

    hits = DB.tools.find_by_position(pos)
    if hits:
        return (hits, "by_position")

    return None


# ============================================================
# FALLBACK SEARCH
# ============================================================
def fallback_search(text: str):
    fuzzy = DB.tools.search_name(text)
    if fuzzy:
        return (fuzzy, "fallback")
    return None


# ============================================================
# RELATED ITEMS
# ============================================================
def build_related(tool_id: int):
    out = {
        "positions": [],
        "tasks": [],
        "problems": [],
        "solutions": [],
    }

    try:
        out["positions"] = DB.tools.get_positions_for_tool(tool_id)
    except:
        pass

    try:
        out["tasks"] = DB.tools.get_tasks_for_tool(tool_id)
    except:
        pass

    try:
        out["problems"] = DB.tools.get_problems_for_tool(tool_id)
    except:
        pass

    try:
        out["solutions"] = DB.tools.get_solutions_for_tool(tool_id)
    except:
        pass

    return out


# ============================================================
# MAIN ROUTER ENTRY POINT
# ============================================================
def tools_router(
    *,
    text: str,
    intent: str,
    confidence: float,
    entities: Dict[str, Any],
):
    return router.route(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        priority_handlers=[
            by_tool_id,
            by_name,
            by_metadata,
            by_position,
        ],
        fallback=fallback_search,
    )
