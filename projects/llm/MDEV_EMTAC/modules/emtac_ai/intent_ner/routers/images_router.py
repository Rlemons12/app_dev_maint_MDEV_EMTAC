"""
Images intent router.

Updated to use ImagePositionService and correct ImageService API.

Routing priorities:
    1. image_id → direct lookup
    2. title / file_name searches
    3. position-based lookup via ImagePositionService
    4. hierarchy-based lookup (area, model, equipment group, etc.)
    5. document-linked images
    6. hybrid pgvector search
    7. simple metadata search
    8. nothing found
"""

from typing import Any, Dict, List
from modules.services import DBServices

DB = DBServices()


# ======================================================================
# MAIN ROUTER
# ======================================================================
def images_router(
    *,
    text: str,
    intent: str,
    confidence: float,
    entities: Dict[str, Any],
) -> Dict[str, Any]:

    image_id = entities.get("image_id")
    title = entities.get("title")
    file_name = entities.get("file_name")

    # Extract full hierarchy
    position_id = entities.get("position")
    area_id = entities.get("area_id")
    equipment_group_id = entities.get("equipment_group_id")
    model_id = entities.get("model_id")
    asset_number_id = entities.get("asset_number_id")
    location_id = entities.get("location_id")
    subassembly_id = entities.get("subassembly_id")
    component_assembly_id = entities.get("component_assembly_id")
    assembly_view_id = entities.get("assembly_view_id")
    site_location_id = entities.get("site_location_id")

    complete_document = entities.get("complete_document")

    # ==================================================================
    # 1) Direct ID Lookup
    # ==================================================================
    if image_id:
        img = DB.images_service.get(image_id)
        if img:
            related = DB.images_service.find_related(image_id)
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=_serialize_image(img),
                related=related,
                matched_on="image_id",
            )

    # ==================================================================
    # 2) TITLE search
    # ==================================================================
    if title:
        hits = DB.images_service.find(title=title)
        if hits:
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=[_serialize_image(i) for i in hits],
                matched_on="title",
            )

    # ==================================================================
    # 3) FILE NAME search
    # ==================================================================
    if file_name:
        hits = DB.images_service.find(file_path=file_name)
        if hits:
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=[_serialize_image(i) for i in hits],
                matched_on="file_name",
            )

    # ==================================================================
    # 4) POSITION-based lookup
    # ==================================================================
    if position_id:
        hits = DB.image_position_service.get_images_by_position(position_id)
        if hits:
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=[_serialize_image(i) for i in hits],
                matched_on="position",
            )

    # ==================================================================
    # 5) Hierarchy-based lookup
    # ==================================================================
    hierarchy_filters = {
        "area_id": area_id,
        "equipment_group_id": equipment_group_id,
        "model_id": model_id,
        "asset_number_id": asset_number_id,
        "location_id": location_id,
        "subassembly_id": subassembly_id,
        "component_assembly_id": component_assembly_id,
        "assembly_view_id": assembly_view_id,
        "site_location_id": site_location_id,
    }

    if any(hierarchy_filters.values()):
        hits = DB.image_position_service.get_images_by_hierarchy(**hierarchy_filters)
        if hits:
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=[_serialize_image(i) for i in hits],
                matched_on="hierarchy",
            )

    # ==================================================================
    # 6) Document-linked images
    # ==================================================================
    if complete_document:
        hits = DB.images_service.find(complete_document_id=complete_document)
        if hits:
            return _response(
                text=text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                results=[_serialize_image(i) for i in hits],
                matched_on="complete_document",
            )

    # ==================================================================
    # 7) Hybrid PGVector search
    # ==================================================================
    hybrid = DB.images_service.search_images(query=text, limit=15)
    if hybrid:
        return _response(
            text=text,
            intent=intent,
            confidence=confidence,
            entities=entities,
            results=_serialize_hybrid(hybrid),
            matched_on="hybrid_pgvector",
        )

    # ==================================================================
    # 8) Simple fallback metadata search
    # ==================================================================
    basic = DB.images_service.find(title=text)
    if basic:
        return _response(
            text=text,
            intent=intent,
            confidence=confidence,
            entities=entities,
            results=[_serialize_image(i) for i in basic],
            matched_on="basic_find",
        )

    # ==================================================================
    # 9) NOTHING FOUND
    # ==================================================================
    return _response(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        results=[],
        matched_on="no_results",
    )


# ======================================================================
# RESPONSE FORMATTER
# ======================================================================

def _response(text, intent, confidence, entities, results, matched_on, related=None):
    # Normalize results into a list
    if results is None:
        results = []
    elif isinstance(results, dict):
        results = [results]

    return {
        "router": "Images",
        "intent": intent,
        "confidence": confidence,
        "query": text,
        "matched_on": matched_on,
        "entities": entities,
        "results": results,
        "related": related,
    }

# ======================================================================
# SERIALIZERS
# ======================================================================
def _serialize_image(img):
    return {
        "id": img.id,
        "title": img.title,
        "description": img.description,
        "file_path": img.file_path,
    }


def _serialize_hybrid(rows):
    out = []
    for r in rows:
        out.append({
            "id": r.get("image_id"),
            "title": r.get("title"),
            "file_path": r.get("file_path"),
            "score": float(r.get("score", 0)),
        })
    return out
