from typing import Any, Dict, List, Callable
from modules.services import DBServices
from modules.emtac_ai.intent_ner.routers.base_router import BaseRouter

DB = DBServices()


def serialize_problem(p):
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
    }


# ---------------------------------------------
# Priority handlers
# ---------------------------------------------
def _by_problem_id(entities):
    pid = entities.get("problem_id")
    if not pid:
        return None

    prob = DB.troubleshooting.get_problem(pid)
    if not prob:
        return None

    return [prob], "problem_id"      # FIXED (no nested list)


def _by_problem_name(entities):
    name = entities.get("problem_name")
    if not name:
        return None

    matches = DB.troubleshooting.search_problems(name)
    if not matches:
        return None

    if len(matches) == 1:
        return [matches[0]], "problem_name"     # FIXED
    else:
        return matches, "problem_name"          # FIXED


def _fallback(text):
    resolved = DB.troubleshooting.resolve_query(text)
    status = resolved.get("status")

    if status == "no_match":
        return [], "no_results", None

    if status == "multiple_matches":
        return resolved["choices"], "multiple_results", None

    if status == "resolved":
        # IMPORTANT FIX:
        # return BOTH the problem AND the tree so BaseRouter can output it
        problem = resolved["problem"]
        tree = resolved.get("tree")
        return [problem], "resolved", tree

    return None

# ---------------------------------------------
# Main router (using BaseRouter)
# ---------------------------------------------
def troubleshooting_router(*, text, intent, confidence, entities):

    router = BaseRouter(
        service=DB.troubleshooting,
        serializer=serialize_problem,
    )

    # Run the router
    result = router.route(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        priority_handlers=[
            _by_problem_id,
            _by_problem_name,
        ],
        fallback=_fallback,
    )

    # --- FIX FOR TREE IN RESOLVED CASE ---
    # BaseRouter does not know how to unpack the third return value from fallback,
    # so we patch here when BaseRouter outputs matched_on="resolved".
    if result.get("matched_on") == "resolved":
        # fallback returns ([problem], "resolved", tree)
        resolved = DB.troubleshooting.resolve_query(text)
        tree = resolved.get("tree")
        result["related"] = tree

    return result


__all__ = ["troubleshooting_router"]
