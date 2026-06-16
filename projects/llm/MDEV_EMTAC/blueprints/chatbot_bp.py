# routes/chatbot_bp.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import inspect
import json
import time

from flask import Blueprint, request, jsonify

from modules.configuration.log_config import (
    logger,
    with_request_id,
    error_id,
    get_request_id,
)
from modules.decorators.trace_decorator import trace_entrypoint
from modules.observability.high_end_tracer import tracer
from modules.coordinators.chat_coordinator import ChatCoordinator
from modules.coordinators.chat_payload_coordinator import ChatPayloadCoordinator
from modules.coordinators.feedback_coordinator import FeedbackCoordinator
from modules.coordinators.health_coordinator import HealthCoordinator
from modules.coordinators.analytics_coordinator import AnalyticsCoordinator
from modules.coordinators.dashboard_coordinator import DashboardCoordinator

# Optional user comment coordinator.
# The existing file already uses user_comment_coordinator near the bottom,
# but the pasted version did not show the import/instance.
# This keeps the chatbot routes from failing at import time if that coordinator
# is not present in the current branch.
try:
    from modules.coordinators.user_comment_coordinator import UserCommentCoordinator
except Exception as import_error:
    UserCommentCoordinator = None
    logger.warning(
        "[ChatbotRoutes] UserCommentCoordinator unavailable: %s",
        import_error,
        exc_info=True,
    )


chatbot_bp = Blueprint("chatbot_bp", __name__)

chat_coordinator = ChatCoordinator()
chat_payload_coordinator = ChatPayloadCoordinator()
feedback_coordinator = FeedbackCoordinator()
health_coordinator = HealthCoordinator()
analytics_coordinator = AnalyticsCoordinator()
dashboard_coordinator = DashboardCoordinator()

user_comment_coordinator = (
    UserCommentCoordinator()
    if UserCommentCoordinator is not None
    else None
)



# ==========================================================
# PAYLOAD PAGINATION SETTINGS
# ==========================================================
# The images panel can explode if one matched position has thousands of
# related images. Keep the first /ask/payload response small, then let the
# frontend request more pages from /ask/payload/images.
IMAGE_PAYLOAD_INITIAL_PAGE_SIZE = 24
IMAGE_PAYLOAD_MAX_PAGE_SIZE = 100
IMAGE_PAYLOAD_DEFAULT_PAGE_SIZE = 24
IMAGE_PAYLOAD_ENDPOINT = "/chatbot/ask/payload/images"

# ==========================================================
# SHARED HELPERS
# ==========================================================

def _empty_blocks() -> Dict[str, list]:
    return {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }


def _empty_answer_response(
    *,
    status: str,
    answer: str,
    request_id: Optional[str],
    conversation_id: Optional[str] = None,
    payload_status: str = "unavailable",
    payload_endpoint: Optional[str] = None,
    document_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "document_scope": document_scope,
        "payload_status": payload_status,
        "payload_endpoint": payload_endpoint,
        "blocks": _empty_blocks(),
        "documents": [],
        "parts": [],
        "images": [],
        "drawings": [],
    }


def _empty_payload_response(
    *,
    status: str,
    payload_status: str,
    message: str,
    request_id: Optional[str],
    payload_route_request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "payload_status": payload_status,
        "message": message,
        "request_id": request_id,
        "payload_route_request_id": payload_route_request_id,
        "conversation_id": conversation_id,
        "blocks": _empty_blocks(),
        "documents": [],
        "parts": [],
        "images": [],
        "drawings": [],
        "relationship_summary": None,
    }




def _mixed_request_data() -> Dict[str, Any]:
    """
    Return request data that works for both:
        - GET query string endpoints
        - POST JSON endpoints

    Query-string values win only when JSON does not provide the key.
    """

    data: Dict[str, Any] = {}

    try:
        for key, value in request.args.items():
            data[key] = value
    except Exception:
        pass

    json_body = _json_data()

    if isinstance(json_body, dict):
        data.update(json_body)

    return data


def _extract_original_request_id(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract the original /ask request_id used to load saved payload data.
    """

    value = (
        data.get("requestId")
        or data.get("request_id")
        or data.get("originalRequestId")
        or data.get("original_request_id")
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    """
    Convert browser/query values into a bounded positive integer.
    """

    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = default

    if parsed < minimum:
        parsed = minimum

    if maximum is not None and parsed > maximum:
        parsed = maximum

    return parsed


def _extract_images_for_payload(result: Dict[str, Any]) -> list:
    """
    Read images from the normalized payload.

    Supports both current shapes:
        result["images"]
        result["blocks"]["images-container"]
    """

    if not isinstance(result, dict):
        return []

    images = result.get("images")

    if isinstance(images, list):
        return images

    blocks = result.get("blocks")

    if isinstance(blocks, dict):
        block_images = blocks.get("images-container")

        if isinstance(block_images, list):
            return block_images

    return []


def _set_images_for_payload(
    result: Dict[str, Any],
    images: list,
    *,
    total_images: int,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """
    Store a paged image list back into the payload in all shapes the frontend
    knows how to read.
    """

    if not isinstance(result, dict):
        return result

    safe_images = images if isinstance(images, list) else []
    safe_total = max(int(total_images or 0), len(safe_images))
    safe_page = max(int(page or 1), 1)
    safe_page_size = max(int(page_size or IMAGE_PAYLOAD_DEFAULT_PAGE_SIZE), 1)

    result["images"] = safe_images

    blocks = result.get("blocks")

    if not isinstance(blocks, dict):
        blocks = _empty_blocks()
        result["blocks"] = blocks

    blocks["images-container"] = safe_images

    returned_count = len(safe_images)
    has_more = (safe_page * safe_page_size) < safe_total

    result["images_total"] = safe_total
    result["image_count_total"] = safe_total
    result["total_images"] = safe_total

    result["images_returned"] = returned_count
    result["images_page"] = safe_page
    result["images_page_size"] = safe_page_size
    result["images_has_more"] = has_more
    result["images_truncated"] = has_more
    result["images_next_page"] = safe_page + 1 if has_more else None
    result["image_pagination_endpoint"] = IMAGE_PAYLOAD_ENDPOINT
    result["images_endpoint"] = IMAGE_PAYLOAD_ENDPOINT

    return result


def _page_images(
    images: list,
    *,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    """
    Return one page of images and pagination metadata.
    """

    safe_images = images if isinstance(images, list) else []
    safe_page = max(int(page or 1), 1)
    safe_page_size = max(int(page_size or IMAGE_PAYLOAD_DEFAULT_PAGE_SIZE), 1)

    total = len(safe_images)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size

    page_images = safe_images[start:end]
    has_more = end < total

    return {
        "images": page_images,
        "total": total,
        "page": safe_page,
        "page_size": safe_page_size,
        "returned": len(page_images),
        "has_more": has_more,
        "next_page": safe_page + 1 if has_more else None,
    }


def _cap_initial_payload_images(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the normal /ask/payload response small.

    Important:
        This is a route-level safety cap. It prevents a huge JSON response from
        being sent to the browser. The deeper service should still eventually
        stop loading thousands of images from the database in the first place.
    """

    if not isinstance(result, dict):
        return result

    all_images = _extract_images_for_payload(result)
    page = _page_images(
        all_images,
        page=1,
        page_size=IMAGE_PAYLOAD_INITIAL_PAGE_SIZE,
    )

    return _set_images_for_payload(
        result,
        page["images"],
        total_images=page["total"],
        page=page["page"],
        page_size=page["page_size"],
    )

def _resolve_route_request_id(request_id: Optional[str]) -> str:
    """
    Resolve the active request ID.

    Important:
        @with_request_id may create/bind a request ID for logging context,
        but it does not necessarily pass that value into the Flask route
        function argument.

    Therefore:
        route argument request_id may be None
        get_request_id() should still return the active logging request ID.
    """

    return request_id or get_request_id()


def _json_data() -> Dict[str, Any]:
    data = request.get_json(silent=True) or {}

    if not isinstance(data, dict):
        return {}

    return data


def _extract_user_id(data: Dict[str, Any]) -> str:
    """
    Normalize common user ID keys used by different frontend versions.
    """

    user_id = (
        data.get("userId")
        or data.get("user_id")
        or data.get("user")
        or "anonymous"
    )

    return str(user_id or "anonymous").strip() or "anonymous"


def _extract_client_type(data: Dict[str, Any]) -> str:
    """
    Normalize common client type keys used by different frontend versions.
    """

    client_type = (
        data.get("clientType")
        or data.get("client_type")
        or "web"
    )

    return str(client_type or "web").strip().lower() or "web"


def _extract_conversation_id(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract the active conversational-memory ID from the request body.

    Supported aliases:
        - conversation_id
        - conversationId
        - chatSessionId
        - chat_session_id
        - sessionId
        - session_id

    In the memory implementation we added, this maps to ChatSession.session_id.
    """

    conversation_id = (
        data.get("conversation_id")
        or data.get("conversationId")
        or data.get("chatSessionId")
        or data.get("chat_session_id")
        or data.get("sessionId")
        or data.get("session_id")
    )

    if conversation_id is None:
        return None

    conversation_id = str(conversation_id).strip()

    return conversation_id or None


def _extract_document_scope(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract and normalize document-scoped conversation mode from the request body.

    Expected frontend shape:
        {
            "enabled": true,
            "scope_type": "complete_document",
            "document_id": null,
            "complete_document_id": 30,
            "document_name": "Example Manual"
        }

    Supported aliases are accepted so older/newer frontend versions can coexist.

    Important:
        This route only validates and forwards the scope.
        Actual retrieval filtering happens later in UnifiedSearchService / RAG retrieval.
    """

    raw_scope = (
        data.get("document_scope")
        or data.get("documentScope")
        or data.get("active_document_scope")
        or data.get("activeDocumentScope")
    )

    if raw_scope is None:
        return None

    if not isinstance(raw_scope, dict):
        logger.warning(
            "[ChatAskRoute] Ignoring invalid document_scope. Expected dict, got %s.",
            type(raw_scope).__name__,
        )
        return None

    enabled = raw_scope.get("enabled", True)

    if enabled is False:
        return None

    scope_type = (
        raw_scope.get("scope_type")
        or raw_scope.get("scopeType")
        or "complete_document"
    )

    scope_type = str(scope_type or "").strip() or "complete_document"

    if scope_type != "complete_document":
        logger.warning(
            "[ChatAskRoute] Ignoring unsupported document_scope scope_type=%s raw_scope=%s",
            scope_type,
            raw_scope,
        )
        return None

    complete_document_id = (
        raw_scope.get("complete_document_id")
        or raw_scope.get("completed_document_id")
        or raw_scope.get("completeDocumentId")
        or raw_scope.get("completeDocumentID")
    )

    complete_document_id = _coerce_int_or_none(complete_document_id)

    if complete_document_id is None:
        logger.warning(
            "[ChatAskRoute] Ignoring document_scope because complete_document_id is missing. raw_scope=%s",
            raw_scope,
        )
        return None

    document_id = (
        raw_scope.get("document_id")
        or raw_scope.get("documentId")
    )

    document_id = _coerce_int_or_none(document_id)

    document_name = (
        raw_scope.get("document_name")
        or raw_scope.get("documentName")
        or raw_scope.get("name")
        or raw_scope.get("title")
        or f"Document #{complete_document_id}"
    )

    document_name = str(document_name or "").strip() or f"Document #{complete_document_id}"

    return {
        "enabled": True,
        "scope_type": "complete_document",
        "document_id": document_id,
        "complete_document_id": complete_document_id,
        "document_name": document_name,
    }


def _coerce_int_or_none(value: Any) -> Optional[int]:
    """
    Convert user/browser supplied IDs into int values safely.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        text = str(value).strip()

        if not text:
            return None

        return int(text)

    except (TypeError, ValueError):
        return None


def _process_question_with_optional_document_scope(
    *,
    user_id: str,
    question: str,
    client_type: str,
    request_id: str,
    conversation_id: Optional[str],
    document_scope: Optional[Dict[str, Any]],
    conversation_enabled: bool = True,
) -> Any:
    """
    Call ChatCoordinator.process_question with optional document_scope and
    optional conversation_enabled support.

    Behavior:
        - conversation_enabled=True:
            pass conversation_id normally.
        - conversation_enabled=False:
            force conversation_id=None and ask backend to treat this as
            single-turn / NEW_TOPIC mode.

    Safe rollout:
        - If ChatCoordinator supports document_scope, forward it.
        - If ChatCoordinator supports conversation_enabled, forward it.
        - If not, continue with the older supported arguments and log warnings.
    """

    normalized_conversation_enabled = bool(conversation_enabled)
    normalized_conversation_id = (
        conversation_id if normalized_conversation_enabled else None
    )

    base_kwargs: Dict[str, Any] = {
        "user_id": user_id,
        "question": question,
        "client_type": client_type,
        "request_id": request_id,
        "conversation_id": normalized_conversation_id,
    }

    try:
        signature = inspect.signature(chat_coordinator.process_question)
        parameters = signature.parameters

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if "conversation_enabled" in parameters or accepts_kwargs:
            base_kwargs["conversation_enabled"] = normalized_conversation_enabled

            logger.info(
                "[ChatAskRoute] Forwarding conversation_enabled to ChatCoordinator "
                "request_id=%s conversation_enabled=%s conversation_id=%s",
                request_id,
                normalized_conversation_enabled,
                normalized_conversation_id,
            )

        elif not normalized_conversation_enabled:
            logger.warning(
                "[ChatAskRoute] conversation_enabled=False received but "
                "ChatCoordinator.process_question does not accept conversation_enabled yet. "
                "Continuing with conversation_id=None only. request_id=%s",
                request_id,
            )

        if document_scope and ("document_scope" in parameters or accepts_kwargs):
            base_kwargs["document_scope"] = document_scope

            logger.info(
                "[ChatAskRoute] Forwarding document_scope to ChatCoordinator "
                "request_id=%s complete_document_id=%s document_name=%s",
                request_id,
                document_scope.get("complete_document_id"),
                document_scope.get("document_name"),
            )

        elif document_scope:
            logger.warning(
                "[ChatAskRoute] document_scope received but "
                "ChatCoordinator.process_question does not accept document_scope yet. "
                "Continuing without scoped backend retrieval. "
                "request_id=%s complete_document_id=%s document_name=%s",
                request_id,
                document_scope.get("complete_document_id"),
                document_scope.get("document_name"),
            )

        return chat_coordinator.process_question(**base_kwargs)

    except (TypeError, ValueError) as signature_error:
        logger.warning(
            "[ChatAskRoute] Could not inspect ChatCoordinator.process_question signature. "
            "Retrying with legacy-safe kwargs. request_id=%s error=%s",
            request_id,
            signature_error,
            exc_info=True,
        )

        legacy_kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "question": question,
            "client_type": client_type,
            "request_id": request_id,
            "conversation_id": normalized_conversation_id,
        }

        return chat_coordinator.process_question(**legacy_kwargs)


def _normalize_blocks_and_payload_lists(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure every response has the frontend payload keys.

    The answer-first route should normally return empty payload containers.
    The payload route fills them later.
    """

    if "blocks" not in result or not isinstance(result["blocks"], dict):
        result["blocks"] = _empty_blocks()

    for key, value in _empty_blocks().items():
        result["blocks"].setdefault(key, list(value))

        if result["blocks"][key] is None:
            result["blocks"][key] = []

    result.setdefault("documents", [])
    result.setdefault("parts", [])
    result.setdefault("images", [])
    result.setdefault("drawings", [])

    if result["documents"] is None:
        result["documents"] = []
    if result["parts"] is None:
        result["parts"] = []
    if result["images"] is None:
        result["images"] = []
    if result["drawings"] is None:
        result["drawings"] = []

    return result


def _normalize_answer_result(
    *,
    result: Any,
    effective_request_id: str,
    incoming_conversation_id: Optional[str],
    document_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize the /ask response.

    Critical IDs:
        request_id:
            Used by /chatbot/ask/payload to find the saved QandA/raw_response seed.

        conversation_id:
            Used by the frontend on the next /chatbot/ask call so conversational
            memory continues in the same ChatSession.

        document_scope:
            Used by document-scoped conversation mode.
            The frontend already stores it, but echoing it back helps debugging.
    """

    if not isinstance(result, dict):
        result = _empty_answer_response(
            status="error",
            answer="Invalid coordinator response.",
            request_id=effective_request_id,
            conversation_id=incoming_conversation_id,
            document_scope=document_scope,
            payload_status="unavailable",
            payload_endpoint=None,
        )

    result.setdefault("status", "success")
    result.setdefault("answer", "")

    if result.get("answer") is None:
        result["answer"] = ""

    # The frontend needs request_id for /chatbot/ask/payload.
    result["request_id"] = result.get("request_id") or effective_request_id

    # The frontend needs conversation_id for the next /chatbot/ask request.
    result["conversation_id"] = (
        result.get("conversation_id")
        or result.get("conversationId")
        or result.get("chatSessionId")
        or incoming_conversation_id
    )

    # Echo document_scope for debug visibility.
    # Later layers may also return their own normalized scope; prefer that if present.
    result["document_scope"] = (
        result.get("document_scope")
        or result.get("documentScope")
        or document_scope
    )

    result["document_scope_enabled"] = bool(
        isinstance(result.get("document_scope"), dict)
        and result["document_scope"].get("enabled") is True
    )

    result.setdefault("payload_status", "pending")
    result.setdefault("payload_endpoint", "/ask/payload")

    if result.get("status") in {"error", "invalid_input", "session_ended"}:
        result["payload_status"] = result.get("payload_status") or "unavailable"

        if result["payload_status"] == "unavailable":
            result["payload_endpoint"] = None

    # Keep payload endpoint consistent for the frontend.
    if result.get("payload_status") == "pending":
        result["payload_endpoint"] = (
            result.get("payload_endpoint")
            or "/ask/payload"
        )

    result = _normalize_blocks_and_payload_lists(result)

    return result


def _normalize_payload_result(
    *,
    result: Any,
    original_request_id: Optional[str],
    payload_route_request_id: str,
    conversation_id: Optional[str],
) -> Dict[str, Any]:
    """
    Normalize the /ask/payload response.

    Payload lookup remains request_id-based. conversation_id is included only
    for debugging/frontend continuity; it is not used to load payload.
    """

    if not isinstance(result, dict):
        result = _empty_payload_response(
            status="error",
            payload_status="error",
            message="Invalid payload coordinator response.",
            request_id=original_request_id,
            payload_route_request_id=payload_route_request_id,
            conversation_id=conversation_id,
        )

    result["request_id"] = original_request_id
    result["payload_route_request_id"] = payload_route_request_id
    result["conversation_id"] = result.get("conversation_id") or conversation_id

    result = _normalize_blocks_and_payload_lists(result)
    result.setdefault("relationship_summary", None)

    return result


def _user_comment_service_unavailable_response(
    *,
    request_id: str,
    message: str,
) -> tuple:
    return jsonify({
        "status": "error",
        "message": message,
        "request_id": request_id,
    }), 501


# ==========================================================
# ASK
# ==========================================================
# Answer-first route.
#
# This route returns the chatbot text answer quickly.
# It does NOT wait for documents/images/parts/drawings payload projection.
#
# Frontend flow:
#   1. POST /chatbot/ask
#   2. Render result["answer"]
#   3. Store result["conversation_id"] in frontend memory
#   4. Send that same conversation_id on the next /chatbot/ask call
#   5. If result["payload_status"] == "pending", call /chatbot/ask/payload
#      using result["request_id"]
#
# Document mode flow:
#   1. User clicks "Ask this document" from the document payload panel.
#   2. Frontend stores document_scope in sessionStorage.
#   3. Frontend sends document_scope with /chatbot/ask.
#   4. Route validates/normalizes document_scope.
#   5. Route forwards document_scope when ChatCoordinator supports it.
# ==========================================================

@chatbot_bp.route("/ask", methods=["POST"])
@with_request_id
@trace_entrypoint(
    name="chat.ask",
    enabled_env="EMTAC_TRACE_CHAT_ENABLED",
    deep_profile=False,
    capture_args=False,
    capture_return=False,
)
def ask(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        data = _json_data()

        conversation_mode = str(
            data.get("conversation_mode")
            or data.get("conversationMode")
            or ""
        ).strip().lower()

        conversation_enabled = _coerce_bool(
            data.get("conversation_enabled", data.get("conversationEnabled")),
            default=True,
        )

        if conversation_mode in {
            "single_turn",
            "single-turn",
            "new_topic",
            "new-topic",
            "off",
            "disabled",
        }:
            conversation_enabled = False

        incoming_conversation_id = _extract_conversation_id(data)

        # Critical:
        # When Conversation is OFF, do not allow the old session id through.
        if not conversation_enabled:
            incoming_conversation_id = None

        document_scope = _extract_document_scope(data)

        logger.info(
            "[ChatAskRoute] Incoming ask request request_id=%s "
            "conversation_enabled=%s conversation_mode=%s conversation_id=%s "
            "has_document_scope=%s complete_document_id=%s document_name=%s",
            effective_request_id,
            conversation_enabled,
            conversation_mode or ("conversation" if conversation_enabled else "single_turn"),
            incoming_conversation_id,
            bool(document_scope),
            document_scope.get("complete_document_id") if document_scope else None,
            document_scope.get("document_name") if document_scope else None,
        )

        result = _process_question_with_optional_document_scope(
            user_id=_extract_user_id(data),
            question=data.get("question", ""),
            client_type=_extract_client_type(data),
            request_id=effective_request_id,
            conversation_id=incoming_conversation_id,
            document_scope=document_scope,
            conversation_enabled=conversation_enabled,
        )

        result = _normalize_answer_result(
            result=result,
            effective_request_id=effective_request_id,
            incoming_conversation_id=incoming_conversation_id,
            document_scope=document_scope,
        )

        # Keep this visible in the frontend/debug response.
        result["conversation_enabled"] = bool(conversation_enabled)
        result["conversation_mode"] = (
            "conversation" if conversation_enabled else "single_turn"
        )

        logger.info(
            "[ChatAskRoute] Answer route complete request_id=%s "
            "conversation_enabled=%s conversation_mode=%s conversation_id=%s "
            "status=%s payload_status=%s document_scope_enabled=%s complete_document_id=%s",
            result.get("request_id"),
            result.get("conversation_enabled"),
            result.get("conversation_mode"),
            result.get("conversation_id"),
            result.get("status"),
            result.get("payload_status"),
            result.get("document_scope_enabled"),
            (
                result.get("document_scope", {}).get("complete_document_id")
                if isinstance(result.get("document_scope"), dict)
                else None
            ),
        )

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/ask route failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify(_empty_answer_response(
            status="error",
            answer="An unexpected error occurred.",
            request_id=effective_request_id,
            conversation_id=None,
            document_scope=None,
            payload_status="unavailable",
            payload_endpoint=None,
        )), 500


# ==========================================================
# ASK PAYLOAD
# ==========================================================
# Supporting payload route.
#
# This route loads the heavier UI payload:
#   - documents
#   - parts
#   - images
#   - drawings
#
# Frontend should call this AFTER /ask returns.
#
# Expected request body:
# {
#   "requestId": "<request_id returned from /ask>",
#   "conversation_id": "<conversation_id returned from /ask>",
#   "clientType": "web"
# }
#
# Optional fallback:
# {
#   "requestId": "<request_id>",
#   "payload_seed": {...},
#   "clientType": "web"
# }
# ==========================================================

@chatbot_bp.route("/ask/payload", methods=["POST"])
@with_request_id
@trace_entrypoint(
    name="chat.payload",
    enabled_env="EMTAC_TRACE_PAYLOAD_ENABLED",
    deep_profile=False,
    capture_args=False,
    capture_return=False,
)
def ask_payload(request_id=None):
    """
    Supporting payload route.

    Important:
        payload_route_request_id belongs to THIS /ask/payload request.
        original_request_id belongs to the first /ask request and is used
        to find the saved QandA raw_response seed.

    conversation_id is accepted for debugging/frontend continuity, but payload
    loading remains request_id-based.
    """

    payload_route_request_id = _resolve_route_request_id(request_id)
    original_request_id = None
    conversation_id = None
    route_start = time.perf_counter()

    try:
        data = _json_data()

        original_request_id = _extract_original_request_id(data)

        conversation_id = _extract_conversation_id(data)
        payload_seed = data.get("payload_seed")

        if not original_request_id and not payload_seed:
            response_body = _empty_payload_response(
                status="invalid_input",
                payload_status="unavailable",
                message="Missing requestId or payload_seed.",
                request_id=None,
                payload_route_request_id=payload_route_request_id,
                conversation_id=conversation_id,
            )

            logger.info(
                "[ChatPayloadRoute] Invalid payload request "
                "payload_route_request_id=%s original_request_id=%s conversation_id=%s elapsed=%.3fs",
                payload_route_request_id,
                original_request_id,
                conversation_id,
                time.perf_counter() - route_start,
            )

            return jsonify(response_body), 400

        logger.info(
            "[ChatPayloadRoute] Starting payload load "
            "payload_route_request_id=%s original_request_id=%s conversation_id=%s "
            "has_payload_seed=%s",
            payload_route_request_id,
            original_request_id,
            conversation_id,
            payload_seed is not None,
        )

        load_start = time.perf_counter()

        result = chat_payload_coordinator.load_payload(
            request_id=original_request_id,
            payload_seed=payload_seed,
            client_type=_extract_client_type(data),
        )

        load_time = time.perf_counter() - load_start

        logger.info(
            "[ChatPayloadRoute] Coordinator payload load complete "
            "payload_route_request_id=%s original_request_id=%s conversation_id=%s load_time=%.3fs",
            payload_route_request_id,
            original_request_id,
            conversation_id,
            load_time,
        )

        result = _normalize_payload_result(
            result=result,
            original_request_id=original_request_id,
            payload_route_request_id=payload_route_request_id,
            conversation_id=conversation_id,
        )

        result = _cap_initial_payload_images(result)

        documents_count = len(result.get("documents") or [])
        parts_count = len(result.get("parts") or [])
        images_count = len(result.get("images") or [])
        drawings_count = len(result.get("drawings") or [])

        # ------------------------------------------------------
        # Serialization probe
        # ------------------------------------------------------
        # This tells us whether the delay is caused by converting the
        # Python payload into JSON before Flask sends it to the browser.
        serialize_probe_start = time.perf_counter()

        try:
            payload_size_bytes = len(
                json.dumps(result, default=str).encode("utf-8")
            )
        except Exception as exc:
            payload_size_bytes = -1
            logger.exception(
                "[ChatPayloadRoute] Failed payload serialization probe "
                "payload_route_request_id=%s original_request_id=%s conversation_id=%s error=%s",
                payload_route_request_id,
                original_request_id,
                conversation_id,
                exc,
            )

        serialize_probe_time = time.perf_counter() - serialize_probe_start

        logger.info(
            "[ChatPayloadRoute] Payload ready for jsonify "
            "payload_route_request_id=%s original_request_id=%s conversation_id=%s "
            "status=%s payload_status=%s documents=%s images=%s parts=%s drawings=%s "
            "payload_size_bytes=%s serialize_probe_time=%.3fs route_elapsed_before_jsonify=%.3fs",
            payload_route_request_id,
            original_request_id,
            conversation_id,
            result.get("status"),
            result.get("payload_status"),
            documents_count,
            images_count,
            parts_count,
            drawings_count,
            payload_size_bytes,
            serialize_probe_time,
            time.perf_counter() - route_start,
        )

        # ------------------------------------------------------
        # Flask jsonify timing
        # ------------------------------------------------------
        jsonify_start = time.perf_counter()
        flask_response = jsonify(result)
        jsonify_time = time.perf_counter() - jsonify_start

        logger.info(
            "[ChatPayloadRoute] jsonify complete "
            "payload_route_request_id=%s original_request_id=%s conversation_id=%s "
            "jsonify_time=%.3fs payload_size_bytes=%s total_route_time_before_return=%.3fs",
            payload_route_request_id,
            original_request_id,
            conversation_id,
            jsonify_time,
            payload_size_bytes,
            time.perf_counter() - route_start,
        )

        return flask_response

    except Exception as e:
        error_id(
            f"/ask/payload route failure: {e}",
            payload_route_request_id,
            exc_info=True,
        )

        response_body = _empty_payload_response(
            status="error",
            payload_status="error",
            message="An unexpected error occurred while loading supporting payload.",
            request_id=original_request_id,
            payload_route_request_id=payload_route_request_id,
            conversation_id=conversation_id,
        )

        logger.info(
            "[ChatPayloadRoute] Returning error payload "
            "payload_route_request_id=%s original_request_id=%s conversation_id=%s elapsed=%.3fs",
            payload_route_request_id,
            original_request_id,
            conversation_id,
            time.perf_counter() - route_start,
        )

        return jsonify(response_body), 500


# ==========================================================
# ASK PAYLOAD IMAGES
# ==========================================================
# Paginated image-only endpoint for the Images panel.
#
# Frontend flow:
#   1. POST /chatbot/ask
#   2. POST /chatbot/ask/payload
#      - returns first 24 images only
#      - returns images_total/images_has_more/images_next_page
#   3. GET /chatbot/ask/payload/images?request_id=<original_request_id>&page=2&page_size=24
#
# This is intentionally route-level pagination so it can be dropped in without
# requiring an immediate rewrite of ChunkRelationshipService.
#
# Long-term improvement:
#   Move the limit/offset down into ChunkRelationshipService so the DB never
#   returns thousands of images for the first payload.
# ==========================================================

@chatbot_bp.route("/ask/payload/images", methods=["GET", "POST"])
@with_request_id
@trace_entrypoint(
    name="chat.payload.images",
    enabled_env="EMTAC_TRACE_PAYLOAD_ENABLED",
    deep_profile=False,
    capture_args=False,
    capture_return=False,
)
def ask_payload_images(request_id=None):
    image_route_request_id = _resolve_route_request_id(request_id)
    route_start = time.perf_counter()

    original_request_id = None
    conversation_id = None

    try:
        data = _mixed_request_data()

        original_request_id = _extract_original_request_id(data)
        conversation_id = _extract_conversation_id(data)

        page = _coerce_positive_int(
            data.get("page"),
            default=1,
            minimum=1,
        )

        page_size = _coerce_positive_int(
            data.get("page_size") or data.get("pageSize"),
            default=IMAGE_PAYLOAD_DEFAULT_PAGE_SIZE,
            minimum=1,
            maximum=IMAGE_PAYLOAD_MAX_PAGE_SIZE,
        )

        payload_seed = data.get("payload_seed")

        if not original_request_id and not payload_seed:
            logger.info(
                "[ChatPayloadImagesRoute] Invalid image payload request "
                "image_route_request_id=%s original_request_id=%s conversation_id=%s elapsed=%.3fs",
                image_route_request_id,
                original_request_id,
                conversation_id,
                time.perf_counter() - route_start,
            )

            return jsonify({
                "status": "invalid_input",
                "payload_status": "unavailable",
                "message": "Missing request_id/requestId or payload_seed.",
                "request_id": original_request_id,
                "image_route_request_id": image_route_request_id,
                "conversation_id": conversation_id,
                "images": [],
                "images_total": 0,
                "images_returned": 0,
                "images_page": page,
                "images_page_size": page_size,
                "images_has_more": False,
                "images_next_page": None,
            }), 400

        logger.info(
            "[ChatPayloadImagesRoute] Starting image page load "
            "image_route_request_id=%s original_request_id=%s conversation_id=%s "
            "page=%s page_size=%s has_payload_seed=%s",
            image_route_request_id,
            original_request_id,
            conversation_id,
            page,
            page_size,
            payload_seed is not None,
        )

        load_start = time.perf_counter()

        result = chat_payload_coordinator.load_payload(
            request_id=original_request_id,
            payload_seed=payload_seed,
            client_type=_extract_client_type(data),
        )

        load_time = time.perf_counter() - load_start

        result = _normalize_payload_result(
            result=result,
            original_request_id=original_request_id,
            payload_route_request_id=image_route_request_id,
            conversation_id=conversation_id,
        )

        all_images = _extract_images_for_payload(result)

        page_result = _page_images(
            all_images,
            page=page,
            page_size=page_size,
        )

        response_body = {
            "status": "success",
            "payload_status": "complete",
            "request_id": original_request_id,
            "image_route_request_id": image_route_request_id,
            "conversation_id": conversation_id,

            "images": page_result["images"],
            "images_total": page_result["total"],
            "image_count_total": page_result["total"],
            "total_images": page_result["total"],
            "images_returned": page_result["returned"],
            "images_page": page_result["page"],
            "images_page_size": page_result["page_size"],
            "images_has_more": page_result["has_more"],
            "images_truncated": page_result["has_more"],
            "images_next_page": page_result["next_page"],
            "image_pagination_endpoint": IMAGE_PAYLOAD_ENDPOINT,
            "images_endpoint": IMAGE_PAYLOAD_ENDPOINT,
        }

        logger.info(
            "[ChatPayloadImagesRoute] Image page ready "
            "image_route_request_id=%s original_request_id=%s conversation_id=%s "
            "page=%s page_size=%s returned=%s total=%s has_more=%s "
            "load_time=%.3fs elapsed=%.3fs",
            image_route_request_id,
            original_request_id,
            conversation_id,
            page_result["page"],
            page_result["page_size"],
            page_result["returned"],
            page_result["total"],
            page_result["has_more"],
            load_time,
            time.perf_counter() - route_start,
        )

        return jsonify(response_body)

    except Exception as e:
        error_id(
            f"/ask/payload/images route failure: {e}",
            image_route_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "payload_status": "error",
            "message": "An unexpected error occurred while loading image page.",
            "request_id": original_request_id,
            "image_route_request_id": image_route_request_id,
            "conversation_id": conversation_id,
            "images": [],
            "images_total": 0,
            "images_returned": 0,
            "images_page": None,
            "images_page_size": None,
            "images_has_more": False,
            "images_next_page": None,
        }), 500


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True

    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False

    return default


# ==========================================================
# UPDATE Q&A
# ==========================================================

@chatbot_bp.route("/update_qanda", methods=["POST"])
@with_request_id
def update_qanda(request_id=None):
    route_request_id = _resolve_route_request_id(request_id)

    try:
        data = _json_data()

        # This is the request_id from the original /chatbot/ask call.
        # That is the one attached to the qanda row.
        original_answer_request_id = (
            data.get("request_id")
            or data.get("requestId")
            or data.get("original_request_id")
            or data.get("originalRequestId")
        )

        conversation_id = _extract_conversation_id(data)

        logger.info(
            "[QandAFeedbackRoute] Feedback submit route_request_id=%s "
            "original_answer_request_id=%s conversation_id=%s "
            "has_question=%s has_answer=%s has_rating=%s has_comment=%s",
            route_request_id,
            original_answer_request_id,
            conversation_id,
            bool(data.get("question")),
            bool(data.get("answer")),
            data.get("rating") is not None,
            bool(data.get("comment")),
        )

        result = feedback_coordinator.process_feedback(
            user_id=(
                data.get("user_id")
                or data.get("userId")
                or "anonymous"
            ),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            rating=data.get("rating"),
            comment=data.get("comment"),

            # Important:
            # pass the original /ask request_id, not this feedback route request_id.
            request_id=original_answer_request_id,
        )

        if result.get("status") == "success":
            return jsonify({
                "status": "success",
                "message": result.get("message", "Feedback saved."),
                "request_id": original_answer_request_id,
                "conversation_id": conversation_id,
                "feedback_route_request_id": route_request_id,
            })

        return jsonify({
            "status": "error",
            "message": result.get("message", "Unable to save feedback."),
            "request_id": original_answer_request_id,
            "conversation_id": conversation_id,
            "feedback_route_request_id": route_request_id,
        }), 400

    except Exception as e:
        error_id(
            f"/update_qanda failure: {e}",
            route_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
            "request_id": None,
            "conversation_id": None,
            "feedback_route_request_id": route_request_id,
        }), 500


# ==========================================================
# HEALTH
# ==========================================================

@chatbot_bp.route("/health", methods=["GET"])
@with_request_id
def health_check(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        result = health_coordinator.check_health(
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/health failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# METRICS
# ==========================================================

@chatbot_bp.route("/metrics", methods=["GET"])
@with_request_id
def metrics(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        result = analytics_coordinator.get_metrics(
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/metrics failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# PERFORMANCE RECOMMENDATIONS
# ==========================================================

@chatbot_bp.route("/performance/recommendations", methods=["GET"])
@with_request_id
def performance_recommendations(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        hours = request.args.get("hours", 24, type=int)

        result = analytics_coordinator.get_recommendations(
            hours=hours,
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/performance/recommendations failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# PERFORMANCE DASHBOARD
# ==========================================================

@chatbot_bp.route("/performance/dashboard", methods=["GET"])
@with_request_id
def performance_dashboard(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        hours = request.args.get("hours", 24, type=int)

        result = dashboard_coordinator.get_dashboard(
            hours=hours,
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/performance/dashboard failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# USER COMMENTS
# ==========================================================

@chatbot_bp.route("/user-comments", methods=["POST"])
@with_request_id
def submit_user_comment(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    if user_comment_coordinator is None:
        return _user_comment_service_unavailable_response(
            request_id=effective_request_id,
            message="User comment service is not available in this build.",
        )

    try:
        data = _json_data()

        page_url = (
            data.get("page_url")
            or data.get("pageUrl")
            or request.headers.get("Referer")
            or ""
        )

        result = user_comment_coordinator.submit_comment(
            user_id=(
                data.get("user_id")
                or data.get("userId")
                or data.get("user")
            ),
            comment=data.get("comment", ""),
            page_url=page_url,
            screenshot_path=(
                data.get("screenshot_path")
                or data.get("screenshotPath")
            ),
            request_id=effective_request_id,
        )

        if result.get("status") == "success":
            return jsonify(result), 201

        if result.get("status") == "invalid_input":
            return jsonify(result), 400

        return jsonify(result), 500

    except Exception as e:
        error_id(
            f"/user-comments POST failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred while saving the comment.",
            "comment": None,
            "request_id": effective_request_id,
        }), 500


@chatbot_bp.route("/user-comments", methods=["GET"])
@with_request_id
def list_user_comments(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    if user_comment_coordinator is None:
        return _user_comment_service_unavailable_response(
            request_id=effective_request_id,
            message="User comment service is not available in this build.",
        )

    try:
        result = user_comment_coordinator.list_comments(
            user_id=request.args.get("user_id") or request.args.get("userId"),
            page_url=request.args.get("page_url") or request.args.get("pageUrl"),
            limit=request.args.get("limit", 50, type=int),
            offset=request.args.get("offset", 0, type=int),
            request_id=effective_request_id,
        )

        if result.get("status") == "success":
            return jsonify(result)

        if result.get("status") == "invalid_input":
            return jsonify(result), 400

        return jsonify(result), 500

    except Exception as e:
        error_id(
            f"/user-comments GET failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred while loading comments.",
            "comments": [],
            "count": 0,
            "request_id": effective_request_id,
        }), 500


@chatbot_bp.route("/user-comments/<int:comment_id>", methods=["GET"])
@with_request_id
def get_user_comment(comment_id, request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    if user_comment_coordinator is None:
        return _user_comment_service_unavailable_response(
            request_id=effective_request_id,
            message="User comment service is not available in this build.",
        )

    try:
        result = user_comment_coordinator.get_comment(
            comment_id=comment_id,
            request_id=effective_request_id,
        )

        if result.get("status") == "success":
            return jsonify(result)

        if result.get("status") == "not_found":
            return jsonify(result), 404

        if result.get("status") == "invalid_input":
            return jsonify(result), 400

        return jsonify(result), 500

    except Exception as e:
        error_id(
            f"/user-comments/<comment_id> GET failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred while loading the comment.",
            "comment": None,
            "request_id": effective_request_id,
        }), 500