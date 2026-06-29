"""
blueprints/upload_search_db/search_drawing.py

Thin Flask routes for drawing search.

Business logic lives in:
    modules/coordinators/drawing_search_coordinator.py
    modules/orchestrators/drawing_search_orchestrator.py
    modules/services/drawing_search_service.py

Route design:
    /drawings/search
        Search drawings.

    /drawings/types
        Get available drawing types.

    /drawings/search/by-type/<drawing_type>
        Search drawings by type.

    /drawings/view/<drawing_id>
        Legacy route. Redirects to the distinct print viewer.

    /drawings/print-viewer/<drawing_id>
        EMTAC drawing/print viewer page.

    /drawings/file-status/<drawing_id>
        JSON status used by frontend to enable/disable View/Open/Print buttons.

    /drawings/file/<drawing_id>
        Serves the raw file inline. Does not force download.
"""

from __future__ import annotations

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from modules.configuration.log_config import get_request_id, debug_id
from modules.coordinators.drawing_search_coordinator import DrawingSearchCoordinator


drawing_routes = Blueprint("drawing_routes", __name__)


PREVIEWABLE_EXTENSIONS = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "svg",
}


def _get_extension(download_name: str | None) -> str:
    """
    Safely extract a lowercase file extension from a file name.
    """
    if not download_name:
        return ""

    if "." not in download_name:
        return ""

    return download_name.rsplit(".", 1)[-1].lower().strip()


def _can_browser_preview(mime_type: str | None, download_name: str | None) -> bool:
    """
    Decide whether the browser can reasonably preview this file.

    Browsers can usually preview:
        - PDFs
        - common image formats

    Browsers usually cannot preview:
        - SolidWorks drawings such as .SLDDRW
        - CAD files
        - many vendor-specific print/drawing files
    """
    safe_mime_type = mime_type or "application/octet-stream"
    extension = _get_extension(download_name)

    return (
        safe_mime_type == "application/pdf"
        or safe_mime_type.startswith("image/")
        or extension in PREVIEWABLE_EXTENSIONS
    )


def _safe_payload_dict(payload):
    """
    Make sure route logic can safely call .get() on the coordinator payload.
    """
    if isinstance(payload, dict):
        return payload

    return {
        "success": False,
        "message": "Unexpected drawing response format.",
        "raw_payload": str(payload),
    }


@drawing_routes.route("/drawings/search", methods=["GET"])
def search_drawings():
    """
    Search endpoint for drawings.
    """
    request_id = get_request_id()
    debug_id("[search_drawing route] Starting /drawings/search", request_id)

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.search_from_request_args(
        request_args=request.args,
        request_id=request_id,
    )

    return jsonify(payload), status_code


@drawing_routes.route("/drawings/types", methods=["GET"])
def get_drawing_types():
    """
    Get all available drawing types.
    """
    request_id = get_request_id()
    debug_id("[search_drawing route] Starting /drawings/types", request_id)

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.get_drawing_types(
        request_id=request_id,
    )

    return jsonify(payload), status_code


@drawing_routes.route("/drawings/search/by-type/<drawing_type>", methods=["GET"])
def search_drawings_by_type(drawing_type: str):
    """
    Search drawings by drawing type.
    """
    request_id = get_request_id()
    debug_id(
        f"[search_drawing route] Starting /drawings/search/by-type/{drawing_type}",
        request_id,
    )

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.search_by_type(
        drawing_type=drawing_type,
        request_args=request.args,
        request_id=request_id,
    )

    return jsonify(payload), status_code


@drawing_routes.route("/drawings/view/<int:drawing_id>", methods=["GET"])
def view_drawing_legacy(drawing_id: int):
    """
    Backward-compatible route.

    Old search links may still point to:
        /drawings/view/<id>

    Redirect them to the distinct print viewer:
        /drawings/print-viewer/<id>
    """
    request_id = get_request_id()
    debug_id(
        f"[search_drawing route] Redirecting legacy /drawings/view/{drawing_id}",
        request_id,
    )

    return redirect(
        url_for("drawing_routes.print_viewer", drawing_id=drawing_id)
    )


@drawing_routes.route("/drawings/print-viewer/<int:drawing_id>", methods=["GET"])
def print_viewer(drawing_id: int):
    """
    Distinct EMTAC drawing/print viewer page.

    This page shows:
        - drawing metadata
        - print option
        - preview when browser-supported
        - disabled/greyed-out open/print controls when no file exists

    Raw file serving is handled by:
        /drawings/file/<drawing_id>
    """
    request_id = get_request_id()
    debug_id(
        f"[search_drawing route] Starting /drawings/print-viewer/{drawing_id}",
        request_id,
    )

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.get_drawing_file(
        drawing_id=drawing_id,
        request_id=request_id,
    )

    payload = _safe_payload_dict(payload)

    if status_code != 200:
        return render_template(
            "upload_search_database/drawing_print_viewer.html",
            drawing_id=drawing_id,
            payload=payload,
            drawing=payload.get("drawing"),
            file_url=None,
            can_preview=False,
            can_open_file=False,
            error=payload,
        ), status_code

    mime_type = payload.get("mime_type") or "application/octet-stream"
    download_name = payload.get("download_name") or ""

    can_preview = _can_browser_preview(
        mime_type=mime_type,
        download_name=download_name,
    )

    file_url = url_for(
        "drawing_routes.serve_drawing_file",
        drawing_id=drawing_id,
    )

    return render_template(
        "upload_search_database/drawing_print_viewer.html",
        drawing_id=drawing_id,
        payload=payload,
        drawing=payload.get("drawing"),
        file_url=file_url,
        can_preview=can_preview,
        can_open_file=True,
        error=None,
    )


@drawing_routes.route("/drawings/file-status/<int:drawing_id>", methods=["GET"])
def drawing_file_status(drawing_id: int):
    """
    JSON helper route for the frontend.

    Use this route when rendering search results so the UI can:
        - enable View/Open/Print when a real file exists
        - grey out View/Open/Print when no file exists
        - avoid showing download options
    """
    request_id = get_request_id()
    debug_id(
        f"[search_drawing route] Starting /drawings/file-status/{drawing_id}",
        request_id,
    )

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.get_drawing_file(
        drawing_id=drawing_id,
        request_id=request_id,
    )

    payload = _safe_payload_dict(payload)

    if status_code != 200:
        return jsonify(
            {
                "success": False,
                "drawing_id": drawing_id,
                "can_open_file": False,
                "can_preview": False,
                "file_url": None,
                "viewer_url": url_for(
                    "drawing_routes.print_viewer",
                    drawing_id=drawing_id,
                ),
                "error": payload,
            }
        ), status_code

    mime_type = payload.get("mime_type") or "application/octet-stream"
    download_name = payload.get("download_name") or ""

    can_preview = _can_browser_preview(
        mime_type=mime_type,
        download_name=download_name,
    )

    return jsonify(
        {
            "success": True,
            "drawing_id": drawing_id,
            "drawing": payload.get("drawing"),
            "mime_type": mime_type,
            "download_name": download_name,
            "can_open_file": True,
            "can_preview": can_preview,
            "file_url": url_for(
                "drawing_routes.serve_drawing_file",
                drawing_id=drawing_id,
            ),
            "viewer_url": url_for(
                "drawing_routes.print_viewer",
                drawing_id=drawing_id,
            ),
        }
    ), 200


@drawing_routes.route("/drawings/file/<int:drawing_id>", methods=["GET"])
def serve_drawing_file(drawing_id: int):
    """
    Serve the raw drawing/print file.

    This route does not force download.

    Important:
        For files like .SLDDRW, the browser may still hand the file off to
        another app because browsers cannot preview SolidWorks drawings natively.
    """
    request_id = get_request_id()
    debug_id(
        f"[search_drawing route] Starting /drawings/file/{drawing_id}",
        request_id,
    )

    coordinator = DrawingSearchCoordinator()

    payload, status_code = coordinator.get_drawing_file(
        drawing_id=drawing_id,
        request_id=request_id,
    )

    payload = _safe_payload_dict(payload)

    if status_code != 200:
        return jsonify(payload), status_code

    physical_file_path = payload.get("physical_file_path")

    if not physical_file_path:
        return jsonify(
            {
                "success": False,
                "drawing_id": drawing_id,
                "message": "Drawing file path was not found.",
            }
        ), 404

    return send_file(
        physical_file_path,
        mimetype=payload.get("mime_type") or "application/octet-stream",
        as_attachment=False,
        download_name=payload.get("download_name"),
        conditional=True,
    )