from __future__ import annotations

from flask import Blueprint, jsonify

from modules.theme_utils.theme_utils  import get_available_themes


theme_utils_bp = Blueprint(
    "theme_utils_bp",
    __name__,
    url_prefix="",
)


@theme_utils_bp.route("/api/themes", methods=["GET"])
def api_get_themes():
    """
    Return the available theme CSS files as JSON.
    """
    return jsonify(get_available_themes())


@theme_utils_bp.app_context_processor
def inject_theme_options():
    """
    Make available_themes accessible to templates.
    """
    return {
        "available_themes": get_available_themes()
    }