from __future__ import annotations

from app.api.ai_chat_route import ai_bp, init_ai_blueprint
from app.api.dashboard_routes import dashboard_bp

__all__ = [
    "ai_bp",
    "init_ai_blueprint",
    "dashboard_bp",
]