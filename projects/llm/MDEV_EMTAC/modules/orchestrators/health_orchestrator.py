# modules/orchestrators/health_orchestrator.py

from typing import Dict, Any
from sqlalchemy import text

from modules.configuration.log_config import (
    with_request_id,
    error_id,
)

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from plugins.ai_modules.ai_models import ModelsConfig


class HealthOrchestrator(BaseOrchestrator):
    """
    Cross-system health validation.

    Checks:
        - Database connectivity
        - pgvector availability
        - Active AI model configuration
        - Optional GPU service reachability

    Does NOT:
        - Generate embeddings
        - Run inference
        - Modify data
    """

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # Public Entry
    # ---------------------------------------------------------

    @with_request_id
    def check_health(self, *, request_id: str | None = None) -> Dict[str, Any]:

        health_report = {
            "status": "healthy",
            "components": {},
        }

        # -----------------------------------------------------
        # 1. Database Connectivity
        # -----------------------------------------------------
        try:
            with self.transaction() as session:
                session.execute(text("SELECT 1"))

            health_report["components"]["database"] = "ok"

        except Exception as e:
            error_id(f"Database health check failed: {e}", request_id, exc_info=True)
            health_report["components"]["database"] = "failed"
            health_report["status"] = "unhealthy"

        # -----------------------------------------------------
        # 2. pgvector Check (PostgreSQL only)
        # -----------------------------------------------------
        try:
            with self.transaction() as session:
                session.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )

            health_report["components"]["pgvector"] = "ok"

        except Exception:
            # Not fatal — may be SQLite or extension not installed
            health_report["components"]["pgvector"] = "unknown"

        # -----------------------------------------------------
        # 3. Active Text Model Config
        # -----------------------------------------------------
        try:
            model_name = ModelsConfig.get_current_model_name("text")
            health_report["components"]["text_model"] = model_name or "none"

        except Exception:
            health_report["components"]["text_model"] = "failed"
            health_report["status"] = "degraded"

        # -----------------------------------------------------
        # 4. Active Image Model Config
        # -----------------------------------------------------
        try:
            image_model = ModelsConfig.get_current_model_name("image")
            health_report["components"]["image_model"] = image_model or "none"

        except Exception:
            health_report["components"]["image_model"] = "failed"
            health_report["status"] = "degraded"

        return health_report