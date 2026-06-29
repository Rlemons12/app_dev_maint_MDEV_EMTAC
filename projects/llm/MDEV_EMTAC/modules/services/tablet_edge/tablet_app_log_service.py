"""
Tablet app log service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_app_log_service.py

Responsibilities:
    - Validate tablet app log payloads.
    - Normalize log level, source, message, context, and timestamps.
    - Batch insert tablet app logs.
    - Query recent tablet app logs.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import TabletAppLog


class TabletAppLogService:
    """
    Domain service for tablet_edge.tablet_app_log.
    """

    VALID_LOG_LEVELS = {
        "DEBUG",
        "INFO",
        "WARN",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    DEFAULT_MAX_BATCH_SIZE = 500

    @staticmethod
    def normalize_string(
        value: Any,
        *,
        max_length: int | None = None,
        allow_empty: bool = False,
    ) -> str | None:
        """
        Normalize optional string values.

        Empty strings become None unless allow_empty=True.
        """
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized and not allow_empty:
            return None

        if max_length is not None and len(normalized) > max_length:
            normalized = normalized[:max_length]

        return normalized

    @staticmethod
    def normalize_datetime(value: Any) -> datetime | None:
        """
        Normalize optional datetime values.

        Accepts:
            - datetime objects
            - ISO strings
            - ISO strings ending in Z

        If a datetime is naive, UTC is assumed.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        normalized = str(value).strip()

        if not normalized:
            return None

        try:
            normalized = normalized.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except Exception as exc:
            raise ValueError(f"Invalid datetime value: {value!r}") from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    def normalize_log_level(self, value: Any) -> str:
        """
        Normalize log_level.

        Unknown log levels are allowed but converted to INFO and logged.
        """
        log_level = self.normalize_string(value, max_length=50)

        if not log_level:
            return "INFO"

        log_level = log_level.upper()

        if log_level == "WARNING":
            log_level = "WARN"

        if log_level not in self.VALID_LOG_LEVELS:
            logger.warning(
                "[TABLET_EDGE_APP_LOG] Unknown log_level received: %s. Defaulting to INFO.",
                log_level,
            )
            return "INFO"

        return log_level

    @staticmethod
    def validate_context(value: Any) -> dict[str, Any] | None:
        """
        Validate context for JSONB storage.

        Expected:
            None or JSON object / Python dict.
        """
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError("context must be a JSON object when provided.")

        try:
            json.dumps(value, default=str)
        except Exception as exc:
            raise ValueError("context is not JSON serializable.") from exc

        return value

    def validate_log_payload(self, log_item: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize one app log payload.

        Required:
            message

        Optional:
            log_level
            log_source
            context
            client_created_at
        """
        if not isinstance(log_item, dict):
            raise ValueError("Each app log must be a JSON object.")

        message = self.normalize_string(log_item.get("message"))

        if not message:
            raise ValueError("message is required.")

        return {
            "log_level": self.normalize_log_level(log_item.get("log_level")),
            "log_source": self.normalize_string(log_item.get("log_source"), max_length=150),
            "message": message,
            "context": self.validate_context(log_item.get("context")),
            "client_created_at": self.normalize_datetime(log_item.get("client_created_at")),
        }

    def validate_logs_list(
        self,
        logs: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Validate the incoming logs array.
        """
        if not isinstance(logs, list):
            raise ValueError("logs must be a list.")

        if len(logs) > max_batch_size:
            raise ValueError(
                f"Too many app logs in one request. "
                f"max_batch_size={max_batch_size}, received={len(logs)}"
            )

        normalized_logs: list[dict[str, Any]] = []

        for index, log_item in enumerate(logs):
            try:
                normalized_logs.append(self.validate_log_payload(log_item))
            except Exception as exc:
                raise ValueError(f"Invalid app log at index {index}: {exc}") from exc

        return normalized_logs

    def record_logs(
        self,
        session: Session,
        tablet_device_id: int | None,
        logs: list[dict[str, Any]],
    ) -> list[TabletAppLog]:
        """
        Insert app logs for a tablet.

        Args:
            session:
                Existing SQLAlchemy session owned by orchestrator.
            tablet_device_id:
                ID from tablet_edge.tablet_device. Can be None if the tablet
                could not be resolved but the route still wants to preserve logs.
            logs:
                Raw log payload list from the tablet/client.

        Returns:
            List of inserted TabletAppLog ORM objects.
        """
        normalized_logs = self.validate_logs_list(logs)

        inserted_logs: list[TabletAppLog] = []

        for log_data in normalized_logs:
            app_log = TabletAppLog(
                tablet_device_id=tablet_device_id,
                log_level=log_data["log_level"],
                log_source=log_data["log_source"],
                message=log_data["message"],
                context=log_data["context"],
                client_created_at=log_data["client_created_at"],
            )

            session.add(app_log)
            inserted_logs.append(app_log)

        session.flush()

        logger.info(
            "[TABLET_EDGE_APP_LOG] Recorded %s app log(s) for tablet_device_id=%s",
            len(inserted_logs),
            tablet_device_id,
        )

        return inserted_logs

    def get_recent_logs(
        self,
        session: Session,
        tablet_device_id: int | None = None,
        *,
        limit: int = 100,
    ) -> list[TabletAppLog]:
        """
        Return recent app logs.

        If tablet_device_id is provided, only logs for that tablet are returned.
        """
        safe_limit = max(1, min(int(limit), 500))

        stmt = select(TabletAppLog)

        if tablet_device_id is not None:
            stmt = stmt.where(TabletAppLog.tablet_device_id == tablet_device_id)

        stmt = stmt.order_by(desc(TabletAppLog.created_at)).limit(safe_limit)

        return list(session.execute(stmt).scalars().all())

    def build_record_logs_response(
        self,
        inserted_logs: list[TabletAppLog],
        *,
        failed: int = 0,
    ) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/app-logs.
        """
        return {
            "success": True,
            "accepted": len(inserted_logs),
            "failed": failed,
            "log_ids": [log_item.id for log_item in inserted_logs],
        }