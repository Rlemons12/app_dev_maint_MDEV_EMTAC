"""
Tablet offline event service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_offline_event_service.py

Responsibilities:
    - Validate offline event sync payloads.
    - Normalize local_event_id UUID values.
    - Normalize event_type values.
    - Validate JSONB event_payload data.
    - Insert new offline events.
    - Detect duplicate tablet/local_event_id pairs.
    - Keep service logic session-safe.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import TabletOfflineEvent


class TabletOfflineEventService:
    """
    Domain service for tablet_edge.tablet_offline_event.
    """

    VALID_EVENT_TYPES = {
        "feedback_submitted",
        "ai_answer_rating",
        "comment_created",
        "search_query_attempted",
        "image_search_attempted",
        "training_progress_update",
        "part_viewed",
        "drawing_viewed",
        "document_viewed",
        "route_test_event",
    }

    VALID_PROCESSING_STATUSES = {
        "pending",
        "processed",
        "failed",
    }

    DEFAULT_MAX_BATCH_SIZE = 500

    @staticmethod
    def normalize_uuid(value: Any, *, field_name: str = "uuid") -> UUID:
        """
        Convert string/UUID value into a UUID object.

        Raises:
            ValueError if missing or invalid.
        """
        if isinstance(value, UUID):
            return value

        if value is None:
            raise ValueError(f"{field_name} is required.")

        try:
            return UUID(str(value).strip())
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc

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
    def normalize_datetime(value: Any, *, field_name: str = "datetime") -> datetime:
        """
        Normalize required datetime values.

        Accepts:
            - datetime objects
            - ISO strings
            - ISO strings ending in Z

        If a datetime is naive, UTC is assumed.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if value is None:
            raise ValueError(f"{field_name} is required.")

        normalized = str(value).strip()

        if not normalized:
            raise ValueError(f"{field_name} is required.")

        try:
            normalized = normalized.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    @staticmethod
    def normalize_optional_datetime(value: Any) -> datetime | None:
        """
        Normalize optional datetime values.
        """
        if value is None:
            return None

        if isinstance(value, str) and not value.strip():
            return None

        return TabletOfflineEventService.normalize_datetime(value)

    def normalize_event_type(self, value: Any) -> str:
        """
        Normalize event_type.

        Unknown event types are allowed but logged so the Android app can add
        event types without breaking older server builds.
        """
        event_type = self.normalize_string(value, max_length=100)

        if not event_type:
            raise ValueError("event_type is required.")

        event_type = event_type.lower()

        if event_type not in self.VALID_EVENT_TYPES:
            logger.warning(
                "[TABLET_EDGE_OFFLINE] Unknown offline event_type received: %s",
                event_type,
            )

        return event_type

    def normalize_processing_status(self, value: Any) -> str:
        """
        Normalize processing_status.

        Used by mark_processed/mark_failed helpers.
        """
        status = self.normalize_string(value, max_length=50)

        if not status:
            return "pending"

        status = status.lower()

        if status not in self.VALID_PROCESSING_STATUSES:
            raise ValueError(f"Invalid processing_status: {value!r}")

        return status

    @staticmethod
    def validate_event_payload(value: Any) -> dict[str, Any]:
        """
        Validate event_payload for JSONB storage.

        Expected:
            JSON object / Python dict.

        This intentionally rejects arbitrary objects so bad payloads do not
        make it to the database layer.
        """
        if value is None:
            raise ValueError("event_payload is required.")

        if not isinstance(value, dict):
            raise ValueError("event_payload must be a JSON object.")

        try:
            json.dumps(value, default=str)
        except Exception as exc:
            raise ValueError("event_payload is not JSON serializable.") from exc

        return value

    def validate_offline_event_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize one offline event payload.

        Required:
            local_event_id
            event_type
            client_created_at
            event_payload
        """
        if not isinstance(event, dict):
            raise ValueError("Each offline event must be a JSON object.")

        return {
            "local_event_id": self.normalize_uuid(
                event.get("local_event_id"),
                field_name="local_event_id",
            ),
            "event_type": self.normalize_event_type(event.get("event_type")),
            "event_payload": self.validate_event_payload(event.get("event_payload")),
            "client_created_at": self.normalize_datetime(
                event.get("client_created_at"),
                field_name="client_created_at",
            ),
        }

    def validate_events_list(
        self,
        events: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Validate the incoming offline events array.
        """
        if not isinstance(events, list):
            raise ValueError("events must be a list.")

        if len(events) > max_batch_size:
            raise ValueError(
                f"Too many offline events in one request. "
                f"max_batch_size={max_batch_size}, received={len(events)}"
            )

        normalized_events: list[dict[str, Any]] = []

        for index, event in enumerate(events):
            try:
                normalized_events.append(self.validate_offline_event_payload(event))
            except Exception as exc:
                raise ValueError(f"Invalid offline event at index {index}: {exc}") from exc

        return normalized_events

    def get_existing_local_event_ids(
        self,
        session: Session,
        tablet_device_id: int,
        local_event_ids: list[UUID],
    ) -> set[UUID]:
        """
        Return local_event_id values that already exist for this tablet.

        This prevents duplicate inserts before hitting the database unique
        constraint.
        """
        if not local_event_ids:
            return set()

        stmt = select(TabletOfflineEvent.local_event_id).where(
            TabletOfflineEvent.tablet_device_id == tablet_device_id,
            TabletOfflineEvent.local_event_id.in_(local_event_ids),
        )

        return set(session.execute(stmt).scalars().all())

    def sync_offline_events(
        self,
        session: Session,
        tablet_device_id: int,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Accept offline events from the tablet.

        Args:
            session:
                Existing SQLAlchemy session owned by orchestrator.
            tablet_device_id:
                ID from tablet_edge.tablet_device.
            events:
                Raw event payload list from the tablet/client.

        Returns:
            Response-safe dictionary containing accepted, duplicates, and failed counts.
        """
        normalized_events = self.validate_events_list(events)

        local_event_ids = [
            event_data["local_event_id"]
            for event_data in normalized_events
        ]

        existing_ids = self.get_existing_local_event_ids(
            session=session,
            tablet_device_id=tablet_device_id,
            local_event_ids=local_event_ids,
        )

        inserted_events: list[TabletOfflineEvent] = []
        duplicate_ids: list[str] = []

        for event_data in normalized_events:
            local_event_id = event_data["local_event_id"]

            if local_event_id in existing_ids:
                duplicate_ids.append(str(local_event_id))
                continue

            offline_event = TabletOfflineEvent(
                tablet_device_id=tablet_device_id,
                local_event_id=local_event_id,
                event_type=event_data["event_type"],
                event_payload=event_data["event_payload"],
                client_created_at=event_data["client_created_at"],
                processing_status="pending",
            )

            session.add(offline_event)
            inserted_events.append(offline_event)

        session.flush()

        logger.info(
            "[TABLET_EDGE_OFFLINE] Synced offline events for tablet_device_id=%s accepted=%s duplicates=%s failed=%s",
            tablet_device_id,
            len(inserted_events),
            len(duplicate_ids),
            0,
        )

        return {
            "success": True,
            "accepted": len(inserted_events),
            "duplicates": len(duplicate_ids),
            "failed": 0,
            "event_ids": [event.id for event in inserted_events],
            "duplicate_local_event_ids": duplicate_ids,
        }

    def mark_processed(
        self,
        session: Session,
        event_id: int,
    ) -> TabletOfflineEvent:
        """
        Mark an offline event as processed.

        Transaction ownership belongs to the orchestrator.
        """
        event = session.get(TabletOfflineEvent, event_id)

        if event is None:
            raise ValueError(f"Offline event not found: {event_id}")

        event.processing_status = "processed"
        event.processed_at = datetime.now(timezone.utc)
        event.error_message = None

        session.flush()

        logger.info(
            "[TABLET_EDGE_OFFLINE] Marked offline event processed event_id=%s",
            event_id,
        )

        return event

    def mark_failed(
        self,
        session: Session,
        event_id: int,
        error_message: str,
    ) -> TabletOfflineEvent:
        """
        Mark an offline event as failed.

        Transaction ownership belongs to the orchestrator.
        """
        event = session.get(TabletOfflineEvent, event_id)

        if event is None:
            raise ValueError(f"Offline event not found: {event_id}")

        event.processing_status = "failed"
        event.processed_at = datetime.now(timezone.utc)
        event.error_message = self.normalize_string(error_message) or "Unknown error"

        session.flush()

        logger.warning(
            "[TABLET_EDGE_OFFLINE] Marked offline event failed event_id=%s error=%s",
            event_id,
            event.error_message,
        )

        return event

    def get_pending_events(
        self,
        session: Session,
        tablet_device_id: int | None = None,
        *,
        limit: int = 100,
    ) -> list[TabletOfflineEvent]:
        """
        Return pending offline events.

        If tablet_device_id is provided, only pending events for that tablet are returned.
        """
        safe_limit = max(1, min(int(limit), 500))

        stmt = select(TabletOfflineEvent).where(
            TabletOfflineEvent.processing_status == "pending"
        )

        if tablet_device_id is not None:
            stmt = stmt.where(TabletOfflineEvent.tablet_device_id == tablet_device_id)

        stmt = stmt.order_by(TabletOfflineEvent.created_at.asc()).limit(safe_limit)

        return list(session.execute(stmt).scalars().all())

    def build_sync_response(self, sync_result: dict[str, Any]) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/offline-events/sync.
        """
        return {
            "success": bool(sync_result.get("success", False)),
            "accepted": int(sync_result.get("accepted", 0)),
            "duplicates": int(sync_result.get("duplicates", 0)),
            "failed": int(sync_result.get("failed", 0)),
            "event_ids": sync_result.get("event_ids", []),
            "duplicate_local_event_ids": sync_result.get("duplicate_local_event_ids", []),
        }