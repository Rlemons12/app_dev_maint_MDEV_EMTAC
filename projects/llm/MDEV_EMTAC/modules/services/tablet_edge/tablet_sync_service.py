"""
Tablet sync service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_sync_service.py

Responsibilities:
    - Record sync attempts between tablet and EMTAC server.
    - Track sync direction.
    - Track sync status.
    - Track records sent/received/failed.
    - Track sync duration.
    - Query recent sync events.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import TabletSyncEvent


class TabletSyncService:
    """
    Domain service for tablet_edge.tablet_sync_event.
    """

    VALID_SYNC_TYPES = {
        "network_events",
        "health_samples",
        "dropdown_cache",
        "offline_events",
        "app_logs",
        "heartbeat",
        "registration",
        "unknown",
    }

    VALID_SYNC_DIRECTIONS = {
        "tablet_to_server",
        "server_to_tablet",
        "bidirectional",
        "unknown",
    }

    VALID_STATUSES = {
        "started",
        "success",
        "failed",
        "partial",
        "cancelled",
        "unknown",
    }

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
    def normalize_int(
        value: Any,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
        default: int | None = None,
    ) -> int | None:
        """
        Normalize optional integer values.
        """
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        try:
            normalized = int(value)
        except Exception as exc:
            raise ValueError(f"Invalid integer value: {value!r}") from exc

        if min_value is not None and normalized < min_value:
            normalized = min_value

        if max_value is not None and normalized > max_value:
            normalized = max_value

        return normalized

    @staticmethod
    def utc_now() -> datetime:
        """
        Return timezone-aware UTC datetime.
        """
        return datetime.now(timezone.utc)

    @staticmethod
    def calculate_duration_ms(
        started_at: datetime | None,
        completed_at: datetime | None,
    ) -> int | None:
        """
        Calculate duration in milliseconds.
        """
        if started_at is None or completed_at is None:
            return None

        return max(0, int((completed_at - started_at).total_seconds() * 1000))

    def normalize_sync_type(self, value: Any) -> str:
        """
        Normalize sync_type.

        Unknown values are allowed but converted to 'unknown'.
        """
        sync_type = self.normalize_string(value, max_length=100)

        if not sync_type:
            return "unknown"

        sync_type = sync_type.lower()

        if sync_type not in self.VALID_SYNC_TYPES:
            logger.warning(
                "[TABLET_EDGE_SYNC] Unknown sync_type received: %s",
                sync_type,
            )
            return "unknown"

        return sync_type

    def normalize_sync_direction(self, value: Any) -> str:
        """
        Normalize sync_direction.
        """
        sync_direction = self.normalize_string(value, max_length=50)

        if not sync_direction:
            return "unknown"

        sync_direction = sync_direction.lower()

        if sync_direction not in self.VALID_SYNC_DIRECTIONS:
            logger.warning(
                "[TABLET_EDGE_SYNC] Unknown sync_direction received: %s",
                sync_direction,
            )
            return "unknown"

        return sync_direction

    def normalize_status(self, value: Any) -> str:
        """
        Normalize sync status.
        """
        status = self.normalize_string(value, max_length=50)

        if not status:
            return "unknown"

        status = status.lower()

        if status not in self.VALID_STATUSES:
            logger.warning(
                "[TABLET_EDGE_SYNC] Unknown sync status received: %s",
                status,
            )
            return "unknown"

        return status

    def start_sync(
        self,
        session: Session,
        tablet_device_id: int,
        sync_type: str,
        sync_direction: str,
    ) -> TabletSyncEvent:
        """
        Start a sync event.

        Args:
            session:
                Existing SQLAlchemy session owned by orchestrator.
            tablet_device_id:
                ID from tablet_edge.tablet_device.
            sync_type:
                Example: network_events, health_samples, dropdown_cache.
            sync_direction:
                Example: tablet_to_server, server_to_tablet, bidirectional.
        """
        event = TabletSyncEvent(
            tablet_device_id=tablet_device_id,
            sync_type=self.normalize_sync_type(sync_type),
            sync_direction=self.normalize_sync_direction(sync_direction),
            status="started",
            records_sent=0,
            records_received=0,
            records_failed=0,
            started_at=self.utc_now(),
        )

        session.add(event)
        session.flush()

        logger.info(
            "[TABLET_EDGE_SYNC] Started sync event id=%s tablet_device_id=%s sync_type=%s direction=%s",
            event.id,
            tablet_device_id,
            event.sync_type,
            event.sync_direction,
        )

        return event

    def complete_sync(
        self,
        session: Session,
        sync_event_id: int,
        *,
        records_sent: int = 0,
        records_received: int = 0,
        records_failed: int = 0,
        status: str | None = None,
    ) -> TabletSyncEvent:
        """
        Mark a sync event complete.

        If records_failed is greater than 0 and status is not provided,
        the status becomes 'partial'. Otherwise it becomes 'success'.
        """
        event = session.get(TabletSyncEvent, sync_event_id)

        if event is None:
            raise ValueError(f"Sync event not found: {sync_event_id}")

        completed_at = self.utc_now()

        safe_records_sent = self.normalize_int(records_sent, min_value=0, default=0) or 0
        safe_records_received = self.normalize_int(records_received, min_value=0, default=0) or 0
        safe_records_failed = self.normalize_int(records_failed, min_value=0, default=0) or 0

        if status:
            final_status = self.normalize_status(status)
        elif safe_records_failed > 0:
            final_status = "partial"
        else:
            final_status = "success"

        event.status = final_status
        event.records_sent = safe_records_sent
        event.records_received = safe_records_received
        event.records_failed = safe_records_failed
        event.completed_at = completed_at
        event.duration_ms = self.calculate_duration_ms(event.started_at, completed_at)
        event.error_message = None

        session.flush()

        logger.info(
            "[TABLET_EDGE_SYNC] Completed sync event id=%s status=%s sent=%s received=%s failed=%s duration_ms=%s",
            event.id,
            event.status,
            event.records_sent,
            event.records_received,
            event.records_failed,
            event.duration_ms,
        )

        return event

    def fail_sync(
        self,
        session: Session,
        sync_event_id: int,
        error_message: str,
        *,
        records_sent: int = 0,
        records_received: int = 0,
        records_failed: int = 0,
    ) -> TabletSyncEvent:
        """
        Mark a sync event failed.
        """
        event = session.get(TabletSyncEvent, sync_event_id)

        if event is None:
            raise ValueError(f"Sync event not found: {sync_event_id}")

        completed_at = self.utc_now()

        event.status = "failed"
        event.records_sent = self.normalize_int(records_sent, min_value=0, default=0) or 0
        event.records_received = self.normalize_int(records_received, min_value=0, default=0) or 0
        event.records_failed = self.normalize_int(records_failed, min_value=0, default=0) or 0
        event.completed_at = completed_at
        event.duration_ms = self.calculate_duration_ms(event.started_at, completed_at)
        event.error_message = self.normalize_string(error_message) or "Unknown sync error"

        session.flush()

        logger.warning(
            "[TABLET_EDGE_SYNC] Failed sync event id=%s error=%s",
            event.id,
            event.error_message,
        )

        return event

    def cancel_sync(
        self,
        session: Session,
        sync_event_id: int,
        reason: str | None = None,
    ) -> TabletSyncEvent:
        """
        Mark a sync event cancelled.
        """
        event = session.get(TabletSyncEvent, sync_event_id)

        if event is None:
            raise ValueError(f"Sync event not found: {sync_event_id}")

        completed_at = self.utc_now()

        event.status = "cancelled"
        event.completed_at = completed_at
        event.duration_ms = self.calculate_duration_ms(event.started_at, completed_at)
        event.error_message = self.normalize_string(reason) or "Sync cancelled."

        session.flush()

        logger.info(
            "[TABLET_EDGE_SYNC] Cancelled sync event id=%s reason=%s",
            event.id,
            event.error_message,
        )

        return event

    def record_quick_sync(
        self,
        session: Session,
        tablet_device_id: int,
        *,
        sync_type: str,
        sync_direction: str,
        status: str,
        records_sent: int = 0,
        records_received: int = 0,
        records_failed: int = 0,
        error_message: str | None = None,
    ) -> TabletSyncEvent:
        """
        Record a sync event in one call.

        Useful when the route already performed the work and only needs to log
        the result.
        """
        started_at = self.utc_now()
        completed_at = self.utc_now()

        final_status = self.normalize_status(status)

        event = TabletSyncEvent(
            tablet_device_id=tablet_device_id,
            sync_type=self.normalize_sync_type(sync_type),
            sync_direction=self.normalize_sync_direction(sync_direction),
            status=final_status,
            records_sent=self.normalize_int(records_sent, min_value=0, default=0) or 0,
            records_received=self.normalize_int(records_received, min_value=0, default=0) or 0,
            records_failed=self.normalize_int(records_failed, min_value=0, default=0) or 0,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=self.calculate_duration_ms(started_at, completed_at),
            error_message=self.normalize_string(error_message),
        )

        session.add(event)
        session.flush()

        logger.info(
            "[TABLET_EDGE_SYNC] Recorded quick sync id=%s tablet_device_id=%s sync_type=%s status=%s",
            event.id,
            tablet_device_id,
            event.sync_type,
            event.status,
        )

        return event

    def get_recent_sync_events(
        self,
        session: Session,
        tablet_device_id: int | None = None,
        *,
        limit: int = 100,
    ) -> list[TabletSyncEvent]:
        """
        Return recent sync events.

        If tablet_device_id is provided, only sync events for that tablet are returned.
        """
        safe_limit = max(1, min(int(limit), 500))

        stmt = select(TabletSyncEvent)

        if tablet_device_id is not None:
            stmt = stmt.where(TabletSyncEvent.tablet_device_id == tablet_device_id)

        stmt = stmt.order_by(desc(TabletSyncEvent.started_at)).limit(safe_limit)

        return list(session.execute(stmt).scalars().all())

    def build_sync_event_response(
        self,
        event: TabletSyncEvent,
    ) -> dict[str, Any]:
        """
        Build a response-safe dictionary for one sync event.
        """
        return {
            "success": True,
            "sync_event": event.to_dict(),
        }

    def build_recent_sync_events_response(
        self,
        events: list[TabletSyncEvent],
    ) -> dict[str, Any]:
        """
        Build a response-safe dictionary for recent sync events.
        """
        return {
            "success": True,
            "count": len(events),
            "sync_events": [event.to_dict() for event in events],
        }
