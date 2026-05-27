"""
Tablet device service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_device_service.py

Responsibilities:
    - Resolve tablet devices by tablet_uid.
    - Register a new tablet.
    - Update an existing tablet.
    - Update heartbeat / last_seen_at.
    - Keep service logic session-safe.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from modules.emtacdb.tablet_edge.tablet_edge_models import TabletDevice
from modules.configuration.log_config import logger


class TabletDeviceService:
    """
    Domain service for tablet_edge.tablet_device.
    """

    REQUIRED_REGISTER_FIELDS = {
        "tablet_uid",
        "tablet_name",
    }

    @staticmethod
    def normalize_uuid(value: Any) -> UUID:
        """
        Convert a string/UUID value into a UUID object.

        Raises:
            ValueError if value is missing or invalid.
        """
        if isinstance(value, UUID):
            return value

        if value is None:
            raise ValueError("tablet_uid is required.")

        try:
            return UUID(str(value).strip())
        except Exception as exc:
            raise ValueError(f"Invalid tablet_uid: {value!r}") from exc

    @staticmethod
    def normalize_string(value: Any, *, max_length: int | None = None) -> str | None:
        """
        Normalize incoming optional string values.

        Empty strings become None unless required by the caller.
        """
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        if max_length is not None and len(normalized) > max_length:
            normalized = normalized[:max_length]

        return normalized

    def validate_register_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize register payload.

        Required:
            tablet_uid
            tablet_name
        """
        if not isinstance(payload, dict):
            raise ValueError("Register payload must be a JSON object.")

        missing = [
            field
            for field in self.REQUIRED_REGISTER_FIELDS
            if not self.normalize_string(payload.get(field))
        ]

        if missing:
            raise ValueError(f"Missing required register fields: {', '.join(missing)}")

        tablet_uid = self.normalize_uuid(payload.get("tablet_uid"))
        tablet_name = self.normalize_string(payload.get("tablet_name"), max_length=150)

        if not tablet_name:
            raise ValueError("tablet_name is required.")

        return {
            "tablet_uid": tablet_uid,
            "tablet_name": tablet_name,
            "device_make": self.normalize_string(payload.get("device_make"), max_length=100),
            "device_model": self.normalize_string(payload.get("device_model"), max_length=100),
            "android_version": self.normalize_string(payload.get("android_version"), max_length=50),
            "app_version": self.normalize_string(payload.get("app_version"), max_length=50),
            "assigned_area": self.normalize_string(payload.get("assigned_area"), max_length=150),
            "assigned_station": self.normalize_string(payload.get("assigned_station"), max_length=150),
            "assigned_role": self.normalize_string(payload.get("assigned_role"), max_length=100),
        }

    def get_by_uid(
        self,
        session: Session,
        tablet_uid: str | UUID,
    ) -> TabletDevice | None:
        """
        Return a tablet by tablet_uid, or None if not found.
        """
        normalized_uid = self.normalize_uuid(tablet_uid)

        stmt = select(TabletDevice).where(TabletDevice.tablet_uid == normalized_uid)

        return session.execute(stmt).scalar_one_or_none()

    def require_active_by_uid(
        self,
        session: Session,
        tablet_uid: str | UUID,
    ) -> TabletDevice:
        """
        Return an active tablet by tablet_uid.

        Raises:
            ValueError if not found or inactive.
        """
        tablet = self.get_by_uid(session, tablet_uid)

        if tablet is None:
            raise ValueError(f"Tablet not registered: {tablet_uid}")

        if not tablet.is_active:
            raise ValueError(f"Tablet is inactive: {tablet_uid}")

        return tablet

    def register_or_update(
        self,
        session: Session,
        payload: dict[str, Any],
    ) -> TabletDevice:
        """
        Register a new tablet or update an existing one.

        Transaction ownership belongs to the orchestrator.
        """
        data = self.validate_register_payload(payload)
        now_utc = datetime.now(timezone.utc)

        tablet = self.get_by_uid(session, data["tablet_uid"])

        if tablet is None:
            tablet = TabletDevice(
                tablet_uid=data["tablet_uid"],
                tablet_name=data["tablet_name"],
                device_make=data["device_make"],
                device_model=data["device_model"],
                android_version=data["android_version"],
                app_version=data["app_version"],
                assigned_area=data["assigned_area"],
                assigned_station=data["assigned_station"],
                assigned_role=data["assigned_role"],
                is_active=True,
                last_seen_at=now_utc,
            )

            session.add(tablet)
            session.flush()

            logger.info(
                "[TABLET_EDGE_REGISTER] Registered new tablet id=%s uid=%s name=%s",
                tablet.id,
                tablet.tablet_uid,
                tablet.tablet_name,
            )

            return tablet

        tablet.tablet_name = data["tablet_name"]
        tablet.device_make = data["device_make"]
        tablet.device_model = data["device_model"]
        tablet.android_version = data["android_version"]
        tablet.app_version = data["app_version"]
        tablet.assigned_area = data["assigned_area"]
        tablet.assigned_station = data["assigned_station"]
        tablet.assigned_role = data["assigned_role"]
        tablet.last_seen_at = now_utc

        session.flush()

        logger.info(
            "[TABLET_EDGE_REGISTER] Updated existing tablet id=%s uid=%s name=%s",
            tablet.id,
            tablet.tablet_uid,
            tablet.tablet_name,
        )

        return tablet

    def update_heartbeat(
        self,
        session: Session,
        tablet_uid: str | UUID,
        payload: dict[str, Any] | None = None,
    ) -> TabletDevice:
        """
        Update last_seen_at and lightweight heartbeat metadata.

        Allowed heartbeat fields:
            app_version
        """
        payload = payload or {}
        now_utc = datetime.now(timezone.utc)

        tablet = self.require_active_by_uid(session, tablet_uid)

        app_version = self.normalize_string(payload.get("app_version"), max_length=50)

        if app_version:
            tablet.app_version = app_version

        tablet.last_seen_at = now_utc

        session.flush()

        logger.info(
            "[TABLET_EDGE_HEARTBEAT] Heartbeat updated tablet_id=%s uid=%s",
            tablet.id,
            tablet.tablet_uid,
        )

        return tablet

    def set_active_state(
        self,
        session: Session,
        tablet_uid: str | UUID,
        is_active: bool,
    ) -> TabletDevice:
        """
        Activate or deactivate a tablet.
        """
        tablet = self.get_by_uid(session, tablet_uid)

        if tablet is None:
            raise ValueError(f"Tablet not found: {tablet_uid}")

        tablet.is_active = bool(is_active)
        session.flush()

        logger.info(
            "[TABLET_EDGE_DEVICE] Set active state tablet_id=%s uid=%s is_active=%s",
            tablet.id,
            tablet.tablet_uid,
            tablet.is_active,
        )

        return tablet

    def build_register_response(self, tablet: TabletDevice) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/register.
        """
        return {
            "success": True,
            "tablet_device_id": tablet.id,
            "tablet_uid": str(tablet.tablet_uid),
            "tablet_name": tablet.tablet_name,
            "registered": True,
            "is_active": tablet.is_active,
            "last_seen_at": tablet.last_seen_at.isoformat() if tablet.last_seen_at else None,
        }

    def build_heartbeat_response(self, tablet: TabletDevice) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/heartbeat.
        """
        return {
            "success": True,
            "tablet_device_id": tablet.id,
            "tablet_uid": str(tablet.tablet_uid),
            "tablet_name": tablet.tablet_name,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "last_seen_at": tablet.last_seen_at.isoformat() if tablet.last_seen_at else None,
        }