"""
Tablet app update service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_app_update_service.py

Responsibilities:
    - Check whether an EMTAC tablet has an APK update available.
    - Resolve APK release download metadata.
    - Record tablet-side update lifecycle reports.
    - Keep database work inside a provided SQLAlchemy session.
    - Do not create sessions.
    - Do not commit or rollback.

Routes supported through coordinator/orchestrator:
    GET  /tablet-edge/app-update/check
    GET  /tablet-edge/app-update/download/<release_id>
    POST /tablet-edge/app-update/report
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


DEFAULT_APP_PACKAGE = "com.example.emtactablet"
DEFAULT_RELEASE_CHANNEL = "stable"

APK_MIME_TYPE = "application/vnd.android.package-archive"


class TabletAppUpdateService:
    """
    Service for EMTAC Tablet Edge Agent update checks and reports.
    """

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def normalize_string(value: Any, *, default: str | None = None) -> str | None:
        if value is None:
            return default

        normalized = str(value).strip()

        if not normalized:
            return default

        return normalized

    @staticmethod
    def normalize_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None

        if isinstance(value, str) and not value.strip():
            return None

        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None

        return dict(row._mapping)

    @staticmethod
    def _json_safe_datetime(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()

        return value

    def get_tablet_by_uid(
        self,
        session: Session,
        tablet_uid: str,
    ) -> dict[str, Any]:
        """
        Resolve an active tablet by tablet_uid.
        """
        safe_tablet_uid = self.normalize_string(tablet_uid)

        if not safe_tablet_uid:
            raise ValueError("tablet_uid is required.")

        row = session.execute(
            text(
                """
                SELECT
                    id,
                    tablet_uid,
                    tablet_name,
                    app_version,
                    app_version_code,
                    is_active
                FROM tablet_edge.tablet_device
                WHERE tablet_uid = CAST(:tablet_uid AS UUID)
                LIMIT 1;
                """
            ),
            {"tablet_uid": safe_tablet_uid},
        ).fetchone()

        tablet = self._row_to_dict(row)

        if tablet is None:
            raise ValueError(f"Tablet was not found for tablet_uid={safe_tablet_uid}.")

        if not tablet.get("is_active", False):
            raise ValueError(f"Tablet is inactive for tablet_uid={safe_tablet_uid}.")

        return tablet

    def update_tablet_version_if_supplied(
        self,
        session: Session,
        *,
        tablet_uid: str,
        version_name: str | None = None,
        version_code: int | None = None,
    ) -> None:
        """
        Update tablet_device app version fields when the tablet reports them.

        This is useful after install_completed and also safe for heartbeat/check
        style requests later.
        """
        safe_tablet_uid = self.normalize_string(tablet_uid)

        if not safe_tablet_uid:
            raise ValueError("tablet_uid is required.")

        if version_name is None and version_code is None:
            return

        session.execute(
            text(
                """
                UPDATE tablet_edge.tablet_device
                SET
                    app_version = COALESCE(:version_name, app_version),
                    app_version_code = COALESCE(:version_code, app_version_code),
                    updated_at = NOW()
                WHERE tablet_uid = CAST(:tablet_uid AS UUID);
                """
            ),
            {
                "tablet_uid": safe_tablet_uid,
                "version_name": version_name,
                "version_code": version_code,
            },
        )

    def get_latest_active_release(
        self,
        session: Session,
        *,
        app_package: str = DEFAULT_APP_PACKAGE,
        release_channel: str = DEFAULT_RELEASE_CHANNEL,
        current_version_code: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get the newest active release that is currently published and not retired.
        """
        row = session.execute(
            text(
                """
                SELECT
                    id,
                    app_package,
                    release_channel,
                    version_name,
                    version_code,
                    apk_filename,
                    apk_file_path,
                    apk_sha256,
                    apk_size_bytes,
                    release_notes,
                    min_supported_version_code,
                    max_supported_version_code,
                    is_active,
                    is_required,
                    rollout_percent,
                    created_by,
                    published_at,
                    retired_at,
                    created_at,
                    updated_at
                FROM tablet_edge.tablet_app_release
                WHERE app_package = :app_package
                  AND release_channel = :release_channel
                  AND is_active = TRUE
                  AND (published_at IS NULL OR published_at <= NOW())
                  AND retired_at IS NULL
                  AND (
                        :current_version_code IS NULL
                        OR min_supported_version_code IS NULL
                        OR :current_version_code >= min_supported_version_code
                  )
                  AND (
                        :current_version_code IS NULL
                        OR max_supported_version_code IS NULL
                        OR :current_version_code <= max_supported_version_code
                  )
                ORDER BY version_code DESC
                LIMIT 1;
                """
            ),
            {
                "app_package": app_package,
                "release_channel": release_channel,
                "current_version_code": current_version_code,
            },
        ).fetchone()

        release = self._row_to_dict(row)

        if release is None:
            return None

        for key in ("published_at", "retired_at", "created_at", "updated_at"):
            release[key] = self._json_safe_datetime(release.get(key))

        return release

    def build_update_check_response(
        self,
        session: Session,
        *,
        tablet_uid: str | None = None,
        current_version_code: int | None = None,
        current_version_name: str | None = None,
        app_package: str = DEFAULT_APP_PACKAGE,
        release_channel: str = DEFAULT_RELEASE_CHANNEL,
    ) -> dict[str, Any]:
        """
        Build the update check response for a tablet.
        """
        safe_app_package = self.normalize_string(
            app_package,
            default=DEFAULT_APP_PACKAGE,
        ) or DEFAULT_APP_PACKAGE

        safe_release_channel = self.normalize_string(
            release_channel,
            default=DEFAULT_RELEASE_CHANNEL,
        ) or DEFAULT_RELEASE_CHANNEL

        tablet: dict[str, Any] | None = None

        if tablet_uid:
            tablet = self.get_tablet_by_uid(session, tablet_uid)

            if current_version_code is None:
                current_version_code = tablet.get("app_version_code")

            if current_version_name is None:
                current_version_name = tablet.get("app_version")

        latest_release = self.get_latest_active_release(
            session=session,
            app_package=safe_app_package,
            release_channel=safe_release_channel,
            current_version_code=current_version_code,
        )

        update_available = False
        update_status = "no_release_found"

        if latest_release is not None:
            latest_version_code = int(latest_release["version_code"])

            if current_version_code is None:
                update_available = True
                update_status = "tablet_version_unknown"
            elif latest_version_code > current_version_code:
                update_available = True
                update_status = "update_available"
            elif latest_version_code == current_version_code:
                update_available = False
                update_status = "up_to_date"
            else:
                update_available = False
                update_status = "tablet_newer_than_release"

        response: dict[str, Any] = {
            "success": True,
            "service": "tablet_edge",
            "update_available": update_available,
            "update_status": update_status,
            "app_package": safe_app_package,
            "release_channel": safe_release_channel,
            "installed_version_name": current_version_name,
            "installed_version_code": current_version_code,
            "server_time_utc": self.utc_now_iso(),
        }

        if tablet is not None:
            response["tablet_device_id"] = tablet["id"]
            response["tablet_uid"] = str(tablet["tablet_uid"])
            response["tablet_name"] = tablet["tablet_name"]

        if latest_release is not None:
            release_id = latest_release["id"]

            response["latest_release"] = {
                "release_id": release_id,
                "version_name": latest_release["version_name"],
                "version_code": latest_release["version_code"],
                "apk_filename": latest_release["apk_filename"],
                "apk_size_bytes": latest_release["apk_size_bytes"],
                "apk_sha256": latest_release["apk_sha256"],
                "release_notes": latest_release["release_notes"],
                "is_required": latest_release["is_required"],
                "rollout_percent": latest_release["rollout_percent"],
                "published_at": latest_release["published_at"],
            }

            response["download_path"] = (
                f"/tablet-edge/app-update/download/{release_id}"
            )

        return response

    def get_download_release(
        self,
        session: Session,
        *,
        release_id: int,
    ) -> dict[str, Any]:
        """
        Resolve a release for APK download.
        """
        row = session.execute(
            text(
                """
                SELECT
                    id,
                    app_package,
                    release_channel,
                    version_name,
                    version_code,
                    apk_filename,
                    apk_file_path,
                    apk_sha256,
                    apk_size_bytes,
                    release_notes,
                    is_active,
                    is_required,
                    published_at,
                    retired_at
                FROM tablet_edge.tablet_app_release
                WHERE id = :release_id
                LIMIT 1;
                """
            ),
            {"release_id": release_id},
        ).fetchone()

        release = self._row_to_dict(row)

        if release is None:
            raise ValueError(f"Release was not found for release_id={release_id}.")

        if not release.get("is_active", False):
            raise ValueError(f"Release {release_id} is not active.")

        if release.get("retired_at") is not None:
            raise ValueError(f"Release {release_id} has been retired.")

        apk_path = Path(str(release["apk_file_path"])).expanduser()

        if not apk_path.exists():
            raise FileNotFoundError(f"APK file does not exist: {apk_path}")

        if not apk_path.is_file():
            raise FileNotFoundError(f"APK path is not a file: {apk_path}")

        release["apk_file_path"] = str(apk_path)
        release["mime_type"] = APK_MIME_TYPE

        for key in ("published_at", "retired_at"):
            release[key] = self._json_safe_datetime(release.get(key))

        return release

    def calculate_file_sha256(self, file_path: str | Path) -> str:
        """
        Calculate SHA-256 for a file.

        This is not used on every download by default because APK files can be
        large. Use it from test/admin scripts if you want to verify stored hashes.
        """
        path = Path(file_path)
        sha256 = hashlib.sha256()

        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def record_update_report(
        self,
        session: Session,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Record an update event reported by the tablet.

        Expected payload:
            {
                "tablet_uid": "...",
                "event_type": "install_completed",
                "release_id": 1,
                "version_name": "1.0.2",
                "version_code": 3,
                "message": "optional",
                "details": {}
            }

        Common event_type values:
            update_available_seen
            download_started
            download_completed
            download_failed
            install_launched
            install_completed
            install_failed
        """
        tablet_uid = self.normalize_string(payload.get("tablet_uid"))

        if not tablet_uid:
            raise ValueError("tablet_uid is required.")

        event_type = self.normalize_string(
            payload.get("event_type"),
            default="app_update_report",
        ) or "app_update_report"

        release_id = self.normalize_int(
            payload.get("release_id"),
            field_name="release_id",
        )

        version_name = self.normalize_string(payload.get("version_name"))

        version_code = self.normalize_int(
            payload.get("version_code"),
            field_name="version_code",
        )

        message = self.normalize_string(payload.get("message"))

        details = payload.get("details")

        if details is not None and not isinstance(details, dict):
            raise ValueError("details must be a JSON object when supplied.")

        tablet = self.get_tablet_by_uid(session, tablet_uid)

        if event_type == "install_completed":
            self.update_tablet_version_if_supplied(
                session=session,
                tablet_uid=tablet_uid,
                version_name=version_name,
                version_code=version_code,
            )

        context = {
            "event_type": event_type,
            "release_id": release_id,
            "version_name": version_name,
            "version_code": version_code,
            "message": message,
            "details": details or {},
        }

        inserted_log_id = session.execute(
            text(
                """
                INSERT INTO tablet_edge.tablet_app_log (
                    tablet_device_id,
                    log_level,
                    log_source,
                    message,
                    context,
                    client_created_at,
                    server_received_at,
                    created_at
                )
                VALUES (
                    :tablet_device_id,
                    :log_level,
                    :log_source,
                    :message,
                    CAST(:context AS JSONB),
                    NOW(),
                    NOW(),
                    NOW()
                )
                RETURNING id;
                """
            ),
            {
                "tablet_device_id": tablet["id"],
                "log_level": "INFO",
                "log_source": "tablet_app_update",
                "message": message or f"Tablet app update event: {event_type}",
                "context": json.dumps(context),
            },
        ).scalar_one()

        return {
            "success": True,
            "accepted": True,
            "tablet_device_id": tablet["id"],
            "tablet_uid": str(tablet["tablet_uid"]),
            "tablet_name": tablet["tablet_name"],
            "event_type": event_type,
            "release_id": release_id,
            "app_log_id": inserted_log_id,
            "server_time_utc": self.utc_now_iso(),
        }