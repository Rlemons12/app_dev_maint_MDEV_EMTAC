import logging
from typing import Dict, Optional, Set

from sqlalchemy import text

from modules.services.tablet_edge.tablet_user_session_report_dtos import (
    TabletUserSessionReportRequest,
    TabletUserSessionReportResult,
)


logger = logging.getLogger(__name__)


class TabletUserSessionReportService:
    """
    Handles tablet-reported user session state.

    This makes the tablet heartbeat/report the source of truth for:
        - which tablet is online
        - what tablet name is being reported
        - who the tablet reports is currently logged in
        - what page the tablet is currently on

    This service intentionally uses SQL text with column introspection so it
    can work even if the tablet_edge tables have slightly different columns
    between development versions.
    """

    DEVICE_SCHEMA = "tablet_edge"
    DEVICE_TABLE = "tablet_device"
    SESSION_SCHEMA = "tablet_edge"
    SESSION_TABLE = "tablet_user_session"

    @classmethod
    def report_user_session(
        cls,
        db_session,
        report: TabletUserSessionReportRequest,
    ) -> TabletUserSessionReportResult:
        tablet_uid = cls._clean(report.tablet_uid)
        tablet_name = cls._clean(report.tablet_name)
        username = cls._clean(report.username)
        display_name = cls._clean(report.display_name)
        current_page_url = cls._clean(report.current_page_url)
        event_type = cls._normalize_event_type(report.event_type)

        if not tablet_uid:
            return TabletUserSessionReportResult(
                success=False,
                message="tablet_uid is required.",
                event_type=event_type,
            )

        device_id = cls._get_or_create_tablet_device(
            db_session=db_session,
            tablet_uid=tablet_uid,
            tablet_name=tablet_name,
            app_version=cls._clean(report.app_version),
            app_version_code=report.app_version_code,
            ip_address=cls._clean(report.ip_address),
            user_agent=cls._clean(report.user_agent),
        )

        if event_type == "logout":
            closed_count = cls._close_active_sessions_for_device(
                db_session=db_session,
                tablet_device_id=device_id,
                logout_reason="tablet_reported_logout",
            )

            return TabletUserSessionReportResult(
                success=True,
                message=f"Tablet logout reported. Closed active sessions: {closed_count}",
                tablet_device_id=device_id,
                active_session_id=None,
                tablet_uid=tablet_uid,
                tablet_name=tablet_name,
                username=None,
                display_name=None,
                event_type=event_type,
                is_active=False,
            )

        if not username:
            active_session = cls._get_active_session_for_device(
                db_session=db_session,
                tablet_device_id=device_id,
            )

            if active_session:
                cls._touch_tablet_user_session(
                    db_session=db_session,
                    session_id=active_session["id"],
                    current_page_url=current_page_url,
                    ip_address=cls._clean(report.ip_address),
                    user_agent=cls._clean(report.user_agent),
                )

                return TabletUserSessionReportResult(
                    success=True,
                    message="Tablet heartbeat received. Existing active user session touched.",
                    tablet_device_id=device_id,
                    active_session_id=active_session["id"],
                    tablet_uid=tablet_uid,
                    tablet_name=tablet_name,
                    username=active_session.get("username"),
                    display_name=active_session.get("display_name"),
                    event_type=event_type,
                    is_active=True,
                )

            return TabletUserSessionReportResult(
                success=True,
                message="Tablet heartbeat received. No user reported as logged in.",
                tablet_device_id=device_id,
                active_session_id=None,
                tablet_uid=tablet_uid,
                tablet_name=tablet_name,
                username=None,
                display_name=None,
                event_type=event_type,
                is_active=False,
            )

        active_session = cls._get_active_session_for_device(
            db_session=db_session,
            tablet_device_id=device_id,
        )

        if active_session and active_session.get("username") == username:
            cls._touch_tablet_user_session(
                db_session=db_session,
                session_id=active_session["id"],
                current_page_url=current_page_url,
                ip_address=cls._clean(report.ip_address),
                user_agent=cls._clean(report.user_agent),
                display_name=display_name,
            )

            return TabletUserSessionReportResult(
                success=True,
                message="Tablet user session heartbeat updated.",
                tablet_device_id=device_id,
                active_session_id=active_session["id"],
                tablet_uid=tablet_uid,
                tablet_name=tablet_name,
                username=username,
                display_name=display_name or active_session.get("display_name"),
                event_type=event_type,
                is_active=True,
            )

        if active_session:
            cls._close_active_sessions_for_device(
                db_session=db_session,
                tablet_device_id=device_id,
                logout_reason="tablet_reported_user_switch",
            )

        new_session_id = cls._create_tablet_user_session(
            db_session=db_session,
            tablet_device_id=device_id,
            username=username,
            display_name=display_name,
            user_id=report.user_id,
            current_page_url=current_page_url,
            ip_address=cls._clean(report.ip_address),
            user_agent=cls._clean(report.user_agent),
        )

        return TabletUserSessionReportResult(
            success=True,
            message="Tablet user session created from tablet report.",
            tablet_device_id=device_id,
            active_session_id=new_session_id,
            tablet_uid=tablet_uid,
            tablet_name=tablet_name,
            username=username,
            display_name=display_name,
            event_type=event_type,
            is_active=True,
        )

    @classmethod
    def _get_or_create_tablet_device(
        cls,
        db_session,
        tablet_uid: str,
        tablet_name: Optional[str],
        app_version: Optional[str],
        app_version_code: Optional[int],
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> int:
        existing = db_session.execute(
            text(
                """
                SELECT id
                FROM tablet_edge.tablet_device
                WHERE tablet_uid = :tablet_uid
                LIMIT 1
                """
            ),
            {"tablet_uid": tablet_uid},
        ).mappings().first()

        if existing:
            device_id = int(existing["id"])
            cls._update_tablet_device(
                db_session=db_session,
                tablet_device_id=device_id,
                tablet_uid=tablet_uid,
                tablet_name=tablet_name,
                app_version=app_version,
                app_version_code=app_version_code,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            return device_id

        device_columns = cls._get_table_columns(
            db_session=db_session,
            schema_name=cls.DEVICE_SCHEMA,
            table_name=cls.DEVICE_TABLE,
        )

        values = {
            "tablet_uid": tablet_uid,
            "tablet_name": tablet_name,
            "device_name": tablet_name,
            "app_version": app_version,
            "app_version_code": app_version_code,
            "last_ip_address": ip_address,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "is_active": True,
        }

        timestamp_columns = {
            "created_at": "NOW()",
            "updated_at": "NOW()",
            "last_seen_at": "NOW()",
        }

        insert_columns = []
        insert_values = []
        params = {}

        for column_name, column_value in values.items():
            if column_name in device_columns and column_value is not None:
                insert_columns.append(column_name)
                insert_values.append(f":{column_name}")
                params[column_name] = column_value

        for column_name, sql_value in timestamp_columns.items():
            if column_name in device_columns:
                insert_columns.append(column_name)
                insert_values.append(sql_value)

        if "tablet_uid" not in insert_columns:
            raise RuntimeError(
                "tablet_edge.tablet_device must have a tablet_uid column."
            )

        sql = f"""
            INSERT INTO tablet_edge.tablet_device (
                {", ".join(insert_columns)}
            )
            VALUES (
                {", ".join(insert_values)}
            )
            RETURNING id
        """

        created = db_session.execute(text(sql), params).mappings().first()

        if not created:
            raise RuntimeError("Failed to create tablet_device row.")

        return int(created["id"])

    @classmethod
    def _update_tablet_device(
        cls,
        db_session,
        tablet_device_id: int,
        tablet_uid: str,
        tablet_name: Optional[str],
        app_version: Optional[str],
        app_version_code: Optional[int],
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> None:
        device_columns = cls._get_table_columns(
            db_session=db_session,
            schema_name=cls.DEVICE_SCHEMA,
            table_name=cls.DEVICE_TABLE,
        )

        values = {
            "tablet_uid": tablet_uid,
            "tablet_name": tablet_name,
            "device_name": tablet_name,
            "app_version": app_version,
            "app_version_code": app_version_code,
            "last_ip_address": ip_address,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "is_active": True,
        }

        timestamp_values = {
            "last_seen_at": "NOW()",
            "updated_at": "NOW()",
        }

        set_clauses = []
        params = {"tablet_device_id": tablet_device_id}

        for column_name, column_value in values.items():
            if column_name in device_columns and column_value is not None:
                set_clauses.append(f"{column_name} = :{column_name}")
                params[column_name] = column_value

        for column_name, sql_value in timestamp_values.items():
            if column_name in device_columns:
                set_clauses.append(f"{column_name} = {sql_value}")

        if not set_clauses:
            return

        sql = f"""
            UPDATE tablet_edge.tablet_device
            SET {", ".join(set_clauses)}
            WHERE id = :tablet_device_id
        """

        db_session.execute(text(sql), params)

    @classmethod
    def _get_active_session_for_device(
        cls,
        db_session,
        tablet_device_id: int,
    ) -> Optional[Dict]:
        row = db_session.execute(
            text(
                """
                SELECT
                    id,
                    username,
                    display_name
                FROM tablet_edge.tablet_user_session
                WHERE tablet_device_id = :tablet_device_id
                  AND is_active = TRUE
                ORDER BY last_seen_at DESC NULLS LAST, login_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            ),
            {"tablet_device_id": tablet_device_id},
        ).mappings().first()

        return dict(row) if row else None

    @classmethod
    def _close_active_sessions_for_device(
        cls,
        db_session,
        tablet_device_id: int,
        logout_reason: str,
    ) -> int:
        session_columns = cls._get_table_columns(
            db_session=db_session,
            schema_name=cls.SESSION_SCHEMA,
            table_name=cls.SESSION_TABLE,
        )

        set_clauses = [
            "is_active = FALSE",
        ]

        params = {
            "tablet_device_id": tablet_device_id,
            "logout_reason": logout_reason,
        }

        if "logout_at" in session_columns:
            set_clauses.append("logout_at = COALESCE(logout_at, NOW())")

        if "logout_reason" in session_columns:
            set_clauses.append("logout_reason = :logout_reason")

        if "updated_at" in session_columns:
            set_clauses.append("updated_at = NOW()")

        sql = f"""
            UPDATE tablet_edge.tablet_user_session
            SET {", ".join(set_clauses)}
            WHERE tablet_device_id = :tablet_device_id
              AND is_active = TRUE
        """

        result = db_session.execute(text(sql), params)
        return int(result.rowcount or 0)

    @classmethod
    def _touch_tablet_user_session(
        cls,
        db_session,
        session_id: int,
        current_page_url: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        display_name: Optional[str] = None,
    ) -> None:
        session_columns = cls._get_table_columns(
            db_session=db_session,
            schema_name=cls.SESSION_SCHEMA,
            table_name=cls.SESSION_TABLE,
        )

        set_clauses = []
        params = {"session_id": session_id}

        if "last_seen_at" in session_columns:
            set_clauses.append("last_seen_at = NOW()")

        if "updated_at" in session_columns:
            set_clauses.append("updated_at = NOW()")

        if "current_page_url" in session_columns and current_page_url:
            set_clauses.append("current_page_url = :current_page_url")
            params["current_page_url"] = current_page_url

        if "last_ip_address" in session_columns and ip_address:
            set_clauses.append("last_ip_address = :last_ip_address")
            params["last_ip_address"] = ip_address

        if "login_ip_address" in session_columns and ip_address:
            set_clauses.append("login_ip_address = COALESCE(login_ip_address, :login_ip_address)")
            params["login_ip_address"] = ip_address

        if "user_agent" in session_columns and user_agent:
            set_clauses.append("user_agent = :user_agent")
            params["user_agent"] = user_agent

        if "display_name" in session_columns and display_name:
            set_clauses.append("display_name = :display_name")
            params["display_name"] = display_name

        if not set_clauses:
            return

        sql = f"""
            UPDATE tablet_edge.tablet_user_session
            SET {", ".join(set_clauses)}
            WHERE id = :session_id
        """

        db_session.execute(text(sql), params)

    @classmethod
    def _create_tablet_user_session(
        cls,
        db_session,
        tablet_device_id: int,
        username: str,
        display_name: Optional[str],
        user_id: Optional[int],
        current_page_url: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> int:
        session_columns = cls._get_table_columns(
            db_session=db_session,
            schema_name=cls.SESSION_SCHEMA,
            table_name=cls.SESSION_TABLE,
        )

        values = {
            "tablet_device_id": tablet_device_id,
            "username": username,
            "display_name": display_name,
            "user_id": user_id,
            "current_page_url": current_page_url,
            "login_ip_address": ip_address,
            "last_ip_address": ip_address,
            "user_agent": user_agent,
            "is_active": True,
        }

        timestamp_columns = {
            "login_at": "NOW()",
            "last_seen_at": "NOW()",
            "created_at": "NOW()",
            "updated_at": "NOW()",
        }

        insert_columns = []
        insert_values = []
        params = {}

        for column_name, column_value in values.items():
            if column_name in session_columns and column_value is not None:
                insert_columns.append(column_name)
                insert_values.append(f":{column_name}")
                params[column_name] = column_value

        for column_name, sql_value in timestamp_columns.items():
            if column_name in session_columns:
                insert_columns.append(column_name)
                insert_values.append(sql_value)

        required_columns = {"tablet_device_id", "username", "is_active"}

        missing_required = [
            column_name
            for column_name in required_columns
            if column_name not in insert_columns
        ]

        if missing_required:
            raise RuntimeError(
                "tablet_edge.tablet_user_session is missing required columns: "
                + ", ".join(missing_required)
            )

        sql = f"""
            INSERT INTO tablet_edge.tablet_user_session (
                {", ".join(insert_columns)}
            )
            VALUES (
                {", ".join(insert_values)}
            )
            RETURNING id
        """

        created = db_session.execute(text(sql), params).mappings().first()

        if not created:
            raise RuntimeError("Failed to create tablet_user_session row.")

        return int(created["id"])

    @classmethod
    def _get_table_columns(
        cls,
        db_session,
        schema_name: str,
        table_name: str,
    ) -> Set[str]:
        rows = db_session.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = :table_name
                """
            ),
            {
                "schema_name": schema_name,
                "table_name": table_name,
            },
        ).mappings().all()

        return {row["column_name"] for row in rows}

    @staticmethod
    def _clean(value) -> Optional[str]:
        if value is None:
            return None

        cleaned = str(value).strip()

        if not cleaned:
            return None

        return cleaned

    @staticmethod
    def _normalize_event_type(event_type: Optional[str]) -> str:
        cleaned = (event_type or "heartbeat").strip().lower()

        allowed = {
            "login",
            "logout",
            "heartbeat",
            "page_view",
            "user_changed",
        }

        if cleaned not in allowed:
            return "heartbeat"

        return cleaned