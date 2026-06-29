"""
Tablet user session tracking service.

File:
    modules/services/tablet_edge/tablet_user_session_service.py

Purpose:
    Tracks which EMTAC web user is logged into which physical EMTAC tablet.

This service writes to:

    tablet_edge.tablet_user_session

Main responsibilities:
    - Resolve a tablet device from tablet_uid or tablet_device_id.
    - Start a tablet user session after successful login.
    - Update last_seen/current_page_url while the tablet is active.
    - End a tablet user session during logout.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.emtacdb.tablet_edge.tablet_edge_models import (
    TabletDevice,
    TabletUserSession,
)

logger = logging.getLogger(__name__)


class TabletUserSessionService:
    """
    Service for tracking EMTAC user sessions on physical tablets.
    """

    @staticmethod
    def _normalize_tablet_uid(tablet_uid: Optional[str]) -> Optional[UUID]:
        """
        Convert a tablet UID string into a UUID object.

        Returns None if the value is missing or invalid.
        """
        if not tablet_uid:
            return None

        try:
            return UUID(str(tablet_uid).strip())
        except ValueError:
            logger.warning(
                "Invalid tablet_uid received for tablet user session tracking: %r",
                tablet_uid,
            )
            return None

    @classmethod
    def resolve_tablet_device(
        cls,
        db_session: Session,
        tablet_uid: Optional[str] = None,
        tablet_device_id: Optional[int] = None,
    ) -> Optional[TabletDevice]:
        """
        Resolve a TabletDevice by tablet_device_id or tablet_uid.

        tablet_device_id is preferred when available.
        tablet_uid is used when the Android/WebView side sends the stable UUID.
        """
        if tablet_device_id is not None:
            return (
                db_session.query(TabletDevice)
                .filter(TabletDevice.id == tablet_device_id)
                .one_or_none()
            )

        normalized_uid = cls._normalize_tablet_uid(tablet_uid)

        if normalized_uid is None:
            return None

        return (
            db_session.query(TabletDevice)
            .filter(TabletDevice.tablet_uid == normalized_uid)
            .one_or_none()
        )

    @classmethod
    def start_user_session(
        cls,
        db_session: Session,
        tablet_uid: Optional[str],
        user_id: Optional[int],
        username: Optional[str],
        display_name: Optional[str] = None,
        session_id: Optional[str] = None,
        login_ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        current_page_url: Optional[str] = None,
        close_existing_for_tablet: bool = True,
    ) -> Optional[TabletUserSession]:
        """
        Start a new active user session for a tablet.

        This should be called immediately after a successful EMTAC login.

        If close_existing_for_tablet=True, any existing active session for this
        tablet is marked inactive before creating the new one. That prevents
        multiple users from appearing active on the same tablet.
        """
        tablet_device = cls.resolve_tablet_device(
            db_session=db_session,
            tablet_uid=tablet_uid,
        )

        if tablet_device is None:
            logger.warning(
                "Could not start tablet user session because tablet was not found. "
                "tablet_uid=%r username=%r user_id=%r",
                tablet_uid,
                username,
                user_id,
            )
            return None

        if close_existing_for_tablet:
            (
                db_session.query(TabletUserSession)
                .filter(
                    TabletUserSession.tablet_device_id == tablet_device.id,
                    TabletUserSession.is_active.is_(True),
                )
                .update(
                    {
                        TabletUserSession.is_active: False,
                        TabletUserSession.logout_at: func.now(),
                        TabletUserSession.logout_reason: "replaced_by_new_login",
                        TabletUserSession.updated_at: func.now(),
                    },
                    synchronize_session=False,
                )
            )

        tablet_user_session = TabletUserSession(
            tablet_device_id=tablet_device.id,
            user_id=user_id,
            username=username,
            display_name=display_name,
            session_id=session_id,
            login_ip_address=login_ip_address,
            last_ip_address=login_ip_address,
            user_agent=user_agent,
            current_page_url=current_page_url,
            is_active=True,
        )

        db_session.add(tablet_user_session)
        db_session.flush()

        logger.info(
            "Started tablet user session. "
            "tablet_device_id=%s tablet_uid=%s username=%s user_id=%s session_id=%s",
            tablet_device.id,
            tablet_device.tablet_uid,
            username,
            user_id,
            session_id,
        )

        return tablet_user_session

    @classmethod
    def touch_active_session(
        cls,
        db_session: Session,
        tablet_uid: Optional[str] = None,
        tablet_device_id: Optional[int] = None,
        session_id: Optional[str] = None,
        current_page_url: Optional[str] = None,
        last_ip_address: Optional[str] = None,
    ) -> Optional[TabletUserSession]:
        """
        Update last_seen/current page for the active tablet user session.

        This can be called from:
            - tablet heartbeat route
            - page activity route
            - any tablet-edge request where tablet identity is known
        """
        query = db_session.query(TabletUserSession).filter(
            TabletUserSession.is_active.is_(True)
        )

        if session_id:
            query = query.filter(TabletUserSession.session_id == session_id)
        else:
            tablet_device = cls.resolve_tablet_device(
                db_session=db_session,
                tablet_uid=tablet_uid,
                tablet_device_id=tablet_device_id,
            )

            if tablet_device is None:
                return None

            query = query.filter(
                TabletUserSession.tablet_device_id == tablet_device.id
            )

        tablet_user_session = (
            query.order_by(TabletUserSession.last_seen_at.desc())
            .first()
        )

        if tablet_user_session is None:
            return None

        tablet_user_session.last_seen_at = func.now()
        tablet_user_session.updated_at = func.now()

        if current_page_url:
            tablet_user_session.current_page_url = current_page_url

        if last_ip_address:
            tablet_user_session.last_ip_address = last_ip_address

        db_session.flush()

        return tablet_user_session

    @classmethod
    def end_user_session(
        cls,
        db_session: Session,
        session_id: Optional[str] = None,
        tablet_uid: Optional[str] = None,
        tablet_device_id: Optional[int] = None,
        logout_reason: str = "user_logout",
    ) -> int:
        """
        End active user sessions.

        Preferred matching order:
            1. session_id
            2. tablet_device_id
            3. tablet_uid

        Returns the number of rows updated.
        """
        query = db_session.query(TabletUserSession).filter(
            TabletUserSession.is_active.is_(True)
        )

        if session_id:
            query = query.filter(TabletUserSession.session_id == session_id)
        else:
            tablet_device = cls.resolve_tablet_device(
                db_session=db_session,
                tablet_uid=tablet_uid,
                tablet_device_id=tablet_device_id,
            )

            if tablet_device is None:
                logger.warning(
                    "Could not end tablet user session because tablet was not found. "
                    "tablet_uid=%r tablet_device_id=%r session_id=%r",
                    tablet_uid,
                    tablet_device_id,
                    session_id,
                )
                return 0

            query = query.filter(
                TabletUserSession.tablet_device_id == tablet_device.id
            )

        updated_count = query.update(
            {
                TabletUserSession.is_active: False,
                TabletUserSession.logout_at: func.now(),
                TabletUserSession.logout_reason: logout_reason,
                TabletUserSession.updated_at: func.now(),
            },
            synchronize_session=False,
        )

        logger.info(
            "Ended tablet user session(s). count=%s session_id=%s tablet_uid=%s "
            "tablet_device_id=%s logout_reason=%s",
            updated_count,
            session_id,
            tablet_uid,
            tablet_device_id,
            logout_reason,
        )

        return updated_count