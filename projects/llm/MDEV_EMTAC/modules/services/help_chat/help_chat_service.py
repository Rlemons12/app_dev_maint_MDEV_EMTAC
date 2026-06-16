from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from flask import request, session, url_for

from modules.configuration.config_env import DatabaseConfig
from modules.help_chat.help_chat_models import HelpChatMessage, HelpChatSession

logger = logging.getLogger(__name__)


class HelpChatService:
    def __init__(self) -> None:
        self.db_config = DatabaseConfig()

    def get_session(self):
        return self.db_config.get_main_session()

    @staticmethod
    def get_client_ip() -> Optional[str]:
        x_forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()

        x_real_ip = request.headers.get("X-Real-IP", "").strip()
        if x_real_ip:
            return x_real_ip

        return request.remote_addr

    @classmethod
    def get_ip_details(cls) -> dict[str, Optional[str]]:
        return {
            "client_ip": cls.get_client_ip(),
            "remote_addr": request.remote_addr,
            "x_forwarded_for": request.headers.get("X-Forwarded-For"),
            "x_real_ip": request.headers.get("X-Real-IP"),
        }

    @staticmethod
    def get_display_name_from_flask_session() -> str:
        first_name = str(session.get("first_name") or "").strip()
        last_name = str(session.get("last_name") or "").strip()
        employee_id = str(session.get("employee_id") or "").strip()
        user_id = str(session.get("user_id") or "").strip()

        display_name = " ".join(part for part in [first_name, last_name] if part).strip()

        if display_name:
            return display_name

        if employee_id:
            return employee_id

        if user_id:
            return f"User {user_id}"

        return "EMTAC User"

    @staticmethod
    def get_current_page_from_request() -> Optional[str]:
        json_data = request.get_json(silent=True) or {}

        current_page = (
            request.args.get("page")
            or request.form.get("page")
            or json_data.get("current_page")
            or json_data.get("page")
            or request.referrer
        )

        if current_page:
            return str(current_page).strip()[:500]

        return None

    @staticmethod
    def update_session_network_info(chat_session: HelpChatSession) -> None:
        ip_details = HelpChatService.get_ip_details()

        chat_session.user_identifier = ip_details["client_ip"]
        chat_session.client_ip = ip_details["client_ip"]
        chat_session.remote_addr = ip_details["remote_addr"]
        chat_session.x_forwarded_for = ip_details["x_forwarded_for"]
        chat_session.x_real_ip = ip_details["x_real_ip"]

    def get_or_create_user_chat_session(self) -> dict[str, Any]:
        db_session = self.get_session()

        try:
            session_uuid = session.get("help_chat_session_uuid")

            if session_uuid:
                existing = (
                    db_session.query(HelpChatSession)
                    .filter(HelpChatSession.session_uuid == session_uuid)
                    .first()
                )

                if existing:
                    self.update_session_network_info(existing)
                    existing.display_name = self.get_display_name_from_flask_session()
                    existing.user_id = str(session.get("user_id") or "") or None
                    existing.employee_id = str(session.get("employee_id") or "") or None
                    existing.user_level = str(session.get("user_level") or "") or None

                    current_page = self.get_current_page_from_request()
                    if current_page:
                        existing.current_page = current_page

                    existing.last_seen = datetime.utcnow()
                    existing.updated_at = datetime.utcnow()

                    db_session.commit()
                    db_session.refresh(existing)

                    return self.serialize_chat_session(existing)

            new_uuid = str(uuid4())
            ip_details = self.get_ip_details()

            chat_session = HelpChatSession(
                session_uuid=new_uuid,
                display_name=self.get_display_name_from_flask_session(),
                user_identifier=ip_details["client_ip"],
                user_id=str(session.get("user_id") or "") or None,
                employee_id=str(session.get("employee_id") or "") or None,
                user_level=str(session.get("user_level") or "") or None,
                client_ip=ip_details["client_ip"],
                remote_addr=ip_details["remote_addr"],
                x_forwarded_for=ip_details["x_forwarded_for"],
                x_real_ip=ip_details["x_real_ip"],
                status="open",
                current_page=self.get_current_page_from_request(),
                is_online=False,
                last_seen=datetime.utcnow(),
            )

            db_session.add(chat_session)
            db_session.commit()
            db_session.refresh(chat_session)

            session["help_chat_session_uuid"] = new_uuid

            return self.serialize_chat_session(chat_session)

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to get or create help chat session.")
            raise
        finally:
            db_session.close()

    def save_user_message(
        self,
        *,
        session_uuid: str,
        message_text: str,
        current_page: Optional[str] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._save_message(
            session_uuid=session_uuid,
            sender_type="user",
            sender_name=None,
            message_text=message_text,
            current_page=current_page,
        )

    def save_admin_message(
        self,
        *,
        session_uuid: str,
        message_text: str,
        sender_name: str = "Admin",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._save_message(
            session_uuid=session_uuid,
            sender_type="admin",
            sender_name=sender_name,
            message_text=message_text,
            current_page=None,
        )

    def save_system_message(
        self,
        *,
        session_uuid: str,
        message_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._save_message(
            session_uuid=session_uuid,
            sender_type="system",
            sender_name="System",
            message_text=message_text,
            current_page=None,
        )

    def _save_message(
        self,
        *,
        session_uuid: str,
        sender_type: str,
        sender_name: Optional[str],
        message_text: str,
        current_page: Optional[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        db_session = self.get_session()

        try:
            chat_session = (
                db_session.query(HelpChatSession)
                .filter(HelpChatSession.session_uuid == session_uuid)
                .first()
            )

            if not chat_session:
                raise ValueError("Help chat session not found.")

            final_sender_name = sender_name
            if sender_type == "user":
                final_sender_name = chat_session.display_name

            msg = HelpChatMessage(
                help_chat_session_id=chat_session.id,
                sender_type=sender_type,
                sender_name=final_sender_name,
                message_type="text",
                message_text=message_text.strip(),
            )

            db_session.add(msg)

            if current_page:
                chat_session.current_page = current_page[:500]

            if sender_type == "user":
                self.update_session_network_info(chat_session)
                chat_session.last_seen = datetime.utcnow()

            chat_session.updated_at = datetime.utcnow()

            db_session.commit()
            db_session.refresh(msg)
            db_session.refresh(chat_session)

            message_payload = self.serialize_message(msg)
            session_payload = self.serialize_chat_session(chat_session)

            return message_payload, session_payload

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to save help chat message.")
            raise
        finally:
            db_session.close()

    def mark_online(self, *, session_uuid: str, online: bool) -> Optional[dict[str, Any]]:
        db_session = self.get_session()

        try:
            chat_session = (
                db_session.query(HelpChatSession)
                .filter(HelpChatSession.session_uuid == session_uuid)
                .first()
            )

            if not chat_session:
                return None

            self.update_session_network_info(chat_session)
            chat_session.is_online = online
            chat_session.last_seen = datetime.utcnow()
            chat_session.updated_at = datetime.utcnow()

            db_session.commit()
            db_session.refresh(chat_session)

            return self.serialize_chat_session(chat_session)

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to update online state.")
            raise
        finally:
            db_session.close()

    def update_status(self, *, session_uuid: str, status: str) -> dict[str, Any]:
        db_session = self.get_session()

        try:
            normalized_status = status.strip().lower()

            if normalized_status not in {"open", "closed", "waiting"}:
                raise ValueError("Invalid status.")

            chat_session = (
                db_session.query(HelpChatSession)
                .filter(HelpChatSession.session_uuid == session_uuid)
                .first()
            )

            if not chat_session:
                raise ValueError("Help chat session not found.")

            chat_session.status = normalized_status
            chat_session.updated_at = datetime.utcnow()

            db_session.commit()
            db_session.refresh(chat_session)

            return self.serialize_chat_session(chat_session)

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to update help chat status.")
            raise
        finally:
            db_session.close()

    def get_recent_sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        db_session = self.get_session()

        try:
            sessions = (
                db_session.query(HelpChatSession)
                .order_by(HelpChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )

            return [
                self.serialize_chat_session(item)
                for item in sessions
            ]

        finally:
            db_session.close()

    def get_messages_for_session(
        self,
        *,
        session_uuid: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        db_session = self.get_session()

        try:
            chat_session = (
                db_session.query(HelpChatSession)
                .filter(HelpChatSession.session_uuid == session_uuid)
                .first()
            )

            if not chat_session:
                raise ValueError("Help chat session not found.")

            messages = list(chat_session.messages)

            session_payload = self.serialize_chat_session(chat_session)
            message_payloads = [
                self.serialize_message(message)
                for message in messages
            ]

            return session_payload, message_payloads

        finally:
            db_session.close()

    @staticmethod
    def serialize_message(message: HelpChatMessage) -> dict[str, Any]:
        return {
            "id": message.id,
            "help_chat_session_id": message.help_chat_session_id,
            "sender_type": message.sender_type,
            "sender_name": message.sender_name,
            "message_type": message.message_type,
            "message_text": message.message_text,
            "attachment_filename": message.attachment_filename,
            "attachment_original_name": message.attachment_original_name,
            "attachment_url": (
                url_for("help_chat_bp.uploaded_file", filename=message.attachment_filename)
                if message.attachment_filename
                else None
            ),
            "created_at": (
                message.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if message.created_at
                else None
            ),
        }

    @staticmethod
    def serialize_chat_session(chat_session: HelpChatSession) -> dict[str, Any]:
        try:
            messages = list(getattr(chat_session, "messages", []) or [])
        except Exception:
            messages = []

        last_message = messages[-1] if messages else None

        return {
            "id": chat_session.id,
            "session_uuid": chat_session.session_uuid,
            "display_name": chat_session.display_name,
            "user_identifier": chat_session.user_identifier,
            "user_id": chat_session.user_id,
            "employee_id": chat_session.employee_id,
            "user_level": chat_session.user_level,
            "status": chat_session.status,
            "current_page": chat_session.current_page,
            "client_ip": chat_session.client_ip,
            "remote_addr": chat_session.remote_addr,
            "x_forwarded_for": chat_session.x_forwarded_for,
            "x_real_ip": chat_session.x_real_ip,
            "last_seen": (
                chat_session.last_seen.strftime("%Y-%m-%d %H:%M:%S")
                if chat_session.last_seen
                else None
            ),
            "is_online": chat_session.is_online,
            "created_at": (
                chat_session.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if chat_session.created_at
                else None
            ),
            "updated_at": (
                chat_session.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                if chat_session.updated_at
                else None
            ),
            "last_message_preview": (
                last_message.message_text[:80]
                if last_message and last_message.message_text
                else (
                    f"[file] {last_message.attachment_original_name}"
                    if last_message and last_message.attachment_original_name
                    else ""
                )
            ),
        }