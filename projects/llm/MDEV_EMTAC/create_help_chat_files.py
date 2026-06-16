from pathlib import Path

ROOT = Path.cwd()

FILES = {
    "modules/help_chat/__init__.py": "",

    "modules/help_chat/help_chat_models.py": r'''from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from modules.emtacdb.emtacdb_fts import Base


class HelpChatSession(Base):
    __tablename__ = "help_chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="EMTAC User")
    user_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    employee_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    user_level: Mapped[str | None] = mapped_column(String(100), nullable=True)

    client_ip: Mapped[str | None] = mapped_column(String(255), nullable=True)
    remote_addr: Mapped[str | None] = mapped_column(String(255), nullable=True)
    x_forwarded_for: Mapped[str | None] = mapped_column(String(500), nullable=True)
    x_real_ip: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="open")
    current_page: Mapped[str | None] = mapped_column(String(500), nullable=True)

    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_online: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    messages: Mapped[list["HelpChatMessage"]] = relationship(
        "HelpChatMessage",
        back_populates="help_chat_session",
        cascade="all, delete-orphan",
        order_by="HelpChatMessage.created_at.asc()",
    )


class HelpChatMessage(Base):
    __tablename__ = "help_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    help_chat_session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("help_chat_sessions.id"),
        nullable=False,
        index=True,
    )

    sender_type: Mapped[str] = mapped_column(String(50), nullable=False)
    sender_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    message_type: Mapped[str] = mapped_column(String(50), nullable=False, default="text")
    message_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    attachment_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attachment_original_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attachment_mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    help_chat_session: Mapped[HelpChatSession] = relationship(
        "HelpChatSession",
        back_populates="messages",
    )
''',

    "modules/services/help_chat/__init__.py": "",

    "modules/services/help_chat/help_chat_service.py": r'''from __future__ import annotations

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

    def get_or_create_user_chat_session(self) -> HelpChatSession:
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
                    db_session.expunge(existing)
                    return existing

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

            db_session.expunge(chat_session)
            return chat_session

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
    ) -> tuple[HelpChatMessage, HelpChatSession]:
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
    ) -> tuple[HelpChatMessage, HelpChatSession]:
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
    ) -> tuple[HelpChatMessage, HelpChatSession]:
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
    ) -> tuple[HelpChatMessage, HelpChatSession]:
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

            chat_session.messages

            db_session.expunge(msg)
            db_session.expunge(chat_session)

            return msg, chat_session

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to save help chat message.")
            raise
        finally:
            db_session.close()

    def mark_online(self, *, session_uuid: str, online: bool) -> Optional[HelpChatSession]:
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
            chat_session.messages

            db_session.expunge(chat_session)
            return chat_session

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to update online state.")
            raise
        finally:
            db_session.close()

    def update_status(self, *, session_uuid: str, status: str) -> HelpChatSession:
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
            chat_session.messages

            db_session.expunge(chat_session)
            return chat_session

        except Exception:
            db_session.rollback()
            logger.exception("[HELP_CHAT] Failed to update help chat status.")
            raise
        finally:
            db_session.close()

    def get_recent_sessions(self, *, limit: int = 100) -> list[HelpChatSession]:
        db_session = self.get_session()

        try:
            sessions = (
                db_session.query(HelpChatSession)
                .order_by(HelpChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )

            for item in sessions:
                item.messages
                db_session.expunge(item)

            return sessions

        finally:
            db_session.close()

    def get_messages_for_session(self, *, session_uuid: str) -> tuple[HelpChatSession, list[HelpChatMessage]]:
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

            for message in messages:
                db_session.expunge(message)

            db_session.expunge(chat_session)
            return chat_session, messages

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
        last_message = chat_session.messages[-1] if chat_session.messages else None

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
''',

    "blueprints/help_chat/__init__.py": "",

    "blueprints/help_chat/help_chat_bp.py": r'''from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
)
from werkzeug.utils import secure_filename

from modules.configuration.config_env import DatabaseConfig
from modules.help_chat.help_chat_models import HelpChatMessage
from modules.services.help_chat.help_chat_service import HelpChatService

logger = logging.getLogger(__name__)

help_chat_bp = Blueprint(
    "help_chat_bp",
    __name__,
    url_prefix="/help-chat",
)

DEFAULT_ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "gif", "webp", "bmp",
    "pdf", "txt", "log",
}


def admin_allowed() -> bool:
    user_level = str(session.get("user_level") or "").upper()
    return user_level in {"ADMIN", "LEVEL_III"} or bool(session.get("is_admin"))


def get_upload_folder() -> Path:
    configured_folder = current_app.config.get("HELP_CHAT_UPLOAD_FOLDER")

    if configured_folder:
        upload_folder = Path(configured_folder)
    else:
        upload_folder = Path(current_app.instance_path) / "help_chat_uploads"

    upload_folder.mkdir(parents=True, exist_ok=True)
    return upload_folder


def get_allowed_extensions() -> set[str]:
    configured = current_app.config.get("HELP_CHAT_ALLOWED_EXTENSIONS")

    if configured:
        return {str(item).lower().lstrip(".") for item in configured}

    return DEFAULT_ALLOWED_EXTENSIONS


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()
    return ext in get_allowed_extensions()


@help_chat_bp.route("/admin", methods=["GET"])
def help_chat_admin_page():
    if not admin_allowed():
        return "Unauthorized", 401

    return render_template("help_chat/help_chat_admin.html")


@help_chat_bp.route("/api/session", methods=["GET", "POST"])
def get_or_create_session():
    service = HelpChatService()

    try:
        chat_session = service.get_or_create_user_chat_session()
        _, messages = service.get_messages_for_session(session_uuid=chat_session.session_uuid)

        return jsonify({
            "success": True,
            "session": service.serialize_chat_session(chat_session),
            "messages": [service.serialize_message(message) for message in messages],
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to get/create help chat session.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/api/messages", methods=["GET"])
def get_current_session_messages():
    service = HelpChatService()

    try:
        chat_session = service.get_or_create_user_chat_session()
        _, messages = service.get_messages_for_session(session_uuid=chat_session.session_uuid)

        return jsonify({
            "success": True,
            "session": service.serialize_chat_session(chat_session),
            "messages": [service.serialize_message(message) for message in messages],
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load help chat messages.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/api/messages", methods=["POST"])
def post_user_message():
    service = HelpChatService()

    try:
        data = request.get_json(silent=True) or {}

        message_text = str(data.get("message_text") or "").strip()
        current_page = str(data.get("current_page") or request.referrer or "").strip()

        if not message_text:
            return jsonify({"success": False, "message": "Message text is required."}), 400

        chat_session = service.get_or_create_user_chat_session()

        message, updated_session = service.save_user_message(
            session_uuid=chat_session.session_uuid,
            message_text=message_text,
            current_page=current_page,
        )

        payload = service.serialize_message(message)
        session_payload = service.serialize_chat_session(updated_session)

        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "help_chat_new_message",
                payload,
                room=f"help_chat_{updated_session.session_uuid}",
            )
            socketio.emit(
                "help_chat_admin_new_message",
                payload,
                room="help_chat_admins",
            )
            socketio.emit(
                "help_chat_session_updated",
                session_payload,
                room="help_chat_admins",
            )

        return jsonify({
            "success": True,
            "session": session_payload,
            "message": payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to save help chat message.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/api/upload", methods=["POST"])
def upload_file():
    service = HelpChatService()

    try:
        chat_session = service.get_or_create_user_chat_session()

        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file part provided."}), 400

        file = request.files["file"]
        message_text = request.form.get("message_text", "").strip()

        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "File type not allowed."}), 400

        original_name = file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid4().hex}_{safe_name}"

        upload_path = get_upload_folder() / unique_name
        file.save(upload_path)

        db_config = DatabaseConfig()
        db_session = db_config.get_main_session()

        try:
            msg = HelpChatMessage(
                help_chat_session_id=chat_session.id,
                sender_type="user",
                sender_name=chat_session.display_name,
                message_type="file",
                message_text=message_text or None,
                attachment_filename=unique_name,
                attachment_original_name=original_name,
                attachment_mime_type=file.mimetype,
            )

            db_session.add(msg)
            db_session.commit()
            db_session.refresh(msg)

            payload = service.serialize_message(msg)

        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "help_chat_new_message",
                payload,
                room=f"help_chat_{chat_session.session_uuid}",
            )
            socketio.emit(
                "help_chat_admin_new_message",
                payload,
                room="help_chat_admins",
            )

        return jsonify({"success": True, "message": payload})

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to upload help chat attachment.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(get_upload_folder(), filename)


@help_chat_bp.route("/admin/api/sessions", methods=["GET"])
def admin_api_sessions():
    if not admin_allowed():
        return jsonify({"success": False, "message": "Unauthorized."}), 401

    service = HelpChatService()

    try:
        chat_sessions = service.get_recent_sessions(limit=200)

        return jsonify({
            "success": True,
            "sessions": [
                service.serialize_chat_session(chat_session)
                for chat_session in chat_sessions
            ],
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load admin help chat sessions.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/admin/api/session/<session_uuid>/messages", methods=["GET"])
def admin_api_session_messages(session_uuid: str):
    if not admin_allowed():
        return jsonify({"success": False, "message": "Unauthorized."}), 401

    service = HelpChatService()

    try:
        chat_session, messages = service.get_messages_for_session(session_uuid=session_uuid)

        return jsonify({
            "success": True,
            "session": service.serialize_chat_session(chat_session),
            "messages": [service.serialize_message(message) for message in messages],
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load admin help chat messages.")
        return jsonify({"success": False, "message": str(exc)}), 500


@help_chat_bp.route("/admin/api/session/<session_uuid>/status", methods=["POST"])
def admin_api_session_status(session_uuid: str):
    if not admin_allowed():
        return jsonify({"success": False, "message": "Unauthorized."}), 401

    service = HelpChatService()

    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get("status") or "").strip().lower()

        chat_session = service.update_status(session_uuid=session_uuid, status=status)
        payload = service.serialize_chat_session(chat_session)

        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "help_chat_session_updated",
                payload,
                room="help_chat_admins",
            )

        return jsonify({"success": True, "session": payload})

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to update help chat status.")
        return jsonify({"success": False, "message": str(exc)}), 500
''',

    "modules/help_chat/help_chat_socket.py": r'''from __future__ import annotations

import logging

from flask import session
from flask_socketio import emit, join_room, leave_room

from modules.services.help_chat.help_chat_service import HelpChatService

logger = logging.getLogger(__name__)


def _admin_allowed() -> bool:
    user_level = str(session.get("user_level") or "").upper()
    return user_level in {"ADMIN", "LEVEL_III"} or bool(session.get("is_admin"))


def register_help_chat_socket_events(socketio) -> None:
    @socketio.on("help_chat_connect_check")
    def help_chat_connect_check():
        emit("help_chat_connected", {"success": True})

    @socketio.on("help_chat_join_user_room")
    def help_chat_join_user_room(data):
        service = HelpChatService()

        try:
            data = data or {}
            session_uuid = data.get("session_uuid")

            if not session_uuid:
                chat_session = service.get_or_create_user_chat_session()
                session_uuid = chat_session.session_uuid

            room_name = f"help_chat_{session_uuid}"
            join_room(room_name)

            chat_session = service.mark_online(session_uuid=session_uuid, online=True)

            emit("help_chat_joined_user_room", {
                "success": True,
                "session_uuid": session_uuid,
                "room": room_name,
            })

            if chat_session:
                socketio.emit(
                    "help_chat_session_updated",
                    service.serialize_chat_session(chat_session),
                    room="help_chat_admins",
                )

        except Exception as exc:
            logger.exception("[HELP_CHAT] Failed to join user room.")
            emit("help_chat_error", {"message": str(exc)})

    @socketio.on("help_chat_leave_user_room")
    def help_chat_leave_user_room(data):
        service = HelpChatService()

        try:
            data = data or {}
            session_uuid = data.get("session_uuid")

            if not session_uuid:
                return

            room_name = f"help_chat_{session_uuid}"
            leave_room(room_name)

            chat_session = service.mark_online(session_uuid=session_uuid, online=False)

            emit("help_chat_left_user_room", {
                "success": True,
                "session_uuid": session_uuid,
                "room": room_name,
            })

            if chat_session:
                socketio.emit(
                    "help_chat_session_updated",
                    service.serialize_chat_session(chat_session),
                    room="help_chat_admins",
                )

        except Exception as exc:
            logger.exception("[HELP_CHAT] Failed to leave user room.")
            emit("help_chat_error", {"message": str(exc)})

    @socketio.on("help_chat_user_send_message")
    def help_chat_user_send_message(data):
        service = HelpChatService()

        try:
            data = data or {}
            session_uuid = data.get("session_uuid")
            message_text = str(data.get("message_text") or "").strip()
            current_page = str(data.get("current_page") or "").strip()

            if not session_uuid:
                chat_session = service.get_or_create_user_chat_session()
                session_uuid = chat_session.session_uuid

            if not message_text:
                emit("help_chat_error", {"message": "Message text is required."})
                return

            message, chat_session = service.save_user_message(
                session_uuid=session_uuid,
                message_text=message_text,
                current_page=current_page,
            )

            message_payload = service.serialize_message(message)
            session_payload = service.serialize_chat_session(chat_session)

            socketio.emit(
                "help_chat_new_message",
                message_payload,
                room=f"help_chat_{session_uuid}",
            )

            socketio.emit(
                "help_chat_admin_new_message",
                message_payload,
                room="help_chat_admins",
            )

            socketio.emit(
                "help_chat_session_updated",
                session_payload,
                room="help_chat_admins",
            )

        except Exception as exc:
            logger.exception("[HELP_CHAT] Failed to send user message.")
            emit("help_chat_error", {"message": str(exc)})

    @socketio.on("help_chat_admin_join")
    def help_chat_admin_join():
        if not _admin_allowed():
            emit("help_chat_error", {"message": "Unauthorized admin action."})
            return

        join_room("help_chat_admins")
        emit("help_chat_admin_joined", {"success": True, "room": "help_chat_admins"})

    @socketio.on("help_chat_admin_send_message")
    def help_chat_admin_send_message(data):
        if not _admin_allowed():
            emit("help_chat_error", {"message": "Unauthorized admin action."})
            return

        service = HelpChatService()

        try:
            data = data or {}
            session_uuid = data.get("session_uuid")
            message_text = str(data.get("message_text") or "").strip()

            if not session_uuid or not message_text:
                emit("help_chat_error", {"message": "Missing session or message text."})
                return

            sender_name = str(session.get("first_name") or "Admin").strip() or "Admin"

            message, chat_session = service.save_admin_message(
                session_uuid=session_uuid,
                message_text=message_text,
                sender_name=sender_name,
            )

            message_payload = service.serialize_message(message)
            session_payload = service.serialize_chat_session(chat_session)

            socketio.emit(
                "help_chat_admin_message_sent",
                message_payload,
                room="help_chat_admins",
            )

            socketio.emit(
                "help_chat_new_message",
                message_payload,
                room=f"help_chat_{session_uuid}",
            )

            socketio.emit(
                "help_chat_session_updated",
                session_payload,
                room="help_chat_admins",
            )

        except Exception as exc:
            logger.exception("[HELP_CHAT] Failed to send admin message.")
            emit("help_chat_error", {"message": str(exc)})

    @socketio.on("help_chat_admin_notice")
    def help_chat_admin_notice(data):
        if not _admin_allowed():
            emit("help_chat_error", {"message": "Unauthorized admin action."})
            return

        service = HelpChatService()

        try:
            data = data or {}
            session_uuid = data.get("session_uuid")
            message_text = str(data.get("message_text") or "").strip()

            if not session_uuid or not message_text:
                emit("help_chat_error", {"message": "Missing session or notice text."})
                return

            message, chat_session = service.save_system_message(
                session_uuid=session_uuid,
                message_text=message_text,
            )

            message_payload = service.serialize_message(message)
            session_payload = service.serialize_chat_session(chat_session)

            socketio.emit(
                "help_chat_new_message",
                message_payload,
                room=f"help_chat_{session_uuid}",
            )

            socketio.emit(
                "help_chat_session_updated",
                session_payload,
                room="help_chat_admins",
            )

        except Exception as exc:
            logger.exception("[HELP_CHAT] Failed to send admin notice.")
            emit("help_chat_error", {"message": str(exc)})
''',

    "templates/module_template_html/partials/help_chat_popup.html": r'''<div id="helpChatPopup" class="help-chat-popup" style="display: none;">
    <div class="help-chat-header">
        <span>EMTAC Help Chat</span>
        <button id="helpChatCloseButton" type="button">X</button>
    </div>

    <div id="helpChatMessages" class="help-chat-messages"></div>

    <div class="help-chat-input-row">
        <input id="helpChatInput" type="text" placeholder="Ask for help...">
        <button id="helpChatSendButton" type="button">Send</button>
    </div>
</div>
''',

    "templates/help_chat/help_chat_admin.html": r'''{% extends "module_template_html/base_template.html" %}

{% block title %}EMTAC Help Chat Admin{% endblock %}

{% block extra_head %}
<link
    rel="stylesheet"
    href="{{ url_for('static', filename='css/help_chat/help_chat_admin.css') }}?v=help_chat_admin_20260602_1"
>
{% endblock %}

{% block content %}
<div class="help-chat-admin-page">
    <div class="help-chat-admin-header">
        <h1>Help Chat Admin</h1>
        <p>Switch between active user chats using the tabs below.</p>
    </div>

    <div class="help-chat-admin-layout">
        <aside class="help-chat-session-panel">
            <div class="help-chat-session-panel-header">
                <strong>User Chats</strong>
                <button id="helpChatRefreshSessionsButton" type="button">Refresh</button>
            </div>

            <div id="helpChatAdminSessionTabs" class="help-chat-admin-session-tabs">
                <div class="help-chat-admin-empty">No chats loaded.</div>
            </div>
        </aside>

        <section class="help-chat-admin-chat-panel">
            <div id="helpChatAdminActiveHeader" class="help-chat-admin-active-header">
                Select a user chat.
            </div>

            <div id="helpChatAdminMessages" class="help-chat-admin-messages"></div>

            <div class="help-chat-admin-input-row">
                <input
                    id="helpChatAdminInput"
                    type="text"
                    placeholder="Type admin reply..."
                    disabled
                >
                <button id="helpChatAdminSendButton" type="button" disabled>Send</button>
            </div>
        </section>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
{{ super() }}
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script
    src="{{ url_for('static', filename='js/help_chat/help_chat_admin.js') }}?v=help_chat_admin_20260602_1"
></script>
{% endblock %}
''',

    "static/js/help_chat/help_chat_popup.js": r'''document.addEventListener("DOMContentLoaded", function () {
    console.log("[HELP_CHAT] help_chat_popup.js loaded");

    const helpChatLink = document.getElementById("helpChatSidebarLink");
    const helpChatPopup = document.getElementById("helpChatPopup");
    const helpChatCloseButton = document.getElementById("helpChatCloseButton");
    const helpChatMessages = document.getElementById("helpChatMessages");
    const helpChatInput = document.getElementById("helpChatInput");
    const helpChatSendButton = document.getElementById("helpChatSendButton");

    let socket = null;
    let sessionUuid = null;

    if (!helpChatLink || !helpChatPopup || !helpChatMessages || !helpChatInput || !helpChatSendButton) {
        console.warn("[HELP_CHAT] Missing required help chat elements.");
        return;
    }

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function addMessage(sender, text, senderType) {
        const message = document.createElement("div");
        message.className = "help-chat-message help-chat-message-" + (senderType || "system");

        message.innerHTML = `
            <div class="help-chat-message-sender">${escapeHtml(sender)}</div>
            <div class="help-chat-message-text">${escapeHtml(text)}</div>
        `;

        helpChatMessages.appendChild(message);
        helpChatMessages.scrollTop = helpChatMessages.scrollHeight;
    }

    function renderMessages(messages) {
        helpChatMessages.innerHTML = "";

        if (!messages || messages.length === 0) {
            addMessage("EMTAC Help", "How can I help you?", "system");
            return;
        }

        messages.forEach(function (message) {
            addMessage(
                message.sender_name || message.sender_type || "Unknown",
                message.message_text || "",
                message.sender_type || "system"
            );
        });
    }

    async function ensureSession() {
        const response = await fetch("/help-chat/api/session", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                current_page: window.location.href
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || "Failed to create help chat session.");
        }

        sessionUuid = data.session.session_uuid;
        renderMessages(data.messages || []);

        return data.session;
    }

    function ensureSocket() {
        if (socket || typeof io === "undefined") {
            return;
        }

        socket = io();

        socket.on("connect", function () {
            console.log("[HELP_CHAT] Socket connected");

            if (sessionUuid) {
                socket.emit("help_chat_join_user_room", {
                    session_uuid: sessionUuid
                });
            }
        });

        socket.on("help_chat_new_message", function (message) {
            if (!message) {
                return;
            }

            addMessage(
                message.sender_name || message.sender_type || "Unknown",
                message.message_text || "",
                message.sender_type || "system"
            );
        });

        socket.on("help_chat_error", function (payload) {
            console.warn("[HELP_CHAT] Socket error:", payload);
            addMessage("System", payload.message || "Help chat error.", "system");
        });
    }

    async function openHelpChat() {
        helpChatPopup.style.display = "flex";

        try {
            await ensureSession();
            ensureSocket();

            if (socket && socket.connected && sessionUuid) {
                socket.emit("help_chat_join_user_room", {
                    session_uuid: sessionUuid
                });
            }

            helpChatInput.focus();
        } catch (error) {
            console.error("[HELP_CHAT] Failed to open help chat:", error);
            addMessage("System", error.message || "Failed to open help chat.", "system");
        }
    }

    function closeHelpChat() {
        helpChatPopup.style.display = "none";

        if (socket && sessionUuid) {
            socket.emit("help_chat_leave_user_room", {
                session_uuid: sessionUuid
            });
        }
    }

    function sendHelpMessage() {
        const messageText = helpChatInput.value.trim();

        if (!messageText) {
            return;
        }

        if (!sessionUuid) {
            addMessage("System", "Help chat session is not ready yet.", "system");
            return;
        }

        helpChatInput.value = "";

        if (socket && socket.connected) {
            socket.emit("help_chat_user_send_message", {
                session_uuid: sessionUuid,
                message_text: messageText,
                current_page: window.location.href
            });
            return;
        }

        fetch("/help-chat/api/messages", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message_text: messageText,
                current_page: window.location.href
            })
        })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    throw new Error(data.message || "Failed to send message.");
                }

                addMessage(
                    data.message.sender_name || "You",
                    data.message.message_text || "",
                    data.message.sender_type || "user"
                );
            })
            .catch(error => {
                console.error("[HELP_CHAT] Failed to send message:", error);
                addMessage("System", error.message || "Failed to send message.", "system");
            });
    }

    helpChatLink.addEventListener("click", function (event) {
        event.preventDefault();
        openHelpChat();
    });

    if (helpChatCloseButton) {
        helpChatCloseButton.addEventListener("click", closeHelpChat);
    }

    helpChatSendButton.addEventListener("click", sendHelpMessage);

    helpChatInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendHelpMessage();
        }
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && helpChatPopup.style.display !== "none") {
            closeHelpChat();
        }
    });
});
''',

    "static/js/help_chat/help_chat_admin.js": r'''document.addEventListener("DOMContentLoaded", function () {
    console.log("[HELP_CHAT_ADMIN] help_chat_admin.js loaded");

    const tabsContainer = document.getElementById("helpChatAdminSessionTabs");
    const messagesContainer = document.getElementById("helpChatAdminMessages");
    const activeHeader = document.getElementById("helpChatAdminActiveHeader");
    const input = document.getElementById("helpChatAdminInput");
    const sendButton = document.getElementById("helpChatAdminSendButton");
    const refreshButton = document.getElementById("helpChatRefreshSessionsButton");

    let socket = null;
    let activeSessionUuid = null;
    let sessions = [];

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function formatSessionLabel(session) {
        const name = session.display_name || session.employee_id || session.user_id || "Unknown User";
        const status = session.is_online ? "online" : "offline";
        return `${name} (${status})`;
    }

    function renderTabs() {
        tabsContainer.innerHTML = "";

        if (!sessions.length) {
            tabsContainer.innerHTML = '<div class="help-chat-admin-empty">No help chats yet.</div>';
            return;
        }

        sessions.forEach(function (session) {
            const tab = document.createElement("button");
            tab.type = "button";
            tab.className = "help-chat-admin-tab";

            if (session.session_uuid === activeSessionUuid) {
                tab.classList.add("active");
            }

            tab.innerHTML = `
                <div class="help-chat-admin-tab-name">${escapeHtml(formatSessionLabel(session))}</div>
                <div class="help-chat-admin-tab-meta">${escapeHtml(session.current_page || "")}</div>
                <div class="help-chat-admin-tab-preview">${escapeHtml(session.last_message_preview || "")}</div>
            `;

            tab.addEventListener("click", function () {
                loadSessionMessages(session.session_uuid);
            });

            tabsContainer.appendChild(tab);
        });
    }

    function addMessage(message) {
        const item = document.createElement("div");
        item.className = "help-chat-admin-message help-chat-admin-message-" + (message.sender_type || "system");

        item.innerHTML = `
            <div class="help-chat-admin-message-meta">
                <strong>${escapeHtml(message.sender_name || message.sender_type || "Unknown")}</strong>
                <span>${escapeHtml(message.created_at || "")}</span>
            </div>
            <div class="help-chat-admin-message-text">${escapeHtml(message.message_text || "")}</div>
        `;

        messagesContainer.appendChild(item);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function renderMessages(messages) {
        messagesContainer.innerHTML = "";

        if (!messages || !messages.length) {
            messagesContainer.innerHTML = '<div class="help-chat-admin-empty">No messages in this chat yet.</div>';
            return;
        }

        messages.forEach(addMessage);
    }

    async function loadSessions() {
        const response = await fetch("/help-chat/admin/api/sessions");
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || "Failed to load sessions.");
        }

        sessions = data.sessions || [];
        renderTabs();

        if (!activeSessionUuid && sessions.length) {
            await loadSessionMessages(sessions[0].session_uuid);
        }
    }

    async function loadSessionMessages(sessionUuid) {
        activeSessionUuid = sessionUuid;
        renderTabs();

        const response = await fetch(`/help-chat/admin/api/session/${encodeURIComponent(sessionUuid)}/messages`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || "Failed to load messages.");
        }

        activeHeader.innerHTML = `
            <strong>${escapeHtml(data.session.display_name || "Unknown User")}</strong>
            <span>${escapeHtml(data.session.current_page || "")}</span>
        `;

        input.disabled = false;
        sendButton.disabled = false;

        renderMessages(data.messages || []);
    }

    function sendAdminMessage() {
        const messageText = input.value.trim();

        if (!messageText || !activeSessionUuid) {
            return;
        }

        input.value = "";

        socket.emit("help_chat_admin_send_message", {
            session_uuid: activeSessionUuid,
            message_text: messageText
        });
    }

    function ensureSocket() {
        if (socket || typeof io === "undefined") {
            return;
        }

        socket = io();

        socket.on("connect", function () {
            console.log("[HELP_CHAT_ADMIN] Socket connected");
            socket.emit("help_chat_admin_join");
        });

        socket.on("help_chat_admin_new_message", function (message) {
            loadSessions().catch(console.error);

            if (message && message.help_chat_session_id) {
                const active = sessions.find(s => s.session_uuid === activeSessionUuid);
                if (active) {
                    loadSessionMessages(activeSessionUuid).catch(console.error);
                }
            }
        });

        socket.on("help_chat_new_message", function () {
            if (activeSessionUuid) {
                loadSessionMessages(activeSessionUuid).catch(console.error);
            }
        });

        socket.on("help_chat_admin_message_sent", function () {
            if (activeSessionUuid) {
                loadSessionMessages(activeSessionUuid).catch(console.error);
            }

            loadSessions().catch(console.error);
        });

        socket.on("help_chat_session_updated", function () {
            loadSessions().catch(console.error);
        });

        socket.on("help_chat_error", function (payload) {
            console.warn("[HELP_CHAT_ADMIN] Socket error:", payload);
            alert(payload.message || "Help chat admin error.");
        });
    }

    sendButton.addEventListener("click", sendAdminMessage);

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendAdminMessage();
        }
    });

    refreshButton.addEventListener("click", function () {
        loadSessions().catch(console.error);
    });

    ensureSocket();
    loadSessions().catch(console.error);
});
''',

    "static/css/help_chat/help_chat_popup.css": r'''.help-chat-popup {
    position: fixed;
    right: 30px;
    bottom: 30px;
    width: 380px;
    height: 500px;
    background: #111827;
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
    z-index: 99999;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    flex-direction: column;
    overflow: hidden;
}

.help-chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: #1f2937;
    border-bottom: 1px solid #374151;
}

.help-chat-header button {
    background: #374151;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
}

.help-chat-messages {
    flex: 1;
    padding: 12px;
    overflow-y: auto;
    font-size: 14px;
}

.help-chat-message {
    margin-bottom: 10px;
    padding: 8px;
    border-radius: 8px;
    background: #1f2937;
}

.help-chat-message-user {
    background: #164e63;
}

.help-chat-message-admin {
    background: #365314;
}

.help-chat-message-system {
    background: #374151;
}

.help-chat-message-sender {
    font-weight: bold;
    margin-bottom: 4px;
}

.help-chat-input-row {
    display: flex;
    gap: 8px;
    padding: 12px;
    border-top: 1px solid #374151;
    background: #1f2937;
}

#helpChatInput {
    flex: 1;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #4b5563;
}

#helpChatSendButton {
    padding: 8px 12px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
}
''',

    "static/css/help_chat/help_chat_admin.css": r'''.help-chat-admin-page {
    color: #f9fafb;
    padding: 20px;
}

.help-chat-admin-header {
    margin-bottom: 16px;
}

.help-chat-admin-layout {
    display: grid;
    grid-template-columns: 320px minmax(400px, 1fr);
    gap: 16px;
    height: calc(100vh - 220px);
    min-height: 520px;
}

.help-chat-session-panel,
.help-chat-admin-chat-panel {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 10px;
    overflow: hidden;
}

.help-chat-session-panel {
    display: flex;
    flex-direction: column;
}

.help-chat-session-panel-header,
.help-chat-admin-active-header {
    padding: 12px;
    background: #1f2937;
    border-bottom: 1px solid #374151;
}

.help-chat-session-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.help-chat-admin-session-tabs {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.help-chat-admin-tab {
    width: 100%;
    text-align: left;
    background: #1f2937;
    color: #f9fafb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    cursor: pointer;
}

.help-chat-admin-tab.active {
    outline: 2px solid #60a5fa;
}

.help-chat-admin-tab-name {
    font-weight: bold;
}

.help-chat-admin-tab-meta,
.help-chat-admin-tab-preview {
    color: #cbd5e1;
    font-size: 12px;
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.help-chat-admin-chat-panel {
    display: flex;
    flex-direction: column;
}

.help-chat-admin-active-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.help-chat-admin-active-header span {
    color: #cbd5e1;
    font-size: 12px;
}

.help-chat-admin-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
}

.help-chat-admin-message {
    background: #1f2937;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}

.help-chat-admin-message-user {
    background: #164e63;
}

.help-chat-admin-message-admin {
    background: #365314;
}

.help-chat-admin-message-system {
    background: #374151;
}

.help-chat-admin-message-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 12px;
    color: #d1d5db;
}

.help-chat-admin-input-row {
    display: flex;
    gap: 8px;
    padding: 12px;
    background: #1f2937;
    border-top: 1px solid #374151;
}

#helpChatAdminInput {
    flex: 1;
    padding: 10px;
}

#helpChatAdminSendButton,
#helpChatRefreshSessionsButton {
    padding: 8px 12px;
    cursor: pointer;
}

.help-chat-admin-empty {
    color: #cbd5e1;
    padding: 12px;
}
''',

    "sql/create_help_chat_tables.sql": r'''CREATE TABLE IF NOT EXISTS public.help_chat_sessions (
    id SERIAL PRIMARY KEY,
    session_uuid VARCHAR(64) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL DEFAULT 'EMTAC User',
    user_identifier VARCHAR(255),
    user_id VARCHAR(255),
    employee_id VARCHAR(255),
    user_level VARCHAR(100),
    client_ip VARCHAR(255),
    remote_addr VARCHAR(255),
    x_forwarded_for VARCHAR(500),
    x_real_ip VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'open',
    current_page VARCHAR(500),
    last_seen TIMESTAMP,
    is_online BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS public.help_chat_messages (
    id SERIAL PRIMARY KEY,
    help_chat_session_id INTEGER NOT NULL,
    sender_type VARCHAR(50) NOT NULL,
    sender_name VARCHAR(255),
    message_type VARCHAR(50) NOT NULL DEFAULT 'text',
    message_text TEXT,
    attachment_filename VARCHAR(500),
    attachment_original_name VARCHAR(500),
    attachment_mime_type VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_help_chat_messages_session
        FOREIGN KEY (help_chat_session_id)
        REFERENCES public.help_chat_sessions (id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_help_chat_sessions_session_uuid
ON public.help_chat_sessions (session_uuid);

CREATE INDEX IF NOT EXISTS ix_help_chat_sessions_user_id
ON public.help_chat_sessions (user_id);

CREATE INDEX IF NOT EXISTS ix_help_chat_sessions_employee_id
ON public.help_chat_sessions (employee_id);

CREATE INDEX IF NOT EXISTS ix_help_chat_sessions_updated_at
ON public.help_chat_sessions (updated_at DESC);

CREATE INDEX IF NOT EXISTS ix_help_chat_messages_help_chat_session_id
ON public.help_chat_messages (help_chat_session_id);

CREATE INDEX IF NOT EXISTS ix_help_chat_messages_created_at
ON public.help_chat_messages (created_at);
''',
}


def write_file(relative_path: str, content: str, overwrite: bool = False) -> None:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        print(f"[SKIP] Exists: {relative_path}")
        return

    path.write_text(content, encoding="utf-8")
    print(f"[WRITE] {relative_path}")


def main() -> None:
    overwrite = "--overwrite" in {arg.lower() for arg in __import__("sys").argv[1:]}

    print(f"[HELP_CHAT_GENERATOR] Root: {ROOT}")
    print(f"[HELP_CHAT_GENERATOR] Overwrite: {overwrite}")

    for relative_path, content in FILES.items():
        write_file(relative_path, content, overwrite=overwrite)

    print("\nDone.")
    print("\nManual updates still needed:")
    print("1. Add this to your sidebar for all users:")
    print('   <a href="#help-chat" id="helpChatSidebarLink" class="nav-link highlight">Help Chat</a>')
    print("\n2. Add this to your sidebar for ADMIN/LEVEL_III:")
    print('   <a href="/help-chat/admin" class="nav-link highlight">Help Chat Admin</a>')
    print("\n3. Include popup CSS in your base template head:")
    print('   <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/help_chat/help_chat_popup.css\') }}?v=help_chat_20260602_1">')
    print("\n4. Include popup panel in your base template:")
    print('   {% include "module_template_html/partials/help_chat_popup.html" %}')
    print("\n5. Include popup JS in your base template scripts:")
    print('   <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>')
    print('   <script src="{{ url_for(\'static\', filename=\'js/help_chat/help_chat_popup.js\') }}?v=help_chat_20260602_1"></script>')
    print("\n6. Register the blueprint:")
    print("   from blueprints.help_chat.help_chat_bp import help_chat_bp")
    print("   app.register_blueprint(help_chat_bp)")
    print("\n7. Register Socket.IO events where socketio exists:")
    print("   from modules.help_chat.help_chat_socket import register_help_chat_socket_events")
    print("   register_help_chat_socket_events(socketio)")
    print("\n8. Run sql/create_help_chat_tables.sql in DBeaver/PostgreSQL.")


if __name__ == "__main__":
    main()