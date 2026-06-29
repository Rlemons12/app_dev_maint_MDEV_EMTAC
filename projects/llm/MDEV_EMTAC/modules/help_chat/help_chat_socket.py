from __future__ import annotations

import logging
from typing import Any, Optional

from flask import session
from flask_socketio import emit, join_room, leave_room

from modules.services.help_chat.help_chat_service import HelpChatService

logger = logging.getLogger(__name__)


def _admin_allowed() -> bool:
    user_level = str(session.get("user_level") or "").upper()
    return user_level in {"ADMIN", "LEVEL_III"} or bool(session.get("is_admin"))


def _get_payload_session_uuid(payload: Optional[dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None

    session_uuid = payload.get("session_uuid")
    if session_uuid:
        return str(session_uuid)

    return None


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
                session_payload = service.get_or_create_user_chat_session()
                session_uuid = _get_payload_session_uuid(session_payload)

            if not session_uuid:
                emit("help_chat_error", {"message": "Session UUID is required."})
                return

            room_name = f"help_chat_{session_uuid}"
            join_room(room_name)

            session_payload = service.mark_online(
                session_uuid=session_uuid,
                online=True,
            )

            emit(
                "help_chat_joined_user_room",
                {
                    "success": True,
                    "session_uuid": session_uuid,
                    "room": room_name,
                },
            )

            if session_payload:
                socketio.emit(
                    "help_chat_session_updated",
                    session_payload,
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

            session_uuid = str(session_uuid)
            room_name = f"help_chat_{session_uuid}"

            leave_room(room_name)

            session_payload = service.mark_online(
                session_uuid=session_uuid,
                online=False,
            )

            emit(
                "help_chat_left_user_room",
                {
                    "success": True,
                    "session_uuid": session_uuid,
                    "room": room_name,
                },
            )

            if session_payload:
                socketio.emit(
                    "help_chat_session_updated",
                    session_payload,
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
                session_payload = service.get_or_create_user_chat_session()
                session_uuid = _get_payload_session_uuid(session_payload)

            if not session_uuid:
                emit("help_chat_error", {"message": "Session UUID is required."})
                return

            if not message_text:
                emit("help_chat_error", {"message": "Message text is required."})
                return

            message_payload, session_payload = service.save_user_message(
                session_uuid=str(session_uuid),
                message_text=message_text,
                current_page=current_page,
            )

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

        emit(
            "help_chat_admin_joined",
            {
                "success": True,
                "room": "help_chat_admins",
            },
        )

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

            first_name = str(session.get("first_name") or "").strip()
            last_name = str(session.get("last_name") or "").strip()
            sender_name = " ".join(
                part for part in [first_name, last_name] if part
            ).strip() or "Admin"

            message_payload, session_payload = service.save_admin_message(
                session_uuid=str(session_uuid),
                message_text=message_text,
                sender_name=sender_name,
            )

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

            message_payload, session_payload = service.save_system_message(
                session_uuid=str(session_uuid),
                message_text=message_text,
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
            logger.exception("[HELP_CHAT] Failed to send admin notice.")
            emit("help_chat_error", {"message": str(exc)})