from __future__ import annotations

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
        chat_session_payload = service.get_or_create_user_chat_session()
        session_uuid = str(chat_session_payload["session_uuid"])

        session_payload, messages_payload = service.get_messages_for_session(
            session_uuid=session_uuid
        )

        return jsonify({
            "success": True,
            "session": session_payload,
            "messages": messages_payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to get/create help chat session.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/api/messages", methods=["GET"])
def get_current_session_messages():
    service = HelpChatService()

    try:
        chat_session_payload = service.get_or_create_user_chat_session()
        session_uuid = str(chat_session_payload["session_uuid"])

        session_payload, messages_payload = service.get_messages_for_session(
            session_uuid=session_uuid
        )

        return jsonify({
            "success": True,
            "session": session_payload,
            "messages": messages_payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load help chat messages.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/api/messages", methods=["POST"])
def post_user_message():
    service = HelpChatService()

    try:
        data = request.get_json(silent=True) or {}

        message_text = str(data.get("message_text") or "").strip()
        current_page = str(data.get("current_page") or request.referrer or "").strip()

        if not message_text:
            return jsonify({
                "success": False,
                "message": "Message text is required.",
            }), 400

        chat_session_payload = service.get_or_create_user_chat_session()
        session_uuid = str(chat_session_payload["session_uuid"])

        message_payload, session_payload = service.save_user_message(
            session_uuid=session_uuid,
            message_text=message_text,
            current_page=current_page,
        )

        socketio = current_app.extensions.get("socketio")
        if socketio:
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

        return jsonify({
            "success": True,
            "session": session_payload,
            "message": message_payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to save help chat message.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/api/upload", methods=["POST"])
def upload_file():
    service = HelpChatService()

    try:
        chat_session_payload = service.get_or_create_user_chat_session()
        session_uuid = str(chat_session_payload["session_uuid"])
        help_chat_session_id = int(chat_session_payload["id"])
        display_name = str(chat_session_payload.get("display_name") or "EMTAC User")

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "message": "No file part provided.",
            }), 400

        file = request.files["file"]
        message_text = request.form.get("message_text", "").strip()

        if file.filename == "":
            return jsonify({
                "success": False,
                "message": "No file selected.",
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": "File type not allowed.",
            }), 400

        original_name = file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{uuid4().hex}_{safe_name}"

        upload_path = get_upload_folder() / unique_name
        file.save(upload_path)

        db_config = DatabaseConfig()
        db_session = db_config.get_main_session()

        try:
            msg = HelpChatMessage(
                help_chat_session_id=help_chat_session_id,
                sender_type="user",
                sender_name=display_name,
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
                room=f"help_chat_{session_uuid}",
            )
            socketio.emit(
                "help_chat_admin_new_message",
                payload,
                room="help_chat_admins",
            )

        return jsonify({
            "success": True,
            "message": payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to upload help chat attachment.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(get_upload_folder(), filename)


@help_chat_bp.route("/admin/api/sessions", methods=["GET"])
def admin_api_sessions():
    if not admin_allowed():
        return jsonify({
            "success": False,
            "message": "Unauthorized.",
        }), 401

    service = HelpChatService()

    try:
        chat_sessions = service.get_recent_sessions(limit=25)

        return jsonify({
            "success": True,
            "sessions": chat_sessions,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load admin help chat sessions.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/admin/api/session/<session_uuid>/messages", methods=["GET"])
def admin_api_session_messages(session_uuid: str):
    if not admin_allowed():
        return jsonify({
            "success": False,
            "message": "Unauthorized.",
        }), 401

    service = HelpChatService()

    try:
        session_payload, messages_payload = service.get_messages_for_session(
            session_uuid=session_uuid
        )

        return jsonify({
            "success": True,
            "session": session_payload,
            "messages": messages_payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to load admin help chat messages.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500


@help_chat_bp.route("/admin/api/session/<session_uuid>/status", methods=["POST"])
def admin_api_session_status(session_uuid: str):
    if not admin_allowed():
        return jsonify({
            "success": False,
            "message": "Unauthorized.",
        }), 401

    service = HelpChatService()

    try:
        data = request.get_json(silent=True) or {}
        status = str(data.get("status") or "").strip().lower()

        session_payload = service.update_status(
            session_uuid=session_uuid,
            status=status,
        )

        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "help_chat_session_updated",
                session_payload,
                room="help_chat_admins",
            )

        return jsonify({
            "success": True,
            "session": session_payload,
        })

    except Exception as exc:
        logger.exception("[HELP_CHAT] Failed to update help chat status.")
        return jsonify({
            "success": False,
            "message": str(exc),
        }), 500