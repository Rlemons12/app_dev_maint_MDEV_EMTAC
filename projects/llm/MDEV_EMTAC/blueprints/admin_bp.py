import json
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from sqlalchemy.orm import subqueryload
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.orm import subqueryload, joinedload
from modules.configuration import config
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger
from modules.ai.config.models_config import ModelsConfig
from modules.emtacdb.emtacdb_fts import User, UserComments, UserLevel, UserLogin


admin_bp = Blueprint("admin_bp", __name__)


@admin_bp.route("/admin_dashboard")
def admin_dashboard():
    logger.debug("Admin dashboard accessed by user: %s", session.get("user_id"))

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning("Unauthorized access attempt by user: %s", session.get("user_id"))
        flash("You do not have permission to access this page.", "error")
        return redirect(url_for("login_bp.login"))

    session_db = None

    try:
        db_config = DatabaseConfig()
        session_db = db_config.get_main_session()

        logger.debug("Fetching users, comments, and active sessions from the database.")

        users = session_db.query(User).all()
        comments = session_db.query(UserComments).options(
            subqueryload(UserComments.user)
        ).all()

        user_map = {}
        for user in users:
            user_map[str(user.id)] = {
                "id": user.id,
                "employee_id": user.employee_id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "user_level": user.user_level.name,
            }
            user_map[user.employee_id] = user_map[str(user.id)]

        current_time = datetime.now()
        time_threshold = current_time - timedelta(hours=6)

        active_logins = (
            session_db.query(UserLogin)
            .options(joinedload(UserLogin.user))
            .join(User)
            .filter(UserLogin.is_active == True)
            .filter(UserLogin.last_activity >= time_threshold)
            .order_by(UserLogin.last_activity.desc())
            .all()
        )

        current_ai_model = ModelsConfig.get_config_value("ai", "CURRENT_MODEL", "NoAIModel")
        current_embedding_model = ModelsConfig.get_config_value("embedding", "CURRENT_MODEL", "NoEmbeddingModel")
        current_image_model = ModelsConfig.get_config_value("image", "CURRENT_MODEL", "NoImageModel")

        logger.info(
            "Fetched %d users, %d comments, and %d active logins.",
            len(users),
            len(comments),
            len(active_logins),
        )

    except Exception as e:
        logger.error("Error loading admin dashboard data: %s", e, exc_info=True)
        flash(f"Error loading dashboard: {e}", "error")
        return redirect(url_for("admin_bp.admin_dashboard"))

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return render_template(
        "admin_dashboard.html",
        users=users,
        comments=comments,
        active_sessions=active_logins,
        user_map=user_map,
        current_ai_model=current_ai_model,
        current_embedding_model=current_embedding_model,
        current_image_model=current_image_model,
        current_time=current_time,
    )


@admin_bp.route("/reset_user_password", methods=["POST"])
def reset_user_password():
    logger.debug("Password reset requested by admin user: %s", session.get("user_id"))

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning(
            "Unauthorized password reset attempt by user: %s",
            session.get("user_id"),
        )
        flash("You do not have permission to perform this action.", "error")
        return redirect(url_for("login_bp.login"))

    user_id = request.form.get("user_id")
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")

    if not user_id:
        flash("User ID is required.", "error")
        return redirect(url_for("admin_bp.admin_dashboard"))

    if not new_password or not confirm_password:
        flash("New password and confirmation are required.", "error")
        return redirect(url_for("admin_bp.admin_dashboard"))

    if new_password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("admin_bp.admin_dashboard"))

    if len(new_password) < 8:
        flash("Password must be at least 8 characters long.", "error")
        return redirect(url_for("admin_bp.admin_dashboard"))

    session_db = None

    try:
        db_config = DatabaseConfig()
        session_db = db_config.get_main_session()

        user = session_db.query(User).filter_by(id=user_id).first()

        if not user:
            flash("User not found.", "error")
            logger.warning("Password reset failed. User not found: %s", user_id)
            return redirect(url_for("admin_bp.admin_dashboard"))

        user.hashed_password = generate_password_hash(new_password)
        user.must_change_password = True
        user.password_last_changed = datetime.now()

        session_db.commit()

        flash(
            f"Temporary password set for {user.first_name} {user.last_name}. "
            "User must change it at next login.",
            "success",
        )

        logger.info(
            "Temporary password reset for user_id=%s by admin_user_id=%s",
            user_id,
            session.get("user_id"),
        )

    except Exception as e:
        if session_db:
            session_db.rollback()

        logger.error("Error resetting user password: %s", e, exc_info=True)
        flash(f"Error resetting password: {e}", "error")

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return redirect(url_for("admin_bp.admin_dashboard"))


@admin_bp.route("/change_password", methods=["GET", "POST"])
def change_password():
    if not session.get("user_id"):
        flash("Please log in first.", "error")
        return redirect(url_for("login_bp.login"))

    if request.method == "GET":
        return render_template("change_password.html")

    current_password = request.form.get("current_password")
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")

    if not current_password or not new_password or not confirm_password:
        flash("All password fields are required.", "error")
        return redirect(url_for("admin_bp.change_password"))

    if new_password != confirm_password:
        flash("New passwords do not match.", "error")
        return redirect(url_for("admin_bp.change_password"))

    if len(new_password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("admin_bp.change_password"))

    session_db = None

    try:
        db_config = DatabaseConfig()
        session_db = db_config.get_main_session()

        user = session_db.query(User).filter_by(id=session.get("user_id")).first()

        if not user:
            flash("User not found.", "error")
            return redirect(url_for("login_bp.login"))

        if not check_password_hash(user.hashed_password, current_password):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("admin_bp.change_password"))

        user.hashed_password = generate_password_hash(new_password)
        user.must_change_password = False
        user.password_last_changed = datetime.now()

        session.pop("force_password_change", None)

        session_db.commit()

        flash("Password changed successfully.", "success")
        return redirect(url_for("admin_bp.admin_dashboard"))

    except Exception as e:
        if session_db:
            session_db.rollback()

        logger.error("Error changing password: %s", e, exc_info=True)
        flash(f"Error changing password: {e}", "error")
        return redirect(url_for("admin_bp.change_password"))

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")


@admin_bp.route("/change_user_level", methods=["POST"])
def change_user_level():
    logger.debug("User level change requested by user: %s", session.get("user_id"))

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning(
            "Unauthorized attempt to change user level by user: %s",
            session.get("user_id"),
        )
        flash("You do not have permission to perform this action.", "error")
        return redirect(url_for("login_bp.login"))

    user_id = request.form["user_id"]
    new_user_level = request.form["user_level"]
    session_db = None

    try:
        db_config = DatabaseConfig()
        session_db = db_config.get_main_session()

        logger.debug("Fetching user with ID: %s", user_id)

        user = session_db.query(User).filter_by(id=user_id).first()

        if user:
            user.user_level = UserLevel(new_user_level)
            session_db.commit()

            flash("User level updated successfully.", "success")
            logger.info("User level changed for user: %s to %s", user_id, new_user_level)
        else:
            flash("User not found.", "error")
            logger.warning("User not found for user ID: %s", user_id)

    except Exception as e:
        if session_db:
            session_db.rollback()

        logger.error("Error changing user level: %s", e, exc_info=True)
        flash(f"Error changing user level: {e}", "error")

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return redirect(url_for("admin_bp.admin_dashboard"))


@admin_bp.route("/set_models", methods=["POST"])
def set_models():
    logger.debug("Model change requested by user: %s", session.get("user_id"))

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning(
            "Unauthorized attempt to change models by user: %s",
            session.get("user_id"),
        )
        flash("You do not have permission to perform this action.", "error")
        return redirect(url_for("login_bp.login"))

    ai_model = request.form["ai_model"]
    embedding_model = request.form["embedding_model"]
    image_model = request.form["image_model"]

    try:
        config.CURRENT_AI_MODEL = ai_model
        config.CURRENT_EMBEDDING_MODEL = embedding_model
        config.CURRENT_IMAGE_MODEL = image_model

        logger.debug(
            "New model configurations: AI Model: %s, Embedding Model: %s, Image Model: %s",
            ai_model,
            embedding_model,
            image_model,
        )

        ModelsConfig.set_current_ai_model(ai_model)
        ModelsConfig.set_current_embedding_model(embedding_model)
        ModelsConfig.set_current_image_model(image_model)

        logger.info("Model configurations updated successfully.")
        flash("Models updated successfully.", "success")

    except Exception as e:
        logger.error("Error updating models: %s", e, exc_info=True)
        flash(f"Error updating models: {e}", "error")

    return redirect(url_for("admin_bp.admin_dashboard"))


@admin_bp.route("/get_available_models", methods=["GET"])
def get_available_models():
    logger.debug("Fetching available models for admin dashboard")

    try:
        ai_models = ModelsConfig.get_available_models("ai")
        embedding_models = ModelsConfig.get_available_models("embedding")
        image_models = ModelsConfig.get_available_models("image")

        current_ai = ModelsConfig.get_config_value("ai", "CURRENT_MODEL", "NoAIModel")
        current_embedding = ModelsConfig.get_config_value("embedding", "CURRENT_MODEL", "NoEmbeddingModel")
        current_image = ModelsConfig.get_config_value("image", "CURRENT_MODEL", "NoImageModel")

        return jsonify({
            "status": "success",
            "models": {
                "ai": ai_models,
                "embedding": embedding_models,
                "image": image_models,
            },
            "current": {
                "ai": current_ai,
                "embedding": current_embedding,
                "image": current_image,
            },
        })

    except Exception as e:
        logger.error("Error fetching available models: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@admin_bp.route("/register_model", methods=["POST"])
def register_model():
    if session.get("user_level") != UserLevel.ADMIN.name:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json or {}

    required = {"model_type", "name", "label", "backend"}

    if not required.issubset(data):
        return jsonify({"error": "Missing required fields"}), 400

    success = ModelsConfig.register_available_model(
        model_type=data["model_type"],
        model_info={
            "name": data["name"],
            "label": data.get("label", data["name"]),
            "backend": data.get("backend", "gpu_service"),
            "enabled": data.get("enabled", True),
            "context_window": data.get("context_window"),
        },
    )

    return jsonify({"success": success})


@admin_bp.route("/add_model", methods=["POST"])
def add_model():
    logger.debug("Add model requested by user: %s", session.get("user_id"))

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning(
            "Unauthorized attempt to add model by user: %s",
            session.get("user_id"),
        )
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    data = request.get_json(silent=True) or {}

    model_type = (data.get("model_type") or "").strip()
    name = (data.get("name") or "").strip()
    label = (data.get("label") or name).strip()
    path = (data.get("path") or "").strip()
    backend = (data.get("backend") or "gpu_service").strip()
    gpu_key = (data.get("gpu_key") or "").strip()
    enabled = bool(data.get("enabled", True))

    if not model_type or not name:
        return jsonify({
            "status": "error",
            "message": "model_type and name are required",
        }), 400

    try:
        models = ModelsConfig.get_available_models(model_type)

        for model in models:
            if isinstance(model, dict) and model.get("name") == name:
                return jsonify({
                    "status": "error",
                    "message": f"Model '{name}' already exists",
                }), 409

        entry = {
            "name": name,
            "label": label,
            "enabled": enabled,
            "backend": backend,
        }

        if path:
            entry["path"] = path

        if gpu_key:
            entry["gpu_key"] = gpu_key

        models.append(entry)

        ModelsConfig.set_config_value(
            model_type,
            "available_models",
            json.dumps(models),
        )

        return jsonify({
            "status": "success",
            "added": entry,
            "total": len(models),
        })

    except Exception as e:
        logger.error("Error adding model: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@admin_bp.route("/promote_latest_trained_model", methods=["POST"])
def promote_latest_trained_model():
    logger.debug(
        "Promote latest trained model requested by user: %s",
        session.get("user_id"),
    )

    if session.get("user_level") != UserLevel.ADMIN.name:
        logger.warning(
            "Unauthorized promote attempt by user: %s",
            session.get("user_id"),
        )
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    try:
        success = ModelsConfig.auto_promote_latest_trained_model()

        if success:
            return jsonify({
                "status": "success",
                "message": "Latest trained model promoted successfully.",
            })

        return jsonify({
            "status": "error",
            "message": "Failed to promote latest trained model.",
        }), 500

    except Exception as e:
        logger.error("Error promoting latest trained model: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500