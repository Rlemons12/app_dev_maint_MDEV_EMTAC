import os
import sys
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

# ---------------------------------------------------------------------
# PATH SETUP
# File location:
#   MDEV_EMTAC/modules/ui_emtac/app.py
#
# Needed import roots:
#   MDEV_EMTAC
#   MDEV_EMTAC/modules
#   MDEV_EMTAC/modules/ui_emtac
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
MODULES_DIR = CURRENT_DIR.parent
PROJECT_ROOT = MODULES_DIR.parent

for path in (PROJECT_ROOT, MODULES_DIR, CURRENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_LAYOUTS = {
    "Default": {
        "Problem / Solution": {
            "x": 10.0,
            "y": 10.0,
            "width": 940.0,
            "height": 929.0,
            "is_expanded": False,
        },
        "Documents": {
            "x": 960.0,
            "y": 489.5,
            "width": 470.0,
            "height": 459.5,
            "is_expanded": False,
        },
        "Parts": {
            "x": 1440.0,
            "y": 489.5,
            "width": 470.0,
            "height": 459.5,
            "is_expanded": False,
        },
        "Images": {
            "x": 960.0,
            "y": 10.0,
            "width": 470.0,
            "height": 459.5,
            "is_expanded": False,
        },
        "Drawings": {
            "x": 1440.0,
            "y": 10.0,
            "width": 470.0,
            "height": 459.5,
            "is_expanded": False,
        },
    }
}


def load_backend():
    """
    Load shared backend dependencies for the Flask UI.

    Import strategy:
    1. Prefer absolute imports through the project package layout.
    2. Fall back where appropriate.
    3. Raise a clear error message if something still fails.
    """
    print("=== load_backend() starting ===")
    print(f"CURRENT_DIR: {CURRENT_DIR}")
    print(f"MODULES_DIR: {MODULES_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # -----------------------------
    # DataService import
    # -----------------------------
    data_service_errors = []
    DataService = None

    try:
        print("Trying import: modules.ui_emtac.data_service")
        from modules.ui_emtac.data_service import DataService
        print("SUCCESS: modules.ui_emtac.data_service")
    except Exception as exc:
        data_service_errors.append(f"modules.ui_emtac.data_service -> {exc}")
        print(f"FAILED: modules.ui_emtac.data_service -> {exc}")
        try:
            print("Trying import: data_service")
            from data_service import DataService
            print("SUCCESS: data_service")
        except Exception as exc2:
            data_service_errors.append(f"data_service -> {exc2}")
            print(f"FAILED: data_service -> {exc2}")

    if DataService is None:
        raise ImportError(
            "Unable to import DataService. Tried:\n- "
            + "\n- ".join(data_service_errors)
        )

    # -----------------------------
    # Configuration imports
    # -----------------------------
    try:
        print("Trying import: modules.configuration.config")
        from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
        print("SUCCESS: modules.configuration.config")
    except Exception as exc:
        print(f"FAILED: modules.configuration.config -> {exc}")
        try:
            print("Trying import: configuration.config")
            from configuration.config import DATABASE_PATH_IMAGES_FOLDER
            print("SUCCESS: configuration.config")
        except Exception as exc2:
            print(f"FAILED: configuration.config -> {exc2}")
            raise ImportError(
                "Unable to import DATABASE_PATH_IMAGES_FOLDER.\n"
                f"modules.configuration.config -> {exc}\n"
                f"configuration.config -> {exc2}"
            ) from exc2

    try:
        print("Trying import: modules.configuration.config_env")
        from modules.configuration.config_env import DatabaseConfig
        print("SUCCESS: modules.configuration.config_env")
    except Exception as exc:
        print(f"FAILED: modules.configuration.config_env -> {exc}")
        try:
            print("Trying import: configuration.config_env")
            from configuration.config_env import DatabaseConfig
            print("SUCCESS: configuration.config_env")
        except Exception as exc2:
            print(f"FAILED: configuration.config_env -> {exc2}")
            raise ImportError(
                "Unable to import DatabaseConfig.\n"
                f"modules.configuration.config_env -> {exc}\n"
                f"configuration.config_env -> {exc2}"
            ) from exc2

    try:
        print("Trying import: modules.configuration.log_config")
        from modules.configuration.log_config import initial_log_cleanup, logger
        print("SUCCESS: modules.configuration.log_config")
    except Exception as exc:
        print(f"FAILED: modules.configuration.log_config -> {exc}")
        try:
            print("Trying import: configuration.log_config")
            from configuration.log_config import initial_log_cleanup, logger
            print("SUCCESS: configuration.log_config")
        except Exception as exc2:
            print(f"FAILED: configuration.log_config -> {exc2}")
            raise ImportError(
                "Unable to import log configuration.\n"
                f"modules.configuration.log_config -> {exc}\n"
                f"configuration.log_config -> {exc2}"
            ) from exc2

    # -----------------------------
    # Database model imports
    # -----------------------------
    try:
        print("Trying import: modules.emtacdb.emtacdb_fts")
        from modules.emtacdb.emtacdb_fts import (
            AssetNumber,
            KivyUser,
            Position,
            Task,
            User,
        )
        print("SUCCESS: modules.emtacdb.emtacdb_fts")
    except Exception as exc:
        print(f"FAILED: modules.emtacdb.emtacdb_fts -> {exc}")
        try:
            print("Trying import: emtacdb.emtacdb_fts")
            from emtacdb.emtacdb_fts import (
                AssetNumber,
                KivyUser,
                Position,
                Task,
                User,
            )
            print("SUCCESS: emtacdb.emtacdb_fts")
        except Exception as exc2:
            print(f"FAILED: emtacdb.emtacdb_fts -> {exc2}")
            raise ImportError(
                "Unable to import database models.\n"
                f"modules.emtacdb.emtacdb_fts -> {exc}\n"
                f"emtacdb.emtacdb_fts -> {exc2}"
            ) from exc2

    # -----------------------------
    # Initialize backend
    # -----------------------------
    print("Running initial_log_cleanup()")
    initial_log_cleanup()

    print("Creating DatabaseConfig session")
    db_session = DatabaseConfig().get_main_session()
    print("SUCCESS: Database session created")

    return {
        "data_service": DataService(db_session),
        "db_session": db_session,
        "DATABASE_PATH_IMAGES_FOLDER": DATABASE_PATH_IMAGES_FOLDER,
        "Task": Task,
        "User": User,
        "KivyUser": KivyUser,
        "AssetNumber": AssetNumber,
        "Position": Position,
        "logger": logger,
    }


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY",
        "maintenance-troubleshooting-dev",
    )

    backend_cache = {"value": None, "error": None}

    def backend():
        if backend_cache["value"] is not None:
            return backend_cache["value"]
        if backend_cache["error"] is not None:
            raise backend_cache["error"]

        try:
            backend_cache["value"] = load_backend()
            return backend_cache["value"]
        except Exception as exc:
            backend_cache["error"] = exc
            raise

    def serialize_basic(obj, fields):
        payload = {}
        for field in fields:
            payload[field] = getattr(obj, field, None)
        return payload

    def parse_int(value):
        if value in (None, "", "null", "undefined"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def require_login(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    def render_backend_error(status_code=503):
        message = (
            "The Flask UI is in place, but the shared maintenance backend is not "
            "importable in this environment. Check the console output from "
            "load_backend() to see which import failed."
        )
        return render_template("login.html", error=message), status_code

    def current_user():
        user_id = session.get("user_id")
        if not user_id:
            return None
        ctx = backend()
        return (
            ctx["db_session"]
            .query(ctx["User"])
            .filter(ctx["User"].id == user_id)
            .first()
        )

    def ensure_layout_user():
        """
        Ensure we can access layout methods from KivyUser, matching the original
        Kivy behavior that attempted to use/ensure a KivyUser for layout storage.
        """
        ctx = backend()
        user = current_user()
        if not user:
            return None

        session_obj = ctx["db_session"]
        KivyUser = ctx["KivyUser"]

        layout_user = session_obj.query(KivyUser).filter(KivyUser.id == user.id).first()
        if layout_user:
            return layout_user

        if hasattr(KivyUser, "ensure_kivy_user"):
            layout_user = KivyUser.ensure_kivy_user(session_obj, user)
            try:
                session_obj.commit()
            except Exception:
                session_obj.rollback()
                raise
            return layout_user

        return None

    def serialize_option(obj):
        label = (
            getattr(obj, "name", None)
            or getattr(obj, "title", None)
            or getattr(obj, "number", None)
            or getattr(obj, "part_number", None)
            or getattr(obj, "drawing_number", None)
            or getattr(obj, "document_number", None)
            or getattr(obj, "employee_id", None)
            or str(getattr(obj, "id", ""))
        )
        return {"id": getattr(obj, "id", None), "label": label}

    def serialize_problem(problem):
        return serialize_basic(problem, ["id", "name", "description"])

    def serialize_solution(solution):
        return serialize_basic(solution, ["id", "name", "description"])

    def serialize_task(task):
        return serialize_basic(
            task,
            ["id", "name", "description", "task_number", "instructions"],
        )

    def serialize_tool(tool):
        return serialize_basic(tool, ["id", "name", "description", "tool_number"])

    def serialize_part(part):
        return serialize_basic(
            part,
            ["id", "part_number", "name", "description", "type", "image_path"],
        )

    def serialize_document(document):
        return serialize_basic(
            document,
            ["id", "title", "description", "file_path", "document_number"],
        )

    def serialize_drawing(drawing):
        return serialize_basic(
            drawing,
            ["id", "title", "description", "file_path", "drawing_number"],
        )

    def serialize_position(position):
        if not position:
            return None
        return {
            "id": getattr(position, "id", None),
            "name": getattr(position, "name", None),
            "description": getattr(position, "description", None),
            "area_id": getattr(position, "area_id", None),
            "equipment_group_id": getattr(position, "equipment_group_id", None),
            "model_id": getattr(position, "model_id", None),
            "asset_number_id": getattr(position, "asset_number_id", None),
            "location_id": getattr(position, "location_id", None),
            "subassembly_id": getattr(position, "subassembly_id", None),
            "component_assembly_id": getattr(position, "component_assembly_id", None),
            "assembly_view_id": getattr(position, "assembly_view_id", None),
        }

    def serialize_asset(asset):
        return {
            "id": getattr(asset, "id", None),
            "number": getattr(asset, "number", None),
            "description": getattr(asset, "description", None),
            "model_id": getattr(asset, "model_id", None),
            "label": (
                f"{getattr(asset, 'number', '')} - "
                f"{getattr(asset, 'description', '') or ''}"
            ).strip(" -"),
        }

    def serialize_layouts(layouts):
        return layouts or {}

    def resolve_position_from_filters(session_obj, filters):
        Position = backend()["Position"]

        query = session_obj.query(Position).distinct()

        if filters.get("area_id"):
            query = query.filter(Position.area_id == filters["area_id"])
        if filters.get("equipment_group_id"):
            query = query.filter(
                Position.equipment_group_id == filters["equipment_group_id"]
            )
        if filters.get("model_id"):
            query = query.filter(Position.model_id == filters["model_id"])
        if filters.get("asset_number_id"):
            query = query.filter(Position.asset_number_id == filters["asset_number_id"])
        if filters.get("location_id"):
            query = query.filter(Position.location_id == filters["location_id"])

        return query.first()

    def load_asset_position(asset):
        """
        Mirrors the Kivy load_asset_position logic:
        1) Try Position.asset_number_id == asset.id
        2) Fallback to AssetNumber.number -> model_id -> Position.model_id
        """
        ctx = backend()
        session_obj = ctx["db_session"]
        Position = ctx["Position"]
        AssetNumber = ctx["AssetNumber"]

        pos = (
            session_obj.query(Position)
            .filter(Position.asset_number_id == asset.id)
            .first()
        )
        if pos:
            return pos

        an = (
            session_obj.query(AssetNumber)
            .filter(AssetNumber.number == asset.number)
            .first()
        )
        if not an:
            return None

        pos = (
            session_obj.query(Position)
            .filter(Position.model_id == an.model_id)
            .first()
        )
        return pos

    def _serialize_image_payload(image):
        if isinstance(image, dict):
            payload = dict(image)
            file_path = payload.get("file_path")
            payload["file_url"] = (
                url_for("media_proxy", kind="image", file_path=file_path)
                if file_path
                else None
            )
            return payload

        payload = serialize_basic(image, ["id", "title", "description", "file_path"])
        payload["file_url"] = (
            url_for("media_proxy", kind="image", file_path=payload["file_path"])
            if payload["file_path"]
            else None
        )
        return payload

    @app.get("/")
    def root():
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            employee_id = request.form.get("employee_id", "").strip()
            password = request.form.get("password", "")

            if not employee_id or not password:
                return render_template(
                    "login.html",
                    error="Please enter both Employee ID and Password",
                )

            try:
                ctx = backend()
                user = (
                    ctx["db_session"]
                    .query(ctx["User"])
                    .filter(ctx["User"].employee_id == employee_id)
                    .first()
                )

                if not user:
                    return render_template("login.html", error="Employee ID not found")

                if not user.check_password_hash(password):
                    return render_template("login.html", error="Invalid password")

                session["user_id"] = user.id
                return redirect(url_for("dashboard"))

            except Exception:
                return render_backend_error()

        return render_template("login.html", error=None)

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/dashboard")
    @require_login
    def dashboard():
        try:
            ctx = backend()
        except Exception:
            return render_backend_error()

        user = current_user()
        areas = [serialize_option(area) for area in ctx["data_service"].get_all_areas()]
        return render_template("dashboard.html", user=user, areas=areas)

    @app.get("/api/options/areas")
    @require_login
    def areas():
        ctx = backend()
        return jsonify(
            [serialize_option(area) for area in ctx["data_service"].get_all_areas()]
        )

    @app.get("/api/options/equipment-groups")
    @require_login
    def equipment_groups():
        ctx = backend()
        area_id = parse_int(request.args.get("area_id"))
        if not area_id:
            return jsonify([])
        return jsonify(
            [
                serialize_option(item)
                for item in ctx["data_service"].get_equipment_groups_by_area(area_id)
            ]
        )

    @app.get("/api/options/models")
    @require_login
    def models():
        ctx = backend()
        equipment_group_id = parse_int(request.args.get("equipment_group_id"))
        if not equipment_group_id:
            return jsonify([])
        return jsonify(
            [
                serialize_option(item)
                for item in ctx["data_service"].get_models_by_equipment_group(
                    equipment_group_id
                )
            ]
        )

    @app.get("/api/options/locations")
    @require_login
    def locations():
        ctx = backend()
        model_id = parse_int(request.args.get("model_id"))
        if not model_id:
            return jsonify([])
        return jsonify(
            [
                serialize_option(item)
                for item in ctx["data_service"].get_locations_by_model(model_id)
            ]
        )

    @app.get("/api/options/asset-numbers")
    @require_login
    def asset_numbers():
        ctx = backend()
        model_id = parse_int(request.args.get("model_id"))
        if not model_id:
            return jsonify([])
        return jsonify(
            [
                serialize_option(item)
                for item in ctx["data_service"].get_asset_numbers_by_model(model_id)
            ]
        )

    @app.get("/api/search/assets")
    @require_login
    def search_assets():
        ctx = backend()
        AssetNumber = ctx["AssetNumber"]
        db_session = ctx["db_session"]

        q = request.args.get("q", "").strip()
        if len(q) < 2:
            return jsonify([])

        if hasattr(AssetNumber, "search_asset_numbers"):
            results = AssetNumber.search_asset_numbers(db_session, q)
            normalized = []
            for item in results:
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "id": item.get("id"),
                            "number": item.get("number"),
                            "description": item.get("description"),
                            "model_id": item.get("model_id"),
                            "label": f"{item.get('number', '')} - {item.get('description', '') or ''}".strip(
                                " -"
                            ),
                        }
                    )
            return jsonify(normalized)

        assets = (
            db_session.query(AssetNumber)
            .filter(AssetNumber.number.ilike(f"%{q}%"))
            .limit(20)
            .all()
        )
        return jsonify([serialize_asset(item) for item in assets])

    @app.get("/api/assets/<int:asset_id>/position")
    @require_login
    def asset_position(asset_id):
        ctx = backend()
        AssetNumber = ctx["AssetNumber"]
        db_session = ctx["db_session"]

        asset = db_session.query(AssetNumber).filter(AssetNumber.id == asset_id).first()
        if not asset:
            abort(404)

        position = load_asset_position(asset)

        return jsonify(
            {
                "asset": serialize_asset(asset),
                "position": serialize_position(position),
            }
        )

    @app.get("/api/positions/current")
    @require_login
    def current_position_from_filters():
        ctx = backend()
        filters = {
            "area_id": parse_int(request.args.get("area_id")),
            "equipment_group_id": parse_int(request.args.get("equipment_group_id")),
            "model_id": parse_int(request.args.get("model_id")),
            "asset_number_id": parse_int(request.args.get("asset_number_id")),
            "location_id": parse_int(request.args.get("location_id")),
        }

        position = resolve_position_from_filters(
            ctx["db_session"],
            filters,
        )

        return jsonify(
            {
                "filters": filters,
                "position": serialize_position(position),
            }
        )

    @app.get("/api/problems")
    @require_login
    def problems():
        ctx = backend()
        filters = {
            "area_id": parse_int(request.args.get("area_id")),
            "equipment_group_id": parse_int(request.args.get("equipment_group_id")),
            "model_id": parse_int(request.args.get("model_id")),
            "asset_number_id": parse_int(request.args.get("asset_number_id")),
            "location_id": parse_int(request.args.get("location_id")),
        }
        items = ctx["data_service"].get_problems_by_filters(**filters)
        return jsonify([serialize_problem(item) for item in items])

    @app.get("/api/positions/<int:position_id>/problems")
    @require_login
    def problems_by_position(position_id):
        ctx = backend()
        items = ctx["data_service"].get_problems_by_position(position_id)
        return jsonify([serialize_problem(item) for item in items])

    @app.get("/api/problems/<int:problem_id>/solutions")
    @require_login
    def problem_solutions(problem_id):
        ctx = backend()
        return jsonify(
            [
                serialize_solution(item)
                for item in ctx["data_service"].get_solutions_by_problem(problem_id)
            ]
        )

    @app.get("/api/solutions/<int:solution_id>/tasks")
    @require_login
    def solution_tasks(solution_id):
        ctx = backend()
        return jsonify(
            [
                serialize_task(item)
                for item in ctx["data_service"].get_tasks_by_solution(solution_id)
            ]
        )

    @app.get("/api/tasks/<int:task_id>")
    @require_login
    def task_details(task_id):
        ctx = backend()

        task = (
            ctx["db_session"]
            .query(ctx["Task"])
            .filter(ctx["Task"].id == task_id)
            .first()
        )
        if task is None:
            abort(404)

        position = ctx["data_service"].get_position_by_task_id(task_id)
        tools = ctx["data_service"].get_tools_by_task(task_id)
        parts = ctx["data_service"].get_parts_by_task(task_id)
        documents = ctx["data_service"].get_documents_by_task_id(task_id)
        images = ctx["data_service"].get_images_by_task(task_id)
        drawings = ctx["data_service"].get_drawings_by_task(task_id)

        return jsonify(
            {
                "task": serialize_task(task),
                "position": serialize_position(position),
                "tools": [serialize_tool(item) for item in tools],
                "parts": [serialize_part(item) for item in parts],
                "documents": [serialize_document(item) for item in documents],
                "images": [_serialize_image_payload(item) for item in images],
                "drawings": [serialize_drawing(item) for item in drawings],
            }
        )

    @app.get("/api/positions/<int:position_id>/parts")
    @require_login
    def parts_by_position(position_id):
        ctx = backend()
        items = ctx["data_service"].get_parts_by_position(position_id)
        return jsonify([serialize_part(item) for item in items])

    @app.get("/api/positions/<int:position_id>/documents")
    @require_login
    def documents_by_position(position_id):
        ctx = backend()
        items = ctx["data_service"].get_documents_by_position(position_id)
        return jsonify([serialize_document(item) for item in items])

    @app.get("/api/positions/<int:position_id>/images")
    @require_login
    def images_by_position(position_id):
        ctx = backend()
        items = ctx["data_service"].get_images_by_position(position_id)
        return jsonify([_serialize_image_payload(item) for item in items])

    @app.get("/api/positions/<int:position_id>/drawings")
    @require_login
    def drawings_by_position(position_id):
        ctx = backend()
        items = ctx["data_service"].get_drawings_by_position(position_id)
        return jsonify([serialize_drawing(item) for item in items])

    @app.get("/api/layouts")
    @require_login
    def get_layouts():
        layout_user = ensure_layout_user()

        saved_layouts = {}
        if layout_user and hasattr(layout_user, "get_all_layouts"):
            saved_layouts = layout_user.get_all_layouts() or {}

        merged = dict(DEFAULT_LAYOUTS)
        merged.update(saved_layouts)

        return jsonify(
            {
                "layouts": serialize_layouts(merged),
                "default_layout_name": "Default",
            }
        )

    @app.get("/api/layouts/<layout_name>")
    @require_login
    def get_layout(layout_name):
        layout_user = ensure_layout_user()

        if layout_name in DEFAULT_LAYOUTS:
            return jsonify(
                {
                    "name": layout_name,
                    "layout": DEFAULT_LAYOUTS[layout_name],
                    "is_default": True,
                }
            )

        layout = None
        if layout_user and hasattr(layout_user, "get_layout"):
            layout = layout_user.get_layout(layout_name)

        if not layout:
            abort(404)

        return jsonify(
            {
                "name": layout_name,
                "layout": layout,
                "is_default": False,
            }
        )

    @app.post("/api/layouts")
    @require_login
    def save_layout():
        payload = request.get_json(silent=True) or {}
        layout_name = (payload.get("name") or "").strip()
        layout_data = payload.get("layout")

        if not layout_name:
            return jsonify({"error": "Layout name is required"}), 400
        if not isinstance(layout_data, dict) or not layout_data:
            return jsonify({"error": "Layout data is required"}), 400
        if layout_name == "Default":
            return jsonify({"error": "Default layout cannot be overwritten"}), 400

        layout_user = ensure_layout_user()
        if not layout_user or not hasattr(layout_user, "save_layout"):
            return jsonify({"error": "Layout storage is unavailable for this user"}), 500

        layout_user.save_layout(layout_name, layout_data)

        return jsonify(
            {
                "message": f"Layout '{layout_name}' saved successfully",
                "name": layout_name,
                "layout": layout_data,
            }
        )

    @app.delete("/api/layouts/<layout_name>")
    @require_login
    def delete_layout(layout_name):
        if layout_name == "Default":
            return jsonify({"error": "Default layout cannot be deleted"}), 400

        layout_user = ensure_layout_user()
        if not layout_user or not hasattr(layout_user, "delete_layout"):
            return jsonify({"error": "Layout storage is unavailable for this user"}), 500

        deleted = layout_user.delete_layout(layout_name)
        if not deleted:
            abort(404)

        return jsonify({"message": f"Layout '{layout_name}' deleted successfully"})

    @app.get("/media/<kind>")
    @require_login
    def media_proxy(kind):
        file_path = request.args.get("file_path", "")
        if not file_path:
            abort(404)

        if kind != "image":
            abort(404)

        ctx = backend()
        resolved = Path(file_path).resolve()
        allowed_roots = [Path(ctx["DATABASE_PATH_IMAGES_FOLDER"]).resolve()]

        if not any(root == resolved or root in resolved.parents for root in allowed_roots):
            abort(403)

        if not resolved.exists() or not resolved.is_file():
            abort(404)

        return send_file(resolved)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)