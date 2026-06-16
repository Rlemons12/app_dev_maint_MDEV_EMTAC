from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict

from flask import Flask, abort, render_template, request, session


BASE_DIR = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
THEMES_DIR = STATIC_DIR / "css" / "module_template" / "themes"

HOST = "127.0.0.1"
PORT = 5000
DEBUG = True

# Map friendly query keys to real partial template paths
ALLOWED_PARTIALS: Dict[str, str] = {
    "panels_2": "module_template_html/partials/layout_panels_2.html",
    "panels_3": "module_template_html/partials/layout_panels_3_bottom_span.html",
    "panels_4": "module_template_html/partials/layout_panels_4_grid.html",
    "panels_5": "module_template_html/partials/layout_panels_5_main_left.html",
    "panels_6": "module_template_html/partials/layout_panels_6_grid.html",
}

DEFAULT_PARTIAL_KEY = "panels_6"
NEW_TEMPLATE_PATH = "module_template_html/New_template.html"


class ThemeOption(TypedDict):
    file: str
    label: str


app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

app.secret_key = "emtac-new-template-standalone-secret-key"


def build_theme_label(filename: str) -> str:
    """
    Convert a theme filename into a readable label.

    Example:
        crimson_night.css -> Crimson Night
        theme-blue-grey.css -> Blue Grey
    """
    name = Path(filename).name

    if name.lower().endswith(".css"):
        name = name[:-4]

    if name.lower().startswith("theme-"):
        name = name[6:]

    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = " ".join(name.split())

    return name.title()


def get_available_themes() -> List[ThemeOption]:
    """
    Read theme CSS files from:
        static/css/module_template/themes
    """
    if not THEMES_DIR.exists() or not THEMES_DIR.is_dir():
        return []

    theme_files = sorted(
        (
            path
            for path in THEMES_DIR.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() == ".css"
        ),
        key=lambda p: p.name.lower(),
    )

    return [
        {
            "file": theme_file.name,
            "label": build_theme_label(theme_file.name),
        }
        for theme_file in theme_files
    ]


def validate_template_exists(template_path: str) -> str:
    """
    Ensure a template file exists before rendering it.
    """
    full_path = TEMPLATES_DIR / template_path
    if not full_path.exists():
        abort(404, description=f"Template file not found: {full_path}")
    return template_path


def resolve_partial_from_request() -> str:
    """
    Resolve the requested partial from the query string.

    Examples:
        /new-template
        /new-template?partial=panels_4
        /new-template?partial=panels_6
    """
    partial_key = request.args.get("partial", DEFAULT_PARTIAL_KEY).strip()
    selected_partial = ALLOWED_PARTIALS.get(partial_key)

    if not selected_partial:
        abort(404, description=f"Unknown partial key: {partial_key}")

    return validate_template_exists(selected_partial)


@app.before_request
def ensure_demo_session_defaults() -> None:
    """
    Provide the session values expected by base_template.html and sidebar partials.
    """
    session.setdefault("user_level", "ADMIN")
    session.setdefault("first_name", "Demo")


@app.context_processor
def inject_shared_template_context() -> Dict[str, object]:
    """
    Shared template variables expected by the base template/sidebar.
    """
    return {
        "available_themes": get_available_themes(),
        "current_page": "search",
        "current_ai_model": "Demo Model",
        "current_embedding_model": "Demo Embedding Model",
    }


@app.route("/", methods=["GET"])
def home():
    """
    Convenience route to the new template page.
    """
    return new_template_page()


@app.route("/new-template", methods=["GET"])
def new_template_page():
    """
    Render the real New_template.html page and pass in a selected partial.
    """
    available_themes = get_available_themes()
    if not available_themes:
        abort(500, description=f"No theme CSS files found in: {THEMES_DIR}")

    validate_template_exists(NEW_TEMPLATE_PATH)
    selected_partial = resolve_partial_from_request()

    return render_template(
        NEW_TEMPLATE_PATH,
        selected_partial=selected_partial,
    )


@app.route("/health", methods=["GET"])
def health() -> Dict[str, object]:
    """
    Simple health endpoint for startup verification.
    """
    return {
        "status": "ok",
        "template": NEW_TEMPLATE_PATH,
        "default_partial": ALLOWED_PARTIALS[DEFAULT_PARTIAL_KEY],
        "themes_found": len(get_available_themes()),
    }


if __name__ == "__main__":
    discovered_themes = get_available_themes()

    print("Starting EMTAC New_template standalone page...")
    print(f"Templates folder:  {TEMPLATES_DIR}")
    print(f"Static folder:     {STATIC_DIR}")
    print(f"Themes folder:     {THEMES_DIR}")
    print(f"Template page:     {NEW_TEMPLATE_PATH}")
    print(f"Default partial:   {ALLOWED_PARTIALS[DEFAULT_PARTIAL_KEY]}")
    print("Available partial keys:")
    for key, value in ALLOWED_PARTIALS.items():
        print(f"  - {key}: {value}")

    print("Discovered themes:")
    for theme in discovered_themes:
        print(f"  - {theme['file']} -> {theme['label']}")

    print(f"Open in browser:   http://{HOST}:{PORT}")
    print(f"Panels 6 example:  http://{HOST}:{PORT}/new-template?partial=panels_6")
    print(f"Panels 4 example:  http://{HOST}:{PORT}/new-template?partial=panels_4")

    app.run(debug=DEBUG, host=HOST, port=PORT)