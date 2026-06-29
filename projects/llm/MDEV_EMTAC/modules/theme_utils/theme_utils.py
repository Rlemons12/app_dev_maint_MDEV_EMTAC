from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from flask import current_app


def build_theme_label(filename: str) -> str:
    """
    Convert a CSS filename into a readable dropdown label.

    Examples:
        theme-blue-grey.css -> Blue Grey
        theme-classic-grey-navy.css -> Classic Grey Navy
        green-modern.css -> Green Modern
    """
    name = Path(filename).name

    if name.lower().endswith(".css"):
        name = name[:-4]

    if name.startswith("theme-"):
        name = name[len("theme-"):]

    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = " ".join(name.split())

    return name.title()


def get_themes_directory() -> Path:
    """
    Return the absolute path to the theme directory under Flask static.

    Expected location:
        static/css/module_template/themes
    """
    return Path(current_app.static_folder) / "css" / "module_template" / "themes"


def get_available_themes() -> List[Dict[str, str]]:
    """
    Read CSS theme files from:
        static/css/module_template/themes

    Returns:
        [
            {"file": "theme-blue-grey.css", "label": "Blue Grey"},
            {"file": "theme-classic-grey-navy.css", "label": "Classic Grey Navy"},
            ...
        ]

    Notes:
        - The folder contents are the source of truth.
        - No hardcoded theme file list is used.
        - Only .css files are returned.
        - Results are sorted alphabetically by filename.
    """
    themes_dir = get_themes_directory()

    if not themes_dir.exists() or not themes_dir.is_dir():
        return []

    theme_files = sorted(
        (
            path
            for path in themes_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".css"
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