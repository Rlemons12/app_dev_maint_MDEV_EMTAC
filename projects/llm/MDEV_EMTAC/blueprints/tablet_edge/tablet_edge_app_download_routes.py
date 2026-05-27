"""
Tablet Edge Agent app download page routes.

Purpose:
    Provides a simple internal web page where users can download the latest
    EMTAC Tablet Edge Agent APK.

Routes:
    GET /tablet-edge/app
        Shows the latest active release and a download button.

    GET /tablet-edge/app-download/latest
        Downloads the newest active release.

    GET /tablet-edge/app-download/<release_id>
        Downloads a specific release by database ID.

Database table:
    tablet_edge.tablet_app_release
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from flask import Blueprint, abort, current_app, redirect, render_template, send_file, url_for

logger = logging.getLogger(__name__)

tablet_edge_app_download_bp = Blueprint(
    "tablet_edge_app_download",
    __name__,
    url_prefix="/tablet-edge",
)


DEFAULT_UPDATES_DIR = Path("E:/emtac/tablet_updates")
APK_MIME_TYPE = "application/vnd.android.package-archive"


@dataclass
class TabletAppRelease:
    id: int
    app_package: str
    release_channel: str
    version_name: str
    version_code: int
    apk_filename: str
    apk_file_path: str
    apk_sha256: str
    apk_size_bytes: Optional[int]
    release_notes: Optional[str]
    is_active: bool
    is_required: bool
    rollout_percent: int
    published_at: Optional[str]
    updated_at: Optional[str]

    @property
    def display_size(self) -> str:
        return format_bytes(self.apk_size_bytes)

    @property
    def short_sha256(self) -> str:
        if not self.apk_sha256:
            return "Not available"
        return self.apk_sha256[:16] + "..."


def get_database_url() -> str:
    """
    Reads DATABASE_URL from Flask config or environment.

    Supports SQLAlchemy-style PostgreSQL URL:
        postgresql+psycopg2://user:pass@host:5432/db

    Converts it to:
        postgresql://user:pass@host:5432/db
    """
    database_url = (
        current_app.config.get("DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("SQLALCHEMY_DATABASE_URI")
        or current_app.config.get("SQLALCHEMY_DATABASE_URI")
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL or SQLALCHEMY_DATABASE_URI is not configured."
        )

    database_url = str(database_url).strip()

    if database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace(
            "postgresql+psycopg2://",
            "postgresql://",
            1,
        )

    return database_url


def get_connection():
    database_url = get_database_url()
    return psycopg2.connect(database_url)


def row_to_release(row: dict) -> TabletAppRelease:
    return TabletAppRelease(
        id=row["id"],
        app_package=row["app_package"],
        release_channel=row["release_channel"],
        version_name=row["version_name"],
        version_code=row["version_code"],
        apk_filename=row["apk_filename"],
        apk_file_path=row["apk_file_path"],
        apk_sha256=row["apk_sha256"],
        apk_size_bytes=row.get("apk_size_bytes"),
        release_notes=row.get("release_notes"),
        is_active=row["is_active"],
        is_required=row["is_required"],
        rollout_percent=row["rollout_percent"],
        published_at=str(row["published_at"]) if row.get("published_at") else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
    )


def get_latest_active_release(
    app_package: str = "com.example.emtactablet",
    release_channel: str = "stable",
) -> Optional[TabletAppRelease]:
    sql = """
        SELECT
            id,
            app_package,
            release_channel,
            version_name,
            version_code,
            apk_filename,
            apk_file_path,
            apk_sha256,
            apk_size_bytes,
            release_notes,
            is_active,
            is_required,
            rollout_percent,
            published_at,
            updated_at
        FROM tablet_edge.tablet_app_release
        WHERE app_package = %s
          AND release_channel = %s
          AND is_active = TRUE
          AND rollout_percent > 0
        ORDER BY version_code DESC
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, (app_package, release_channel))
            row = cursor.fetchone()

    if not row:
        return None

    return row_to_release(row)


def get_release_by_id(release_id: int) -> Optional[TabletAppRelease]:
    sql = """
        SELECT
            id,
            app_package,
            release_channel,
            version_name,
            version_code,
            apk_filename,
            apk_file_path,
            apk_sha256,
            apk_size_bytes,
            release_notes,
            is_active,
            is_required,
            rollout_percent,
            published_at,
            updated_at
        FROM tablet_edge.tablet_app_release
        WHERE id = %s
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, (release_id,))
            row = cursor.fetchone()

    if not row:
        return None

    return row_to_release(row)


def resolve_apk_path(release: TabletAppRelease) -> Path:
    """
    Resolves APK path from database.

    Primary:
        release.apk_file_path

    Fallback:
        E:/emtac/tablet_updates/<apk_filename>
    """
    raw_path = release.apk_file_path.replace("\\", "/").strip()
    apk_path = Path(raw_path)

    if apk_path.exists() and apk_path.is_file():
        return apk_path

    fallback_dir = Path(
        current_app.config.get("TABLET_UPDATES_DIR")
        or os.environ.get("TABLET_UPDATES_DIR")
        or DEFAULT_UPDATES_DIR
    )

    fallback_path = fallback_dir / release.apk_filename

    if fallback_path.exists() and fallback_path.is_file():
        return fallback_path

    raise FileNotFoundError(
        f"APK file not found. db_path={release.apk_file_path}, "
        f"fallback_path={fallback_path}"
    )


def format_bytes(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "Unknown size"

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size:.1f} TB"


def get_public_base_url() -> str:
    """
    Optional config for showing a stable URL on the page.

    Set in Flask config or .env if desired:
        TABLET_EDGE_PUBLIC_BASE_URL=http://172.19.194.129:5000
    """
    return (
        current_app.config.get("TABLET_EDGE_PUBLIC_BASE_URL")
        or os.environ.get("TABLET_EDGE_PUBLIC_BASE_URL")
        or ""
    ).rstrip("/")


@tablet_edge_app_download_bp.route("/app", methods=["GET"])
def app_download_page():
    try:
        release = get_latest_active_release()
    except Exception as exc:
        logger.exception("Failed to load latest EMTAC Tablet Edge Agent release.")
        return (
            render_template(
                "tablet_edge/app.html",
                release=None,
                error_message=f"Could not load latest app release: {exc}",
                download_url=None,
                latest_download_url=None,
                public_page_url=None,
            ),
            500,
        )

    if not release:
        return (
            render_template(
                "tablet_edge/app.html",
                release=None,
                error_message="No active EMTAC Tablet Edge Agent release is available.",
                download_url=None,
                latest_download_url=None,
                public_page_url=None,
            ),
            404,
        )

    public_base_url = get_public_base_url()

    download_url = url_for(
        "tablet_edge_app_download.download_release",
        release_id=release.id,
    )

    latest_download_url = url_for(
        "tablet_edge_app_download.download_latest_release",
    )

    public_page_url = None
    if public_base_url:
        public_page_url = f"{public_base_url}/tablet-edge/app"

    return render_template(
        "tablet_edge/app.html",
        release=release,
        error_message=None,
        download_url=download_url,
        latest_download_url=latest_download_url,
        public_page_url=public_page_url,
    )


@tablet_edge_app_download_bp.route("/app-download/latest", methods=["GET"])
def download_latest_release():
    release = get_latest_active_release()

    if not release:
        abort(404, description="No active EMTAC Tablet Edge Agent release is available.")

    return redirect(
        url_for(
            "tablet_edge_app_download.download_release",
            release_id=release.id,
        )
    )


@tablet_edge_app_download_bp.route("/app-download/<int:release_id>", methods=["GET"])
def download_release(release_id: int):
    release = get_release_by_id(release_id)

    if not release:
        abort(404, description=f"Release ID {release_id} was not found.")

    if not release.is_active:
        abort(404, description=f"Release ID {release_id} is not active.")

    try:
        apk_path = resolve_apk_path(release)
    except FileNotFoundError as exc:
        logger.exception("APK file missing for release_id=%s", release_id)
        abort(404, description=str(exc))

    logger.info(
        "Serving EMTAC Tablet Edge Agent APK. release_id=%s version=%s code=%s path=%s",
        release.id,
        release.version_name,
        release.version_code,
        apk_path,
    )

    try:
        return send_file(
            apk_path,
            mimetype=APK_MIME_TYPE,
            as_attachment=True,
            download_name=release.apk_filename,
            max_age=0,
        )
    except TypeError:
        # Fallback for older Flask versions.
        return send_file(
            str(apk_path),
            mimetype=APK_MIME_TYPE,
            as_attachment=True,
            attachment_filename=release.apk_filename,
            cache_timeout=0,
        )
