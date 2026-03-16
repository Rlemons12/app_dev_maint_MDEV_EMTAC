from __future__ import annotations

import csv
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, redirect,url_for
from werkzeug.utils import secure_filename
from sqlalchemy import or_

from modules.decorators import trace_entrypoint
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id, debug_id
from modules.coordinators.batch_processing_coordinator import BatchProcessingCoordinator
from modules.emtacdb.emtacdb_fts import (
    FileLog,
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Location,
)


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------
db_config = DatabaseConfig()
Session = db_config.get_main_session_registry()

# ---------------------------------------------------------
# BLUEPRINT
# ---------------------------------------------------------
batch_processing_bp = Blueprint("batch_processing_bp", __name__)

# ---------------------------------------------------------
# LOCAL LOG FILE SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

LOG_FILE = "script_sql_speed_test.csv"
log_file_path = os.path.join(LOG_FOLDER, LOG_FILE)

if not os.path.exists(log_file_path):
    with open(log_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file_path", "total_time_ms"])

# ---------------------------------------------------------
# FILE SUPPORT
# ---------------------------------------------------------
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".rtf",
    ".csv",
    ".xls",
    ".xlsx",
    ".md",
    ".json",
    ".xml",
    ".ppt",
    ".pptx",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _safe_int(value: Any) -> Optional[int]:
    text = _safe_str(value)
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _is_supported_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in ALLOWED_EXTENSIONS


def _get_area_id(session, raw_value: Any) -> Optional[int]:
    text = _safe_str(raw_value)
    if not text:
        return None

    numeric = _safe_int(text)
    if numeric is not None:
        return numeric

    row = (
        session.query(Area)
        .filter(Area.name == text)
        .first()
    )
    return row.id if row else None


def _get_equipment_group_id(session, raw_value: Any) -> Optional[int]:
    text = _safe_str(raw_value)
    if not text:
        return None

    numeric = _safe_int(text)
    if numeric is not None:
        return numeric

    row = (
        session.query(EquipmentGroup)
        .filter(EquipmentGroup.name == text)
        .first()
    )
    return row.id if row else None


def _get_model_id(session, raw_value: Any) -> Optional[int]:
    text = _safe_str(raw_value)
    if not text:
        return None

    numeric = _safe_int(text)
    if numeric is not None:
        return numeric

    row = (
        session.query(Model)
        .filter(Model.name == text)
        .first()
    )
    return row.id if row else None


def _get_asset_number_id(session, raw_value: Any) -> Optional[int]:
    text = _safe_str(raw_value)
    if not text:
        return None

    numeric = _safe_int(text)
    if numeric is not None:
        return numeric

    row = (
        session.query(AssetNumber)
        .filter(AssetNumber.number == text)
        .first()
    )
    return row.id if row else None


def _get_location_id(session, raw_value: Any) -> Optional[int]:
    text = _safe_str(raw_value)
    if not text:
        return None

    numeric = _safe_int(text)
    if numeric is not None:
        return numeric

    row = (
        session.query(Location)
        .filter(Location.name == text)
        .first()
    )
    return row.id if row else None


def _build_batch_metadata(form_data, *, session, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Keeps compatibility with existing frontend field names.

    Supports both:
      - posted numeric IDs
      - posted display names
    """
    batch_area = _safe_str(form_data.get("batchArea"))
    batch_equipment_group = _safe_str(form_data.get("batchEquipmentGroup"))
    batch_model = _safe_str(form_data.get("batchModel"))
    batch_asset_number = _safe_str(form_data.get("batchAssetNumber"))
    batch_location = _safe_str(form_data.get("batchLocation"))

    area_id = _get_area_id(session, batch_area)
    equipment_group_id = _get_equipment_group_id(session, batch_equipment_group)
    model_id = _get_model_id(session, batch_model)
    asset_number_id = _get_asset_number_id(session, batch_asset_number)
    location_id = _get_location_id(session, batch_location)

    metadata = {
        "title": _safe_str(form_data.get("batchTitle")),
        "description": "",
        "source": "batch_processing_route",
        "document_type": "batch_upload",
        "tags": "batch,upload",
        "priority": "normal",
        "area": batch_area,
        "equipment_group": batch_equipment_group,
        "model": batch_model,
        "asset_number": batch_asset_number,
        "location": batch_location,
        "site_location": "",
        "room_number": "Unknown",
        "department": "",
        "area_id": area_id,
        "equipment_group_id": equipment_group_id,
        "model_id": model_id,
        "asset_number_id": asset_number_id,
        "location_id": location_id,
        "site_location_id": None,
    }

    debug_id(
        f"[batch_processing_bp] Resolved metadata for batch processing: {metadata}",
        request_id,
    )

    return metadata


def _copy_folder_contents(
    *,
    source_folder: str,
    destination_folder: str,
    request_id: Optional[str] = None,
) -> int:
    """
    Copy supported files from source folder into temp working folder.
    Preserves relative structure.
    """
    source_root = Path(source_folder)
    destination_root = Path(destination_folder)
    copied_count = 0

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        rel_path = path.relative_to(source_root)
        destination_path = destination_root / rel_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(path, destination_path)
        copied_count += 1

    info_id(
        f"[batch_processing_bp] Copied {copied_count} file(s) from folder source: {source_folder}",
        request_id,
    )
    return copied_count


def _save_uploaded_files(
    *,
    uploaded_files,
    destination_folder: str,
    request_id: Optional[str] = None,
) -> int:
    """
    Save uploaded files into the temp working folder.
    """
    saved_count = 0
    destination_root = Path(destination_folder)

    for file_obj in uploaded_files:
        raw_name = getattr(file_obj, "filename", "") or ""
        filename = secure_filename(raw_name)

        if not filename:
            warning_id(
                "[batch_processing_bp] Skipping uploaded file with empty filename",
                request_id,
            )
            continue

        if not _is_supported_file(filename):
            warning_id(
                f"[batch_processing_bp] Skipping unsupported uploaded file: {filename}",
                request_id,
            )
            continue

        destination_path = destination_root / filename
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        file_obj.save(str(destination_path))
        saved_count += 1

    info_id(
        f"[batch_processing_bp] Saved {saved_count} uploaded file(s) into working folder",
        request_id,
    )
    return saved_count


def _append_csv_file_logs(results: List[Dict[str, Any]]) -> None:
    try:
        with open(log_file_path, "a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            for result in results:
                writer.writerow([
                    result.get("file_path", ""),
                    result.get("duration_ms", 0),
                ])
    except Exception as exc:
        logger.error(f"Failed writing CSV file logs: {exc}")


def _write_db_file_logs(
    *,
    session_id: str,
    session_datetime: datetime,
    results: List[Dict[str, Any]],
) -> None:
    """
    Assumes FileLog.session is a STRING column.
    """
    session = Session()
    try:
        for result in results:
            file_log_entry = FileLog(
                session=session_id,
                session_datetime=session_datetime,
                file_processed=result.get("file_path", ""),
                total_time=str(result.get("duration_ms", 0)),
            )
            session.add(file_log_entry)

        session.commit()
        logger.info("Batch file logs saved to database")
    except Exception as db_error:
        session.rollback()
        logger.error(f"Database error saving file log entries: {db_error}")
    finally:
        session.close()


def _cleanup_temp_folder(path: Optional[str], request_id: Optional[str] = None) -> None:
    if not path:
        return

    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            info_id(
                f"[batch_processing_bp] Removed temp working folder: {path}",
                request_id,
            )
    except Exception as exc:
        warning_id(
            f"[batch_processing_bp] Failed removing temp working folder '{path}': {exc}",
            request_id,
        )


def _serialize_area(area: Area) -> Dict[str, Any]:
    return {
        "id": area.id,
        "name": getattr(area, "name", ""),
    }


def _serialize_equipment_group(group: EquipmentGroup) -> Dict[str, Any]:
    return {
        "id": group.id,
        "name": getattr(group, "name", ""),
        "area_id": getattr(group, "area_id", None),
    }


def _serialize_model(model: Model) -> Dict[str, Any]:
    return {
        "id": model.id,
        "name": getattr(model, "name", ""),
        "equipment_group_id": getattr(model, "equipment_group_id", None),
    }


def _serialize_asset_number(asset_number: AssetNumber) -> Dict[str, Any]:
    return {
        "id": asset_number.id,
        "number": (
            getattr(asset_number, "number", None)
            or getattr(asset_number, "asset_number", None)
            or getattr(asset_number, "name", "")
        ),
        "model_id": getattr(asset_number, "model_id", None),
    }


def _serialize_location(location: Location) -> Dict[str, Any]:
    return {
        "id": location.id,
        "name": getattr(location, "name", ""),
        "model_id": getattr(location, "model_id", None),
    }

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@batch_processing_bp.route("/batch_processing", methods=["POST"])
@trace_entrypoint(
    deep_profile=True,
    capture_args=True,
    capture_return=True,
)
def batch_processing():
    """
    Supports:
      - folder path only
      - uploaded files only
      - both together

    Existing frontend can remain unchanged.
    """
    request_id = str(uuid.uuid4())
    session_datetime = datetime.now()
    session_id = session_datetime.strftime("%Y%m%d%H%M%S")

    logger.info("Received batch processing request")
    logger.info(request.form)

    folder_path = _safe_str(request.form.get("batchFolderPath"))
    uploaded_files = request.files.getlist("files")

    if not folder_path and not uploaded_files:
        return jsonify({
            "status": "validation_error",
            "message": "No folder path or uploaded files provided",
        }), 400

    if folder_path:
        folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            return jsonify({
                "status": "validation_error",
                "message": "Folder path does not exist",
                "folder_path": folder_path,
            }), 400

        if not os.path.isdir(folder_path):
            return jsonify({
                "status": "validation_error",
                "message": "Provided path is not a folder",
                "folder_path": folder_path,
            }), 400

    temp_work_dir: Optional[str] = None
    metadata_session = None

    try:
        metadata_session = Session()
        metadata = _build_batch_metadata(
            request.form,
            session=metadata_session,
            request_id=request_id,
        )

        logger.info(f"Folder Path: {folder_path}")
        logger.info(f"Uploaded files count: {len(uploaded_files)}")
        logger.info(f"Batch metadata: {metadata}")

        temp_work_dir = tempfile.mkdtemp(prefix="emtac_batch_work_")
        info_id(
            f"[batch_processing_bp] Created temp working folder: {temp_work_dir}",
            request_id,
        )

        staged_count = 0

        if folder_path:
            staged_count += _copy_folder_contents(
                source_folder=folder_path,
                destination_folder=temp_work_dir,
                request_id=request_id,
            )

        if uploaded_files:
            staged_count += _save_uploaded_files(
                uploaded_files=uploaded_files,
                destination_folder=temp_work_dir,
                request_id=request_id,
            )

        if staged_count == 0:
            return jsonify({
                "status": "validation_error",
                "message": "No supported files found to process",
                "folder_path": folder_path,
                "uploaded_file_count": len(uploaded_files),
            }), 400

        coordinator = BatchProcessingCoordinator()

        success, response, http_status = coordinator.process_folder(
            folder_path=temp_work_dir,
            metadata=metadata,
            include_subfolders=True,
            concurrent=False,
            max_workers=4,
            request_id=request_id,
        )

        if not isinstance(response, dict):
            response = {
                "status": "processing_error",
                "message": "Coordinator returned invalid response",
                "raw_response": str(response),
            }
            success = False
            http_status = 500

        results = response.get("results", []) or []

        response["route_context"] = {
            "success": success,
            "folder_path_provided": bool(folder_path),
            "folder_path": folder_path,
            "uploaded_files_provided": len(uploaded_files),
            "staged_files": staged_count,
            "session_id": session_id,
            "session_datetime": session_datetime.isoformat(),
            "resolved_position_context": {
                "area_id": metadata.get("area_id"),
                "equipment_group_id": metadata.get("equipment_group_id"),
                "model_id": metadata.get("model_id"),
                "asset_number_id": metadata.get("asset_number_id"),
                "location_id": metadata.get("location_id"),
                "site_location_id": metadata.get("site_location_id"),
            },
        }

        _write_db_file_logs(
            session_id=session_id,
            session_datetime=session_datetime,
            results=results,
        )
        _append_csv_file_logs(results)

        logger.info(
            f"Batch processing completed | success={success} | "
            f"http_status={http_status} | processed={response.get('processed')} | "
            f"failed={response.get('failed')}"
        )

        if success:
            return redirect(url_for('upload_image_page'))

        return jsonify(response), http_status

    except Exception as exc:
        error_id(
            f"[batch_processing_bp] Unhandled batch processing error: {exc}",
            request_id,
            exc_info=True,
        )
        return jsonify({
            "status": "processing_error",
            "message": "Batch processing failed",
            "detail": str(exc),
        }), 500

    finally:
        if metadata_session is not None:
            metadata_session.close()
        _cleanup_temp_folder(temp_work_dir, request_id=request_id)


@batch_processing_bp.route("/add_batch_folder", methods=["POST"])
def add_batch_folder():
    """
    Backward-compatible alias route.
    """
    return batch_processing()


@batch_processing_bp.route("/batch/get_batch_list_data", methods=["GET"])
def get_batch_list_data():
    """
    Supports existing frontend JS:
        /batch/get_batch_list_data
    """
    session = Session()

    try:
        areas = session.query(Area).order_by(Area.name.asc()).all()
        equipment_groups = session.query(EquipmentGroup).order_by(EquipmentGroup.name.asc()).all()
        models = session.query(Model).order_by(Model.name.asc()).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).order_by(Location.name.asc()).all()

        return jsonify({
            "areas": [_serialize_area(area) for area in areas],
            "equipment_groups": [_serialize_equipment_group(group) for group in equipment_groups],
            "models": [_serialize_model(model) for model in models],
            "asset_numbers": [_serialize_asset_number(asset_number) for asset_number in asset_numbers],
            "locations": [_serialize_location(location) for location in locations],
        }), 200

    except Exception as exc:
        logger.exception("Failed to fetch batch dropdown data")
        return jsonify({
            "status": "processing_error",
            "message": "Failed to fetch batch dropdown data",
            "detail": str(exc),
        }), 500

    finally:
        session.close()