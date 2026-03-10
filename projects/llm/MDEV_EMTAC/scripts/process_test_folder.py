from __future__ import annotations

import os
import sys
import uuid
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.coordinators.batch_processing_coordinator import BatchProcessingCoordinator
from modules.configuration.log_config import (
    info_id,
    warning_id,
    error_id,
)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

TEST_FOLDER = r"E:\emtac\data\raw_documention\test_doc"

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".xlsx",
    ".xls",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".gif",
    ".bmp",
    ".webp",
    ".md",
    ".json",
    ".xml",
    ".rtf",
}

DEFAULT_PDF_PAGES = 5
KEEP_PREPARED_TEST_FOLDER = False

# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EMTAC Test Folder Ingestion")

    parser.add_argument(
        "--folder",
        type=str,
        default=TEST_FOLDER,
        help="Folder to ingest",
    )

    parser.add_argument(
        "--pages",
        type=str,
        default=str(DEFAULT_PDF_PAGES),
        help="Number of PDF pages to process (example: 5 or 'all')",
    )

    parser.add_argument(
        "--keep-prepared-folder",
        action="store_true",
        help="Keep the prepared temporary folder used for testing",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def collect_files(folder_path: str) -> List[str]:
    collected: List[str] = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()

            if ext in ALLOWED_EXTENSIONS:
                collected.append(os.path.join(root, file_name))
            else:
                warning_id(f"Skipping unsupported file type: {file_name}", None)

    collected.sort(key=lambda p: p.lower())
    return collected


def build_metadata(source_folder: str) -> Dict[str, Any]:
    """
    Keep metadata simple for coordinator testing.
    Titles for individual files will be derived by the batch coordinator when blank.
    """
    return {
        "title": "",
        "description": "",
        "source": "folder_test_ingestion",
        "document_type": "test_batch",
        "tags": "test,bulk",
        "area": "",
        "equipment_group": "",
        "model": "",
        "asset_number": "",
        "location": "",
        "site_location": "",
        "room_number": "Unknown",
        "department": "",
        "priority": "normal",
        "original_source_folder": source_folder,
    }


def _safe_makedirs(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def extract_first_n_pages_pdf(
    source_pdf_path: str,
    *,
    max_pages: int,
    output_pdf_path: str,
    request_id: Optional[str] = None,
) -> str:
    try:
        with fitz.open(source_pdf_path) as src:
            total = int(src.page_count)
            pages_to_copy = min(max_pages, total)

            info_id(
                f"[process_test_folder] Creating first-pages PDF | "
                f"src='{source_pdf_path}' pages={pages_to_copy}/{total} out='{output_pdf_path}'",
                request_id,
            )

            new_doc = fitz.open()

            for i in range(pages_to_copy):
                new_doc.insert_pdf(src, from_page=i, to_page=i)

            new_doc.save(output_pdf_path)
            new_doc.close()

        return output_pdf_path

    except Exception as e:
        error_id(
            f"[process_test_folder] Failed extracting first {max_pages} pages "
            f"src='{source_pdf_path}' err={e}",
            request_id,
        )
        raise


def build_prepared_test_folder(
    *,
    source_folder: str,
    file_paths: List[str],
    max_pages: Optional[int],
    request_id: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a temp folder that mirrors the source files for coordinator testing.

    Rules:
      - non-PDF files are copied as-is
      - PDFs are copied whole if max_pages is None
      - PDFs are trimmed to first N pages if max_pages is an int
    """
    temp_root = tempfile.mkdtemp(prefix="emtac_batch_test_")
    manifest: List[Dict[str, Any]] = []

    info_id(
        f"[process_test_folder] Preparing temp ingestion folder: {temp_root}",
        request_id,
    )

    for src_path in file_paths:
        rel_path = os.path.relpath(src_path, source_folder)
        dst_path = os.path.join(temp_root, rel_path)

        dst_dir = os.path.dirname(dst_path)
        _safe_makedirs(dst_dir)

        ext = os.path.splitext(src_path)[1].lower()

        try:
            if ext == ".pdf" and max_pages is not None:
                extract_first_n_pages_pdf(
                    src_path,
                    max_pages=max_pages,
                    output_pdf_path=dst_path,
                    request_id=request_id,
                )
                processed_mode = f"first_{max_pages}_pages"
            else:
                shutil.copy2(src_path, dst_path)
                processed_mode = "full_copy"

            manifest.append(
                {
                    "source_path": src_path,
                    "prepared_path": dst_path,
                    "relative_path": rel_path,
                    "extension": ext,
                    "mode": processed_mode,
                }
            )

        except Exception as e:
            error_id(
                f"[process_test_folder] Failed to prepare file '{src_path}' err={e}",
                request_id,
            )
            manifest.append(
                {
                    "source_path": src_path,
                    "prepared_path": dst_path,
                    "relative_path": rel_path,
                    "extension": ext,
                    "mode": "prepare_failed",
                    "error": str(e),
                }
            )

    return temp_root, manifest


def print_manifest(manifest: List[Dict[str, Any]]):
    print("\nPREPARED FILES")
    print("-" * 80)

    for entry in manifest:
        mode = entry.get("mode", "unknown")
        rel_path = entry.get("relative_path", "")
        ext = entry.get("extension", "")
        if "error" in entry:
            print(f"FAILED PREP  | {ext:<6} | {mode:<16} | {rel_path} | {entry['error']}")
        else:
            print(f"READY        | {ext:<6} | {mode:<16} | {rel_path}")


def safe_remove_tree(path: str, request_id: Optional[str] = None):
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path)
            info_id(f"[process_test_folder] Removed temp folder: {path}", request_id)
    except Exception as e:
        warning_id(
            f"[process_test_folder] Failed removing temp folder '{path}' err={e}",
            request_id,
        )


def print_results(response: Dict[str, Any]):
    results = response.get("results", [])

    print("\nPER-FILE RESULTS")
    print("-" * 120)

    if not results:
        print("No per-file results returned.")
        return

    for result in results:
        file_type = (
            result.get("file_type")
            or result.get("category")
            or "unknown"
        )
        status = result.get("status", "unknown")
        http_status = result.get("http_status", "n/a")
        duration_ms = result.get("duration_ms", 0)
        file_name = result.get("file_name", "")
        message = result.get("message", "")

        print(
            f"{status.upper():<10} | "
            f"{file_type:<10} | "
            f"HTTP {str(http_status):<4} | "
            f"{str(duration_ms):>6} ms | "
            f"{file_name} | {message}"
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    args = parse_args()
    request_id = str(uuid.uuid4())

    source_folder = os.path.abspath(args.folder)

    if args.pages.lower() == "all":
        max_pages = None
    else:
        max_pages = int(args.pages)

    print("=" * 80)
    print("EMTAC TEST FOLDER INGESTION")
    print("=" * 80)
    print(f"SOURCE FOLDER : {source_folder}")

    if max_pages is None:
        print("PDF MODE      : FULL DOCUMENT")
    else:
        print(f"PDF MODE      : FIRST {max_pages} PAGES")

    print("=" * 80)

    if not os.path.exists(source_folder):
        raise RuntimeError(f"Folder does not exist: {source_folder}")

    file_paths = collect_files(source_folder)

    if not file_paths:
        print("No supported files found.")
        return

    print(f"\nFound {len(file_paths)} supported files in source folder\n")

    prepared_folder = None

    try:
        prepared_folder, manifest = build_prepared_test_folder(
            source_folder=source_folder,
            file_paths=file_paths,
            max_pages=max_pages,
            request_id=request_id,
        )

        print(f"PREPARED FOLDER: {prepared_folder}")
        print_manifest(manifest)

        prepare_failures = [m for m in manifest if m.get("mode") == "prepare_failed"]
        if prepare_failures:
            print(f"\nPreparation failures: {len(prepare_failures)}")

        metadata = build_metadata(source_folder=source_folder)

        coordinator = BatchProcessingCoordinator()

        success, response, status = coordinator.process_folder(
            folder_path=prepared_folder,
            metadata=metadata,
            include_subfolders=True,
            concurrent=False,
            max_workers=4,
            request_id=request_id,
        )

        print("\n" + "=" * 80)
        print("BATCH COORDINATOR RESPONSE")
        print("=" * 80)
        print(f"SUCCESS      : {success}")
        print(f"HTTP STATUS  : {status}")
        print(f"MESSAGE      : {response.get('message')}")
        print(f"TOTAL FILES  : {response.get('total_files_found')}")
        print(f"PROCESSED    : {response.get('processed')}")
        print(f"FAILED       : {response.get('failed')}")
        print(f"SKIPPED      : {response.get('skipped', 0)}")
        print("=" * 80)

        print_results(response)

    except Exception as e:
        error_id(f"[process_test_folder] Unhandled exception: {e}", request_id, exc_info=True)
        print(f"\nCRITICAL FAILURE: {e}")

    finally:
        keep_folder = args.keep_prepared_folder or KEEP_PREPARED_TEST_FOLDER

        if prepared_folder and not keep_folder:
            safe_remove_tree(prepared_folder, request_id=request_id)
        elif prepared_folder:
            print(f"\nPrepared folder kept for inspection: {prepared_folder}")


if __name__ == "__main__":
    main()