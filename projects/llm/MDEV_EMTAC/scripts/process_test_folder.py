# scripts/process_test_folder.py

import os
import sys
import uuid
import argparse
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.applications.file_processing_coordinator import FileProcessingCoordinator
from modules.adapters.local_file_adapter import LocalFileAdapter
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
    ".docx",
    ".txt",
    ".xlsx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

DEFAULT_PDF_PAGES = 5
KEEP_TEMP_PDFS = False


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EMTAC Test Folder Ingestion")

    parser.add_argument(
        "--pages",
        type=str,
        default=str(DEFAULT_PDF_PAGES),
        help="Number of PDF pages to process (example: 5 or 'all')",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def collect_files(folder_path: str) -> List[str]:

    collected: List[str] = []

    for root, _, files in os.walk(folder_path):
        for file in files:

            ext = os.path.splitext(file)[1].lower()

            if ext in ALLOWED_EXTENSIONS:
                collected.append(os.path.join(root, file))
            else:
                warning_id(f"Skipping unsupported file type: {file}", None)

    return collected


def build_metadata(file_path: str, *, original_path: Optional[str] = None) -> Dict[str, Any]:

    return {
        "source": "folder_test_ingestion",
        "uploader_id": 0,
        "document_type": "test_batch",
        "tags": ["test", "bulk"],
        "area": None,
        "equipment_group": None,
        "model": None,
        "original_path": original_path or file_path,
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
    request_id: Optional[str] = None,
) -> str:

    base_dir = os.path.dirname(source_pdf_path)
    base_name = os.path.splitext(os.path.basename(source_pdf_path))[0]

    temp_dir = os.path.join(base_dir, "__temp_first_pages__")
    _safe_makedirs(temp_dir)

    out_path = os.path.join(temp_dir, f"{base_name}_FIRST_{max_pages}_PAGES.pdf")

    try:

        with fitz.open(source_pdf_path) as src:

            total = int(src.page_count)
            pages_to_copy = min(max_pages, total)

            info_id(
                f"[process_test_folder] Creating first-pages PDF | "
                f"src='{source_pdf_path}' pages={pages_to_copy}/{total} out='{out_path}'",
                request_id,
            )

            new_doc = fitz.open()

            for i in range(pages_to_copy):
                new_doc.insert_pdf(src, from_page=i, to_page=i)

            new_doc.save(out_path)
            new_doc.close()

        return out_path

    except Exception as e:

        error_id(
            f"[process_test_folder] Failed extracting first {max_pages} pages "
            f"src='{source_pdf_path}' err={e}",
            request_id,
        )

        raise


def safe_remove(path: str, request_id: Optional[str] = None):

    try:

        if path and os.path.exists(path):

            os.remove(path)

            info_id(f"[process_test_folder] Removed temp file: {path}", request_id)

    except Exception as e:

        warning_id(
            f"[process_test_folder] Failed removing temp file '{path}' err={e}",
            request_id,
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    args = parse_args()

    if args.pages.lower() == "all":
        max_pages = None
    else:
        max_pages = int(args.pages)

    print("=" * 80)
    print("EMTAC TEST FOLDER INGESTION")
    print("=" * 80)

    if max_pages is None:
        print("PDF MODE : FULL DOCUMENT")
    else:
        print(f"PDF MODE : FIRST {max_pages} PAGES")

    print("=" * 80)

    if not os.path.exists(TEST_FOLDER):
        raise RuntimeError(f"Folder does not exist: {TEST_FOLDER}")

    file_paths = collect_files(TEST_FOLDER)

    if not file_paths:
        print("No supported files found.")
        return

    print(f"\nFound {len(file_paths)} files\n")

    coordinator = FileProcessingCoordinator()

    success_count = 0
    failure_count = 0

    for file_path in file_paths:

        request_id = str(uuid.uuid4())

        print("-" * 80)
        print(f"Processing: {file_path}")
        print(f"Request ID: {request_id}")

        ext = os.path.splitext(file_path)[1].lower()

        file_path_to_process = file_path
        original_path = file_path

        temp_pdf_path = None

        if ext == ".pdf" and max_pages is not None:

            try:

                temp_pdf_path = extract_first_n_pages_pdf(
                    file_path,
                    max_pages=max_pages,
                    request_id=request_id,
                )

                file_path_to_process = temp_pdf_path

            except Exception as e:

                failure_count += 1
                print(f"CRITICAL FAILURE (preprocess): {e}")
                continue

        metadata = build_metadata(file_path_to_process, original_path=original_path)

        adapter = LocalFileAdapter(file_path_to_process)

        try:

            success, response, status = coordinator.process_upload(
                files=[adapter],
                metadata=metadata,
                concurrent=False,
                request_id=request_id,
            )

            if success:

                success_count += 1
                print(f"SUCCESS | HTTP {status}")

            else:

                failure_count += 1
                print(f"FAILED  | HTTP {status}")
                print(f"Error: {response}")

        except Exception as e:

            failure_count += 1

            error_id(f"Unhandled exception for {file_path}: {e}", request_id)

            print(f"CRITICAL FAILURE: {e}")

        finally:

            if temp_pdf_path and not KEEP_TEMP_PDFS:

                safe_remove(temp_pdf_path, request_id=request_id)

    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    print(f"Total Files  : {len(file_paths)}")
    print(f"Success      : {success_count}")
    print(f"Failures     : {failure_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()