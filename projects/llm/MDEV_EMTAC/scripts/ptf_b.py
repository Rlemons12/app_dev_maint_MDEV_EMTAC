from __future__ import annotations

import os
import uuid
import requests
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

# ------------------------------------------------------------
# GPU SERVICE CONFIG
# ------------------------------------------------------------

from app.config.gpu_service_config import GPU_SERVICE_CONFIG


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

TEST_FOLDER = os.getenv(
    "GPU_TEST_FOLDER",
    r"E:\emtac\data\raw_documention\test_doc"
)

GPU_SERVICE_URL = os.getenv(
    "GPU_SERVICE_URL",
    "http://127.0.0.1:8000/vision/pdf"
)

PAGE_BATCH_SIZE = int(os.getenv("GPU_PAGE_BATCH_SIZE", "5"))
PAGE_BATCH_COUNT = int(os.getenv("GPU_PAGE_BATCH_COUNT", "3"))

PARALLEL_WORKERS = int(os.getenv("GPU_PARALLEL_WORKERS", "3"))

MODEL_NAME = os.getenv("GPU_MODEL_NAME", "nu_markdown")

DPI = int(os.getenv("GPU_DPI", "200"))
MAX_NEW_TOKENS = int(os.getenv("GPU_MAX_NEW_TOKENS", "1024"))

KEEP_TEMP_PDFS = os.getenv("GPU_KEEP_TEMP", "false").lower() == "true"


# ------------------------------------------------------------
# FILE COLLECTION
# ------------------------------------------------------------

def collect_files(folder_path: str) -> List[str]:

    collected = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                collected.append(os.path.join(root, file))

    return collected


# ------------------------------------------------------------
# PAGE WINDOWS
# ------------------------------------------------------------

def build_page_windows() -> List[Tuple[int, int]]:

    windows = []

    for i in range(PAGE_BATCH_COUNT):

        start = i * PAGE_BATCH_SIZE + 1
        end = start + PAGE_BATCH_SIZE - 1

        windows.append((start, end))

    return windows


# ------------------------------------------------------------
# PDF PAGE EXTRACTION
# ------------------------------------------------------------

def extract_page_range_pdf(
    source_pdf_path: str,
    start_page: int,
    end_page: int,
):

    base_name = Path(source_pdf_path).stem

    # store temporary batch PDFs inside artifact directory
    temp_dir = GPU_SERVICE_CONFIG.pdf_dir / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    out_path = temp_dir / f"{base_name}_PAGES_{start_page}_{end_page}.pdf"

    with fitz.open(source_pdf_path) as src:

        total = src.page_count

        start_idx = max(start_page - 1, 0)
        end_idx = min(end_page - 1, total - 1)

        new_doc = fitz.open()

        for i in range(start_idx, end_idx + 1):
            new_doc.insert_pdf(src, from_page=i, to_page=i)

        new_doc.save(out_path)
        new_doc.close()

    return str(out_path)


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

def safe_remove(path: str):

    try:

        if path and os.path.exists(path):
            os.remove(path)

    except Exception as e:

        print(f"Failed removing temp file {path} err={e}")


# ------------------------------------------------------------
# GPU SERVICE CALL
# ------------------------------------------------------------

def call_gpu_service(pdf_path: str):

    with open(pdf_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(pdf_path),
                f,
                "application/pdf",
            )
        }

        params = {
            "model": MODEL_NAME,
            "max_pages": 50,
            "dpi": DPI,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        response = requests.post(
            GPU_SERVICE_URL,
            params=params,
            files=files,
            timeout=600,
        )

        if response.status_code != 200:

            raise RuntimeError(
                f"GPU service error {response.status_code}: {response.text}"
            )

        return response.json()


# ------------------------------------------------------------
# WINDOW PROCESSOR
# ------------------------------------------------------------

def process_window(file_path, start_page, end_page):

    request_id = str(uuid.uuid4())

    temp_pdf_path = None

    try:

        print(
            f"Processing window {start_page}-{end_page} | request {request_id}"
        )

        temp_pdf_path = extract_page_range_pdf(
            file_path,
            start_page,
            end_page,
        )

        result = call_gpu_service(temp_pdf_path)

        print(
            f"SUCCESS window {start_page}-{end_page} "
            f"chunks={len(result.get('chunks', []))}"
        )

        return True

    except Exception as e:

        print(
            f"FAILED window {start_page}-{end_page} err={e}"
        )

        return False

    finally:

        if temp_pdf_path and not KEEP_TEMP_PDFS:
            safe_remove(temp_pdf_path)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 80)
    print("EMTAC GPU PARALLEL PAGE WINDOW TEST")
    print("=" * 80)

    print("\nGPU SERVICE CONFIG")
    print("-" * 40)

    for k, v in GPU_SERVICE_CONFIG.describe().items():
        print(f"{k}: {v}")

    files = collect_files(TEST_FOLDER)

    if not files:
        print("No PDFs found.")
        return

    page_windows = build_page_windows()

    success = 0
    failures = 0

    for file_path in files:

        print(f"\nProcessing PDF: {file_path}")

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:

            futures = []

            for start_page, end_page in page_windows:

                futures.append(
                    executor.submit(
                        process_window,
                        file_path,
                        start_page,
                        end_page,
                    )
                )

            for future in as_completed(futures):

                if future.result():
                    success += 1
                else:
                    failures += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Success : {success}")
    print(f"Failures: {failures}")

    print("=" * 80)


if __name__ == "__main__":
    main()