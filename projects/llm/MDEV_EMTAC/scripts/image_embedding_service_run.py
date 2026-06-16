from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from pprint import pprint

from werkzeug.datastructures import FileStorage

# ---------------------------------------------------------
# PROJECT PATH BOOTSTRAP
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # ...\MDEV_EMTAC

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from modules.orchestrators.image_orchestrator import ImageOrchestrator
from modules.configuration.log_config import logger
from modules.configuration.config import (
    DATABASE_DIR,
    DATABASE_PATH_IMAGES_FOLDER,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TEST_FOLDER = Path(r"E:\emtac\data\raw_documention\test_doc")

# Set this to force a specific image, or leave as None to auto-pick first image
TEST_IMAGE_NAME: Optional[str] = None

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

TEST_METADATA: Dict[str, Any] = {
    # Leave title blank to confirm orchestrator fallback uses filename stem
    "title": "",
    "description": "",
    "img_metadata": {
        "source": "image_orchestrator_end_to_end_test",
        "test_script": "scripts/image_embedding_service_run.py",
        "purpose": "verify image upload + embedding + db persistence",
    },
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def find_test_image(folder: Path, file_name: Optional[str] = None) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Test folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Test folder is not a directory: {folder}")

    if file_name:
        candidate = folder / file_name
        if not candidate.exists():
            raise FileNotFoundError(f"Requested test image not found: {candidate}")

        if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise RuntimeError(
                f"Requested file is not a supported image type: {candidate.name}"
            )

        return candidate

    images = sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    if not images:
        raise RuntimeError(
            f"No supported image files found in folder: {folder}\n"
            f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    return images[0]


def guess_content_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }

    return mapping.get(suffix, "application/octet-stream")


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_kv(label: str, value: Any) -> None:
    print(f"{label:<28}: {value}")


def summarize_upload_result(result: Dict[str, Any]) -> None:
    print_header("UPLOAD RESULT SUMMARY")

    print_kv("status", result.get("status"))
    print_kv("message", result.get("message"))
    print_kv("processed", result.get("processed"))
    print_kv("failed", result.get("failed"))
    print()

    for idx, item in enumerate(result.get("results", []), start=1):
        print(f"Result #{idx}")
        print("-" * 80)
        print_kv("success", item.get("success"))
        print_kv("file_name", item.get("file_name"))
        print_kv("file_path", item.get("file_path"))
        print_kv("image_id", item.get("image_id"))
        print_kv("http_status", item.get("http_status"))

        if item.get("error"):
            print_kv("error", item.get("error"))

        if item.get("detail"):
            print_kv("detail", item.get("detail"))

        result_payload = item.get("result") or {}
        image_payload = result_payload.get("image") or {}
        embedding_payload = result_payload.get("embedding") or {}

        if image_payload:
            print_kv("image.title", image_payload.get("title"))
            print_kv("image.description", image_payload.get("description"))
            print_kv("image.file_path", image_payload.get("file_path"))

        if embedding_payload:
            print_kv("embedding.id", embedding_payload.get("id"))
            print_kv("embedding.image_id", embedding_payload.get("image_id"))
            print_kv("embedding.model_name", embedding_payload.get("model_name"))

        print()


def main() -> None:
    print_header("ImageOrchestrator end-to-end test")

    print_kv("Project root", PROJECT_ROOT)
    print_kv("Test folder", TEST_FOLDER)
    print_kv("Database dir", DATABASE_DIR)
    print_kv("Images folder", DATABASE_PATH_IMAGES_FOLDER)
    print()

    test_image = find_test_image(TEST_FOLDER, TEST_IMAGE_NAME)

    print_kv("Selected image", test_image)
    print_kv("Selected suffix", test_image.suffix.lower())
    print()

    orchestrator = ImageOrchestrator()

    print("Clearing image model cache before test...")
    orchestrator.image_model_service.clear_model_cache()
    print()

    print("Opening test image as FileStorage and sending through orchestrator...")
    with open(test_image, "rb") as fh:
        file_obj = FileStorage(
            stream=fh,
            filename=test_image.name,
            name="files",
            content_type=guess_content_type(test_image),
        )

        upload_result = orchestrator.process_upload(
            files=[file_obj],
            metadata=TEST_METADATA,
        )

    summarize_upload_result(upload_result)

    # -----------------------------------------------------
    # VERIFY SUCCESS
    # -----------------------------------------------------
    results = upload_result.get("results", [])
    if not results:
        raise RuntimeError("No per-file results returned from orchestrator")

    first_result = results[0]
    if not first_result.get("success"):
        raise RuntimeError(
            f"Upload failed: {first_result.get('error') or first_result.get('detail') or 'unknown error'}"
        )

    image_id = first_result.get("image_id")
    if not image_id:
        raise RuntimeError("Upload succeeded but no image_id was returned")

    print_header("VERIFYING IMAGE GRAPH FROM DATABASE")

    graph = orchestrator.resolve_image_graph(image_id=image_id)

    print_kv("graph.status", graph.get("status"))
    print_kv("graph.image_id", graph.get("image_id"))

    image_payload = graph.get("image") or {}
    embeddings = graph.get("embeddings") or []
    documents = graph.get("documents") or []
    positions = graph.get("positions") or []
    tasks = graph.get("tasks") or []
    problems = graph.get("problems") or []

    print_kv("image.title", image_payload.get("title"))
    print_kv("image.description", image_payload.get("description"))
    print_kv("image.file_path", image_payload.get("file_path"))
    print_kv("embeddings.count", len(embeddings))
    print_kv("documents.count", len(documents))
    print_kv("positions.count", len(positions))
    print_kv("tasks.count", len(tasks))
    print_kv("problems.count", len(problems))
    print()

    if embeddings:
        print("First embedding row:")
        pprint(embeddings[0], sort_dicts=False)
        print()
    else:
        raise RuntimeError(
            "Image row was created, but no embedding rows were found in resolve_image_graph()"
        )

    print_header("TEST PASSED")
    print("The image orchestrator completed end-to-end processing successfully.")
    print("This confirms:")
    print("  1. file save worked")
    print("  2. image row was created")
    print("  3. image embedding was generated")
    print("  4. image_embedding row was stored")
    print("  5. DB verification readback succeeded")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("image_orchestrator_end_to_end_test failed")
        print()
        print_header("TEST FAILED")
        print(f"Error: {exc}")
        raise