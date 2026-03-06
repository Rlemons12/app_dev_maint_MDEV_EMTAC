import os
import sys
import logging
from pathlib import Path
from pprint import pprint
from datetime import datetime

# ------------------------------------------------------------
# Ensure project root is on path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------
# Force DEBUG logging for trace
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from modules.configuration.log_config import get_request_id
from modules.configuration.config_env import get_db_config
from modules.applications.file_processing_coordinator import FileProcessingCoordinator

from modules.emtacdb.emtacdb_fts import (
    CompleteDocument,
    Document,
    DocumentEmbedding,
    Image,
    ImageCompletedDocumentAssociation,
    CompletedDocumentPositionAssociation,
    Position,
)

# ------------------------------------------------------------
# Simple file-like wrapper
# ------------------------------------------------------------
class SimpleFile:
    def __init__(self, path: str):
        self.path = path
        self.filename = Path(path).name

    def read(self):
        with open(self.path, "rb") as f:
            return f.read()

# ------------------------------------------------------------
# Session helper (matches your DatabaseConfig reality)
# ------------------------------------------------------------
def _open_session(db_config):
    """
    Your DatabaseConfig does NOT expose get_session().
    In your logs it suggests a registry-style session: db_config.main_session

    This helper:
    - Uses main_session when present
    - Falls back to get_session if you later add it
    - Returns (session, should_close)
    """
    if hasattr(db_config, "main_session") and db_config.main_session is not None:
        return db_config.main_session, True

    if hasattr(db_config, "get_session"):
        # In case you add it later; supports context-manager style if it exists.
        return db_config.get_session(), False  # caller must handle context manager usage

    raise AttributeError(
        "DatabaseConfig has neither 'main_session' nor 'get_session'. "
        "Expected db_config.main_session based on your current stack."
    )

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------
def verify_relationships(success, result):

    print("\n" + "=" * 60)
    print(" DATABASE RELATIONSHIP VERIFICATION ")
    print("=" * 60)

    if not success:
        print("Pipeline reported failure — skipping verification.")
        return

    if not result or not result.get("document_ids"):
        print("No document_ids returned — nothing to verify.")
        return

    db_config = get_db_config()

    # ✅ CORRECT — use context manager
    with db_config.main_session() as session:
        _verify_relationships_with_session(session, result)

    print("\n" + "=" * 60)
    print(" RELATIONSHIP TRACE COMPLETE ")
    print("=" * 60)


def _verify_relationships_with_session(session, result):
    """
    Prints relationship chains:
    CompleteDocument -> Position
    CompleteDocument -> Chunks -> Embeddings
    CompleteDocument -> Images
    """

    for complete_id in result["document_ids"]:

        print("\n" + "-" * 60)
        print(f"CompleteDocument ID: {complete_id}")
        print("-" * 60)

        complete = session.get(CompleteDocument, complete_id)

        if not complete:
            print("CompleteDocument not found in DB.")
            continue

        print("Title     :", getattr(complete, "title", None))
        print("File Path :", getattr(complete, "file_path", None))

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------
        pos_links = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        position_ids = [p.position_id for p in pos_links]

        print("\nPosition Links:", len(pos_links))
        print("Position IDs  :", position_ids or "None")

        if position_ids:
            positions = (
                session.query(Position)
                .filter(Position.id.in_(position_ids))
                .all()
            )

            for p in positions:
                details = []
                for k in [
                    "site_location_id",
                    "area_id",
                    "equipment_group_id",
                    "model_id",
                    "asset_number_id",
                    "location_id",
                ]:
                    if hasattr(p, k):
                        details.append(f"{k}={getattr(p, k)}")

                print(
                    f"  Position {p.id} → "
                    + (", ".join(details) if details else "(no FK fields found)")
                )

        # ----------------------------------------------------
        # CHUNKS
        # ----------------------------------------------------
        chunks = (
            session.query(Document)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        chunk_ids = [c.id for c in chunks]

        print("\nChunks Created:", len(chunk_ids))
        print("Chunk IDs     :", chunk_ids or "None")

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------
        embeddings = []

        if chunk_ids:
            embeddings = (
                session.query(DocumentEmbedding)
                .filter(DocumentEmbedding.document_id.in_(chunk_ids))
                .all()
            )

        print("\nEmbeddings Created:", len(embeddings))

        for emb in embeddings:
            dims = getattr(emb, "actual_dimensions", None)
            extra = f", dims={dims}" if dims else ""
            print(
                f"  Embedding {emb.id} "
                f"→ document_id={emb.document_id} "
                f"→ model={emb.model_name}{extra}"
            )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------
        image_links = (
            session.query(ImageCompletedDocumentAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        image_ids = [l.image_id for l in image_links]

        print("\nImage Links:", len(image_links))
        print("Image IDs  :", image_ids or "None")

        for link in image_links:
            print(
                f"  Assoc {link.id} → image_id={link.image_id}"
            )

        if image_ids:
            images = (
                session.query(Image)
                .filter(Image.id.in_(image_ids))
                .all()
            )

            for img in images:
                print(
                    f"  Image {img.id} "
                    f"→ title='{img.title}' "
                    f"→ file_path='{img.file_path}'"
                )


def _verify_relationships_with_session(session, result):

    for complete_id in result["document_ids"]:

        print("\n" + "-" * 60)
        print(f"CompleteDocument ID: {complete_id}")
        print("-" * 60)

        complete = session.get(CompleteDocument, complete_id)

        if not complete:
            print("CompleteDocument not found in DB.")
            continue

        print("Title     :", complete.title)
        print("File Path :", complete.file_path)

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------
        pos_links = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        position_ids = [p.position_id for p in pos_links]
        print("Position IDs:", position_ids or "None")

        # ----------------------------------------------------
        # CHUNKS
        # ----------------------------------------------------
        chunks = (
            session.query(Document)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        chunk_ids = [c.id for c in chunks]
        print("Chunks Created:", len(chunk_ids))
        print("Chunk IDs     :", chunk_ids or "None")

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------
        if chunk_ids:
            embeddings = (
                session.query(DocumentEmbedding)
                .filter(DocumentEmbedding.document_id.in_(chunk_ids))
                .all()
            )
        else:
            embeddings = []

        print("Embeddings Created:", len(embeddings))

        for emb in embeddings:
            print(
                f"  Embedding ID {emb.id} "
                f"→ document_id={emb.document_id} "
                f"→ model={emb.model_name}"
            )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------
        image_links = (
            session.query(ImageCompletedDocumentAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        image_ids = [l.image_id for l in image_links]
        print("Images Associated:", image_ids or "None")

        if image_ids:
            images = (
                session.query(Image)
                .filter(Image.id.in_(image_ids))
                .all()
            )

            for img in images:
                print(
                    f"  Image ID {img.id} "
                    f"→ title='{img.title}' "
                    f"→ file_path='{img.file_path}'"
                )

    print("\n" + "=" * 60)
    print(" RELATIONSHIP TRACE COMPLETE ")
    print("=" * 60)

def _verify_relationships_with_session(session, result):
    """
    Uses the provided SQLAlchemy session to print relationship chains:
    CompleteDocument -> CompletedDocumentPositionAssociation -> Position
    CompleteDocument -> Document (chunks) -> DocumentEmbedding
    CompleteDocument -> ImageCompletedDocumentAssociation -> Image
    """

    for complete_id in result["document_ids"]:

        print("\n" + "-" * 60)
        print(f"CompleteDocument ID: {complete_id}")
        print("-" * 60)

        complete = session.get(CompleteDocument, complete_id)

        if not complete:
            print("CompleteDocument not found in DB.")
            continue

        print("Title     :", getattr(complete, "title", None))
        print("File Path :", getattr(complete, "file_path", None))

        # ----------------------------------------------------
        # POSITION RELATIONSHIP (through association table)
        # ----------------------------------------------------
        pos_links = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        position_ids = [p.position_id for p in pos_links]
        print("\nPosition Links:", len(pos_links))
        print("Position IDs  :", position_ids or "None")

        if position_ids:
            positions = (
                session.query(Position)
                .filter(Position.id.in_(position_ids))
                .all()
            )
            for p in positions:
                # Print the FK chain that created the Position row (if present on model)
                details = []
                for k in [
                    "site_location_id",
                    "area_id",
                    "equipment_group_id",
                    "model_id",
                    "asset_number_id",
                    "location_id",
                ]:
                    if hasattr(p, k):
                        details.append(f"{k}={getattr(p, k)}")
                print(f"  Position {p.id} → " + (", ".join(details) if details else "(no FK fields found on model)"))

        # ----------------------------------------------------
        # CHUNKS
        # ----------------------------------------------------
        chunks = (
            session.query(Document)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        chunk_ids = [c.id for c in chunks]
        print("\nChunks Created:", len(chunk_ids))
        print("Chunk IDs     :", chunk_ids or "None")

        # ----------------------------------------------------
        # CHUNK → EMBEDDINGS
        # ----------------------------------------------------
        if chunk_ids:
            embeddings = (
                session.query(DocumentEmbedding)
                .filter(DocumentEmbedding.document_id.in_(chunk_ids))
                .all()
            )
        else:
            embeddings = []

        print("\nEmbeddings Created:", len(embeddings))
        for emb in embeddings:
            model_name = getattr(emb, "model_name", None)
            dims = getattr(emb, "actual_dimensions", None) if hasattr(emb, "actual_dimensions") else None
            extra = f", dims={dims}" if dims is not None else ""
            print(
                f"  Embedding ID {emb.id} "
                f"→ document_id={emb.document_id} "
                f"→ model={model_name}{extra}"
            )

        # ----------------------------------------------------
        # IMAGES (CompleteDocument Level)
        # ----------------------------------------------------
        image_links = (
            session.query(ImageCompletedDocumentAssociation)
            .filter_by(complete_document_id=complete_id)
            .all()
        )

        image_ids = [l.image_id for l in image_links]
        print("\nImage Links      :", len(image_links))
        print("Image IDs        :", image_ids or "None")

        # If the association model has more info, print it too
        for l in image_links:
            parts = [f"assoc_id={getattr(l, 'id', None)}", f"image_id={l.image_id}", f"complete_document_id={complete_id}"]
            if hasattr(l, "document_id"):
                parts.append(f"document_id={getattr(l, 'document_id')}")
            if hasattr(l, "page_number"):
                parts.append(f"page_number={getattr(l, 'page_number')}")
            print("  ImageLink → " + " | ".join(parts))

        if image_ids:
            images = (
                session.query(Image)
                .filter(Image.id.in_(image_ids))
                .all()
            )

            for img in images:
                print(
                    f"  Image ID {img.id} "
                    f"→ title='{getattr(img, 'title', None)}' "
                    f"→ file_path='{getattr(img, 'file_path', None)}'"
                )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    print("\n" + "=" * 60)
    print(" EMTAC INGESTION TRACE TEST ")
    print("=" * 60)

    # --------------------------------------------------------
    # Prompt for file path
    # --------------------------------------------------------
    user_path = input("\nEnter FULL file path to process: ").strip()

    if not user_path:
        print("No file path entered.")
        return

    # Strip quotes if user pasted them
    user_path = user_path.strip().strip('"').strip("'")
    file_path = Path(os.path.normpath(user_path))

    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return

    print(f"\nFile: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print("-" * 60)

    metadata = {
        "area_id": 1,
        "equipment_group_id": 1,
        "model_id": 1,
        "asset_number_id": 1,
        "location_id": 1,
        "title": file_path.stem,
        "tags": "standalone_test",
    }

    coordinator = FileProcessingCoordinator()
    file_obj = SimpleFile(str(file_path))

    start_time = datetime.now()

    success, result, status = coordinator.process_upload(
        files=[file_obj],
        metadata=metadata,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(" PIPELINE RESULT ")
    print("=" * 60)

    print("Request ID :", get_request_id())
    print("Success    :", success)
    print("HTTP Status:", status)
    print("Duration   :", f"{duration:.2f} seconds")
    print("-" * 60)

    pprint(result)

    verify_relationships(success, result)

    print("\n" + "=" * 60)
    print(" TRACE COMPLETE ")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()