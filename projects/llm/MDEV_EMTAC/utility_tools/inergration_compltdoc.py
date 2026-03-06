import os
import sys
from werkzeug.datastructures import FileStorage

# Adjust if needed so project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.emtac_ai.orchestrators.complete_document_orchestrator import (
    CompleteDocumentOrchestrator,
)

PDF_PATH = r"E:\emtac\data\raw_documention\FB4-GENERAL\Align Lot Change Checks.pdf"


def load_pdf(path: str) -> FileStorage:
    """
    Wrap a real file exactly like Flask would.
    """
    file_stream = open(path, "rb")
    return FileStorage(
        stream=file_stream,
        filename=os.path.basename(path),
        content_type="application/pdf",
    )


def main():
    print("==========================================")
    print("CompleteDocumentOrchestrator Standalone")
    print("==========================================")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Instantiate real orchestrator
    orchestrator = CompleteDocumentOrchestrator()

    # Prepare upload object
    file_obj = load_pdf(PDF_PATH)

    # Metadata must match your _resolve_position expectations
    metadata = {
        "position_id": 1,   # change if needed
        "uploaded_by": "standalone_script",
    }

    try:
        result = orchestrator.process_upload(
            files=[file_obj],
            metadata=metadata,
        )

        print("\nPipeline Completed Successfully\n")
        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as e:
        print("\nPipeline Failed\n")
        print(str(e))
        raise

    finally:
        file_obj.stream.close()


if __name__ == "__main__":
    main()
