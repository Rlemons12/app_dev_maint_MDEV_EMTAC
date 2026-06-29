from __future__ import annotations

import argparse
import json

from modules.database_manager.orchestrators.complete_document_summary_maintenance_orchestrator import (
    CompleteDocumentSummaryMaintenanceOrchestrator,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean up complete_document summaries, clean RAG metadata, "
            "retrieval text, and summary embeddings."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to process per batch.",
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Maximum documents to process. 0 means no limit.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-summarize documents even if summaries already exist. "
            "Use this to replace older polluted summaries."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview target documents without changing the database.",
    )

    parser.add_argument(
        "--missing-summary-only",
        action="store_true",
        help=(
            "Only process rows missing summary/rag_metadata. "
            "By default, rows missing summary_embedding_vector are also processed."
        ),
    )

    parser.add_argument(
        "--no-summary-embedding",
        action="store_true",
        help="Do not create/store summary embeddings.",
    )

    args = parser.parse_args()

    orchestrator = CompleteDocumentSummaryMaintenanceOrchestrator(
        store_summary_embedding=not args.no_summary_embedding,
    )

    result = orchestrator.run(
        batch_size=args.batch_size,
        max_documents=args.max_documents,
        force=args.force,
        dry_run=args.dry_run,
        include_missing_embedding=not args.missing_summary_only,
        request_id="run-complete-document-summary-maintenance",
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()