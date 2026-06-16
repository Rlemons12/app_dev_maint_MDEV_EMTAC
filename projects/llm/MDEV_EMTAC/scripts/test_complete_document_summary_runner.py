from __future__ import annotations

import argparse
from typing import List, Optional

from sqlalchemy import text

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id
from modules.services.complete_document_summary_service import (
    CompleteDocumentSummaryService,
)


def get_target_document_ids(
    session,
    *,
    document_id: Optional[int],
    limit: int,
    missing_only: bool,
) -> List[int]:
    if document_id:
        return [document_id]

    if missing_only:
        sql = text(
            """
            SELECT id
            FROM public.complete_document
            WHERE summary IS NULL
               OR TRIM(summary) = ''
            ORDER BY id DESC
            LIMIT :limit
            """
        )
    else:
        sql = text(
            """
            SELECT id
            FROM public.complete_document
            ORDER BY id DESC
            LIMIT :limit
            """
        )

    rows = session.execute(sql, {"limit": limit}).fetchall()
    return [int(row[0]) for row in rows]


def run_summary_test(
    *,
    document_id: Optional[int],
    limit: int,
    force: bool,
    missing_only: bool,
    store_summary_embedding: bool,
) -> None:
    request_id = "test-complete-document-summary-runner"

    db_config = DatabaseConfig()
    summary_service = CompleteDocumentSummaryService(
        store_summary_embedding=store_summary_embedding,
    )

    try:
        with db_config.main_session() as session:
            document_ids = get_target_document_ids(
                session,
                document_id=document_id,
                limit=limit,
                missing_only=missing_only,
            )

            if not document_ids:
                print("No documents found to summarize.")
                return

            print(f"Documents selected: {document_ids}")

            for complete_document_id in document_ids:
                print("=" * 80)
                print(f"Summarizing complete_document_id={complete_document_id}")

                result = summary_service.summarize_complete_document_chunks(
                    session=session,
                    complete_document_id=complete_document_id,
                    force=force,
                    request_id=request_id,
                )

                session.commit()

                print(f"Status: {result.get('status')}")
                print(f"Success: {result.get('success')}")
                print(f"Chunks available: {result.get('chunks_available')}")
                print(f"Chunks summarized: {result.get('chunks_summarized')}")
                print(f"Summary embedding stored: {result.get('summary_embedding_stored')}")
                print()
                print("Summary:")
                print(result.get("summary") or "[no summary returned]")

                info_id(
                    f"Summary test completed for complete_document_id={complete_document_id}: {result}",
                    request_id,
                )

    except Exception as e:
        error_id(
            f"Summary test runner failed: {e}",
            request_id,
            exc_info=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test CompleteDocument chunk-based summarization."
    )

    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Specific complete_document.id to summarize.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of documents to process when --document-id is not provided.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize even if summary already exists.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Include documents that already have summaries.",
    )

    parser.add_argument(
        "--no-summary-embedding",
        action="store_true",
        help="Skip storing summary embedding.",
    )

    args = parser.parse_args()

    run_summary_test(
        document_id=args.document_id,
        limit=args.limit,
        force=args.force,
        missing_only=not args.all,
        store_summary_embedding=not args.no_summary_embedding,
    )


if __name__ == "__main__":
    main()