from __future__ import annotations

import hashlib
import os
from typing import Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
    get_request_id,
)


class UploadIdempotencyService:
    """
    Detects if a file has already been ingested.

    Strategy:
      1) Try complete_document.file_sha256 (if column exists)
      2) Fallback to file_path match

    HARD RULES:
      - No session creation
      - No outer commit/rollback
      - Safe against transaction poisoning
    """

    # =========================================================
    # SIGNATURE GENERATION
    # =========================================================
    @with_request_id
    def compute_signature(
        self,
        *,
        file_path: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        rid = request_id or get_request_id()

        sha = self._sha256_file(file_path)
        size = os.path.getsize(file_path) if os.path.exists(file_path) else None

        sig = {
            "file_sha256": sha,
            "file_size": size,
            "file_basename": os.path.basename(file_path),
        }

        debug_id(f"[IDEMPOTENT] sig={sig}", rid)
        return sig

    # =========================================================
    # LOOKUP
    # =========================================================
    @with_request_id
    def find_existing_complete_document_id(
        self,
        *,
        session,
        file_sha256: str,
        file_path: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[int]:

        rid = request_id or get_request_id()

        if not file_sha256:
            return None

        # -----------------------------------------------------
        # 1️⃣ Try using file_sha256 column safely
        # -----------------------------------------------------
        savepoint = session.begin_nested()

        try:
            sql = text("""
                SELECT id
                FROM complete_document
                WHERE file_sha256 = :sha
                ORDER BY id DESC
                LIMIT 1
            """)

            row = session.execute(sql, {"sha": file_sha256}).fetchone()

            savepoint.commit()

            if row:
                info_id(
                    f"[IDEMPOTENT] found via file_sha256 id={row[0]}",
                    rid,
                )
                return int(row[0])

        except ProgrammingError:
            # Column does not exist
            savepoint.rollback()

            warning_id(
                "[IDEMPOTENT] file_sha256 column not found — falling back to file_path",
                rid,
            )

        except SQLAlchemyError:
            # Any other SQL failure inside savepoint
            savepoint.rollback()

        # -----------------------------------------------------
        # 2️⃣ Fallback to file_path (safe)
        # -----------------------------------------------------
        if file_path:
            try:
                sql = text("""
                    SELECT id
                    FROM complete_document
                    WHERE file_path = :path
                    ORDER BY id DESC
                    LIMIT 1
                """)

                row = session.execute(sql, {"path": file_path}).fetchone()

                if row:
                    info_id(
                        f"[IDEMPOTENT] found via file_path id={row[0]}",
                        rid,
                    )
                    return int(row[0])

            except SQLAlchemyError:
                # Do NOT abort outer transaction
                warning_id(
                    "[IDEMPOTENT] file_path fallback lookup failed",
                    rid,
                )

        return None

    # =========================================================
    # HASH HELPER
    # =========================================================
    def _sha256_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()