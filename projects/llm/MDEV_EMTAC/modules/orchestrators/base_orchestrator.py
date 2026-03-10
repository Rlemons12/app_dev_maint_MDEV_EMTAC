"""
Base Orchestrator (Transaction Aware)

Responsibilities:
    - Provide transactional boundary
    - Integrate with logging + request ID
    - Remain transport agnostic
    - Provide common orchestration utilities

Does NOT:
    - Own services
    - Access global registry
    - Perform domain logic
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Any

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    log_timed_operation,
    get_request_id,
)

from modules.configuration.config_env import get_db_config


class BaseOrchestrator:
    """
    Infrastructure-level orchestrator base class.

    Child orchestrators must explicitly instantiate
    and manage their own services.

    Transaction ownership rules:
        - Orchestrator owns session lifecycle
        - Orchestrator owns commit / rollback
        - Services must never commit / rollback / close sessions
    """

    def __init__(self):
        self._db_config = get_db_config()

    # -----------------------------------------------------
    # Logging Helpers
    # -----------------------------------------------------

    def _rid(self) -> str:
        return get_request_id()

    def _debug(self, message: str) -> None:
        debug_id(message, self._rid())

    def _info(self, message: str) -> None:
        info_id(message, self._rid())

    def _warning(self, message: str) -> None:
        warning_id(message, self._rid())

    def _error(self, message: str, *, exc_info: bool = False) -> None:
        error_id(message, self._rid(), exc_info=exc_info)

    def _timed(self, operation_name: str):
        return log_timed_operation(operation_name, self._rid())

    # -----------------------------------------------------
    # Transaction Boundary
    # -----------------------------------------------------

    @contextmanager
    def transaction(
        self,
        *,
        read_only: bool = False,
        operation_name: str | None = None,
    ) -> Generator[Any, None, None]:
        """
        Workflow-level transactional boundary.

        Example:
            with self.transaction() as session:
                self.document_service.save(session=session, ...)
                self.embedding_service.add_pgvector(session=session, ...)

        Read-only example:
            with self.transaction(read_only=True) as session:
                row = self.document_service.get(session=session, ...)

        Orchestrator owns:
            - Session lifecycle
            - Commit / rollback
            - Read-only vs write semantics

        Notes:
            - For read_only=True, no commit is issued.
            - On exception, rollback is attempted.
            - Session is always closed by the underlying session context.
        """

        rid = self._rid()
        op_name = operation_name or self.__class__.__name__

        debug_id(
            f"[{op_name}] Opening {'read-only' if read_only else 'write'} transaction",
            rid,
        )

        with self._db_config.main_session() as session:
            try:
                yield session

                if read_only:
                    debug_id(f"[{op_name}] Read-only transaction complete", rid)
                else:
                    session.commit()
                    debug_id(f"[{op_name}] Transaction committed", rid)

            except Exception as exc:
                try:
                    session.rollback()
                    warning_id(
                        f"[{op_name}] Transaction rolled back due to error: {exc}",
                        rid,
                    )
                except Exception as rollback_exc:
                    error_id(
                        f"[{op_name}] Rollback failed after error: {rollback_exc}",
                        rid,
                        exc_info=True,
                    )

                raise

            finally:
                debug_id(f"[{op_name}] Transaction scope closed", rid)