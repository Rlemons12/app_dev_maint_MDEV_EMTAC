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

from contextlib import contextmanager

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
    """

    def __init__(self):
        self._db_config = get_db_config()

    # -----------------------------------------------------
    # Logging Helpers
    # -----------------------------------------------------

    def _rid(self):
        return get_request_id()

    def _debug(self, message: str):
        debug_id(message, self._rid())

    def _info(self, message: str):
        info_id(message, self._rid())

    def _warning(self, message: str):
        warning_id(message, self._rid())

    def _error(self, message: str):
        error_id(message, self._rid())

    def _timed(self, operation_name: str):
        return log_timed_operation(operation_name, self._rid())

    # -----------------------------------------------------
    # Transaction Boundary
    # -----------------------------------------------------

    @contextmanager
    def transaction(self):
        """
        Workflow-level transactional boundary.

        Example:

            with self.transaction() as session:
                self.document_service.save(session=session, ...)
                self.embedding_service.add_pgvector(session=session, ...)

        Orchestrator owns:
            - Session lifecycle
            - Commit / rollback (via context manager)
        """

        with self._db_config.main_session() as session:
            yield session
