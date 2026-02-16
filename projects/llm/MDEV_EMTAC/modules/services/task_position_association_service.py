# modules/services/task_position_association_service.py

from typing import Optional, List, Tuple, Dict
from sqlalchemy import func
from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Task,
    Position,
    TaskPositionAssociation,
)


class TaskPositionAssociationService:
    """
    Service layer for TaskPositionAssociation.

    Responsibilities:
    - Associate / dissociate tasks and positions
    - Query tasks by position
    - Query positions by task
    - Batch association helpers

    NO ORM joins outside this layer.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------
    # Session handling
    # ---------------------------------------------------------
    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------
    @with_request_id
    def get_tasks_for_position(
        self,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> List[Task]:
        """
        Return all Tasks linked to a Position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.get_tasks_for_position", rid
            ):
                tasks = TaskPositionAssociation.get_tasks_by_position_id(
                    session=sess,
                    position_id=position_id,
                )
                info_id(
                    f"Found {len(tasks)} tasks for position_id={position_id}",
                    rid,
                )
                return tasks

        except Exception as e:
            error_id(
                f"Error getting tasks for position_id={position_id}: {e}",
                rid,
                exc_info=True,
            )
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_positions_for_task(
        self,
        task_id: int,
        session: Optional[SASession] = None,
    ) -> List[Position]:
        """
        Return all Positions linked to a Task.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.get_positions_for_task", rid
            ):
                positions = TaskPositionAssociation.get_positions_by_task_id(
                    session=sess,
                    task_id=task_id,
                )
                info_id(
                    f"Found {len(positions)} positions for task_id={task_id}",
                    rid,
                )
                return positions

        except Exception as e:
            error_id(
                f"Error getting positions for task_id={task_id}: {e}",
                rid,
                exc_info=True,
            )
            return []
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Association management
    # ---------------------------------------------------------
    @with_request_id
    def associate(
        self,
        task_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[TaskPositionAssociation]:
        """
        Associate a Task with a Position (idempotent).
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.associate", rid
            ):
                assoc = TaskPositionAssociation.associate_task_position(
                    task_id=task_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
                return assoc

        except Exception as e:
            error_id(
                f"Error associating task_id={task_id} position_id={position_id}: {e}",
                rid,
                exc_info=True,
            )
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def dissociate(
        self,
        task_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> bool:
        """
        Remove a Task–Position association.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.dissociate", rid
            ):
                return TaskPositionAssociation.dissociate_task_position(
                    task_id=task_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )

        except Exception as e:
            error_id(
                f"Error dissociating task_id={task_id} position_id={position_id}: {e}",
                rid,
                exc_info=True,
            )
            return False
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Batch helpers
    # ---------------------------------------------------------
    @with_request_id
    def associate_tasks_to_position(
        self,
        task_ids: List[int],
        position_id: int,
        session: Optional[SASession] = None,
    ) -> Dict[int, bool]:
        """
        Associate many Tasks to one Position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.associate_tasks_to_position", rid
            ):
                return TaskPositionAssociation.associate_multiple_tasks_to_position(
                    task_ids=task_ids,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def associate_task_to_positions(
        self,
        task_id: int,
        position_ids: List[int],
        session: Optional[SASession] = None,
    ) -> Dict[int, bool]:
        """
        Associate one Task to many Positions.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.associate_task_to_positions", rid
            ):
                return TaskPositionAssociation.associate_task_to_multiple_positions(
                    task_id=task_id,
                    position_ids=position_ids,
                    request_id=rid,
                    session=sess,
                )
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # REQUIRED BATCH QUERY (USED BY CHUNK SEARCH)
    # ---------------------------------------------------------
    @with_request_id
    def get_tasks_for_positions(
        self,
        position_ids: List[int],
        session: Optional[SASession] = None,
    ) -> List[Task]:
        """
        Return all unique Tasks linked to multiple Positions.

        Required for ChunkAssociationSearch tier-2 enrichment.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[TaskPositionAssociationService.get_tasks_for_positions] "
                "No position_ids provided",
                rid,
            )
            return []

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "TaskPositionAssociationService.get_tasks_for_positions", rid
            ):
                tasks = (
                    sess.query(Task)
                    .join(
                        TaskPositionAssociation,
                        Task.id == TaskPositionAssociation.task_id,
                    )
                    .filter(
                        TaskPositionAssociation.position_id.in_(position_ids)
                    )
                    .distinct()
                    .all()
                )

                debug_id(
                    f"Found {len(tasks)} tasks for "
                    f"{len(position_ids)} positions",
                    rid,
                )

                return tasks

        except Exception as e:
            error_id(
                f"Error getting tasks for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return []
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # LIGHTWEIGHT COUNTS (USED BY CHUNK GRAPH / PROBES)
    # ---------------------------------------------------------
    @with_request_id
    def count_tasks(
            self,
            position_ids: List[int],
            session: Optional[SASession] = None,
    ) -> int:
        """
        Count unique Tasks associated with given Positions.

        Lightweight helper for relationship summaries.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[TaskPositionAssociationService.count_tasks] No position_ids provided",
                rid,
            )
            return 0

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "TaskPositionAssociationService.count_tasks", rid
            ):
                count = (
                    sess.query(func.count(func.distinct(TaskPositionAssociation.task_id)))
                    .filter(
                        TaskPositionAssociation.position_id.in_(position_ids)
                    )
                    .scalar()
                )

                debug_id(
                    f"Counted {count} unique tasks for {len(position_ids)} positions",
                    rid,
                )

                return count or 0

        except Exception as e:
            error_id(
                f"Error counting tasks for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return 0
        finally:
            if created_here:
                sess.close()
