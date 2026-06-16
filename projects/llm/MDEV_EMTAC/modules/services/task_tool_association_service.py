from typing import Optional, List, Tuple

from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    error_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Task,
    Tool,
    TaskToolAssociation,
)


class TaskToolAssociationService:
    """
    Service layer for TaskToolAssociation.

    Responsibilities:
    - Associate / dissociate tasks and tools
    - Query tools by task
    - Query tasks by tool

    No ORM joins leak outside this layer.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # --------------------------------------------------
    # Session handling
    # --------------------------------------------------
    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # --------------------------------------------------
    # Association management
    # --------------------------------------------------
    @with_request_id
    def associate(
        self,
        task_id: int,
        tool_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[TaskToolAssociation]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("TaskToolAssociationService.associate", rid):
                return TaskToolAssociation.associate_task_tool(
                    task_id=task_id,
                    tool_id=tool_id,
                    request_id=rid,
                    session=sess,
                )
        except Exception as e:
            error_id(
                f"Failed to associate task_id={task_id} tool_id={tool_id}: {e}",
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
        tool_id: int,
        session: Optional[SASession] = None,
    ) -> bool:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("TaskToolAssociationService.dissociate", rid):
                return TaskToolAssociation.dissociate_task_tool(
                    task_id=task_id,
                    tool_id=tool_id,
                    request_id=rid,
                    session=sess,
                )
        finally:
            if created_here:
                sess.close()

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------
    @with_request_id
    def get_tools_for_task(
        self,
        task_id: int,
        session: Optional[SASession] = None,
    ) -> List[Tool]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("TaskToolAssociationService.get_tools_for_task", rid):
                tools = TaskToolAssociation.get_tools_by_task(
                    task_id=task_id,
                    request_id=rid,
                    session=sess,
                )
                info_id(f"Found {len(tools)} tools for task_id={task_id}", rid)
                return tools
        except Exception as e:
            error_id(f"Error getting tools for task_id={task_id}: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_tasks_for_tool(
        self,
        tool_id: int,
        session: Optional[SASession] = None,
    ) -> List[Task]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("TaskToolAssociationService.get_tasks_for_tool", rid):
                tasks = TaskToolAssociation.get_tasks_by_tool(
                    tool_id=tool_id,
                    request_id=rid,
                    session=sess,
                )
                info_id(f"Found {len(tasks)} tasks for tool_id={tool_id}", rid)
                return tasks
        except Exception as e:
            error_id(f"Error getting tasks for tool_id={tool_id}: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()
