from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import (
    debug_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Problem,
    ProblemPositionAssociation
)


class TroubleshootingService:

    def __init__(self):
        self.db_config = get_db_config()

    # ---------------------------------------------------------
    # Session Helper
    # ---------------------------------------------------------

    def _get_session(self, session: Optional[Session]) -> Tuple[Session, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # CRUD (PROBLEM)
    # ---------------------------------------------------------

    @with_request_id
    def add_problem(self, name: str, description: str, session: Optional[Session] = None) -> int:

        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            problem_id = Problem.add_to_db(sess, name=name, description=description)

            if created_here:
                sess.commit()

            return problem_id

        except Exception as e:
            if created_here:
                sess.rollback()
            error_id(f"Error adding problem: {e}", rid, exc_info=True)
            raise
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_problem(self, problem_id: int, session: Optional[Session] = None):

        sess, created_here = self._get_session(session)
        try:
            return Problem.get_by_id(sess, problem_id)
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    @with_request_id
    def search_problems(self, text: str, exact: bool = False, limit: int = 50,
                        session: Optional[Session] = None):

        sess, created_here = self._get_session(session)
        try:
            return Problem.search(sess, text=text, exact=exact, limit=limit)
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # ASSOCIATIONS
    # ---------------------------------------------------------

    @with_request_id
    def attach_problem_to_position(self, problem_id: int, position_id: int,
                                   session: Optional[Session] = None):

        sess, created_here = self._get_session(session)

        try:
            assoc = ProblemPositionAssociation.add_to_db(
                session=sess,
                problem_id=problem_id,
                position_id=position_id
            )

            if created_here:
                sess.commit()

            return assoc

        except Exception:
            if created_here:
                sess.rollback()
            raise
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # TREE BUILDING (DATA ONLY)
    # ---------------------------------------------------------

    @with_request_id
    def find_related(self, problem_id: int,
                     session: Optional[Session] = None) -> Optional[Dict[str, Any]]:

        sess, created_here = self._get_session(session)

        try:
            problem = Problem.get_by_id(sess, problem_id)
            if not problem:
                return None

            # Data assembly only — no branching logic
            solutions = [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "tasks": [
                        {
                            "id": ts.task.id,
                            "name": ts.task.name,
                            "description": ts.task.description
                        }
                        for ts in s.task_solutions
                    ]
                }
                for s in problem.solutions
            ]

            return {
                "problem": {
                    "id": problem.id,
                    "name": problem.name,
                    "description": problem.description,
                },
                "downward": {
                    "solutions": solutions
                }
            }

        finally:
            if created_here:
                sess.close()
