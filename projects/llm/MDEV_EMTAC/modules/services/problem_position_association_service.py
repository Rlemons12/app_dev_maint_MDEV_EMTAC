# modules/services/problem_position_association_service.py

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ProblemPositionAssociation,
    Problem,
    Position,
)


class ProblemPositionAssociationService:
    """
    Service layer for Problem ↔ Position associations.

    Responsibilities:
    - Create / delete associations
    - Query problems by position
    - Query positions by problem
    - Hierarchy-aware filtering
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------
    # Session helper
    # ---------------------------------------------------------
    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # CREATE (Get-or-create)
    # ---------------------------------------------------------
    @with_request_id
    def associate(
        self,
        problem_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[ProblemPositionAssociation]:
        """
        Get-or-create a ProblemPositionAssociation.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        debug_id(
            f"[ProblemPositionAssociationService.associate] "
            f"problem_id={problem_id}, position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation(
                "ProblemPositionAssociationService.associate", rid
            ):
                if not problem_id or not position_id:
                    raise ValueError("Both problem_id and position_id are required")

                existing = (
                    sess.query(ProblemPositionAssociation)
                    .filter_by(problem_id=problem_id, position_id=position_id)
                    .first()
                )

                if existing:
                    debug_id(
                        f"Existing association found id={existing.id}", rid
                    )
                    return existing

                assoc = ProblemPositionAssociation(
                    problem_id=problem_id,
                    position_id=position_id,
                )
                sess.add(assoc)
                sess.commit()

                info_id(
                    f"Created ProblemPositionAssociation id={assoc.id}", rid
                )
                return assoc

        except SQLAlchemyError as e:
            sess.rollback()
            error_id(
                f"Failed to associate problem and position: {e}",
                rid,
                exc_info=True,
            )
            return None
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------
    @with_request_id
    def dissociate(
        self,
        problem_id: Optional[int] = None,
        position_id: Optional[int] = None,
        association_id: Optional[int] = None,
        session: Optional[SASession] = None,
    ) -> bool:
        """
        Remove a problem-position association.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "ProblemPositionAssociationService.dissociate", rid
            ):
                if association_id:
                    assoc = sess.query(ProblemPositionAssociation).get(association_id)
                elif problem_id and position_id:
                    assoc = (
                        sess.query(ProblemPositionAssociation)
                        .filter_by(problem_id=problem_id, position_id=position_id)
                        .first()
                    )
                else:
                    raise ValueError(
                        "Provide association_id OR both problem_id and position_id"
                    )

                if not assoc:
                    warning_id("Association not found", rid)
                    return False

                sess.delete(assoc)
                sess.commit()

                info_id(
                    f"Deleted ProblemPositionAssociation id={assoc.id}", rid
                )
                return True

        except Exception as e:
            sess.rollback()
            error_id(
                f"Failed to delete ProblemPositionAssociation: {e}",
                rid,
                exc_info=True,
            )
            return False
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # QUERIES
    # ---------------------------------------------------------
    @with_request_id
    def get_positions_for_problem(
        self,
        problem_id: int,
        session: Optional[SASession] = None,
    ) -> List[Position]:
        """
        Get all Positions linked to a Problem.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            positions = (
                sess.query(Position)
                .join(
                    ProblemPositionAssociation,
                    Position.id == ProblemPositionAssociation.position_id,
                )
                .filter(ProblemPositionAssociation.problem_id == problem_id)
                .all()
            )

            debug_id(
                f"Found {len(positions)} positions for problem_id={problem_id}",
                rid,
            )
            return positions

        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_problems_for_position(
        self,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> List[Problem]:
        """
        Get all Problems linked to a Position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            problems = (
                sess.query(Problem)
                .join(
                    ProblemPositionAssociation,
                    Problem.id == ProblemPositionAssociation.problem_id,
                )
                .filter(ProblemPositionAssociation.position_id == position_id)
                .all()
            )

            debug_id(
                f"Found {len(problems)} problems for position_id={position_id}",
                rid,
            )
            return problems

        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # HIERARCHY-AWARE QUERY
    # ---------------------------------------------------------
    @with_request_id
    def get_positions_for_problem_by_hierarchy(
        self,
        problem_id: int,
        level_filters: Optional[Dict[str, Any]] = None,
        session: Optional[SASession] = None,
    ) -> List[Position]:
        """
        Get positions associated with a problem filtered by hierarchy.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            query = (
                sess.query(Position)
                .join(
                    ProblemPositionAssociation,
                    Position.id == ProblemPositionAssociation.position_id,
                )
                .filter(ProblemPositionAssociation.problem_id == problem_id)
            )

            if level_filters:
                for field, value in level_filters.items():
                    if hasattr(Position, field) and value is not None:
                        query = query.filter(getattr(Position, field) == value)

            positions = query.all()

            info_id(
                f"Found {len(positions)} positions for problem_id={problem_id} "
                f"with hierarchy filters",
                rid,
            )
            return positions

        except SQLAlchemyError as e:
            error_id(
                f"Hierarchy query failed: {e}", rid, exc_info=True
            )
            return []
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # BATCH QUERY (REQUIRED BY CHUNK SEARCH)
    # ---------------------------------------------------------
    @with_request_id
    def get_problems_for_positions(
        self,
        position_ids: List[int],
        session: Optional[SASession] = None,
    ) -> List[Problem]:
        """
        Get all unique Problems linked to multiple Positions.

        This is the REQUIRED batch method used by ChunkAssociationSearch
        for tier-2 enrichment.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[ProblemPositionAssociationService.get_problems_for_positions] "
                "No position_ids provided",
                rid,
            )
            return []

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "ProblemPositionAssociationService.get_problems_for_positions", rid
            ):
                problems = (
                    sess.query(Problem)
                    .join(
                        ProblemPositionAssociation,
                        Problem.id == ProblemPositionAssociation.problem_id,
                    )
                    .filter(
                        ProblemPositionAssociation.position_id.in_(position_ids)
                    )
                    .distinct()
                    .all()
                )

                debug_id(
                    f"Found {len(problems)} problems for "
                    f"{len(position_ids)} positions",
                    rid,
                )

                return problems

        except SQLAlchemyError as e:
            error_id(
                f"Failed to fetch problems for positions: {e}",
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
    def count_problems(
            self,
            position_ids: list[int],
            session: Optional[SASession] = None,
    ) -> int:
        """
        Count unique Problems associated with given Positions.

        Lightweight helper for relationship summaries.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[ProblemPositionAssociationService.count_problems] No position_ids provided",
                rid,
            )
            return 0

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "ProblemPositionAssociationService.count_problems", rid
            ):
                count = (
                    sess.query(func.count(func.distinct(ProblemPositionAssociation.problem_id)))
                    .filter(
                        ProblemPositionAssociation.position_id.in_(position_ids)
                    )
                    .scalar()
                )

                debug_id(
                    f"Counted {count} unique problems for {len(position_ids)} positions",
                    rid,
                )

                return count or 0

        except Exception as e:
            error_id(
                f"Error counting problems for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return 0
        finally:
            if created_here:
                sess.close()
