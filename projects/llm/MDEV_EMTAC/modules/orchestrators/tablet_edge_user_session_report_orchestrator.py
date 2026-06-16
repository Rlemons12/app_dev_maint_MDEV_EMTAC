import logging

from modules.services.tablet_edge.tablet_user_session_report_dtos import (
    TabletUserSessionReportRequest,
    TabletUserSessionReportResult,
)
from modules.services.tablet_edge.tablet_user_session_report_service import (
    TabletUserSessionReportService,
)


logger = logging.getLogger(__name__)


class TabletEdgeUserSessionReportOrchestrator:
    """
    Owns the transaction for tablet-reported user session state.
    """

    def __init__(self, db_config):
        self.db_config = db_config

    def report_user_session(
        self,
        report: TabletUserSessionReportRequest,
    ) -> TabletUserSessionReportResult:
        db_session = self.db_config.get_main_session()

        try:
            result = TabletUserSessionReportService.report_user_session(
                db_session=db_session,
                report=report,
            )

            if result.success:
                db_session.commit()
            else:
                db_session.rollback()

            return result

        except Exception as exc:
            db_session.rollback()

            logger.exception(
                "Tablet user session report failed. tablet_uid=%s event_type=%s error=%s",
                report.tablet_uid,
                report.event_type,
                exc,
            )

            return TabletUserSessionReportResult(
                success=False,
                message=f"Tablet user session report failed: {exc}",
                tablet_uid=report.tablet_uid,
                tablet_name=report.tablet_name,
                username=report.username,
                display_name=report.display_name,
                event_type=report.event_type,
                is_active=False,
            )

        finally:
            db_session.close()