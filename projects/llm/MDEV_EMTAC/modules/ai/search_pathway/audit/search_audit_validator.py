from __future__ import annotations

from typing import Any

from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_logger,
)


logger = get_search_audit_logger()


class SearchAuditValidator:
    """
    Validates search audit results.

    This class should eventually check things like:

        - returned chunks exist
        - returned documents exist
        - returned images exist
        - returned drawings exist
        - returned parts exist
        - files exist on disk
        - relationship paths are valid
        - duplicate payload items were not returned
    """

    @staticmethod
    def validate_basic_counts(
        extracted_payload: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        counts = {
            key: len(value)
            for key, value in extracted_payload.items()
        }

        result = {
            "check_name": "basic_payload_counts",
            "check_status": "passed",
            "counts": counts,
        }

        logger.debug(
            "AUDIT_VALIDATION check=basic_payload_counts status=passed counts=%s",
            counts,
        )

        return result
