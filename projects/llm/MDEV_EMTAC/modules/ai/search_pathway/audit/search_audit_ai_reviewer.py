from __future__ import annotations

from typing import Any


class SearchAuditAIReviewer:
    """
    Future AI-based review layer.

    Purpose:
        Judge whether the returned search result actually matched the user's intent.

    This is different from technical validation.

    Technical validation asks:
        Did the returned IDs/files/relationships exist?

    AI review asks:
        Did the returned data make sense for the question?
    """

    @staticmethod
    def review_result_placeholder(
        *,
        question: str,
        answer: str | None,
        extracted_payload: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "review_status": "not_implemented",
            "question": question,
            "answer_present": bool(answer),
            "payload_counts": {
                key: len(value)
                for key, value in extracted_payload.items()
            },
        }
