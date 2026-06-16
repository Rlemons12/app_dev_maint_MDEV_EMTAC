from __future__ import annotations

from typing import Any


class SearchAuditPayloadExtractor:
    """
    Extracts audit-friendly data from a search or payload response.

    The goal is to normalize different response shapes into predictable groups:

        - chunks
        - documents
        - images
        - drawings
        - parts

    This keeps the audit service from needing to know every possible
    frontend/backend payload structure.
    """

    @staticmethod
    def _as_list(value: Any) -> list[dict[str, Any]]:
        """
        Convert a possible payload section into a list of dictionaries.

        Supported shapes:
            list[dict]
            {"items": list[dict]}
            {"results": list[dict]}
            {"documents": list[dict]}
            {"images": list[dict]}
            {"drawings": list[dict]}
            {"parts": list[dict]}
            {"chunks": list[dict]}
        """

        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

        if isinstance(value, dict):
            for key in (
                "items",
                "results",
                "documents",
                "images",
                "drawings",
                "parts",
                "chunks",
            ):
                nested = value.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]

        return []

    @classmethod
    def extract_chunks(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
        response = response or {}

        return cls._as_list(
            response.get("chunks")
            or response.get("used_chunks")
            or response.get("retrieved_chunks")
        )

    @classmethod
    def extract_documents(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
        response = response or {}

        return cls._as_list(
            response.get("documents")
            or response.get("document_panel")
            or response.get("docs")
        )

    @classmethod
    def extract_images(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
        response = response or {}

        return cls._as_list(
            response.get("images")
            or response.get("image_panel")
            or response.get("thumbnails")
        )

    @classmethod
    def extract_drawings(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
        response = response or {}

        return cls._as_list(
            response.get("drawings")
            or response.get("drawing_panel")
        )

    @classmethod
    def extract_parts(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
        response = response or {}

        return cls._as_list(
            response.get("parts")
            or response.get("part_panel")
        )

    @classmethod
    def extract_all(cls, response: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        response = response or {}

        return {
            "chunks": cls.extract_chunks(response),
            "documents": cls.extract_documents(response),
            "images": cls.extract_images(response),
            "drawings": cls.extract_drawings(response),
            "parts": cls.extract_parts(response),
        }
