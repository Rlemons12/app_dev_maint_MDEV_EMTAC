from typing import Dict, Any

from modules.configuration.log_config import debug_id


class ChatService:
    """
    Pure response formatting service.

    Responsibilities:
        - Convert raw AI result into stable UI contract
        - Guarantee block structure
        - Preserve legacy fields (method)
        - Preserve new architecture fields (strategy, model_name)
        - Stateless
    """

    # ---------------------------------------------------------
    # Response Formatting
    # ---------------------------------------------------------

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts raw AI domain result into UI contract.

        Guarantees:
            - Stable block structure
            - method field (legacy compatibility)
            - strategy field (new architecture)
            - model_name passthrough
        """

        if not isinstance(result, dict):
            return self._error_response("Invalid AI response format.")

        documents = result.get("documents") or []
        parts = result.get("parts") or []
        images = result.get("images") or []
        drawings = result.get("drawings") or []

        # Defensive type enforcement
        documents = documents if isinstance(documents, list) else []
        parts = parts if isinstance(parts, list) else []
        images = images if isinstance(images, list) else []
        drawings = drawings if isinstance(drawings, list) else []

        strategy = result.get("strategy", "rag")

        response = {
            "status": result.get("status", "success"),
            "answer": result.get("answer", ""),
            "method": strategy,               # Legacy parity
            "strategy": strategy,             # New architecture
            "model_name": result.get("model_name"),
            "blocks": {
                "documents-container": documents,
                "parts-container": parts,
                "images-container": images,
                "drawings-container": drawings,
            },
        }

        debug_id(
            f"ChatService formatted response "
            f"(docs={len(documents)}, parts={len(parts)}, "
            f"images={len(images)}, drawings={len(drawings)})",
            None,
        )

        return response

    # ---------------------------------------------------------
    # Error Helper
    # ---------------------------------------------------------

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "answer": message,
            "method": "error",
            "strategy": "error",
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
        }