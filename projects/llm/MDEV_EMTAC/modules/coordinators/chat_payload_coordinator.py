from __future__ import annotations

from typing import Dict, Any, Optional

from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
)

from modules.orchestrators.chat_payload_orchestrator import ChatPayloadOrchestrator


class ChatPayloadCoordinator:
    """
    Application-layer coordinator for supporting chat payload requests.

    Responsibilities:
        - Validate payload request
        - Delegate to ChatPayloadOrchestrator.load_payload()
        - Normalize supporting payload response

    Does NOT:
        - Generate the answer
        - Open sessions
        - Commit/rollback
        - Access ORM directly
        - Render frontend HTML
    """

    EMPTY_BLOCKS = {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }

    TOP_LEVEL_TO_BLOCK_KEY = {
        "documents": "documents-container",
        "parts": "parts-container",
        "images": "images-container",
        "drawings": "drawings-container",
    }

    def __init__(
        self,
        *,
        orchestrator: Optional[ChatPayloadOrchestrator] = None,
    ):
        self.orchestrator = orchestrator or ChatPayloadOrchestrator()

    @with_request_id
    def load_payload(
        self,
        *,
        request_id: Optional[str],
        payload_seed: Optional[Dict[str, Any]] = None,
        client_type: str = "web",
    ) -> Dict[str, Any]:

        info_id("ChatPayloadCoordinator.load_payload called", request_id)

        normalized_client_type = (client_type or "web").strip().lower() or "web"

        if not request_id and not payload_seed:
            warning_id(
                "ChatPayloadCoordinator rejected request: missing request_id and payload_seed",
                request_id,
            )

            return self._normalize_response({
                "status": "invalid_input",
                "payload_status": "unavailable",
                "message": "Missing request_id or payload seed.",
            })

        result = self.orchestrator.load_payload(
            request_id=request_id,
            payload_seed=payload_seed,
            client_type=normalized_client_type,
        )

        return self._normalize_response(result)

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(response, dict):
            response = {
                "status": "error",
                "payload_status": "error",
                "message": "Invalid payload response format.",
            }

        response.setdefault("status", "success")
        response.setdefault("payload_status", "complete")
        response.setdefault("message", "")

        if "blocks" not in response or not isinstance(response["blocks"], dict):
            response["blocks"] = {}

        for block_key in self.EMPTY_BLOCKS:
            response["blocks"].setdefault(block_key, [])

            if response["blocks"][block_key] is None:
                response["blocks"][block_key] = []

        for top_level_key in self.TOP_LEVEL_TO_BLOCK_KEY:
            response.setdefault(top_level_key, [])

            if response[top_level_key] is None:
                response[top_level_key] = []

        for top_level_key, block_key in self.TOP_LEVEL_TO_BLOCK_KEY.items():
            top_level_items = response.get(top_level_key) or []

            if top_level_items and not response["blocks"].get(block_key):
                response["blocks"][block_key] = top_level_items

            if response["blocks"].get(block_key) and not response.get(top_level_key):
                response[top_level_key] = response["blocks"][block_key]

        return response