# modules/services/ai_steward_manager_service.py

from __future__ import annotations

from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id,
    error_id,
    debug_id,
    warning_id,
)

from modules.services.unified_search_service import UnifiedSearchService
from modules.observability.high_end_tracer import tracer


class AIStewardManagerService:
    """
    Stateless AI execution service.

    Responsibilities:
        - Call UnifiedSearchService
        - Return raw/enriched domain result
        - No persistence
        - No transaction ownership
        - No session creation
        - No UI formatting

    Notes:
        - ChatOrchestrator owns the session/transaction.
        - ChatService owns final UI formatting into blocks.
        - This service is the bridge between UnifiedSearchService/RAG and
          richer document payload data.
    """

    def __init__(self):
        self.search_service = UnifiedSearchService(
            enable_intent=False,
            enable_vector=True,
            enable_fts=True,
        )

    # ---------------------------------------------------------
    # Main Execution
    # ---------------------------------------------------------

    @with_request_id
    def execute(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        client_type: Optional[str] = None,
        request_id: Optional[str] = None,
        rag_only: bool = True,
        forced_chunk_id: Optional[int] = None,
    ) -> Dict[str, Any]:

        question = (question or "").strip()

        if not question:
            return self._empty_response(
                strategy="invalid_input",
                answer="Please provide a more detailed question.",
                model_name=None,
            )

        try:
            # --------------------------------------------------
            # 1. Execute search / RAG
            # --------------------------------------------------
            with tracer.span(
                "unified_search_execute",
                meta={
                    "rag_only": rag_only,
                    "client_type": client_type,
                    "forced_chunk_id": forced_chunk_id,
                },
            ):
                result = self.search_service.execute(
                    session=session,
                    user_id=user_id,
                    question=question,
                    request_id=request_id,
                    rag_only=rag_only,
                    forced_chunk_id=forced_chunk_id,
                )

            if not isinstance(result, dict):
                warning_id(
                    "[AIStewardManagerService] UnifiedSearchService returned "
                    "non-dict result; replacing with empty result.",
                    request_id,
                )
                result = {}

            # --------------------------------------------------
            # 2. Normalize raw result shape
            # --------------------------------------------------
            result.setdefault("strategy", "rag")
            result.setdefault("method", result.get("strategy", "rag"))
            result.setdefault("answer", "")
            result.setdefault("chunks", [])
            result.setdefault("used_chunks", [])
            result.setdefault("documents", [])
            result.setdefault("images", [])
            result.setdefault("drawings", [])
            result.setdefault("parts", [])

            # --------------------------------------------------
            # 3. Inject model_name safely
            # --------------------------------------------------
            result["model_name"] = self._resolve_model_name(
                fallback=result.get("model_name"),
                request_id=request_id,
            )

            # --------------------------------------------------
            # 4. Enrich document/UI payload when relationship data exists
            # --------------------------------------------------
            with tracer.span("ai_result_enrichment"):
                result = self._apply_relationship_projection(
                    session=session,
                    result=result,
                    request_id=request_id,
                )

            debug_id(
                "[AIStewardManagerService] Result ready "
                f"(strategy={result.get('strategy')}, "
                f"docs={len(result.get('documents') or [])}, "
                f"images={len(result.get('images') or [])}, "
                f"parts={len(result.get('parts') or [])}, "
                f"drawings={len(result.get('drawings') or [])})",
                request_id,
            )

            return result

        except Exception as e:
            error_id(
                f"AIStewardManagerService failure: {e}",
                request_id,
                exc_info=True,
            )

            return self._empty_response(
                strategy="error",
                answer="AI processing error.",
                model_name=None,
            )

    # ---------------------------------------------------------
    # Relationship Projection Hook
    # ---------------------------------------------------------

    def _apply_relationship_projection(
            self,
            *,
            session: Session,
            result: Dict[str, Any],
            request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Converts RAG chunks + relationship_map into richer document payloads.

        Required input fields from UnifiedSearchService:
            - used_chunks or chunks
            - relationship_map

        Output fields expected by ChatService:
            - documents
            - images
            - parts
            - drawings

        Notes:
            - This method does not create relationship_map.
            - UnifiedSearchService is responsible for creating relationship_map.
            - This method only projects/enriches data if relationship_map exists.
        """

        relationship_map = result.get("relationship_map")

        debug_id(
            "[AIStewardManagerService] projection precheck "
            f"strategy={result.get('strategy')} "
            f"method={result.get('method')} "
            f"used_chunks={len(result.get('used_chunks') or [])} "
            f"chunks={len(result.get('chunks') or [])} "
            f"documents={len(result.get('documents') or [])} "
            f"images={len(result.get('images') or [])} "
            f"relationship_map={'yes' if isinstance(relationship_map, dict) and relationship_map else 'no'}",
            request_id,
        )

        if not isinstance(relationship_map, dict) or not relationship_map:
            debug_id(
                "[AIStewardManagerService] No relationship_map found; "
                "leaving document payload unchanged.",
                request_id,
            )
            return result

        chunks = self._resolve_projection_chunks(result)

        if not chunks:
            debug_id(
                "[AIStewardManagerService] No chunks available for relationship "
                "projection; leaving document payload unchanged.",
                request_id,
            )
            return result

        try:
            from modules.ai.search_pathway.rag_core.chunk_relationship_projection import (
                ChunkRelationshipProjection,
            )
        except Exception as e:
            warning_id(
                "[AIStewardManagerService] ChunkRelationshipProjection import failed; "
                f"leaving payload unchanged. Error: {e}",
                request_id,
            )
            return result

        try:
            projection = ChunkRelationshipProjection(session=session)

            projected = projection.project_chunks_for_ui(
                chunks=chunks,
                relationship_map=relationship_map,
            )

            if not isinstance(projected, dict):
                warning_id(
                    "[AIStewardManagerService] Relationship projection returned "
                    "non-dict payload; leaving payload unchanged.",
                    request_id,
                )
                return result

            documents = projected.get("documents-container")

            if isinstance(documents, list):
                result["documents"] = documents

            summary = projected.get("summary")
            if summary is not None:
                result["relationship_summary"] = summary

            # --------------------------------------------------
            # Copy top-level containers if projection provides them
            # --------------------------------------------------
            for result_key, projected_key in (
                    ("images", "images-container"),
                    ("parts", "parts-container"),
                    ("drawings", "drawings-container"),
            ):
                items = projected.get(projected_key)
                if isinstance(items, list):
                    result[result_key] = items

            # --------------------------------------------------
            # Safety net:
            # If projection only placed images/parts/drawings inside documents,
            # promote them to top-level result containers too.
            # --------------------------------------------------
            result["images"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("images", "part_images"),
                existing_items=result.get("images") or [],
            )

            result["parts"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("parts",),
                existing_items=result.get("parts") or [],
            )

            result["drawings"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("drawings",),
                existing_items=result.get("drawings") or [],
            )

            debug_id(
                "[AIStewardManagerService] Applied relationship projection "
                f"documents={len(result.get('documents') or [])} "
                f"images={len(result.get('images') or [])} "
                f"parts={len(result.get('parts') or [])} "
                f"drawings={len(result.get('drawings') or [])}",
                request_id,
            )

            return result

        except Exception as e:
            warning_id(
                "[AIStewardManagerService] Relationship projection failed; "
                f"leaving payload unchanged. Error: {e}",
                request_id,
            )
            return result

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _resolve_projection_chunks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finds the best chunk list for UI projection.

        Preference:
            1. used_chunks - cleaned/context-selected chunks from RAG
            2. chunks - raw chunk list if used_chunks is unavailable
        """

        used_chunks = result.get("used_chunks")
        if isinstance(used_chunks, list) and used_chunks:
            return used_chunks

        chunks = result.get("chunks")
        if isinstance(chunks, list) and chunks:
            return chunks

        return []

    @staticmethod
    def _promote_unique_items_from_documents(
            *,
            documents: List[Dict[str, Any]],
            field_names: tuple[str, ...],
            existing_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Promotes nested document-level items to top-level result containers.

        Example:
            doc["images"] -> result["images"]
            doc["part_images"] -> result["images"]

        Dedupes by:
            1. item["id"] if available
            2. item["src"] if available
            3. object identity fallback
        """

        promoted: List[Dict[str, Any]] = []
        seen = set()

        def add_item(item: Any) -> None:
            if not isinstance(item, dict):
                return

            item_id = item.get("id")
            item_src = item.get("src")

            if item_id is not None:
                key = ("id", item_id)
            elif item_src:
                key = ("src", item_src)
            else:
                key = ("obj", id(item))

            if key in seen:
                return

            seen.add(key)
            promoted.append(item)

        for item in existing_items or []:
            add_item(item)

        for doc in documents or []:
            if not isinstance(doc, dict):
                continue

            for field_name in field_names:
                nested = doc.get(field_name)

                if isinstance(nested, list):
                    for item in nested:
                        add_item(item)

        return promoted

    def _resolve_model_name(
        self,
        *,
        fallback: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Safely resolves the active model name from the search/RAG stack.

        This must never break the chat flow.
        """

        if fallback:
            return fallback

        try:
            rag_pipeline = getattr(self.search_service, "rag_pipeline", None)

            if not rag_pipeline:
                return None

            answer_generator = getattr(rag_pipeline, "answer_generator", None)

            if not answer_generator:
                return None

            model_name = getattr(answer_generator, "model_name", None)
            if model_name:
                return model_name

            model_config = getattr(answer_generator, "model_config", None)
            if model_config:
                return getattr(model_config, "model_name", None)

            return None

        except Exception as e:
            debug_id(
                f"[AIStewardManagerService] Could not resolve model name: {e}",
                request_id,
            )
            return None

    @staticmethod
    def _empty_response(
        *,
        strategy: str,
        answer: str,
        model_name: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "status": "error" if strategy == "error" else "success",
            "strategy": strategy,
            "method": strategy,
            "answer": answer,
            "chunks": [],
            "used_chunks": [],
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
            "model_name": model_name,
        }

    @with_request_id
    def project_payload(
        self,
        *,
        session: Session,
        result: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds supporting UI payload from an existing answer/RAG result.

        This is intended for the second request:
            ChatPayloadOrchestrator.load_payload()

        It expects the result to already contain:
            - chunks or used_chunks
            - relationship_map

        It returns:
            - documents
            - images
            - parts
            - drawings
            - relationship_summary when available
        """

        if not isinstance(result, dict):
            warning_id(
                "[AIStewardManagerService] project_payload received non-dict result.",
                request_id,
            )
            result = {}

        result.setdefault("strategy", "rag")
        result.setdefault("method", result.get("strategy", "rag"))
        result.setdefault("answer", "")
        result.setdefault("chunks", [])
        result.setdefault("used_chunks", [])
        result.setdefault("documents", [])
        result.setdefault("images", [])
        result.setdefault("drawings", [])
        result.setdefault("parts", [])

        with tracer.span("ai_payload_projection"):
            result = self._apply_relationship_projection(
                session=session,
                result=result,
                request_id=request_id,
            )

        result["payload_status"] = "complete"

        debug_id(
            "[AIStewardManagerService] Payload projection ready "
            f"documents={len(result.get('documents') or [])} "
            f"images={len(result.get('images') or [])} "
            f"parts={len(result.get('parts') or [])} "
            f"drawings={len(result.get('drawings') or [])}",
            request_id,
        )

        return result