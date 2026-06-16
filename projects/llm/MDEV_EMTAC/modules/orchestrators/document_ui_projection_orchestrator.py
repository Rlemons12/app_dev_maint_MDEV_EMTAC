from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id, debug_id, get_request_id

from modules.orchestrators.chunk_graph_orchestrator import ChunkGraphOrchestrator
from modules.services.drawing_navigation_projection import DrawingNavigationProjection


class DocumentUIProjectionOrchestrator(BaseOrchestrator):

    def __init__(self):
        super().__init__()
        self.chunk_graph_orchestrator = ChunkGraphOrchestrator()

    @with_request_id
    def build_from_chunk(
        self,
        *,
        chunk_id: int,
        include_embeddings: bool = False,
        include_reverse: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            search_result = self.chunk_graph_orchestrator.build_graph(
                chunk_id=chunk_id,
                include_embeddings=include_embeddings,
                include_2nd_tier=True,
            )

            if "error" in search_result:
                return search_result

            documents = [search_result.get("chunk", {})]

            tier1 = search_result.get("1st_tier", {})
            tier2 = search_result.get("2nd_tier", {})

            images = tier1.get("images", {})
            chunk_images = images.get("chunk_level", [])
            document_images = images.get("document_level", [])

            for doc in documents:

                doc["images"] = chunk_images + document_images

                positions = tier2.get("positions", [])
                doc["positions"] = positions
                doc["position_ids"] = positions

                if doc["position_ids"]:
                    drawing_nav = DrawingNavigationProjection(session=session)

                    doc["drawing_navigation"] = drawing_nav.build_navigation(
                        complete_document_id=doc.get("complete_document_id"),
                        position_ids=doc["position_ids"],
                    )

                doc["parts"] = tier2.get("parts", [])

                for key in ("tasks", "tools", "problems"):
                    if key in tier2:
                        doc[key] = tier2.get(key, [])

            return {
                "documents-container": documents,
                "summary": search_result.get("summary", {}),
            }