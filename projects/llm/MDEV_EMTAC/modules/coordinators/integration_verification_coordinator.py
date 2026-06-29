from __future__ import annotations

from typing import Dict, Any, Optional

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    error_id,
)

from modules.orchestrators.integration_verification_orchestrator import (
    IntegrationVerificationOrchestrator,
)

from modules.emtac_ai.search.rag_core.rag_pipeline import (
    get_default_rag,
)


class IntegrationVerificationCoordinator:
    """
    Application-layer integration test harness.

    Responsibilities:
        - Drive full system verification
        - Execute deterministic payload verification
        - Execute RAG pipeline
        - Compare outputs
        - Produce structured integration report

    Does NOT:
        - Own DB session
        - Execute domain logic directly
        - Perform AI logic itself
    """

    def __init__(self):
        self.verification_orchestrator = IntegrationVerificationOrchestrator()
        self.rag_pipeline = get_default_rag()

    # ---------------------------------------------------------
    # MAIN INTEGRATION TEST ENTRY
    # ---------------------------------------------------------
    @with_request_id
    def run_document_integration_test(
        self,
        *,
        document_id: int,
        test_question: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        debug_id(
            f"[IntegrationCoordinator] Starting integration test for document_id={document_id}",
            request_id,
        )

        try:
            # -------------------------------------------------
            # 1. Deterministic Graph Verification
            # -------------------------------------------------
            deterministic_result = (
                self.verification_orchestrator.verify_complete_document(
                    document_id=document_id,
                    request_id=request_id,
                )
            )

            # -------------------------------------------------
            # 2. Optional RAG Execution
            # -------------------------------------------------
            rag_result = None

            if test_question:
                rag_result = self.rag_pipeline.run(
                    question=test_question,
                    top_k=5,
                    request_id=request_id,
                )

            # -------------------------------------------------
            # 3. Comparison Layer
            # -------------------------------------------------
            comparison = self._compare_results(
                deterministic_result,
                rag_result,
            )

            info_id(
                f"[IntegrationCoordinator] Completed integration test for document_id={document_id}",
                request_id,
            )

            return {
                "deterministic": deterministic_result,
                "rag": rag_result,
                "comparison": comparison,
            }

        except Exception as e:
            error_id(
                f"[IntegrationCoordinator] Integration test failed: {e}",
                request_id,
                exc_info=True,
            )
            raise

    # ---------------------------------------------------------
    # INTERNAL COMPARISON LOGIC
    # ---------------------------------------------------------
    def _compare_results(
        self,
        deterministic: Dict[str, Any],
        rag: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not rag:
            return {"note": "RAG not executed."}

        deterministic_docs = deterministic.get("payload", [])
        rag_docs = rag.get("documents", [])

        return {
            "deterministic_doc_count": len(deterministic_docs),
            "rag_doc_count": len(rag_docs),
            "doc_count_match": len(deterministic_docs) == len(rag_docs),
        }