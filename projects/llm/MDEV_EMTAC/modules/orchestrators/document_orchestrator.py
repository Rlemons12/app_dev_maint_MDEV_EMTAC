from typing import Optional, List, Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.document_service import DocumentService
from modules.services.document_embedding_service import DocumentEmbeddingService


class DocumentOrchestrator(BaseOrchestrator):
    """
    Document domain workflow owner.

    Owns:
    - Document lifecycle
    - Document ↔ Embedding coordination
    - Document search

    Does NOT:
    - Generate embeddings
    - Perform RAG
    - Perform ingestion
    """

    def __init__(self):
        super().__init__()
        self.document_service = DocumentService()
        self.embedding_service = DocumentEmbeddingService()

    # ==========================================================
    # CREATE DOCUMENT
    # ==========================================================

    def create_document(
        self,
        *,
        name: str,
        file_path: str,
        content: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        rev: str = "R0",
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:

        with self._timed("DocumentOrchestrator.create_document"):
            with self.transaction() as session:

                doc = self.document_service.save(
                    session=session,
                    name=name,
                    file_path=file_path,
                    content=content,
                    complete_document_id=complete_document_id,
                    rev=rev,
                    doc_metadata=doc_metadata,
                    request_id=self._rid(),
                )

                self._info(f"Document created id={doc.id}")
                return doc.id

    # ==========================================================
    # UPDATE DOCUMENT
    # ==========================================================

    def update_document(
        self,
        *,
        doc_id: int,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        rev: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:

        with self._timed("DocumentOrchestrator.update_document"):
            with self.transaction() as session:

                doc = self.document_service.get(
                    session=session,
                    doc_id=doc_id,
                    request_id=self._rid(),
                )

                if not doc:
                    raise ValueError(f"Document id={doc_id} not found")

                if name is not None:
                    doc.name = name
                if file_path is not None:
                    doc.file_path = file_path
                if content is not None:
                    doc.content = content
                if rev is not None:
                    doc.rev = rev
                if doc_metadata is not None:
                    doc.doc_metadata = doc_metadata

                self._info(f"Document updated id={doc_id}")
                return True

    # ==========================================================
    # DELETE DOCUMENT
    # ==========================================================

    def delete_document(
        self,
        *,
        doc_id: int,
        delete_embeddings: bool = True,
    ) -> bool:

        with self._timed("DocumentOrchestrator.delete_document"):
            with self.transaction() as session:

                if delete_embeddings:
                    embeddings = self.embedding_service.get_by_document(
                        session=session,
                        document_id=doc_id,
                        request_id=self._rid(),
                    )

                    for emb in embeddings:
                        session.delete(emb)

                deleted = self.document_service.remove(
                    session=session,
                    doc_id=doc_id,
                    request_id=self._rid(),
                )

                self._info(f"Document deleted id={doc_id}")
                return deleted

    # ==========================================================
    # GET DOCUMENT WITH EMBEDDINGS
    # ==========================================================

    def get_document_with_embeddings(
        self,
        *,
        doc_id: int,
    ) -> Dict[str, Any]:

        with self._timed("DocumentOrchestrator.get_document_with_embeddings"):
            with self.transaction(read_only=True) as session:

                doc = self.document_service.get(
                    session=session,
                    doc_id=doc_id,
                    request_id=self._rid(),
                )

                if not doc:
                    return {}

                embeddings = self.embedding_service.get_by_document(
                    session=session,
                    document_id=doc_id,
                    request_id=self._rid(),
                )

                return {
                    "document": doc,
                    "embeddings": embeddings,
                }

    # ==========================================================
    # SEARCH (FTS)
    # ==========================================================

    def search_documents(
        self,
        *,
        search_text: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:

        with self._timed("DocumentOrchestrator.search_documents"):
            with self.transaction(read_only=True) as session:

                return self.document_service.search_fts(
                    session=session,
                    search_text=search_text,
                    limit=limit,
                    request_id=self._rid(),
                )

    # ==========================================================
    # VECTOR SEARCH
    # ==========================================================

    def search_similar_documents(
        self,
        *,
        query_embedding: List[float],
        model_name: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:

        with self._timed("DocumentOrchestrator.search_similar_documents"):
            with self.transaction(read_only=True) as session:

                return self.embedding_service.search_similar(
                    session=session,
                    query_embedding=query_embedding,
                    model_name=model_name,
                    limit=limit,
                    threshold=threshold,
                    request_id=self._rid(),
                )
