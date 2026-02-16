#MDEV_EMTAC/modules/services/db_services.py

from modules.configuration.config_env import DatabaseConfig

from modules.services.area_service import AreaService
from modules.services.position_service import PositionService
from modules.services.part_service import PartService
from modules.services.tool_service import ToolService
from modules.services.document_service import DocumentService
from modules.services.complete_document_service import CompleteDocumentService
from modules.services.image_service import ImageService
from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.task_position_association_service import TaskPositionAssociationService
from modules.services.tool_position_association_service import ToolPositionAssociationService
from modules.services.chunk_search_service import ChunkAssociationSearchService
from modules.services.drawing_part_association_service import DrawingPartAssociationService
from modules.services.document_ui_projection_service import DocumentUIProjectionService


class _ChunkSearchProxy:
    """
    Allows BOTH:
      services.chunk_search.search_from_chunk(...)
      services.chunk_search(session=...).search_from_chunk(...)
    """

    def __init__(self, db_config):
        self._db_config = db_config

    def __call__(self, *, session, request_id=None):
        return ChunkAssociationSearchService(
            session=session,
            request_id=request_id,
        )

    def __getattr__(self, name):
        raise AttributeError(
            "Chunk search requires a session. "
            "Use services.chunk_search(session=...)."
        )


class DBServices:
    """
    SAFE service façade (NO AI / RAG execution).

    ✔ Owns DatabaseConfig
    ✔ Session-safe
    ✔ Backward compatible
    """

    def __init__(self):
        self.db_config = DatabaseConfig()

        # -------------------------
        # CORE DATA SERVICES
        # -------------------------
        self.areas = AreaService(self.db_config)
        self.positions = PositionService(self.db_config)
        self.parts = PartService(self.db_config)
        self.tools = ToolService(self.db_config)

        self.documents = DocumentService(self.db_config)
        self.complete_documents = CompleteDocumentService(self.db_config)
        self.images = ImageService(self.db_config)

        # -------------------------
        # ASSOCIATIONS
        # -------------------------
        self.parts_position_images = PartsPositionImageService(self.db_config)
        self.task_positions = TaskPositionAssociationService(self.db_config)
        self.tool_positions = ToolPositionAssociationService(self.db_config)
        self.drawing_part_associations = DrawingPartAssociationService(self.db_config)
        # -------------------------
        # SEARCH (BACKWARD SAFE)
        # -------------------------
        self.chunk_search = _ChunkSearchProxy(self.db_config)

        # -------------------------
        # UI PROJECTION
        # -------------------------
        self.ui_projection = lambda session: DocumentUIProjectionService(
            session=session
        )
