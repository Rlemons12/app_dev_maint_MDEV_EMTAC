from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import inspect
import os
import tempfile
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from PIL import Image as PILImage

from modules.configuration.config import DATABASE_DIR
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import Image, ImageCompletedDocumentAssociation
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)

from modules.services.image_extraction_service import ImageExtractionService
from modules.services.image_embedding_service import ImageEmbeddingService



class ImageGuidedAssociationService:
    """
    Guided image ingestion layer.

    Responsibilities:
      - Attempt guided extraction
      - Fallback to basic extraction
      - Generate image embeddings (ALL paths)
      - Handle synthetic VLM visual descriptions
      - Never own transactions
    """

    def __init__(self):
        self.basic = ImageExtractionService()
        self.embedding_service = ImageEmbeddingService()
        self.association_service = ImageCompletedDocumentAssociationService()
    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    @staticmethod
    def _resolve_absolute_image_path(file_path: str) -> str:
        """
        Convert DB-stored relative image path to absolute path when needed.
        """
        if not file_path:
            raise ValueError("image file_path is empty")

        if os.path.isabs(file_path):
            return file_path

        return os.path.join(DATABASE_DIR, file_path)

    @staticmethod
    def _build_document_key(
        *,
        complete_document_id: int,
        request_id: Optional[str],
    ) -> str:
        if request_id:
            return f"doc:{complete_document_id}:req:{request_id}"
        return f"doc:{complete_document_id}"

    def _load_new_images_for_document(
        self,
        *,
        session,
        complete_document_id: int,
        created: int,
    ) -> List[Image]:
        """
        Load only the newly created images for the document.

        We fetch newest first, then reverse to preserve creation order.
        """
        rows: List[Image] = (
            session.query(Image)
            .join(ImageCompletedDocumentAssociation)
            .filter(
                ImageCompletedDocumentAssociation.complete_document_id
                == complete_document_id
            )
            .order_by(Image.id.desc())
            .limit(created)
            .all()
        )

        rows.reverse()
        return rows

    def _prepare_batch_image_inputs(
        self,
        *,
        images: List[Image],
        request_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[PILImage.Image]]:
        """
        Convert DB image rows into batch-ready PIL images.

        Returns:
            prepared_rows: metadata aligned to pil_images
            pil_images: opened PIL images aligned to prepared_rows
        """
        prepared_rows: List[Dict[str, Any]] = []
        pil_images: List[PILImage.Image] = []

        for image in images:
            try:
                absolute_path = self._resolve_absolute_image_path(image.file_path)

                if not os.path.exists(absolute_path):
                    warning_id(
                        f"[GUIDED IMG] image file missing image_id={image.id} path={absolute_path}",
                        request_id,
                    )
                    continue

                pil_img = PILImage.open(absolute_path).convert("RGB")

                prepared_rows.append(
                    {
                        "image_id": image.id,
                        "image_obj": image,
                        "absolute_path": absolute_path,
                    }
                )
                pil_images.append(pil_img)

            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] failed loading image image_id={getattr(image, 'id', None)}: {e}",
                    request_id,
                )
                continue

        return prepared_rows, pil_images

    def _close_pil_images(self, pil_images: List[PILImage.Image]) -> None:
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass

    def _call_with_optional_request_id(
        self,
        fn,
        *args,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Call helper that may or may not accept request_id.
        """
        if not callable(fn):
            raise TypeError("fn must be callable")

        try:
            sig = inspect.signature(fn)
            if "request_id" in sig.parameters:
                return fn(*args, request_id=request_id, **kwargs)
            return fn(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)

    def _resolve_image_model_handler(
        self,
        *,
        embedding_model_service,
        model_name: Optional[str],
        request_id: Optional[str] = None,
    ):
        """
        Resolve the active image model handler robustly.

        Resolution order:
          1) embedding_model_service.get_image_model_handler(...)
          2) embedding_model_service.load_image_model(...)
          3) embedding_model_service._load_local_model(...)
          4) embedding_model_service.clip_handler
          5) direct import fallback for CLIPModelHandler
        """
        rid = request_id or get_request_id()

        if embedding_model_service is None:
            warning_id(
                "[GUIDED IMG] embedding_model_service is None; cannot resolve image model handler",
                rid,
            )
            return None

        debug_id(
            f"[GUIDED IMG] resolving image model handler | model_name={model_name}",
            rid,
        )

        # -------------------------------------------------
        # 1) get_image_model_handler
        # -------------------------------------------------
        get_handler = getattr(embedding_model_service, "get_image_model_handler", None)
        if callable(get_handler):
            try:
                try:
                    handler = self._call_with_optional_request_id(
                        get_handler,
                        model_name,
                        request_id=rid,
                    )
                except TypeError:
                    handler = self._call_with_optional_request_id(
                        get_handler,
                        request_id=rid,
                    )

                if handler is not None:
                    debug_id(
                        f"[GUIDED IMG] resolved handler via get_image_model_handler | type={type(handler).__name__}",
                        rid,
                    )
                    return handler
            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] get_image_model_handler failed: {e}",
                    rid,
                )

        # -------------------------------------------------
        # 2) load_image_model
        # -------------------------------------------------
        load_handler = getattr(embedding_model_service, "load_image_model", None)
        if callable(load_handler):
            try:
                try:
                    handler = self._call_with_optional_request_id(
                        load_handler,
                        model_name,
                        request_id=rid,
                    )
                except TypeError:
                    handler = self._call_with_optional_request_id(
                        load_handler,
                        request_id=rid,
                    )

                if handler is not None:
                    debug_id(
                        f"[GUIDED IMG] resolved handler via load_image_model | type={type(handler).__name__}",
                        rid,
                    )
                    return handler
            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] load_image_model failed: {e}",
                    rid,
                )

        # -------------------------------------------------
        # 3) _load_local_model
        # -------------------------------------------------
        load_local_model = getattr(embedding_model_service, "_load_local_model", None)
        if callable(load_local_model):
            try:
                if model_name:
                    try:
                        handler = self._call_with_optional_request_id(
                            load_local_model,
                            model_name,
                            request_id=rid,
                        )
                    except TypeError:
                        handler = self._call_with_optional_request_id(
                            load_local_model,
                            request_id=rid,
                        )
                else:
                    handler = self._call_with_optional_request_id(
                        load_local_model,
                        request_id=rid,
                    )

                if handler is not None:
                    debug_id(
                        f"[GUIDED IMG] resolved handler via _load_local_model | type={type(handler).__name__}",
                        rid,
                    )
                    return handler
            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] _load_local_model failed: {e}",
                    rid,
                )

        # -------------------------------------------------
        # 4) clip_handler attribute
        # -------------------------------------------------
        clip_handler = getattr(embedding_model_service, "clip_handler", None)
        if clip_handler is not None:
            debug_id(
                f"[GUIDED IMG] resolved handler via clip_handler attr | type={type(clip_handler).__name__}",
                rid,
            )
            return clip_handler

        # -------------------------------------------------
        # 5) direct import fallback
        # -------------------------------------------------
        if model_name in (None, "", "CLIPModelHandler"):
            try:
                from modules.ai.image.models.clip_model_handler import CLIPModelHandler

                handler = CLIPModelHandler()
                debug_id(
                    "[GUIDED IMG] resolved handler via direct CLIPModelHandler import fallback",
                    rid,
                )
                return handler
            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] direct CLIPModelHandler import fallback failed: {e}",
                    rid,
                )

        warning_id(
            f"[GUIDED IMG] unable to resolve image model handler for batch embedding | model_name={model_name}",
            rid,
        )
        return None

    def _batch_embed_extracted_images(
        self,
        *,
        session,
        complete_document_id: int,
        created: int,
        embedding_model_service,
        request_id: Optional[str] = None,
    ) -> int:
        """
        Batch embed extracted images for a single document.

        Preferred CLIP handler methods:
        - begin_document_embedding_session(document_key)
        - get_image_embeddings_batch(images, document_key=..., request_id=...)
        - end_document_embedding_session(document_key)
        """
        rid = request_id or get_request_id()

        if created <= 0:
            debug_id("[GUIDED IMG] no new images to batch embed", rid)
            return 0

        new_images = self._load_new_images_for_document(
            session=session,
            complete_document_id=complete_document_id,
            created=created,
        )

        if not new_images:
            debug_id("[GUIDED IMG] no new images found after extraction", rid)
            return 0

        prepared_rows, pil_images = self._prepare_batch_image_inputs(
            images=new_images,
            request_id=rid,
        )

        if not prepared_rows or not pil_images:
            warning_id("[GUIDED IMG] no valid images available for batch embedding", rid)
            return 0

        model_name = None
        try:
            model_name = embedding_model_service.get_current_model_name(
                request_id=rid
            )
        except TypeError:
            model_name = embedding_model_service.get_current_model_name()
        except Exception as e:
            warning_id(
                f"[GUIDED IMG] failed to resolve current image model name: {e}",
                rid,
            )

        clip_handler = self._resolve_image_model_handler(
            embedding_model_service=embedding_model_service,
            model_name=model_name,
            request_id=rid,
        )

        if clip_handler is None:
            self._close_pil_images(pil_images)
            return 0

        document_key = self._build_document_key(
            complete_document_id=complete_document_id,
            request_id=rid,
        )

        stored = 0

        try:
            begin_session = getattr(
                clip_handler,
                "begin_document_embedding_session",
                None,
            )
            if callable(begin_session):
                try:
                    begin_session(document_key)
                except TypeError:
                    begin_session(document_key=document_key)

            vectors = None
            get_batch = getattr(clip_handler, "get_image_embeddings_batch", None)

            if callable(get_batch):
                try:
                    vectors = get_batch(
                        pil_images,
                        document_key=document_key,
                        request_id=rid,
                    )
                except TypeError:
                    try:
                        vectors = get_batch(
                            pil_images,
                            document_key=document_key,
                        )
                    except TypeError:
                        vectors = get_batch(pil_images)

            if vectors is None:
                warning_id(
                    "[GUIDED IMG] batch image embedding API unavailable; falling back to per-image embedding",
                    rid,
                )
                vectors = []

                get_single = getattr(clip_handler, "get_image_embedding", None)
                if not callable(get_single):
                    warning_id(
                        "[GUIDED IMG] clip handler missing get_image_embedding",
                        rid,
                    )
                    return 0

                for pil_img in pil_images:
                    try:
                        try:
                            vec = get_single(
                                pil_img,
                                document_key=document_key,
                                request_id=rid,
                            )
                        except TypeError:
                            try:
                                vec = get_single(
                                    pil_img,
                                    document_key=document_key,
                                )
                            except TypeError:
                                vec = get_single(
                                    pil_img,
                                    request_id=rid,
                                )
                        vectors.append(vec)
                    except Exception as e:
                        warning_id(
                            f"[GUIDED IMG] per-image embedding failed: {e}",
                            rid,
                        )
                        vectors.append(None)

            if not isinstance(vectors, list):
                warning_id(
                    f"[GUIDED IMG] batch embedding returned invalid result type={type(vectors).__name__}",
                    rid,
                )
                return 0

            if len(vectors) != len(prepared_rows):
                warning_id(
                    f"[GUIDED IMG] vector count mismatch expected={len(prepared_rows)} got={len(vectors)}",
                    rid,
                )
                return 0

            for row, vector in zip(prepared_rows, vectors):
                image_id = row["image_id"]

                if not vector:
                    warning_id(
                        f"[GUIDED IMG] embedding failed image_id={image_id}: empty vector",
                        rid,
                    )
                    continue

                try:
                    self.embedding_service.add_pgvector(
                        session=session,
                        image_id=image_id,
                        model_name=model_name or "CLIPModelHandler",
                        embedding=vector,
                        request_id=rid,
                    )
                    stored += 1

                except Exception as e:
                    warning_id(
                        f"[GUIDED IMG] storing embedding failed image_id={image_id}: {e}",
                        rid,
                    )

            debug_id(
                f"[GUIDED IMG] batch image embeddings stored={stored} complete_document_id={complete_document_id}",
                rid,
            )
            return stored

        finally:
            try:
                end_session = getattr(
                    clip_handler,
                    "end_document_embedding_session",
                    None,
                )
                if callable(end_session):
                    try:
                        end_session(document_key)
                    except TypeError:
                        end_session(document_key=document_key)
            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] failed ending document embedding session: {e}",
                    rid,
                )

            self._close_pil_images(pil_images)

    # =========================================================
    # NEW — VLM STRUCTURED VISUAL INGESTION
    # =========================================================

    @with_request_id
    def create_from_description(
        self,
        *,
        session,
        complete_document_id: int,
        position_id: Optional[int],
        label: str,
        visual_type: str,
        description: str,
        embedding_model_service=None,
        request_id: Optional[str] = None,
    ) -> int:

        rid = request_id or get_request_id()

        try:
            temp_dir = tempfile.gettempdir()
            fake_path = os.path.join(
                temp_dir,
                f"vlm_visual_{complete_document_id}_{label}.txt",
            )

            with open(fake_path, "w", encoding="utf-8") as f:
                f.write(description)

            image = Image(
                title=label,
                description=f"{visual_type}: {description}",
                file_path=fake_path,
            )

            session.add(image)
            session.flush()

            assoc = ImageCompletedDocumentAssociation(
                image_id=image.id,
                complete_document_id=complete_document_id,
            )

            session.add(assoc)
            session.flush()

            debug_id(
                f"[VLM IMG] created synthetic image | id={image.id} | label={label}",
                rid,
            )

            if embedding_model_service:
                try:
                    model_name = embedding_model_service.get_current_model_name(
                        request_id=rid
                    )

                    vector = embedding_model_service.get_text_embedding(
                        description,
                        request_id=rid,
                    )

                    if vector:
                        self.embedding_service.add_pgvector(
                            session=session,
                            image_id=image.id,
                            model_name=model_name,
                            embedding=vector,
                            request_id=rid,
                        )

                except Exception as e:
                    warning_id(
                        f"[VLM IMG] embedding failed image_id={image.id}: {e}",
                        rid,
                    )

            return 1

        except Exception as e:
            error_id(f"[VLM IMG] failed create_from_description: {e}", rid)
            return 0

    # =========================================================
    # EXISTING PDF EXTRACTION FLOW (UPDATED FOR BATCH CLIP)
    # =========================================================

    @with_request_id
    def extract_and_associate(
        self,
        *,
        session,
        file_path: str,
        complete_document_id: int,
        position_id: Optional[int],
        embedding_model_service=None,
        request_id: Optional[str] = None,
    ) -> int:

        rid = request_id or get_request_id()
        created = 0

        def _count_images() -> int:
            return int(
                session.query(Image.id)
                .join(ImageCompletedDocumentAssociation)
                .filter(
                    ImageCompletedDocumentAssociation.complete_document_id
                    == complete_document_id
                )
                .count()
            )

        before_count = _count_images()

        try:
            info_id(
                f"[GUIDED IMG] starting guided extraction doc_id={complete_document_id}",
                rid,
            )

            payload: Dict[str, Any] = {
                "complete_document_id": complete_document_id,
                "position_id": position_id,
            }

            from modules.services.image_completed_document_association_service import (
                ImageCompletedDocumentAssociationService,
            )

            assoc_service = ImageCompletedDocumentAssociationService()

            success, result, status_code = self.association_service.guided_extraction_with_mapping(
                session=session,
                file_path=file_path,
                metadata=payload,
                request_id=rid,
            )

            if success:
                after_count = _count_images()
                created = max(0, after_count - before_count)

                debug_id(
                    f"[GUIDED IMG] guided extraction succeeded "
                    f"complete_document_id={complete_document_id} "
                    f"status_code={status_code} "
                    f"result={result} "
                    f"created={created}",
                    rid,
                )
            else:
                warning_id(
                    f"[GUIDED IMG] guided extraction failed "
                    f"complete_document_id={complete_document_id} "
                    f"status_code={status_code} "
                    f"result={result}",
                    rid,
                )

        except Exception as e:
            warning_id(
                f"[GUIDED IMG] guided extraction raised for complete_document_id={complete_document_id}: {e}",
                rid,
            )

        if created == 0:
            try:
                info_id(
                    f"[GUIDED IMG] falling back to basic image extraction "
                    f"complete_document_id={complete_document_id}",
                    rid,
                )

                basic_fn = getattr(self.basic, "extract_pdf_images", None)
                if callable(basic_fn):
                    basic_fn(
                        session=session,
                        file_path=file_path,
                        complete_document_id=complete_document_id,
                        position_id=position_id,
                        request_id=rid,
                    )

                    after_count = _count_images()
                    created = max(0, after_count - before_count)

                    debug_id(
                        f"[GUIDED IMG] fallback extraction complete "
                        f"complete_document_id={complete_document_id} "
                        f"created={created}",
                        rid,
                    )
                else:
                    warning_id(
                        "[GUIDED IMG] fallback extraction unavailable: "
                        "self.basic.extract_pdf_images not found",
                        rid,
                    )

            except Exception as e:
                error_id(
                    f"[GUIDED IMG] fallback extraction failed "
                    f"complete_document_id={complete_document_id}: {e}",
                    rid,
                    exc_info=True,
                )
                return 0

        if created > 0 and embedding_model_service:
            try:
                stored = self._batch_embed_extracted_images(
                    session=session,
                    complete_document_id=complete_document_id,
                    created=created,
                    embedding_model_service=embedding_model_service,
                    request_id=rid,
                )

                debug_id(
                    f"[GUIDED IMG] extracted={created} embedded={stored} "
                    f"complete_document_id={complete_document_id}",
                    rid,
                )

            except Exception as e:
                warning_id(
                    f"[GUIDED IMG] batch embedding stage failed "
                    f"for complete_document_id={complete_document_id}: {e}",
                    rid,
                )

        return created