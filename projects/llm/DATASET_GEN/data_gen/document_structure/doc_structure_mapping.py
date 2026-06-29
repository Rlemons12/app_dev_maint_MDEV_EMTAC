#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Structure Mapping (Core Models)
----------------------------------------

This file contains lightweight dataclasses used by all layers of the
structure-aware pipeline:
    - Layer 1: Structure Analyzer
    - Layer 2: Guided Extraction
    - Layer 3: Structure-Aware Q&A
    - Layer 4: ORPO Correction Feedback
    - Layer 5: Long-term continuous learning

This version contains NO database dependencies.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ================================================================
# IMAGE POSITION
# ================================================================

@dataclass
class ImagePosition:
    """
    Represents an image's position on a page.

    page_number: PDF page index (0-based)
    bbox: (x0, y0, x1, y1) rectangle for the image
    image_index: unique sequence number for page
    estimated_size: width & height
    content_type: "image/raster", "image/svg+xml", etc.
    """
    page_number: int
    bbox: Tuple[float, float, float, float]
    image_index: int
    estimated_size: Tuple[int, int]
    content_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "bbox": self.bbox,
            "image_index": self.image_index,
            "estimated_size": self.estimated_size,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImagePosition":
        return cls(
            page_number=data["page_number"],
            bbox=tuple(data["bbox"]),
            image_index=data["image_index"],
            estimated_size=tuple(data["estimated_size"]),
            content_type=data["content_type"],
        )


# ================================================================
# CHUNK BOUNDARY
# ================================================================

@dataclass
class ChunkBoundary:
    """
    Defines a chunk of content on a page.

    start_position: top Y coordinate
    end_position: bottom Y coordinate
    chunk_type: paragraph, heading, caption, etc.
    associated_images: list of image indexes
    context_data: additional metadata (layout type, preview text, etc.)
    """
    page_number: int
    start_position: float
    end_position: float
    chunk_type: str
    associated_images: List[int]
    context_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "chunk_type": self.chunk_type,
            "associated_images": self.associated_images,
            "context_data": self.context_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkBoundary":
        return cls(
            page_number=data["page_number"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            chunk_type=data["chunk_type"],
            associated_images=data["associated_images"],
            context_data=data["context_data"],
        )


# ================================================================
# DOCUMENT STRUCTURE MAP
# ================================================================

@dataclass
class DocumentStructureMap:
    """
    Complete structure representation of a document.

    Backward-compatible with the original EMTAC version:
        - Allows incremental building (add_page_layout, add_image_position, add_chunk_boundary)
        - Supports regenerating extraction plans
        - Fully serializable
    """

    total_pages: int
    image_positions: List[ImagePosition] = None
    chunk_boundaries: List[ChunkBoundary] = None
    page_layouts: Dict[int, Dict[str, Any]] = None
    extraction_plan: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    # ------------------------------------------------------------
    # INIT + DEFAULTS
    # ------------------------------------------------------------
    def __post_init__(self):
        if self.image_positions is None:
            self.image_positions = []
        if self.chunk_boundaries is None:
            self.chunk_boundaries = []
        if self.page_layouts is None:
            self.page_layouts = {}
        if self.extraction_plan is None:
            self.extraction_plan = {}
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------
    # UPDATE METHODS (required by tests)
    # ------------------------------------------------------------
    def add_page_layout(self, page_number: int, layout: Dict[str, Any]):
        """Add or update a page's layout metadata."""
        self.page_layouts[page_number] = layout

    def add_image_position(self, img: ImagePosition):
        """Append an image position entry."""
        self.image_positions.append(img)

    def add_chunk_boundary(self, chunk: ChunkBoundary):
        """Append a chunk boundary entry."""
        self.chunk_boundaries.append(chunk)

    # ------------------------------------------------------------
    # EXTRACTION PLAN (needed by tests)
    # ------------------------------------------------------------
    def create_extraction_plan(self) -> Dict[str, Any]:
        """
        Rebuild the basic extraction plan.
        Backward-compatible with EMTAC pipeline (chunk_extraction_map & image_extraction_map included).
        """
        plan = {
            "strategy": "structure_guided",
            "pages": list(range(self.total_pages)),
            "chunks": {},
            "images": {},
            "chunk_extraction_map": {},
            "image_extraction_map": {},  # <-- NEW
        }

        # Process Chunks
        for idx, c in enumerate(self.chunk_boundaries):
            cid = f"chunk_{c.page_number}_{idx}"

            plan["chunks"][cid] = {
                "page_number": c.page_number,
                "start_y": c.start_position,
                "end_y": c.end_position,
                "type": c.chunk_type,
                "images": c.associated_images,
            }

            # Chunk extraction actions (EMTAC legacy)
            actions = ["extract_text"]
            if c.associated_images:
                actions.append("extract_images")
            plan["chunk_extraction_map"][cid] = actions

        # Process Images
        for img in self.image_positions:
            iid = f"image_{img.page_number}_{img.image_index}"

            plan["images"][iid] = {
                "page_number": img.page_number,
                "index": img.image_index,
                "bbox": img.bbox,
                "type": img.content_type,
            }

            # Image extraction actions (EMTAC legacy)
            img_actions = ["extract_image"]
            if img.content_type == "image/svg+xml":
                img_actions.append("extract_vector")
            plan["image_extraction_map"][iid] = img_actions

        self.extraction_plan = plan
        return plan

    # ------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "image_positions": [img.to_dict() for img in self.image_positions],
            "chunk_boundaries": [c.to_dict() for c in self.chunk_boundaries],
            "page_layouts": self.page_layouts,
            "extraction_plan": self.extraction_plan,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStructureMap":
        return cls(
            total_pages=data["total_pages"],
            image_positions=[ImagePosition.from_dict(d) for d in data["image_positions"]],
            chunk_boundaries=[ChunkBoundary.from_dict(d) for d in data["chunk_boundaries"]],
            page_layouts=data.get("page_layouts", {}),
            extraction_plan=data.get("extraction_plan", {}),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------
    # FILE HELPERS
    # ------------------------------------------------------------
    def save_to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "DocumentStructureMap":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
