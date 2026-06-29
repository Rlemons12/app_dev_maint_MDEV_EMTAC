#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LAYER 1 — STRUCTURE ANALYZER
============================

This module performs high-level document structure analysis:

    - Page layout detection
    - Text block extraction
    - Block-type classification (heading, caption, paragraph, list)
    - Image extraction (raster + vector)
    - Chunk boundary creation
    - Extraction plan creation
    - Returns a DocumentStructureMap

NO database integration in this layer.
"""

import re
import fitz  # PyMuPDF
from typing import Any, Dict, List

from .doc_structure_mapping import (
    DocumentStructureMap,
    ImagePosition,
    ChunkBoundary,
)


# ================================================================
# MAIN ANALYZER
# ================================================================

class DocumentStructureAnalyzer:
    """
    Standalone document structure analyzer.

    Input: PDF path
    Output: DocumentStructureMap
    """

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def analyze(self, file_path: str) -> DocumentStructureMap:
        """
        Perform full document structure analysis.
        """
        pdf = fitz.open(file_path)
        total_pages = len(pdf)

        structure = DocumentStructureMap(
            total_pages=total_pages,
            image_positions=[],
            chunk_boundaries=[],
            page_layouts={},
            extraction_plan={},
            metadata={
                "file_path": file_path,
                "analysis_version": "L1-StructureAnalyzer",
                "pg_engine": "pg_config_engin (placeholder)",
            }
        )

        # --------------------------------------------------------
        # PAGE LOOP
        # --------------------------------------------------------
        for page_number in range(total_pages):
            page = pdf[page_number]

            page_result = self._analyze_page(page, page_number)

            structure.page_layouts[page_number] = page_result["layout"]
            structure.image_positions.extend(page_result["images"])
            structure.chunk_boundaries.extend(page_result["chunks"])

        pdf.close()

        # Final extraction plan
        structure.extraction_plan = self._create_extraction_plan(structure)

        return structure

    # ============================================================
    # PAGE ANALYSIS
    # ============================================================

    def _analyze_page(self, page, page_number: int) -> Dict[str, Any]:
        """
        Analyze a single page.
        Returns image positions, text blocks, chunk boundaries, metadata.
        """

        analysis = {
            "layout": {
                "page_size": tuple(page.rect),
                "rotation": page.rotation,
                "text_blocks": [],
                "layout_type": "unknown",
            },
            "images": [],
            "chunks": [],
        }

        # --------------------------------------------------------
        # A) Images (PNG/JPG)
        # --------------------------------------------------------
        for image_index, img_info in enumerate(page.get_images(full=True)):
            rects = page.get_image_rects(img_info[0])
            rect = rects[0] if rects else fitz.Rect(0, 0, 50, 50)

            analysis["images"].append(
                ImagePosition(
                    page_number=page_number,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    image_index=image_index,
                    estimated_size=(int(rect.width), int(rect.height)),
                    content_type="image/raster"
                )
            )

        # --------------------------------------------------------
        # B) Vector drawings (SVG-like)
        # --------------------------------------------------------
        drawings = page.get_drawings()
        offset = len(analysis["images"])

        for i, dr in enumerate(drawings):
            if "rect" not in dr:
                continue

            r = dr["rect"]

            analysis["images"].append(
                ImagePosition(
                    page_number=page_number,
                    bbox=(r.x0, r.y0, r.x1, r.y1),
                    image_index=offset + i,
                    estimated_size=(int(r.width), int(r.height)),
                    content_type="image/svg+xml"
                )
            )

        # --------------------------------------------------------
        # C) Text extraction + classification
        # --------------------------------------------------------
        text_dict = page.get_text("dict")
        self._extract_text_blocks(text_dict, analysis)

        # Layout detection (single vs two column)
        analysis["layout"]["layout_type"] = self._detect_layout_type(
            analysis["layout"]["text_blocks"]
        )

        # --------------------------------------------------------
        # D) Create chunk boundaries
        # --------------------------------------------------------
        analysis["chunks"] = self._make_chunk_boundaries(
            analysis["layout"]["text_blocks"],
            analysis["images"],
            page_number
        )

        return analysis

    # ============================================================
    # TEXT BLOCK EXTRACTION
    # ============================================================

    def _extract_text_blocks(self, text_dict, analysis):
        """
        Extract text blocks and classify them.
        """
        blocks = analysis["layout"]["text_blocks"]

        for block in text_dict.get("blocks", []):
            if block["type"] != 0:
                continue

            bbox = block["bbox"]

            # Aggregate spans into a single string
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")

            text = text.strip()
            if not text:
                continue

            blocks.append({
                "bbox": bbox,
                "text": text,
                "block_type": self._classify_block(text),
            })

    # ------------------------------------------------------------
    # BLOCK TYPE CLASSIFIER
    # ------------------------------------------------------------

    def _classify_block(self, text: str) -> str:
        t = text.strip()

        # HEADINGS (uppercase or short + numeric prefix)
        if len(t) < 100:
            if t.isupper():
                return "heading"
            if re.match(r"^\d+[\.\)]", t):
                return "numbered_heading"

        # CAPTIONS
        if re.match(r"^(figure|table|image)\s+\d+", t.lower()):
            return "caption"

        # LIST ITEMS
        if re.match(r"^[\-\*\•]\s+", t):
            return "list_item"
        if re.match(r"^\d+\.", t):
            return "list_item"

        return "paragraph"

    # ============================================================
    # LAYOUT DETECTION
    # ============================================================

    def _detect_layout_type(self, blocks):
        if not blocks:
            return "unknown"

        left_positions = [round(b["bbox"][0], 0) for b in blocks]
        unique_lefts = list(set(left_positions))

        if len(unique_lefts) >= 2:
            return "two_column"

        return "single_column"

    # ============================================================
    # CHUNK CREATION
    # ============================================================

    def _make_chunk_boundaries(self, blocks, images, page_number):
        """
        Convert text blocks → chunk boundaries.
        """
        if not blocks:
            return []

        # Sort by y-position (top → bottom)
        blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][1])

        chunks = []
        chunk_start = None
        associated_imgs = []

        for idx, blk in enumerate(blocks_sorted):
            y0, y1 = blk["bbox"][1], blk["bbox"][3]
            block_type = blk["block_type"]
            text_preview = blk["text"][:80]

            if chunk_start is None:
                chunk_start = y0

            # Split triggers
            is_split = block_type in ("heading", "numbered_heading", "caption")
            is_last = idx == len(blocks_sorted) - 1

            if is_split or is_last:
                chunks.append(
                    ChunkBoundary(
                        page_number=page_number,
                        start_position=chunk_start,
                        end_position=y1,
                        chunk_type=block_type,
                        associated_images=associated_imgs.copy(),
                        context_data={"text_preview": text_preview},
                    )
                )
                chunk_start = None
                associated_imgs = []

        return chunks

    # ============================================================
    # EXTRACTION PLAN
    # ============================================================

    def _create_extraction_plan(self, struct: DocumentStructureMap):
        """
        Create a simple extraction map for layers 2+.
        """
        plan = {
            "strategy": "structure_guided",
            "pages": list(range(struct.total_pages)),
            "chunks": {},
            "images": {},
        }

        # Chunks
        for idx, c in enumerate(struct.chunk_boundaries):
            cid = f"chunk_{c.page_number}_{idx}"
            plan["chunks"][cid] = {
                "page_number": c.page_number,
                "start_y": c.start_position,
                "end_y": c.end_position,
                "type": c.chunk_type,
                "images": c.associated_images,
            }

        # Images
        for img in struct.image_positions:
            iid = f"image_{img.page_number}_{img.image_index}"
            plan["images"][iid] = {
                "page_number": img.page_number,
                "index": img.image_index,
                "bbox": img.bbox,
                "type": img.content_type,
            }

        return plan


# ================================================================
# MANUAL TEST (optional)
# ================================================================

# ================================================================
# TEST SUITE COMPATIBILITY WRAPPER
# ================================================================

class StructureAnalyzer(DocumentStructureAnalyzer):
    """
    Thin wrapper required by test_structure_layer.

    Tests import:
        from data_gen.document_structure_extractor.structure_analyzer import StructureAnalyzer

    This wrapper keeps a clean public API while preserving the
    fully featured internal class: DocumentStructureAnalyzer.
    """
    pass


if __name__ == "__main__":
    analyzer = DocumentStructureAnalyzer()
    doc_map = analyzer.analyze("test.pdf")
    doc_map.save_to_file("structure_output.json")
    print("[DONE] Structure analysis complete.")
