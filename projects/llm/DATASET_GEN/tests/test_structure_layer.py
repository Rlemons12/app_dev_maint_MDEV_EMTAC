import sys
import os

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_gen.document_structure.doc_structure_mapping import DocumentStructureMap, ImagePosition, ChunkBoundary
from data_gen.document_structure.structure_analyzer import StructureAnalyzer


def test_structure_map_basic():
    print("\n--- Smoke Test: DocumentStructureMap ---")

    structure = DocumentStructureMap(
        total_pages=3,
        metadata={"test": True}
    )

    # Add some fake page layout info
    structure.add_page_layout(0, {"rotation": 0, "dummy": True})
    structure.add_page_layout(1, {"rotation": 90})
    structure.add_page_layout(2, {"rotation": 180})

    # Add fake images
    pos = ImagePosition(
        page_number=1,
        bbox=(10, 20, 200, 240),
        image_index=0,
        estimated_size=(190, 220),
        content_type="image/png",
    )
    structure.add_image_position(pos)

    # Add fake chunk boundary
    boundary = ChunkBoundary(
        page_number=1,
        start_position=0,
        end_position=500,
        chunk_type="body_text",
        associated_images=[0],
        context_data={"preview": "fake text"}
    )
    structure.add_chunk_boundary(boundary)

    # Create extraction plan
    plan = structure.create_extraction_plan()

    print("Extraction plan created:")
    print(plan)

    assert structure.total_pages == 3
    assert len(structure.image_positions) == 1
    assert len(structure.chunk_boundaries) == 1
    assert "chunk_extraction_map" in plan
    assert "image_extraction_map" in plan

    print("✓ DocumentStructureMap smoke test passed!")


def test_structure_analyzer_basic():
    print("\n--- Smoke Test: StructureAnalyzer ---")

    analyzer = StructureAnalyzer()

    # Instead of analyzing a real PDF, test a minimal stub
    fake_raw_content = {
        "pages": [
            {"page_index": 0, "text": "Hello page 0", "images": [], "ocr_used": False},
            {"page_index": 1, "text": "Page 1 with image", "images": ["img_path"], "ocr_used": False},
        ],
        "metadata": {"source": "TEST_FAKE"}
    }

    structure_map = analyzer.analyze_raw_document(fake_raw_content)

    print("Structure map produced:")
    print("Pages:", structure_map.total_pages)
    print("Image positions:", len(structure_map.image_positions))
    print("Chunks:", len(structure_map.chunk_boundaries))

    assert structure_map.total_pages == 2
    assert len(structure_map.image_positions) >= 1
    assert len(structure_map.chunk_boundaries) >= 1

    print("✓ StructureAnalyzer smoke test passed!")


if __name__ == "__main__":
    test_structure_map_basic()
    test_structure_analyzer_basic()
