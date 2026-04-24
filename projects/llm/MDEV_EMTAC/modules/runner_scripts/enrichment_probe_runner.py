from __future__ import annotations

from modules.coordinators.enrichment_probe_coordinator import (
    EnrichmentProbeCoordinator,
)


def main() -> int:
    coordinator = EnrichmentProbeCoordinator()

    # --------------------------------------------------
    # OPTION 1: auto-find a trigger chunk
    # --------------------------------------------------
    # result = coordinator.run_probe()

    # --------------------------------------------------
    # OPTION 2: probe a specific chunk
    # --------------------------------------------------
    # result = coordinator.probe_chunk(chunk_id=726)

    # --------------------------------------------------
    # OPTION 3: scan tier summary across chunks
    # --------------------------------------------------
    result = coordinator.scan_tier_summary()
    # result = coordinator.scan_tier_summary(limit=200)

    print("\n=== ENRICHMENT PROBE RESULT ===")
    print(f"Status : {result.get('status')}")
    print(f"Message: {result.get('message')}")

    # --------------------------------------------------
    # TIER SUMMARY MODE
    # --------------------------------------------------
    if "tier_summary" in result:
        tier = result.get("tier_summary", {})

        print("\n--- TIER SUMMARY ---")
        print(f"Total Chunks Scanned           : {tier.get('total_chunks_scanned', 0)}")
        print(f"Chunks With Chunk-Level Images : {tier.get('chunks_with_chunk_level_images', 0)}")
        print(f"Chunks With Document Images    : {tier.get('chunks_with_document_level_images', 0)}")
        print(f"Chunks With Any Images         : {tier.get('chunks_with_any_images', 0)}")
        print(f"Chunks With Positions          : {tier.get('chunks_with_positions', 0)}")
        print(f"Chunks With Parts              : {tier.get('chunks_with_parts', 0)}")
        print(f"Chunks With Drawings           : {tier.get('chunks_with_drawings', 0)}")
        print(f"Chunks With Images + Parts     : {tier.get('chunks_with_images_and_parts', 0)}")
        print(f"Chunks With Images + Drawings  : {tier.get('chunks_with_images_and_drawings', 0)}")
        print(f"Chunks With Parts + Drawings   : {tier.get('chunks_with_parts_and_drawings', 0)}")
        print(f"Chunks With Full Payload       : {tier.get('chunks_with_full_payload', 0)}")

        samples = tier.get("sample_chunk_ids", {})

        print("\n--- SAMPLE CHUNK IDS ---")
        print(f"Chunk-Level Images : {samples.get('chunk_level_images', [])}")
        print(f"Document Images    : {samples.get('document_level_images', [])}")
        print(f"Parts              : {samples.get('parts', [])}")
        print(f"Drawings           : {samples.get('drawings', [])}")
        print(f"Full Payload       : {samples.get('full_payload', [])}")

        if result.get("status") == "success":
            print("\n✔ Tier summary scan completed successfully")
            return 0

        print("\nTier summary scan failed")
        return 2

    # --------------------------------------------------
    # PROBE / SPECIFIC CHUNK MODE
    # --------------------------------------------------
    chunk = result.get("chunk")
    if chunk:
        print("\n--- CHUNK INFO ---")
        print(f"Chunk ID            : {chunk.get('chunk_id')}")
        print(f"Chunk Name          : {chunk.get('chunk_name')}")
        print(f"Complete Document ID: {chunk.get('complete_document_id')}")
        print(f"Content Length      : {chunk.get('content_length')}")
        print("Preview:")
        print(chunk.get("preview", ""))

    graph = result.get("graph_summary", {})
    if graph:
        print("\n--- GRAPH SUMMARY ---")
        print(f"Images          : {graph.get('image_count', 0)}")
        print(f"Embeddings      : {graph.get('embedding_count', 0)}")
        print(f"Positions       : {graph.get('position_count', 0)}")
        print(f"Parts           : {graph.get('part_count', 0)}")
        print(f"Drawings        : {graph.get('drawing_count', 0)}")
        print(f"Has Trigger     : {graph.get('has_trigger_chunk', False)}")

    summary = result.get("ui_payload_summary", {})

    print("\n--- UI PAYLOAD SUMMARY ---")
    print(f"Documents Returned: {summary.get('document_count', 0)}")

    for doc in summary.get("documents", []):
        print("\nDocument:")
        print(f"  Complete Doc ID : {doc.get('complete_document_id')}")
        print(f"  Chunks          : {doc.get('chunk_count')}")
        print(f"  Images          : {doc.get('image_count')}")
        print(f"  Positions       : {doc.get('position_count')}")
        print(f"  Parts           : {doc.get('part_count')}")
        print(f"  Drawing Nav     : {doc.get('has_drawing_navigation')}")

    if result.get("status") == "success":
        print("\n✔ Enrichment probe completed successfully")
        return 0

    if result.get("status") == "not_found":
        print("\nNo trigger chunk found")
        return 1

    print("\nEnrichment probe failed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())