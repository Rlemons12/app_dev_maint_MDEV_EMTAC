#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script that uses:

    - DocumentStructureExtractor  (Stage 1)
    - StructureChunkLoader        (Stage 2)

This script:
1. Extracts the document structure
2. Saves _structure.json
3. Loads the structure
4. Produces clean, Q&A-ready chunks
5. Prints them or saves them if needed

Perfect test harness before Q&A generation.
"""

import argparse
from pathlib import Path

from structure_extractor import DocumentStructureExtractor
from structure_chunk_loader import StructureChunkLoader


def main():
    parser = argparse.ArgumentParser(description="Extract + Clean document chunks")
    parser.add_argument("input", help="Path to document (.pdf, .docx, .pptx, .txt)")
    parser.add_argument("--out-dir", default="structure_maps", help="Where to save structure json")
    parser.add_argument("--min-length", type=int, default=40, help="Min chunk length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--no-merge-headings", action="store_true", help="Disable heading merging")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Stage 1: Extract structure.json
    # ----------------------------------------------------------
    print("\n[1] Extracting document structure...")
    extractor = DocumentStructureExtractor(str(input_path))
    structure = extractor.extract()

    structure_json_path = out_dir / f"{input_path.stem}_structure.json"
    extractor.save(structure, structure_json_path)

    print(f"[SUCCESS] Structure saved to: {structure_json_path}")

    # ----------------------------------------------------------
    # Stage 2: Load + clean chunks
    # ----------------------------------------------------------
    print("\n[2] Cleaning chunks for Q&A generation...")

    loader = StructureChunkLoader(
        structure_path=str(structure_json_path),
        min_length=args.min_length,
        dedupe=not args.no_dedupe,
        merge_headings=not args.no_merge_headings
    )

    clean_chunks = loader.load_clean_chunks()

    print(f"[SUCCESS] {len(clean_chunks)} clean chunks ready.")

    print("\n------------- CLEAN CHUNKS -------------")
    for i, c in enumerate(clean_chunks):
        print(f"\n---- Chunk {i} (page {c['page']}) ----")
        print(c["text"])
    print("----------------------------------------\n")

    print("[DONE]")


if __name__ == "__main__":
    main()
