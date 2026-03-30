#!/usr/bin/env python
"""
build_clean_file_locations.py

Purpose:
    Read the source inventory CSV and generate a clean list of file locations
    for real documents/images while excluding HTML-export support assets.

Updated behavior:
    - Adds an "area" column
    - Infers area from file path:
        * /AddVBagFab  -> MMABF
        * /ConvBagFab  -> MMCBF

Typical usage:
    python build_clean_file_locations.py ^
      --inventory-csv "E:\\emtac\\projects\\llm\\MDEV_EMTAC\\mi_au_maint\\migration_analysis\\2026-03-25_032802\\inventory\\source_inventory.csv" ^
      --output-dir "E:\\emtac\\projects\\llm\\MDEV_EMTAC\\mi_au_maint\\migration_analysis\\2026-03-25_032802\\reports\\clean_file_locations" ^
      --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


LOGGER_NAME = "build_clean_file_locations"
logger = logging.getLogger(LOGGER_NAME)


INCLUDED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

LOW_VALUE_FILENAMES = {
    "thumbs.db",
    "filelist.xml",
    "cachedata.xml",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if logger.handlers:
        logger.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_lower(value) -> str:
    return safe_str(value).lower()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_slashes(path_value: str) -> str:
    return safe_str(path_value).replace("\\", "/")


def infer_area(relative_path: str, parent_folder: str, ancestor_path: str) -> tuple[str, str]:
    """
    Returns:
        (area_code, area_status)

    area_status values:
        ""          -> normal
        "obsolete"  -> area is obsolete
    """
    combined = " | ".join(
        [
            normalize_slashes(relative_path),
            normalize_slashes(parent_folder),
            normalize_slashes(ancestor_path),
        ]
    ).lower()

    # ---------- OBSOLETE ----------
    if "/fb5" in combined or "fb5" in combined:
        return "", "obsolete"

    if "/colpitt" in combined or "colpitt" in combined:
        return "", "obsolete"

    # ---------- MMDFL ----------
    if "/bag load" in combined or "bag load" in combined or "/bag_load" in combined or "bag_load" in combined:
        return "MMDFL", ""

    if "/taco guides" in combined or "taco guides" in combined or "/taco_guides" in combined or "taco_guides" in combined:
        return "MMDFL", ""

    if "/fb4" in combined or "fb4" in combined:
        return "MMDFL", ""

    # ---------- MMLFL ----------
    if "/fb1-2" in combined or "fb1-2" in combined or "/fb1_2" in combined or "fb1_2" in combined:
        return "MMLFL", ""

    if "/fb3-gen" in combined or "fb3-gen" in combined or "/fb3_gen" in combined or "fb3_gen" in combined:
        return "MMLFL", ""

    if "/fb6" in combined or "fb6" in combined:
        return "MMLFL", ""

    # ---------- MMSHR ----------
    if "/aaahirisegloba" in combined or "aaahirisegloba" in combined:
        return "MMSHR", ""

    if "/6_hirise" in combined or "6_hirise" in combined:
        return "MMSHR", ""

    # ---------- Others ----------
    if "/addvbagfab" in combined or "addvbagfab" in combined:
        return "MMABF", ""

    if "/convbagfab" in combined or "convbagfab" in combined:
        return "MMCBF", ""

    if "/3_overwrap" in combined or "3_overwrap" in combined:
        return "MMOWP", ""

    if "/4_sterilization" in combined or "4_sterilization" in combined:
        return "MMSTZ", ""

    if "/5_packout" in combined or "5_packout" in combined:
        return "MMPKG", ""

    if "/subassembly" in combined or "subassembly" in combined:
        return "MMSSU", ""

    if "/extrusion" in combined or "extrusion" in combined:
        return "MMSBX", ""

    return "", ""


def is_html_asset_path(relative_path: str, parent_folder: str, ancestor_path: str) -> bool:
    rel = safe_lower(relative_path).replace("\\", "/")
    parent = safe_lower(parent_folder).replace("\\", "/")
    ancestor = safe_lower(ancestor_path).replace("\\", "/")

    path_parts = [p for p in rel.split("/") if p]

    if any(part.endswith("_files") for part in path_parts):
        return True

    if parent.endswith("_files"):
        return True

    if "/_files/" in rel:
        return True

    if any(part.endswith(".htm_files") or part.endswith(".html_files") for part in path_parts):
        return True

    if "_files" in ancestor:
        return True

    return False


def classify_row(row: pd.Series) -> tuple[bool, str]:
    file_name = safe_lower(row.get("file_name", ""))
    extension = safe_lower(row.get("extension", ""))
    relative_path = safe_str(row.get("relative_path", ""))
    parent_folder = safe_str(row.get("parent_folder", ""))
    ancestor_path = safe_str(row.get("ancestor_path", ""))

    if not extension or extension not in INCLUDED_EXTENSIONS:
        return False, "excluded_extension"

    if file_name in LOW_VALUE_FILENAMES:
        return False, "low_value_support_file"

    if extension in IMAGE_EXTENSIONS and is_html_asset_path(relative_path, parent_folder, ancestor_path):
        return False, "html_support_image"

    return True, "included"


def build_output_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        include, reason = classify_row(row)

        if not include:
            continue

        root_path = safe_str(row.get("root_path", ""))
        relative_path = safe_str(row.get("relative_path", ""))
        extension = safe_lower(row.get("extension", ""))
        file_name = safe_str(row.get("file_name", ""))
        parent_folder = safe_str(row.get("parent_folder", ""))
        ancestor_path = safe_str(row.get("ancestor_path", ""))

        full_path = str(Path(root_path) / relative_path) if root_path and relative_path else ""

        if extension in DOCUMENT_EXTENSIONS:
            file_group = "document"
        elif extension in IMAGE_EXTENSIONS:
            file_group = "image"
        else:
            file_group = "other"

        area = infer_area(
            relative_path=relative_path,
            parent_folder=parent_folder,
            ancestor_path=ancestor_path,
        )

        rows.append(
            {
                "area": area,
                "root_path": root_path,
                "relative_path": relative_path,
                "full_path": full_path,
                "file_name": file_name,
                "extension": extension,
                "file_group": file_group,
                "file_category": safe_str(row.get("file_category", "")),
                "parent_folder": parent_folder,
                "ancestor_path": ancestor_path,
                "include_reason": reason,
            }
        )

    out_df = pd.DataFrame(rows)

    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["area", "file_group", "extension", "relative_path", "file_name"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)

    return out_df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    logger.info("Writing CSV: %s", path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: dict, path: Path) -> None:
    logger.info("Writing JSON: %s", path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean file-location list from source inventory."
    )
    parser.add_argument(
        "--inventory-csv",
        required=True,
        help="Path to source_inventory.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    inventory_csv = Path(args.inventory_csv)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    if not inventory_csv.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory_csv}")

    logger.info("Loading inventory CSV: %s", inventory_csv)
    df = pd.read_csv(inventory_csv, dtype=str, keep_default_na=False)

    required_cols = {"root_path", "relative_path", "file_name", "extension"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Inventory CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    clean_df = build_output_df(df)

    documents_df = clean_df[clean_df["file_group"] == "document"].copy()
    images_df = clean_df[clean_df["file_group"] == "image"].copy()

    all_csv = output_dir / "clean_file_locations_all.csv"
    documents_csv = output_dir / "clean_file_locations_documents.csv"
    images_csv = output_dir / "clean_file_locations_images.csv"
    summary_json = output_dir / "clean_file_locations_summary.json"

    write_csv(clean_df, all_csv)
    write_csv(documents_df, documents_csv)
    write_csv(images_df, images_csv)

    summary = {
        "inventory_csv": str(inventory_csv),
        "output_dir": str(output_dir),
        "total_clean_rows": int(len(clean_df)),
        "document_rows": int(len(documents_df)),
        "image_rows": int(len(images_df)),
        "counts_by_extension": (
            clean_df["extension"].value_counts().sort_index().to_dict()
            if not clean_df.empty else {}
        ),
        "counts_by_area": (
            clean_df["area"].replace("", "[blank]").value_counts().sort_index().to_dict()
            if not clean_df.empty else {}
        ),
    }
    write_json(summary, summary_json)

    print("Clean file location build complete.")
    print(f"Output directory: {output_dir}")
    print(f"All CSV: {all_csv}")
    print(f"Documents CSV: {documents_csv}")
    print(f"Images CSV: {images_csv}")
    print(f"Summary JSON: {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())