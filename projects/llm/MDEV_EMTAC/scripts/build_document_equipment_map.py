from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("document_equipment_map")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


@dataclass
class DocumentAssociationRecord:
    root_path: str
    relative_path: str
    file_name: str
    stem: str
    extension: str
    file_category: str
    is_document_like: bool
    parent_folder: str
    ancestor_path: str
    side: str
    area: str
    nearest_equipment_folder: str
    equipment_number: str
    equipment_number_source: str
    document_title: str
    association_confidence: str
    notes: str = ""


class TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


EQUIPMENT_PATTERNS = [
    re.compile(r"\b([A-Z]{2,6}\d{3,8}(?:-\d{1,4})?)\b"),
    re.compile(r"\b([A-Z]{2,10}-\d{2,10}(?:-\d{1,4})?)\b"),
]

AREA_KEYWORDS = {
    "filling": ["fill", "filling", "fillbay"],
    "overwrap": ["overwrap", "ow"],
    "packout": ["packout", "pack"],
    "hirise": ["hirise", "hi rise"],
    "extrusion": ["extrusion"],
    "subassembly": ["subassembly"],
    "sterilization": ["ster", "sterilization"],
    "conveyors": ["conveyor", "conveyors"],
    "robots": ["fanuc", "robot", "robots"],
    "solutions": ["solution", "solutions"],
    "dryside": ["dryside", "dry side"],
    "wetside": ["wetside", "wet side"],
    "agvs": ["agv", "agvs"],
    "cranes": ["crane", "cranes", "srm"],
}

DOCUMENT_EXTENSIONS = {
    ".htm", ".html", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".ppt", ".pptx", ".txt", ".rtf"
}


def now_iso() -> str:
    return datetime.now().isoformat()


def safe_read_text(file_path: Path) -> str:
    encodings = ["utf-8", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding, errors="strict")
        except Exception:
            continue
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def extract_html_title(file_path: Path) -> str:
    if file_path.suffix.lower() not in {".htm", ".html"}:
        return ""

    text = safe_read_text(file_path)
    if not text:
        return ""

    parser = TitleParser()
    try:
        parser.feed(text)
    except Exception:
        pass

    return " ".join(parser.title.split()).strip()


def guess_category(extension: str, mime_type: str) -> str:
    ext = extension.lower()

    if ext in {".htm", ".html"}:
        return "html"
    if ext in {".pdf"}:
        return "pdf"
    if ext in {".doc", ".docx"}:
        return "word"
    if ext in {".xls", ".xlsx", ".xlsm"}:
        return "excel"
    if ext in {".ppt", ".pptx"}:
        return "powerpoint"
    if ext in {".txt", ".rtf", ".md"}:
        return "text"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".tif", ".tiff"}:
        return "image"
    if ext in {".xml"}:
        return "xml"
    if ext in {".js"}:
        return "javascript"
    if ext in {".css"}:
        return "css"

    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("text/"):
        return "text"

    return "other"


def find_equipment_number(value: str) -> Optional[str]:
    if not value:
        return None

    value = value.replace("_", " ").replace("\\", " ").replace("/", " ")
    for pattern in EQUIPMENT_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    return None


def infer_side(relative_parts: List[str]) -> str:
    joined = " ".join(relative_parts).lower()
    if "wetside" in joined or "wetside" in joined:
        return "wetside"
    if "dryside" in joined or "dry side" in joined:
        return "dryside"
    return ""


def infer_area(relative_parts: List[str]) -> str:
    joined = " ".join(relative_parts).lower()
    for area, keywords in AREA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in joined:
                return area
    return ""


def nearest_equipment_folder(parts: List[str]) -> str:
    for part in reversed(parts):
        if find_equipment_number(part):
            return part
    return ""


def determine_equipment_number(file_path: Path, relative_parts: List[str], html_title: str) -> Tuple[str, str, str]:
    """
    Returns (equipment_number, source, confidence)
    """
    filename_hit = find_equipment_number(file_path.name)
    if filename_hit:
        return filename_hit, "file_name", "high"

    stem_hit = find_equipment_number(file_path.stem)
    if stem_hit:
        return stem_hit, "file_stem", "high"

    title_hit = find_equipment_number(html_title)
    if title_hit:
        return title_hit, "html_title", "medium"

    for part in reversed(relative_parts):
        folder_hit = find_equipment_number(part)
        if folder_hit:
            return folder_hit, "ancestor_folder", "medium"

    return "", "", "low"


def build_document_equipment_map(root_dir: Path) -> Tuple[List[DocumentAssociationRecord], Dict]:
    root_dir = root_dir.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    logger.info("Building document-equipment map for root: %s", root_dir)

    records: List[DocumentAssociationRecord] = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(root_dir).as_posix()
        relative_parts = list(path.relative_to(root_dir).parts)
        parent_folder = "." if path.parent == root_dir else path.parent.relative_to(root_dir).as_posix()
        ancestor_path = " / ".join(relative_parts[:-1])

        mime_type = mimetypes.guess_type(str(path))[0] or ""
        extension = path.suffix.lower()
        category = guess_category(extension, mime_type)
        is_document_like = extension in DOCUMENT_EXTENSIONS or category in {"html", "pdf", "word", "excel", "powerpoint", "text"}

        html_title = extract_html_title(path) if extension in {".htm", ".html"} else ""

        side = infer_side([p.lower() for p in relative_parts])
        area = infer_area([p.lower() for p in relative_parts])
        equipment_folder = nearest_equipment_folder(relative_parts[:-1])

        equipment_number, equipment_source, confidence = determine_equipment_number(
            file_path=path,
            relative_parts=relative_parts[:-1],
            html_title=html_title,
        )

        record = DocumentAssociationRecord(
            root_path=str(root_dir),
            relative_path=relative_path,
            file_name=path.name,
            stem=path.stem,
            extension=extension,
            file_category=category,
            is_document_like=is_document_like,
            parent_folder=parent_folder,
            ancestor_path=ancestor_path,
            side=side,
            area=area,
            nearest_equipment_folder=equipment_folder,
            equipment_number=equipment_number,
            equipment_number_source=equipment_source,
            document_title=html_title,
            association_confidence=confidence,
            notes="",
        )
        records.append(record)

    summary = build_summary(records, root_dir)
    logger.info("Document-equipment map complete. Records=%s", len(records))
    return records, summary


def build_summary(records: List[DocumentAssociationRecord], root_dir: Path) -> Dict:
    by_category: Dict[str, int] = {}
    by_area: Dict[str, int] = {}
    by_side: Dict[str, int] = {}
    by_confidence: Dict[str, int] = {}
    by_equipment: Dict[str, int] = {}

    for rec in records:
        by_category[rec.file_category] = by_category.get(rec.file_category, 0) + 1
        by_area[rec.area or "[blank]"] = by_area.get(rec.area or "[blank]", 0) + 1
        by_side[rec.side or "[blank]"] = by_side.get(rec.side or "[blank]", 0) + 1
        by_confidence[rec.association_confidence] = by_confidence.get(rec.association_confidence, 0) + 1
        if rec.equipment_number:
            by_equipment[rec.equipment_number] = by_equipment.get(rec.equipment_number, 0) + 1

    top_equipment = sorted(by_equipment.items(), key=lambda x: (-x[1], x[0]))[:100]

    return {
        "generated_at": now_iso(),
        "root_path": str(root_dir),
        "total_records": len(records),
        "counts_by_category": dict(sorted(by_category.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_area": dict(sorted(by_area.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_side": dict(sorted(by_side.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_confidence": dict(sorted(by_confidence.items(), key=lambda x: (-x[1], x[0]))),
        "top_equipment_numbers": [{"equipment_number": k, "count": v} for k, v in top_equipment],
    }


def write_csv(records: List[DocumentAssociationRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(asdict(records[0]).keys()) if records else list(DocumentAssociationRecord.__dataclass_fields__.keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def write_unresolved_csv(records: List[DocumentAssociationRecord], output_path: Path) -> None:
    unresolved = [r for r in records if not r.equipment_number]
    write_csv(unresolved, output_path)


def write_equipment_summary_csv(records: List[DocumentAssociationRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, int] = {}
    for rec in records:
        key = rec.equipment_number or "[unresolved]"
        grouped[key] = grouped.get(key, 0) + 1

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["equipment_number", "file_count"])
        writer.writeheader()
        for equipment_number, count in sorted(grouped.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow({"equipment_number": equipment_number, "file_count": count})


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only document-to-folder-and-equipment association map."
    )
    parser.add_argument("root", type=str, help="Root folder to scan.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="document_equipment_output",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    root_dir = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()

    records, summary = build_document_equipment_map(root_dir)

    main_csv = output_dir / "document_equipment_associations.csv"
    unresolved_csv = output_dir / "unresolved_documents.csv"
    equipment_csv = output_dir / "equipment_groups_summary.csv"
    summary_json = output_dir / "document_equipment_summary.json"

    write_csv(records, main_csv)
    write_unresolved_csv(records, unresolved_csv)
    write_equipment_summary_csv(records, equipment_csv)
    write_json(
        {
            "summary": summary,
            "records": [asdict(r) for r in records],
        },
        summary_json,
    )

    print("\nDocument-equipment map complete.")
    print(f"Root scanned: {root_dir}")
    print(f"Total records: {summary['total_records']}")
    print(f"Associations CSV: {main_csv}")
    print(f"Unresolved CSV: {unresolved_csv}")
    print(f"Equipment summary CSV: {equipment_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()