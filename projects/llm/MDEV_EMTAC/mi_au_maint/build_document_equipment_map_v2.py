from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACKAGE_DIR = CURRENT_FILE.parent.resolve()


logger = logging.getLogger("document_equipment_map_v2")


# ============================================================
# Logging
# ============================================================
def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ============================================================
# Data models
# ============================================================
@dataclass
class HierarchyContext:
    page_relative_path: str
    page_title: str
    level: int
    guessed_page_role: str
    first_discovered_from: str
    first_discovered_via_text: str
    first_discovered_via_target: str


@dataclass
class DocumentAssociationRecordV2:
    root_path: str
    relative_path: str
    file_name: str
    stem: str
    extension: str
    file_category: str
    is_document_like: bool
    parent_folder: str
    ancestor_path: str

    page_title: str
    hierarchy_level: int
    guessed_page_role: str
    first_discovered_from: str
    first_discovered_via_text: str

    side_direct: str
    side_inherited: str
    side_final: str

    area_direct: str
    area_inherited: str
    area_final: str

    equipment_number_direct: str
    equipment_number_direct_source: str

    inherited_equipment_number: str
    inherited_equipment_source: str

    equipment_number_final: str
    equipment_number_final_source: str

    nearest_equipment_folder: str
    nearest_launcher_ancestor: str
    nearest_launcher_ancestor_title: str
    nearest_content_ancestor: str
    nearest_content_ancestor_title: str

    association_confidence: str
    notes: str = ""


# ============================================================
# HTML title parser
# ============================================================
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


# ============================================================
# Constants / heuristics
# ============================================================
EQUIPMENT_PATTERNS = [
    re.compile(r"\b([A-Z]{2,6}\d{3,8}(?:-\d{1,4})?)\b"),
    re.compile(r"\b([A-Z]{2,10}-\d{2,10}(?:-\d{1,4})?)\b"),
]

AREA_KEYWORDS = {
    "filling": ["fill", "filling", "fillbay"],
    "overwrap": ["overwrap", "ow"],
    "packout": ["packout", "pack"],
    "hirise": ["hirise", "hi rise", "hi-rise"],
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
    ".htm",
    ".html",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ppt",
    ".pptx",
    ".txt",
    ".rtf",
}


# ============================================================
# Helpers
# ============================================================
def now_iso() -> str:
    return datetime.now().isoformat()


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def ensure_dir_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


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
    if ext in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".svg",
        ".webp",
        ".tif",
        ".tiff",
    }:
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

    normalized = value.replace("_", " ").replace("\\", " ").replace("/", " ")
    for pattern in EQUIPMENT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1)
    return None


def infer_side_from_text(value: str) -> str:
    text = (value or "").lower()
    if "wetside" in text or "wet side" in text:
        return "wetside"
    if "dryside" in text or "dry side" in text:
        return "dryside"
    return ""


def infer_side(relative_parts: List[str]) -> str:
    return infer_side_from_text(" ".join(relative_parts))


def infer_area_from_text(value: str) -> str:
    text = (value or "").lower()
    for area, keywords in AREA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return area
    return ""


def infer_area(relative_parts: List[str]) -> str:
    return infer_area_from_text(" ".join(relative_parts))


def nearest_equipment_folder(parts: List[str]) -> str:
    for part in reversed(parts):
        if find_equipment_number(part):
            return part
    return ""


def determine_direct_equipment_number(
    file_path: Path,
    relative_parts: List[str],
    html_title: str,
) -> Tuple[str, str, str]:
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


def relative_to_root(path: Path, root_dir: Path) -> str:
    return path.relative_to(root_dir).as_posix()


# ============================================================
# Hierarchy loading
# ============================================================
def load_hierarchy_contexts(hierarchy_json_path: Path) -> Dict[str, HierarchyContext]:
    ensure_file_exists(hierarchy_json_path, "Hierarchy JSON")

    data = json.loads(hierarchy_json_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])

    contexts: Dict[str, HierarchyContext] = {}
    for page in pages:
        relative_path = page.get("page_relative_path", "")
        if not relative_path:
            continue

        contexts[relative_path] = HierarchyContext(
            page_relative_path=relative_path,
            page_title=page.get("page_title", "") or "",
            level=int(page.get("level", -1)),
            guessed_page_role=page.get("guessed_page_role", "") or "",
            first_discovered_from=page.get("first_discovered_from", "") or "",
            first_discovered_via_text=page.get("first_discovered_via_text", "") or "",
            first_discovered_via_target=page.get("first_discovered_via_target", "") or "",
        )

    return contexts


# ============================================================
# Inheritance logic
# ============================================================
def resolve_hierarchy_chain(
    relative_path: str,
    hierarchy_contexts: Dict[str, HierarchyContext],
) -> List[HierarchyContext]:
    """
    Build chain from current page upward through first_discovered_from.
    """
    chain: List[HierarchyContext] = []
    seen = set()

    current = hierarchy_contexts.get(relative_path)
    while current and current.page_relative_path not in seen:
        chain.append(current)
        seen.add(current.page_relative_path)

        parent_path = current.first_discovered_from
        if not parent_path:
            break

        current = hierarchy_contexts.get(parent_path)

    return chain


def find_nearest_launcher_ancestor(
    chain: List[HierarchyContext],
    current_relative_path: str,
) -> Tuple[str, str]:
    """
    Find nearest ancestor in the chain with launcher/root-navigation role.
    Skip the current page itself if possible.
    """
    for ctx in chain[1:]:
        if ctx.guessed_page_role in {"launcher", "root_navigation", "entry_page"}:
            return ctx.page_relative_path, ctx.page_title

    if chain and chain[0].guessed_page_role in {
        "launcher",
        "root_navigation",
        "entry_page",
    }:
        return chain[0].page_relative_path, chain[0].page_title

    return "", ""


def find_nearest_content_ancestor(
    chain: List[HierarchyContext],
) -> Tuple[str, str]:
    """
    Find nearest non-generic ancestor page with useful title.
    """
    for ctx in chain[1:]:
        title = (ctx.page_title or "").strip()
        if title and title.lower() not in {"page title", "drafting"}:
            return ctx.page_relative_path, ctx.page_title

    if chain:
        title = (chain[0].page_title or "").strip()
        if title and title.lower() not in {"page title", "drafting"}:
            return chain[0].page_relative_path, chain[0].page_title

    return "", ""


def infer_inherited_equipment_number(
    chain: List[HierarchyContext],
) -> Tuple[str, str]:
    """
    Walk upward through the hierarchy chain and try to infer equipment number
    from page title, path, or discovery text.
    """
    for ctx in chain:
        path_hit = find_equipment_number(ctx.page_relative_path)
        if path_hit:
            return path_hit, "hierarchy_page_path"

        title_hit = find_equipment_number(ctx.page_title)
        if title_hit:
            return title_hit, "hierarchy_page_title"

        discovery_hit = find_equipment_number(ctx.first_discovered_via_text)
        if discovery_hit:
            return discovery_hit, "hierarchy_discovery_text"

    return "", ""


def infer_inherited_side(
    chain: List[HierarchyContext],
) -> str:
    for ctx in chain:
        for candidate in (
            ctx.page_relative_path,
            ctx.page_title,
            ctx.first_discovered_via_text,
        ):
            hit = infer_side_from_text(candidate)
            if hit:
                return hit
    return ""


def infer_inherited_area(
    chain: List[HierarchyContext],
) -> str:
    for ctx in chain:
        for candidate in (
            ctx.page_relative_path,
            ctx.page_title,
            ctx.first_discovered_via_text,
        ):
            hit = infer_area_from_text(candidate)
            if hit:
                return hit
    return ""


def choose_final_value(direct_value: str, inherited_value: str) -> str:
    return direct_value or inherited_value


def determine_final_confidence(
    direct_equipment_number: str,
    direct_source: str,
    inherited_equipment_number: str,
    side_final: str,
    area_final: str,
) -> str:
    if direct_equipment_number and direct_source in {"file_name", "file_stem"}:
        return "high"

    if direct_equipment_number:
        return "medium"

    if inherited_equipment_number and side_final and area_final:
        return "medium"

    if inherited_equipment_number:
        return "low-medium"

    if side_final or area_final:
        return "low"

    return "low"


# ============================================================
# Main builder
# ============================================================
def build_document_equipment_map_v2(
    root_dir: Path,
    hierarchy_json_path: Path,
) -> Tuple[List[DocumentAssociationRecordV2], Dict]:
    root_dir = root_dir.resolve()
    hierarchy_json_path = hierarchy_json_path.resolve()

    ensure_dir_exists(root_dir, "Root directory")
    ensure_file_exists(hierarchy_json_path, "Hierarchy JSON")

    hierarchy_contexts = load_hierarchy_contexts(hierarchy_json_path)
    logger.info("Loaded hierarchy contexts: %s", len(hierarchy_contexts))

    records: List[DocumentAssociationRecordV2] = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        relative_path = relative_to_root(path, root_dir)
        relative_parts = list(path.relative_to(root_dir).parts)
        parent_folder = "." if path.parent == root_dir else relative_to_root(path.parent, root_dir)
        ancestor_path = " / ".join(relative_parts[:-1])

        mime_type = mimetypes.guess_type(str(path))[0] or ""
        extension = path.suffix.lower()
        category = guess_category(extension, mime_type)
        is_document_like = extension in DOCUMENT_EXTENSIONS or category in {
            "html",
            "pdf",
            "word",
            "excel",
            "powerpoint",
            "text",
        }

        html_title = extract_html_title(path) if extension in {".htm", ".html"} else ""

        side_direct = infer_side([p.lower() for p in relative_parts])
        area_direct = infer_area([p.lower() for p in relative_parts])
        equipment_folder = nearest_equipment_folder(relative_parts[:-1])

        direct_equipment_number, direct_equipment_source, _ = determine_direct_equipment_number(
            file_path=path,
            relative_parts=relative_parts[:-1],
            html_title=html_title,
        )

        ctx = hierarchy_contexts.get(relative_path)
        hierarchy_level = ctx.level if ctx else -1
        guessed_page_role = ctx.guessed_page_role if ctx else ""
        first_discovered_from = ctx.first_discovered_from if ctx else ""
        first_discovered_via_text = ctx.first_discovered_via_text if ctx else ""

        chain = resolve_hierarchy_chain(relative_path, hierarchy_contexts) if ctx else []

        inherited_equipment_number, inherited_equipment_source = infer_inherited_equipment_number(chain)
        side_inherited = infer_inherited_side(chain)
        area_inherited = infer_inherited_area(chain)

        side_final = choose_final_value(side_direct, side_inherited)
        area_final = choose_final_value(area_direct, area_inherited)

        equipment_number_final = direct_equipment_number or inherited_equipment_number
        equipment_number_final_source = direct_equipment_source or inherited_equipment_source

        nearest_launcher_ancestor, nearest_launcher_ancestor_title = find_nearest_launcher_ancestor(
            chain,
            relative_path,
        )
        nearest_content_ancestor, nearest_content_ancestor_title = find_nearest_content_ancestor(chain)

        final_confidence = determine_final_confidence(
            direct_equipment_number=direct_equipment_number,
            direct_source=direct_equipment_source,
            inherited_equipment_number=inherited_equipment_number,
            side_final=side_final,
            area_final=area_final,
        )

        record = DocumentAssociationRecordV2(
            root_path=str(root_dir),
            relative_path=relative_path,
            file_name=path.name,
            stem=path.stem,
            extension=extension,
            file_category=category,
            is_document_like=is_document_like,
            parent_folder=parent_folder,
            ancestor_path=ancestor_path,

            page_title=html_title or (ctx.page_title if ctx else ""),
            hierarchy_level=hierarchy_level,
            guessed_page_role=guessed_page_role,
            first_discovered_from=first_discovered_from,
            first_discovered_via_text=first_discovered_via_text,

            side_direct=side_direct,
            side_inherited=side_inherited,
            side_final=side_final,

            area_direct=area_direct,
            area_inherited=area_inherited,
            area_final=area_final,

            equipment_number_direct=direct_equipment_number,
            equipment_number_direct_source=direct_equipment_source,

            inherited_equipment_number=inherited_equipment_number,
            inherited_equipment_source=inherited_equipment_source,

            equipment_number_final=equipment_number_final,
            equipment_number_final_source=equipment_number_final_source,

            nearest_equipment_folder=equipment_folder,
            nearest_launcher_ancestor=nearest_launcher_ancestor,
            nearest_launcher_ancestor_title=nearest_launcher_ancestor_title,
            nearest_content_ancestor=nearest_content_ancestor,
            nearest_content_ancestor_title=nearest_content_ancestor_title,

            association_confidence=final_confidence,
            notes="",
        )
        records.append(record)

    summary = build_summary(records, root_dir, hierarchy_json_path)
    logger.info("Document-equipment map v2 complete. Records=%s", len(records))
    return records, summary


# ============================================================
# Summary / outputs
# ============================================================
def build_summary(
    records: List[DocumentAssociationRecordV2],
    root_dir: Path,
    hierarchy_json_path: Path,
) -> Dict:
    by_category: Dict[str, int] = {}
    by_area: Dict[str, int] = {}
    by_side: Dict[str, int] = {}
    by_confidence: Dict[str, int] = {}
    by_equipment: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    hierarchy_bound_count = 0

    for rec in records:
        by_category[rec.file_category] = by_category.get(rec.file_category, 0) + 1
        by_area[rec.area_final or "[blank]"] = by_area.get(rec.area_final or "[blank]", 0) + 1
        by_side[rec.side_final or "[blank]"] = by_side.get(rec.side_final or "[blank]", 0) + 1
        by_confidence[rec.association_confidence] = by_confidence.get(rec.association_confidence, 0) + 1
        by_source[rec.equipment_number_final_source or "[blank]"] = by_source.get(
            rec.equipment_number_final_source or "[blank]",
            0,
        ) + 1

        if rec.hierarchy_level >= 0:
            hierarchy_bound_count += 1

        if rec.equipment_number_final:
            by_equipment[rec.equipment_number_final] = by_equipment.get(
                rec.equipment_number_final,
                0,
            ) + 1

    top_equipment = sorted(by_equipment.items(), key=lambda x: (-x[1], x[0]))[:100]

    return {
        "generated_at": now_iso(),
        "root_path": str(root_dir),
        "hierarchy_json_path": str(hierarchy_json_path),
        "total_records": len(records),
        "records_with_hierarchy_context": hierarchy_bound_count,
        "counts_by_category": dict(sorted(by_category.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_area_final": dict(sorted(by_area.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_side_final": dict(sorted(by_side.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_confidence": dict(sorted(by_confidence.items(), key=lambda x: (-x[1], x[0]))),
        "counts_by_equipment_source": dict(sorted(by_source.items(), key=lambda x: (-x[1], x[0]))),
        "top_equipment_numbers": [
            {"equipment_number": k, "count": v}
            for k, v in top_equipment
        ],
    }


def write_csv(records: List[DocumentAssociationRecordV2], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        list(asdict(records[0]).keys())
        if records
        else list(DocumentAssociationRecordV2.__dataclass_fields__.keys())
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def write_unresolved_csv(
    records: List[DocumentAssociationRecordV2],
    output_path: Path,
) -> None:
    unresolved = [r for r in records if not r.equipment_number_final]
    write_csv(unresolved, output_path)


def write_hierarchy_bound_csv(
    records: List[DocumentAssociationRecordV2],
    output_path: Path,
) -> None:
    hierarchy_bound = [r for r in records if r.hierarchy_level >= 0]
    write_csv(hierarchy_bound, output_path)


def write_equipment_summary_csv(
    records: List[DocumentAssociationRecordV2],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, int] = {}
    for rec in records:
        key = rec.equipment_number_final or "[unresolved]"
        grouped[key] = grouped.get(key, 0) + 1

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["equipment_number_final", "file_count"],
        )
        writer.writeheader()
        for equipment_number, count in sorted(grouped.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow(
                {
                    "equipment_number_final": equipment_number,
                    "file_count": count,
                }
            )


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only document-to-folder-and-equipment association map "
            "using site_hierarchy.json inheritance."
        )
    )
    parser.add_argument("root", type=str, help="Root folder to scan.")
    parser.add_argument(
        "--hierarchy-json",
        type=str,
        required=True,
        help="Path to site_hierarchy.json produced by build_site_hierarchy.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PACKAGE_DIR),
        help="Directory for canonical output files.",
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
    hierarchy_json_path = Path(args.hierarchy_json).resolve()
    output_dir = Path(args.output_dir).resolve()

    ensure_dir_exists(root_dir, "Root directory")
    ensure_file_exists(hierarchy_json_path, "Hierarchy JSON")
    output_dir.mkdir(parents=True, exist_ok=True)

    records, summary = build_document_equipment_map_v2(root_dir, hierarchy_json_path)

    main_csv = output_dir / "document_equipment_associations_v2.csv"
    unresolved_csv = output_dir / "unresolved_documents_v2.csv"
    hierarchy_bound_csv = output_dir / "hierarchy_bound_documents_v2.csv"
    equipment_csv = output_dir / "equipment_groups_summary_v2.csv"
    summary_json = output_dir / "document_equipment_summary_v2.json"

    write_csv(records, main_csv)
    write_unresolved_csv(records, unresolved_csv)
    write_hierarchy_bound_csv(records, hierarchy_bound_csv)
    write_equipment_summary_csv(records, equipment_csv)
    write_json(
        {
            "summary": summary,
            "records": [asdict(r) for r in records],
        },
        summary_json,
    )

    print("\nDocument-equipment map v2 complete.")
    print(f"Root scanned: {root_dir}")
    print(f"Hierarchy JSON: {hierarchy_json_path}")
    print(f"Output dir: {output_dir}")
    print(f"Total records: {summary['total_records']}")
    print(f"Records with hierarchy context: {summary['records_with_hierarchy_context']}")
    print(f"Associations CSV: {main_csv}")
    print(f"Hierarchy-bound CSV: {hierarchy_bound_csv}")
    print(f"Unresolved CSV: {unresolved_csv}")
    print(f"Equipment summary CSV: {equipment_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()