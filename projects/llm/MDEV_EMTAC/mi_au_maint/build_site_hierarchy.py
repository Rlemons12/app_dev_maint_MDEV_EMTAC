from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent.resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("site_hierarchy")


def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ============================================================
# Data models
# ============================================================
@dataclass
class HierarchyPageRecord:
    """
    One row per crawled HTML page.
    """
    root_path: str
    seed_page: str
    page_relative_path: str
    page_file_name: str
    page_title: str
    page_parent_folder: str
    level: int
    first_discovered_from: str
    first_discovered_via_text: str
    first_discovered_via_target: str
    inbound_html_link_count: int
    outbound_html_link_count: int
    external_link_count: int
    document_link_count: int
    asset_reference_count: int
    missing_local_reference_count: int
    anchor_reference_count: int
    guessed_page_role: str
    crawl_status: str
    notes: str = ""


@dataclass
class HierarchyEdgeRecord:
    """
    One row per parent -> child relationship or reference discovered.
    """
    root_path: str
    seed_page: str
    source_page: str
    source_title: str
    source_level: int
    target_raw: str
    target_normalized: str
    target_kind: str
    target_relative_path: str
    target_exists: bool
    target_is_crawled_html: bool
    target_level: int
    is_external: bool
    is_anchor_only: bool
    link_text: str
    tag_name: str
    attribute_name: str
    notes: str = ""


@dataclass
class MissingReferenceRecord:
    """
    One row per missing local reference.
    """
    root_path: str
    seed_page: str
    source_page: str
    source_title: str
    source_level: int
    target_raw: str
    target_normalized: str
    target_kind: str
    expected_relative_path: str
    tag_name: str
    attribute_name: str
    link_text: str
    notes: str = ""


# ============================================================
# Utility helpers
# ============================================================
def now_iso() -> str:
    return datetime.now().isoformat()


def safe_read_text(file_path: Path) -> str:
    """
    Read text using a small fallback chain suitable for exported HTML.
    """
    encodings = ["utf-8", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding, errors="strict")
        except Exception:
            continue

    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Could not read text file: %s | reason=%s", file_path, exc)
        return ""


def normalize_reference(raw_value: str) -> str:
    """
    Normalize an HTML reference target conservatively.
    """
    value = (raw_value or "").strip()
    if not value:
        return ""
    value = value.replace("\\", "/")
    value = unquote(value)
    return value


def is_external_target(target: str) -> bool:
    """
    Determine whether a target is external.
    """
    if not target:
        return False

    parsed = urlparse(target)

    if parsed.scheme and parsed.scheme.lower() not in {"file"}:
        return True

    if target.startswith("//"):
        return True

    return False


def is_data_uri(target: str) -> bool:
    return target.lower().startswith("data:")


def is_anchor_only(target: str) -> bool:
    return target.startswith("#")


def strip_fragment_and_query(target: str) -> str:
    """
    Remove query/fragment and return path portion only.
    """
    if not target:
        return ""
    parsed = urlparse(target)
    return parsed.path or ""


def resolve_local_target(source_page: Path, normalized_target: str) -> Path:
    """
    Resolve a local target relative to the source page.
    """
    cleaned = strip_fragment_and_query(normalized_target)
    return (source_page.parent / cleaned).resolve()


def is_within_root(path: Path, root_dir: Path) -> bool:
    """
    Check whether a resolved path stays within the crawl root.
    """
    try:
        path.relative_to(root_dir.resolve())
        return True
    except Exception:
        return False


def guess_category(extension: str, mime_type: str) -> str:
    """
    Assign a rough category.
    """
    ext = extension.lower()

    if ext in {".htm", ".html"}:
        return "html"
    if ext in {".xml"}:
        return "xml"
    if ext in {".pub"}:
        return "publisher"
    if ext in {".css"}:
        return "css"
    if ext in {".js"}:
        return "javascript"
    if ext in {".json"}:
        return "json"
    if ext in {".csv"}:
        return "csv"
    if ext in {".txt", ".md", ".rtf"}:
        return "text"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".tif", ".tiff"}:
        return "image"
    if ext in {".pdf"}:
        return "pdf"
    if ext in {".doc", ".docx"}:
        return "word"
    if ext in {".xls", ".xlsx", ".xlsm"}:
        return "excel"
    if ext in {".ppt", ".pptx"}:
        return "powerpoint"
    if ext in {".zip", ".7z", ".rar", ".tar", ".gz"}:
        return "archive"
    if ext in {".mp4", ".avi", ".mov", ".wmv", ".mkv"}:
        return "video"
    if ext in {".mp3", ".wav", ".m4a", ".aac"}:
        return "audio"

    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("text/"):
        return "text"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"

    return "other"


def classify_target_kind(
    tag_name: str,
    attribute_name: str,
    normalized_target: str,
    local_extension: str,
    local_category: str,
) -> str:
    """
    Classify a discovered reference.
    """
    if is_data_uri(normalized_target):
        return "data_uri"

    if is_anchor_only(normalized_target):
        return "anchor"

    if attribute_name == "src" and tag_name == "img":
        return "image"

    if attribute_name == "src" and tag_name == "script":
        return "script"

    if tag_name == "link" and attribute_name == "href":
        if local_extension == ".css":
            return "stylesheet"
        return "link_resource"

    if local_category == "html":
        return "html"
    if local_category == "image":
        return "image"
    if local_category in {"pdf", "word", "excel", "powerpoint"}:
        return "document"
    if local_category == "javascript":
        return "script"
    if local_category == "css":
        return "stylesheet"
    if local_category == "xml":
        return "xml"

    if tag_name == "a" and attribute_name == "href":
        return "href"

    return "other"


def guess_page_role(
    page_title: str,
    page_file_name: str,
    outbound_html_link_count: int,
    inbound_html_link_count: int,
) -> str:
    """
    Light heuristic for planning.
    """
    title = (page_title or "").lower()
    name = (page_file_name or "").lower()

    if "main" in title or "launch" in title or "menu" in title or "home" == title.strip():
        return "launcher"

    if outbound_html_link_count >= 4 and inbound_html_link_count == 0:
        return "root_navigation"

    if name in {"index.htm", "index.html", "default.htm", "default.html"}:
        return "entry_page"

    if name.startswith("page"):
        return "publisher_subpage"

    return "content_page"


# ============================================================
# HTML parsing
# ============================================================
class LinkExtractingHtmlParser(HTMLParser):
    """
    Extract title and references from HTML.
    """

    LINK_ATTRS = {
        "a": ["href"],
        "img": ["src"],
        "script": ["src"],
        "link": ["href"],
        "iframe": ["src"],
        "embed": ["src"],
        "source": ["src", "srcset"],
        "object": ["data"],
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self._in_title: bool = False

        self.references: List[Dict[str, str]] = []

        self._inside_anchor: bool = False
        self._current_anchor_text_parts: List[str] = []
        self._current_anchor_index: Optional[int] = None

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs if k}

        if tag == "title":
            self._in_title = True

        if tag == "a":
            self._inside_anchor = True
            self._current_anchor_text_parts = []

        if tag in self.LINK_ATTRS:
            for attr_name in self.LINK_ATTRS[tag]:
                if attr_name not in attr_map:
                    continue

                raw_value = attr_map.get(attr_name, "").strip()
                if not raw_value:
                    continue

                if tag == "source" and attr_name == "srcset":
                    candidates = [
                        part.strip().split(" ")[0]
                        for part in raw_value.split(",")
                        if part.strip()
                    ]
                    for candidate in candidates:
                        self.references.append(
                            {
                                "tag_name": tag,
                                "attribute_name": "src",
                                "raw_target": candidate,
                                "link_text": "",
                            }
                        )
                    continue

                self.references.append(
                    {
                        "tag_name": tag,
                        "attribute_name": attr_name,
                        "raw_target": raw_value,
                        "link_text": "",
                    }
                )

                if tag == "a" and attr_name == "href":
                    self._current_anchor_index = len(self.references) - 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = False

        if tag == "a":
            if (
                self._current_anchor_index is not None
                and 0 <= self._current_anchor_index < len(self.references)
            ):
                anchor_text = " ".join(
                    part.strip()
                    for part in self._current_anchor_text_parts
                    if part.strip()
                ).strip()
                self.references[self._current_anchor_index]["link_text"] = anchor_text

            self._inside_anchor = False
            self._current_anchor_text_parts = []
            self._current_anchor_index = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data

        if self._inside_anchor:
            self._current_anchor_text_parts.append(data)


# ============================================================
# Writers
# ============================================================
def write_csv_from_dataclasses(
    records: List[object],
    output_path: Path,
    fallback_headers: Optional[List[str]] = None,
) -> None:
    """
    Generic CSV writer for dataclass lists.
    """
    logger.info("Writing CSV: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        headers = fallback_headers or []
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    headers = list(asdict(records[0]).keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def write_json(payload: Dict, output_path: Path) -> None:
    """
    Write JSON payload.
    """
    logger.info("Writing JSON: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# Crawl internals
# ============================================================
def parse_html_page(page_path: Path) -> Tuple[str, List[Dict[str, str]]]:
    """
    Parse one HTML page and return its title + extracted references.
    """
    html_text = safe_read_text(page_path)
    parser = LinkExtractingHtmlParser()

    try:
        parser.feed(html_text)
    except Exception as exc:
        logger.warning(
            "Could not fully parse HTML page: %s | reason=%s",
            page_path,
            exc,
        )

    page_title = " ".join(parser.title.split()).strip()
    return page_title, parser.references


def make_relative(path: Path, root_dir: Path) -> str:
    """
    Convert absolute path to root-relative POSIX string.
    """
    return path.resolve().relative_to(root_dir.resolve()).as_posix()


def build_site_hierarchy(
    seed_page: Path,
    root_dir: Path,
    max_depth: Optional[int] = None,
    max_pages: Optional[int] = None,
    progress_every: int = 100,
) -> Tuple[List[HierarchyPageRecord], List[HierarchyEdgeRecord], List[MissingReferenceRecord], Dict]:
    """
    Crawl local HTML pages breadth-first from a seed page.
    """
    root_dir = root_dir.resolve()
    seed_page = seed_page.resolve()

    if not seed_page.exists():
        raise FileNotFoundError(f"Seed page does not exist: {seed_page}")
    if not seed_page.is_file():
        raise FileNotFoundError(f"Seed page is not a file: {seed_page}")
    if seed_page.suffix.lower() not in {".htm", ".html"}:
        raise ValueError(f"Seed page is not an HTML file: {seed_page}")
    if not is_within_root(seed_page, root_dir):
        raise ValueError(
            f"Seed page is not within root directory.\nSeed: {seed_page}\nRoot: {root_dir}"
        )

    seed_relative = make_relative(seed_page, root_dir)

    logger.info("Starting BFS site hierarchy crawl.")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Package dir: %s", PACKAGE_DIR)
    logger.info("Root: %s", root_dir)
    logger.info("Seed: %s", seed_page)
    logger.info("Max depth: %s", max_depth if max_depth is not None else "unlimited")
    logger.info("Max pages: %s", max_pages if max_pages is not None else "unlimited")

    queue: Deque[Tuple[Path, int, str, str, str]] = deque()
    queue.append((seed_page, 0, "", "", seed_relative))

    visited_pages: Set[str] = set()
    page_records_by_path: Dict[str, HierarchyPageRecord] = {}
    edge_records: List[HierarchyEdgeRecord] = []
    missing_records: List[MissingReferenceRecord] = []

    inbound_html_counter: Counter[str] = Counter()
    first_discovery: Dict[str, Tuple[str, str, str]] = {
        seed_relative: ("", "", seed_relative)
    }

    parsed_page_count = 0

    while queue:
        (
            current_page,
            current_level,
            discovered_from,
            discovered_via_text,
            discovered_via_target,
        ) = queue.popleft()

        current_relative = make_relative(current_page, root_dir)

        if current_relative in visited_pages:
            continue

        if max_pages is not None and parsed_page_count >= max_pages:
            logger.info("Reached max_pages=%s. Stopping crawl.", max_pages)
            break

        if max_depth is not None and current_level > max_depth:
            logger.debug("Skipping page beyond max depth: %s", current_relative)
            continue

        visited_pages.add(current_relative)
        parsed_page_count += 1

        if progress_every > 0 and parsed_page_count % progress_every == 0:
            logger.info(
                "Progress | parsed_pages=%s | queue_remaining=%s | current_level=%s | current_page=%s",
                parsed_page_count,
                len(queue),
                current_level,
                current_relative,
            )

        page_title, references = parse_html_page(current_page)

        outbound_html_link_count = 0
        external_link_count = 0
        document_link_count = 0
        asset_reference_count = 0
        missing_local_reference_count = 0
        anchor_reference_count = 0

        first_from, first_text, first_target = first_discovery.get(
            current_relative,
            (discovered_from, discovered_via_text, discovered_via_target),
        )

        page_record = HierarchyPageRecord(
            root_path=str(root_dir),
            seed_page=seed_relative,
            page_relative_path=current_relative,
            page_file_name=current_page.name,
            page_title=page_title,
            page_parent_folder="."
            if current_page.parent == root_dir
            else make_relative(current_page.parent, root_dir),
            level=current_level,
            first_discovered_from=first_from,
            first_discovered_via_text=first_text,
            first_discovered_via_target=first_target,
            inbound_html_link_count=0,
            outbound_html_link_count=0,
            external_link_count=0,
            document_link_count=0,
            asset_reference_count=0,
            missing_local_reference_count=0,
            anchor_reference_count=0,
            guessed_page_role="",
            crawl_status="parsed",
            notes="",
        )

        for ref in references:
            tag_name = ref["tag_name"]
            attribute_name = ref["attribute_name"]
            raw_target = ref["raw_target"]
            link_text = " ".join((ref.get("link_text") or "").split()).strip()

            normalized_target = normalize_reference(raw_target)
            is_external = is_external_target(normalized_target)
            anchor_only = is_anchor_only(normalized_target)
            data_uri = is_data_uri(normalized_target)

            target_relative_path = ""
            target_exists = False
            local_extension = ""
            local_category = ""
            target_level = -1
            target_is_crawled_html = False

            if is_external:
                external_link_count += 1
                target_kind = "external"

            elif data_uri:
                target_kind = "data_uri"

            elif anchor_only:
                anchor_reference_count += 1
                target_kind = "anchor"

            else:
                resolved_path = resolve_local_target(current_page, normalized_target)

                if is_within_root(resolved_path, root_dir):
                    try:
                        target_relative_path = make_relative(resolved_path, root_dir)
                    except Exception:
                        target_relative_path = ""
                else:
                    target_relative_path = ""

                target_exists = resolved_path.exists() and resolved_path.is_file()
                local_extension = resolved_path.suffix.lower()
                mime_type = mimetypes.guess_type(str(resolved_path))[0] or ""
                local_category = guess_category(local_extension, mime_type)

                target_kind = classify_target_kind(
                    tag_name=tag_name,
                    attribute_name=attribute_name,
                    normalized_target=normalized_target,
                    local_extension=local_extension,
                    local_category=local_category,
                )

                if target_kind == "html":
                    outbound_html_link_count += 1
                    if target_relative_path:
                        inbound_html_counter[target_relative_path] += 1

                    target_is_crawled_html = (
                        target_exists and target_relative_path not in visited_pages
                    )
                    target_level = current_level + 1 if target_exists else -1

                    if target_exists and target_relative_path:
                        if target_relative_path not in first_discovery:
                            first_discovery[target_relative_path] = (
                                current_relative,
                                link_text,
                                normalized_target,
                            )

                        if max_depth is None or (current_level + 1) <= max_depth:
                            queue.append(
                                (
                                    resolved_path,
                                    current_level + 1,
                                    current_relative,
                                    link_text,
                                    normalized_target,
                                )
                            )

                elif target_kind == "document":
                    document_link_count += 1

                elif target_kind in {
                    "image",
                    "script",
                    "stylesheet",
                    "xml",
                    "link_resource",
                    "other",
                    "href",
                }:
                    asset_reference_count += 1

                if not target_exists:
                    missing_local_reference_count += 1
                    missing_records.append(
                        MissingReferenceRecord(
                            root_path=str(root_dir),
                            seed_page=seed_relative,
                            source_page=current_relative,
                            source_title=page_title,
                            source_level=current_level,
                            target_raw=raw_target,
                            target_normalized=normalized_target,
                            target_kind=target_kind,
                            expected_relative_path=target_relative_path,
                            tag_name=tag_name,
                            attribute_name=attribute_name,
                            link_text=link_text,
                            notes="",
                        )
                    )

            edge_records.append(
                HierarchyEdgeRecord(
                    root_path=str(root_dir),
                    seed_page=seed_relative,
                    source_page=current_relative,
                    source_title=page_title,
                    source_level=current_level,
                    target_raw=raw_target,
                    target_normalized=normalized_target,
                    target_kind=target_kind,
                    target_relative_path=target_relative_path,
                    target_exists=target_exists,
                    target_is_crawled_html=target_is_crawled_html,
                    target_level=target_level,
                    is_external=is_external,
                    is_anchor_only=anchor_only,
                    link_text=link_text,
                    tag_name=tag_name,
                    attribute_name=attribute_name,
                    notes="",
                )
            )

        page_record.outbound_html_link_count = outbound_html_link_count
        page_record.external_link_count = external_link_count
        page_record.document_link_count = document_link_count
        page_record.asset_reference_count = asset_reference_count
        page_record.missing_local_reference_count = missing_local_reference_count
        page_record.anchor_reference_count = anchor_reference_count

        page_records_by_path[current_relative] = page_record

    for page_relative, page_record in page_records_by_path.items():
        inbound_count = inbound_html_counter.get(page_relative, 0)
        page_record.inbound_html_link_count = inbound_count
        page_record.guessed_page_role = guess_page_role(
            page_title=page_record.page_title,
            page_file_name=page_record.page_file_name,
            outbound_html_link_count=page_record.outbound_html_link_count,
            inbound_html_link_count=inbound_count,
        )

    page_records = sorted(
        page_records_by_path.values(),
        key=lambda r: (r.level, r.page_relative_path.lower()),
    )

    level_counts = Counter(rec.level for rec in page_records)
    role_counts = Counter(rec.guessed_page_role for rec in page_records)
    target_kind_counts = Counter(edge.target_kind for edge in edge_records)

    summary = {
        "generated_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "package_dir": str(PACKAGE_DIR),
        "root_path": str(root_dir),
        "seed_page": seed_relative,
        "total_pages_crawled": len(page_records),
        "total_edges": len(edge_records),
        "total_missing_local_references": len(missing_records),
        "max_depth_reached": max(level_counts.keys()) if level_counts else 0,
        "pages_by_level": dict(sorted(level_counts.items(), key=lambda x: x[0])),
        "pages_by_role": dict(sorted(role_counts.items(), key=lambda x: (-x[1], x[0]))),
        "edges_by_target_kind": dict(
            sorted(target_kind_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
    }

    logger.info(
        "Crawl complete | pages=%s | edges=%s | missing_local_refs=%s | max_depth=%s",
        summary["total_pages_crawled"],
        summary["total_edges"],
        summary["total_missing_local_references"],
        summary["max_depth_reached"],
    )

    return page_records, edge_records, missing_records, summary


# ============================================================
# Starter map writers
# ============================================================
def write_page_migration_starter(
    page_records: List[HierarchyPageRecord],
    output_path: Path,
) -> None:
    """
    Write starter CSV for page-level migration planning.
    """
    logger.info("Writing page migration starter CSV: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "page_relative_path",
        "page_title",
        "level",
        "guessed_page_role",
        "first_discovered_from",
        "outbound_html_link_count",
        "document_link_count",
        "asset_reference_count",
        "missing_local_reference_count",
        "migration_group",
        "target_route",
        "target_template",
        "migration_action",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for page in page_records:
            writer.writerow(
                {
                    "page_relative_path": page.page_relative_path,
                    "page_title": page.page_title,
                    "level": page.level,
                    "guessed_page_role": page.guessed_page_role,
                    "first_discovered_from": page.first_discovered_from,
                    "outbound_html_link_count": page.outbound_html_link_count,
                    "document_link_count": page.document_link_count,
                    "asset_reference_count": page.asset_reference_count,
                    "missing_local_reference_count": page.missing_local_reference_count,
                    "migration_group": "",
                    "target_route": "",
                    "target_template": "",
                    "migration_action": "",
                    "notes": "",
                }
            )


def write_edge_migration_starter(
    edge_records: List[HierarchyEdgeRecord],
    output_path: Path,
) -> None:
    """
    Write starter CSV for relationship-level migration planning.
    """
    logger.info("Writing edge migration starter CSV: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "source_page",
        "source_level",
        "target_raw",
        "target_normalized",
        "target_kind",
        "target_relative_path",
        "target_exists",
        "target_level",
        "link_text",
        "target_component",
        "target_model",
        "migration_action",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for edge in edge_records:
            writer.writerow(
                {
                    "source_page": edge.source_page,
                    "source_level": edge.source_level,
                    "target_raw": edge.target_raw,
                    "target_normalized": edge.target_normalized,
                    "target_kind": edge.target_kind,
                    "target_relative_path": edge.target_relative_path,
                    "target_exists": edge.target_exists,
                    "target_level": edge.target_level,
                    "link_text": edge.link_text,
                    "target_component": "",
                    "target_model": "",
                    "migration_action": "",
                    "notes": "",
                }
            )


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only site hierarchy from a seed HTML page using "
            "breadth-first crawl of local HTML links."
        )
    )
    parser.add_argument(
        "seed",
        type=str,
        help="Seed HTML page path to start crawling from.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder that bounds the crawl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PACKAGE_DIR),
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum depth to crawl. Example: 2",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of HTML pages to parse.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N parsed pages. Use 0 to disable periodic progress logs.",
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

    seed_page = Path(args.seed).resolve()
    root_dir = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Package dir: %s", PACKAGE_DIR)
    logger.info("Seed page: %s", seed_page)
    logger.info("Root directory: %s", root_dir)
    logger.info("Output directory: %s", output_dir)

    page_records, edge_records, missing_records, summary = build_site_hierarchy(
        seed_page=seed_page,
        root_dir=root_dir,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        progress_every=args.progress_every,
    )

    pages_csv = output_dir / "site_hierarchy_pages.csv"
    edges_csv = output_dir / "site_hierarchy_edges.csv"
    missing_csv = output_dir / "missing_local_references.csv"
    summary_json = output_dir / "site_hierarchy_summary.json"
    hierarchy_json = output_dir / "site_hierarchy.json"
    page_migration_csv = output_dir / "site_hierarchy_page_migration_starter.csv"
    edge_migration_csv = output_dir / "site_hierarchy_edge_migration_starter.csv"

    write_csv_from_dataclasses(
        page_records,
        pages_csv,
        fallback_headers=list(HierarchyPageRecord.__dataclass_fields__.keys()),
    )
    write_csv_from_dataclasses(
        edge_records,
        edges_csv,
        fallback_headers=list(HierarchyEdgeRecord.__dataclass_fields__.keys()),
    )
    write_csv_from_dataclasses(
        missing_records,
        missing_csv,
        fallback_headers=list(MissingReferenceRecord.__dataclass_fields__.keys()),
    )
    write_json(summary, summary_json)
    write_json(
        {
            "summary": summary,
            "pages": [asdict(rec) for rec in page_records],
            "edges": [asdict(rec) for rec in edge_records],
            "missing_local_references": [asdict(rec) for rec in missing_records],
        },
        hierarchy_json,
    )
    write_page_migration_starter(page_records, page_migration_csv)
    write_edge_migration_starter(edge_records, edge_migration_csv)

    print("\nSite hierarchy build complete.")
    print(f"Seed page: {seed_page}")
    print(f"Root: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pages crawled: {summary['total_pages_crawled']}")
    print(f"Edges recorded: {summary['total_edges']}")
    print(f"Missing local references: {summary['total_missing_local_references']}")
    print(f"Max depth reached: {summary['max_depth_reached']}")
    print(f"Pages by level: {summary['pages_by_level']}")
    print(f"Pages CSV: {pages_csv}")
    print(f"Edges CSV: {edges_csv}")
    print(f"Missing refs CSV: {missing_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Hierarchy JSON: {hierarchy_json}")
    print(f"Page migration starter: {page_migration_csv}")
    print(f"Edge migration starter: {edge_migration_csv}")


if __name__ == "__main__":
    main()