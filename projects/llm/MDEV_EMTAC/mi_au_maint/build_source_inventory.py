from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import mimetypes
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional
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
logger = logging.getLogger("source_inventory")


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
class FileRecord:
    """
    A single inventory record for one file discovered during the walk.
    """
    root_path: str
    relative_path: str
    parent_folder: str
    file_name: str
    stem: str
    extension: str
    suffixes: str
    size_bytes: int
    size_kb: float
    mime_type: str
    guessed_category: str
    created_time: str
    modified_time: str
    accessed_time: str
    is_hidden: bool
    depth: int
    sha256: str = ""


@dataclass
class HtmlPageRecord:
    """
    A single HTML page summary record.
    """
    root_path: str
    page_relative_path: str
    page_file_name: str
    page_title: str
    page_parent_folder: str
    page_depth: int
    internal_link_count: int
    external_link_count: int
    local_asset_reference_count: int
    image_reference_count: int
    script_reference_count: int
    stylesheet_reference_count: int
    document_reference_count: int
    anchor_reference_count: int
    missing_local_reference_count: int
    inbound_link_count: int
    guessed_page_role: str
    notes: str = ""


@dataclass
class HtmlLinkRecord:
    """
    A single extracted HTML relationship/reference row.
    """
    root_path: str
    source_page: str
    source_page_title: str
    tag_name: str
    attribute_name: str
    raw_target: str
    normalized_target: str
    target_kind: str
    is_external: bool
    is_data_uri: bool
    is_anchor_only: bool
    local_target_relative_path: str
    local_target_exists: bool
    local_target_extension: str
    local_target_category: str
    link_text: str
    notes: str = ""


# ============================================================
# Utility helpers
# ============================================================
def iso_from_timestamp(ts: float) -> str:
    """
    Convert a filesystem timestamp to ISO-8601 string.
    """
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return ""


def safe_stat(path: Path) -> Optional[os.stat_result]:
    """
    Safely stat a path. Returns None on failure.
    """
    try:
        return path.stat()
    except Exception as exc:
        logger.warning("Could not stat path: %s | reason=%s", path, exc)
        return None


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 hash for a file in a memory-safe way.
    """
    sha = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as exc:
        logger.warning("Could not hash file: %s | reason=%s", file_path, exc)
        return ""


def is_hidden_path(path: Path) -> bool:
    """
    Determine whether a file should be considered hidden.
    """
    try:
        return any(part.startswith(".") for part in path.parts)
    except Exception:
        return False


def guess_category(extension: str, mime_type: str) -> str:
    """
    Assign a rough category to help with later migration mapping.
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


def normalize_suffixes(path: Path) -> str:
    """
    Return all suffixes as a single semicolon-separated string.
    Example: .tar.gz -> '.tar;.gz'
    """
    if not path.suffixes:
        return ""
    return ";".join(path.suffixes)


def is_html_extension(extension: str) -> bool:
    return extension.lower() in {".htm", ".html"}


def normalize_reference(raw_value: str) -> str:
    """
    Normalize an HTML reference target conservatively.
    This is still read-only and intended only for analysis.
    """
    value = (raw_value or "").strip()
    if not value:
        return ""

    value = value.replace("\\", "/")
    value = unquote(value)
    return value


def is_external_target(target: str) -> bool:
    """
    Check whether a target is external (http, https, mailto, tel, ftp, etc.).
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
    Remove fragment and query from a reference.
    """
    if not target:
        return ""

    parsed = urlparse(target)
    path = parsed.path or ""
    return path


def resolve_local_target(source_page: Path, normalized_target: str) -> Path:
    """
    Resolve a local reference relative to the source page.
    """
    cleaned = strip_fragment_and_query(normalized_target)
    return (source_page.parent / cleaned).resolve()


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


def classify_target_kind(
    tag_name: str,
    attribute_name: str,
    normalized_target: str,
    local_extension: str,
    local_category: str,
) -> str:
    """
    Classify a discovered reference for migration planning.
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
    internal_link_count: int,
    inbound_link_count: int,
) -> str:
    """
    A very light heuristic for later migration planning.
    """
    title = (page_title or "").lower()
    name = (page_file_name or "").lower()

    if "main" in title or "launch" in title or "menu" in title:
        return "launcher"

    if internal_link_count >= 4 and inbound_link_count == 0:
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
    Lightweight HTML parser that extracts title, href/src references,
    and simple anchor text without requiring external dependencies.
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
                        ref = {
                            "tag_name": tag,
                            "attribute_name": "src",
                            "raw_target": candidate,
                            "link_text": "",
                        }
                        self.references.append(ref)
                    continue

                ref = {
                    "tag_name": tag,
                    "attribute_name": attr_name,
                    "raw_target": raw_value,
                    "link_text": "",
                }
                self.references.append(ref)

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
# Core inventory builder
# ============================================================
def build_inventory(
    root_dir: Path,
    include_hash: bool = False,
    include_hidden: bool = True,
) -> List[FileRecord]:
    """
    Recursively walk root_dir and collect metadata for all files.
    This function does not modify any files or folders.
    """
    logger.info("Starting inventory build for root: %s", root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    records: List[FileRecord] = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        hidden = is_hidden_path(path)
        if hidden and not include_hidden:
            logger.debug("Skipping hidden file: %s", path)
            continue

        stat_info = safe_stat(path)
        if stat_info is None:
            continue

        relative_path = path.relative_to(root_dir).as_posix()
        parent_folder = (
            path.parent.relative_to(root_dir).as_posix()
            if path.parent != root_dir
            else "."
        )
        extension = path.suffix.lower()
        mime_type = mimetypes.guess_type(str(path))[0] or ""
        category = guess_category(extension, mime_type)
        depth = len(path.relative_to(root_dir).parts) - 1

        sha256 = compute_sha256(path) if include_hash else ""

        record = FileRecord(
            root_path=str(root_dir.resolve()),
            relative_path=relative_path,
            parent_folder=parent_folder,
            file_name=path.name,
            stem=path.stem,
            extension=extension,
            suffixes=normalize_suffixes(path),
            size_bytes=stat_info.st_size,
            size_kb=round(stat_info.st_size / 1024, 3),
            mime_type=mime_type,
            guessed_category=category,
            created_time=iso_from_timestamp(stat_info.st_ctime),
            modified_time=iso_from_timestamp(stat_info.st_mtime),
            accessed_time=iso_from_timestamp(stat_info.st_atime),
            is_hidden=hidden,
            depth=depth,
            sha256=sha256,
        )
        records.append(record)

    logger.info("Inventory build complete. Files found: %s", len(records))
    return records


# ============================================================
# HTML map builders
# ============================================================
def build_html_maps(
    root_dir: Path,
    file_records: List[FileRecord],
) -> tuple[List[HtmlPageRecord], List[HtmlLinkRecord]]:
    """
    Build page-level and link-level maps for all HTML files found in the inventory.
    """
    logger.info("Building HTML page/link maps.")

    html_files = [rec for rec in file_records if rec.guessed_category == "html"]
    page_records: List[HtmlPageRecord] = []
    link_records: List[HtmlLinkRecord] = []

    inbound_counter: Counter[str] = Counter()

    for file_rec in html_files:
        page_path = root_dir / Path(file_rec.relative_path)

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

        page_link_rows: List[HtmlLinkRecord] = []

        internal_link_count = 0
        external_link_count = 0
        local_asset_reference_count = 0
        image_reference_count = 0
        script_reference_count = 0
        stylesheet_reference_count = 0
        document_reference_count = 0
        anchor_reference_count = 0
        missing_local_reference_count = 0

        for ref in parser.references:
            tag_name = ref["tag_name"]
            attribute_name = ref["attribute_name"]
            raw_target = ref["raw_target"]
            link_text = " ".join((ref.get("link_text") or "").split()).strip()

            normalized_target = normalize_reference(raw_target)
            external = is_external_target(normalized_target)
            data_uri = is_data_uri(normalized_target)
            anchor_only = is_anchor_only(normalized_target)

            local_target_relative_path = ""
            local_target_exists = False
            local_target_extension = ""
            local_target_category = ""

            if not external and not data_uri and not anchor_only:
                local_target_path = resolve_local_target(page_path, normalized_target)

                try:
                    local_target_relative_path = (
                        local_target_path.relative_to(root_dir.resolve()).as_posix()
                    )
                except Exception:
                    local_target_relative_path = ""

                local_target_exists = (
                    local_target_path.exists() and local_target_path.is_file()
                )
                local_target_extension = local_target_path.suffix.lower()
                mime_type = mimetypes.guess_type(str(local_target_path))[0] or ""
                local_target_category = guess_category(
                    local_target_extension,
                    mime_type,
                )

            target_kind = classify_target_kind(
                tag_name=tag_name,
                attribute_name=attribute_name,
                normalized_target=normalized_target,
                local_extension=local_target_extension,
                local_category=local_target_category,
            )

            if external:
                external_link_count += 1
            elif anchor_only:
                anchor_reference_count += 1
            elif data_uri:
                pass
            else:
                if target_kind == "html":
                    internal_link_count += 1
                    if local_target_relative_path:
                        inbound_counter[local_target_relative_path] += 1
                else:
                    local_asset_reference_count += 1

                if not local_target_exists:
                    missing_local_reference_count += 1

            if target_kind == "image":
                image_reference_count += 1
            elif target_kind == "script":
                script_reference_count += 1
            elif target_kind == "stylesheet":
                stylesheet_reference_count += 1
            elif target_kind == "document":
                document_reference_count += 1

            page_link_rows.append(
                HtmlLinkRecord(
                    root_path=str(root_dir.resolve()),
                    source_page=file_rec.relative_path,
                    source_page_title=page_title,
                    tag_name=tag_name,
                    attribute_name=attribute_name,
                    raw_target=raw_target,
                    normalized_target=normalized_target,
                    target_kind=target_kind,
                    is_external=external,
                    is_data_uri=data_uri,
                    is_anchor_only=anchor_only,
                    local_target_relative_path=local_target_relative_path,
                    local_target_exists=local_target_exists,
                    local_target_extension=local_target_extension,
                    local_target_category=local_target_category,
                    link_text=link_text,
                    notes="",
                )
            )

        guessed_role = guess_page_role(
            page_title=page_title,
            page_file_name=file_rec.file_name,
            internal_link_count=internal_link_count,
            inbound_link_count=0,
        )

        page_records.append(
            HtmlPageRecord(
                root_path=str(root_dir.resolve()),
                page_relative_path=file_rec.relative_path,
                page_file_name=file_rec.file_name,
                page_title=page_title,
                page_parent_folder=file_rec.parent_folder,
                page_depth=file_rec.depth,
                internal_link_count=internal_link_count,
                external_link_count=external_link_count,
                local_asset_reference_count=local_asset_reference_count,
                image_reference_count=image_reference_count,
                script_reference_count=script_reference_count,
                stylesheet_reference_count=stylesheet_reference_count,
                document_reference_count=document_reference_count,
                anchor_reference_count=anchor_reference_count,
                missing_local_reference_count=missing_local_reference_count,
                inbound_link_count=0,
                guessed_page_role=guessed_role,
                notes="",
            )
        )

        link_records.extend(page_link_rows)

    page_index: Dict[str, int] = {
        rec.page_relative_path: idx for idx, rec in enumerate(page_records)
    }

    for page_path, count in inbound_counter.items():
        idx = page_index.get(page_path)
        if idx is not None:
            page_records[idx].inbound_link_count = count

    for idx, page_rec in enumerate(page_records):
        page_records[idx].guessed_page_role = guess_page_role(
            page_title=page_rec.page_title,
            page_file_name=page_rec.page_file_name,
            internal_link_count=page_rec.internal_link_count,
            inbound_link_count=page_rec.inbound_link_count,
        )

    logger.info(
        "HTML map build complete. Pages=%s | Relationships=%s",
        len(page_records),
        len(link_records),
    )

    return page_records, link_records


# ============================================================
# Summary builders
# ============================================================
def build_summary(records: List[FileRecord], root_dir: Path) -> Dict:
    """
    Build summary statistics for the inventory run.
    """
    extension_counts = Counter()
    category_counts = Counter()
    folder_counts = Counter()

    total_size_bytes = 0

    for rec in records:
        extension_counts[rec.extension or "[no_extension]"] += 1
        category_counts[rec.guessed_category] += 1
        folder_counts[rec.parent_folder] += 1
        total_size_bytes += rec.size_bytes

    largest_files = sorted(records, key=lambda x: x.size_bytes, reverse=True)[:25]

    summary = {
        "project_root": str(PROJECT_ROOT),
        "package_dir": str(PACKAGE_DIR),
        "root_path": str(root_dir.resolve()),
        "generated_at": datetime.now().isoformat(),
        "total_files": len(records),
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 3),
        "counts_by_extension": dict(
            sorted(extension_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "counts_by_category": dict(
            sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "counts_by_folder": dict(
            sorted(folder_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "largest_files": [
            {
                "relative_path": rec.relative_path,
                "size_bytes": rec.size_bytes,
                "guessed_category": rec.guessed_category,
            }
            for rec in largest_files
        ],
    }

    return summary


def build_html_summary(
    page_records: List[HtmlPageRecord],
    link_records: List[HtmlLinkRecord],
    root_dir: Path,
) -> Dict:
    """
    Build summary statistics for HTML map outputs.
    """
    role_counts = Counter()
    target_kind_counts = Counter()
    external_count = 0
    missing_local_count = 0

    for page in page_records:
        role_counts[page.guessed_page_role] += 1

    for link in link_records:
        target_kind_counts[link.target_kind] += 1
        if link.is_external:
            external_count += 1
        if (
            not link.is_external
            and not link.is_data_uri
            and not link.is_anchor_only
            and not link.local_target_exists
        ):
            missing_local_count += 1

    return {
        "project_root": str(PROJECT_ROOT),
        "package_dir": str(PACKAGE_DIR),
        "root_path": str(root_dir.resolve()),
        "generated_at": datetime.now().isoformat(),
        "total_html_pages": len(page_records),
        "total_html_relationships": len(link_records),
        "external_reference_count": external_count,
        "missing_local_reference_count": missing_local_count,
        "counts_by_page_role": dict(
            sorted(role_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "counts_by_target_kind": dict(
            sorted(target_kind_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
    }


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


def write_json(records: List[FileRecord], summary: Dict, output_path: Path) -> None:
    """
    Write inventory records and summary to JSON.
    """
    logger.info("Writing JSON inventory: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "files": [asdict(rec) for rec in records],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_html_json(
    page_records: List[HtmlPageRecord],
    link_records: List[HtmlLinkRecord],
    summary: Dict,
    output_path: Path,
) -> None:
    """
    Write HTML page/link map payload to JSON.
    """
    logger.info("Writing HTML JSON map: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "pages": [asdict(rec) for rec in page_records],
        "relationships": [asdict(rec) for rec in link_records],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary_json(summary: Dict, output_path: Path) -> None:
    """
    Write summary-only JSON.
    """
    logger.info("Writing summary JSON: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_folder_tree(root_dir: Path, output_path: Path) -> None:
    """
    Write a text tree of folders and files.
    Read-only inspection only.
    """
    logger.info("Writing folder tree: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [str(root_dir.resolve())]

    def walk(current: Path, prefix: str = "") -> None:
        try:
            entries = sorted(
                current.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except Exception as exc:
            lines.append(f"{prefix}[ERROR] {current.name} :: {exc}")
            return

        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():
                extension = "    " if idx == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    walk(root_dir)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Migration map starter
# ============================================================
def write_migration_map_starter(records: List[FileRecord], output_path: Path) -> None:
    """
    Write a starter CSV for later migration planning.
    """
    logger.info("Writing migration map starter CSV: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "source_relative_path",
        "file_name",
        "extension",
        "guessed_category",
        "size_bytes",
        "parent_folder",
        "target_name",
        "target_path",
        "target_type",
        "migration_action",
        "logical_group",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for rec in records:
            writer.writerow(
                {
                    "source_relative_path": rec.relative_path,
                    "file_name": rec.file_name,
                    "extension": rec.extension,
                    "guessed_category": rec.guessed_category,
                    "size_bytes": rec.size_bytes,
                    "parent_folder": rec.parent_folder,
                    "target_name": "",
                    "target_path": "",
                    "target_type": "",
                    "migration_action": "",
                    "logical_group": "",
                    "notes": "",
                }
            )


def write_html_migration_map_starter(
    page_records: List[HtmlPageRecord],
    link_records: List[HtmlLinkRecord],
    output_dir: Path,
) -> None:
    """
    Write starter CSVs specifically for HTML/page migration planning.
    """
    page_map_path = output_dir / "html_page_migration_map_starter.csv"
    relationship_map_path = output_dir / "html_relationship_migration_map_starter.csv"

    logger.info("Writing HTML page migration starter CSV: %s", page_map_path)
    page_headers = [
        "page_relative_path",
        "page_file_name",
        "page_title",
        "guessed_page_role",
        "internal_link_count",
        "external_link_count",
        "missing_local_reference_count",
        "target_route",
        "target_template",
        "migration_action",
        "logical_group",
        "notes",
    ]

    with page_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=page_headers)
        writer.writeheader()
        for page in page_records:
            writer.writerow(
                {
                    "page_relative_path": page.page_relative_path,
                    "page_file_name": page.page_file_name,
                    "page_title": page.page_title,
                    "guessed_page_role": page.guessed_page_role,
                    "internal_link_count": page.internal_link_count,
                    "external_link_count": page.external_link_count,
                    "missing_local_reference_count": page.missing_local_reference_count,
                    "target_route": "",
                    "target_template": "",
                    "migration_action": "",
                    "logical_group": "",
                    "notes": "",
                }
            )

    logger.info(
        "Writing HTML relationship migration starter CSV: %s",
        relationship_map_path,
    )
    relationship_headers = [
        "source_page",
        "source_page_title",
        "tag_name",
        "attribute_name",
        "raw_target",
        "normalized_target",
        "target_kind",
        "is_external",
        "local_target_relative_path",
        "local_target_exists",
        "target_component",
        "target_model",
        "migration_action",
        "notes",
    ]

    with relationship_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=relationship_headers)
        writer.writeheader()
        for link in link_records:
            writer.writerow(
                {
                    "source_page": link.source_page,
                    "source_page_title": link.source_page_title,
                    "tag_name": link.tag_name,
                    "attribute_name": link.attribute_name,
                    "raw_target": link.raw_target,
                    "normalized_target": link.normalized_target,
                    "target_kind": link.target_kind,
                    "is_external": link.is_external,
                    "local_target_relative_path": link.local_target_relative_path,
                    "local_target_exists": link.local_target_exists,
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
            "Recursively inventory a source folder for migration planning without "
            "modifying any files, and optionally build HTML page/link maps."
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root folder to walk recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PACKAGE_DIR),
        help="Folder where inventory files will be written.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help=(
            "Compute SHA256 for each file. Slower, but useful for "
            "deduplication and verification."
        ),
    )
    parser.add_argument(
        "--exclude-hidden",
        action="store_true",
        help="Exclude hidden/dot-prefixed files from the inventory.",
    )
    parser.add_argument(
        "--skip-html-map",
        action="store_true",
        help="Skip HTML page/link extraction.",
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

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Package dir: %s", PACKAGE_DIR)
    logger.info("Root directory: %s", root_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Hash enabled: %s", args.hash)
    logger.info("Exclude hidden: %s", args.exclude_hidden)
    logger.info("Skip HTML map: %s", args.skip_html_map)

    records = build_inventory(
        root_dir=root_dir,
        include_hash=args.hash,
        include_hidden=not args.exclude_hidden,
    )

    summary = build_summary(records, root_dir)

    csv_path = output_dir / "source_inventory.csv"
    json_path = output_dir / "source_inventory.json"
    summary_path = output_dir / "source_inventory_summary.json"
    tree_path = output_dir / "source_tree.txt"
    migration_map_path = output_dir / "migration_map_starter.csv"

    write_csv_from_dataclasses(
        records,
        csv_path,
        fallback_headers=list(FileRecord.__dataclass_fields__.keys()),
    )
    write_json(records, summary, json_path)
    write_summary_json(summary, summary_path)
    write_folder_tree(root_dir, tree_path)
    write_migration_map_starter(records, migration_map_path)

    html_page_count = 0
    html_relationship_count = 0

    if not args.skip_html_map:
        page_records, link_records = build_html_maps(root_dir, records)
        html_summary = build_html_summary(page_records, link_records, root_dir)

        html_pages_csv_path = output_dir / "html_pages.csv"
        html_links_csv_path = output_dir / "html_relationships.csv"
        html_json_path = output_dir / "html_map.json"
        html_summary_path = output_dir / "html_map_summary.json"

        write_csv_from_dataclasses(
            page_records,
            html_pages_csv_path,
            fallback_headers=list(HtmlPageRecord.__dataclass_fields__.keys()),
        )
        write_csv_from_dataclasses(
            link_records,
            html_links_csv_path,
            fallback_headers=list(HtmlLinkRecord.__dataclass_fields__.keys()),
        )
        write_html_json(page_records, link_records, html_summary, html_json_path)
        write_summary_json(html_summary, html_summary_path)
        write_html_migration_map_starter(page_records, link_records, output_dir)

        html_page_count = len(page_records)
        html_relationship_count = len(link_records)

    logger.info("Inventory complete.")
    logger.info("CSV: %s", csv_path)
    logger.info("JSON: %s", json_path)
    logger.info("Summary JSON: %s", summary_path)
    logger.info("Tree TXT: %s", tree_path)
    logger.info("Migration starter CSV: %s", migration_map_path)

    print("\nInventory complete.")
    print(f"Root scanned: {root_dir}")
    print(f"Files found: {summary['total_files']}")
    print(f"Total size (MB): {summary['total_size_mb']}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")
    print(f"Tree: {tree_path}")
    print(f"Migration map starter: {migration_map_path}")

    if not args.skip_html_map:
        print(f"HTML pages: {html_page_count}")
        print(f"HTML relationships: {html_relationship_count}")
        print(f"HTML pages CSV: {output_dir / 'html_pages.csv'}")
        print(f"HTML relationships CSV: {output_dir / 'html_relationships.csv'}")
        print(f"HTML JSON map: {output_dir / 'html_map.json'}")
        print(f"HTML summary: {output_dir / 'html_map_summary.json'}")
        print(
            "HTML page migration starter: "
            f"{output_dir / 'html_page_migration_map_starter.csv'}"
        )
        print(
            "HTML relationship migration starter: "
            f"{output_dir / 'html_relationship_migration_map_starter.csv'}"
        )


if __name__ == "__main__":
    main()