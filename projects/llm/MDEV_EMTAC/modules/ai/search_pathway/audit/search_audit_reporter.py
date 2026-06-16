# ============================================================
# File:
#   modules/ai/search_pathway/audit/search_audit_reporter.py
#
# Purpose:
#   Generate reports from the PostgreSQL audit schema for AI/search pathway
#   auditing.
#
# Output behavior:
#   Each report run creates its own folder inside:
#
#       logs/search_audit/reports/
#
#   Example:
#
#       logs/search_audit/reports/request_a10c3a51_all_pathways_20260508_051200/
#           full_report.json
#           run_summary.csv
#           count_summary.csv
#           item_details.csv
#           validation_details.csv
#           report.md
#
# Example commands:
#
#   cd E:\emtac\projects\llm\MDEV_EMTAC
#
#   python -m modules.ai.search_pathway.audit.search_audit_reporter ^
#       --request-id 3931161f ^
#       --format all
#
#   python -m modules.ai.search_pathway.audit.search_audit_reporter ^
#       --user-id anonymous ^
#       --start "2026-05-08 03:00" ^
#       --end "2026-05-08 04:00" ^
#       --format csv
#
#   python -m modules.ai.search_pathway.audit.search_audit_reporter ^
#       --pathway payload_projection ^
#       --start "2026-05-08" ^
#       --end "2026-05-08" ^
#       --include-evidence ^
#       --format json
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import text

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


# ============================================================
# Project path setup
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# File path:
#   modules/ai/search_pathway/audit/search_audit_reporter.py
#
# parents[0] = audit
# parents[1] = search_pathway
# parents[2] = ai
# parents[3] = modules
# parents[4] = project root
PROJECT_ROOT = CURRENT_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from modules.configuration.config_env import get_db_config

try:
    from modules.configuration.config import LOGS_DIR
except Exception:
    LOGS_DIR = str(PROJECT_ROOT / "logs")

try:
    from modules.ai.search_pathway.audit.search_audit_logger import (
        get_search_audit_logger,
    )

    logger = get_search_audit_logger()
except Exception:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("search_audit_reporter")


# ============================================================
# Constants
# ============================================================

DEFAULT_TIMEZONE_NAME = "America/Chicago"

DEFAULT_REPORT_DIR = (
    Path(LOGS_DIR)
    / "search_audit"
    / "reports"
)

VALID_FORMATS = {"csv", "json", "md", "all"}


# ============================================================
# SQL
# ============================================================

RUN_SUMMARY_SQL = """
SELECT
    r.id::text AS audit_run_id,
    r.request_id,
    r.qanda_id::text AS qanda_id,
    r.user_id,
    r.session_id::text AS session_id,
    r.pathway_name,
    r.pathway_version,
    r.search_mode,
    r.payload_status,
    r.validation_status,
    r.model_name,
    r.duration_ms,
    r.question,
    LEFT(COALESCE(r.final_answer, ''), 1000) AS final_answer_preview,
    r.started_at,
    r.completed_at
FROM audit.search_audit_run r
WHERE 1 = 1
{filters}
ORDER BY r.started_at DESC, r.pathway_name;
"""


COUNT_SUMMARY_SQL = """
SELECT
    r.id::text AS audit_run_id,
    r.request_id,
    r.pathway_name,
    r.payload_status,
    r.validation_status,
    i.item_type,
    i.source_table,
    COUNT(i.id) AS returned_count,
    COUNT(DISTINCT i.source_id) AS distinct_source_id_count,
    MIN(i.rank) AS min_rank,
    MAX(i.rank) AS max_rank,
    MIN(i.created_at) AS first_item_created_at,
    MAX(i.created_at) AS last_item_created_at
FROM audit.search_audit_run r
LEFT JOIN audit.search_audit_payload_item i
    ON i.audit_run_id = r.id
WHERE 1 = 1
{filters}
GROUP BY
    r.id,
    r.request_id,
    r.pathway_name,
    r.payload_status,
    r.validation_status,
    i.item_type,
    i.source_table
ORDER BY
    r.started_at DESC,
    r.pathway_name,
    returned_count DESC;
"""


ITEM_DETAIL_SQL = """
SELECT
    r.id::text AS audit_run_id,
    r.request_id,
    r.pathway_name,
    r.payload_status,
    r.validation_status AS run_validation_status,
    r.started_at AS run_started_at,

    i.id AS audit_item_id,
    i.item_type,
    i.source_table,
    i.source_id,
    i.title,
    i.label,
    i.file_path,
    i.url,
    i.rank,
    i.score,
    i.relationship_path,
    i.exists_in_db,
    i.exists_on_disk,
    i.validation_status AS item_validation_status,
    i.validation_message,
    i.created_at AS item_created_at
    {evidence_select}
FROM audit.search_audit_run r
JOIN audit.search_audit_payload_item i
    ON i.audit_run_id = r.id
WHERE 1 = 1
{filters}
ORDER BY
    r.started_at DESC,
    r.pathway_name,
    i.item_type,
    i.rank,
    i.source_id
{limit_clause};
"""


VALIDATION_DETAIL_SQL = """
SELECT
    r.id::text AS audit_run_id,
    r.request_id,
    r.pathway_name,
    r.payload_status,
    r.validation_status AS run_validation_status,
    r.started_at AS run_started_at,

    v.id AS validation_id,
    v.check_name,
    v.check_status,
    v.expected_count,
    v.actual_count,
    v.details,
    v.created_at AS validation_created_at
FROM audit.search_audit_run r
JOIN audit.search_audit_validation v
    ON v.audit_run_id = r.id
WHERE 1 = 1
{filters}
ORDER BY
    r.started_at DESC,
    r.pathway_name,
    v.created_at,
    v.check_name;
"""


# ============================================================
# Utility helpers
# ============================================================

def get_default_timezone():
    if ZoneInfo is None:
        return None

    return ZoneInfo(DEFAULT_TIMEZONE_NAME)


def parse_datetime_filter(value: str | None, *, is_end: bool = False) -> datetime | None:
    """
    Parse a CLI date/time value.

    Supported examples:
        2026-05-08
        2026-05-08 03:00
        2026-05-08T03:00
        2026-05-08T03:00:00-05:00

    If no timezone is provided, America/Chicago is assumed.

    If only a date is provided:
        start -> 00:00:00
        end   -> 23:59:59.999999
    """

    if not value:
        return None

    raw = value.strip()

    if not raw:
        return None

    tz = get_default_timezone()

    # Date only
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        parsed_date = date.fromisoformat(raw)

        parsed_time = time.max if is_end else time.min
        parsed = datetime.combine(parsed_date, parsed_time)

        if tz:
            return parsed.replace(tzinfo=tz)

        return parsed

    normalized = raw.replace(" ", "T", 1)

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid date/time value: {value!r}. "
            "Use formats like '2026-05-08', '2026-05-08 03:00', "
            "or '2026-05-08T03:00:00-05:00'."
        ) from exc

    if parsed.tzinfo is None and tz:
        parsed = parsed.replace(tzinfo=tz)

    return parsed


def json_default(value: Any) -> str | int | float | bool | None:
    """
    JSON serializer fallback.
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return str(value)


def rows_to_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    """
    Convert SQLAlchemy mapping rows to plain dicts.
    """

    return [dict(row) for row in rows]


def safe_filename_part(value: str | None, fallback: str) -> str:
    """
    Make a string safe for filenames/folder names.
    """

    if not value:
        value = fallback

    value = str(value).strip()

    if not value:
        value = fallback

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ensure_report_dir(out_dir: Path) -> Path:
    """
    Create a directory if it does not exist.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_report_folder_name(args: argparse.Namespace) -> str:
    """
    Build a dedicated folder name for one report run.

    Examples:
        request_a10c3a51_all_pathways_20260508_050500
        user_anonymous_payload_projection_20260508_050500
        search_audit_all_pathways_20260508_050500
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.request_id:
        target = f"request_{safe_filename_part(args.request_id, 'unknown')}"
    elif args.user_id:
        target = f"user_{safe_filename_part(args.user_id, 'unknown')}"
    else:
        target = "search_audit"

    pathway = safe_filename_part(args.pathway or "all_pathways", "all_pathways")

    return f"{target}_{pathway}_{timestamp}"


# ============================================================
# SQL filter builder
# ============================================================

def build_filters(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    filters: list[str] = []
    params: dict[str, Any] = {}

    if args.request_id:
        filters.append("AND r.request_id = :request_id")
        params["request_id"] = args.request_id.strip()

    if args.user_id:
        filters.append("AND r.user_id = :user_id")
        params["user_id"] = args.user_id.strip()

    if args.pathway:
        filters.append("AND r.pathway_name = :pathway_name")
        params["pathway_name"] = args.pathway.strip()

    if args.validation_status:
        filters.append("AND r.validation_status = :validation_status")
        params["validation_status"] = args.validation_status.strip()

    if args.payload_status:
        filters.append("AND r.payload_status = :payload_status")
        params["payload_status"] = args.payload_status.strip()

    if args.start_dt:
        filters.append("AND r.started_at >= :start_dt")
        params["start_dt"] = args.start_dt

    if args.end_dt:
        filters.append("AND r.started_at <= :end_dt")
        params["end_dt"] = args.end_dt

    return "\n".join(filters), params


# ============================================================
# Query runner
# ============================================================

class SearchAuditReporter:
    """
    Generates search audit reports from the audit schema.

    This class opens a read-only reporting session through the normal
    DatabaseConfig. It does not modify audit data.
    """

    def __init__(self) -> None:
        self.db_config = get_db_config()

    def run_report(self, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
        session = None

        try:
            session = self.db_config.get_main_session()

            filters_sql, params = build_filters(args)

            run_summary = self._fetch_run_summary(
                session=session,
                filters_sql=filters_sql,
                params=params,
            )

            count_summary = self._fetch_count_summary(
                session=session,
                filters_sql=filters_sql,
                params=params,
            )

            item_details = self._fetch_item_details(
                session=session,
                filters_sql=filters_sql,
                params=params,
                include_evidence=args.include_evidence,
                max_items=args.max_items,
            )

            validation_details = self._fetch_validation_details(
                session=session,
                filters_sql=filters_sql,
                params=params,
            )

            return {
                "run_summary": run_summary,
                "count_summary": count_summary,
                "item_details": item_details,
                "validation_details": validation_details,
            }

        finally:
            if session:
                session.close()

    @staticmethod
    def _fetch_run_summary(
        *,
        session,
        filters_sql: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query = RUN_SUMMARY_SQL.format(filters=filters_sql)
        rows = session.execute(text(query), params).mappings().all()
        return rows_to_dicts(rows)

    @staticmethod
    def _fetch_count_summary(
        *,
        session,
        filters_sql: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query = COUNT_SUMMARY_SQL.format(filters=filters_sql)
        rows = session.execute(text(query), params).mappings().all()
        return rows_to_dicts(rows)

    @staticmethod
    def _fetch_item_details(
        *,
        session,
        filters_sql: str,
        params: dict[str, Any],
        include_evidence: bool,
        max_items: int,
    ) -> list[dict[str, Any]]:
        evidence_select = ",\n    i.evidence" if include_evidence else ""

        query_params = dict(params)

        if max_items and max_items > 0:
            limit_clause = "\nLIMIT :max_items"
            query_params["max_items"] = max_items
        else:
            limit_clause = ""

        query = ITEM_DETAIL_SQL.format(
            filters=filters_sql,
            evidence_select=evidence_select,
            limit_clause=limit_clause,
        )

        rows = session.execute(text(query), query_params).mappings().all()
        return rows_to_dicts(rows)

    @staticmethod
    def _fetch_validation_details(
        *,
        session,
        filters_sql: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query = VALIDATION_DETAIL_SQL.format(filters=filters_sql)
        rows = session.execute(text(query), params).mappings().all()
        return rows_to_dicts(rows)


# ============================================================
# Writers
# ============================================================

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write rows to CSV.

    If rows is empty, writes an empty file with no header.
    """

    with path.open("w", newline="", encoding="utf-8") as file:
        if not rows:
            file.write("")
            return

        fieldnames = list(rows[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            cleaned = {
                key: json.dumps(value, default=json_default)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(cleaned)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=json_default,
            ensure_ascii=False,
        )


def markdown_table(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> str:
    if not rows:
        return "_No rows found._\n"

    selected_rows = rows if max_rows is None else rows[:max_rows]
    headers = list(selected_rows[0].keys())

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in selected_rows:
        values = []

        for header in headers:
            value = row.get(header)

            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=json_default)

            if isinstance(value, (datetime, date)):
                value = value.isoformat()

            text_value = "" if value is None else str(value)
            text_value = text_value.replace("\n", " ").replace("|", "\\|")

            if len(text_value) > 250:
                text_value = text_value[:250] + "...[truncated]"

            values.append(text_value)

        lines.append("| " + " | ".join(values) + " |")

    if max_rows is not None and len(rows) > max_rows:
        lines.append("")
        lines.append(f"_Showing {max_rows} of {len(rows)} rows in Markdown preview._")

    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    *,
    args: argparse.Namespace,
    report_data: dict[str, list[dict[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# Search Audit Report\n\n")

        file.write("## Filters\n\n")
        file.write(f"- request_id: `{args.request_id or ''}`\n")
        file.write(f"- user_id: `{args.user_id or ''}`\n")
        file.write(f"- pathway: `{args.pathway or ''}`\n")
        file.write(f"- payload_status: `{args.payload_status or ''}`\n")
        file.write(f"- validation_status: `{args.validation_status or ''}`\n")
        file.write(f"- start: `{args.start or ''}`\n")
        file.write(f"- end: `{args.end or ''}`\n")
        file.write(f"- max_items: `{args.max_items}`\n")
        file.write(f"- include_evidence: `{args.include_evidence}`\n\n")

        file.write("## Run Summary\n\n")
        file.write(markdown_table(report_data["run_summary"]))

        file.write("\n## Payload Count Summary\n\n")
        file.write(markdown_table(report_data["count_summary"]))

        file.write("\n## Validation Details\n\n")
        file.write(markdown_table(report_data["validation_details"]))

        file.write("\n## Item Details\n\n")
        file.write(
            markdown_table(
                report_data["item_details"],
                max_rows=args.md_item_limit,
            )
        )


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    report_data: dict[str, list[dict[str, Any]]],
    written_files: list[Path],
) -> None:
    """
    Write a small manifest describing this report run.
    """

    manifest = {
        "created_at": datetime.now().isoformat(),
        "filters": {
            "request_id": args.request_id,
            "user_id": args.user_id,
            "pathway": args.pathway,
            "payload_status": args.payload_status,
            "validation_status": args.validation_status,
            "start": args.start,
            "end": args.end,
            "max_items": args.max_items,
            "include_evidence": args.include_evidence,
        },
        "row_counts": {
            "run_summary": len(report_data["run_summary"]),
            "count_summary": len(report_data["count_summary"]),
            "item_details": len(report_data["item_details"]),
            "validation_details": len(report_data["validation_details"]),
        },
        "files": [
            str(file_path.name)
            for file_path in written_files
        ],
    }

    write_json(path, manifest)


def write_reports(
    *,
    out_dir: Path,
    report_format: str,
    report_data: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[Path]:
    """
    Write report files into a dedicated report folder.

    File names are intentionally simple because the folder name already
    contains request/user/pathway/timestamp context.
    """

    written_files: list[Path] = []

    if report_format in {"json", "all"}:
        json_path = out_dir / "full_report.json"
        write_json(json_path, report_data)
        written_files.append(json_path)

    if report_format in {"csv", "all"}:
        csv_targets = {
            "run_summary": report_data["run_summary"],
            "count_summary": report_data["count_summary"],
            "item_details": report_data["item_details"],
            "validation_details": report_data["validation_details"],
        }

        for name, rows in csv_targets.items():
            csv_path = out_dir / f"{name}.csv"
            write_csv(csv_path, rows)
            written_files.append(csv_path)

    if report_format in {"md", "all"}:
        md_path = out_dir / "report.md"
        write_markdown_report(
            md_path,
            args=args,
            report_data=report_data,
        )
        written_files.append(md_path)

    manifest_path = out_dir / "manifest.json"
    write_manifest(
        manifest_path,
        args=args,
        report_data=report_data,
        written_files=written_files,
    )
    written_files.append(manifest_path)

    return written_files


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate search audit reports from audit schema.",
    )

    parser.add_argument(
        "--request-id",
        default=None,
        help="Filter by request_id, for example 3931161f.",
    )

    parser.add_argument(
        "--user-id",
        default=None,
        help="Filter by user_id.",
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Start date/time filter. Examples: "
            "'2026-05-08', '2026-05-08 03:00', "
            "'2026-05-08T03:00:00-05:00'. "
            "Naive times are treated as America/Chicago."
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "End date/time filter. Examples: "
            "'2026-05-08', '2026-05-08 04:00', "
            "'2026-05-08T04:00:00-05:00'. "
            "Naive times are treated as America/Chicago."
        ),
    )

    parser.add_argument(
        "--pathway",
        default=None,
        help=(
            "Filter by pathway_name, for example "
            "'rag' or 'payload_projection'."
        ),
    )

    parser.add_argument(
        "--payload-status",
        default=None,
        help="Filter by payload_status, for example pending, complete, error.",
    )

    parser.add_argument(
        "--validation-status",
        default=None,
        help="Filter by run validation_status, for example passed, warning, failed.",
    )

    parser.add_argument(
        "--format",
        default="all",
        choices=sorted(VALID_FORMATS),
        help="Output format: csv, json, md, or all.",
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_REPORT_DIR),
        help=f"Base output directory. Default: {DEFAULT_REPORT_DIR}",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Maximum item detail rows to export. "
            "Use 0 for unlimited. Default: 0."
        ),
    )

    parser.add_argument(
        "--md-item-limit",
        type=int,
        default=200,
        help=(
            "Maximum item rows to include in Markdown preview. "
            "CSV/JSON still honor --max-items. Default: 200."
        ),
    )

    parser.add_argument(
        "--include-evidence",
        action="store_true",
        help=(
            "Include item evidence JSONB in item details. "
            "This can make reports very large."
        ),
    )

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.start_dt = parse_datetime_filter(args.start, is_end=False)
    args.end_dt = parse_datetime_filter(args.end, is_end=True)

    if args.max_items < 0:
        raise ValueError("--max-items cannot be negative.")

    if args.md_item_limit < 0:
        raise ValueError("--md-item-limit cannot be negative.")

    args.out_dir = str(Path(args.out_dir).resolve())

    return args


def print_report_summary(
    *,
    report_data: dict[str, list[dict[str, Any]]],
    written_files: list[Path],
    report_out_dir: Path,
) -> None:
    run_count = len(report_data["run_summary"])
    count_rows = len(report_data["count_summary"])
    item_rows = len(report_data["item_details"])
    validation_rows = len(report_data["validation_details"])

    logger.info("Report complete.")
    logger.info("Report folder: %s", report_out_dir)
    logger.info("Runs found: %s", run_count)
    logger.info("Count rows: %s", count_rows)
    logger.info("Item detail rows: %s", item_rows)
    logger.info("Validation rows: %s", validation_rows)

    for file_path in written_files:
        logger.info("Wrote report file: %s", file_path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_args(args)

    base_out_dir = ensure_report_dir(Path(args.out_dir))
    report_folder_name = build_report_folder_name(args)
    report_out_dir = ensure_report_dir(base_out_dir / report_folder_name)

    logger.info("Starting search audit report.")
    logger.info("Base output directory: %s", base_out_dir)
    logger.info("Report output directory: %s", report_out_dir)

    reporter = SearchAuditReporter()
    report_data = reporter.run_report(args)

    written_files = write_reports(
        out_dir=report_out_dir,
        report_format=args.format,
        report_data=report_data,
        args=args,
    )

    print_report_summary(
        report_data=report_data,
        written_files=written_files,
        report_out_dir=report_out_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())