"""
Tablet dropdown cache service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_dropdown_cache_service.py

Responsibilities:
    - Build mobile-friendly dropdown/cache payloads.
    - Build cache status responses.
    - Calculate a practical cache version.
    - Update per-tablet dropdown cache manifest rows.
    - Keep service logic session-safe.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import (
    TabletDropdownCacheManifest,
)


class TabletDropdownCacheService:
    """
    Domain service for tablet dropdown/cache payloads.

    This service intentionally imports core EMTAC lookup models lazily so that
    the service can still import cleanly while model names are being refactored.
    """

    CACHE_SPECS: dict[str, dict[str, Any]] = {
        "site_locations": {
            "model_candidates": ["SiteLocation"],
            "display_fields": ["title", "name", "room_number", "site_area"],
            "description_fields": ["description", "site_area", "room_number"],
            "parent_fields": ["building_id"],
        },
        "areas": {
            "model_candidates": ["Area"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "equipment_groups": {
            "model_candidates": ["EquipmentGroup"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["area_id"],
        },
        "models": {
            "model_candidates": ["Model"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["equipment_group_id"],
        },
        "asset_numbers": {
            "model_candidates": ["AssetNumber"],
            "display_fields": ["number", "name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["model_id"],
        },
        "locations": {
            "model_candidates": ["Location"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["model_id", "asset_number_id"],
        },
        "subassemblies": {
            "model_candidates": ["Subassembly"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "component_assemblies": {
            "model_candidates": ["ComponentAssembly"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["subassembly_id"],
        },
        "assembly_views": {
            "model_candidates": ["AssemblyView", "ComponentView"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": ["component_assembly_id", "subassembly_id"],
        },
        "tool_categories": {
            "model_candidates": ["ToolCategory"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "part_categories": {
            "model_candidates": ["PartCategory"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "problem_categories": {
            "model_candidates": ["ProblemCategory"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "solution_categories": {
            "model_candidates": ["SolutionCategory"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
        "task_categories": {
            "model_candidates": ["TaskCategory"],
            "display_fields": ["name", "title"],
            "description_fields": ["description"],
            "parent_fields": [],
        },
    }

    @staticmethod
    def utc_now() -> datetime:
        """
        Return timezone-aware UTC datetime.
        """
        return datetime.now(timezone.utc)

    @staticmethod
    def datetime_to_version(value: datetime | None = None) -> str:
        """
        Convert a datetime into a stable cache version string.
        """
        value = value or TabletDropdownCacheService.utc_now()

        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    @staticmethod
    def json_safe_value(value: Any) -> Any:
        """
        Convert Python/SQLAlchemy values into response-safe values.
        """
        if isinstance(value, datetime):
            return value.isoformat()

        return value

    @staticmethod
    def normalize_string(value: Any, *, max_length: int | None = None) -> str | None:
        """
        Normalize optional string values.
        """
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        if max_length is not None and len(normalized) > max_length:
            normalized = normalized[:max_length]

        return normalized

    def _get_emtacdb_fts_module(self) -> Any:
        """
        Import the main EMTAC ORM module lazily.
        """
        import modules.emtacdb.emtacdb_fts as emtacdb_fts

        return emtacdb_fts

    def resolve_model_class(self, model_candidates: list[str]) -> type[Any] | None:
        """
        Resolve the first available model class from modules.emtacdb.emtacdb_fts.

        This supports current and older model naming while avoiding hard import
        failures when optional category classes do not exist yet.
        """
        emtacdb_fts = self._get_emtacdb_fts_module()

        for model_name in model_candidates:
            model_class = getattr(emtacdb_fts, model_name, None)

            if model_class is not None:
                return model_class

        return None

    @staticmethod
    def model_has_field(model_class: type[Any], field_name: str) -> bool:
        """
        Return True if the ORM model has a field/attribute.
        """
        return hasattr(model_class, field_name)

    def get_first_available_value(
        self,
        row: Any,
        field_names: list[str],
    ) -> Any:
        """
        Return the first non-empty value from a list of candidate fields.
        """
        for field_name in field_names:
            if hasattr(row, field_name):
                value = getattr(row, field_name)

                if value is not None and str(value).strip():
                    return value

        return None

    def build_parent_map(
        self,
        row: Any,
        parent_fields: list[str],
    ) -> dict[str, Any]:
        """
        Return parent IDs for cache rows.

        Example:
            {
                "area_id": 1,
                "model_id": 12
            }
        """
        parent_map: dict[str, Any] = {}

        for field_name in parent_fields:
            if hasattr(row, field_name):
                value = getattr(row, field_name)

                if value is not None:
                    parent_map[field_name] = value

        return parent_map

    def get_row_updated_at(self, row: Any) -> Any:
        """
        Return best available updated timestamp from a row.
        """
        for field_name in ("updated_at", "modified_at", "created_at"):
            if hasattr(row, field_name):
                value = getattr(row, field_name)

                if value is not None:
                    return value

        return None

    def get_is_active(self, row: Any) -> bool:
        """
        Return best available active/enabled flag.

        If the row does not have an active flag, default to True.
        """
        for field_name in ("is_active", "active", "enabled"):
            if hasattr(row, field_name):
                value = getattr(row, field_name)

                if value is not None:
                    return bool(value)

        return True

    def build_cache_row(
        self,
        row: Any,
        *,
        cache_name: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert one ORM row into a tablet/mobile-friendly cache row.
        """
        record_id = getattr(row, "id", None)

        display_value = self.get_first_available_value(
            row,
            spec.get("display_fields", []),
        )

        description_value = self.get_first_available_value(
            row,
            spec.get("description_fields", []),
        )

        if display_value is None:
            display_value = f"{cache_name}:{record_id}"

        parent_map = self.build_parent_map(
            row,
            spec.get("parent_fields", []),
        )

        return {
            "id": record_id,
            "record_id": record_id,
            "display_name": self.normalize_string(display_value),
            "description": self.normalize_string(description_value),
            "parents": parent_map,
            "updated_at": self.json_safe_value(self.get_row_updated_at(row)),
            "is_active": self.get_is_active(row),
        }

    def query_cache_rows(
        self,
        session: Session,
        *,
        cache_name: str,
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Query and serialize rows for one cache.

        If the model does not exist yet, return an empty list and log a warning.
        """
        model_class = self.resolve_model_class(spec.get("model_candidates", []))

        if model_class is None:
            logger.warning(
                "[TABLET_EDGE_CACHE] No model class found for cache_name=%s candidates=%s",
                cache_name,
                spec.get("model_candidates", []),
            )
            return []

        stmt = select(model_class)

        if self.model_has_field(model_class, "id"):
            stmt = stmt.order_by(getattr(model_class, "id").asc())

        rows = list(session.execute(stmt).scalars().all())

        return [
            self.build_cache_row(
                row,
                cache_name=cache_name,
                spec=spec,
            )
            for row in rows
        ]

    def get_cache_max_timestamp(
        self,
        session: Session,
        *,
        spec: dict[str, Any],
    ) -> datetime | None:
        """
        Get max updated/modified/created timestamp for a cache's backing model.
        """
        model_class = self.resolve_model_class(spec.get("model_candidates", []))

        if model_class is None:
            return None

        for field_name in ("updated_at", "modified_at", "created_at"):
            if self.model_has_field(model_class, field_name):
                column = getattr(model_class, field_name)
                return session.execute(select(func.max(column))).scalar_one_or_none()

        return None

    def calculate_cache_version(self, session: Session) -> str:
        """
        Calculate a global cache version.

        Best effort:
            - Uses the latest updated_at/modified_at/created_at timestamp from
              available lookup tables.
            - Falls back to current UTC time if no timestamp columns exist.
        """
        max_timestamp: datetime | None = None

        for spec in self.CACHE_SPECS.values():
            cache_timestamp = self.get_cache_max_timestamp(session, spec=spec)

            if cache_timestamp is None:
                continue

            if max_timestamp is None or cache_timestamp > max_timestamp:
                max_timestamp = cache_timestamp

        return self.datetime_to_version(max_timestamp or self.utc_now())

    def get_cache_status(
        self,
        session: Session,
    ) -> dict[str, Any]:
        """
        Build dropdown cache status response.

        This does not require a tablet UID. It reports the current server-side
        cache version and record counts for each available cache.
        """
        cache_version = self.calculate_cache_version(session)

        caches: dict[str, dict[str, Any]] = {}

        for cache_name, spec in self.CACHE_SPECS.items():
            rows = self.query_cache_rows(
                session,
                cache_name=cache_name,
                spec=spec,
            )

            cache_timestamp = self.get_cache_max_timestamp(session, spec=spec)
            version = self.datetime_to_version(cache_timestamp) if cache_timestamp else cache_version

            caches[cache_name] = {
                "version": version,
                "record_count": len(rows),
            }

        logger.info(
            "[TABLET_EDGE_CACHE] Built cache status for %s cache(s). version=%s",
            len(caches),
            cache_version,
        )

        return {
            "success": True,
            "current_version": cache_version,
            "caches": caches,
        }

    def build_full_cache(
        self,
        session: Session,
    ) -> dict[str, Any]:
        """
        Build the full dropdown cache payload for the tablet.
        """
        cache_version = self.calculate_cache_version(session)

        payload: dict[str, Any] = {
            "success": True,
            "cache_version": cache_version,
        }

        total_records = 0

        for cache_name, spec in self.CACHE_SPECS.items():
            rows = self.query_cache_rows(
                session,
                cache_name=cache_name,
                spec=spec,
            )

            payload[cache_name] = rows
            total_records += len(rows)

        payload["cache_summary"] = {
            "cache_count": len(self.CACHE_SPECS),
            "total_records": total_records,
            "generated_at": self.datetime_to_version(self.utc_now()),
        }

        logger.info(
            "[TABLET_EDGE_CACHE] Built full dropdown cache. cache_count=%s total_records=%s version=%s",
            len(self.CACHE_SPECS),
            total_records,
            cache_version,
        )

        return payload

    def build_delta_cache(
        self,
        session: Session,
        since: str | datetime | None,
    ) -> dict[str, Any]:
        """
        Placeholder delta cache response.

        Version 1 uses full refresh. Delta can be added later once every lookup
        table has reliable updated_at/deleted handling.
        """
        cache_version = self.calculate_cache_version(session)

        logger.info(
            "[TABLET_EDGE_CACHE] Delta cache requested but full refresh is required. since=%s version=%s",
            since,
            cache_version,
        )

        return {
            "success": True,
            "cache_version": cache_version,
            "full_refresh_required": True,
            "updated": {},
            "deleted": {},
            "message": "Delta cache is not implemented yet. Request full cache refresh.",
        }

    def get_manifest(
        self,
        session: Session,
        tablet_device_id: int,
        cache_name: str,
    ) -> TabletDropdownCacheManifest | None:
        """
        Return cache manifest row for a tablet/cache name.
        """
        stmt = select(TabletDropdownCacheManifest).where(
            TabletDropdownCacheManifest.tablet_device_id == tablet_device_id,
            TabletDropdownCacheManifest.cache_name == cache_name,
        )

        return session.execute(stmt).scalar_one_or_none()

    def update_manifest(
        self,
        session: Session,
        tablet_device_id: int,
        cache_name: str,
        version: str,
        record_count: int,
        *,
        sync_status: str = "success",
        full_sync: bool = True,
    ) -> TabletDropdownCacheManifest:
        """
        Create or update tablet dropdown cache manifest.

        Transaction ownership belongs to the orchestrator.
        """
        now_utc = self.utc_now()

        manifest = self.get_manifest(
            session=session,
            tablet_device_id=tablet_device_id,
            cache_name=cache_name,
        )

        if manifest is None:
            manifest = TabletDropdownCacheManifest(
                tablet_device_id=tablet_device_id,
                cache_name=cache_name,
                cache_version=version,
                record_count=max(0, int(record_count)),
                sync_status=self.normalize_string(sync_status, max_length=50) or "unknown",
            )
            session.add(manifest)
        else:
            manifest.cache_version = version
            manifest.record_count = max(0, int(record_count))
            manifest.sync_status = self.normalize_string(sync_status, max_length=50) or "unknown"

        if full_sync:
            manifest.last_full_sync_at = now_utc
        else:
            manifest.last_delta_sync_at = now_utc

        session.flush()

        logger.info(
            "[TABLET_EDGE_CACHE] Updated manifest tablet_device_id=%s cache_name=%s version=%s record_count=%s status=%s",
            tablet_device_id,
            cache_name,
            version,
            record_count,
            sync_status,
        )

        return manifest

    def update_manifests_from_full_cache(
        self,
        session: Session,
        tablet_device_id: int,
        full_cache_payload: dict[str, Any],
    ) -> list[TabletDropdownCacheManifest]:
        """
        Update all manifest rows based on a full cache payload.
        """
        version = str(full_cache_payload.get("cache_version") or self.calculate_cache_version(session))

        updated_manifests: list[TabletDropdownCacheManifest] = []

        for cache_name in self.CACHE_SPECS:
            rows = full_cache_payload.get(cache_name, [])

            if not isinstance(rows, list):
                rows = []

            manifest = self.update_manifest(
                session=session,
                tablet_device_id=tablet_device_id,
                cache_name=cache_name,
                version=version,
                record_count=len(rows),
                sync_status="success",
                full_sync=True,
            )

            updated_manifests.append(manifest)

        return updated_manifests

    def build_manifest_response(
        self,
        manifests: list[TabletDropdownCacheManifest],
    ) -> dict[str, Any]:
        """
        Build response-safe manifest response.
        """
        return {
            "success": True,
            "count": len(manifests),
            "manifests": [manifest.to_dict() for manifest in manifests],
        }