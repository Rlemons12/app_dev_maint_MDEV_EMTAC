"""
Tablet Edge orchestrator for the EMTAC Tablet Edge Agent.

File:
    modules/orchestrators/tablet_edge_orchestrator.py

Responsibilities:
    - Own database session lifecycle.
    - Own commit/rollback/close behavior.
    - Call tablet_edge services.
    - Convert service results into response-safe dictionaries.
    - Coordinate Wi-Fi/router/access-point tracking.
    - Return route-friendly tuples:
        (success: bool, response_body: dict, status_code: int)

Important:
    Coordinators should call this orchestrator.
    Services should not create/close sessions or commit/rollback.

Router/AP tracking:
    The coordinator normalizes Wi-Fi/router fields.
    This orchestrator wires those normalized fields into the service layer.

    Main tables involved:
        tablet_edge.tablet_wifi_access_point
        tablet_edge.tablet_wifi_observation
        tablet_edge.tablet_network_event
        tablet_edge.tablet_health_sample
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from sqlalchemy.orm import Session

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger
from modules.services.tablet_edge.tablet_app_update_service import (TabletAppUpdateService)
from modules.services.tablet_edge.tablet_app_log_service import TabletAppLogService
from modules.services.tablet_edge.tablet_device_service import TabletDeviceService
from modules.services.tablet_edge.tablet_dropdown_cache_service import (
    TabletDropdownCacheService,
)
from modules.services.tablet_edge.tablet_health_sample_service import (
    TabletHealthSampleService,
)
from modules.services.tablet_edge.tablet_network_event_service import (
    TabletNetworkEventService,
)
from modules.services.tablet_edge.tablet_offline_event_service import (
    TabletOfflineEventService,
)
from modules.services.tablet_edge.tablet_sync_service import TabletSyncService
from modules.services.tablet_edge.tablet_wifi_tracking_service import (
    TabletWifiTrackingService,
)


ResponseTuple = tuple[bool, dict[str, Any], int]


class TabletEdgeOrchestrator:
    """
    Orchestrator for the EMTAC Tablet Edge Agent backend.

    This class follows the EMTAC pattern:

        Coordinator
            ↓
        Orchestrator
            ↓
        Service
            ↓
        Database

    The orchestrator owns transactions. Services receive a session and do
    domain work only.
    """

    NETWORK_SIGNAL_KEYS: tuple[str, ...] = (
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "ip_address",
        "gateway_address",
        "dhcp_server_address",
        "dns_servers",
        "wifi_rssi",
        "signal_level",
        "frequency_mhz",
        "wifi_band",
        "link_speed_mbps",
        "is_online",
        "server_url",
        "server_reachable",
        "server_latency_ms",
        "quality_level",
    )

    def __init__(self, db_config: DatabaseConfig | None = None) -> None:
        self.db_config = db_config or DatabaseConfig()

        self.device_service = TabletDeviceService()
        self.network_event_service = TabletNetworkEventService()
        self.health_sample_service = TabletHealthSampleService()
        self.dropdown_cache_service = TabletDropdownCacheService()
        self.offline_event_service = TabletOfflineEventService()
        self.app_log_service = TabletAppLogService()
        self.sync_service = TabletSyncService()
        self.wifi_tracking_service = TabletWifiTrackingService()
        self.app_update_service = TabletAppUpdateService()

    @staticmethod
    def utc_now_iso() -> str:
        """
        Return current UTC time as an ISO string.
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _success_response(
        response_body: dict[str, Any],
        status_code: int = 200,
    ) -> ResponseTuple:
        """
        Build successful route tuple.
        """
        return True, response_body, status_code

    @staticmethod
    def _error_response(
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "tablet_edge_error",
        details: dict[str, Any] | None = None,
    ) -> ResponseTuple:
        """
        Build error route tuple.
        """
        response_body: dict[str, Any] = {
            "success": False,
            "error": message,
            "error_type": error_type,
            "server_time_utc": TabletEdgeOrchestrator.utc_now_iso(),
        }

        if details:
            response_body["details"] = details

        return False, response_body, status_code

    def _get_session(self) -> Session:
        """
        Create a main database session using the existing EMTAC DatabaseConfig.

        This project uses DatabaseConfig.get_main_session() as the canonical
        session creation method.
        """
        if not hasattr(self.db_config, "get_main_session"):
            raise RuntimeError(
                "DatabaseConfig does not expose get_main_session(). "
                "TabletEdgeOrchestrator requires a main database session."
            )

        session = self.db_config.get_main_session()

        if session is None:
            raise RuntimeError("DatabaseConfig.get_main_session() returned None.")

        return session

    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """
        Open a database session and manage commit/rollback/close.

        Services must not commit or rollback. They receive this session.
        """
        session = self._get_session()

        try:
            yield session
            session.commit()

        except Exception:
            session.rollback()
            logger.exception("[TABLET_EDGE] Transaction failed and was rolled back.")
            raise

        finally:
            session.close()

    @staticmethod
    def _get_required_value(
        payload: dict[str, Any],
        field_name: str,
    ) -> Any:
        """
        Get required field from payload.
        """
        value = payload.get(field_name)

        if value is None:
            raise ValueError(f"{field_name} is required.")

        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} is required.")

        return value

    @staticmethod
    def _get_list_value(
        payload: dict[str, Any],
        field_name: str,
    ) -> list[dict[str, Any]]:
        """
        Get required list field from payload.
        """
        value = payload.get(field_name)

        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list.")

        return value

    @staticmethod
    def _payload_has_network_signal(payload: dict[str, Any]) -> bool:
        """
        Return True when the payload contains enough network/AP information
        to justify recording a Wi-Fi observation.
        """
        for key in TabletEdgeOrchestrator.NETWORK_SIGNAL_KEYS:
            value = payload.get(key)

            if value not in (None, ""):
                return True

        return False

    @staticmethod
    def _merge_parent_fields_into_item(
        *,
        parent_payload: dict[str, Any],
        item_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge useful top-level fields into an event/sample item when missing.

        The coordinator already does a lot of this, but keeping this here makes
        the orchestrator tolerant of direct script tests or older clients.
        """
        merged = dict(item_payload)

        inheritable_keys = (
            "tablet_uid",
            "tablet_name",
            "server_url",
            "page_url",
            "current_page_url",
            "is_online",
            "ssid",
            "bssid",
            "router_ip",
            "router_name",
            "ip_address",
            "gateway_address",
            "dhcp_server_address",
            "dns_servers",
            "wifi_rssi",
            "signal_level",
            "frequency_mhz",
            "wifi_band",
            "link_speed_mbps",
            "server_reachable",
            "server_latency_ms",
            "quality_level",
        )

        for key in inheritable_keys:
            if merged.get(key) in (None, "") and parent_payload.get(key) not in (
                None,
                "",
            ):
                merged[key] = parent_payload.get(key)

        return merged

    def _get_events_from_payload(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Accept both list-style and single-event payloads.

        Supported:
            {"events": [...]}
            {"network_events": [...]}
            {"event_type": "...", "quality_level": "..."}
        """
        if isinstance(payload.get("events"), list):
            events = payload["events"]

        elif isinstance(payload.get("network_events"), list):
            events = payload["network_events"]

        elif payload.get("event_type"):
            events = [payload]

        else:
            raise ValueError("events must be a list.")

        normalized_events: list[dict[str, Any]] = []

        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"events item #{index} must be a JSON object.")

            normalized_events.append(
                self._merge_parent_fields_into_item(
                    parent_payload=payload,
                    item_payload=event,
                )
            )

        return normalized_events

    def _get_samples_from_payload(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Accept both list-style and single-sample payloads.

        Supported:
            {"samples": [...]}
            {"health_samples": [...]}
            {"quality_level": "...", "server_reachable": true}
        """
        if isinstance(payload.get("samples"), list):
            samples = payload["samples"]

        elif isinstance(payload.get("health_samples"), list):
            samples = payload["health_samples"]

        elif (
            payload.get("quality_level")
            or "server_reachable" in payload
            or self._payload_has_network_signal(payload)
        ):
            samples = [payload]

        else:
            raise ValueError("samples must be a list.")

        normalized_samples: list[dict[str, Any]] = []

        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"samples item #{index} must be a JSON object.")

            normalized_samples.append(
                self._merge_parent_fields_into_item(
                    parent_payload=payload,
                    item_payload=sample,
                )
            )

        return normalized_samples

    def _attach_access_point_to_payload(
        self,
        *,
        session: Session,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int | None]:
        """
        Upsert/find the access point for a payload and attach access_point_id.

        Returns:
            (updated_payload, access_point_id)
        """
        updated_payload = dict(payload)

        if not self._payload_has_network_signal(updated_payload):
            return updated_payload, None

        access_point = self.wifi_tracking_service.upsert_access_point_from_payload(
            session=session,
            payload=updated_payload,
        )

        if access_point is None:
            return updated_payload, None

        updated_payload["access_point_id"] = access_point.id

        return updated_payload, int(access_point.id)

    def _record_wifi_observation_if_available(
        self,
        *,
        session: Session,
        tablet_device_id: int,
        payload: dict[str, Any],
        source: str,
    ) -> int | None:
        """
        Record a tablet Wi-Fi observation when useful network/router/AP data is
        present in the payload.
        """
        if not self._payload_has_network_signal(payload):
            return None

        observation = self.wifi_tracking_service.record_observation_from_payload(
            session=session,
            tablet_device_id=tablet_device_id,
            payload=payload,
            source=source,
        )

        if observation is None:
            return None

        return int(observation.id)

    def _prepare_network_items_for_insert(
        self,
        *,
        session: Session,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, list[int]]:
        """
        Attach access_point_id to event/sample items before they are inserted.

        Returns:
            prepared_items,
            access_points_resolved_count,
            access_point_ids
        """
        prepared_items: list[dict[str, Any]] = []
        access_point_ids: list[int] = []

        for item in items:
            prepared_item, access_point_id = self._attach_access_point_to_payload(
                session=session,
                payload=item,
            )

            prepared_items.append(prepared_item)

            if access_point_id is not None:
                access_point_ids.append(access_point_id)

        unique_access_point_ids = sorted(set(access_point_ids))

        return prepared_items, len(unique_access_point_ids), unique_access_point_ids

    def register_tablet(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Register or update a tablet device.

        Route:
            POST /tablet-edge/register
        """
        logger.info("[TABLET_EDGE_REGISTER] Register tablet requested.")

        try:
            with self.transaction() as session:
                tablet = self.device_service.register_or_update(
                    session=session,
                    payload=payload,
                )

                prepared_payload, access_point_id = self._attach_access_point_to_payload(
                    session=session,
                    payload=payload,
                )

                wifi_observation_id = self._record_wifi_observation_if_available(
                    session=session,
                    tablet_device_id=tablet.id,
                    payload=prepared_payload,
                    source="registration",
                )

                self.sync_service.record_quick_sync(
                    session=session,
                    tablet_device_id=tablet.id,
                    sync_type="registration",
                    sync_direction="tablet_to_server",
                    status="success",
                    records_sent=1,
                    records_received=1,
                    records_failed=0,
                )

                response_body = self.device_service.build_register_response(tablet)
                response_body["access_point_id"] = access_point_id
                response_body["wifi_observation_id"] = wifi_observation_id
                response_body["wifi_observation_recorded"] = (
                    wifi_observation_id is not None
                )
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_REGISTER] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="register_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_REGISTER] Registration failed.")
            return self._error_response(
                "Tablet registration failed.",
                status_code=500,
                error_type="register_failed",
                details={"exception": str(exc)},
            )

    def heartbeat(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Update tablet last_seen_at and return server time.

        Route:
            POST /tablet-edge/heartbeat

        If Wi-Fi/router/AP fields are present, this also records a lightweight
        Wi-Fi observation.
        """
        logger.info("[TABLET_EDGE_HEARTBEAT] Heartbeat requested.")

        try:
            tablet_uid = self._get_required_value(payload, "tablet_uid")

            with self.transaction() as session:
                tablet = self.device_service.update_heartbeat(
                    session=session,
                    tablet_uid=tablet_uid,
                    payload=payload,
                )

                prepared_payload, access_point_id = self._attach_access_point_to_payload(
                    session=session,
                    payload=payload,
                )

                wifi_observation_id = self._record_wifi_observation_if_available(
                    session=session,
                    tablet_device_id=tablet.id,
                    payload=prepared_payload,
                    source="heartbeat",
                )

                self.sync_service.record_quick_sync(
                    session=session,
                    tablet_device_id=tablet.id,
                    sync_type="heartbeat",
                    sync_direction="tablet_to_server",
                    status="success",
                    records_sent=1,
                    records_received=1,
                    records_failed=0,
                )

                response_body = self.device_service.build_heartbeat_response(tablet)
                response_body["access_point_id"] = access_point_id
                response_body["wifi_observation_id"] = wifi_observation_id
                response_body["wifi_observation_recorded"] = (
                    wifi_observation_id is not None
                )

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_HEARTBEAT] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="heartbeat_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_HEARTBEAT] Heartbeat failed.")
            return self._error_response(
                "Tablet heartbeat failed.",
                status_code=500,
                error_type="heartbeat_failed",
                details={"exception": str(exc)},
            )

    def record_network_events(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Record tablet network events.

        Route:
            POST /tablet-edge/network-events

        Also:
            - upserts access point records when BSSID is present
            - attaches access_point_id to network events
            - records Wi-Fi observations for event payloads that contain network data
        """
        logger.info("[TABLET_EDGE_NETWORK] Network events requested.")

        try:
            tablet_uid = self._get_required_value(payload, "tablet_uid")
            events = self._get_events_from_payload(payload)

            with self.transaction() as session:
                tablet = self.device_service.require_active_by_uid(
                    session=session,
                    tablet_uid=tablet_uid,
                )

                prepared_events, access_points_resolved, access_point_ids = (
                    self._prepare_network_items_for_insert(
                        session=session,
                        items=events,
                    )
                )

                inserted_events = self.network_event_service.record_events(
                    session=session,
                    tablet_device_id=tablet.id,
                    events=prepared_events,
                )

                wifi_observation_ids: list[int] = []

                for event in prepared_events:
                    observation_id = self._record_wifi_observation_if_available(
                        session=session,
                        tablet_device_id=tablet.id,
                        payload=event,
                        source="network_event",
                    )

                    if observation_id is not None:
                        wifi_observation_ids.append(observation_id)

                self.sync_service.record_quick_sync(
                    session=session,
                    tablet_device_id=tablet.id,
                    sync_type="network_events",
                    sync_direction="tablet_to_server",
                    status="success",
                    records_sent=len(inserted_events),
                    records_received=0,
                    records_failed=0,
                )

                response_body = self.network_event_service.build_record_events_response(
                    inserted_events
                )
                response_body["access_points_resolved"] = access_points_resolved
                response_body["access_point_ids"] = access_point_ids
                response_body["wifi_observations_recorded"] = len(
                    wifi_observation_ids
                )
                response_body["wifi_observation_ids"] = wifi_observation_ids
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_NETWORK] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="network_events_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_NETWORK] Recording network events failed.")
            return self._error_response(
                "Recording network events failed.",
                status_code=500,
                error_type="network_events_failed",
                details={"exception": str(exc)},
            )

    def record_health_samples(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Record tablet health samples.

        Route:
            POST /tablet-edge/health-samples

        Also:
            - upserts access point records when BSSID is present
            - attaches access_point_id to health samples
            - records Wi-Fi observations for sample payloads that contain network data
        """
        logger.info("[TABLET_EDGE_HEALTH] Health samples requested.")

        try:
            tablet_uid = self._get_required_value(payload, "tablet_uid")
            samples = self._get_samples_from_payload(payload)

            with self.transaction() as session:
                tablet = self.device_service.require_active_by_uid(
                    session=session,
                    tablet_uid=tablet_uid,
                )

                prepared_samples, access_points_resolved, access_point_ids = (
                    self._prepare_network_items_for_insert(
                        session=session,
                        items=samples,
                    )
                )

                inserted_samples = self.health_sample_service.record_samples(
                    session=session,
                    tablet_device_id=tablet.id,
                    samples=prepared_samples,
                )

                wifi_observation_ids: list[int] = []

                for sample in prepared_samples:
                    observation_id = self._record_wifi_observation_if_available(
                        session=session,
                        tablet_device_id=tablet.id,
                        payload=sample,
                        source="health_sample",
                    )

                    if observation_id is not None:
                        wifi_observation_ids.append(observation_id)

                self.sync_service.record_quick_sync(
                    session=session,
                    tablet_device_id=tablet.id,
                    sync_type="health_samples",
                    sync_direction="tablet_to_server",
                    status="success",
                    records_sent=len(inserted_samples),
                    records_received=0,
                    records_failed=0,
                )

                response_body = self.health_sample_service.build_record_samples_response(
                    inserted_samples
                )
                response_body["access_points_resolved"] = access_points_resolved
                response_body["access_point_ids"] = access_point_ids
                response_body["wifi_observations_recorded"] = len(
                    wifi_observation_ids
                )
                response_body["wifi_observation_ids"] = wifi_observation_ids
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_HEALTH] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="health_samples_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_HEALTH] Recording health samples failed.")
            return self._error_response(
                "Recording health samples failed.",
                status_code=500,
                error_type="health_samples_failed",
                details={"exception": str(exc)},
            )

    def get_dropdown_cache_status(
        self,
        tablet_uid: str | None = None,
    ) -> ResponseTuple:
        """
        Return current dropdown cache status.

        Route:
            GET /tablet-edge/dropdown-cache/status

        tablet_uid is optional. If supplied, it is validated when possible.
        """
        logger.info("[TABLET_EDGE_CACHE] Dropdown cache status requested.")

        try:
            with self.transaction() as session:
                response_body = self.dropdown_cache_service.get_cache_status(session)

                if tablet_uid:
                    tablet = self.device_service.require_active_by_uid(
                        session=session,
                        tablet_uid=tablet_uid,
                    )
                    response_body["tablet_device_id"] = tablet.id
                    response_body["tablet_uid"] = str(tablet.tablet_uid)
                    response_body["tablet_name"] = tablet.tablet_name

                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_CACHE] Invalid cache status request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="dropdown_cache_status_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_CACHE] Cache status failed.")
            return self._error_response(
                "Dropdown cache status failed.",
                status_code=500,
                error_type="dropdown_cache_status_failed",
                details={"exception": str(exc)},
            )

    def get_full_dropdown_cache(
        self,
        tablet_uid: str | None = None,
    ) -> ResponseTuple:
        """
        Return full dropdown cache payload.

        Route:
            GET /tablet-edge/dropdown-cache/full

        tablet_uid is optional. If supplied, manifests are updated for that tablet.
        """
        logger.info("[TABLET_EDGE_CACHE] Full dropdown cache requested.")

        try:
            with self.transaction() as session:
                tablet = None

                if tablet_uid:
                    tablet = self.device_service.require_active_by_uid(
                        session=session,
                        tablet_uid=tablet_uid,
                    )

                full_cache_payload = self.dropdown_cache_service.build_full_cache(session)

                if tablet is not None:
                    manifests = self.dropdown_cache_service.update_manifests_from_full_cache(
                        session=session,
                        tablet_device_id=tablet.id,
                        full_cache_payload=full_cache_payload,
                    )

                    full_cache_payload["tablet_device_id"] = tablet.id
                    full_cache_payload["tablet_uid"] = str(tablet.tablet_uid)
                    full_cache_payload["tablet_name"] = tablet.tablet_name
                    full_cache_payload["manifest_count"] = len(manifests)

                    self.sync_service.record_quick_sync(
                        session=session,
                        tablet_device_id=tablet.id,
                        sync_type="dropdown_cache",
                        sync_direction="server_to_tablet",
                        status="success",
                        records_sent=int(
                            full_cache_payload.get("cache_summary", {}).get(
                                "total_records", 0
                            )
                        ),
                        records_received=0,
                        records_failed=0,
                    )

                full_cache_payload["server_time_utc"] = self.utc_now_iso()

                return self._success_response(full_cache_payload, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_CACHE] Invalid full cache request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="dropdown_cache_full_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_CACHE] Full cache failed.")
            return self._error_response(
                "Full dropdown cache failed.",
                status_code=500,
                error_type="dropdown_cache_full_failed",
                details={"exception": str(exc)},
            )

    def get_delta_dropdown_cache(
        self,
        since: str | None = None,
    ) -> ResponseTuple:
        """
        Return delta dropdown cache placeholder.

        Route:
            GET /tablet-edge/dropdown-cache/delta?since=<timestamp>
        """
        logger.info("[TABLET_EDGE_CACHE] Delta dropdown cache requested.")

        try:
            with self.transaction() as session:
                response_body = self.dropdown_cache_service.build_delta_cache(
                    session=session,
                    since=since,
                )
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except Exception as exc:
            logger.exception("[TABLET_EDGE_CACHE] Delta cache failed.")
            return self._error_response(
                "Delta dropdown cache failed.",
                status_code=500,
                error_type="dropdown_cache_delta_failed",
                details={"exception": str(exc)},
            )

    def sync_offline_events(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Sync offline events from the tablet.

        Route:
            POST /tablet-edge/offline-events/sync
        """
        logger.info("[TABLET_EDGE_OFFLINE] Offline event sync requested.")

        try:
            tablet_uid = self._get_required_value(payload, "tablet_uid")
            events = self._get_list_value(payload, "events")

            with self.transaction() as session:
                tablet = self.device_service.require_active_by_uid(
                    session=session,
                    tablet_uid=tablet_uid,
                )

                sync_result = self.offline_event_service.sync_offline_events(
                    session=session,
                    tablet_device_id=tablet.id,
                    events=events,
                )

                self.sync_service.record_quick_sync(
                    session=session,
                    tablet_device_id=tablet.id,
                    sync_type="offline_events",
                    sync_direction="tablet_to_server",
                    status="success",
                    records_sent=len(events),
                    records_received=0,
                    records_failed=int(sync_result.get("failed", 0)),
                )

                response_body = self.offline_event_service.build_sync_response(
                    sync_result
                )
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_OFFLINE] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="offline_events_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_OFFLINE] Offline event sync failed.")
            return self._error_response(
                "Offline event sync failed.",
                status_code=500,
                error_type="offline_events_sync_failed",
                details={"exception": str(exc)},
            )

    def record_app_logs(self, payload: dict[str, Any]) -> ResponseTuple:
        """
        Record app logs from the tablet.

        Route:
            POST /tablet-edge/app-logs

        This route accepts logs even when tablet_uid is missing or cannot be
        resolved. In that case tablet_device_id is stored as NULL.
        """
        logger.info("[TABLET_EDGE_APP_LOG] App logs requested.")

        try:
            logs = self._get_list_value(payload, "logs")
            tablet_uid = payload.get("tablet_uid")

            with self.transaction() as session:
                tablet_device_id: int | None = None

                if tablet_uid:
                    try:
                        tablet = self.device_service.get_by_uid(
                            session=session,
                            tablet_uid=tablet_uid,
                        )

                        if tablet is not None and tablet.is_active:
                            tablet_device_id = tablet.id

                    except Exception as exc:
                        logger.warning(
                            "[TABLET_EDGE_APP_LOG] Could not resolve tablet_uid=%s. "
                            "Logs will be stored without tablet_device_id. error=%s",
                            tablet_uid,
                            exc,
                        )

                inserted_logs = self.app_log_service.record_logs(
                    session=session,
                    tablet_device_id=tablet_device_id,
                    logs=logs,
                )

                if tablet_device_id is not None:
                    self.sync_service.record_quick_sync(
                        session=session,
                        tablet_device_id=tablet_device_id,
                        sync_type="app_logs",
                        sync_direction="tablet_to_server",
                        status="success",
                        records_sent=len(inserted_logs),
                        records_received=0,
                        records_failed=0,
                    )

                response_body = self.app_log_service.build_record_logs_response(
                    inserted_logs
                )
                response_body["server_time_utc"] = self.utc_now_iso()

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_APP_LOG] Invalid request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_logs_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_APP_LOG] Recording app logs failed.")
            return self._error_response(
                "Recording app logs failed.",
                status_code=500,
                error_type="app_logs_failed",
                details={"exception": str(exc)},
            )
    def check_app_update(
        self,
        *,
        tablet_uid: str | None = None,
        current_version_code: Any = None,
        current_version_name: str | None = None,
        app_package: str | None = None,
        release_channel: str | None = None,
    ) -> ResponseTuple:
        """
        Check whether a tablet has an app update available.

        Route:
            GET /tablet-edge/app-update/check
        """
        logger.info("[TABLET_EDGE_APP_UPDATE] App update check requested.")

        try:
            normalized_version_code = self.app_update_service.normalize_int(
                current_version_code,
                field_name="version_code",
            )

            with self.transaction() as session:
                response_body = self.app_update_service.build_update_check_response(
                    session=session,
                    tablet_uid=tablet_uid,
                    current_version_code=normalized_version_code,
                    current_version_name=current_version_name,
                    app_package=app_package or "com.example.emtactablet",
                    release_channel=release_channel or "stable",
                )

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_APP_UPDATE] Invalid update check: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_update_check_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_APP_UPDATE] App update check failed.")
            return self._error_response(
                "App update check failed.",
                status_code=500,
                error_type="app_update_check_failed",
                details={"exception": str(exc)},
            )

    def get_app_update_download_file(
        self,
        *,
        release_id: int,
    ) -> ResponseTuple:
        """
        Resolve APK release download metadata.

        Route:
            GET /tablet-edge/app-update/download/<release_id>

        The Flask route handles send_file(). The orchestrator only resolves and
        validates the release/file metadata.
        """
        logger.info(
            "[TABLET_EDGE_APP_UPDATE] APK download requested. release_id=%s",
            release_id,
        )

        try:
            with self.transaction() as session:
                release = self.app_update_service.get_download_release(
                    session=session,
                    release_id=release_id,
                )

                response_body = {
                    "success": True,
                    "release_id": release["id"],
                    "app_package": release["app_package"],
                    "release_channel": release["release_channel"],
                    "version_name": release["version_name"],
                    "version_code": release["version_code"],
                    "apk_filename": release["apk_filename"],
                    "apk_file_path": release["apk_file_path"],
                    "apk_sha256": release["apk_sha256"],
                    "apk_size_bytes": release["apk_size_bytes"],
                    "mime_type": release["mime_type"],
                    "server_time_utc": self.utc_now_iso(),
                }

                return self._success_response(response_body, 200)

        except FileNotFoundError as exc:
            logger.warning("[TABLET_EDGE_APP_UPDATE] APK file not found: %s", exc)
            return self._error_response(
                str(exc),
                status_code=404,
                error_type="app_update_apk_file_not_found",
            )

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_APP_UPDATE] Invalid download request: %s", exc)
            return self._error_response(
                str(exc),
                status_code=404,
                error_type="app_update_download_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_APP_UPDATE] APK download lookup failed.")
            return self._error_response(
                "APK download lookup failed.",
                status_code=500,
                error_type="app_update_download_failed",
                details={"exception": str(exc)},
            )

    def report_app_update(
        self,
        payload: dict[str, Any],
    ) -> ResponseTuple:
        """
        Record an app update event from the tablet.

        Route:
            POST /tablet-edge/app-update/report
        """
        logger.info("[TABLET_EDGE_APP_UPDATE] App update report requested.")

        try:
            with self.transaction() as session:
                response_body = self.app_update_service.record_update_report(
                    session=session,
                    payload=payload,
                )

                tablet_device_id = response_body.get("tablet_device_id")

                if tablet_device_id is not None:
                    self.sync_service.record_quick_sync(
                        session=session,
                        tablet_device_id=tablet_device_id,
                        sync_type="app_update_report",
                        sync_direction="tablet_to_server",
                        status=str(response_body.get("event_type", "reported")),
                        records_sent=1,
                        records_received=0,
                        records_failed=0,
                    )

                return self._success_response(response_body, 200)

        except ValueError as exc:
            logger.warning("[TABLET_EDGE_APP_UPDATE] Invalid update report: %s", exc)
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_update_report_validation_error",
            )

        except Exception as exc:
            logger.exception("[TABLET_EDGE_APP_UPDATE] Update report failed.")
            return self._error_response(
                "App update report failed.",
                status_code=500,
                error_type="app_update_report_failed",
                details={"exception": str(exc)},
            )