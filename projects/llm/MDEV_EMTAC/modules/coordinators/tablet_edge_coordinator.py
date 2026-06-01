"""
Tablet Edge coordinator for the EMTAC Tablet Edge Agent.

File:
    modules/coordinators/tablet_edge_coordinator.py

Responsibilities:
    - Normalize incoming route/request data.
    - Validate that payloads are JSON objects where required.
    - Normalize Wi-Fi/router/access-point fields from tablet payloads.
    - Delegate business/database work to TabletEdgeOrchestrator.
    - Return route-friendly tuples:
        (success: bool, response_body: dict, status_code: int)

Important:
    The coordinator does NOT create database sessions.
    The coordinator does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.

Router/AP tracking notes:
    The database now supports:

        tablet_edge.tablet_wifi_access_point
        tablet_edge.tablet_wifi_observation

    And these network fields on health/network records:

        ssid
        bssid
        router_ip
        router_name
        ip_address
        gateway_address
        dhcp_server_address
        dns_servers
        wifi_rssi
        signal_level
        frequency_mhz
        wifi_band
        link_speed_mbps

    This coordinator normalizes those fields only. The orchestrator/service layer
    still owns database inserts/updates.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from modules.configuration.log_config import logger
from modules.orchestrators.tablet_edge_orchestrator import TabletEdgeOrchestrator


ResponseTuple = tuple[bool, dict[str, Any], int]


class TabletEdgeCoordinator:
    """
    Coordinator for Tablet Edge HTTP-facing workflows.

    Pattern:

        Blueprint
            ↓
        Coordinator
            ↓
        Orchestrator
            ↓
        Service
            ↓
        Database
    """

    _INT_PATTERN = re.compile(r"-?\d+")

    STRING_NORMALIZATION_FIELDS: tuple[str, ...] = (
        "tablet_uid",
        "tablet_name",
        "device_make",
        "device_model",
        "android_version",
        "app_version",
        "assigned_area",
        "assigned_station",
        "assigned_role",
        "event_type",
        "quality_level",
        "server_url",
        "page_url",
        "current_page_url",
        "message",
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "ip_address",
        "gateway_address",
        "dhcp_server_address",
        "dns_servers",
        "wifi_band",
        "log_level",
        "log_source",
        "cache_name",
        "cache_version",
        "sync_type",
        "sync_direction",
        "status",
        "error_message",
        "processing_status",
    )

    INTEGER_NORMALIZATION_FIELDS: tuple[str, ...] = (
        "latency_ms",
        "avg_latency_ms",
        "consecutive_failures",
        "wifi_rssi",
        "signal_level",
        "frequency_mhz",
        "link_speed_mbps",
        "server_latency_ms",
        "battery_percent",
        "records_sent",
        "records_received",
        "records_failed",
        "duration_ms",
        "record_count",
    )

    BOOLEAN_NORMALIZATION_FIELDS: tuple[str, ...] = (
        "is_online",
        "server_reachable",
        "is_charging",
        "app_foreground",
        "is_active",
    )

    NETWORK_CONTEXT_KEYS: tuple[str, ...] = (
        "wifi",
        "network",
        "router",
        "access_point",
        "connection",
        "network_info",
        "wifi_info",
    )

    NETWORK_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "ssid": (
            "SSID",
            "wifi_ssid",
            "network_ssid",
        ),
        "bssid": (
            "BSSID",
            "wifi_bssid",
            "access_point_bssid",
            "ap_bssid",
            "mac_address",
            "access_point_mac",
        ),
        "router_ip": (
            "routerIp",
            "router_ip_address",
            "router_address",
            "gateway_ip",
            "gatewayIp",
            "default_gateway",
            "defaultGateway",
        ),
        "router_name": (
            "routerName",
            "access_point_name",
            "ap_name",
            "friendly_name",
            "wifi_name",
        ),
        "ip_address": (
            "ipAddress",
            "tablet_ip",
            "tabletIp",
            "local_ip",
            "localIp",
            "device_ip",
            "deviceIp",
        ),
        "gateway_address": (
            "gatewayAddress",
            "gateway_ip",
            "gatewayIp",
            "default_gateway",
            "defaultGateway",
            "router_ip",
            "routerIp",
        ),
        "dhcp_server_address": (
            "dhcpServerAddress",
            "dhcp_server_ip",
            "dhcpServerIp",
            "dhcp_server",
            "dhcpServer",
        ),
        "dns_servers": (
            "dnsServers",
            "dns_server",
            "dnsServer",
            "dns",
        ),
        "wifi_rssi": (
            "wifiRssi",
            "rssi",
            "rssi_dbm",
            "rssiDbm",
            "wifi_rssi_dbm",
            "wifiRssiDbm",
        ),
        "signal_level": (
            "signalLevel",
            "wifi_level",
            "wifiLevel",
            "level",
        ),
        "frequency_mhz": (
            "frequencyMhz",
            "frequency",
            "wifi_frequency",
            "wifiFrequency",
            "wifi_frequency_mhz",
            "wifiFrequencyMhz",
        ),
        "wifi_band": (
            "wifiBand",
            "band",
            "network_band",
            "networkBand",
        ),
        "link_speed_mbps": (
            "linkSpeedMbps",
            "link_speed",
            "linkSpeed",
            "wifi_link_speed",
            "wifiLinkSpeed",
            "wifi_link_speed_mbps",
            "wifiLinkSpeedMbps",
        ),
        "server_latency_ms": (
            "serverLatencyMs",
            "server_latency",
            "serverLatency",
        ),
        "latency_ms": (
            "latencyMs",
            "latency",
        ),
        "avg_latency_ms": (
            "avgLatencyMs",
            "average_latency_ms",
            "averageLatencyMs",
        ),
        "consecutive_failures": (
            "consecutiveFailures",
            "failure_count",
            "failureCount",
        ),
        "is_online": (
            "isOnline",
            "online",
            "network_online",
            "networkOnline",
        ),
        "server_reachable": (
            "serverReachable",
            "server_available",
            "serverAvailable",
        ),
        "app_foreground": (
            "appForeground",
            "is_app_foreground",
            "isAppForeground",
        ),
        "is_charging": (
            "isCharging",
            "charging",
        ),
        "battery_percent": (
            "batteryPercent",
            "battery",
        ),
    }

    def __init__(self, orchestrator: TabletEdgeOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or TabletEdgeOrchestrator()

    @staticmethod
    def utc_now_iso() -> str:
        """
        Return current UTC time as an ISO string.
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _error_response(
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "tablet_edge_coordinator_error",
        details: dict[str, Any] | None = None,
    ) -> ResponseTuple:
        """
        Build a route-friendly error tuple.
        """
        response_body: dict[str, Any] = {
            "success": False,
            "error": message,
            "error_type": error_type,
            "server_time_utc": TabletEdgeCoordinator.utc_now_iso(),
        }

        if details:
            response_body["details"] = details

        return False, response_body, status_code

    @staticmethod
    def ensure_dict_payload(payload: Any) -> dict[str, Any]:
        """
        Ensure incoming request payload is a JSON object/dict.

        Flask's request.get_json(silent=True) can return None.
        Routes should normally pass {} in that case, but this keeps the
        coordinator defensive.
        """
        if payload is None:
            return {}

        if not isinstance(payload, dict):
            raise ValueError("Request payload must be a JSON object.")

        return payload

    @staticmethod
    def normalize_string(value: Any, *, max_length: int | None = None) -> str | None:
        """
        Normalize string-ish request values.

        Empty strings become None.
        """
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        if max_length is not None and len(normalized) > max_length:
            normalized = normalized[:max_length]

        return normalized

    @staticmethod
    def normalize_optional_query_value(value: Any) -> str | None:
        """
        Normalize optional query string values.
        """
        return TabletEdgeCoordinator.normalize_string(value)

    @classmethod
    def normalize_int(cls, value: Any) -> int | None:
        """
        Normalize integer-ish values.

        Handles Android/client strings like:
            "-61 dBm"
            "35ms"
            "72 Mbps"
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(round(value))

        normalized = str(value).strip()

        if not normalized:
            return None

        match = cls._INT_PATTERN.search(normalized)

        if not match:
            return None

        try:
            return int(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def normalize_bool(value: Any) -> bool | None:
        """
        Normalize boolean-ish values.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value != 0

        normalized = str(value).strip().lower()

        if not normalized:
            return None

        if normalized in {"1", "true", "t", "yes", "y", "online", "on"}:
            return True

        if normalized in {"0", "false", "f", "no", "n", "offline", "off"}:
            return False

        return None

    @classmethod
    def normalize_bssid(cls, value: Any) -> str | None:
        """
        Normalize BSSID values.

        Android may return placeholder values when location/Wi-Fi permissions are
        missing. Those placeholders are not useful as real access-point IDs, so
        they are treated as unavailable.
        """
        normalized = cls.normalize_string(value, max_length=64)

        if not normalized:
            return None

        normalized = normalized.strip().replace("-", ":").upper()

        unavailable_values = {
            "00:00:00:00:00:00",
            "02:00:00:00:00:00",
            "UNKNOWN",
            "UNKNOWN_BSSID",
            "<UNKNOWN BSSID>",
        }

        if normalized in unavailable_values:
            return None

        return normalized

    @staticmethod
    def infer_wifi_band(frequency_mhz: int | None) -> str | None:
        """
        Infer Wi-Fi band from frequency in MHz.
        """
        if frequency_mhz is None:
            return None

        if 2400 <= frequency_mhz < 2500:
            return "2.4GHz"

        if 4900 <= frequency_mhz < 5925:
            return "5GHz"

        if 5925 <= frequency_mhz <= 7125:
            return "6GHz"

        return None

    @staticmethod
    def _copy_alias_if_missing(
        payload: dict[str, Any],
        canonical_key: str,
        alias_keys: tuple[str, ...],
    ) -> None:
        """
        Copy a known alias value to a canonical key when the canonical key is
        missing or empty.
        """
        current_value = payload.get(canonical_key)

        if current_value not in (None, ""):
            return

        for alias_key in alias_keys:
            alias_value = payload.get(alias_key)

            if alias_value not in (None, ""):
                payload[canonical_key] = alias_value
                return

    @classmethod
    def _merge_nested_network_context(cls, payload: dict[str, Any]) -> None:
        """
        Merge nested Wi-Fi/network/router dictionaries into the current payload.

        Example accepted shapes:

            {
                "tablet_uid": "...",
                "wifi": {
                    "ssid": "...",
                    "bssid": "...",
                    "rssi": -61
                }
            }

            {
                "tablet_uid": "...",
                "network": {
                    "gatewayIp": "192.168.1.1"
                }
            }

        Existing top-level keys win.
        """
        for context_key in cls.NETWORK_CONTEXT_KEYS:
            nested_value = payload.get(context_key)

            if not isinstance(nested_value, dict):
                continue

            for nested_key, nested_item_value in nested_value.items():
                if nested_key not in payload or payload.get(nested_key) in (None, ""):
                    payload[nested_key] = nested_item_value

    @classmethod
    def normalize_network_fields(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize common network/router/access-point fields in a payload.

        This method is intentionally tolerant. It accepts snake_case and several
        common Android/camelCase variants, then adds canonical snake_case keys
        for the orchestrator/service layer.
        """
        normalized_payload = dict(payload)

        cls._merge_nested_network_context(normalized_payload)

        for canonical_key, alias_keys in cls.NETWORK_FIELD_ALIASES.items():
            cls._copy_alias_if_missing(
                normalized_payload,
                canonical_key,
                alias_keys,
            )

        for field_name in cls.STRING_NORMALIZATION_FIELDS:
            if field_name in normalized_payload:
                max_length = 64 if field_name == "bssid" else None
                normalized_payload[field_name] = cls.normalize_string(
                    normalized_payload.get(field_name),
                    max_length=max_length,
                )

        if "bssid" in normalized_payload:
            normalized_payload["bssid"] = cls.normalize_bssid(
                normalized_payload.get("bssid")
            )

        for field_name in cls.INTEGER_NORMALIZATION_FIELDS:
            if field_name in normalized_payload:
                normalized_payload[field_name] = cls.normalize_int(
                    normalized_payload.get(field_name)
                )

        for field_name in cls.BOOLEAN_NORMALIZATION_FIELDS:
            if field_name in normalized_payload:
                normalized_payload[field_name] = cls.normalize_bool(
                    normalized_payload.get(field_name)
                )

        signal_level = normalized_payload.get("signal_level")
        if isinstance(signal_level, int):
            normalized_payload["signal_level"] = max(0, min(signal_level, 4))

        frequency_mhz = normalized_payload.get("frequency_mhz")
        wifi_band = cls.normalize_string(normalized_payload.get("wifi_band"))

        if not wifi_band:
            wifi_band = cls.infer_wifi_band(frequency_mhz)

        normalized_payload["wifi_band"] = wifi_band

        gateway_address = cls.normalize_string(normalized_payload.get("gateway_address"))
        router_ip = cls.normalize_string(normalized_payload.get("router_ip"))

        if not router_ip and gateway_address:
            normalized_payload["router_ip"] = gateway_address

        if not gateway_address and router_ip:
            normalized_payload["gateway_address"] = router_ip

        dns_servers = normalized_payload.get("dns_servers")
        if isinstance(dns_servers, list):
            normalized_payload["dns_servers"] = ", ".join(
                str(item).strip()
                for item in dns_servers
                if str(item).strip()
            )

        return normalized_payload

    @classmethod
    def _inherit_top_level_network_fields(
        cls,
        *,
        top_level_payload: dict[str, Any],
        item_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Copy top-level tablet/network fields into a list item when missing.

        This keeps payloads flexible. The tablet can send either:

            {
                "tablet_uid": "...",
                "ssid": "...",
                "bssid": "...",
                "events": [...]
            }

        or:

            {
                "events": [
                    {
                        "tablet_uid": "...",
                        "ssid": "...",
                        "bssid": "..."
                    }
                ]
            }
        """
        merged_item = dict(item_payload)

        inheritable_keys = (
            "tablet_uid",
            "tablet_name",
            "device_make",
            "device_model",
            "android_version",
            "app_version",
            "assigned_area",
            "assigned_station",
            "assigned_role",
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
        )

        for key in inheritable_keys:
            if merged_item.get(key) in (None, "") and top_level_payload.get(key) not in (
                None,
                "",
            ):
                merged_item[key] = top_level_payload.get(key)

        return cls.normalize_network_fields(merged_item)

    @classmethod
    def normalize_network_collection_payload(
        cls,
        payload: dict[str, Any],
        *,
        collection_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        Normalize a payload that may contain a collection of network-style items.

        Supported examples:
            {"events": [...]}
            {"network_events": [...]}
            {"samples": [...]}
            {"health_samples": [...]}

        If no collection key exists, the top-level payload is normalized.
        """
        normalized_payload = cls.normalize_network_fields(deepcopy(payload))

        for collection_key in collection_keys:
            collection_value = normalized_payload.get(collection_key)

            if collection_value is None:
                continue

            if not isinstance(collection_value, list):
                raise ValueError(f"Payload field '{collection_key}' must be a list.")

            normalized_items: list[dict[str, Any]] = []

            for index, item in enumerate(collection_value):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Payload field '{collection_key}' item #{index} "
                        "must be a JSON object."
                    )

                normalized_items.append(
                    cls._inherit_top_level_network_fields(
                        top_level_payload=normalized_payload,
                        item_payload=item,
                    )
                )

            normalized_payload[collection_key] = normalized_items

        return normalized_payload

    def register_tablet(self, payload: Any) -> ResponseTuple:
        """
        Coordinate tablet registration.

        Route:
            POST /tablet-edge/register
        """
        logger.info("[TABLET_EDGE_COORDINATOR] register_tablet requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)
            normalized_payload = self.normalize_network_fields(safe_payload)

            return self.orchestrator.register_tablet(normalized_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] register_tablet validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="register_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] register_tablet failed unexpectedly."
            )
            return self._error_response(
                "Tablet registration request failed.",
                status_code=500,
                error_type="register_coordinator_failed",
                details={"exception": str(exc)},
            )

    def heartbeat(self, payload: Any) -> ResponseTuple:
        """
        Coordinate tablet heartbeat.

        Route:
            POST /tablet-edge/heartbeat

        Note:
            Heartbeat payloads may now include Wi-Fi/router/AP fields. The
            orchestrator can use those fields to update last_seen_at and/or
            record a lightweight Wi-Fi observation.
        """
        logger.info("[TABLET_EDGE_COORDINATOR] heartbeat requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)
            normalized_payload = self.normalize_network_fields(safe_payload)

            return self.orchestrator.heartbeat(normalized_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] heartbeat validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="heartbeat_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] heartbeat failed unexpectedly."
            )
            return self._error_response(
                "Tablet heartbeat request failed.",
                status_code=500,
                error_type="heartbeat_coordinator_failed",
                details={"exception": str(exc)},
            )

    def record_network_events(self, payload: Any) -> ResponseTuple:
        """
        Coordinate tablet network event submission.

        Route:
            POST /tablet-edge/network-events

        Supported payload shapes:

            {
                "tablet_uid": "...",
                "events": [
                    {
                        "event_type": "server_health_good",
                        "quality_level": "good",
                        "latency_ms": 35,
                        "ssid": "...",
                        "bssid": "...",
                        "gateway_address": "192.168.1.1"
                    }
                ]
            }

            {
                "tablet_uid": "...",
                "event_type": "server_health_good",
                "quality_level": "good",
                "ssid": "...",
                "bssid": "..."
            }
        """
        logger.info("[TABLET_EDGE_COORDINATOR] record_network_events requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)
            normalized_payload = self.normalize_network_collection_payload(
                safe_payload,
                collection_keys=("events", "network_events"),
            )

            return self.orchestrator.record_network_events(normalized_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] network event payload validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="network_events_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] record_network_events failed unexpectedly."
            )
            return self._error_response(
                "Network event request failed.",
                status_code=500,
                error_type="network_events_coordinator_failed",
                details={"exception": str(exc)},
            )

    def record_health_samples(self, payload: Any) -> ResponseTuple:
        """
        Coordinate tablet health sample submission.

        Route:
            POST /tablet-edge/health-samples

        Supported payload shapes:

            {
                "tablet_uid": "...",
                "samples": [
                    {
                        "server_reachable": true,
                        "server_latency_ms": 35,
                        "quality_level": "good",
                        "ssid": "...",
                        "bssid": "...",
                        "gateway_address": "192.168.1.1"
                    }
                ]
            }

            {
                "tablet_uid": "...",
                "server_reachable": true,
                "quality_level": "good",
                "ssid": "...",
                "bssid": "..."
            }
        """
        logger.info("[TABLET_EDGE_COORDINATOR] record_health_samples requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)
            normalized_payload = self.normalize_network_collection_payload(
                safe_payload,
                collection_keys=("samples", "health_samples"),
            )

            return self.orchestrator.record_health_samples(normalized_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] health sample payload validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="health_samples_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] record_health_samples failed unexpectedly."
            )
            return self._error_response(
                "Health sample request failed.",
                status_code=500,
                error_type="health_samples_coordinator_failed",
                details={"exception": str(exc)},
            )

    def get_dropdown_cache_status(
        self,
        tablet_uid: str | None = None,
    ) -> ResponseTuple:
        """
        Coordinate dropdown cache status request.

        Route:
            GET /tablet-edge/dropdown-cache/status

        Query parameter:
            tablet_uid optional
        """
        logger.info("[TABLET_EDGE_COORDINATOR] get_dropdown_cache_status requested.")

        try:
            safe_tablet_uid = self.normalize_optional_query_value(tablet_uid)

            return self.orchestrator.get_dropdown_cache_status(
                tablet_uid=safe_tablet_uid
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] get_dropdown_cache_status failed unexpectedly."
            )
            return self._error_response(
                "Dropdown cache status request failed.",
                status_code=500,
                error_type="dropdown_cache_status_coordinator_failed",
                details={"exception": str(exc)},
            )

    def get_full_dropdown_cache(
        self,
        tablet_uid: str | None = None,
    ) -> ResponseTuple:
        """
        Coordinate full dropdown cache request.

        Route:
            GET /tablet-edge/dropdown-cache/full

        Query parameter:
            tablet_uid optional
        """
        logger.info("[TABLET_EDGE_COORDINATOR] get_full_dropdown_cache requested.")

        try:
            safe_tablet_uid = self.normalize_optional_query_value(tablet_uid)

            return self.orchestrator.get_full_dropdown_cache(
                tablet_uid=safe_tablet_uid
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] get_full_dropdown_cache failed unexpectedly."
            )
            return self._error_response(
                "Full dropdown cache request failed.",
                status_code=500,
                error_type="dropdown_cache_full_coordinator_failed",
                details={"exception": str(exc)},
            )

    def get_delta_dropdown_cache(
        self,
        since: str | None = None,
    ) -> ResponseTuple:
        """
        Coordinate delta dropdown cache request.

        Route:
            GET /tablet-edge/dropdown-cache/delta?since=<timestamp>
        """
        logger.info("[TABLET_EDGE_COORDINATOR] get_delta_dropdown_cache requested.")

        try:
            safe_since = self.normalize_optional_query_value(since)

            return self.orchestrator.get_delta_dropdown_cache(
                since=safe_since
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] get_delta_dropdown_cache failed unexpectedly."
            )
            return self._error_response(
                "Delta dropdown cache request failed.",
                status_code=500,
                error_type="dropdown_cache_delta_coordinator_failed",
                details={"exception": str(exc)},
            )

    def sync_offline_events(self, payload: Any) -> ResponseTuple:
        """
        Coordinate offline event sync.

        Route:
            POST /tablet-edge/offline-events/sync
        """
        logger.info("[TABLET_EDGE_COORDINATOR] sync_offline_events requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)

            return self.orchestrator.sync_offline_events(safe_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] offline event payload validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="offline_events_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] sync_offline_events failed unexpectedly."
            )
            return self._error_response(
                "Offline event sync request failed.",
                status_code=500,
                error_type="offline_events_coordinator_failed",
                details={"exception": str(exc)},
            )

    def record_app_logs(self, payload: Any) -> ResponseTuple:
        """
        Coordinate tablet app log submission.

        Route:
            POST /tablet-edge/app-logs
        """
        logger.info("[TABLET_EDGE_COORDINATOR] record_app_logs requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)

            return self.orchestrator.record_app_logs(safe_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] app log payload validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_logs_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] record_app_logs failed unexpectedly."
            )
            return self._error_response(
                "App log request failed.",
                status_code=500,
                error_type="app_logs_coordinator_failed",
                details={"exception": str(exc)},
            )

    def check_app_update(
        self,
        *,
        tablet_uid: Any = None,
        version_code: Any = None,
        version_name: Any = None,
        app_package: Any = None,
        release_channel: Any = None,
    ) -> ResponseTuple:
        """
        Coordinate tablet app update check.

        Route:
            GET /tablet-edge/app-update/check
        """
        logger.info("[TABLET_EDGE_COORDINATOR] check_app_update requested.")

        try:
            safe_tablet_uid = self.normalize_optional_query_value(tablet_uid)
            safe_version_name = self.normalize_optional_query_value(version_name)
            safe_app_package = self.normalize_optional_query_value(app_package)
            safe_release_channel = self.normalize_optional_query_value(release_channel)

            return self.orchestrator.check_app_update(
                tablet_uid=safe_tablet_uid,
                current_version_code=version_code,
                current_version_name=safe_version_name,
                app_package=safe_app_package,
                release_channel=safe_release_channel,
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] check_app_update failed unexpectedly."
            )
            return self._error_response(
                "App update check request failed.",
                status_code=500,
                error_type="app_update_check_coordinator_failed",
                details={"exception": str(exc)},
            )

    def get_app_update_download_file(
        self,
        *,
        release_id: Any,
    ) -> ResponseTuple:
        """
        Coordinate APK download lookup.

        Route:
            GET /tablet-edge/app-update/download/<release_id>
        """
        logger.info("[TABLET_EDGE_COORDINATOR] get_app_update_download_file requested.")

        try:
            try:
                safe_release_id = int(release_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("release_id must be an integer.") from exc

            return self.orchestrator.get_app_update_download_file(
                release_id=safe_release_id,
            )

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] APK download validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_update_download_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] get_app_update_download_file failed unexpectedly."
            )
            return self._error_response(
                "APK download request failed.",
                status_code=500,
                error_type="app_update_download_coordinator_failed",
                details={"exception": str(exc)},
            )

    def report_app_update(
        self,
        payload: Any,
    ) -> ResponseTuple:
        """
        Coordinate tablet app update event report.

        Route:
            POST /tablet-edge/app-update/report
        """
        logger.info("[TABLET_EDGE_COORDINATOR] report_app_update requested.")

        try:
            safe_payload = self.ensure_dict_payload(payload)

            return self.orchestrator.report_app_update(safe_payload)

        except ValueError as exc:
            logger.warning(
                "[TABLET_EDGE_COORDINATOR] update report payload validation failed: %s",
                exc,
            )
            return self._error_response(
                str(exc),
                status_code=400,
                error_type="app_update_report_payload_error",
            )

        except Exception as exc:
            logger.exception(
                "[TABLET_EDGE_COORDINATOR] report_app_update failed unexpectedly."
            )
            return self._error_response(
                "App update report request failed.",
                status_code=500,
                error_type="app_update_report_coordinator_failed",
                details={"exception": str(exc)},
            )