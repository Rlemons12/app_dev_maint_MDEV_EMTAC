"""
Tablet Wi-Fi/router/access-point tracking service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_wifi_tracking_service.py

Responsibilities:
    - Upsert known Wi-Fi access point/router records.
    - Record tablet Wi-Fi observations.
    - Normalize defensive Wi-Fi/router values when called outside the coordinator.
    - Return SQLAlchemy model instances to the orchestrator.

Important:
    This service does NOT create database sessions.
    This service does NOT commit, rollback, or close sessions.
    The orchestrator owns transaction lifecycle.

Tables:
    tablet_edge.tablet_wifi_access_point
    tablet_edge.tablet_wifi_observation
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import (
    TabletWifiAccessPoint,
    TabletWifiObservation,
)


class TabletWifiTrackingService:
    """
    Service for tracking tablet Wi-Fi/router/access-point information.

    Primary identity:
        bssid

    Human-readable labels:
        router_name
        friendly_name

    Network routing fields:
        router_ip
        gateway_address
    """

    _INT_PATTERN = re.compile(r"-?\d+")

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

    @staticmethod
    def utc_now() -> datetime:
        """
        Return timezone-aware UTC datetime.
        """
        return datetime.now(timezone.utc)

    @staticmethod
    def normalize_string(value: Any, *, max_length: int | None = None) -> str | None:
        """
        Normalize string-ish values.

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

    @classmethod
    def normalize_bssid(cls, value: Any) -> str | None:
        """
        Normalize BSSID values.

        Android may return placeholder values when location/Wi-Fi permissions are
        missing. Those placeholders are not useful as real access-point IDs.
        """
        normalized = cls.normalize_string(value, max_length=64)

        if not normalized:
            return None

        normalized = normalized.replace("-", ":").upper()

        unavailable_values = {
            "00:00:00:00:00:00",
            "02:00:00:00:00:00",
            "UNKNOWN",
            "UNKNOWN_BSSID",
            "<UNKNOWN BSSID>",
            "<UNKNOWN SSID>",
        }

        if normalized in unavailable_values:
            return None

        return normalized

    @classmethod
    def normalize_int(cls, value: Any) -> int | None:
        """
        Normalize integer-ish values.

        Handles values like:
            -61
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

    @staticmethod
    def parse_datetime(value: Any) -> datetime | None:
        """
        Parse datetime values from payloads.

        Accepts:
            datetime object
            ISO string
            ISO string with trailing Z
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)

            return value

        raw_value = str(value).strip()

        if not raw_value:
            return None

        try:
            cleaned = raw_value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)

            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)

            return parsed

        except ValueError:
            logger.warning(
                "[TABLET_WIFI_TRACKING] Could not parse datetime value: %s",
                raw_value,
            )
            return None

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
    def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
        """
        Return the first non-empty value from payload by key list.
        """
        for key in keys:
            value = payload.get(key)

            if value not in (None, ""):
                return value

        return None

    @classmethod
    def _normalize_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Defensive normalization for calls that bypass the coordinator.

        The coordinator should already normalize most values, but this keeps the
        service safe for direct scripts, tests, and older clients.
        """
        normalized: dict[str, Any] = dict(payload)

        normalized["ssid"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "ssid",
                    "SSID",
                    "wifi_ssid",
                    "network_ssid",
                ),
            ),
            max_length=150,
        )

        normalized["bssid"] = cls.normalize_bssid(
            cls._first_present(
                normalized,
                (
                    "bssid",
                    "BSSID",
                    "wifi_bssid",
                    "access_point_bssid",
                    "ap_bssid",
                    "mac_address",
                    "access_point_mac",
                ),
            )
        )

        router_ip = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "router_ip",
                    "routerIp",
                    "router_ip_address",
                    "router_address",
                    "gateway_ip",
                    "gatewayIp",
                    "gateway_address",
                    "gatewayAddress",
                    "default_gateway",
                    "defaultGateway",
                ),
            ),
            max_length=100,
        )

        gateway_address = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "gateway_address",
                    "gatewayAddress",
                    "gateway_ip",
                    "gatewayIp",
                    "default_gateway",
                    "defaultGateway",
                    "router_ip",
                    "routerIp",
                ),
            ),
            max_length=100,
        )

        if not router_ip and gateway_address:
            router_ip = gateway_address

        if not gateway_address and router_ip:
            gateway_address = router_ip

        normalized["router_ip"] = router_ip
        normalized["gateway_address"] = gateway_address

        normalized["router_name"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "router_name",
                    "routerName",
                    "access_point_name",
                    "ap_name",
                    "friendly_name",
                    "wifi_name",
                ),
            ),
            max_length=150,
        )

        normalized["ip_address"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "ip_address",
                    "ipAddress",
                    "tablet_ip",
                    "tabletIp",
                    "local_ip",
                    "localIp",
                    "device_ip",
                    "deviceIp",
                ),
            ),
            max_length=100,
        )

        normalized["dhcp_server_address"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "dhcp_server_address",
                    "dhcpServerAddress",
                    "dhcp_server_ip",
                    "dhcpServerIp",
                    "dhcp_server",
                    "dhcpServer",
                ),
            ),
            max_length=100,
        )

        dns_servers = cls._first_present(
            normalized,
            (
                "dns_servers",
                "dnsServers",
                "dns_server",
                "dnsServer",
                "dns",
            ),
        )

        if isinstance(dns_servers, list):
            dns_servers = ", ".join(
                str(item).strip()
                for item in dns_servers
                if str(item).strip()
            )

        normalized["dns_servers"] = cls.normalize_string(dns_servers)

        normalized["wifi_rssi"] = cls.normalize_int(
            cls._first_present(
                normalized,
                (
                    "wifi_rssi",
                    "wifiRssi",
                    "rssi",
                    "rssi_dbm",
                    "rssiDbm",
                    "wifi_rssi_dbm",
                    "wifiRssiDbm",
                ),
            )
        )

        signal_level = cls.normalize_int(
            cls._first_present(
                normalized,
                (
                    "signal_level",
                    "signalLevel",
                    "wifi_level",
                    "wifiLevel",
                    "level",
                ),
            )
        )

        if signal_level is not None:
            signal_level = max(0, min(signal_level, 4))

        normalized["signal_level"] = signal_level

        normalized["frequency_mhz"] = cls.normalize_int(
            cls._first_present(
                normalized,
                (
                    "frequency_mhz",
                    "frequencyMhz",
                    "frequency",
                    "wifi_frequency",
                    "wifiFrequency",
                    "wifi_frequency_mhz",
                    "wifiFrequencyMhz",
                ),
            )
        )

        wifi_band = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "wifi_band",
                    "wifiBand",
                    "band",
                    "network_band",
                    "networkBand",
                ),
            ),
            max_length=50,
        )

        if not wifi_band:
            wifi_band = cls.infer_wifi_band(normalized["frequency_mhz"])

        normalized["wifi_band"] = wifi_band

        normalized["link_speed_mbps"] = cls.normalize_int(
            cls._first_present(
                normalized,
                (
                    "link_speed_mbps",
                    "linkSpeedMbps",
                    "link_speed",
                    "linkSpeed",
                    "wifi_link_speed",
                    "wifiLinkSpeed",
                    "wifi_link_speed_mbps",
                    "wifiLinkSpeedMbps",
                ),
            )
        )

        normalized["is_online"] = cls.normalize_bool(
            cls._first_present(
                normalized,
                (
                    "is_online",
                    "isOnline",
                    "online",
                    "network_online",
                    "networkOnline",
                ),
            )
        )

        normalized["server_reachable"] = cls.normalize_bool(
            cls._first_present(
                normalized,
                (
                    "server_reachable",
                    "serverReachable",
                    "server_available",
                    "serverAvailable",
                ),
            )
        )

        normalized["server_latency_ms"] = cls.normalize_int(
            cls._first_present(
                normalized,
                (
                    "server_latency_ms",
                    "serverLatencyMs",
                    "server_latency",
                    "serverLatency",
                    "latency_ms",
                    "latencyMs",
                    "latency",
                ),
            )
        )

        normalized["server_url"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "server_url",
                    "serverUrl",
                ),
            )
        )

        normalized["quality_level"] = cls.normalize_string(
            cls._first_present(
                normalized,
                (
                    "quality_level",
                    "qualityLevel",
                    "quality",
                ),
            ),
            max_length=50,
        )

        return normalized

    @classmethod
    def payload_has_network_signal(cls, payload: dict[str, Any]) -> bool:
        """
        Return True when a payload contains useful network/router/AP data.
        """
        normalized = cls._normalize_payload(payload)

        for key in cls.NETWORK_SIGNAL_KEYS:
            value = normalized.get(key)

            if value not in (None, ""):
                return True

        return False

    @classmethod
    def upsert_access_point_from_payload(
        cls,
        *,
        session: Session,
        payload: dict[str, Any],
    ) -> TabletWifiAccessPoint | None:
        """
        Find or create a known Wi-Fi access point/router record.

        Returns None when no usable BSSID is present.

        BSSID is the primary access-point identity. Router IP and router name are
        useful metadata, but they are not unique enough to be the identity.
        """
        normalized = cls._normalize_payload(payload)

        bssid = normalized.get("bssid")

        if not bssid:
            logger.debug(
                "[TABLET_WIFI_TRACKING] No usable BSSID in payload. "
                "Skipping access point upsert."
            )
            return None

        now = cls.utc_now()

        access_point = (
            session.query(TabletWifiAccessPoint)
            .filter(TabletWifiAccessPoint.bssid == bssid)
            .one_or_none()
        )

        if access_point is None:
            access_point = TabletWifiAccessPoint(
                ssid=normalized.get("ssid"),
                bssid=bssid,
                router_ip=normalized.get("router_ip"),
                router_name=normalized.get("router_name"),
                friendly_name=normalized.get("router_name"),
                is_approved=True,
                first_seen_at=now,
                last_seen_at=now,
                created_at=now,
                updated_at=now,
            )

            session.add(access_point)
            session.flush()

            logger.info(
                "[TABLET_WIFI_TRACKING] Created access point id=%s ssid=%s bssid=%s router_ip=%s router_name=%s",
                access_point.id,
                access_point.ssid,
                access_point.bssid,
                access_point.router_ip,
                access_point.router_name,
            )

            return access_point

        changed = False

        if normalized.get("ssid") and access_point.ssid != normalized.get("ssid"):
            access_point.ssid = normalized.get("ssid")
            changed = True

        if normalized.get("router_ip") and access_point.router_ip != normalized.get(
            "router_ip"
        ):
            access_point.router_ip = normalized.get("router_ip")
            changed = True

        if normalized.get("router_name") and access_point.router_name != normalized.get(
            "router_name"
        ):
            access_point.router_name = normalized.get("router_name")
            changed = True

            if not access_point.friendly_name:
                access_point.friendly_name = normalized.get("router_name")

        access_point.last_seen_at = now
        access_point.updated_at = now

        session.flush()

        if changed:
            logger.info(
                "[TABLET_WIFI_TRACKING] Updated access point id=%s ssid=%s bssid=%s router_ip=%s router_name=%s",
                access_point.id,
                access_point.ssid,
                access_point.bssid,
                access_point.router_ip,
                access_point.router_name,
            )
        else:
            logger.debug(
                "[TABLET_WIFI_TRACKING] Access point seen id=%s bssid=%s",
                access_point.id,
                access_point.bssid,
            )

        return access_point

    @classmethod
    def record_observation_from_payload(
        cls,
        *,
        session: Session,
        tablet_device_id: int,
        payload: dict[str, Any],
        source: str | None = None,
    ) -> TabletWifiObservation | None:
        """
        Record a Wi-Fi/router observation for a tablet.

        This records even when BSSID is unavailable, as long as some useful
        network data exists. If BSSID is available, this also links the
        observation to tablet_wifi_access_point through access_point_id.
        """
        normalized = cls._normalize_payload(payload)

        if not cls.payload_has_network_signal(normalized):
            logger.debug(
                "[TABLET_WIFI_TRACKING] No network signal data in payload. "
                "Skipping Wi-Fi observation. source=%s tablet_device_id=%s",
                source,
                tablet_device_id,
            )
            return None

        access_point = cls.upsert_access_point_from_payload(
            session=session,
            payload=normalized,
        )

        sampled_at = (
            cls.parse_datetime(normalized.get("sampled_at"))
            or cls.parse_datetime(normalized.get("client_sampled_at"))
            or cls.parse_datetime(normalized.get("client_created_at"))
            or cls.utc_now()
        )

        observation = TabletWifiObservation(
            tablet_device_id=tablet_device_id,
            access_point_id=access_point.id if access_point is not None else None,
            sampled_at=sampled_at,
            is_online=normalized.get("is_online"),
            ssid=normalized.get("ssid"),
            bssid=normalized.get("bssid"),
            router_ip=normalized.get("router_ip"),
            router_name=normalized.get("router_name"),
            ip_address=normalized.get("ip_address"),
            gateway_address=normalized.get("gateway_address"),
            dhcp_server_address=normalized.get("dhcp_server_address"),
            dns_servers=normalized.get("dns_servers"),
            wifi_rssi=normalized.get("wifi_rssi"),
            signal_level=normalized.get("signal_level"),
            frequency_mhz=normalized.get("frequency_mhz"),
            wifi_band=normalized.get("wifi_band"),
            link_speed_mbps=normalized.get("link_speed_mbps"),
            server_url=normalized.get("server_url"),
            server_reachable=normalized.get("server_reachable"),
            server_latency_ms=normalized.get("server_latency_ms"),
            quality_level=normalized.get("quality_level"),
            created_at=cls.utc_now(),
        )

        session.add(observation)
        session.flush()

        logger.info(
            "[TABLET_WIFI_TRACKING] Recorded Wi-Fi observation id=%s tablet_device_id=%s source=%s ssid=%s bssid=%s rssi=%s signal_level=%s access_point_id=%s",
            observation.id,
            tablet_device_id,
            source,
            observation.ssid,
            observation.bssid,
            observation.wifi_rssi,
            observation.signal_level,
            observation.access_point_id,
        )

        return observation

    @classmethod
    def build_access_point_summary(
        cls,
        access_point: TabletWifiAccessPoint | None,
    ) -> dict[str, Any] | None:
        """
        Build response-safe access point summary.
        """
        if access_point is None:
            return None

        return access_point.to_dict()

    @classmethod
    def build_observation_summary(
        cls,
        observation: TabletWifiObservation | None,
    ) -> dict[str, Any] | None:
        """
        Build response-safe observation summary.
        """
        if observation is None:
            return None

        return observation.to_dict()


__all__ = [
    "TabletWifiTrackingService",
]