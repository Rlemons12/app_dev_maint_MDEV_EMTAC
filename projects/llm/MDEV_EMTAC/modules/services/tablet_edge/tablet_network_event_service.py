"""
Tablet network event service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_network_event_service.py

Responsibilities:
    - Validate network event payloads.
    - Normalize quality levels and event types.
    - Normalize Wi-Fi/router/access-point fields.
    - Batch insert tablet network events.
    - Query recent tablet network events.

Important:
    This service does NOT create database sessions.
    This service does NOT commit or rollback.
    The orchestrator owns transaction lifecycle.

Router/AP tracking fields supported:
    access_point_id
    ssid
    bssid
    router_ip
    router_name
    wifi_rssi
    signal_level
    ip_address
    gateway_address
    dhcp_server_address
    dns_servers
    frequency_mhz
    wifi_band
    link_speed_mbps
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from modules.configuration.log_config import logger
from modules.emtacdb.tablet_edge.tablet_edge_models import TabletNetworkEvent


class TabletNetworkEventService:
    """
    Domain service for tablet_edge.tablet_network_event.
    """

    VALID_EVENT_TYPES = {
        "server_health_good",
        "server_health_slow",
        "server_timeout",
        "server_unreachable",
        "server_reachable",
        "offline_detected",
        "online_restored",
        "weak_wifi_detected",
        "poor_wifi_detected",
        "critical_connection_detected",
        "wifi_signal_poor",
        "wifi_signal_recovered",
        "tablet_offline",
        "tablet_online",
        "sync_failed",
        "sync_restored",
        "heartbeat",
        "network_sample",
        "network_observation",
    }

    VALID_QUALITY_LEVELS = {
        "good",
        "fair",
        "poor",
        "critical",
        "offline",
        "unknown",
    }

    DEFAULT_MAX_BATCH_SIZE = 500

    _INT_PATTERN = re.compile(r"-?\d+")

    @staticmethod
    def normalize_string(
        value: Any,
        *,
        max_length: int | None = None,
        allow_empty: bool = False,
    ) -> str | None:
        """
        Normalize optional string values.

        Empty strings become None unless allow_empty=True.
        """
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized and not allow_empty:
            return None

        if max_length is not None and len(normalized) > max_length:
            normalized = normalized[:max_length]

        return normalized

    @classmethod
    def normalize_int(
        cls,
        value: Any,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int | None:
        """
        Normalize optional integer values.

        Accepts:
            35
            35.0
            "35"
            "35ms"
            "-61 dBm"
            "72 Mbps"

        Returns None for blank values.
        Raises ValueError for invalid numeric values.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            normalized = int(value)

        elif isinstance(value, int):
            normalized = value

        elif isinstance(value, float):
            normalized = int(round(value))

        else:
            raw_value = str(value).strip()

            if not raw_value:
                return None

            match = cls._INT_PATTERN.search(raw_value)

            if not match:
                raise ValueError(f"Invalid integer value: {value!r}")

            try:
                normalized = int(match.group(0))
            except Exception as exc:
                raise ValueError(f"Invalid integer value: {value!r}") from exc

        if min_value is not None and normalized < min_value:
            normalized = min_value

        if max_value is not None and normalized > max_value:
            normalized = max_value

        return normalized

    @staticmethod
    def normalize_bool(value: Any) -> bool | None:
        """
        Normalize optional boolean values.

        Supports real booleans and common string representations.
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

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
            "online",
            "reachable",
            "on",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "offline",
            "unreachable",
            "off",
        }:
            return False

        raise ValueError(f"Invalid boolean value: {value!r}")

    @staticmethod
    def normalize_datetime(value: Any) -> datetime | None:
        """
        Normalize optional datetime values.

        Accepts:
            - datetime objects
            - ISO strings
            - ISO strings ending in Z
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)

            return value

        normalized = str(value).strip()

        if not normalized:
            return None

        try:
            normalized = normalized.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except Exception as exc:
            raise ValueError(f"Invalid datetime value: {value!r}") from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    @classmethod
    def normalize_bssid(cls, value: Any) -> str | None:
        """
        Normalize BSSID values.

        Android may return placeholder values when Wi-Fi/location permission is
        unavailable. Those values are not useful for AP identity.
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

    def normalize_event_type(self, value: Any) -> str:
        """
        Normalize event_type.

        Unknown event types are allowed but logged. This keeps the tablet
        client forward-compatible while still making unexpected events visible.
        """
        event_type = self.normalize_string(value, max_length=100)

        if not event_type:
            raise ValueError("event_type is required.")

        event_type = event_type.lower()

        if event_type not in self.VALID_EVENT_TYPES:
            logger.warning(
                "[TABLET_EDGE_NETWORK] Unknown network event_type received: %s",
                event_type,
            )

        return event_type

    def normalize_quality_level(self, value: Any) -> str:
        """
        Normalize quality_level.

        Invalid quality values become 'unknown' and are logged.
        """
        quality_level = self.normalize_string(value, max_length=50)

        if not quality_level:
            return "unknown"

        quality_level = quality_level.lower()

        if quality_level not in self.VALID_QUALITY_LEVELS:
            logger.warning(
                "[TABLET_EDGE_NETWORK] Unknown quality_level received: %s",
                quality_level,
            )
            return "unknown"

        return quality_level

    def normalize_dns_servers(self, value: Any) -> str | None:
        """
        Normalize DNS server payload.

        Accepts:
            "192.168.1.1, 8.8.8.8"
            ["192.168.1.1", "8.8.8.8"]
        """
        if value is None:
            return None

        if isinstance(value, list):
            normalized_list = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

            if not normalized_list:
                return None

            return ", ".join(normalized_list)

        return self.normalize_string(value)

    def validate_event_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize one network event payload.

        Required:
            event_type

        Optional:
            quality_level
            server_url
            page_url
            latency_ms
            avg_latency_ms
            consecutive_failures
            is_online
            access_point_id
            ssid
            bssid
            router_ip
            router_name
            wifi_rssi
            signal_level
            ip_address
            gateway_address
            dhcp_server_address
            dns_servers
            frequency_mhz
            wifi_band
            link_speed_mbps
            message
            event_started_at
            event_ended_at
        """
        if not isinstance(event, dict):
            raise ValueError("Each network event must be a JSON object.")

        frequency_mhz = self.normalize_int(
            event.get("frequency_mhz"),
            min_value=0,
        )

        wifi_band = self.normalize_string(
            event.get("wifi_band"),
            max_length=50,
        )

        if not wifi_band:
            wifi_band = self.infer_wifi_band(frequency_mhz)

        router_ip = self.normalize_string(event.get("router_ip"), max_length=100)
        gateway_address = self.normalize_string(
            event.get("gateway_address"),
            max_length=100,
        )

        if not router_ip and gateway_address:
            router_ip = gateway_address

        if not gateway_address and router_ip:
            gateway_address = router_ip

        return {
            "access_point_id": self.normalize_int(
                event.get("access_point_id"),
                min_value=1,
            ),
            "event_type": self.normalize_event_type(event.get("event_type")),
            "quality_level": self.normalize_quality_level(event.get("quality_level")),
            "server_url": self.normalize_string(event.get("server_url")),
            "page_url": self.normalize_string(event.get("page_url")),
            "latency_ms": self.normalize_int(event.get("latency_ms"), min_value=0),
            "avg_latency_ms": self.normalize_int(
                event.get("avg_latency_ms"),
                min_value=0,
            ),
            "consecutive_failures": self.normalize_int(
                event.get("consecutive_failures", 0),
                min_value=0,
            ) or 0,
            "is_online": self.normalize_bool(event.get("is_online")),
            "ssid": self.normalize_string(event.get("ssid"), max_length=150),
            "bssid": self.normalize_bssid(event.get("bssid")),
            "router_ip": router_ip,
            "router_name": self.normalize_string(
                event.get("router_name"),
                max_length=150,
            ),
            "wifi_rssi": self.normalize_int(
                event.get("wifi_rssi"),
                min_value=-150,
                max_value=0,
            ),
            "signal_level": self.normalize_int(
                event.get("signal_level"),
                min_value=0,
                max_value=4,
            ),
            "ip_address": self.normalize_string(
                event.get("ip_address"),
                max_length=100,
            ),
            "gateway_address": gateway_address,
            "dhcp_server_address": self.normalize_string(
                event.get("dhcp_server_address"),
                max_length=100,
            ),
            "dns_servers": self.normalize_dns_servers(event.get("dns_servers")),
            "frequency_mhz": frequency_mhz,
            "wifi_band": wifi_band,
            "link_speed_mbps": self.normalize_int(
                event.get("link_speed_mbps"),
                min_value=0,
            ),
            "message": self.normalize_string(event.get("message")),
            "event_started_at": self.normalize_datetime(event.get("event_started_at")),
            "event_ended_at": self.normalize_datetime(event.get("event_ended_at")),
        }

    def validate_events_list(
        self,
        events: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Validate the incoming events array.
        """
        if not isinstance(events, list):
            raise ValueError("events must be a list.")

        if len(events) > max_batch_size:
            raise ValueError(
                f"Too many network events in one request. "
                f"max_batch_size={max_batch_size}, received={len(events)}"
            )

        normalized_events: list[dict[str, Any]] = []

        for index, event in enumerate(events):
            try:
                normalized_events.append(self.validate_event_payload(event))
            except Exception as exc:
                raise ValueError(f"Invalid network event at index {index}: {exc}") from exc

        return normalized_events

    def record_events(
        self,
        session: Session,
        tablet_device_id: int,
        events: list[dict[str, Any]],
    ) -> list[TabletNetworkEvent]:
        """
        Insert network events for a tablet.

        Args:
            session:
                Existing SQLAlchemy session owned by orchestrator.
            tablet_device_id:
                ID from tablet_edge.tablet_device.
            events:
                Raw event payload list from the tablet/client.

        Returns:
            List of inserted TabletNetworkEvent ORM objects.
        """
        normalized_events = self.validate_events_list(events)

        inserted_events: list[TabletNetworkEvent] = []

        for event_data in normalized_events:
            event = TabletNetworkEvent(
                tablet_device_id=tablet_device_id,
                access_point_id=event_data["access_point_id"],
                event_type=event_data["event_type"],
                quality_level=event_data["quality_level"],
                server_url=event_data["server_url"],
                page_url=event_data["page_url"],
                latency_ms=event_data["latency_ms"],
                avg_latency_ms=event_data["avg_latency_ms"],
                consecutive_failures=event_data["consecutive_failures"],
                is_online=event_data["is_online"],
                ssid=event_data["ssid"],
                bssid=event_data["bssid"],
                router_ip=event_data["router_ip"],
                router_name=event_data["router_name"],
                wifi_rssi=event_data["wifi_rssi"],
                signal_level=event_data["signal_level"],
                ip_address=event_data["ip_address"],
                gateway_address=event_data["gateway_address"],
                dhcp_server_address=event_data["dhcp_server_address"],
                dns_servers=event_data["dns_servers"],
                frequency_mhz=event_data["frequency_mhz"],
                wifi_band=event_data["wifi_band"],
                link_speed_mbps=event_data["link_speed_mbps"],
                message=event_data["message"],
                event_started_at=event_data["event_started_at"],
                event_ended_at=event_data["event_ended_at"],
            )

            session.add(event)
            inserted_events.append(event)

        session.flush()

        logger.info(
            "[TABLET_EDGE_NETWORK] Recorded %s network event(s) for tablet_device_id=%s",
            len(inserted_events),
            tablet_device_id,
        )

        return inserted_events

    def get_recent_events(
        self,
        session: Session,
        tablet_device_id: int,
        *,
        limit: int = 100,
    ) -> list[TabletNetworkEvent]:
        """
        Return recent network events for a tablet.
        """
        safe_limit = max(1, min(int(limit), 500))

        stmt = (
            select(TabletNetworkEvent)
            .where(TabletNetworkEvent.tablet_device_id == tablet_device_id)
            .order_by(desc(TabletNetworkEvent.created_at))
            .limit(safe_limit)
        )

        return list(session.execute(stmt).scalars().all())

    def build_record_events_response(
        self,
        inserted_events: list[TabletNetworkEvent],
        *,
        failed: int = 0,
    ) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/network-events.
        """
        access_point_ids = sorted(
            {
                int(event.access_point_id)
                for event in inserted_events
                if event.access_point_id is not None
            }
        )

        return {
            "success": True,
            "accepted": len(inserted_events),
            "failed": failed,
            "event_ids": [event.id for event in inserted_events],
            "access_point_ids": access_point_ids,
        }


__all__ = [
    "TabletNetworkEventService",
]