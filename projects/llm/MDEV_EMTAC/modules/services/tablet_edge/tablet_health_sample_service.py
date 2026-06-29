"""
Tablet health sample service for the EMTAC Tablet Edge Agent.

File:
    modules/services/tablet_edge/tablet_health_sample_service.py

Responsibilities:
    - Validate tablet health sample payloads.
    - Normalize quality levels, timestamps, battery values, Wi-Fi values, router/AP values, and booleans.
    - Batch insert health samples.
    - Query recent health samples.

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
from modules.emtacdb.tablet_edge.tablet_edge_models import TabletHealthSample


class TabletHealthSampleService:
    """
    Domain service for tablet_edge.tablet_health_sample.
    """

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
            "charging",
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
            "not_charging",
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
                "[TABLET_EDGE_HEALTH] Unknown quality_level received: %s",
                quality_level,
            )
            return "unknown"

        return quality_level

    def validate_sample_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize one health sample payload.

        Required:
            quality_level is recommended, but missing/blank values are normalized
            to 'unknown' for forward compatibility.

        Optional:
            sampled_at
            server_reachable
            server_latency_ms
            battery_percent
            is_charging
            access_point_id
            ssid
            bssid
            router_ip
            router_name
            wifi_rssi
            signal_level
            ip_address
            gateway_address
            frequency_mhz
            wifi_band
            link_speed_mbps
            app_foreground
            current_page_url
        """
        if not isinstance(sample, dict):
            raise ValueError("Each health sample must be a JSON object.")

        sampled_at = self.normalize_datetime(sample.get("sampled_at"))

        if sampled_at is None:
            sampled_at = datetime.now(timezone.utc)

        server_reachable = self.normalize_bool(sample.get("server_reachable"))

        if server_reachable is None:
            server_reachable = False

        frequency_mhz = self.normalize_int(
            sample.get("frequency_mhz"),
            min_value=0,
        )

        wifi_band = self.normalize_string(
            sample.get("wifi_band"),
            max_length=50,
        )

        if not wifi_band:
            wifi_band = self.infer_wifi_band(frequency_mhz)

        router_ip = self.normalize_string(sample.get("router_ip"), max_length=100)
        gateway_address = self.normalize_string(
            sample.get("gateway_address"),
            max_length=100,
        )

        if not router_ip and gateway_address:
            router_ip = gateway_address

        if not gateway_address and router_ip:
            gateway_address = router_ip

        return {
            "sampled_at": sampled_at,
            "server_reachable": server_reachable,
            "server_latency_ms": self.normalize_int(
                sample.get("server_latency_ms"),
                min_value=0,
            ),
            "quality_level": self.normalize_quality_level(sample.get("quality_level")),
            "battery_percent": self.normalize_int(
                sample.get("battery_percent"),
                min_value=0,
                max_value=100,
            ),
            "is_charging": self.normalize_bool(sample.get("is_charging")),
            "access_point_id": self.normalize_int(
                sample.get("access_point_id"),
                min_value=1,
            ),
            "ssid": self.normalize_string(sample.get("ssid"), max_length=150),
            "bssid": self.normalize_bssid(sample.get("bssid")),
            "router_ip": router_ip,
            "router_name": self.normalize_string(
                sample.get("router_name"),
                max_length=150,
            ),
            "wifi_rssi": self.normalize_int(
                sample.get("wifi_rssi"),
                min_value=-150,
                max_value=0,
            ),
            "signal_level": self.normalize_int(
                sample.get("signal_level"),
                min_value=0,
                max_value=4,
            ),
            "ip_address": self.normalize_string(
                sample.get("ip_address"),
                max_length=100,
            ),
            "gateway_address": gateway_address,
            "frequency_mhz": frequency_mhz,
            "wifi_band": wifi_band,
            "link_speed_mbps": self.normalize_int(
                sample.get("link_speed_mbps"),
                min_value=0,
            ),
            "app_foreground": self.normalize_bool(sample.get("app_foreground")),
            "current_page_url": self.normalize_string(sample.get("current_page_url")),
        }

    def validate_samples_list(
        self,
        samples: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Validate the incoming samples array.
        """
        if not isinstance(samples, list):
            raise ValueError("samples must be a list.")

        if len(samples) > max_batch_size:
            raise ValueError(
                f"Too many health samples in one request. "
                f"max_batch_size={max_batch_size}, received={len(samples)}"
            )

        normalized_samples: list[dict[str, Any]] = []

        for index, sample in enumerate(samples):
            try:
                normalized_samples.append(self.validate_sample_payload(sample))
            except Exception as exc:
                raise ValueError(f"Invalid health sample at index {index}: {exc}") from exc

        return normalized_samples

    def record_samples(
        self,
        session: Session,
        tablet_device_id: int,
        samples: list[dict[str, Any]],
    ) -> list[TabletHealthSample]:
        """
        Insert health samples for a tablet.

        Args:
            session:
                Existing SQLAlchemy session owned by orchestrator.
            tablet_device_id:
                ID from tablet_edge.tablet_device.
            samples:
                Raw sample payload list from the tablet/client.

        Returns:
            List of inserted TabletHealthSample ORM objects.
        """
        normalized_samples = self.validate_samples_list(samples)

        inserted_samples: list[TabletHealthSample] = []

        for sample_data in normalized_samples:
            sample = TabletHealthSample(
                tablet_device_id=tablet_device_id,
                access_point_id=sample_data["access_point_id"],
                sampled_at=sample_data["sampled_at"],
                server_reachable=sample_data["server_reachable"],
                server_latency_ms=sample_data["server_latency_ms"],
                quality_level=sample_data["quality_level"],
                battery_percent=sample_data["battery_percent"],
                is_charging=sample_data["is_charging"],
                ssid=sample_data["ssid"],
                bssid=sample_data["bssid"],
                router_ip=sample_data["router_ip"],
                router_name=sample_data["router_name"],
                wifi_rssi=sample_data["wifi_rssi"],
                signal_level=sample_data["signal_level"],
                ip_address=sample_data["ip_address"],
                gateway_address=sample_data["gateway_address"],
                frequency_mhz=sample_data["frequency_mhz"],
                wifi_band=sample_data["wifi_band"],
                link_speed_mbps=sample_data["link_speed_mbps"],
                app_foreground=sample_data["app_foreground"],
                current_page_url=sample_data["current_page_url"],
            )

            session.add(sample)
            inserted_samples.append(sample)

        session.flush()

        logger.info(
            "[TABLET_EDGE_HEALTH] Recorded %s health sample(s) for tablet_device_id=%s",
            len(inserted_samples),
            tablet_device_id,
        )

        return inserted_samples

    def get_recent_samples(
        self,
        session: Session,
        tablet_device_id: int,
        *,
        limit: int = 100,
    ) -> list[TabletHealthSample]:
        """
        Return recent health samples for a tablet.
        """
        safe_limit = max(1, min(int(limit), 500))

        stmt = (
            select(TabletHealthSample)
            .where(TabletHealthSample.tablet_device_id == tablet_device_id)
            .order_by(desc(TabletHealthSample.sampled_at))
            .limit(safe_limit)
        )

        return list(session.execute(stmt).scalars().all())

    def build_record_samples_response(
        self,
        inserted_samples: list[TabletHealthSample],
        *,
        failed: int = 0,
    ) -> dict[str, Any]:
        """
        Build response body for /tablet-edge/health-samples.
        """
        access_point_ids = sorted(
            {
                int(sample.access_point_id)
                for sample in inserted_samples
                if sample.access_point_id is not None
            }
        )

        return {
            "success": True,
            "accepted": len(inserted_samples),
            "failed": failed,
            "sample_ids": [sample.id for sample in inserted_samples],
            "access_point_ids": access_point_ids,
        }


__all__ = [
    "TabletHealthSampleService",
]