"""
SQLAlchemy ORM models for the EMTAC Tablet Edge Agent.

File:
    modules/emtacdb/tablet_edge/tablet_edge_models.py

Database schema:
    tablet_edge

Tables:
    tablet_edge.tablet_device
    tablet_edge.tablet_wifi_access_point
    tablet_edge.tablet_wifi_observation
    tablet_edge.tablet_network_event
    tablet_edge.tablet_health_sample
    tablet_edge.tablet_dropdown_cache_manifest
    tablet_edge.tablet_sync_event
    tablet_edge.tablet_offline_event
    tablet_edge.tablet_app_log

Important:
    These models map to tables created by:

        modules/database_manager/tablet_edge/create_tablet_edge_schema.sql

    If new columns/tables are added here, the SQL schema script must also be
    updated before the application is run against a fresh or existing database.

    Do not put these models under modules/emtac_client. These are server-side
    PostgreSQL ORM models, not Android/client-side code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID as PythonUUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from modules.emtacdb.emtacdb_fts import Base


TABLET_EDGE_SCHEMA = "tablet_edge"


def _json_safe_value(value: Any) -> Any:
    """
    Convert common SQLAlchemy/Python values into JSON-safe values.

    This is useful for service responses, logging summaries, route payloads,
    and quick debugging from scripts.
    """
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, PythonUUID):
        return str(value)

    return value


class TabletDevice(Base):
    """
    One record per physical EMTAC tablet.

    Table:
        tablet_edge.tablet_device
    """

    __tablename__ = "tablet_device"
    __table_args__ = {"schema": TABLET_EDGE_SCHEMA}

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_uid = Column(UUID(as_uuid=True), nullable=False, unique=True)
    tablet_name = Column(String(150), nullable=False)

    device_make = Column(String(100), nullable=True)
    device_model = Column(String(100), nullable=True)
    android_version = Column(String(50), nullable=True)
    app_version = Column(String(50), nullable=True)

    assigned_area = Column(String(150), nullable=True)
    assigned_station = Column(String(150), nullable=True)
    assigned_role = Column(String(100), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))

    first_seen_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    network_events = relationship(
        "TabletNetworkEvent",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    health_samples = relationship(
        "TabletHealthSample",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    wifi_observations = relationship(
        "TabletWifiObservation",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    dropdown_cache_manifests = relationship(
        "TabletDropdownCacheManifest",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    sync_events = relationship(
        "TabletSyncEvent",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    offline_events = relationship(
        "TabletOfflineEvent",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    app_logs = relationship(
        "TabletAppLog",
        back_populates="tablet_device",
        passive_deletes=True,
    )

    user_sessions = relationship(
        "TabletUserSession",
        back_populates="tablet_device",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<TabletDevice id={self.id!r} "
            f"tablet_uid={self.tablet_uid!r} "
            f"tablet_name={self.tablet_name!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_uid": _json_safe_value(self.tablet_uid),
            "tablet_name": self.tablet_name,
            "device_make": self.device_make,
            "device_model": self.device_model,
            "android_version": self.android_version,
            "app_version": self.app_version,
            "assigned_area": self.assigned_area,
            "assigned_station": self.assigned_station,
            "assigned_role": self.assigned_role,
            "is_active": self.is_active,
            "first_seen_at": _json_safe_value(self.first_seen_at),
            "last_seen_at": _json_safe_value(self.last_seen_at),
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }

class TabletUserSession(Base):
    """
    Tracks EMTAC web/app user login sessions associated with a physical tablet.

    This table answers:
        - Who logged into this tablet?
        - When did they log in?
        - Are they still active?
        - When were they last seen?
        - When did they log out?

    Table:
        tablet_edge.tablet_user_session
    """

    __tablename__ = "tablet_user_session"
    __table_args__ = (
        Index(
            "ix_tablet_user_session_tablet_active_last_seen",
            "tablet_device_id",
            "is_active",
            "last_seen_at",
        ),
        Index(
            "ix_tablet_user_session_username_last_seen",
            "username",
            "last_seen_at",
        ),
        Index(
            "ix_tablet_user_session_user_id_last_seen",
            "user_id",
            "last_seen_at",
        ),
        Index(
            "ix_tablet_user_session_session_id",
            "session_id",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(BigInteger, nullable=True)
    username = Column(String(150), nullable=True)
    display_name = Column(String(255), nullable=True)

    session_id = Column(String(255), nullable=True)

    login_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    last_seen_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    logout_at = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))

    login_ip_address = Column(String(100), nullable=True)
    last_ip_address = Column(String(100), nullable=True)
    user_agent = Column(Text, nullable=True)

    current_page_url = Column(Text, nullable=True)

    logout_reason = Column(String(100), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="user_sessions",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletUserSession id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"username={self.username!r} "
            f"is_active={self.is_active!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "session_id": self.session_id,
            "login_at": _json_safe_value(self.login_at),
            "last_seen_at": _json_safe_value(self.last_seen_at),
            "logout_at": _json_safe_value(self.logout_at),
            "is_active": self.is_active,
            "login_ip_address": self.login_ip_address,
            "last_ip_address": self.last_ip_address,
            "user_agent": self.user_agent,
            "current_page_url": self.current_page_url,
            "logout_reason": self.logout_reason,
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }


class TabletWifiAccessPoint(Base):
    """
    Known Wi-Fi router/access point used by EMTAC tablets.

    The BSSID is the best unique identifier for the actual access point radio.
    Router IP and router name are stored as descriptive/routing details, but
    should not be treated as the primary identity.

    Table:
        tablet_edge.tablet_wifi_access_point
    """

    __tablename__ = "tablet_wifi_access_point"
    __table_args__ = (
        UniqueConstraint(
            "bssid",
            name="uq_tablet_wifi_access_point_bssid",
        ),
        Index(
            "ix_tablet_wifi_access_point_ssid",
            "ssid",
        ),
        Index(
            "ix_tablet_wifi_access_point_router_ip",
            "router_ip",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    ssid = Column(String(150), nullable=True)
    bssid = Column(String(64), nullable=False)

    router_ip = Column(String(100), nullable=True)
    router_name = Column(String(150), nullable=True)
    friendly_name = Column(String(150), nullable=True)

    assigned_area = Column(String(150), nullable=True)
    assigned_station = Column(String(150), nullable=True)
    physical_location = Column(String(255), nullable=True)

    is_approved = Column(Boolean, nullable=False, server_default=text("TRUE"))
    notes = Column(Text, nullable=True)

    first_seen_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    wifi_observations = relationship(
        "TabletWifiObservation",
        back_populates="access_point",
        passive_deletes=True,
    )

    network_events = relationship(
        "TabletNetworkEvent",
        back_populates="access_point",
        passive_deletes=True,
    )

    health_samples = relationship(
        "TabletHealthSample",
        back_populates="access_point",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<TabletWifiAccessPoint id={self.id!r} "
            f"ssid={self.ssid!r} "
            f"bssid={self.bssid!r} "
            f"router_ip={self.router_ip!r} "
            f"friendly_name={self.friendly_name!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ssid": self.ssid,
            "bssid": self.bssid,
            "router_ip": self.router_ip,
            "router_name": self.router_name,
            "friendly_name": self.friendly_name,
            "assigned_area": self.assigned_area,
            "assigned_station": self.assigned_station,
            "physical_location": self.physical_location,
            "is_approved": self.is_approved,
            "notes": self.notes,
            "first_seen_at": _json_safe_value(self.first_seen_at),
            "last_seen_at": _json_safe_value(self.last_seen_at),
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }


class TabletWifiObservation(Base):
    """
    Periodic Wi-Fi/router observation reported by a tablet.

    This table answers:
        - What access point/router was this tablet using at this time?
        - What was the Wi-Fi signal strength?
        - What gateway/router IP was being used?
        - Was the EMTAC server reachable at that moment?

    This table is intended for regular network snapshots. Use
    TabletNetworkEvent for important event-style records such as slow network,
    offline transition, recovered connection, or repeated failures.

    Table:
        tablet_edge.tablet_wifi_observation
    """

    __tablename__ = "tablet_wifi_observation"
    __table_args__ = (
        Index(
            "ix_tablet_wifi_observation_tablet_sampled_at",
            "tablet_device_id",
            "sampled_at",
        ),
        Index(
            "ix_tablet_wifi_observation_bssid",
            "bssid",
        ),
        Index(
            "ix_tablet_wifi_observation_access_point_id",
            "access_point_id",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    access_point_id = Column(
        BigInteger,
        ForeignKey(
            f"{TABLET_EDGE_SCHEMA}.tablet_wifi_access_point.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    sampled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    is_online = Column(Boolean, nullable=True)

    ssid = Column(String(150), nullable=True)
    bssid = Column(String(64), nullable=True)

    router_ip = Column(String(100), nullable=True)
    router_name = Column(String(150), nullable=True)

    ip_address = Column(String(100), nullable=True)
    gateway_address = Column(String(100), nullable=True)
    dhcp_server_address = Column(String(100), nullable=True)
    dns_servers = Column(Text, nullable=True)

    wifi_rssi = Column(Integer, nullable=True)
    signal_level = Column(Integer, nullable=True)

    frequency_mhz = Column(Integer, nullable=True)
    wifi_band = Column(String(50), nullable=True)
    link_speed_mbps = Column(Integer, nullable=True)

    server_url = Column(Text, nullable=True)
    server_reachable = Column(Boolean, nullable=True)
    server_latency_ms = Column(Integer, nullable=True)

    quality_level = Column(String(50), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="wifi_observations",
    )

    access_point = relationship(
        "TabletWifiAccessPoint",
        back_populates="wifi_observations",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletWifiObservation id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"ssid={self.ssid!r} "
            f"bssid={self.bssid!r} "
            f"wifi_rssi={self.wifi_rssi!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "access_point_id": self.access_point_id,
            "sampled_at": _json_safe_value(self.sampled_at),
            "is_online": self.is_online,
            "ssid": self.ssid,
            "bssid": self.bssid,
            "router_ip": self.router_ip,
            "router_name": self.router_name,
            "ip_address": self.ip_address,
            "gateway_address": self.gateway_address,
            "dhcp_server_address": self.dhcp_server_address,
            "dns_servers": self.dns_servers,
            "wifi_rssi": self.wifi_rssi,
            "signal_level": self.signal_level,
            "frequency_mhz": self.frequency_mhz,
            "wifi_band": self.wifi_band,
            "link_speed_mbps": self.link_speed_mbps,
            "server_url": self.server_url,
            "server_reachable": self.server_reachable,
            "server_latency_ms": self.server_latency_ms,
            "quality_level": self.quality_level,
            "created_at": _json_safe_value(self.created_at),
        }


class TabletNetworkEvent(Base):
    """
    Important network quality events reported by a tablet.

    Examples:
        - server_health_good
        - server_health_slow
        - server_unreachable
        - wifi_signal_poor
        - wifi_signal_recovered
        - tablet_offline
        - tablet_online

    Table:
        tablet_edge.tablet_network_event
    """

    __tablename__ = "tablet_network_event"
    __table_args__ = (
        Index(
            "ix_tablet_network_event_tablet_created_at",
            "tablet_device_id",
            "created_at",
        ),
        Index(
            "ix_tablet_network_event_bssid",
            "bssid",
        ),
        Index(
            "ix_tablet_network_event_access_point_id",
            "access_point_id",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    access_point_id = Column(
        BigInteger,
        ForeignKey(
            f"{TABLET_EDGE_SCHEMA}.tablet_wifi_access_point.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    event_type = Column(String(100), nullable=False)
    quality_level = Column(String(50), nullable=False)

    server_url = Column(Text, nullable=True)
    page_url = Column(Text, nullable=True)

    latency_ms = Column(Integer, nullable=True)
    avg_latency_ms = Column(Integer, nullable=True)
    consecutive_failures = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    is_online = Column(Boolean, nullable=True)

    ssid = Column(String(150), nullable=True)
    bssid = Column(String(64), nullable=True)

    router_ip = Column(String(100), nullable=True)
    router_name = Column(String(150), nullable=True)

    wifi_rssi = Column(Integer, nullable=True)
    signal_level = Column(Integer, nullable=True)

    ip_address = Column(String(100), nullable=True)
    gateway_address = Column(String(100), nullable=True)
    dhcp_server_address = Column(String(100), nullable=True)
    dns_servers = Column(Text, nullable=True)

    frequency_mhz = Column(Integer, nullable=True)
    wifi_band = Column(String(50), nullable=True)
    link_speed_mbps = Column(Integer, nullable=True)

    message = Column(Text, nullable=True)

    event_started_at = Column(DateTime(timezone=True), nullable=True)
    event_ended_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="network_events",
    )

    access_point = relationship(
        "TabletWifiAccessPoint",
        back_populates="network_events",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletNetworkEvent id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"event_type={self.event_type!r} "
            f"quality_level={self.quality_level!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "access_point_id": self.access_point_id,
            "event_type": self.event_type,
            "quality_level": self.quality_level,
            "server_url": self.server_url,
            "page_url": self.page_url,
            "latency_ms": self.latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "is_online": self.is_online,
            "ssid": self.ssid,
            "bssid": self.bssid,
            "router_ip": self.router_ip,
            "router_name": self.router_name,
            "wifi_rssi": self.wifi_rssi,
            "signal_level": self.signal_level,
            "ip_address": self.ip_address,
            "gateway_address": self.gateway_address,
            "dhcp_server_address": self.dhcp_server_address,
            "dns_servers": self.dns_servers,
            "frequency_mhz": self.frequency_mhz,
            "wifi_band": self.wifi_band,
            "link_speed_mbps": self.link_speed_mbps,
            "message": self.message,
            "event_started_at": _json_safe_value(self.event_started_at),
            "event_ended_at": _json_safe_value(self.event_ended_at),
            "created_at": _json_safe_value(self.created_at),
        }


class TabletHealthSample(Base):
    """
    Periodic tablet health sample.

    This is a lightweight current-condition snapshot. It can include basic
    router/access-point information so Grafana can display which AP/router
    the tablet was using during the sample.

    Table:
        tablet_edge.tablet_health_sample
    """

    __tablename__ = "tablet_health_sample"
    __table_args__ = (
        Index(
            "ix_tablet_health_sample_tablet_sampled_at",
            "tablet_device_id",
            "sampled_at",
        ),
        Index(
            "ix_tablet_health_sample_bssid",
            "bssid",
        ),
        Index(
            "ix_tablet_health_sample_access_point_id",
            "access_point_id",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    access_point_id = Column(
        BigInteger,
        ForeignKey(
            f"{TABLET_EDGE_SCHEMA}.tablet_wifi_access_point.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    sampled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    server_reachable = Column(
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
    )
    server_latency_ms = Column(Integer, nullable=True)

    quality_level = Column(String(50), nullable=False)

    battery_percent = Column(Integer, nullable=True)
    is_charging = Column(Boolean, nullable=True)

    ssid = Column(String(150), nullable=True)
    bssid = Column(String(64), nullable=True)

    router_ip = Column(String(100), nullable=True)
    router_name = Column(String(150), nullable=True)

    wifi_rssi = Column(Integer, nullable=True)
    signal_level = Column(Integer, nullable=True)

    ip_address = Column(String(100), nullable=True)
    gateway_address = Column(String(100), nullable=True)

    frequency_mhz = Column(Integer, nullable=True)
    wifi_band = Column(String(50), nullable=True)
    link_speed_mbps = Column(Integer, nullable=True)

    app_foreground = Column(Boolean, nullable=True)
    current_page_url = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="health_samples",
    )

    access_point = relationship(
        "TabletWifiAccessPoint",
        back_populates="health_samples",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletHealthSample id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"quality_level={self.quality_level!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "access_point_id": self.access_point_id,
            "sampled_at": _json_safe_value(self.sampled_at),
            "server_reachable": self.server_reachable,
            "server_latency_ms": self.server_latency_ms,
            "quality_level": self.quality_level,
            "battery_percent": self.battery_percent,
            "is_charging": self.is_charging,
            "ssid": self.ssid,
            "bssid": self.bssid,
            "router_ip": self.router_ip,
            "router_name": self.router_name,
            "wifi_rssi": self.wifi_rssi,
            "signal_level": self.signal_level,
            "ip_address": self.ip_address,
            "gateway_address": self.gateway_address,
            "frequency_mhz": self.frequency_mhz,
            "wifi_band": self.wifi_band,
            "link_speed_mbps": self.link_speed_mbps,
            "app_foreground": self.app_foreground,
            "current_page_url": self.current_page_url,
            "created_at": _json_safe_value(self.created_at),
        }


class TabletDropdownCacheManifest(Base):
    """
    Tracks dropdown/cache version state for each tablet.

    Table:
        tablet_edge.tablet_dropdown_cache_manifest
    """

    __tablename__ = "tablet_dropdown_cache_manifest"
    __table_args__ = (
        UniqueConstraint(
            "tablet_device_id",
            "cache_name",
            name="uq_tablet_cache_name",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    cache_name = Column(String(150), nullable=False)
    cache_version = Column(String(150), nullable=False)

    record_count = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    last_full_sync_at = Column(DateTime(timezone=True), nullable=True)
    last_delta_sync_at = Column(DateTime(timezone=True), nullable=True)

    sync_status = Column(
        String(50),
        nullable=False,
        server_default=text("'unknown'"),
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="dropdown_cache_manifests",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletDropdownCacheManifest id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"cache_name={self.cache_name!r} "
            f"cache_version={self.cache_version!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "cache_name": self.cache_name,
            "cache_version": self.cache_version,
            "record_count": self.record_count,
            "last_full_sync_at": _json_safe_value(self.last_full_sync_at),
            "last_delta_sync_at": _json_safe_value(self.last_delta_sync_at),
            "sync_status": self.sync_status,
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }


class TabletSyncEvent(Base):
    """
    Tracks sync attempts between tablet and server.

    Table:
        tablet_edge.tablet_sync_event
    """

    __tablename__ = "tablet_sync_event"
    __table_args__ = (
        Index(
            "ix_tablet_sync_event_tablet_created_at",
            "tablet_device_id",
            "created_at",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    sync_type = Column(String(100), nullable=False)
    sync_direction = Column(String(50), nullable=False)

    status = Column(String(50), nullable=False)

    records_sent = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    records_received = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    records_failed = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    duration_ms = Column(Integer, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="sync_events",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletSyncEvent id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"sync_type={self.sync_type!r} "
            f"status={self.status!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "sync_type": self.sync_type,
            "sync_direction": self.sync_direction,
            "status": self.status,
            "records_sent": self.records_sent,
            "records_received": self.records_received,
            "records_failed": self.records_failed,
            "started_at": _json_safe_value(self.started_at),
            "completed_at": _json_safe_value(self.completed_at),
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "created_at": _json_safe_value(self.created_at),
        }


class TabletOfflineEvent(Base):
    """
    Stores queued/offline events sent from the tablet.

    Table:
        tablet_edge.tablet_offline_event
    """

    __tablename__ = "tablet_offline_event"
    __table_args__ = (
        UniqueConstraint(
            "tablet_device_id",
            "local_event_id",
            name="uq_tablet_local_event",
        ),
        Index(
            "ix_tablet_offline_event_tablet_created_at",
            "tablet_device_id",
            "created_at",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="CASCADE"),
        nullable=False,
    )

    local_event_id = Column(UUID(as_uuid=True), nullable=False)

    event_type = Column(String(100), nullable=False)
    event_payload = Column(JSONB, nullable=False)

    client_created_at = Column(DateTime(timezone=True), nullable=False)
    server_received_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    processing_status = Column(
        String(50),
        nullable=False,
        server_default=text("'pending'"),
    )
    processed_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="offline_events",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletOfflineEvent id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"local_event_id={self.local_event_id!r} "
            f"event_type={self.event_type!r} "
            f"processing_status={self.processing_status!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "local_event_id": _json_safe_value(self.local_event_id),
            "event_type": self.event_type,
            "event_payload": self.event_payload,
            "client_created_at": _json_safe_value(self.client_created_at),
            "server_received_at": _json_safe_value(self.server_received_at),
            "processing_status": self.processing_status,
            "processed_at": _json_safe_value(self.processed_at),
            "error_message": self.error_message,
            "created_at": _json_safe_value(self.created_at),
        }


class TabletAppLog(Base):
    """
    App-side logs submitted by the Android Edge Agent.

    Table:
        tablet_edge.tablet_app_log
    """

    __tablename__ = "tablet_app_log"
    __table_args__ = (
        Index(
            "ix_tablet_app_log_tablet_created_at",
            "tablet_device_id",
            "created_at",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    tablet_device_id = Column(
        BigInteger,
        ForeignKey(f"{TABLET_EDGE_SCHEMA}.tablet_device.id", ondelete="SET NULL"),
        nullable=True,
    )

    log_level = Column(String(50), nullable=False)
    log_source = Column(String(150), nullable=True)
    message = Column(Text, nullable=False)

    context = Column(JSONB, nullable=True)

    client_created_at = Column(DateTime(timezone=True), nullable=True)
    server_received_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    tablet_device = relationship(
        "TabletDevice",
        back_populates="app_logs",
    )

    def __repr__(self) -> str:
        return (
            f"<TabletAppLog id={self.id!r} "
            f"tablet_device_id={self.tablet_device_id!r} "
            f"log_level={self.log_level!r} "
            f"log_source={self.log_source!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tablet_device_id": self.tablet_device_id,
            "log_level": self.log_level,
            "log_source": self.log_source,
            "message": self.message,
            "context": self.context,
            "client_created_at": _json_safe_value(self.client_created_at),
            "server_received_at": _json_safe_value(self.server_received_at),
            "created_at": _json_safe_value(self.created_at),
        }


class TabletWifiRouterName(Base):
    """
    Friendly router/gateway name lookup for EMTAC tablet Wi-Fi status.

    This table maps router/gateway IP addresses to human-friendly names.

    Examples:
        100.100.28.1   -> VLAN 81 Gateway
        192.168.255.1  -> VLAN 80 Gateway
        100.100.28.225 -> Engineering
        192.168.255.12 -> Sterilization

    Table:
        tablet_edge.tablet_wifi_router_name
    """

    __tablename__ = "tablet_wifi_router_name"
    __table_args__ = (
        UniqueConstraint(
            "router_ip",
            name="uq_tablet_wifi_router_name_router_ip",
        ),
        Index(
            "ix_tablet_wifi_router_name_router_name",
            "router_name",
        ),
        Index(
            "ix_tablet_wifi_router_name_assigned_area",
            "assigned_area",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    router_ip = Column(String(100), nullable=False)
    router_name = Column(String(150), nullable=False)

    assigned_area = Column(String(150), nullable=True)
    notes = Column(Text, nullable=True)

    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    def __repr__(self) -> str:
        return (
            f"<TabletWifiRouterName id={self.id!r} "
            f"router_ip={self.router_ip!r} "
            f"router_name={self.router_name!r} "
            f"assigned_area={self.assigned_area!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "router_ip": self.router_ip,
            "router_name": self.router_name,
            "assigned_area": self.assigned_area,
            "notes": self.notes,
            "is_active": self.is_active,
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }

class TabletAppRelease(Base):
    """
    EMTAC Tablet Edge Agent APK release record.

    This table stores approved APK releases that EMTAC tablets can check
    against when determining whether an update is available.

    Table:
        tablet_edge.tablet_app_release
    """

    __tablename__ = "tablet_app_release"
    __table_args__ = (
        UniqueConstraint(
            "app_package",
            "release_channel",
            "version_code",
            name="uq_tablet_app_release_package_channel_version_code",
        ),
        Index(
            "ix_tablet_app_release_package_active_version_code",
            "app_package",
            "is_active",
            "version_code",
        ),
        Index(
            "ix_tablet_app_release_channel_active_version_code",
            "release_channel",
            "is_active",
            "version_code",
        ),
        {"schema": TABLET_EDGE_SCHEMA},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_package = Column(
        String(255),
        nullable=False,
        server_default=text("'com.example.emtactablet'"),
    )

    release_channel = Column(
        String(50),
        nullable=False,
        server_default=text("'stable'"),
    )

    version_name = Column(String(50), nullable=False)
    version_code = Column(Integer, nullable=False)

    apk_filename = Column(String(255), nullable=False)
    apk_file_path = Column(Text, nullable=False)

    apk_sha256 = Column(String(64), nullable=False)
    apk_size_bytes = Column(BigInteger, nullable=True)

    release_notes = Column(Text, nullable=True)

    min_supported_version_code = Column(Integer, nullable=True)
    max_supported_version_code = Column(Integer, nullable=True)

    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))
    is_required = Column(Boolean, nullable=False, server_default=text("FALSE"))

    rollout_percent = Column(
        Integer,
        nullable=False,
        server_default=text("100"),
    )

    created_by = Column(String(150), nullable=True)

    published_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    def __repr__(self) -> str:
        return (
            f"<TabletAppRelease id={self.id!r} "
            f"app_package={self.app_package!r} "
            f"release_channel={self.release_channel!r} "
            f"version_name={self.version_name!r} "
            f"version_code={self.version_code!r} "
            f"is_active={self.is_active!r}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "app_package": self.app_package,
            "release_channel": self.release_channel,
            "version_name": self.version_name,
            "version_code": self.version_code,
            "apk_filename": self.apk_filename,
            "apk_file_path": self.apk_file_path,
            "apk_sha256": self.apk_sha256,
            "apk_size_bytes": self.apk_size_bytes,
            "release_notes": self.release_notes,
            "min_supported_version_code": self.min_supported_version_code,
            "max_supported_version_code": self.max_supported_version_code,
            "is_active": self.is_active,
            "is_required": self.is_required,
            "rollout_percent": self.rollout_percent,
            "created_by": self.created_by,
            "published_at": _json_safe_value(self.published_at),
            "retired_at": _json_safe_value(self.retired_at),
            "created_at": _json_safe_value(self.created_at),
            "updated_at": _json_safe_value(self.updated_at),
        }

__all__ = [
    "TABLET_EDGE_SCHEMA",
    "TabletDevice",
    "TabletUserSession",
    "TabletWifiAccessPoint",
    "TabletWifiRouterName",
    "TabletWifiObservation",
    "TabletNetworkEvent",
    "TabletHealthSample",
    "TabletDropdownCacheManifest",
    "TabletSyncEvent",
    "TabletOfflineEvent",
    "TabletAppLog",
    "TabletAppRelease",
]