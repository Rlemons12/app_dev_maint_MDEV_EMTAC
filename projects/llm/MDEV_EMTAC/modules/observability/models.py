#%%
# modules/emtac_observability/models.py

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    Boolean,
    Text,
    ForeignKey,
    BigInteger,
    DateTime,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from modules.configuration.config_env import get_db_config


# ---------------------------------------------------------
# Base Setup (uses existing DatabaseConfig singleton)
# ---------------------------------------------------------

db_config = get_db_config()
Base = db_config.get_main_base()


# =========================================================
# TRACE SESSION (Root)
# =========================================================

class TraceSession(Base):
    __tablename__ = "trace_sessions"
    __table_args__ = {"schema": "emtac_observability"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(String(64), index=True)
    root_span_id = Column(UUID(as_uuid=True))

    service_name = Column(String(100))
    environment = Column(String(50))

    started_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime(timezone=True))
    duration_ms = Column(Numeric)

    status = Column(String(20))
    sampled = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    spans = relationship(
        "TraceSpan",
        back_populates="trace_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# =========================================================
# TRACE SPAN (Function / Operation)
# =========================================================

class TraceSpan(Base):
    __tablename__ = "trace_spans"
    __table_args__ = (
        Index("idx_trace_spans_trace_id", "trace_id"),
        Index("idx_trace_spans_parent", "parent_span_id"),
        Index("idx_trace_spans_request_id", "request_id"),
        Index("idx_trace_spans_status", "status"),
        Index("idx_trace_spans_module_name", "module_name"),
        {"schema": "emtac_observability"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    trace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emtac_observability.trace_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    parent_span_id = Column(UUID(as_uuid=True), nullable=True)

    # -----------------------------
    # Core span identity
    # -----------------------------
    name = Column(String(255), nullable=False)
    qualified_name = Column(String(512), nullable=True)  # e.g. modules.x.y.func
    module_name = Column(String(255), nullable=True)     # e.g. modules.foo.bar
    file_path = Column(String(1024), nullable=True)      # relative path
    line_number = Column(Integer, nullable=True)

    depth = Column(Integer)

    # -----------------------------
    # Timing / metrics
    # -----------------------------
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True))
    duration_ms = Column(Numeric)

    cpu_ms = Column(Numeric)
    memory_kb = Column(Numeric)

    # -----------------------------
    # Status / errors
    # -----------------------------
    status = Column(String(20))
    exception = Column(Text)

    # -----------------------------
    # Correlation
    # -----------------------------
    request_id = Column(String(64))
    thread_id = Column(BigInteger)
    process_id = Column(BigInteger)

    metadata_json = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    trace_session = relationship("TraceSession", back_populates="spans")

    events = relationship(
        "TraceEvent",
        back_populates="span",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    alerts = relationship(
        "TraceAlert",
        back_populates="span",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# =========================================================
# TRACE EVENT (Optional fine-grain events)
# =========================================================

class TraceEvent(Base):
    __tablename__ = "trace_events"
    __table_args__ = {"schema": "emtac_observability"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    span_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emtac_observability.trace_spans.id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type = Column(String(50))
    payload = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    span = relationship("TraceSpan", back_populates="events")


# =========================================================
# TRACE ALERT (Threshold breaches)
# =========================================================

class TraceAlert(Base):
    __tablename__ = "trace_alerts"
    __table_args__ = {"schema": "emtac_observability"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    span_id = Column(
        UUID(as_uuid=True),
        ForeignKey("emtac_observability.trace_spans.id", ondelete="CASCADE"),
        nullable=False,
    )

    metric_type = Column(String(50))
    threshold_value = Column(Numeric)
    actual_value = Column(Numeric)
    severity = Column(String(20))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    span = relationship("TraceSpan", back_populates="alerts")