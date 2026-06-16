from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from modules.emtacdb.emtacdb_fts import Base


class HelpChatSession(Base):
    __tablename__ = "help_chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="EMTAC User")
    user_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    employee_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    user_level: Mapped[str | None] = mapped_column(String(100), nullable=True)

    client_ip: Mapped[str | None] = mapped_column(String(255), nullable=True)
    remote_addr: Mapped[str | None] = mapped_column(String(255), nullable=True)
    x_forwarded_for: Mapped[str | None] = mapped_column(String(500), nullable=True)
    x_real_ip: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="open")
    current_page: Mapped[str | None] = mapped_column(String(500), nullable=True)

    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_online: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    messages: Mapped[list["HelpChatMessage"]] = relationship(
        "HelpChatMessage",
        back_populates="help_chat_session",
        cascade="all, delete-orphan",
        order_by="HelpChatMessage.created_at.asc()",
    )


class HelpChatMessage(Base):
    __tablename__ = "help_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    help_chat_session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("help_chat_sessions.id"),
        nullable=False,
        index=True,
    )

    sender_type: Mapped[str] = mapped_column(String(50), nullable=False)
    sender_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    message_type: Mapped[str] = mapped_column(String(50), nullable=False, default="text")
    message_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    attachment_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attachment_original_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attachment_mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    help_chat_session: Mapped[HelpChatSession] = relationship(
        "HelpChatSession",
        back_populates="messages",
    )
