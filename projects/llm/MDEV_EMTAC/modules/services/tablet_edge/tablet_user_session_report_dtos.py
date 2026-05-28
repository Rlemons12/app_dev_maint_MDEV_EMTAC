from dataclasses import dataclass
from typing import Optional


@dataclass
class TabletUserSessionReportRequest:
    tablet_uid: str
    tablet_name: Optional[str] = None

    username: Optional[str] = None
    display_name: Optional[str] = None
    user_id: Optional[int] = None

    event_type: str = "heartbeat"
    current_page_url: Optional[str] = None

    app_version: Optional[str] = None
    app_version_code: Optional[int] = None

    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class TabletUserSessionReportResult:
    success: bool
    message: str

    tablet_device_id: Optional[int] = None
    active_session_id: Optional[int] = None

    tablet_uid: Optional[str] = None
    tablet_name: Optional[str] = None

    username: Optional[str] = None
    display_name: Optional[str] = None

    event_type: Optional[str] = None
    is_active: bool = False