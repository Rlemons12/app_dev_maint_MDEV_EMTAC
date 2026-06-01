from dataclasses import dataclass
from typing import Optional


@dataclass
class TabletIdentity:
    tablet_uid: Optional[str] = None
    tablet_name: Optional[str] = None


@dataclass
class LoginRequestData:
    employee_id: str
    password: str
    tablet_identity: TabletIdentity
    remote_addr: Optional[str]
    user_agent: Optional[str]
    current_page_url: Optional[str]


@dataclass
class LoginResult:
    success: bool
    message: str = ""

    user_id: Optional[int] = None
    employee_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    primary_area: Optional[str] = None
    age: Optional[int] = None
    education_level: Optional[str] = None
    start_date: Optional[object] = None
    user_level: Optional[object] = None

    login_time: Optional[str] = None
    login_record_id: Optional[int] = None
    emtac_session_tracking_id: Optional[str] = None
    tablet_user_session_started: bool = False


@dataclass
class LogoutRequestData:
    login_record_id: Optional[int]
    emtac_session_tracking_id: Optional[str]
    tablet_identity: TabletIdentity


@dataclass
class LogoutResult:
    success: bool
    message: str = ""
    tablet_identity: Optional[TabletIdentity] = None