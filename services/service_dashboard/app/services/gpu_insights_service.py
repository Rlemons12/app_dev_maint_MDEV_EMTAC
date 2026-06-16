from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from configuration.config import SERVICE_GPU_BASE_URL
from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)


@dataclass
class GPUServiceSnapshot:
    status: str
    pid: Optional[int]
    uptime_seconds: Optional[int]
    cwd: Optional[str]
    command: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
            "cwd": self.cwd,
            "command": self.command,
        }


def fetch_json(url: str, timeout: float = 2.0) -> Dict[str, Any]:
    """
    Fetch JSON from a remote endpoint and return a consistent error structure
    when the request fails.
    """
    dash_debug(f"Fetching JSON from url='{url}' timeout={timeout}")

    try:
        with urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            dash_debug(f"Fetch succeeded for url='{url}'")
            return payload

    except HTTPError as exc:
        dash_warning(f"HTTP error fetching url='{url}' status={exc.code}")
        return {
            "error": f"HTTP {exc.code} from {url}",
            "url": url,
            "error_type": "http_error",
        }

    except URLError as exc:
        dash_warning(f"Connection error fetching url='{url}' reason={exc.reason}")
        return {
            "error": f"Connection error to {url}: {exc.reason}",
            "url": url,
            "error_type": "url_error",
        }

    except json.JSONDecodeError as exc:
        dash_error(f"Invalid JSON returned from url='{url}' error={exc}")
        return {
            "error": f"Invalid JSON returned from {url}: {exc}",
            "url": url,
            "error_type": "json_decode_error",
        }

    except Exception as exc:
        dash_error(f"Unexpected error fetching url='{url}' error={exc}")
        return {
            "error": f"Failed to fetch {url}: {exc}",
            "url": url,
            "error_type": "unknown_error",
        }


def build_service_snapshot(gpu_service: Any) -> GPUServiceSnapshot:
    """
    Build a normalized snapshot of the managed GPU service object from
    ServiceManager.get_service(...).
    """
    snapshot = GPUServiceSnapshot(
        status=gpu_service.get_status(),
        pid=gpu_service.get_pid(),
        uptime_seconds=gpu_service.get_uptime_seconds(),
        cwd=getattr(gpu_service, "cwd", None),
        command=" ".join(gpu_service.command) if getattr(gpu_service, "command", None) else "",
    )
    dash_debug(
        "Built GPU service snapshot "
        f"status='{snapshot.status}' pid='{snapshot.pid}' uptime='{snapshot.uptime_seconds}'"
    )
    return snapshot


def get_gpu_api_urls(base_url: Optional[str] = None) -> Dict[str, str]:
    """
    Build the GPU service API endpoint URLs.
    """
    root = (base_url or SERVICE_GPU_BASE_URL).rstrip("/")
    urls = {
        "status_data": f"{root}/api/status",
        "gpu": f"{root}/api/gpu",
        "process": f"{root}/api/process",
    }
    dash_debug(f"GPU API URLs resolved: {urls}")
    return urls


def get_gpu_service_insights(
    gpu_service: Any,
    base_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """
    Return a single structured payload for the service dashboard.
    """
    dash_info("Building GPU service insights payload")

    if not gpu_service:
        dash_warning("GPU service insights requested but GPU Service is not registered")
        return {
            "success": False,
            "message": "GPU Service is not registered.",
            "service": None,
            "gpu": {"error": "GPU Service is not registered."},
            "status_data": {"error": "GPU Service is not registered."},
            "process": {"error": "GPU Service is not registered."},
        }

    service_snapshot = build_service_snapshot(gpu_service).to_dict()

    if gpu_service.get_status() != "running":
        dash_info("GPU service insights requested while GPU Service is not running")
        not_running = {"error": "GPU Service is not running."}
        return {
            "success": True,
            "service": service_snapshot,
            "gpu": not_running,
            "status_data": not_running,
            "process": not_running,
        }

    urls = get_gpu_api_urls(base_url=base_url)

    status_data = fetch_json(urls["status_data"], timeout=timeout)
    gpu_data = fetch_json(urls["gpu"], timeout=timeout)
    process_data = fetch_json(urls["process"], timeout=timeout)

    dash_info("GPU service insights payload built successfully")

    return {
        "success": True,
        "service": service_snapshot,
        "gpu": gpu_data,
        "status_data": status_data,
        "process": process_data,
    }


def get_gpu_service_status_only(
    gpu_service: Any,
    base_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """
    Optional helper if you only want service status + /api/status.
    """
    dash_info("Building GPU service status-only payload")

    if not gpu_service:
        dash_warning("Status-only requested but GPU Service is not registered")
        return {
            "success": False,
            "message": "GPU Service is not registered.",
            "service": None,
            "status_data": {"error": "GPU Service is not registered."},
        }

    service_snapshot = build_service_snapshot(gpu_service).to_dict()

    if gpu_service.get_status() != "running":
        dash_info("Status-only requested while GPU Service is not running")
        return {
            "success": True,
            "service": service_snapshot,
            "status_data": {"error": "GPU Service is not running."},
        }

    url = get_gpu_api_urls(base_url=base_url)["status_data"]
    status_data = fetch_json(url, timeout=timeout)

    return {
        "success": True,
        "service": service_snapshot,
        "status_data": status_data,
    }


def get_gpu_service_gpu_only(
    gpu_service: Any,
    base_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """
    Optional helper if you only want service status + /api/gpu.
    """
    dash_info("Building GPU-only payload")

    if not gpu_service:
        dash_warning("GPU-only requested but GPU Service is not registered")
        return {
            "success": False,
            "message": "GPU Service is not registered.",
            "service": None,
            "gpu": {"error": "GPU Service is not registered."},
        }

    service_snapshot = build_service_snapshot(gpu_service).to_dict()

    if gpu_service.get_status() != "running":
        dash_info("GPU-only requested while GPU Service is not running")
        return {
            "success": True,
            "service": service_snapshot,
            "gpu": {"error": "GPU Service is not running."},
        }

    url = get_gpu_api_urls(base_url=base_url)["gpu"]
    gpu_data = fetch_json(url, timeout=timeout)

    return {
        "success": True,
        "service": service_snapshot,
        "gpu": gpu_data,
    }


def get_gpu_service_process_only(
    gpu_service: Any,
    base_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """
    Optional helper if you only want service status + /api/process.
    """
    dash_info("Building GPU process-only payload")

    if not gpu_service:
        dash_warning("Process-only requested but GPU Service is not registered")
        return {
            "success": False,
            "message": "GPU Service is not registered.",
            "service": None,
            "process": {"error": "GPU Service is not registered."},
        }

    service_snapshot = build_service_snapshot(gpu_service).to_dict()

    if gpu_service.get_status() != "running":
        dash_info("Process-only requested while GPU Service is not running")
        return {
            "success": True,
            "service": service_snapshot,
            "process": {"error": "GPU Service is not running."},
        }

    url = get_gpu_api_urls(base_url=base_url)["process"]
    process_data = fetch_json(url, timeout=timeout)

    return {
        "success": True,
        "service": service_snapshot,
        "process": process_data,
    }