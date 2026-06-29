# services/service_dashboard/app/services/service_manager.py

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)


class ManagedService:
    def __init__(self, config: dict[str, Any]):
        self.name: str = config["name"]
        self.service_type: str = config.get("service_type", "process")
        self.cwd: Optional[str] = config.get("cwd")

        self.command: list[str] = config.get("command", [])
        self.start_command: list[str] = config.get("start_command", [])
        self.stop_command: list[str] = config.get("stop_command", [])
        self.status_command: list[str] = config.get("status_command", [])

        self.base_url: Optional[str] = config.get("base_url")

        # Optional process environment override.
        self.env: Optional[dict[str, str]] = config.get("env")

        self.log_file: Optional[str] = config.get("log_file")
        self.auto_tail_log: bool = config.get("auto_tail_log", False)

        # Optional explicit health path.
        # Example:
        #   MCP AI Layer -> /api/ai/health
        #   GPU Service  -> /api/status
        #   Grafana      -> /api/health
        self.health_path: Optional[str] = config.get("health_path")

        self.process: Optional[subprocess.Popen[str]] = None
        self.output_buffer: deque[str] = deque(maxlen=1000)

        self.lock = threading.Lock()
        self.operation_lock = threading.RLock()

        self.started_at: Optional[float] = None
        self.last_known_status: str = "stopped"

        # For command-managed services like PostgreSQL where pg_ctl.exe exits
        # but postgres.exe keeps running independently.
        self.external_pid: Optional[int] = None

        self.start_in_progress: bool = False
        self.stop_in_progress: bool = False

        self.log_tail_thread: Optional[threading.Thread] = None
        self.log_tail_stop_event = threading.Event()


    def append_output(self, line: str) -> None:
        clean_line = (line or "").rstrip()
        if not clean_line:
            return

        with self.lock:
            self.output_buffer.append(clean_line)

    def get_output(self) -> list[str]:
        with self.lock:
            return list(self.output_buffer)

    def clear_output(self) -> None:
        with self.lock:
            self.output_buffer.clear()

    def _refresh_command_status(self) -> bool:
        """
        Refresh status for command-managed services.

        PostgreSQL is started through pg_ctl.exe. pg_ctl.exe exits after starting
        postgres.exe, so ServiceManager cannot rely on a Popen handle.

        Source of truth:
            pg_ctl -D <data_dir> status

        Expected running output:
            pg_ctl: server is running (PID: 14800)
        """

        if self.service_type != "command":
            return False

        if not self.status_command:
            return self.last_known_status == "running"

        try:
            result = subprocess.run(
                self.status_command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = f"{stdout}\n{stderr}"
            output_lower = output.lower()

            running = result.returncode == 0 or "server is running" in output_lower
            stopped = (
                "no server running" in output_lower
                or "server is not running" in output_lower
            )

            if running:
                self.last_known_status = "running"

                pid_match = re.search(r"PID:\s*(\d+)", output, flags=re.IGNORECASE)
                if pid_match:
                    try:
                        self.external_pid = int(pid_match.group(1))
                    except Exception:
                        self.external_pid = None

                if self.started_at is None:
                    self.started_at = time.time()

                return True

            if stopped:
                self.last_known_status = "stopped"
                self.external_pid = None
                self.started_at = None
                return False

            # Unknown output. Preserve the last known state.
            return self.last_known_status == "running"

        except Exception as exc:
            self.append_output(f"[SYSTEM] Status check failed: {exc}")
            dash_warning(
                f"Status check failed for service='{self.name}' error={exc}"
            )
            return self.last_known_status == "running"

    def is_running(self) -> bool:
        if self.service_type == "process":
            return self.process is not None and self.process.poll() is None

        if self.service_type == "command":
            return self._refresh_command_status()

        return False

    def get_pid(self) -> Optional[int]:
        if self.service_type == "process" and self.process and self.is_running():
            return self.process.pid

        if self.service_type == "command":
            self._refresh_command_status()
            return self.external_pid

        return None

    def get_status(self) -> str:
        if self.service_type == "command":
            if self.start_in_progress:
                if self._refresh_command_status():
                    self.start_in_progress = False
                    return "running"
                return "starting"

            if self.stop_in_progress:
                if not self._refresh_command_status():
                    self.stop_in_progress = False
                    return "stopped"
                return "stopping"

            return "running" if self._refresh_command_status() else "stopped"

        if self.start_in_progress:
            return "starting"

        if self.stop_in_progress:
            return "stopping"

        return "running" if self.is_running() else self.last_known_status

    def get_uptime_seconds(self) -> Optional[int]:
        if self.started_at and self.get_status() == "running":
            return int(time.time() - self.started_at)

        return None

    def start_log_tailing(self) -> None:
        if not self.log_file or not self.auto_tail_log:
            return

        if self.log_tail_thread and self.log_tail_thread.is_alive():
            return

        self.log_tail_stop_event.clear()
        self.log_tail_thread = threading.Thread(
            target=self._tail_log_file,
            name=f"log-tail-{self.name}",
            daemon=True,
        )
        self.log_tail_thread.start()
        dash_debug(
            f"Started log tail thread for service='{self.name}' log_file='{self.log_file}'"
        )

    def stop_log_tailing(self) -> None:
        self.log_tail_stop_event.set()
        dash_debug(f"Stop requested for log tail thread service='{self.name}'")

    def _tail_log_file(self) -> None:
        if not self.log_file:
            return

        log_path = Path(self.log_file)
        self.append_output(f"[SYSTEM] Starting log tail: {log_path}")
        dash_info(f"Starting log tail for service='{self.name}' path='{log_path}'")

        while not self.log_tail_stop_event.is_set():
            if not log_path.exists():
                time.sleep(1)
                continue

            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(0, os.SEEK_END)

                    while not self.log_tail_stop_event.is_set():
                        line = f.readline()
                        if not line:
                            time.sleep(0.5)
                            continue

                        self.append_output(line)

            except Exception as exc:
                self.append_output(f"[SYSTEM] Log tail error: {exc}")
                dash_warning(
                    f"Log tail error for service='{self.name}' path='{log_path}' error={exc}"
                )
                time.sleep(2)

        self.append_output(f"[SYSTEM] Stopped log tail: {log_path}")
        dash_info(f"Stopped log tail for service='{self.name}' path='{log_path}'")


class ServiceManager:
    def __init__(self):
        self.services: Dict[str, ManagedService] = {}
        self.manager_lock = threading.RLock()

        # When false, the dashboard will not treat an externally reachable
        # process as a dashboard-owned running service.
        #
        # This keeps service ownership clear:
        # - Dashboard-started services show PID/uptime/output.
        # - External/manual services do not get silently treated as owned.
        self.treat_external_reachable_as_running = (
            os.getenv("DASHBOARD_TREAT_EXTERNAL_REACHABLE_AS_RUNNING", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "y", "on"}
        )

        dash_info(
            "ServiceManager initialized | "
            f"treat_external_reachable_as_running="
            f"{self.treat_external_reachable_as_running}"
        )

    def register_service(self, config: dict[str, Any]) -> None:
        name = config["name"]

        with self.manager_lock:
            if name in self.services:
                dash_warning(f"Duplicate service registration attempted for '{name}'")
                raise ValueError(f"Service '{name}' is already registered.")

            self.services[name] = ManagedService(config=config)

        dash_info(
            f"Registered service='{name}' type='{config.get('service_type', 'process')}'"
        )

    def _run_command(
        self,
        command: list[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess[str]:
        dash_debug(
            f"Running command='{command}' cwd='{cwd}' timeout='{timeout}'"
        )

        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def _append_completed_process_output(
        self,
        service: ManagedService,
        result: subprocess.CompletedProcess[str],
    ) -> None:
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if stdout.strip():
            service.append_output("[STDOUT]")
            for line in stdout.splitlines():
                service.append_output(line)

        if stderr.strip():
            service.append_output("[STDERR]")
            for line in stderr.splitlines():
                service.append_output(line)

    # ------------------------------------------------------------------
    # General URL / port guards
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_port_from_url(url: str | None) -> int | None:
        if not url:
            return None

        try:
            parsed = urlparse(url)
            return parsed.port
        except Exception:
            return None

    @staticmethod
    def _normalize_base_url(url: str | None) -> str | None:
        if not url:
            return None

        cleaned = str(url).strip().rstrip("/")

        if not cleaned:
            return None

        if cleaned.startswith("http://0.0.0.0:"):
            cleaned = cleaned.replace("http://0.0.0.0:", "http://127.0.0.1:", 1)

        if cleaned.startswith("https://0.0.0.0:"):
            cleaned = cleaned.replace("https://0.0.0.0:", "https://127.0.0.1:", 1)

        return cleaned

    @staticmethod
    def _is_port_listening(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _service_health_path(self, service: ManagedService) -> str | None:
        if service.health_path:
            return service.health_path

        name = service.name.strip().lower()

        if self._is_gpu_service_name(service):
            return "/api/status"

        if name in {
            "ai gateway",
            "ai_gateway",
            "mcp gateway",
            "emtac mcp gateway",
        }:
            return "/health"

        if name in {
            "mcp ai layer",
            "mcp_ai_layer",
            "emtac mcp ai layer",
            "emtac mcp ai layer rest api",
        }:
            return "/api/ai/health"

        if name in {
            "emtac mcp coordinator",
            "mcp coordinator",
            "mcp_coordinator",
        }:
            return "/mcp"

        if name in {"grafana", "grafana server"}:
            return "/api/health"

        if service.base_url:
            return "/health"

        return None

    def _service_accept_headers(self, service: ManagedService) -> dict[str, str]:
        name = service.name.strip().lower()

        if name in {
            "emtac mcp coordinator",
            "mcp coordinator",
            "mcp_coordinator",
        }:
            return {
                "Accept": "text/event-stream, application/json",
            }

        return {
            "Accept": "application/json",
        }

    def _is_service_reachable(self, service: ManagedService) -> bool:
        base_url = self._normalize_base_url(service.base_url)
        health_path = self._service_health_path(service)

        if not base_url or not health_path:
            return False

        health_url = f"{base_url}{health_path}"

        try:
            req = Request(
                health_url,
                headers=self._service_accept_headers(service),
                method="GET",
            )

            with urlopen(req, timeout=2.0) as response:
                reachable = 200 <= response.status < 500

            if reachable:
                dash_info(
                    f"Service reachability check succeeded "
                    f"service='{service.name}' url='{health_url}'"
                )

            return reachable


        except HTTPError as exc:

            # HTTP error still proves a service is answering on that route.

            # MCP streamable HTTP commonly returns 400/406/405 for plain GET

            # requests because it expects MCP/SSE-style headers and flow.

            service_name = service.name.strip().lower()

            if service_name in {

                "emtac mcp coordinator",

                "mcp coordinator",

                "mcp_coordinator",

            } and exc.code in {400, 405, 406}:
                dash_debug(

                    f"MCP Coordinator reachability check got expected HTTP "

                    f"status service='{service.name}' url='{health_url}' "

                    f"status={exc.code}; treating as reachable"

                )

                return True

            dash_warning(

                f"Service reachability check got HTTP error "

                f"service='{service.name}' url='{health_url}' status={exc.code}"

            )

            return True

        except URLError as exc:
            dash_debug(
                f"Service reachability check failed "
                f"service='{service.name}' url='{health_url}' reason={exc.reason}"
            )
            return False

        except TimeoutError:
            dash_warning(
                f"Service reachability check timed out "
                f"service='{service.name}' url='{health_url}'"
            )
            return False

        except Exception as exc:
            dash_warning(
                f"Service reachability check failed "
                f"service='{service.name}' url='{health_url}' error={exc}"
            )
            return False

    def _is_service_port_in_use(self, service: ManagedService) -> bool:
        base_url = self._normalize_base_url(service.base_url)
        port = self._extract_port_from_url(base_url)

        if port is None:
            return False

        return self._is_port_listening(port)

    def _mark_service_reachable_without_owned_process(
        self,
        service: ManagedService,
        *,
        reason: str,
    ) -> None:
        service.last_known_status = "running"

        if service.started_at is None:
            service.started_at = time.time()

        service.append_output(
            f"[SYSTEM] Service is already reachable on "
            f"{self._normalize_base_url(service.base_url)}; {reason}"
        )

    # ------------------------------------------------------------------
    # GPU service guards
    # ------------------------------------------------------------------

    def _is_gpu_service_name(self, service_or_name: Any) -> bool:
        if isinstance(service_or_name, ManagedService):
            raw_name = service_or_name.name
        else:
            raw_name = str(service_or_name or "")

        service_name = raw_name.strip().lower()

        return service_name in {
            "gpu",
            "gpu service",
            "gpu_service",
            "emtac gpu",
            "emtac gpu service",
        }

    def _get_gpu_service_base_url(self) -> str:
        if "GPU_SERVICE_URL" in os.environ:
            return os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5051").rstrip("/")

        service = self.services.get("GPU Service")

        if service and service.base_url:
            return self._normalize_base_url(service.base_url) or "http://127.0.0.1:5051"

        return "http://127.0.0.1:5051"

    def _is_gpu_service_reachable(self) -> bool:
        status_url = f"{self._get_gpu_service_base_url()}/api/status"

        try:
            req = Request(
                status_url,
                headers={"Accept": "application/json"},
                method="GET",
            )

            with urlopen(req, timeout=2.0) as response:
                reachable = response.status == 200

            if reachable:
                dash_info(
                    f"GPU service reachability check succeeded url='{status_url}'"
                )

            return reachable

        except HTTPError as exc:
            dash_warning(
                f"GPU service reachability check returned HTTP error "
                f"url='{status_url}' status={exc.code}"
            )
            return False

        except URLError as exc:
            dash_debug(
                f"GPU service reachability check failed "
                f"url='{status_url}' reason={exc.reason}"
            )
            return False

        except TimeoutError:
            dash_warning(
                f"GPU service reachability check timed out url='{status_url}'"
            )
            return False

        except Exception as exc:
            dash_warning(
                f"GPU service reachability check failed "
                f"url='{status_url}' error={exc}"
            )
            return False

    def _mark_gpu_service_reachable_without_owned_process(
        self,
        service: ManagedService,
        *,
        reason: str,
    ) -> None:
        service.last_known_status = "running"

        if service.started_at is None:
            service.started_at = time.time()

        service.append_output(
            f"[SYSTEM] GPU Service is already reachable on "
            f"{self._get_gpu_service_base_url()}; {reason}"
        )

    def _get_effective_status(self, service: ManagedService) -> str:
        if service.start_in_progress:
            return "starting"

        if service.stop_in_progress:
            return "stopping"

        # Dashboard-owned process or command-managed service.
        if service.is_running():
            return "running"

        # Optional legacy behavior:
        # Treat a reachable external process as running even if the dashboard
        # did not start it. Default is false so services must be started from
        # the dashboard to be considered dashboard-owned/running.
        if self.treat_external_reachable_as_running:
            if self._is_gpu_service_name(service) and self._is_gpu_service_reachable():
                service.last_known_status = "running"
                return "running"

            if service.service_type == "process" and self._is_service_reachable(service):
                service.last_known_status = "running"
                return "running"

        return service.get_status()

    def _get_effective_pid(self, service: ManagedService) -> Optional[int]:
        return service.get_pid()

    def _get_effective_uptime_seconds(self, service: ManagedService) -> Optional[int]:
        if service.started_at and self._get_effective_status(service) == "running":
            return int(time.time() - service.started_at)

        return None

    # ------------------------------------------------------------------
    # Public service operations
    # ------------------------------------------------------------------

    def start_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)

        if not service:
            dash_warning(f"Start requested for unknown service='{name}'")
            return {
                "success": False,
                "message": f"Service '{name}' not found.",
            }

        with service.operation_lock:
            dash_info(f"Start requested for service='{name}'")

            if service.start_in_progress:
                dash_warning(
                    f"Start skipped; service is already starting service='{service.name}'"
                )
                service.append_output(
                    f"[SYSTEM] Start skipped; service '{service.name}' is already starting."
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' is already starting.",
                    "status": "starting",
                    "pid": service.get_pid(),
                }

            if service.is_running():
                dash_warning(
                    f"Start skipped; service already running according to "
                    f"ServiceManager service='{service.name}'"
                )
                service.append_output(
                    f"[SYSTEM] Start skipped; service '{service.name}' is already running."
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' is already running.",
                    "status": "running",
                    "pid": service.get_pid(),
                }

            if self._is_gpu_service_name(service):
                if self._is_gpu_service_reachable():
                    self._mark_gpu_service_reachable_without_owned_process(
                        service,
                        reason="start skipped",
                    )
                    dash_warning(
                        "GPU Service start skipped because /api/status is already reachable."
                    )
                    return {
                        "success": True,
                        "message": "GPU Service is already running on port 5051.",
                        "status": "running",
                        "pid": service.get_pid(),
                    }

            elif service.service_type == "process":
                if self._is_service_reachable(service):
                    self._mark_service_reachable_without_owned_process(
                        service,
                        reason="start skipped",
                    )
                    dash_warning(
                        f"Start skipped because service is already reachable "
                        f"service='{service.name}' base_url='{service.base_url}'"
                    )
                    return {
                        "success": True,
                        "message": f"Service '{service.name}' is already reachable.",
                        "status": "running",
                        "pid": service.get_pid(),
                    }

                if self._is_service_port_in_use(service):
                    self._mark_service_reachable_without_owned_process(
                        service,
                        reason="port is already in use",
                    )
                    dash_warning(
                        f"Start skipped because service port is already in use "
                        f"service='{service.name}' base_url='{service.base_url}'"
                    )
                    return {
                        "success": True,
                        "message": (
                            f"Service '{service.name}' port is already in use; "
                            "start skipped to prevent duplicate process."
                        ),
                        "status": "running",
                        "pid": service.get_pid(),
                    }

            service.start_in_progress = True
            service.last_known_status = "starting"

            try:
                if service.service_type == "process":
                    return self._start_process_service(service)

                if service.service_type == "command":
                    return self._start_command_service(service)

                dash_error(
                    f"Unknown service type for service='{name}' "
                    f"type='{service.service_type}'"
                )
                return {
                    "success": False,
                    "message": f"Unknown service type for '{name}'.",
                }

            finally:
                service.start_in_progress = False

    def stop_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)

        if not service:
            dash_warning(f"Stop requested for unknown service='{name}'")
            return {
                "success": False,
                "message": f"Service '{name}' not found.",
            }

        with service.operation_lock:
            dash_info(f"Stop requested for service='{name}'")

            if service.stop_in_progress:
                dash_warning(
                    f"Stop skipped; service is already stopping service='{service.name}'"
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' is already stopping.",
                    "status": "stopping",
                    "pid": service.get_pid(),
                }

            service.stop_in_progress = True
            service.last_known_status = "stopping"

            try:
                if service.service_type == "process":
                    return self._stop_process_service(service)

                if service.service_type == "command":
                    return self._stop_command_service(service)

                dash_error(
                    f"Unknown service type for service='{name}' "
                    f"type='{service.service_type}'"
                )
                return {
                    "success": False,
                    "message": f"Unknown service type for '{name}'.",
                }

            finally:
                service.stop_in_progress = False

    def restart_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)

        if not service:
            dash_warning(f"Restart requested for unknown service='{name}'")
            return {
                "success": False,
                "message": f"Service '{name}' not found.",
            }

        with service.operation_lock:
            dash_info(f"Restart requested for service='{name}'")

            if service.start_in_progress or service.stop_in_progress:
                return {
                    "success": True,
                    "message": (
                        f"Service '{service.name}' is already transitioning; "
                        "restart skipped."
                    ),
                    "status": service.get_status(),
                    "pid": service.get_pid(),
                }

            if service.is_running():
                stop_result = (
                    self._stop_process_service(service)
                    if service.service_type == "process"
                    else self._stop_command_service(service)
                )

                if not stop_result["success"]:
                    dash_warning(
                        f"Restart aborted because stop failed for service='{name}'"
                    )
                    return stop_result

                time.sleep(1)

            elif self._is_gpu_service_name(service) and self._is_gpu_service_reachable():
                self._mark_gpu_service_reachable_without_owned_process(
                    service,
                    reason="restart skipped because ServiceManager does not own the process handle",
                )
                dash_warning(
                    "GPU Service restart skipped because it is already reachable "
                    "but ServiceManager does not own the process handle."
                )
                return {
                    "success": True,
                    "message": (
                        "GPU Service is already running on port 5051. "
                        "Restart skipped because this dashboard does not own the "
                        "running process handle."
                    ),
                    "status": "running",
                    "pid": service.get_pid(),
                }

            service.start_in_progress = True
            service.last_known_status = "starting"

            try:
                if service.service_type == "process":
                    return self._start_process_service(service)

                if service.service_type == "command":
                    return self._start_command_service(service)

                return {
                    "success": False,
                    "message": f"Unknown service type for '{name}'.",
                }

            finally:
                service.start_in_progress = False

    # ------------------------------------------------------------------
    # Process services
    # ------------------------------------------------------------------

    def _start_process_service(self, service: ManagedService) -> dict[str, Any]:
        if service.is_running():
            dash_warning(f"Start skipped; service already running='{service.name}'")
            service.append_output(
                f"[SYSTEM] Start skipped; service '{service.name}' is already running."
            )
            return {
                "success": True,
                "message": f"Service '{service.name}' is already running.",
                "status": "running",
                "pid": service.get_pid(),
            }

        if self._is_gpu_service_name(service) and self._is_gpu_service_reachable():
            self._mark_gpu_service_reachable_without_owned_process(
                service,
                reason="process start skipped",
            )
            dash_warning(
                "GPU process start skipped because /api/status is already reachable."
            )
            return {
                "success": True,
                "message": "GPU Service is already running on port 5051.",
                "status": "running",
                "pid": service.get_pid(),
            }

        if not self._is_gpu_service_name(service):
            if self._is_service_reachable(service):
                self._mark_service_reachable_without_owned_process(
                    service,
                    reason="process start skipped",
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' is already reachable.",
                    "status": "running",
                    "pid": service.get_pid(),
                }

            if self._is_service_port_in_use(service):
                self._mark_service_reachable_without_owned_process(
                    service,
                    reason="port is already in use",
                )
                return {
                    "success": True,
                    "message": (
                        f"Service '{service.name}' port is already in use; "
                        "start skipped."
                    ),
                    "status": "running",
                    "pid": service.get_pid(),
                }

        if not service.command:
            dash_error(f"Start failed; no command defined for service='{service.name}'")
            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"No command defined for '{service.name}'.",
            }

        try:
            creationflags = 0
            preexec_fn = None

            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid

            dash_info(
                f"Starting process service='{service.name}' "
                f"command='{service.command}' cwd='{service.cwd}'"
            )

            process = subprocess.Popen(
                service.command,
                cwd=service.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
                shell=False,
                env=service.env,
            )

            service.process = process
            service.started_at = time.time()
            service.last_known_status = "running"
            service.append_output(
                f"[SYSTEM] Started service '{service.name}' with PID {process.pid}"
            )
            dash_info(
                f"Process service started service='{service.name}' pid='{process.pid}'"
            )

            threading.Thread(
                target=self._read_output,
                args=(service,),
                name=f"stdout-reader-{service.name}",
                daemon=True,
            ).start()

            service.start_log_tailing()

            return {
                "success": True,
                "message": f"Service '{service.name}' started successfully.",
                "status": "running",
                "pid": process.pid,
            }

        except Exception as exc:
            service.last_known_status = "stopped"
            dash_error(f"Failed to start process service='{service.name}' error={exc}")
            return {
                "success": False,
                "message": f"Failed to start service '{service.name}': {exc}",
            }

    def _stop_process_service(self, service: ManagedService) -> dict[str, Any]:
        if not service.is_running():
            if self._is_gpu_service_name(service) and self._is_gpu_service_reachable():
                service.last_known_status = "running"
                service.append_output(
                    "[SYSTEM] Stop requested, but GPU Service is running outside "
                    "this ServiceManager process. Stop skipped."
                )
                dash_warning(
                    "GPU Service stop skipped because it is reachable but "
                    "ServiceManager does not own the process handle."
                )
                return {
                    "success": False,
                    "message": (
                        "GPU Service is running on port 5051, but this dashboard "
                        "does not own the process handle. Stop it manually or "
                        "restart the dashboard after stopping the external process."
                    ),
                }

            dash_warning(f"Stop skipped; service not running='{service.name}'")
            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"Service '{service.name}' is not running.",
            }

        try:
            service.append_output(f"[SYSTEM] Stopping service '{service.name}'...")
            dash_info(f"Stopping process service='{service.name}'")

            if not service.process:
                service.last_known_status = "stopped"
                service.started_at = None
                service.stop_log_tailing()
                dash_info(
                    f"Process handle missing but service marked stopped='{service.name}'"
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' stopped successfully.",
                }

            if os.name == "nt":
                service.process.terminate()
            else:
                os.killpg(os.getpgid(service.process.pid), signal.SIGTERM)

            try:
                service.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service.append_output(
                    f"[SYSTEM] Force killing service '{service.name}'..."
                )
                dash_warning(
                    f"Force killing process service='{service.name}' after timeout"
                )

                if os.name == "nt":
                    service.process.kill()
                else:
                    os.killpg(os.getpgid(service.process.pid), signal.SIGKILL)

                service.process.wait(timeout=5)

            service.stop_log_tailing()
            service.append_output(f"[SYSTEM] Service '{service.name}' stopped.")
            service.process = None
            service.started_at = None
            service.last_known_status = "stopped"

            dash_info(f"Process service stopped service='{service.name}'")

            return {
                "success": True,
                "message": f"Service '{service.name}' stopped successfully.",
            }

        except Exception as exc:
            dash_error(f"Failed to stop process service='{service.name}' error={exc}")
            return {
                "success": False,
                "message": f"Failed to stop service '{service.name}': {exc}",
            }

    # ------------------------------------------------------------------
    # Command services
    # ------------------------------------------------------------------

    def _start_command_service(self, service: ManagedService) -> dict[str, Any]:
        if service.is_running():
            dash_warning(
                f"Start skipped; command service already running='{service.name}'"
            )
            service.append_output(
                f"[SYSTEM] Start skipped; service '{service.name}' is already running."
            )
            return {
                "success": True,
                "message": f"Service '{service.name}' is already running.",
                "status": "running",
            }

        if not service.start_command:
            dash_error(
                f"Start failed; no start_command defined for service='{service.name}'"
            )
            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"No start_command defined for '{service.name}'.",
            }

        try:
            service.append_output(
                f"[SYSTEM] Running start command for '{service.name}'..."
            )
            dash_info(
                f"Running start command for service='{service.name}' "
                f"command='{service.start_command}'"
            )

            result = self._run_command(
                service.start_command,
                cwd=service.cwd,
                timeout=60,
            )
            self._append_completed_process_output(service, result)

            if result.returncode == 0:
                service.started_at = time.time()
                service.last_known_status = "running"
                service.start_log_tailing()
                dash_info(
                    f"Command service started successfully service='{service.name}'"
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' started successfully.",
                    "status": "running",
                }

            service.last_known_status = "stopped"
            dash_warning(
                f"Start command failed for service='{service.name}' "
                f"returncode='{result.returncode}'"
            )
            return {
                "success": False,
                "message": (
                    f"Start command failed for '{service.name}' "
                    f"with code {result.returncode}."
                ),
            }

        except subprocess.TimeoutExpired:
            service.last_known_status = "stopped"
            dash_warning(f"Start command timed out for service='{service.name}'")
            return {
                "success": False,
                "message": f"Start command timed out for '{service.name}'.",
            }

        except Exception as exc:
            service.last_known_status = "stopped"
            dash_error(f"Failed to start command service='{service.name}' error={exc}")
            return {
                "success": False,
                "message": f"Failed to start service '{service.name}': {exc}",
            }

    def _stop_command_service(self, service: ManagedService) -> dict[str, Any]:
        if not service.is_running():
            dash_warning(
                f"Stop skipped; command service not running='{service.name}'"
            )
            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"Service '{service.name}' is not running.",
            }

        if not service.stop_command:
            dash_error(
                f"Stop failed; no stop_command defined for service='{service.name}'"
            )
            return {
                "success": False,
                "message": f"No stop_command defined for '{service.name}'.",
            }

        try:
            service.append_output(
                f"[SYSTEM] Running stop command for '{service.name}'..."
            )
            dash_info(
                f"Running stop command for service='{service.name}' "
                f"command='{service.stop_command}'"
            )

            result = self._run_command(
                service.stop_command,
                cwd=service.cwd,
                timeout=60,
            )
            self._append_completed_process_output(service, result)

            if result.returncode == 0:
                service.stop_log_tailing()
                service.started_at = None
                service.last_known_status = "stopped"
                dash_info(
                    f"Command service stopped successfully service='{service.name}'"
                )
                return {
                    "success": True,
                    "message": f"Service '{service.name}' stopped successfully.",
                    "status": "stopped",
                }

            dash_warning(
                f"Stop command failed for service='{service.name}' "
                f"returncode='{result.returncode}'"
            )
            return {
                "success": False,
                "message": (
                    f"Stop command failed for '{service.name}' "
                    f"with code {result.returncode}."
                ),
            }

        except subprocess.TimeoutExpired:
            dash_warning(f"Stop command timed out for service='{service.name}'")
            return {
                "success": False,
                "message": f"Stop command timed out for '{service.name}'.",
            }

        except Exception as exc:
            dash_error(f"Failed to stop command service='{service.name}' error={exc}")
            return {
                "success": False,
                "message": f"Failed to stop service '{service.name}': {exc}",
            }

    # ------------------------------------------------------------------
    # Output reader
    # ------------------------------------------------------------------

    def _read_output(self, service: ManagedService) -> None:
        if not service.process or not service.process.stdout:
            return

        try:
            for line in iter(service.process.stdout.readline, ""):
                if not line:
                    break
                service.append_output(line)

        except Exception as exc:
            service.append_output(f"[SYSTEM] Error reading output: {exc}")
            dash_warning(
                f"Error reading output for service='{service.name}' error={exc}"
            )

        finally:
            return_code = service.process.poll() if service.process else None

            service.append_output(
                f"[SYSTEM] Process exited with return code: {return_code}"
            )
            dash_info(
                f"Process exited for service='{service.name}' "
                f"return_code='{return_code}'"
            )

            if self._is_gpu_service_name(service) and self._is_gpu_service_reachable():
                service.append_output(
                    "[SYSTEM] GPU Service API is still reachable after output "
                    "reader ended; preserving running status."
                )
                service.process = None
                service.last_known_status = "running"

                if service.started_at is None:
                    service.started_at = time.time()

                service.stop_log_tailing()
                return

            if service.service_type == "process" and self._is_service_reachable(service):
                service.append_output(
                    "[SYSTEM] Service API is still reachable after output "
                    "reader ended; preserving running status."
                )
                service.process = None
                service.last_known_status = "running"

                if service.started_at is None:
                    service.started_at = time.time()

                service.stop_log_tailing()
                return

            service.process = None
            service.started_at = None
            service.last_known_status = "stopped"
            service.stop_log_tailing()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_service_data(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []

        for name, service in self.services.items():
            data.append(
                {
                    "name": name,
                    "service_type": service.service_type,
                    "status": self._get_effective_status(service),
                    "pid": self._get_effective_pid(service),
                    "uptime_seconds": self._get_effective_uptime_seconds(service),
                    "command": " ".join(service.command) if service.command else "",
                    "start_command": (
                        " ".join(service.start_command)
                        if service.start_command
                        else ""
                    ),
                    "stop_command": (
                        " ".join(service.stop_command)
                        if service.stop_command
                        else ""
                    ),
                    "status_command": (
                        " ".join(service.status_command)
                        if service.status_command
                        else ""
                    ),
                    "cwd": service.cwd,
                    "base_url": service.base_url,
                    "log_file": service.log_file,
                    "output": service.get_output(),
                }
            )

        dash_debug(f"Collected service data for count='{len(data)}'")
        return data

    def get_service(self, name: str) -> Optional[ManagedService]:
        service = self.services.get(name)

        if service is None:
            dash_debug(f"Service lookup missed for name='{name}'")
        else:
            dash_debug(f"Service lookup hit for name='{name}'")

        return service