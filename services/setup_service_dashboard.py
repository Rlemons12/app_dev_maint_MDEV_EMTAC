from __future__ import annotations

import json
import textwrap
from pathlib import Path


APP_PY = r'''
from __future__ import annotations

import json
import os
from flask import Flask, jsonify, render_template
from service_manager import ServiceManager


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "services_config.json")

app = Flask(__name__)
service_manager = ServiceManager()


def load_services() -> None:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    services = config.get("services", [])
    for svc in services:
        service_manager.register_service(svc)


load_services()


@app.route("/")
def dashboard():
    return render_template("service_dashboard.html")


@app.route("/api/services", methods=["GET"])
def get_services():
    return jsonify({
        "success": True,
        "services": service_manager.get_service_data()
    })


@app.route("/api/services/<service_name>/start", methods=["POST"])
def start_service(service_name):
    result = service_manager.start_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/stop", methods=["POST"])
def stop_service(service_name):
    result = service_manager.stop_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/restart", methods=["POST"])
def restart_service(service_name):
    result = service_manager.restart_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/clear-output", methods=["POST"])
def clear_output(service_name):
    service = service_manager.get_service(service_name)
    if not service:
        return jsonify({
            "success": False,
            "message": f"Service '{service_name}' not found."
        }), 404

    service.clear_output()
    return jsonify({
        "success": True,
        "message": f"Output cleared for '{service_name}'."
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)
'''

SERVICE_MANAGER_PY = r'''
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional


class ManagedService:
    def __init__(self, config: dict[str, Any]):
        self.name: str = config["name"]
        self.service_type: str = config.get("service_type", "process")
        self.cwd: Optional[str] = config.get("cwd")

        self.command: list[str] = config.get("command", [])
        self.start_command: list[str] = config.get("start_command", [])
        self.stop_command: list[str] = config.get("stop_command", [])
        self.status_command: list[str] = config.get("status_command", [])

        self.log_file: Optional[str] = config.get("log_file")
        self.auto_tail_log: bool = config.get("auto_tail_log", False)

        self.process: Optional[subprocess.Popen] = None
        self.output_buffer = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.started_at: Optional[float] = None
        self.last_known_status: str = "stopped"

        self.log_tail_thread: Optional[threading.Thread] = None
        self.log_tail_stop_event = threading.Event()

    def append_output(self, line: str) -> None:
        with self.lock:
            self.output_buffer.append(line.rstrip())

    def get_output(self) -> list[str]:
        with self.lock:
            return list(self.output_buffer)

    def clear_output(self) -> None:
        with self.lock:
            self.output_buffer.clear()

    def is_running(self) -> bool:
        if self.service_type == "process":
            return self.process is not None and self.process.poll() is None

        if self.service_type == "command":
            if not self.status_command:
                return self.last_known_status == "running"

            try:
                result = subprocess.run(
                    self.status_command,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    shell=False
                )

                output = ((result.stdout or "") + "\\n" + (result.stderr or "")).lower()

                if result.returncode == 0:
                    self.last_known_status = "running"
                    return True

                if "server is running" in output:
                    self.last_known_status = "running"
                    return True

                if "no server running" in output:
                    self.last_known_status = "stopped"
                    return False

                return self.last_known_status == "running"

            except Exception as exc:
                self.append_output(f"[SYSTEM] Status check failed: {exc}")
                return self.last_known_status == "running"

        return False

    def get_pid(self) -> Optional[int]:
        if self.service_type == "process" and self.process and self.is_running():
            return self.process.pid
        return None

    def get_status(self) -> str:
        return "running" if self.is_running() else "stopped"

    def get_uptime_seconds(self) -> Optional[int]:
        if self.started_at and self.is_running():
            return int(time.time() - self.started_at)
        return None

    def start_log_tailing(self) -> None:
        if not self.log_file or not self.auto_tail_log:
            return

        if self.log_tail_thread and self.log_tail_thread.is_alive():
            return

        self.log_tail_stop_event.clear()
        self.log_tail_thread = threading.Thread(target=self._tail_log_file, daemon=True)
        self.log_tail_thread.start()

    def stop_log_tailing(self) -> None:
        self.log_tail_stop_event.set()

    def _tail_log_file(self) -> None:
        log_path = Path(self.log_file)
        self.append_output(f"[SYSTEM] Starting log tail: {log_path}")

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
                time.sleep(2)

        self.append_output(f"[SYSTEM] Stopped log tail: {log_path}")


class ServiceManager:
    def __init__(self):
        self.services: Dict[str, ManagedService] = {}

    def register_service(self, config: dict[str, Any]) -> None:
        name = config["name"]
        if name in self.services:
            raise ValueError(f"Service '{name}' is already registered.")
        self.services[name] = ManagedService(config=config)

    def _run_command(self, command: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=False
        )

    def start_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)
        if not service:
            return {"success": False, "message": f"Service '{name}' not found."}

        if service.service_type == "process":
            return self._start_process_service(service)

        if service.service_type == "command":
            return self._start_command_service(service)

        return {"success": False, "message": f"Unknown service type for '{name}'."}

    def stop_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)
        if not service:
            return {"success": False, "message": f"Service '{name}' not found."}

        if service.service_type == "process":
            return self._stop_process_service(service)

        if service.service_type == "command":
            return self._stop_command_service(service)

        return {"success": False, "message": f"Unknown service type for '{name}'."}

    def restart_service(self, name: str) -> dict[str, Any]:
        service = self.services.get(name)
        if not service:
            return {"success": False, "message": f"Service '{name}' not found."}

        if service.is_running():
            stop_result = self.stop_service(name)
            if not stop_result["success"]:
                return stop_result
            time.sleep(1)

        return self.start_service(name)

    def _start_process_service(self, service: ManagedService) -> dict[str, Any]:
        if service.is_running():
            return {"success": False, "message": f"Service '{service.name}' is already running."}

        if not service.command:
            return {"success": False, "message": f"No command defined for '{service.name}'."}

        try:
            creationflags = 0
            preexec_fn = None

            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid

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
                shell=False
            )

            service.process = process
            service.started_at = time.time()
            service.last_known_status = "running"
            service.append_output(f"[SYSTEM] Started service '{service.name}' with PID {process.pid}")

            threading.Thread(
                target=self._read_output,
                args=(service,),
                daemon=True
            ).start()

            service.start_log_tailing()

            return {"success": True, "message": f"Service '{service.name}' started successfully."}

        except Exception as exc:
            return {"success": False, "message": f"Failed to start service '{service.name}': {exc}"}

    def _stop_process_service(self, service: ManagedService) -> dict[str, Any]:
        if not service.is_running():
            return {"success": False, "message": f"Service '{service.name}' is not running."}

        try:
            service.append_output(f"[SYSTEM] Stopping service '{service.name}'...")

            if os.name == "nt":
                service.process.terminate()
            else:
                os.killpg(os.getpgid(service.process.pid), signal.SIGTERM)

            try:
                service.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service.append_output(f"[SYSTEM] Force killing service '{service.name}'...")
                if os.name == "nt":
                    service.process.kill()
                else:
                    os.killpg(os.getpgid(service.process.pid), signal.SIGKILL)

            service.stop_log_tailing()
            service.append_output(f"[SYSTEM] Service '{service.name}' stopped.")
            service.process = None
            service.started_at = None
            service.last_known_status = "stopped"

            return {"success": True, "message": f"Service '{service.name}' stopped successfully."}

        except Exception as exc:
            return {"success": False, "message": f"Failed to stop service '{service.name}': {exc}"}

    def _start_command_service(self, service: ManagedService) -> dict[str, Any]:
        if service.is_running():
            return {"success": False, "message": f"Service '{service.name}' is already running."}

        if not service.start_command:
            return {"success": False, "message": f"No start_command defined for '{service.name}'."}

        try:
            service.append_output(f"[SYSTEM] Running start command for '{service.name}'...")
            result = self._run_command(service.start_command, cwd=service.cwd)

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

            if result.returncode == 0:
                service.started_at = time.time()
                service.last_known_status = "running"
                service.start_log_tailing()
                return {"success": True, "message": f"Service '{service.name}' started successfully."}

            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"Start command failed for '{service.name}' with code {result.returncode}."
            }

        except Exception as exc:
            return {"success": False, "message": f"Failed to start service '{service.name}': {exc}"}

    def _stop_command_service(self, service: ManagedService) -> dict[str, Any]:
        if not service.is_running():
            return {"success": False, "message": f"Service '{service.name}' is not running."}

        if not service.stop_command:
            return {"success": False, "message": f"No stop_command defined for '{service.name}'."}

        try:
            service.append_output(f"[SYSTEM] Running stop command for '{service.name}'...")
            result = self._run_command(service.stop_command, cwd=service.cwd)

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

            if result.returncode == 0:
                service.stop_log_tailing()
                service.started_at = None
                service.last_known_status = "stopped"
                return {"success": True, "message": f"Service '{service.name}' stopped successfully."}

            return {
                "success": False,
                "message": f"Stop command failed for '{service.name}' with code {result.returncode}."
            }

        except Exception as exc:
            return {"success": False, "message": f"Failed to stop service '{service.name}': {exc}"}

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
        finally:
            return_code = service.process.poll() if service.process else None
            service.append_output(f"[SYSTEM] Process exited with return code: {return_code}")
            service.process = None
            service.started_at = None
            service.last_known_status = "stopped"
            service.stop_log_tailing()

    def get_service_data(self) -> list[dict[str, Any]]:
        data = []
        for name, service in self.services.items():
            data.append({
                "name": name,
                "service_type": service.service_type,
                "status": service.get_status(),
                "pid": service.get_pid(),
                "uptime_seconds": service.get_uptime_seconds(),
                "command": " ".join(service.command) if service.command else "",
                "start_command": " ".join(service.start_command) if service.start_command else "",
                "stop_command": " ".join(service.stop_command) if service.stop_command else "",
                "status_command": " ".join(service.status_command) if service.status_command else "",
                "cwd": service.cwd,
                "log_file": service.log_file,
                "output": service.get_output(),
            })
        return data

    def get_service(self, name: str):
        return self.services.get(name)
'''

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMTAC Service Dashboard</title>
</head>
<body>
    Replace this with the full HTML from the response.
</body>
</html>
'''

SERVICES_CONFIG = {
    "services": [
        {
            "name": "GPU Service",
            "service_type": "process",
            "command": [
                r"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe",
                "main.py"
            ],
            "cwd": r"E:\emtac\services\gpu"
        },
        {
            "name": "PostgreSQL Server",
            "service_type": "command",
            "start_command": [
                r"C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe",
                "start",
                "-D",
                r"E:\emtac\postgres\data",
                "-l",
                r"E:\emtac\postgres\logs\postgresql.log",
                "-w"
            ],
            "stop_command": [
                r"C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe",
                "stop",
                "-D",
                r"E:\emtac\postgres\data",
                "-m",
                "fast",
                "-w"
            ],
            "status_command": [
                r"C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe",
                "status",
                "-D",
                r"E:\emtac\postgres\data"
            ],
            "cwd": r"E:\emtac\services",
            "log_file": r"E:\emtac\postgres\logs\postgresql.log",
            "auto_tail_log": True
        }
    ]
}

START_BAT = r'''
@echo off
"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe" "E:\emtac\services\service_dashboard\app.py"
pause
'''

CHECK_FLASK_BAT = r'''
@echo off
"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe" -c "import flask; print('Flask OK:', flask.__version__)"
pause
'''


def write_text_file(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print(f"[OK] Wrote file: {path}")


def write_json_file(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    print(f"[OK] Wrote file: {path}")


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    dashboard_dir = root_dir / "service_dashboard"
    templates_dir = dashboard_dir / "templates"
    logs_dir = dashboard_dir / "logs"

    dashboard_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    write_text_file(dashboard_dir / "app.py", APP_PY)
    write_text_file(dashboard_dir / "service_manager.py", SERVICE_MANAGER_PY)
    write_text_file(templates_dir / "service_dashboard.html", HTML_TEMPLATE)
    write_json_file(dashboard_dir / "services_config.json", SERVICES_CONFIG)
    write_text_file(dashboard_dir / "start_dashboard.bat", START_BAT)
    write_text_file(dashboard_dir / "check_flask_in_gpu_venv.bat", CHECK_FLASK_BAT)

    print("[DONE] Service dashboard setup complete.")


if __name__ == "__main__":
    main()