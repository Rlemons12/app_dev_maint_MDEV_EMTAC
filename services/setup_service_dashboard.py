from __future__ import annotations

import json
import textwrap
from pathlib import Path


APP_PY = r'''import json
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
    return jsonify(result)


@app.route("/api/services/<service_name>/stop", methods=["POST"])
def stop_service(service_name):
    result = service_manager.stop_service(service_name)
    return jsonify(result)


@app.route("/api/services/<service_name>/restart", methods=["POST"])
def restart_service(service_name):
    result = service_manager.restart_service(service_name)
    return jsonify(result)


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


SERVICE_MANAGER_PY = r'''import os
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Dict, Optional, Any


class ManagedService:
    def __init__(self, config: dict[str, Any]):
        self.name: str = config["name"]
        self.service_type: str = config.get("service_type", "process")
        self.cwd: Optional[str] = config.get("cwd")
        self.command: list[str] = config.get("command", [])
        self.start_command: list[str] = config.get("start_command", [])
        self.stop_command: list[str] = config.get("stop_command", [])
        self.status_command: list[str] = config.get("status_command", [])

        self.process: Optional[subprocess.Popen] = None
        self.output_buffer = deque(maxlen=500)
        self.lock = threading.Lock()
        self.started_at: Optional[float] = None
        self.last_known_status: str = "stopped"

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
                output = (result.stdout or "") + "\n" + (result.stderr or "")
                output = output.lower()

                # Common pg_ctl status patterns
                if result.returncode == 0:
                    self.last_known_status = "running"
                    return True

                if "server is running" in output:
                    self.last_known_status = "running"
                    return True

                self.last_known_status = "stopped"
                return False

            except Exception as exc:
                self.append_output(f"[SYSTEM] Status check failed: {exc}")
                return self.last_known_status == "running"

        return False

    def get_pid(self) -> Optional[int]:
        if self.service_type == "process":
            if self.process and self.is_running():
                return self.process.pid
        return None

    def get_status(self) -> str:
        return "running" if self.is_running() else "stopped"

    def get_uptime_seconds(self) -> Optional[int]:
        if self.started_at and self.is_running():
            return int(time.time() - self.started_at)
        return None


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

    def start_service(self, name: str) -> dict:
        if name not in self.services:
            return {"success": False, "message": f"Service '{name}' not found."}

        service = self.services[name]

        if service.service_type == "process":
            return self._start_process_service(service)

        if service.service_type == "command":
            return self._start_command_service(service)

        return {"success": False, "message": f"Unknown service type for '{name}'."}

    def stop_service(self, name: str) -> dict:
        if name not in self.services:
            return {"success": False, "message": f"Service '{name}' not found."}

        service = self.services[name]

        if service.service_type == "process":
            return self._stop_process_service(service)

        if service.service_type == "command":
            return self._stop_command_service(service)

        return {"success": False, "message": f"Unknown service type for '{name}'."}

    def restart_service(self, name: str) -> dict:
        if name not in self.services:
            return {"success": False, "message": f"Service '{name}' not found."}

        service = self.services[name]

        if service.is_running():
            stop_result = self.stop_service(name)
            if not stop_result["success"]:
                return stop_result

        return self.start_service(name)

    def _start_process_service(self, service: ManagedService) -> dict:
        if service.is_running():
            return {"success": False, "message": f"Service '{service.name}' is already running."}

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

            return {"success": True, "message": f"Service '{service.name}' started successfully."}

        except Exception as exc:
            return {"success": False, "message": f"Failed to start service '{service.name}': {exc}"}

    def _stop_process_service(self, service: ManagedService) -> dict:
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

            service.append_output(f"[SYSTEM] Service '{service.name}' stopped.")
            service.process = None
            service.started_at = None
            service.last_known_status = "stopped"

            return {"success": True, "message": f"Service '{service.name}' stopped successfully."}

        except Exception as exc:
            return {"success": False, "message": f"Failed to stop service '{service.name}': {exc}"}

    def _start_command_service(self, service: ManagedService) -> dict:
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
                return {"success": True, "message": f"Service '{service.name}' started successfully."}

            service.last_known_status = "stopped"
            return {
                "success": False,
                "message": f"Start command failed for '{service.name}' with code {result.returncode}."
            }

        except Exception as exc:
            return {"success": False, "message": f"Failed to start service '{service.name}': {exc}"}

    def _stop_command_service(self, service: ManagedService) -> dict:
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
            for line in iter(service.process.stdout.readline, ''):
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

    def get_service_data(self) -> list[dict]:
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
                "output": service.get_output(),
            })
        return data

    def get_service(self, name: str):
        return self.services.get(name)
'''


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMTAC Service Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 20px;
        }

        h1 {
            margin-bottom: 10px;
        }

        .subheading {
            margin-bottom: 20px;
            color: #444;
        }

        .services-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }

        .service-card {
            background: white;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border-left: 6px solid #999;
        }

        .service-card.running {
            border-left-color: #28a745;
        }

        .service-card.stopped {
            border-left-color: #dc3545;
        }

        .service-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }

        .service-title {
            font-size: 20px;
            font-weight: bold;
        }

        .status-badge {
            padding: 6px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            color: white;
        }

        .status-running {
            background: #28a745;
        }

        .status-stopped {
            background: #dc3545;
        }

        .service-meta {
            margin-top: 10px;
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }

        .service-actions {
            margin-top: 14px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            border-radius: 6px;
            padding: 10px 14px;
            cursor: pointer;
            font-weight: bold;
        }

        .btn-start {
            background: #28a745;
            color: white;
        }

        .btn-stop {
            background: #dc3545;
            color: white;
        }

        .btn-restart {
            background: #007bff;
            color: white;
        }

        .btn-clear {
            background: #6c757d;
            color: white;
        }

        .output-box {
            margin-top: 16px;
            background: #111;
            color: #00ff88;
            padding: 12px;
            border-radius: 8px;
            height: 300px;
            overflow-y: auto;
            font-family: Consolas, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            border: 1px solid #222;
        }

        .message-bar {
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 6px;
            display: none;
        }

        .message-success {
            background: #d4edda;
            color: #155724;
        }

        .message-error {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <h1>EMTAC Service Dashboard</h1>
    <div class="subheading">Start, stop, restart, and inspect service output.</div>

    <div id="messageBar" class="message-bar"></div>
    <div id="servicesContainer" class="services-grid"></div>

    <script>
        const servicesContainer = document.getElementById("servicesContainer");
        const messageBar = document.getElementById("messageBar");

        function showMessage(message, isSuccess = true) {
            messageBar.textContent = message;
            messageBar.className = "message-bar " + (isSuccess ? "message-success" : "message-error");
            messageBar.style.display = "block";

            setTimeout(() => {
                messageBar.style.display = "none";
            }, 3000);
        }

        function formatUptime(seconds) {
            if (seconds === null || seconds === undefined) {
                return "N/A";
            }

            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            return `${hrs}h ${mins}m ${secs}s`;
        }

        function escapeHtml(text) {
            const div = document.createElement("div");
            div.innerText = text;
            return div.innerHTML;
        }

        async function fetchServices() {
            try {
                const response = await fetch("/api/services");
                const data = await response.json();

                if (!data.success) {
                    showMessage("Failed to fetch services.", false);
                    return;
                }

                renderServices(data.services);
            } catch (error) {
                showMessage("Error loading services: " + error.message, false);
            }
        }

        function renderServices(services) {
            servicesContainer.innerHTML = "";

            services.forEach(service => {
                const card = document.createElement("div");
                card.className = `service-card ${service.status}`;

                const outputText = service.output.length
                    ? service.output.join("\\n")
                    : "No output yet.";

                const commandText = service.service_type === "process"
                    ? service.command
                    : service.start_command;

                card.innerHTML = `
                    <div class="service-header">
                        <div class="service-title">${service.name}</div>
                        <div class="status-badge ${service.status === 'running' ? 'status-running' : 'status-stopped'}">
                            ${service.status}
                        </div>
                    </div>

                    <div class="service-meta">
                        <div><strong>Type:</strong> ${service.service_type}</div>
                        <div><strong>PID:</strong> ${service.pid ?? "N/A"}</div>
                        <div><strong>Uptime:</strong> ${formatUptime(service.uptime_seconds)}</div>
                        <div><strong>Command:</strong> ${commandText || "N/A"}</div>
                        <div><strong>Working Dir:</strong> ${service.cwd ?? "N/A"}</div>
                    </div>

                    <div class="service-actions">
                        <button class="btn-start" onclick="startService('${service.name}')">Start</button>
                        <button class="btn-stop" onclick="stopService('${service.name}')">Stop</button>
                        <button class="btn-restart" onclick="restartService('${service.name}')">Restart</button>
                        <button class="btn-clear" onclick="clearOutput('${service.name}')">Clear Output</button>
                    </div>

                    <div class="output-box">${escapeHtml(outputText)}</div>
                `;

                servicesContainer.appendChild(card);
            });
        }

        async function startService(serviceName) {
            try {
                const response = await fetch(`/api/services/${encodeURIComponent(serviceName)}/start`, {
                    method: "POST"
                });
                const data = await response.json();
                showMessage(data.message, data.success);
                await fetchServices();
            } catch (error) {
                showMessage("Start failed: " + error.message, false);
            }
        }

        async function stopService(serviceName) {
            try {
                const response = await fetch(`/api/services/${encodeURIComponent(serviceName)}/stop`, {
                    method: "POST"
                });
                const data = await response.json();
                showMessage(data.message, data.success);
                await fetchServices();
            } catch (error) {
                showMessage("Stop failed: " + error.message, false);
            }
        }

        async function restartService(serviceName) {
            try {
                const response = await fetch(`/api/services/${encodeURIComponent(serviceName)}/restart`, {
                    method: "POST"
                });
                const data = await response.json();
                showMessage(data.message, data.success);
                await fetchServices();
            } catch (error) {
                showMessage("Restart failed: " + error.message, false);
            }
        }

        async function clearOutput(serviceName) {
            try {
                const response = await fetch(`/api/services/${encodeURIComponent(serviceName)}/clear-output`, {
                    method: "POST"
                });
                const data = await response.json();
                showMessage(data.message, data.success);
                await fetchServices();
            } catch (error) {
                showMessage("Clear failed: " + error.message, false);
            }
        }

        fetchServices();
        setInterval(fetchServices, 3000);
    </script>
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
            "cwd": r"E:\emtac\services"
        }
    ]
}


START_BAT = r'''@echo off
"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe" "E:\emtac\services\service_dashboard\app.py"
pause
'''

CHECK_FLASK_BAT = r'''@echo off
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

    gpu_python = Path(r"E:\emtac\services\gpu\.venv_gpu\Scripts\python.exe")

    print(f"[INFO] Root directory: {root_dir}")
    print(f"[INFO] Dashboard directory: {dashboard_dir}")

    dashboard_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print(f"[OK] Ensured directory exists: {dashboard_dir}")
    print(f"[OK] Ensured directory exists: {templates_dir}")
    print(f"[OK] Ensured directory exists: {logs_dir}")

    write_text_file(dashboard_dir / "app.py", APP_PY)
    write_text_file(dashboard_dir / "service_manager.py", SERVICE_MANAGER_PY)
    write_text_file(templates_dir / "service_dashboard.html", HTML_TEMPLATE)
    write_json_file(dashboard_dir / "services_config.json", SERVICES_CONFIG)
    write_text_file(dashboard_dir / "start_dashboard.bat", START_BAT)
    write_text_file(dashboard_dir / "check_flask_in_gpu_venv.bat", CHECK_FLASK_BAT)

    if gpu_python.exists():
        print(f"[OK] Found GPU venv python: {gpu_python}")
    else:
        print(f"[WARN] GPU venv python not found: {gpu_python}")

    print()
    print("=" * 70)
    print("[DONE] Service dashboard setup complete.")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  {dashboard_dir / 'app.py'}")
    print(f"  {dashboard_dir / 'service_manager.py'}")
    print(f"  {dashboard_dir / 'services_config.json'}")
    print(f"  {dashboard_dir / 'templates' / 'service_dashboard.html'}")
    print(f"  {dashboard_dir / 'start_dashboard.bat'}")
    print()
    print("IMPORTANT:")
    print("Edit PostgreSQL paths in services_config.json before using the dashboard.")
    print()
    print("Next steps:")
    print("1. Make sure Flask exists in E:\\emtac\\services\\gpu\\.venv_gpu")
    print("2. Edit services_config.json if needed")
    print("3. Start the dashboard by double-clicking:")
    print(f"   {dashboard_dir / 'start_dashboard.bat'}")
    print()
    print("Then open:")
    print("   http://127.0.0.1:5050")
    print()


if __name__ == "__main__":
    main()