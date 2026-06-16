from __future__ import annotations

import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify

from configuration.config import (
    PG_CTL_EXE,
    POSTGRES_BIN_DIR,
    POSTGRES_DATA_DIR,
    POSTGRES_LOG_FILE,
    SERVICE_AI_GATEWAY_BASE_URL,
    SERVICE_AI_GATEWAY_CWD,
    SERVICE_AI_GATEWAY_ENTRY,
    SERVICE_AI_GATEWAY_PYTHON,
    SERVICE_DASHBOARD_HOST,
    SERVICE_DASHBOARD_PORT,
    SERVICE_GPU_BASE_URL,
    SERVICE_GPU_CWD,
    SERVICE_GPU_ENTRY,
    SERVICE_GPU_PYTHON,
    SERVICE_MCP_AI_LAYER_BASE_URL,
    SERVICE_MCP_AI_LAYER_CWD,
    SERVICE_MCP_AI_LAYER_ENTRY,
    SERVICE_MCP_AI_LAYER_HEALTH_URL,
    SERVICE_MCP_AI_LAYER_PYTHON,
    SERVICE_MCP_COORDINATOR_BASE_URL,
    SERVICE_MCP_COORDINATOR_CWD,
    SERVICE_MCP_COORDINATOR_ENTRY,
    SERVICE_MCP_COORDINATOR_PYTHON,
    SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32,
    SERVICE_MCP_COORDINATOR_SITE_PACKAGES,
    SERVICE_MCP_COORDINATOR_WIN32_DIR,
    SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR,
    GRAFANA_URL,
    GRAFANA_EXE,
    GRAFANA_CWD,
)
from app.api import ai_bp, init_ai_blueprint, dashboard_bp
from app.services.gpu_insights_service import get_gpu_service_insights
from app.services.postgres_insights_service import get_postgres_service_insights
from app.services.service_dashboard_logger import (
    dash_error,
    dash_info,
    dash_warning,
)
from app.services.service_manager import ServiceManager


app = Flask(__name__)
service_manager = ServiceManager()


SERVICE_START_ORDER = [
    "PostgreSQL Server",
    "Grafana",
    "GPU Service",
    "EMTAC MCP Coordinator",
    "AI Gateway",
    "MCP AI Layer",
]

SERVICE_STOP_ORDER = list(reversed(SERVICE_START_ORDER))


def _sleep_between_services(seconds: float = 1.0) -> None:
    time.sleep(seconds)


def _service_snapshot() -> list[dict[str, Any]]:
    return service_manager.get_service_data()


def _find_service_snapshot(name: str) -> dict[str, Any] | None:
    for service in _service_snapshot():
        if service.get("name") == name:
            return service
    return None


def _service_is_running(name: str) -> bool:
    service = _find_service_snapshot(name)
    return bool(service and service.get("status") == "running")


def _probe_url(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        req = Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(req, timeout=timeout) as response:
            return {
                "reachable": True,
                "status_code": response.status,
                "error": None,
            }

    except HTTPError as exc:
        return {
            "reachable": True,
            "status_code": exc.code,
            "error": f"HTTP {exc.code}",
        }

    except URLError as exc:
        return {
            "reachable": False,
            "status_code": None,
            "error": f"Connection error: {exc.reason}",
        }

    except Exception as exc:
        return {
            "reachable": False,
            "status_code": None,
            "error": str(exc),
        }


# ---------------------------------------------------------
# Service registration
# ---------------------------------------------------------

def load_services() -> None:
    dash_info(
        f"Registering PostgreSQL Server | "
        f"bin_dir={POSTGRES_BIN_DIR} | "
        f"data_dir={POSTGRES_DATA_DIR}"
    )

    service_manager.register_service(
        {
            "name": "PostgreSQL Server",
            "service_type": "command",
            "start_command": [
                str(PG_CTL_EXE),
                "-D",
                str(POSTGRES_DATA_DIR),
                "-l",
                str(POSTGRES_LOG_FILE),
                "start",
                "-w",
            ],
            "stop_command": [
                str(PG_CTL_EXE),
                "-D",
                str(POSTGRES_DATA_DIR),
                "stop",
                "-m",
                "fast",
                "-w",
            ],
            "status_command": [
                str(PG_CTL_EXE),
                "-D",
                str(POSTGRES_DATA_DIR),
                "status",
            ],
            "cwd": str(POSTGRES_BIN_DIR),
            "log_file": str(POSTGRES_LOG_FILE),
            "auto_tail_log": True,
        }
    )

    dash_info(
        f"Registering Grafana | "
        f"exe={GRAFANA_EXE} | "
        f"cwd={GRAFANA_CWD} | "
        f"url={GRAFANA_URL}"
    )

    service_manager.register_service(
        {
            "name": "Grafana",
            "service_type": "process",
            "command": [
                str(GRAFANA_EXE),
                "server",
                "--homepath",
                str(GRAFANA_CWD),
            ],
            "cwd": str(GRAFANA_CWD),
            "base_url": GRAFANA_URL,
            "health_path": "/api/health",
        }
    )

    dash_info(
        f"Registering GPU Service | "
        f"python={SERVICE_GPU_PYTHON} | "
        f"cwd={SERVICE_GPU_CWD} | "
        f"entry={SERVICE_GPU_ENTRY} | "
        f"url={SERVICE_GPU_BASE_URL}"
    )

    service_manager.register_service(
        {
            "name": "GPU Service",
            "service_type": "process",
            "command": [
                str(SERVICE_GPU_PYTHON),
                SERVICE_GPU_ENTRY,
            ],
            "cwd": str(SERVICE_GPU_CWD),
            "base_url": SERVICE_GPU_BASE_URL,
            "health_path": "/api/status",
        }
    )

    dash_info(
        f"Registering EMTAC MCP Coordinator | "
        f"python={SERVICE_MCP_COORDINATOR_PYTHON} | "
        f"cwd={SERVICE_MCP_COORDINATOR_CWD} | "
        f"entry={SERVICE_MCP_COORDINATOR_ENTRY} | "
        f"url={SERVICE_MCP_COORDINATOR_BASE_URL}"
    )

    mcp_command = [
        str(SERVICE_MCP_COORDINATOR_PYTHON),
        *str(SERVICE_MCP_COORDINATOR_ENTRY).split(),
    ]

    mcp_env = __import__("os").environ.copy()
    mcp_env["PATH"] = (
        f"{SERVICE_MCP_COORDINATOR_PYWIN32_SYSTEM32};"
        f"{SERVICE_MCP_COORDINATOR_WIN32_DIR};"
        f"{mcp_env.get('PATH', '')}"
    )
    mcp_env["PYTHONPATH"] = (
        f"{SERVICE_MCP_COORDINATOR_WIN32_LIB_DIR};"
        f"{SERVICE_MCP_COORDINATOR_WIN32_DIR};"
        f"{SERVICE_MCP_COORDINATOR_SITE_PACKAGES};"
        f"{mcp_env.get('PYTHONPATH', '')}"
    )

    service_manager.register_service(
        {
            "name": "EMTAC MCP Coordinator",
            "service_type": "process",
            "command": mcp_command,
            "cwd": str(SERVICE_MCP_COORDINATOR_CWD),
            "base_url": SERVICE_MCP_COORDINATOR_BASE_URL,
            "health_path": "/mcp",
            "env": mcp_env,
        }
    )

    dash_info(
        f"Registering AI Gateway | "
        f"python={SERVICE_AI_GATEWAY_PYTHON} | "
        f"cwd={SERVICE_AI_GATEWAY_CWD} | "
        f"entry={SERVICE_AI_GATEWAY_ENTRY} | "
        f"url={SERVICE_AI_GATEWAY_BASE_URL}"
    )

    service_manager.register_service(
        {
            "name": "AI Gateway",
            "service_type": "process",
            "command": [
                str(SERVICE_AI_GATEWAY_PYTHON),
                SERVICE_AI_GATEWAY_ENTRY,
            ],
            "cwd": str(SERVICE_AI_GATEWAY_CWD),
            "base_url": SERVICE_AI_GATEWAY_BASE_URL,
            "health_path": "/health",
        }
    )

    dash_info(
        f"Registering MCP AI Layer | "
        f"python={SERVICE_MCP_AI_LAYER_PYTHON} | "
        f"cwd={SERVICE_MCP_AI_LAYER_CWD} | "
        f"entry={SERVICE_MCP_AI_LAYER_ENTRY} | "
        f"url={SERVICE_MCP_AI_LAYER_BASE_URL}"
    )

    mcp_ai_layer_command = [
        str(SERVICE_MCP_AI_LAYER_PYTHON),
        *str(SERVICE_MCP_AI_LAYER_ENTRY).split(),
    ]

    service_manager.register_service(
        {
            "name": "MCP AI Layer",
            "service_type": "process",
            "command": mcp_ai_layer_command,
            "cwd": str(SERVICE_MCP_AI_LAYER_CWD),
            "base_url": SERVICE_MCP_AI_LAYER_BASE_URL,
            "health_path": "/api/ai/health",
        }
    )


# ---------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------

load_services()

init_ai_blueprint(service_manager)
app.register_blueprint(ai_bp)
app.register_blueprint(dashboard_bp)

dash_info("Status A.I. blueprint registered | prefix=/api/ai")
dash_info("Dashboard blueprint registered")


# ---------------------------------------------------------
# Service API
# ---------------------------------------------------------

@app.route("/api/services", methods=["GET"])
def get_services():
    return jsonify(
        {
            "success": True,
            "services": _service_snapshot(),
        }
    )


@app.route("/api/services/<service_name>/start", methods=["POST"])
def start_service(service_name: str):
    result = service_manager.start_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/stop", methods=["POST"])
def stop_service(service_name: str):
    result = service_manager.stop_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/restart", methods=["POST"])
def restart_service(service_name: str):
    result = service_manager.restart_service(service_name)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/services/<service_name>/clear-output", methods=["POST"])
def clear_output(service_name: str):
    service = service_manager.get_service(service_name)

    if not service:
        dash_warning(f"Clear output requested for unknown service: {service_name}")
        return jsonify(
            {
                "success": False,
                "message": f"Service '{service_name}' not found.",
            }
        ), 404

    service.clear_output()
    return jsonify(
        {
            "success": True,
            "message": f"Output cleared for '{service_name}'.",
        }
    )


@app.route("/api/services/start-all", methods=["POST"])
def start_all_services():
    dash_info("Start All requested")

    results: list[dict[str, Any]] = []
    overall_success = True

    for service_name in SERVICE_START_ORDER:
        dash_info(f"Start All: starting service='{service_name}'")
        result = service_manager.start_service(service_name)

        results.append(
            {
                "service": service_name,
                "result": result,
            }
        )

        if not result.get("success"):
            overall_success = False
            dash_warning(
                f"Start All stopped; service failed service='{service_name}' "
                f"result={result}"
            )
            break

        _sleep_between_services(1.0)

    return jsonify(
        {
            "success": overall_success,
            "message": (
                "All services started successfully."
                if overall_success
                else "Start All stopped because one service failed."
            ),
            "start_order": SERVICE_START_ORDER,
            "results": results,
            "services": _service_snapshot(),
        }
    ), 200 if overall_success else 400


@app.route("/api/services/stop-all", methods=["POST"])
def stop_all_services():
    dash_info("Stop All requested")

    results: list[dict[str, Any]] = []
    overall_success = True

    for service_name in SERVICE_STOP_ORDER:
        dash_info(f"Stop All: stopping service='{service_name}'")
        result = service_manager.stop_service(service_name)

        results.append(
            {
                "service": service_name,
                "result": result,
            }
        )

        if not result.get("success"):
            overall_success = False
            dash_warning(
                f"Stop All service reported issue service='{service_name}' "
                f"result={result}"
            )

        _sleep_between_services(0.5)

    return jsonify(
        {
            "success": overall_success,
            "message": (
                "All services stopped successfully."
                if overall_success
                else "Stop All completed, but one or more services reported an issue."
            ),
            "stop_order": SERVICE_STOP_ORDER,
            "results": results,
            "services": _service_snapshot(),
        }
    ), 200


@app.route("/api/services/restart-all", methods=["POST"])
def restart_all_services():
    dash_info("Restart All requested")

    stop_results: list[dict[str, Any]] = []
    start_results: list[dict[str, Any]] = []

    for service_name in SERVICE_STOP_ORDER:
        dash_info(f"Restart All: stopping service='{service_name}'")
        result = service_manager.stop_service(service_name)

        stop_results.append(
            {
                "service": service_name,
                "result": result,
            }
        )

        _sleep_between_services(0.5)

    _sleep_between_services(2.0)

    overall_success = True

    for service_name in SERVICE_START_ORDER:
        dash_info(f"Restart All: starting service='{service_name}'")
        result = service_manager.start_service(service_name)

        start_results.append(
            {
                "service": service_name,
                "result": result,
            }
        )

        if not result.get("success"):
            overall_success = False
            dash_warning(
                f"Restart All failed during start service='{service_name}' "
                f"result={result}"
            )
            break

        _sleep_between_services(1.0)

    return jsonify(
        {
            "success": overall_success,
            "message": (
                "All services restarted successfully."
                if overall_success
                else "Restart All completed stop phase but failed during start phase."
            ),
            "stop_order": SERVICE_STOP_ORDER,
            "start_order": SERVICE_START_ORDER,
            "stop_results": stop_results,
            "start_results": start_results,
            "services": _service_snapshot(),
        }
    ), 200 if overall_success else 400


# ---------------------------------------------------------
# GPU insights API
# ---------------------------------------------------------

@app.route("/api/gpu-insights", methods=["GET"])
def gpu_insights():
    gpu_service = service_manager.get_service("GPU Service")

    if not gpu_service:
        dash_error("GPU insights requested but GPU Service is not registered")
        return jsonify(
            {
                "success": False,
                "message": "GPU Service is not registered.",
            }
        ), 500

    payload = get_gpu_service_insights(gpu_service=gpu_service)
    return jsonify(payload)


# ---------------------------------------------------------
# PostgreSQL insights API
# ---------------------------------------------------------

@app.route("/api/postgres-insights", methods=["GET"])
def postgres_insights():
    pg_service = service_manager.get_service("PostgreSQL Server")

    if not pg_service:
        dash_error("Postgres insights requested but PostgreSQL Server is not registered")
        return jsonify(
            {
                "success": False,
                "message": "PostgreSQL Server is not registered.",
            }
        ), 500

    payload = get_postgres_service_insights(pg_service=pg_service)
    return jsonify(payload)


# ---------------------------------------------------------
# AI Gateway health API
# ---------------------------------------------------------

@app.route("/api/ai-gateway-health", methods=["GET"])
def ai_gateway_health():
    ai_gateway_service = service_manager.get_service("AI Gateway")

    if not ai_gateway_service:
        dash_error("AI Gateway health requested but AI Gateway is not registered")
        return jsonify(
            {
                "success": False,
                "message": "AI Gateway is not registered.",
            }
        ), 500

    service_data = {
        "status": ai_gateway_service.get_status(),
        "pid": ai_gateway_service.get_pid(),
        "uptime_seconds": ai_gateway_service.get_uptime_seconds(),
        "cwd": ai_gateway_service.cwd,
        "command": (
            " ".join(ai_gateway_service.command)
            if ai_gateway_service.command
            else ""
        ),
        "url": SERVICE_AI_GATEWAY_BASE_URL,
    }

    if ai_gateway_service.get_status() != "running":
        return jsonify(
            {
                "success": True,
                "service": service_data,
                "gateway": {
                    "error": "AI Gateway is not running.",
                },
            }
        )

    gateway_probe = _probe_url(
        f"{SERVICE_AI_GATEWAY_BASE_URL.rstrip('/')}/health",
        timeout=5,
    )

    return jsonify(
        {
            "success": True,
            "service": service_data,
            "gateway": gateway_probe,
        }
    )


# ---------------------------------------------------------
# MCP AI Layer health API
# ---------------------------------------------------------

@app.route("/api/mcp-ai-layer-health", methods=["GET"])
def mcp_ai_layer_health():
    mcp_ai_layer_service = service_manager.get_service("MCP AI Layer")

    if not mcp_ai_layer_service:
        dash_error("MCP AI Layer health requested but service is not registered")
        return jsonify(
            {
                "success": False,
                "message": "MCP AI Layer is not registered.",
            }
        ), 500

    service_data = {
        "status": mcp_ai_layer_service.get_status(),
        "pid": mcp_ai_layer_service.get_pid(),
        "uptime_seconds": mcp_ai_layer_service.get_uptime_seconds(),
        "cwd": mcp_ai_layer_service.cwd,
        "command": (
            " ".join(mcp_ai_layer_service.command)
            if mcp_ai_layer_service.command
            else ""
        ),
        "url": SERVICE_MCP_AI_LAYER_BASE_URL,
        "health_url": SERVICE_MCP_AI_LAYER_HEALTH_URL,
    }

    if mcp_ai_layer_service.get_status() != "running":
        return jsonify(
            {
                "success": True,
                "service": service_data,
                "mcp_ai_layer": {
                    "error": "MCP AI Layer is not running.",
                },
            }
        )

    probe = _probe_url(SERVICE_MCP_AI_LAYER_HEALTH_URL, timeout=5)

    return jsonify(
        {
            "success": True,
            "service": service_data,
            "mcp_ai_layer": probe,
        }
    )


# ---------------------------------------------------------
# Grafana health API
# ---------------------------------------------------------

@app.route("/api/grafana-health", methods=["GET"])
def grafana_health():
    grafana_service = service_manager.get_service("Grafana")

    if not grafana_service:
        dash_error("Grafana health requested but Grafana is not registered")
        return jsonify(
            {
                "success": False,
                "message": "Grafana is not registered.",
            }
        ), 500

    service_data = {
        "status": grafana_service.get_status(),
        "pid": grafana_service.get_pid(),
        "uptime_seconds": grafana_service.get_uptime_seconds(),
        "cwd": grafana_service.cwd,
        "command": (
            " ".join(grafana_service.command)
            if grafana_service.command
            else ""
        ),
        "url": GRAFANA_URL,
    }

    if grafana_service.get_status() != "running":
        return jsonify(
            {
                "success": True,
                "service": service_data,
                "grafana": {
                    "error": "Grafana is not running.",
                },
            }
        )

    grafana_probe = _probe_url(
        f"{GRAFANA_URL.rstrip('/')}/api/health",
        timeout=5,
    )

    return jsonify(
        {
            "success": True,
            "service": service_data,
            "grafana": grafana_probe,
        }
    )


# ---------------------------------------------------------
# System health API
# ---------------------------------------------------------

@app.route("/api/system-health", methods=["GET"])
def system_health():
    services = _service_snapshot()

    service_status = {
        item.get("name"): {
            "status": item.get("status"),
            "pid": item.get("pid"),
            "uptime_seconds": item.get("uptime_seconds"),
            "base_url": item.get("base_url"),
        }
        for item in services
    }

    probes = {
        "gpu": {
            "url": f"{SERVICE_GPU_BASE_URL.rstrip('/')}/api/status",
            **_probe_url(f"{SERVICE_GPU_BASE_URL.rstrip('/')}/api/status"),
        },
        "ai_gateway": {
            "url": f"{SERVICE_AI_GATEWAY_BASE_URL.rstrip('/')}/health",
            **_probe_url(f"{SERVICE_AI_GATEWAY_BASE_URL.rstrip('/')}/health"),
        },
        "mcp_coordinator": {
            "url": f"{SERVICE_MCP_COORDINATOR_BASE_URL.rstrip('/')}/mcp",
            **_probe_url(f"{SERVICE_MCP_COORDINATOR_BASE_URL.rstrip('/')}/mcp"),
            "note": "HTTP 406 on /mcp can still indicate a valid streamable HTTP MCP endpoint.",
        },
        "mcp_ai_layer": {
            "url": SERVICE_MCP_AI_LAYER_HEALTH_URL,
            **_probe_url(SERVICE_MCP_AI_LAYER_HEALTH_URL),
        },
        "grafana": {
            "url": f"{GRAFANA_URL.rstrip('/')}/api/health",
            **_probe_url(f"{GRAFANA_URL.rstrip('/')}/api/health"),
        },
        "postgres": {
            "running": _service_is_running("PostgreSQL Server"),
        },
    }

    all_required_running = all(
        _service_is_running(name)
        for name in SERVICE_START_ORDER
    )

    all_required_reachable = all(
        [
            probes["gpu"]["reachable"],
            probes["ai_gateway"]["reachable"],
            probes["mcp_coordinator"]["reachable"],
            probes["mcp_ai_layer"]["reachable"],
            probes["grafana"]["reachable"],
            _service_is_running("PostgreSQL Server"),
        ]
    )

    return jsonify(
        {
            "success": True,
            "dashboard": {
                "status": "running",
                "host": SERVICE_DASHBOARD_HOST,
                "port": SERVICE_DASHBOARD_PORT,
            },
            "all_required_running": all_required_running,
            "all_required_reachable": all_required_reachable,
            "all_services_healthy": all_required_running and all_required_reachable,
            "service_order": {
                "start": SERVICE_START_ORDER,
                "stop": SERVICE_STOP_ORDER,
            },
            "services": service_status,
            "probes": probes,
        }
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    dash_info(
        f"Starting Service Dashboard | "
        f"host={SERVICE_DASHBOARD_HOST} | "
        f"port={SERVICE_DASHBOARD_PORT}"
    )

    app.run(
        debug=True,
        use_reloader=False,
        host=SERVICE_DASHBOARD_HOST,
        port=SERVICE_DASHBOARD_PORT,
    )