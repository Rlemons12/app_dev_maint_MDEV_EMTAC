from __future__ import annotations

import asyncio
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_layer.gateway import ask_ai
from ai_layer.mcp_client import McpToolClient
from ai_layer.settings import get_ai_settings
from mcp_server.settings import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "app" / "web"

DOC_FILES = {
    "postgres-mcp": PROJECT_ROOT / "docs" / "README_POSTGRES_MCP.md",
    "connect-and-use": PROJECT_ROOT / "docs" / "HOW_TO_CONNECT_AND_USE_POSTGRES_MCP.md",
    "ai-gateway": PROJECT_ROOT / "docs" / "README_AI_GATEWAY.md",
}

MCP_PROCESS_LOCK = threading.Lock()
MCP_PROCESS: subprocess.Popen[str] | None = None
MCP_PROCESS_CONFIG: dict[str, Any] = {}


def run_async(value: Any) -> Any:
    return asyncio.run(value)


def json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    payload = json.dumps(data, default=str).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))

    if content_length <= 0:
        return {}

    body = handler.rfile.read(content_length).decode("utf-8")

    if not body.strip():
        return {}

    parsed = json.loads(body)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object request body.")

    return parsed


def safe_static_path(request_path: str) -> Path:
    parsed_path = unquote(urlparse(request_path).path)

    if parsed_path == "/":
        parsed_path = "/index.html"

    relative_path = parsed_path.lstrip("/")
    file_path = (STATIC_ROOT / relative_path).resolve()

    if not file_path.is_relative_to(STATIC_ROOT.resolve()):
        raise ValueError("Invalid static path.")

    return file_path


def text_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    body: str,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    payload = body.encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def docs_index() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for doc_id, path in DOC_FILES.items():
        docs.append(
            {
                "id": doc_id,
                "filename": path.name,
                "url": f"/api/docs/{doc_id}",
                "exists": path.exists(),
            }
        )

    return docs


def read_doc(doc_id: str) -> str:
    path = DOC_FILES.get(doc_id)

    if path is None:
        raise ValueError(f"Unknown documentation id: {doc_id}")

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Documentation file is missing: {path.name}")

    return path.read_text(encoding="utf-8")


def read_docs_bundle() -> str:
    sections: list[str] = []

    for doc_id in DOC_FILES:
        sections.append(f"\n\n<!-- doc_id: {doc_id} -->\n\n")
        sections.append(read_doc(doc_id))

    return "".join(sections).strip() + "\n"


def _mcp_status() -> dict[str, Any]:
    with MCP_PROCESS_LOCK:
        process = MCP_PROCESS
        config = dict(MCP_PROCESS_CONFIG)

    running = process is not None and process.poll() is None
    pid = process.pid if running and process is not None else None

    return {
        "running": running,
        "pid": pid,
        "config": config,
    }


def _mcp_start(
    python_executable: str,
    server_script: str,
    transport: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    with MCP_PROCESS_LOCK:
        global MCP_PROCESS
        global MCP_PROCESS_CONFIG

        if MCP_PROCESS is not None and MCP_PROCESS.poll() is None:
            return {
                "status": "already_running",
                "pid": MCP_PROCESS.pid,
                "config": dict(MCP_PROCESS_CONFIG),
            }

        env = os.environ.copy()
        env["MCP_TRANSPORT"] = transport
        env["MCP_HTTP_HOST"] = host
        env["MCP_HTTP_PORT"] = str(port)

        process = subprocess.Popen(
            [python_executable, server_script],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        MCP_PROCESS = process
        MCP_PROCESS_CONFIG = {
            "python_executable": python_executable,
            "server_script": server_script,
            "transport": transport,
            "host": host,
            "port": port,
        }

        return {
            "status": "started",
            "pid": process.pid,
            "config": dict(MCP_PROCESS_CONFIG),
        }


def _mcp_stop() -> dict[str, Any]:
    with MCP_PROCESS_LOCK:
        global MCP_PROCESS
        global MCP_PROCESS_CONFIG

        process = MCP_PROCESS

        if process is None or process.poll() is not None:
            MCP_PROCESS = None
            MCP_PROCESS_CONFIG = {}
            return {
                "status": "already_stopped",
            }

        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

        stopped_pid = process.pid
        MCP_PROCESS = None
        MCP_PROCESS_CONFIG = {}

    return {
        "status": "stopped",
        "pid": stopped_pid,
    }


class PostgresMcpUiHandler(BaseHTTPRequestHandler):
    server_version = "PostgresMcpUI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self.handle_api_get()
            return

        self.handle_static()

    def do_POST(self) -> None:
        if not self.path.startswith("/api/"):
            json_response(self, 404, {"error": "Not found"})
            return

        self.handle_api_post()

    def handle_api_get(self) -> None:
        try:
            parsed_path = unquote(urlparse(self.path).path)

            if self.path == "/api/settings":
                settings = get_settings()
                ai_settings = get_ai_settings()

                json_response(
                    self,
                    200,
                    {
                        "postgres_host": settings.postgres_host,
                        "postgres_port": settings.postgres_port,
                        "postgres_default_db": settings.postgres_default_db,
                        "postgres_default_schema": settings.postgres_default_schema,
                        "mcp_server_name": settings.mcp_server_name,
                        "max_read_rows": settings.max_read_rows,
                        "ai_provider": ai_settings.ai_provider,
                        "openai_model": ai_settings.openai_model,
                        "openai_api_key_set": bool(ai_settings.openai_api_key),
                        "hf_router_base_url": ai_settings.hf_router_base_url,
                        "hf_router_model": ai_settings.hf_router_model,
                        "hf_token_set": bool(ai_settings.hf_token),
                        "ai_default_database": ai_settings.ai_default_database,
                        "mcp_server_python": ai_settings.resolved_mcp_python(),
                        "mcp_server_script": ai_settings.resolved_mcp_script(),
                        "grafana_url": os.getenv("GRAFANA_URL", "http://localhost:3000"),
                        "require_approval_for_destructive_tools": (
                            ai_settings.require_approval_for_destructive_tools
                        ),
                    },
                )
                return

            if self.path == "/api/mcp-server/status":
                json_response(self, 200, _mcp_status())
                return

            if self.path == "/api/tools":
                tools = run_async(McpToolClient().list_tools())
                json_response(self, 200, {"tools": tools})
                return

            if parsed_path == "/api/docs":
                json_response(
                    self,
                    200,
                    {
                        "docs": docs_index(),
                        "bundle_url": "/api/docs/bundle",
                    },
                )
                return

            if parsed_path == "/api/docs/bundle":
                text_response(
                    self,
                    200,
                    read_docs_bundle(),
                    "text/markdown; charset=utf-8",
                )
                return

            if parsed_path.startswith("/api/docs/"):
                doc_id = parsed_path.removeprefix("/api/docs/").strip("/")

                if not doc_id:
                    json_response(self, 400, {"error": "Documentation id is required."})
                    return

                text_response(
                    self,
                    200,
                    read_doc(doc_id),
                    "text/markdown; charset=utf-8",
                )
                return

            json_response(self, 404, {"error": "Unknown API endpoint."})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def handle_api_post(self) -> None:
        try:
            payload = read_json_body(self)

            if self.path == "/api/ask":
                question = str(payload.get("question") or "").strip()
                allow_destructive = bool(payload.get("allow_destructive"))
                provider = payload.get("provider") or None
                model = payload.get("model") or None
                use_tools = bool(payload.get("use_tools", True))

                if not question:
                    json_response(self, 400, {"error": "question is required."})
                    return

                answer = run_async(
                    ask_ai(
                        question,
                        approve_destructive_tools=allow_destructive,
                        print_tool_calls=False,
                        provider=str(provider).strip() if provider else None,
                        model=str(model).strip() if model else None,
                        use_tools=use_tools,
                    )
                )
                json_response(self, 200, {"answer": answer})
                return

            if self.path == "/api/mcp-server/start":
                ai_settings = get_ai_settings()
                python_executable = str(payload.get("python_executable") or "").strip()
                server_script = str(payload.get("server_script") or "").strip()
                transport = str(payload.get("transport") or "streamable-http").strip().lower()
                host = str(payload.get("host") or "127.0.0.1").strip()
                port = int(payload.get("port") or 8000)

                if transport == "http":
                    transport = "streamable-http"

                if transport not in {"streamable-http", "streamable_http"}:
                    json_response(
                        self,
                        400,
                        {"error": "transport must be streamable-http for managed mode."},
                    )
                    return

                result = _mcp_start(
                    python_executable=python_executable or ai_settings.resolved_mcp_python(),
                    server_script=server_script or ai_settings.resolved_mcp_script(),
                    transport=transport,
                    host=host,
                    port=port,
                )
                json_response(self, 200, result)
                return

            if self.path == "/api/mcp-server/stop":
                json_response(self, 200, _mcp_stop())
                return

            if self.path == "/api/read-query":
                sql = str(payload.get("sql") or "").strip()
                database_name = payload.get("database_name") or None

                if not sql:
                    json_response(self, 400, {"error": "sql is required."})
                    return

                arguments: dict[str, Any] = {"sql": sql}

                if database_name:
                    arguments["database_name"] = database_name

                result = run_async(
                    McpToolClient().call_tool("postgres_read_query", arguments)
                )
                json_response(self, 200, {"result": result})
                return

            if self.path == "/api/tool":
                name = str(payload.get("name") or "").strip()
                arguments = payload.get("arguments") or {}

                if not name:
                    json_response(self, 400, {"error": "name is required."})
                    return

                if not isinstance(arguments, dict):
                    json_response(self, 400, {"error": "arguments must be an object."})
                    return

                result = run_async(McpToolClient().call_tool(name, arguments))
                json_response(self, 200, {"result": result})
                return

            json_response(self, 404, {"error": "Unknown API endpoint."})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def handle_static(self) -> None:
        try:
            file_path = safe_static_path(self.path)

            if not file_path.exists() or not file_path.is_file():
                json_response(self, 404, {"error": "Not found"})
                return

            content = file_path.read_bytes()
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PostgreSQL MCP browser UI.")
    parser.add_argument(
        "--host",
        default=os.getenv("POSTGRES_MCP_UI_HOST", "127.0.0.1"),
        help="Host interface to bind. Defaults to POSTGRES_MCP_UI_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("POSTGRES_MCP_UI_PORT", "8765")),
        help="Port to bind. Defaults to POSTGRES_MCP_UI_PORT or 8765.",
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    server = ThreadingHTTPServer((host, port), PostgresMcpUiHandler)

    print(f"PostgreSQL MCP UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
