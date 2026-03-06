# app/api/dashboard.py

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import gpu_info, gpu_warning, gpu_error
import logging

EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
router = APIRouter()

# ---------------------------------------------------------
# GPU backends
# ---------------------------------------------------------
try:
    import pynvml  # type: ignore

    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_gpu_info_nvml() -> Dict[str, Any]:
    """
    Best option if pynvml is installed.
    """
    assert pynvml is not None

    pynvml.nvmlInit()
    try:
        gpus: List[Dict[str, Any]] = []
        count = pynvml.nvmlDeviceGetCount()

        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw_name = pynvml.nvmlDeviceGetName(h)
            name = raw_name.decode("utf-8", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)

            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            power_w = None
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power_w = None

            # process list (best effort)
            procs: List[Dict[str, Any]] = []
            try:
                running = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                for p in running:
                    procs.append(
                        {
                            "pid": int(p.pid),
                            "used_memory_gb": round(_bytes_to_gb(int(p.usedGpuMemory)), 3),
                        }
                    )
            except Exception:
                pass

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "utilization_gpu_pct": int(util.gpu),
                    "utilization_mem_pct": int(util.memory),
                    "mem_total_gb": round(_bytes_to_gb(int(mem.total)), 3),
                    "mem_used_gb": round(_bytes_to_gb(int(mem.used)), 3),
                    "mem_free_gb": round(_bytes_to_gb(int(mem.free)), 3),
                    "temperature_c": temp,
                    "power_w": round(power_w, 2) if power_w is not None else None,
                    "processes": procs,
                }
            )

        return {
            "backend": "nvml",
            "timestamp": time.time(),
            "gpus": gpus,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _run_nvidia_smi(query: str) -> List[str]:
    """
    Run nvidia-smi with a CSV query and return list of lines (no header).
    """
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _get_gpu_info_smi() -> Dict[str, Any]:
    """
    Fallback backend if pynvml isn't installed.
    """
    # GPU summary
    # note: temperature and power might not exist on some systems; keep simple
    lines = _run_nvidia_smi("index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw")
    gpus: List[Dict[str, Any]] = []

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        # Defensive parse
        idx = _safe_int(parts[0]) if len(parts) > 0 else 0
        name = parts[1] if len(parts) > 1 else "unknown"
        util_gpu = _safe_int(parts[2]) if len(parts) > 2 else 0
        util_mem = _safe_int(parts[3]) if len(parts) > 3 else 0
        mem_total = _safe_float(parts[4]) if len(parts) > 4 else 0.0
        mem_used = _safe_float(parts[5]) if len(parts) > 5 else 0.0
        mem_free = _safe_float(parts[6]) if len(parts) > 6 else 0.0
        temp_c = _safe_int(parts[7]) if len(parts) > 7 else None
        power_w = _safe_float(parts[8]) if len(parts) > 8 else None

        gpus.append(
            {
                "index": idx,
                "name": name,
                "utilization_gpu_pct": util_gpu,
                "utilization_mem_pct": util_mem,
                "mem_total_gb": round(mem_total / 1024.0 if mem_total > 1000 else mem_total, 3),  # smi returns MiB typically; this keeps it readable
                "mem_used_gb": round(mem_used / 1024.0 if mem_used > 1000 else mem_used, 3),
                "mem_free_gb": round(mem_free / 1024.0 if mem_free > 1000 else mem_free, 3),
                "temperature_c": temp_c,
                "power_w": round(power_w, 2) if power_w is not None else None,
                "processes": [],
            }
        )

    return {
        "backend": "nvidia-smi",
        "timestamp": time.time(),
        "gpus": gpus,
    }


def get_gpu_info() -> Dict[str, Any]:
    if _HAS_NVML:
        try:
            return _get_gpu_info_nvml()
        except Exception as e:
            gpu_warning(f"[DASH] NVML failed, falling back to nvidia-smi: {e}")
    try:
        return _get_gpu_info_smi()
    except Exception as e:
        return {
            "backend": "none",
            "timestamp": time.time(),
            "error": f"GPU metrics unavailable: {e}",
            "gpus": [],
        }


# ---------------------------------------------------------
# Log streaming backend
# ---------------------------------------------------------
@dataclass
class LogMessage:
    ts: float
    level: str
    msg: str


class LogHub:
    """
    Central hub for dashboard log subscribers.
    You can publish logs here from your logger, or from file tailer below.
    """
    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def publish(self, level: str, msg: str):
        item = LogMessage(ts=time.time(), level=level, msg=msg)
        async with self._lock:
            subs = list(self._subscribers)

        for q in subs:
            try:
                if q.full():
                    _ = q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass


LOG_HUB = LogHub()


async def _tail_log_file_task():
    """
    Optional background file tailer.
    If GPU_DASH_LOG_FILE is set, this will tail the file and publish to LOG_HUB.
    """
    path = os.getenv("GPU_DASH_LOG_FILE", "").strip()
    if not path:
        gpu_info("[DASH] GPU_DASH_LOG_FILE not set; file tailer disabled")
        return

    gpu_info(f"[DASH] Log tailer enabled | file={path}")

    # Wait for file to exist
    while True:
        if os.path.isfile(path):
            break
        await asyncio.sleep(1.0)

    # Tail
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Seek to end on startup
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.25)
                    continue
                line = line.rstrip("\n")
                if line.strip():
                    await LOG_HUB.publish("INFO", line)
    except Exception as e:
        gpu_error(f"[DASH] Log tailer crashed: {e}")


# ---------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------
@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    """
    Single-page dashboard UI.
    """
    return HTMLResponse(_DASHBOARD_HTML)


@router.get("/api/status")
def api_status():
    """
    Service status: loaded models, meta, GPU count.
    """
    try:
        st = GPU_MODELS.status()
        return JSONResponse(st)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/gpu")
def api_gpu():
    """
    GPU metrics snapshot.
    """
    return JSONResponse(get_gpu_info())


@router.get("/api/process")
def api_process():
    """
    Host process info (CPU/RAM) for the GPU service.
    """
    p = psutil.Process(os.getpid())
    mem = p.memory_info()
    return JSONResponse(
        {
            "pid": p.pid,
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "ram_used_gb": round(mem.rss / (1024 ** 3), 3),
            "ram_vms_gb": round(mem.vms / (1024 ** 3), 3),
            "threads": p.num_threads(),
            "create_time": p.create_time(),
            "timestamp": time.time(),
        }
    )


@router.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    """
    Live log stream.
    Dashboard connects here and receives JSON messages.
    """
    await ws.accept()
    q = await LOG_HUB.subscribe()

    try:
        # Send a hello
        await ws.send_text(json.dumps({"type": "hello", "ts": time.time()}))
        while True:
            item: LogMessage = await q.get()
            await ws.send_text(
                json.dumps(
                    {
                        "type": "log",
                        "ts": item.ts,
                        "level": item.level,
                        "msg": item.msg,
                    }
                )
            )
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await LOG_HUB.unsubscribe(q)


# ---------------------------------------------------------
# Hook: start background tasks
# ---------------------------------------------------------
# ---------------------------------------------------------
# Hook: start background tasks
# ---------------------------------------------------------
def install_dashboard_background_tasks(app, on_loop_ready=None):
    """
    Call this once during app startup:
        install_dashboard_background_tasks(app)

    It starts the file tailer if configured
    and captures the running event loop.
    """

    @app.on_event("startup")
    async def _dash_startup():
        global EVENT_LOOP

        EVENT_LOOP = asyncio.get_running_loop()

        if on_loop_ready:
            on_loop_ready(EVENT_LOOP)

        asyncio.create_task(_tail_log_file_task())

        gpu_info("[DASH] Dashboard background tasks started")


# ---------------------------------------------------------
# Minimal HTML/JS UI (no external assets)
# ---------------------------------------------------------
_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>GPU Service Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; min-width: 320px; flex: 1; }
    .title { font-weight: bold; margin-bottom: 8px; }
    pre { background: #0b0b0b; color: #e6e6e6; padding: 10px; border-radius: 10px; overflow: auto; max-height: 420px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; padding: 6px; text-align: left; font-size: 14px; }
    .small { font-size: 12px; color: #444; }
    .ok { color: #167a16; }
    .warn { color: #a16207; }
    .err { color: #b91c1c; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; font-size: 12px; }
    .toolbar { display:flex; gap: 8px; align-items:center; margin-bottom: 12px; flex-wrap: wrap; }
    button { padding: 6px 10px; border-radius: 8px; border: 1px solid #ddd; background: white; cursor: pointer; }
    button:hover { background: #f9fafb; }
    input { padding: 6px 8px; border-radius: 8px; border: 1px solid #ddd; }
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="title">GPU Service Dashboard</div>
    <span class="pill" id="backendPill">backend: -</span>
    <span class="pill" id="wsPill">logs: disconnected</span>
    <button onclick="refreshAll()">Refresh</button>
    <button onclick="clearLogs()">Clear Logs</button>
    <label class="small">Auto refresh (sec):</label>
    <input id="refreshSec" type="number" value="2" min="1" max="60" style="width:80px"/>
  </div>

  <div class="row">
    <div class="card">
      <div class="title">GPU Usage</div>
      <div id="gpuBlock" class="small">Loading...</div>
    </div>

    <div class="card">
      <div class="title">Service Status</div>
      <div id="statusBlock" class="small">Loading...</div>
    </div>

    <div class="card">
      <div class="title">Process</div>
      <div id="procBlock" class="small">Loading...</div>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="title">Live Logs</div>
    <pre id="logPre"></pre>
  </div>

<script>
let ws = null;
let autoTimer = null;

function fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function setPill(id, text) {
  document.getElementById(id).textContent = text;
}

function appendLog(line) {
  const pre = document.getElementById("logPre");
  pre.textContent += line + "\n";
  pre.scrollTop = pre.scrollHeight;
}

function clearLogs() {
  document.getElementById("logPre").textContent = "";
}

async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return await res.json();
}

function renderGpu(data) {
  setPill("backendPill", "backend: " + (data.backend || "-"));

  if (data.error) {
    document.getElementById("gpuBlock").innerHTML = "<span class='err'>" + data.error + "</span>";
    return;
  }
  const gpus = data.gpus || [];
  if (!gpus.length) {
    document.getElementById("gpuBlock").textContent = "No GPUs detected.";
    return;
  }

  let html = "<table><thead><tr><th>GPU</th><th>Util</th><th>VRAM</th><th>Temp</th><th>Power</th></tr></thead><tbody>";
  for (const g of gpus) {
    const util = (g.utilization_gpu_pct ?? 0) + "%";
    const vram = (g.mem_used_gb ?? 0) + " / " + (g.mem_total_gb ?? 0) + " GB";
    const temp = (g.temperature_c == null ? "-" : (g.temperature_c + " C"));
    const power = (g.power_w == null ? "-" : (g.power_w + " W"));
    html += "<tr>";
    html += "<td>" + g.index + " - " + g.name + "</td>";
    html += "<td>" + util + "</td>";
    html += "<td>" + vram + "</td>";
    html += "<td>" + temp + "</td>";
    html += "<td>" + power + "</td>";
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("gpuBlock").innerHTML = html;
}

function renderStatus(data) {
  const models = data.models || {};
  const keys = Object.keys(models);
  let html = "";
  html += "<div class='small'>GPU count: <b>" + (data.gpus ?? "-") + "</b></div>";
  html += "<div class='small'>Loaded models: <b>" + keys.length + "</b></div>";

  if (!keys.length) {
    html += "<div class='small'>No models loaded.</div>";
    document.getElementById("statusBlock").innerHTML = html;
    return;
  }

  html += "<table><thead><tr><th>Name</th><th>Kind</th><th>Device</th><th>Sharded</th></tr></thead><tbody>";
  keys.sort();
  for (const k of keys) {
    const m = models[k] || {};
    html += "<tr>";
    html += "<td>" + k + "</td>";
    html += "<td>" + (m.kind ?? "-") + "</td>";
    html += "<td>" + (m.device ?? "-") + "</td>";
    html += "<td>" + (m.sharded ? "yes" : "no") + "</td>";
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("statusBlock").innerHTML = html;
}

function renderProc(data) {
  let html = "";
  html += "<div class='small'>PID: <b>" + (data.pid ?? "-") + "</b></div>";
  html += "<div class='small'>CPU: <b>" + (data.cpu_percent ?? 0) + "%</b></div>";
  html += "<div class='small'>RAM (RSS): <b>" + (data.ram_used_gb ?? 0) + " GB</b></div>";
  html += "<div class='small'>Threads: <b>" + (data.threads ?? "-") + "</b></div>";
  document.getElementById("procBlock").innerHTML = html;
}

async function refreshAll() {
  try {
    const [gpu, status, proc] = await Promise.all([
      fetchJson("/api/gpu"),
      fetchJson("/api/status"),
      fetchJson("/api/process"),
    ]);
    renderGpu(gpu);
    renderStatus(status);
    renderProc(proc);
  } catch (e) {
    appendLog(fmtTs(Date.now()/1000) + " ERROR refresh: " + e.message);
  }
}

function connectLogs() {
  const proto = (location.protocol === "https:") ? "wss" : "ws";
  const url = proto + "://" + location.host + "/ws/logs";
  ws = new WebSocket(url);

  ws.onopen = () => setPill("wsPill", "logs: connected");
  ws.onclose = () => setPill("wsPill", "logs: disconnected");
  ws.onerror = () => setPill("wsPill", "logs: error");

  ws.onmessage = (ev) => {
    try {
      const obj = JSON.parse(ev.data);
      if (obj.type === "log") {
        appendLog(fmtTs(obj.ts) + " " + (obj.level || "INFO") + " " + (obj.msg || ""));
      }
    } catch {
      appendLog(ev.data);
    }
  };
}

function startAutoRefresh() {
  if (autoTimer) clearInterval(autoTimer);
  const sec = Math.max(1, parseInt(document.getElementById("refreshSec").value || "2"));
  autoTimer = setInterval(refreshAll, sec * 1000);
}

document.getElementById("refreshSec").addEventListener("change", startAutoRefresh);

refreshAll();
connectLogs();
startAutoRefresh();
</script>
</body>
</html>
"""