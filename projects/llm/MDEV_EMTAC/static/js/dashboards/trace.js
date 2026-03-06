let currentTraceId = null;
const BASE = "/dashboards/trace";

document.addEventListener("DOMContentLoaded", function () {

    const refreshBtn = document.getElementById("refreshBtn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", refreshTraces);
    }

    refreshTraces(); // initial load
});

async function refreshTraces() {
    const res = await fetch(`${BASE}/api/recent?limit=50`);

    if (!res.ok) {
        console.error("Failed to load recent traces:", res.status);
        return;
    }
    console.log("refreshTraces called");
    const data = await res.json();

    const list = document.getElementById("traceList");
    list.innerHTML = "";

    (data.recent || []).forEach(trace => {
        const div = document.createElement("div");
        div.className = "trace-item";
        div.innerHTML = `
    <div><strong>${trace.root_function || "Unknown Root"}</strong></div>
    <div style="font-size:12px;color:#777;">
        ${trace.started_at ? new Date(trace.started_at).toLocaleString() : ""}
        | ${trace.duration_ms || 0}ms
        | spans=${trace.span_count}
        | ${trace.status}
    </div>
    <div style="font-size:11px;color:#999;">
        request_id=${trace.request_id}
    </div>
`;
        div.onclick = () => loadTrace(trace.trace_id);
        list.appendChild(div);
    });
}

async function loadTrace(traceId) {
    currentTraceId = traceId;

    const res = await fetch(`${BASE}/api/graph/${traceId}`);

    if (!res.ok) {
        console.error("Failed to load trace graph:", res.status);
        return;
    }

    const data = await res.json();

    document.getElementById("traceMeta").innerText =
        `Trace ID: ${traceId} | Nodes: ${(data.nodes || []).length}`;

    renderTree(data.nodes || []);
}

function renderTree(nodes) {
    const container = document.getElementById("traceTree");
    container.innerHTML = "";

    nodes.forEach(node => {
        const div = document.createElement("div");
        div.className = "trace-node";
        div.innerText = `${node.function} (${node.duration_ms || 0}ms)`;

        div.onclick = () => {
            document.getElementById("nodeDetails").innerText =
                JSON.stringify(node, null, 2);
        };

        container.appendChild(div);
    });
}