from __future__ import annotations

from typing import Any, Dict, List
from flask import Blueprint, jsonify, request, Response

from modules.configuration.config_env import get_db_config
from modules.observability.models import TraceSession, TraceSpan

db = get_db_config()

TRACE_BP = Blueprint("trace_bp", __name__)


# ---------------------------------------------------------
# HTML UI
# ---------------------------------------------------------

@TRACE_BP.get("/trace")
def trace_index() -> Response:
    return Response(_TRACE_HTML, mimetype="text/html")


# ---------------------------------------------------------
# Recent Traces (from Postgres)
# ---------------------------------------------------------

@TRACE_BP.get("/trace/api/recent")
def trace_recent():

    limit = int(request.args.get("limit", "50"))

    with db.main_session() as session:

        traces: List[TraceSession] = (
            session.query(TraceSession)
            .order_by(TraceSession.started_at.desc())
            .limit(limit)
            .all()
        )

        results = []

        for t in traces:
            span_count = (
                session.query(TraceSpan)
                .filter(TraceSpan.trace_id == t.id)
                .count()
            )

            results.append({
                "trace_id": str(t.id),
                "last_seen": t.started_at.isoformat() if t.started_at else None,
                "span_count": span_count,
            })

    return jsonify({"recent": results})


# ---------------------------------------------------------
# Graph (tree nodes from Postgres)
# ---------------------------------------------------------

@TRACE_BP.get("/trace/api/graph/<trace_id>")
def trace_graph(trace_id: str):

    with db.main_session() as session:

        spans: List[TraceSpan] = (
            session.query(TraceSpan)
            .filter(TraceSpan.trace_id == trace_id)
            .order_by(TraceSpan.started_at.asc())
            .all()
        )

        nodes = []

        for s in spans:
            nodes.append({
                "id": str(s.id),
                "trace_id": str(s.trace_id),
                "parent": str(s.parent_span_id) if s.parent_span_id else None,
                "function": s.name,
                "depth": s.depth,
                "duration_ms": float(s.duration_ms) if s.duration_ms else None,
                "status": s.status,
                "request_id": s.request_id,
                "exception": None,  # you can add this later if stored
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            })

    return jsonify({"nodes": nodes})


# ---------------------------------------------------------
# UI (unchanged)
# ---------------------------------------------------------

_TRACE_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Trace Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .row { display: flex; gap: 16px; }
    .panel { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .left { width: 360px; }
    .right { flex: 1; }
    .trace-item { cursor: pointer; padding: 6px; border-bottom: 1px solid #eee; }
    .trace-item:hover { background: #f7f7f7; }
    .tree ul { list-style: none; padding-left: 18px; margin: 4px 0; }
    .node { margin: 2px 0; }
    .node-header { cursor: pointer; user-select: none; }
    .muted { color: #777; font-size: 12px; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 8px; font-size: 12px; border: 1px solid #ddd; }
    pre { white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 8px; border-radius: 8px; border: 1px solid #eee; }
    .controls { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
    input { padding: 6px; border: 1px solid #ddd; border-radius: 6px; width: 100%; }
    button { padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Trace Dashboard</h2>
  <div class="controls">
    <button id="refreshBtn">Refresh</button>
    <span class="muted">Click a trace to load.</span>
  </div>

  <div class="row">
    <div class="panel left">
      <div style="margin-bottom:8px;">
        <input id="filterInput" placeholder="Filter by trace_id prefix..." />
      </div>
      <div id="traceList"></div>
    </div>

    <div class="panel right">
      <div id="meta" class="muted"></div>
      <div class="tree" id="tree"></div>
      <h4>Selected Node</h4>
      <pre id="nodeDetails">(none)</pre>
    </div>
  </div>

  <script>
    let currentTraceId = null;

    const traceListEl = document.getElementById('traceList');
    const treeEl = document.getElementById('tree');
    const metaEl = document.getElementById('meta');
    const nodeDetailsEl = document.getElementById('nodeDetails');
    const filterInput = document.getElementById('filterInput');

    document.getElementById('refreshBtn').onclick = () => loadRecent();
    filterInput.addEventListener('input', () => loadRecent());

    async function loadRecent() {
      const res = await fetch('/trace/api/recent?limit=50');
      const data = await res.json();
      const filter = (filterInput.value || '').trim();

      traceListEl.innerHTML = '';

      const items = data.recent || [];
      const filtered = filter ? items.filter(x => (x.trace_id || '').startsWith(filter)) : items;

      filtered.forEach(item => {
        const div = document.createElement('div');
        div.className = 'trace-item';
        div.innerHTML = `
          <div><b>${item.trace_id}</b></div>
          <div class="muted">${item.last_seen || ''} | spans=${item.span_count}</div>
        `;
        div.onclick = () => loadTrace(item.trace_id);
        traceListEl.appendChild(div);
      });
    }

    async function loadTrace(traceId) {
      currentTraceId = traceId;
      const res = await fetch('/trace/api/graph/' + traceId);
      const graph = await res.json();

      metaEl.innerText = `trace_id=${traceId} nodes=${(graph.nodes||[]).length}`;

      const tree = buildTree(graph.nodes || []);
      renderTree(tree);
    }

    function buildTree(nodes) {
      const byId = {};
      nodes.forEach(n => byId[n.id] = {...n, children: []});
      const roots = [];
      nodes.forEach(n => {
        if (n.parent && byId[n.parent]) {
          byId[n.parent].children.push(byId[n.id]);
        } else {
          roots.push(byId[n.id]);
        }
      });
      return roots;
    }

    function renderTree(roots) {
      treeEl.innerHTML = '';
      const ul = document.createElement('ul');
      roots.forEach(r => ul.appendChild(renderNode(r)));
      treeEl.appendChild(ul);
    }

    function renderNode(node) {
      const li = document.createElement('li');
      const header = document.createElement('div');
      header.className = 'node-header';
      header.innerHTML = `
        <span class="badge">${node.duration_ms || ''}ms</span>
        <b>${node.function}</b>
      `;
      header.onclick = () => {
        nodeDetailsEl.innerText = JSON.stringify(node, null, 2);
      };
      li.appendChild(header);

      const childUl = document.createElement('ul');
      (node.children || []).forEach(c => childUl.appendChild(renderNode(c)));
      li.appendChild(childUl);
      return li;
    }

    loadRecent();
  </script>
</body>
</html>
"""