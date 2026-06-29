// static/js/dashboards/trace.js
// ============================================================
// EMTAC Trace Dashboard
// Complete drop-in replacement
//
// This file replaces BOTH old script blocks:
//   1) the large inline <script> dashboard logic
//   2) the older simple refreshTraces/loadTrace/renderTree script
//
// Important:
// - Do NOT put Jinja {{ ... }} directly in this static JS file.
// - This file uses window.TRACE_DASHBOARD_CONFIG or window.TRACE_* globals
//   when the template provides them.
// - If no config is provided, it falls back to the default trace routes.
// ============================================================

(() => {
  'use strict';

  // ------------------------------------------------------------
  // Config
  // ------------------------------------------------------------

  const CONFIG = window.TRACE_DASHBOARD_CONFIG || {};

  const TRACE_RECENT_API_URL =
    CONFIG.traceRecentApiUrl ||
    window.TRACE_RECENT_API_URL ||
    '/dashboards/trace/api/recent';

  const TRACE_GRAPH_API_BASE =
    CONFIG.traceGraphApiBase ||
    window.TRACE_GRAPH_API_BASE ||
    '/dashboards/trace/api/graph/__TRACE_ID__';

  const TRACE_SETTINGS_API_URL =
    CONFIG.traceSettingsApiUrl ||
    window.TRACE_SETTINGS_API_URL ||
    '/dashboards/trace/api/settings';

  const TRACE_SETTINGS_UPDATE_API_URL =
    CONFIG.traceSettingsUpdateApiUrl ||
    window.TRACE_SETTINGS_UPDATE_API_URL ||
    '/dashboards/trace/api/settings/update';

  const TRACE_SETTINGS_CLEAR_API_URL =
    CONFIG.traceSettingsClearApiUrl ||
    window.TRACE_SETTINGS_CLEAR_API_URL ||
    '/dashboards/trace/api/settings/clear';

  const TRACE_ENV_TEMPLATE_API_URL =
    CONFIG.traceEnvTemplateApiUrl ||
    window.TRACE_ENV_TEMPLATE_API_URL ||
    '/dashboards/trace/api/settings/env-template';

  // ------------------------------------------------------------
  // State
  // ------------------------------------------------------------

  let RECENT_TRACES = [];
  let FILTERED_RECENT_TRACES = [];
  let CURRENT_TRACE_ID = null;
  let CURRENT_NODES = [];
  let CURRENT_SUMMARY = null;
  let VISIBLE_NODES = [];
  let TOTAL = 1;
  let selectedId = null;
  let currentView = 'waterfall';
  let collapsed = {};
  let autoRefreshTimer = null;
  let TRACE_SETTINGS = {};
  let initialized = false;

  // ------------------------------------------------------------
  // DOM cache
  // ------------------------------------------------------------

  const els = {};

  function cacheDom() {
    els.traceList = document.getElementById('traceList');
    els.sidebarFooter = document.getElementById('sidebarFooter');
    els.selectedTraceId = document.getElementById('selectedTraceId');
    els.traceRange = document.getElementById('traceRange');

    els.statSpans = document.getElementById('statSpans');
    els.statVisible = document.getElementById('statVisible');
    els.statTotal = document.getElementById('statTotal');
    els.statOk = document.getElementById('statOk');
    els.statError = document.getElementById('statError');
    els.statRoots = document.getElementById('statRoots');

    els.view = document.getElementById('view');
    els.axis = document.getElementById('axis');
    els.detail = document.getElementById('detail');

    els.searchInput = document.getElementById('searchInput');
    els.traceStatusFilter = document.getElementById('traceStatusFilter');
    els.limitInput = document.getElementById('limitInput');
    els.refreshBtn = document.getElementById('refreshBtn');
    els.autoRefreshSelect = document.getElementById('autoRefreshSelect');
    els.keepSelectionCheck = document.getElementById('keepSelectionCheck');

    els.reloadTraceBtn = document.getElementById('reloadTraceBtn');
    els.copyTraceIdBtn = document.getElementById('copyTraceIdBtn');
    els.waterfallBtn = document.getElementById('waterfallBtn');
    els.ganttBtn = document.getElementById('ganttBtn');

    els.spanStatusFilter = document.getElementById('spanStatusFilter');
    els.spanSearchInput = document.getElementById('spanSearchInput');
    els.showOnlyRootsCheck = document.getElementById('showOnlyRootsCheck');
    els.expandAllBtn = document.getElementById('expandAllBtn');
    els.collapseAllBtn = document.getElementById('collapseAllBtn');

    els.traceSettingsContainer = document.getElementById('traceSettingsContainer');
    els.traceSettingsMessage = document.getElementById('traceSettingsMessage');
    els.traceSettingsRefreshBtn = document.getElementById('traceSettingsRefreshBtn');
    els.traceSettingsApplyBtn = document.getElementById('traceSettingsApplyBtn');
    els.traceSettingsClearBtn = document.getElementById('traceSettingsClearBtn');
    els.traceEnvTemplateBtn = document.getElementById('traceEnvTemplateBtn');
    els.traceEnvTemplateBox = document.getElementById('traceEnvTemplateBox');

    // Compatibility with older simple HTML versions.
    els.legacyTraceMeta = document.getElementById('traceMeta');
    els.legacyTraceTree = document.getElementById('traceTree');
    els.legacyNodeDetails = document.getElementById('nodeDetails');
  }

  // ------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------

  function hasRequiredDom() {
    return !!(els.traceList && els.view && els.detail);
  }

  function formatMs(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';

    const n = Number(value);

    if (n < 1) return `${n.toFixed(2)}ms`;
    if (n < 10) return `${n.toFixed(1)}ms`;
    if (n < 1000) return `${Math.round(n)}ms`;

    return `${(n / 1000).toFixed(2)}s`;
  }

  function formatDate(value) {
    if (!value) return '—';

    const d = new Date(value);

    if (Number.isNaN(d.getTime())) return value;

    return d.toLocaleString();
  }

  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value == null ? '' : String(value);
    return div.innerHTML;
  }

  function setText(el, value) {
    if (el) el.textContent = value;
  }

  function setHtml(el, value) {
    if (el) el.innerHTML = value;
  }

  function setStatus(message) {
    setText(els.sidebarFooter, message || '');
  }

  function setTraceSettingsMessage(message) {
    setText(els.traceSettingsMessage, message || '');
  }

  function getLimitValue() {
    const raw = els.limitInput ? Number(els.limitInput.value || 50) : 50;
    return Math.max(1, Math.min(raw, 200));
  }

  function normalizeStatus(value) {
    const status = String(value || '').toLowerCase();

    if (['error', 'failed', 'exception', 'failure'].includes(status)) return 'error';
    if (['ok', 'success', 'completed', 'complete', 'passed'].includes(status)) return 'ok';

    return 'unknown';
  }

  async function copyText(value) {
    const text = String(value || '');

    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return;
      }
    } catch (error) {
      // Fallback below.
    }

    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.setAttribute('readonly', 'readonly');
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';

    document.body.appendChild(textArea);
    textArea.select();

    try {
      document.execCommand('copy');
    } finally {
      document.body.removeChild(textArea);
    }
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
      credentials: 'same-origin',
      ...options,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        ...(options.headers || {})
      }
    });

    let payload = null;

    try {
      payload = await response.json();
    } catch (error) {
      payload = {
        status: 'error',
        message: 'Response was not JSON.',
        http_status: response.status
      };
    }

    if (!response.ok) {
      const message = payload.message || payload.error || `HTTP ${response.status}`;
      throw new Error(message);
    }

    return payload;
  }

  function buildRecentApiUrl(limitValue) {
    const recentUrl = new URL(TRACE_RECENT_API_URL, window.location.origin);
    recentUrl.searchParams.set('limit', String(limitValue));
    return recentUrl.toString();
  }

  function buildGraphApiUrl(traceId) {
    return TRACE_GRAPH_API_BASE.replace('__TRACE_ID__', encodeURIComponent(traceId));
  }

  // ------------------------------------------------------------
  // Trace settings
  // ------------------------------------------------------------

  async function loadTraceSettings() {
    if (!els.traceSettingsContainer) return;

    try {
      setTraceSettingsMessage('Loading settings...');

      const payload = await fetchJson(TRACE_SETTINGS_API_URL);

      TRACE_SETTINGS = payload.settings || {};
      renderTraceSettings(TRACE_SETTINGS);

      const snapshot = payload.snapshot || {};
      setTraceSettingsMessage(
        `Settings loaded. Runtime overrides apply immediately. Generated: ${snapshot.generated_at || '—'}`
      );
    } catch (error) {
      setHtml(els.traceSettingsContainer, `<div class="error-box">${escapeHtml(error.message)}</div>`);
      setTraceSettingsMessage(`Failed to load settings: ${error.message}`);
    }
  }

  function renderTraceSettings(settings) {
    if (!els.traceSettingsContainer) return;

    const order = [
      'EMTAC_TRACE_ENABLED',
      'EMTAC_TRACE_CHAT_ENABLED',
      'EMTAC_TRACE_PAYLOAD_ENABLED',
      'EMTAC_TRACE_FEEDBACK_ENABLED',
      'EMTAC_TRACE_HEALTH_ENABLED',
      'EMTAC_TRACE_DEEP_PROFILE',
      'EMTAC_TRACE_CAPTURE_ARGS',
      'EMTAC_TRACE_CAPTURE_RETURN',
      'ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN'
    ];

    const rows = [];

    for (const key of order) {
      const setting = settings[key];

      if (!setting) continue;

      rows.push(`
        <div class="setting-row">
          <div class="setting-info">
            <div class="setting-label">${escapeHtml(setting.label || key)}</div>
            <div class="setting-description">${escapeHtml(setting.description || '')}</div>
            <span class="setting-source">source: ${escapeHtml(setting.source || 'unknown')}</span>
          </div>

          <label class="switch" title="${escapeHtml(key)}">
            <input
              type="checkbox"
              data-trace-setting-key="${escapeHtml(key)}"
              ${setting.value ? 'checked' : ''}
            />
            <span class="slider"></span>
          </label>
        </div>
      `);
    }

    setHtml(
      els.traceSettingsContainer,
      rows.join('') || '<div class="empty">No trace settings returned.</div>'
    );
  }

  function collectTraceSettingsFromUi() {
    const inputs = document.querySelectorAll('[data-trace-setting-key]');
    const settings = {};

    inputs.forEach((input) => {
      settings[input.dataset.traceSettingKey] = input.checked;
    });

    return settings;
  }

  async function saveTraceSettings() {
    try {
      const settings = collectTraceSettingsFromUi();

      setTraceSettingsMessage('Applying runtime settings...');

      const payload = await fetchJson(TRACE_SETTINGS_UPDATE_API_URL, {
        method: 'POST',
        body: JSON.stringify({
          settings,
          updated_by: 'trace_dashboard'
        })
      });

      const updatedKeys = Object.keys(payload.updated || {});
      setTraceSettingsMessage(`Applied runtime settings: ${updatedKeys.join(', ') || 'none'}`);

      await loadTraceSettings();
    } catch (error) {
      setTraceSettingsMessage(`Failed to save settings: ${error.message}`);
    }
  }

  async function clearTraceRuntimeOverrides() {
    const confirmed = window.confirm(
      'Clear all runtime trace overrides? Settings will fall back to .env, then config_trace.py defaults.'
    );

    if (!confirmed) return;

    try {
      setTraceSettingsMessage('Clearing runtime overrides...');

      const payload = await fetchJson(TRACE_SETTINGS_CLEAR_API_URL, {
        method: 'POST',
        body: JSON.stringify({ all: true })
      });

      setTraceSettingsMessage(`Cleared ${payload.cleared_count || 0} runtime override(s).`);
      await loadTraceSettings();
    } catch (error) {
      setTraceSettingsMessage(`Failed to clear runtime overrides: ${error.message}`);
    }
  }

  async function toggleEnvTemplate() {
    if (!els.traceEnvTemplateBox) return;

    if (els.traceEnvTemplateBox.style.display === 'block') {
      els.traceEnvTemplateBox.style.display = 'none';
      return;
    }

    try {
      const payload = await fetchJson(TRACE_ENV_TEMPLATE_API_URL);
      els.traceEnvTemplateBox.textContent = payload.env_template || '';
      els.traceEnvTemplateBox.style.display = 'block';
    } catch (error) {
      els.traceEnvTemplateBox.textContent = `Failed to load .env template: ${error.message}`;
      els.traceEnvTemplateBox.style.display = 'block';
    }
  }

  // ------------------------------------------------------------
  // Trace graph normalization
  // ------------------------------------------------------------

  function computeTotal(nodes, summary) {
    if (summary && summary.total_duration_ms != null) {
      return Math.max(Number(summary.total_duration_ms), 1);
    }

    if (!nodes.length) return 1;

    const maxEnd = Math.max(
      ...nodes.map((node) => {
        const start = Number(node.start || 0);
        const duration = Number(node.duration_ms || 0);
        const end = node.end != null ? Number(node.end) : start + duration;
        return Math.max(end, start + duration, 0);
      })
    );

    return Math.max(maxEnd, 1);
  }

  function normalizeNodes(apiNodes) {
    if (!Array.isArray(apiNodes)) return [];

    return apiNodes.map((node) => {
      const relativeStart = node.relative_start_ms != null
        ? Number(node.relative_start_ms)
        : Number(node.start || 0);

      const relativeEnd = node.relative_end_ms != null
        ? Number(node.relative_end_ms)
        : null;

      const duration = node.duration_ms != null
        ? Number(node.duration_ms)
        : (relativeEnd != null ? Math.max(relativeEnd - relativeStart, 0) : 0);

      const safeStatus = normalizeStatus(node.status || node.raw_status);

      return {
        id: String(node.id),
        trace_id: node.trace_id,
        parent: node.parent ? String(node.parent) : null,
        name: node.function || node.name || '(unnamed span)',
        qualified_name: node.qualified_name || null,
        module_name: node.module_name || node.module || null,
        file_path: node.file_path || node.file || null,
        line_number: node.line_number || node.line || null,
        depth: Number(node.depth || 0),
        duration_ms: duration,
        self_time_ms: node.self_time_ms != null ? Number(node.self_time_ms) : null,
        child_count: node.child_count != null ? Number(node.child_count) : 0,
        status: safeStatus,
        raw_status: node.raw_status || node.status || null,
        request_id: node.request_id || null,
        exception: node.exception || null,
        metadata: node.metadata_json || node.metadata || null,
        started_at: node.started_at || null,
        ended_at: node.ended_at || null,
        start: relativeStart,
        end: relativeEnd != null ? relativeEnd : relativeStart + duration,
        raw: node
      };
    });
  }

  function buildTree(nodes = VISIBLE_NODES) {
    const byId = {};
    const roots = [];

    nodes.forEach((node) => {
      byId[node.id] = { ...node, children: [] };
    });

    nodes.forEach((node) => {
      if (node.parent && byId[node.parent]) {
        byId[node.parent].children.push(byId[node.id]);
      } else {
        roots.push(byId[node.id]);
      }
    });

    roots.sort((a, b) => a.start - b.start);

    Object.values(byId).forEach((node) => {
      node.children.sort((a, b) => a.start - b.start);
    });

    return roots;
  }

  // ------------------------------------------------------------
  // Filters
  // ------------------------------------------------------------

  function getVisibleNodes() {
    const searchTerm = (els.spanSearchInput?.value || '').trim().toLowerCase();
    const statusFilter = els.spanStatusFilter?.value || 'all';
    const rootsOnly = !!els.showOnlyRootsCheck?.checked;

    return CURRENT_NODES.filter((node) => {
      if (rootsOnly && node.parent !== null) return false;
      if (statusFilter !== 'all' && node.status !== statusFilter) return false;

      if (
        searchTerm &&
        !(node.name || '').toLowerCase().includes(searchTerm) &&
        !(node.qualified_name || '').toLowerCase().includes(searchTerm) &&
        !(node.module_name || '').toLowerCase().includes(searchTerm)
      ) {
        return false;
      }

      return true;
    });
  }

  function getFilteredRecentTraces() {
    const searchTerm = (els.searchInput?.value || '').trim().toLowerCase();
    const traceStatus = els.traceStatusFilter?.value || 'all';

    return RECENT_TRACES.filter((trace) => {
      const traceId = String(trace.trace_id || '').toLowerCase();
      const requestId = String(trace.request_id || '').toLowerCase();
      const rootFunction = String(trace.root_function || '').toLowerCase();
      const status = String(trace.status || '').toLowerCase();

      if (
        searchTerm &&
        !traceId.includes(searchTerm) &&
        !requestId.includes(searchTerm) &&
        !rootFunction.includes(searchTerm) &&
        !status.includes(searchTerm)
      ) {
        return false;
      }

      if (traceStatus === 'has_errors' && !(Number(trace.error_count || 0) > 0)) return false;
      if (traceStatus === 'no_errors' && Number(trace.error_count || 0) > 0) return false;

      return true;
    });
  }

  // ------------------------------------------------------------
  // Stats / detail
  // ------------------------------------------------------------

  function renderStats() {
    const summary = CURRENT_SUMMARY || {};
    const rootCount = CURRENT_NODES.filter((node) => node.parent === null).length;

    setText(
      els.statSpans,
      summary.span_count != null ? String(summary.span_count) : String(CURRENT_NODES.length)
    );

    setText(els.statVisible, String(VISIBLE_NODES.length));
    setText(els.statTotal, formatMs(TOTAL));

    setText(
      els.statOk,
      summary.ok_count != null
        ? String(summary.ok_count)
        : String(CURRENT_NODES.filter((node) => node.status === 'ok').length)
    );

    setText(
      els.statError,
      summary.error_count != null
        ? String(summary.error_count)
        : String(CURRENT_NODES.filter((node) => node.status === 'error').length)
    );

    setText(els.statRoots, String(rootCount));

    if (CURRENT_TRACE_ID) {
      setText(els.selectedTraceId, CURRENT_TRACE_ID);

      const started = summary.started_at ? formatDate(summary.started_at) : '—';
      const ended = summary.ended_at ? formatDate(summary.ended_at) : '—';

      setText(els.traceRange, `Started: ${started} · Ended: ${ended}`);
    } else {
      setText(els.selectedTraceId, 'No trace selected');
      setText(els.traceRange, 'Choose a trace from the left panel.');
    }

    if (els.legacyTraceMeta) {
      els.legacyTraceMeta.innerText = CURRENT_TRACE_ID
        ? `Trace ID: ${CURRENT_TRACE_ID} | Nodes: ${CURRENT_NODES.length}`
        : 'No trace selected';
    }
  }

  function renderDetail(node) {
    if (!els.detail) return;

    if (!node) {
      setHtml(els.detail, '<div class="detail-empty">Click any span to inspect it.</div>');
      if (els.legacyNodeDetails) els.legacyNodeDetails.innerText = '';
      return;
    }

    let metadataText = '—';

    if (node.metadata) {
      try {
        metadataText = JSON.stringify(node.metadata, null, 2);
      } catch (error) {
        metadataText = String(node.metadata);
      }
    }

    setHtml(els.detail, `
      <div class="detail-grid">
        <div class="kv">
          <div class="kv-label">Function</div>
          <div class="kv-value">${escapeHtml(node.name)}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Qualified Name</div>
          <div class="kv-value">${escapeHtml(node.qualified_name || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Status</div>
          <div class="kv-value">
            <span class="status-pill ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span>
          </div>
        </div>

        <div class="kv">
          <div class="kv-label">Raw Status</div>
          <div class="kv-value">${escapeHtml(node.raw_status || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Span ID</div>
          <div class="kv-value">${escapeHtml(node.id)}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Parent Span ID</div>
          <div class="kv-value">${escapeHtml(node.parent || 'null')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Trace ID</div>
          <div class="kv-value">${escapeHtml(node.trace_id || CURRENT_TRACE_ID || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Module / File</div>
          <div class="kv-value">${escapeHtml(node.module_name || '—')}\n${escapeHtml(node.file_path || '—')}:${escapeHtml(node.line_number || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Depth / Children</div>
          <div class="kv-value">depth=${escapeHtml(String(node.depth))}, children=${escapeHtml(String(node.child_count || 0))}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Duration</div>
          <div class="kv-value">${escapeHtml(formatMs(node.duration_ms))}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Self Time</div>
          <div class="kv-value">${escapeHtml(node.self_time_ms == null ? '—' : formatMs(node.self_time_ms))}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Relative Start / End</div>
          <div class="kv-value">${escapeHtml(formatMs(node.start))} → ${escapeHtml(formatMs(node.end))}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Started At</div>
          <div class="kv-value">${escapeHtml(node.started_at || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Ended At</div>
          <div class="kv-value">${escapeHtml(node.ended_at || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Request ID</div>
          <div class="kv-value">${escapeHtml(node.request_id || '—')}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Metadata</div>
          <div class="kv-value">${escapeHtml(metadataText)}</div>
        </div>

        <div class="kv">
          <div class="kv-label">Exception</div>
          <div class="kv-value">${escapeHtml(node.exception || '—')}</div>
        </div>

        <div class="toolbar-row">
          <button class="button info" id="copySpanIdBtn" type="button">Copy Span ID</button>
          <button class="button info" id="copyParentIdBtn" type="button">Copy Parent ID</button>
        </div>
      </div>
    `);

    if (els.legacyNodeDetails) {
      els.legacyNodeDetails.innerText = JSON.stringify(node.raw || node, null, 2);
    }

    const copySpanIdBtn = document.getElementById('copySpanIdBtn');
    const copyParentIdBtn = document.getElementById('copyParentIdBtn');

    if (copySpanIdBtn) {
      copySpanIdBtn.onclick = async () => {
        await copyText(node.id || '');
        setStatus(`Copied span id ${node.id}`);
      };
    }

    if (copyParentIdBtn) {
      copyParentIdBtn.onclick = async () => {
        await copyText(node.parent || '');
        setStatus(node.parent ? `Copied parent span id ${node.parent}` : 'No parent span id to copy.');
      };
    }
  }

  function selectNode(node) {
    if (!node) return;

    selectedId = node.id;
    renderDetail(node);
    render();
  }

  // ------------------------------------------------------------
  // Timeline rendering
  // ------------------------------------------------------------

  function createRowShell(node, indent) {
    const row = document.createElement('div');
    row.className = `span-row ${selectedId === node.id ? 'selected' : ''}`;

    const left = Math.max((Number(node.start || 0) / TOTAL) * 100, 0).toFixed(2);
    const width = Math.max((Number(node.duration_ms || 0) / TOTAL) * 100, 1.2).toFixed(2);
    const safeStatus = node.status || 'unknown';

    row.innerHTML = `
      <div class="span-name ${safeStatus}" style="padding-left:${indent}px">
        <span class="span-name-text">${escapeHtml(node.name)}</span>
      </div>

      <div class="bar-track">
        <div class="bar-fill ${safeStatus}" style="left:${left}%;width:${width}%"></div>
      </div>

      <div class="span-ms">${escapeHtml(formatMs(node.duration_ms))}</div>
    `;

    row.onclick = () => selectNode(node);

    return row;
  }

  function renderGantt() {
    if (!els.view) return;

    els.view.innerHTML = '';

    if (!VISIBLE_NODES.length) {
      els.view.innerHTML = '<div class="empty">No spans match the current filters.</div>';
      if (els.axis) els.axis.innerHTML = '';
      return;
    }

    const sorted = [...VISIBLE_NODES].sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      if (a.depth !== b.depth) return a.depth - b.depth;
      return a.name.localeCompare(b.name);
    });

    sorted.forEach((node) => {
      const row = createRowShell(node, node.depth * 14);
      els.view.appendChild(row);
    });

    renderAxis();
  }

  function renderWaterfallNode(node, container) {
    const isCollapsed = !!collapsed[node.id];
    const hasChildren = Array.isArray(node.children) && node.children.length > 0;

    const row = document.createElement('div');
    row.className = `span-row ${selectedId === node.id ? 'selected' : ''}`;

    const left = Math.max((Number(node.start || 0) / TOTAL) * 100, 0).toFixed(2);
    const width = Math.max((Number(node.duration_ms || 0) / TOTAL) * 100, 1.2).toFixed(2);
    const safeStatus = node.status || 'unknown';

    const iconHtml = hasChildren
      ? `<span class="toggle-icon" data-id="${escapeHtml(node.id)}">${isCollapsed ? '▶' : '▼'}</span>`
      : '<span class="toggle-icon"></span>';

    row.innerHTML = `
      <div class="span-name ${safeStatus}" style="padding-left:${node.depth * 14}px">
        ${iconHtml}
        <span class="span-name-text">${escapeHtml(node.name)}</span>
      </div>

      <div class="bar-track">
        <div class="bar-fill ${safeStatus}" style="left:${left}%;width:${width}%"></div>
      </div>

      <div class="span-ms">${escapeHtml(formatMs(node.duration_ms))}</div>
    `;

    const nameText = row.querySelector('.span-name-text');
    const track = row.querySelector('.bar-track');
    const ms = row.querySelector('.span-ms');

    if (nameText) nameText.onclick = () => selectNode(node);
    if (track) track.onclick = () => selectNode(node);
    if (ms) ms.onclick = () => selectNode(node);

    const toggle = row.querySelector('[data-id]');

    if (toggle) {
      toggle.onclick = (event) => {
        event.stopPropagation();
        collapsed[node.id] = !collapsed[node.id];
        render();
      };
    }

    row.onclick = (event) => {
      if (event.target && event.target.hasAttribute && event.target.hasAttribute('data-id')) return;
      selectNode(node);
    };

    container.appendChild(row);

    if (!isCollapsed && hasChildren) {
      node.children.forEach((child) => renderWaterfallNode(child, container));
    }
  }

  function renderWaterfall() {
    if (!els.view) return;

    els.view.innerHTML = '';

    if (!VISIBLE_NODES.length) {
      els.view.innerHTML = '<div class="empty">No spans match the current filters.</div>';
      if (els.axis) els.axis.innerHTML = '';
      return;
    }

    const roots = buildTree(VISIBLE_NODES);
    roots.forEach((root) => renderWaterfallNode(root, els.view));

    renderAxis();
  }

  function renderAxis() {
    if (!els.axis) return;

    els.axis.innerHTML = '';

    if (!VISIBLE_NODES.length) return;

    [0, 25, 50, 75, 100].forEach((pct) => {
      const tick = document.createElement('div');
      tick.className = 'time-tick';
      tick.textContent = formatMs((pct / 100) * TOTAL);
      els.axis.appendChild(tick);
    });
  }

  function renderLegacyTree() {
    if (!els.legacyTraceTree) return;

    els.legacyTraceTree.innerHTML = '';

    VISIBLE_NODES.forEach((node) => {
      const div = document.createElement('div');
      div.className = `trace-node ${selectedId === node.id ? 'selected' : ''}`;
      div.style.paddingLeft = `${10 + node.depth * 14}px`;
      div.innerText = `${node.name} (${formatMs(node.duration_ms)})`;

      div.onclick = () => selectNode(node);

      els.legacyTraceTree.appendChild(div);
    });
  }

  function applySpanFilters() {
    VISIBLE_NODES = getVisibleNodes();

    if (selectedId && !CURRENT_NODES.find((node) => node.id === selectedId)) {
      selectedId = null;
      renderDetail(null);
    }

    const selectedVisible = selectedId && VISIBLE_NODES.find((node) => node.id === selectedId);

    if (!selectedVisible && selectedId) {
      const originalNode = CURRENT_NODES.find((node) => node.id === selectedId);
      renderDetail(originalNode || null);
    }
  }

  function render() {
    applySpanFilters();
    renderStats();
    renderLegacyTree();

    if (!els.view) return;

    if (!CURRENT_NODES.length) {
      els.view.innerHTML = '<div class="empty">Load a trace to render the timeline.</div>';
      if (els.axis) els.axis.innerHTML = '';
      return;
    }

    if (!VISIBLE_NODES.length) {
      els.view.innerHTML = '<div class="empty">No spans match the current filters.</div>';
      if (els.axis) els.axis.innerHTML = '';
      return;
    }

    if (currentView === 'gantt') {
      renderGantt();
    } else {
      renderWaterfall();
    }
  }

  function updateViewButtons() {
    if (!els.waterfallBtn || !els.ganttBtn) return;

    if (currentView === 'waterfall') {
      els.waterfallBtn.classList.add('active');
      els.ganttBtn.classList.remove('active');
    } else {
      els.waterfallBtn.classList.remove('active');
      els.ganttBtn.classList.add('active');
    }
  }

  // ------------------------------------------------------------
  // Recent traces
  // ------------------------------------------------------------

  function renderRecentTraces() {
    if (!els.traceList) return;

    FILTERED_RECENT_TRACES = getFilteredRecentTraces();

    if (!FILTERED_RECENT_TRACES.length) {
      els.traceList.innerHTML = '<div class="empty">No traces match the current filters.</div>';
      return;
    }

    els.traceList.innerHTML = '';

    FILTERED_RECENT_TRACES.forEach((trace) => {
      const card = document.createElement('div');

      card.className = `trace-card trace-item ${trace.trace_id === CURRENT_TRACE_ID ? 'active' : ''}`;

      card.innerHTML = `
        <div class="trace-card-header">
          <div class="trace-id">${escapeHtml(trace.trace_id)}</div>
          <button class="copy-btn-mini" type="button" data-copy-trace-id="${escapeHtml(trace.trace_id)}">Copy</button>
        </div>

        <div class="trace-meta">
          <div>
            <strong>Started</strong>
            <span>${escapeHtml(formatDate(trace.last_seen || trace.started_at))}</span>
          </div>

          <div>
            <strong>Total</strong>
            <span>${escapeHtml(formatMs(trace.total_duration_ms || trace.duration_ms))}</span>
          </div>

          <div>
            <strong>Spans</strong>
            <span>${escapeHtml(String(trace.span_count ?? 0))}</span>
          </div>

          <div>
            <strong>Errors</strong>
            <span>${escapeHtml(String(trace.error_count ?? 0))}</span>
          </div>

          <div>
            <strong>Root</strong>
            <span>${escapeHtml(trace.root_function || 'Unknown Root')}</span>
          </div>

          <div>
            <strong>Status</strong>
            <span>${escapeHtml(trace.status || '—')}</span>
          </div>
        </div>

        <div class="status-bar">
          <span class="mini-pill ok">ok ${escapeHtml(String(trace.ok_count ?? 0))}</span>
          <span class="mini-pill error">error ${escapeHtml(String(trace.error_count ?? 0))}</span>
          <span class="mini-pill info">${escapeHtml(formatMs(trace.total_duration_ms || trace.duration_ms))}</span>
        </div>

        <div class="small-note">
          request_id=${escapeHtml(trace.request_id || '—')}
        </div>
      `;

      card.onclick = () => loadTrace(trace.trace_id);

      const copyBtn = card.querySelector('[data-copy-trace-id]');

      if (copyBtn) {
        copyBtn.onclick = async (event) => {
          event.stopPropagation();

          const value = copyBtn.getAttribute('data-copy-trace-id') || '';

          await copyText(value);
          setStatus(`Copied trace id ${value}`);
        };
      }

      els.traceList.appendChild(card);
    });
  }

  // ------------------------------------------------------------
  // Loaders
  // ------------------------------------------------------------

  async function loadRecentTraces(autoSelect = true) {
    if (!els.traceList) return;

    const limitValue = getLimitValue();

    els.traceList.innerHTML = '<div class="loading">Loading recent traces...</div>';
    setStatus(`Loading recent traces (limit ${limitValue})...`);

    try {
      const payload = await fetchJson(buildRecentApiUrl(limitValue), {
        method: 'GET',
        headers: { Accept: 'application/json' }
      });

      RECENT_TRACES = Array.isArray(payload.recent) ? payload.recent : [];

      renderRecentTraces();

      if (autoSelect && FILTERED_RECENT_TRACES.length) {
        const keepSelection = !!els.keepSelectionCheck?.checked;

        const preferredTrace =
          keepSelection &&
          CURRENT_TRACE_ID &&
          FILTERED_RECENT_TRACES.find((trace) => trace.trace_id === CURRENT_TRACE_ID)
            ? CURRENT_TRACE_ID
            : FILTERED_RECENT_TRACES[0].trace_id;

        await loadTrace(preferredTrace, false);
      } else if (!FILTERED_RECENT_TRACES.length) {
        if (!CURRENT_TRACE_ID || !els.keepSelectionCheck?.checked) {
          CURRENT_TRACE_ID = null;
          CURRENT_NODES = [];
          CURRENT_SUMMARY = null;
          VISIBLE_NODES = [];
          TOTAL = 1;
          selectedId = null;
          renderDetail(null);
          render();
        }
      }

      setStatus(
        `Loaded ${RECENT_TRACES.length} trace${RECENT_TRACES.length === 1 ? '' : 's'}, showing ${FILTERED_RECENT_TRACES.length}.`
      );
    } catch (error) {
      const message = error && error.message ? error.message : 'Unknown error loading recent traces';

      els.traceList.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
      setStatus(message);
    }
  }

  async function loadTrace(traceId, rerenderRecent = true) {
    if (!traceId) return;

    CURRENT_TRACE_ID = traceId;

    setText(els.selectedTraceId, traceId);
    setText(els.traceRange, 'Loading trace...');

    if (els.view) {
      els.view.innerHTML = '<div class="loading">Loading trace graph...</div>';
    }

    if (els.axis) {
      els.axis.innerHTML = '';
    }

    if (rerenderRecent) {
      renderRecentTraces();
    }

    setStatus(`Loading trace ${traceId}...`);

    try {
      const payload = await fetchJson(buildGraphApiUrl(traceId), {
        method: 'GET',
        headers: { Accept: 'application/json' }
      });

      CURRENT_SUMMARY = payload.summary || {};
      CURRENT_NODES = normalizeNodes(payload.nodes || []);
      TOTAL = computeTotal(CURRENT_NODES, CURRENT_SUMMARY);

      if (!CURRENT_NODES.length) {
        selectedId = null;
        renderDetail(null);
      } else {
        const preservedSelection = selectedId && CURRENT_NODES.find((node) => node.id === selectedId);

        if (!preservedSelection) {
          selectedId = CURRENT_NODES[0].id;
          renderDetail(CURRENT_NODES[0]);
        } else {
          renderDetail(preservedSelection);
        }
      }

      render();

      if (rerenderRecent) {
        renderRecentTraces();
      }

      setStatus(
        `Loaded trace ${traceId} with ${CURRENT_NODES.length} span${CURRENT_NODES.length === 1 ? '' : 's'}.`
      );
    } catch (error) {
      const message = error && error.message ? error.message : 'Unknown error loading trace graph';

      if (els.view) {
        els.view.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
      }

      if (els.axis) {
        els.axis.innerHTML = '';
      }

      CURRENT_NODES = [];
      CURRENT_SUMMARY = null;
      VISIBLE_NODES = [];
      TOTAL = 1;

      renderStats();
      renderDetail(null);
      setStatus(message);
    }
  }

  // ------------------------------------------------------------
  // Expand / collapse
  // ------------------------------------------------------------

  function collapseAllVisibleRoots() {
    const roots = buildTree(VISIBLE_NODES);

    roots.forEach((root) => {
      collapsed[root.id] = true;
    });

    render();
  }

  function expandAllNodes() {
    collapsed = {};
    render();
  }

  // ------------------------------------------------------------
  // Auto-refresh
  // ------------------------------------------------------------

  function stopAutoRefresh() {
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
  }

  function startAutoRefresh() {
    stopAutoRefresh();

    const interval = Number(els.autoRefreshSelect?.value || 0);

    if (!interval || interval <= 0) {
      setStatus('Auto-refresh disabled.');
      return;
    }

    autoRefreshTimer = window.setInterval(async () => {
      const keepSelection = !!els.keepSelectionCheck?.checked;

      await loadRecentTraces(keepSelection);

      if (!keepSelection && FILTERED_RECENT_TRACES.length) {
        const firstTrace = FILTERED_RECENT_TRACES[0];

        if (firstTrace && firstTrace.trace_id) {
          await loadTrace(firstTrace.trace_id, false);
        }
      } else if (keepSelection && CURRENT_TRACE_ID) {
        await loadTrace(CURRENT_TRACE_ID, false);
      }
    }, interval);

    setStatus(`Auto-refresh enabled every ${Math.round(interval / 1000)} seconds.`);
  }

  // ------------------------------------------------------------
  // Events
  // ------------------------------------------------------------

  function bindEvents() {
    if (els.traceSettingsRefreshBtn) {
      els.traceSettingsRefreshBtn.onclick = () => loadTraceSettings();
    }

    if (els.traceSettingsApplyBtn) {
      els.traceSettingsApplyBtn.onclick = () => saveTraceSettings();
    }

    if (els.traceSettingsClearBtn) {
      els.traceSettingsClearBtn.onclick = () => clearTraceRuntimeOverrides();
    }

    if (els.traceEnvTemplateBtn) {
      els.traceEnvTemplateBtn.onclick = () => toggleEnvTemplate();
    }

    if (els.refreshBtn) {
      els.refreshBtn.onclick = () => loadRecentTraces(true);
    }

    if (els.reloadTraceBtn) {
      els.reloadTraceBtn.onclick = () => {
        if (CURRENT_TRACE_ID) {
          loadTrace(CURRENT_TRACE_ID, false);
        } else {
          loadRecentTraces(true);
        }
      };
    }

    if (els.copyTraceIdBtn) {
      els.copyTraceIdBtn.onclick = async () => {
        if (!CURRENT_TRACE_ID) {
          setStatus('No trace selected to copy.');
          return;
        }

        await copyText(CURRENT_TRACE_ID);
        setStatus(`Copied trace id ${CURRENT_TRACE_ID}`);
      };
    }

    if (els.waterfallBtn) {
      els.waterfallBtn.onclick = () => {
        currentView = 'waterfall';
        updateViewButtons();
        render();
      };
    }

    if (els.ganttBtn) {
      els.ganttBtn.onclick = () => {
        currentView = 'gantt';
        updateViewButtons();
        render();
      };
    }

    if (els.expandAllBtn) {
      els.expandAllBtn.onclick = () => expandAllNodes();
    }

    if (els.collapseAllBtn) {
      els.collapseAllBtn.onclick = () => collapseAllVisibleRoots();
    }

    if (els.limitInput) {
      els.limitInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          loadRecentTraces(true);
        }
      });
    }

    if (els.searchInput) {
      els.searchInput.addEventListener('input', () => {
        renderRecentTraces();
      });
    }

    if (els.traceStatusFilter) {
      els.traceStatusFilter.addEventListener('change', () => {
        renderRecentTraces();
      });
    }

    if (els.autoRefreshSelect) {
      els.autoRefreshSelect.addEventListener('change', () => {
        startAutoRefresh();
      });
    }

    if (els.spanStatusFilter) {
      els.spanStatusFilter.addEventListener('change', () => render());
    }

    if (els.spanSearchInput) {
      els.spanSearchInput.addEventListener('input', () => render());
    }

    if (els.showOnlyRootsCheck) {
      els.showOnlyRootsCheck.addEventListener('change', () => render());
    }

    window.addEventListener('beforeunload', () => {
      stopAutoRefresh();
    });
  }

  // ------------------------------------------------------------
  // Init
  // ------------------------------------------------------------

  function initTraceDashboard() {
    if (initialized) return;
    initialized = true;

    cacheDom();

    if (!els.traceList) {
      console.warn('[trace.js] traceList element was not found. Trace dashboard was not initialized.');
      return;
    }

    if (!hasRequiredDom()) {
      console.warn(
        '[trace.js] Some newer dashboard elements were not found. The script will still try legacy-compatible rendering.'
      );
    }

    updateViewButtons();
    renderStats();
    renderDetail(null);

    bindEvents();

    loadTraceSettings();
    loadRecentTraces(true);
    startAutoRefresh();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTraceDashboard);
  } else {
    initTraceDashboard();
  }

  // Optional debugging hooks from browser console:
  window.EMTAC_TRACE_DASHBOARD = {
    loadRecentTraces,
    loadTrace,
    renderRecentTraces,
    render,
    getState: () => ({
      RECENT_TRACES,
      FILTERED_RECENT_TRACES,
      CURRENT_TRACE_ID,
      CURRENT_NODES,
      CURRENT_SUMMARY,
      VISIBLE_NODES,
      TOTAL,
      selectedId,
      currentView,
      collapsed,
      TRACE_SETTINGS
    })
  };
})();