// static/js/service_dashboard.js

const servicesContainer = document.getElementById("servicesContainer");
const messageBar = document.getElementById("messageBar");

const gpuUsageBlock = document.getElementById("gpuUsageBlock");
const gpuStatusBlock = document.getElementById("gpuStatusBlock");
const gpuProcessBlock = document.getElementById("gpuProcessBlock");

const gpuBackendPill = document.getElementById("gpuBackendPill");
const gpuServicePill = document.getElementById("gpuServicePill");

const refreshSecondsInput = document.getElementById("refreshSeconds");
const refreshAllBtn = document.getElementById("refreshAllBtn");

const serviceSelector = document.getElementById("serviceSelector");
const selectedServiceName = document.getElementById("selectedServiceName");
const selectedServiceOutput = document.getElementById("selectedServiceOutput");

const serviceCountPill = document.getElementById("serviceCountPill");
const stripSvcCount = document.getElementById("strip-svc-count");
const stripSvcRunning = document.getElementById("strip-svc-running");

const serviceTabsList = document.getElementById("serviceTabsList");

const svcDetailName = document.getElementById("svcDetailName");
const svcDetailMetaTop = document.getElementById("svcDetailMetaTop");
const svcDetailStatusPill = document.getElementById("svcDetailStatusPill");
const svcDetailType = document.getElementById("svcDetailType");
const svcDetailPid = document.getElementById("svcDetailPid");
const svcDetailUptime = document.getElementById("svcDetailUptime");
const svcDetailLogFile = document.getElementById("svcDetailLogFile");
const svcDetailCwd = document.getElementById("svcDetailCwd");
const svcDetailCommand = document.getElementById("svcDetailCommand");
const svcDetailOutput = document.getElementById("svcDetailOutput");

const svcStartBtn = document.getElementById("svcStartBtn");
const svcStopBtn = document.getElementById("svcStopBtn");
const svcRestartBtn = document.getElementById("svcRestartBtn");
const svcClearBtn = document.getElementById("svcClearBtn");
const svcViewOutputBtn = document.getElementById("svcViewOutputBtn");

let autoRefreshTimer = null;
let currentServicesData = [];
let selectedServiceKey = "";

/* ─────────────────────────────────────────────
   General helpers
───────────────────────────────────────────── */
function showMessage(message, isSuccess = true) {
    if (!messageBar) {
        return;
    }

    messageBar.textContent = message;
    messageBar.className = "message-bar " + (isSuccess ? "message-success" : "message-error");
    messageBar.style.display = "block";

    setTimeout(() => {
        messageBar.style.display = "none";
    }, 3000);
}

function safeArray(value) {
    return Array.isArray(value) ? value : [];
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.innerText = text ?? "";
    return div.innerHTML;
}

function getServiceKey(service) {
    return service?.name || "";
}

function encodeServiceName(serviceName) {
    return encodeURIComponent(serviceName || "");
}

function formatUptime(seconds) {
    if (seconds === null || seconds === undefined || seconds === "") {
        return "N/A";
    }

    const value = Number(seconds);

    if (Number.isNaN(value)) {
        return "N/A";
    }

    const hrs = Math.floor(value / 3600);
    const mins = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);

    if (hrs > 0) {
        return `${hrs}h ${mins}m ${secs}s`;
    }

    if (mins > 0) {
        return `${mins}m ${secs}s`;
    }

    return `${secs}s`;
}

function normalizeCommand(command) {
    if (Array.isArray(command)) {
        return command.join(" ");
    }

    if (command === null || command === undefined || command === "") {
        return "N/A";
    }

    return String(command);
}

function getServiceCommand(service) {
    if (!service) {
        return "N/A";
    }

    if (service.service_type === "process") {
        return normalizeCommand(service.command);
    }

    if (service.start_command !== undefined && service.start_command !== null) {
        return normalizeCommand(service.start_command);
    }

    return normalizeCommand(service.command);
}

function lastOutputLine(service) {
    const output = safeArray(service?.output);

    if (!output.length) {
        return "No output yet.";
    }

    return output[output.length - 1] || "No output yet.";
}

function fullOutputText(service) {
    const output = safeArray(service?.output);

    if (!output.length) {
        return "No output yet.";
    }

    return output.join("\n");
}

function getSelectedService(services = currentServicesData) {
    if (!selectedServiceKey) {
        return null;
    }

    return safeArray(services).find(item => getServiceKey(item) === selectedServiceKey) || null;
}

function getStatusDotClass(status) {
    const value = String(status || "").toLowerCase();

    if (value === "running") {
        return "svc-dot running";
    }

    if (value === "stopped") {
        return "svc-dot stopped";
    }

    return "svc-dot unknown";
}

function getStatusPillClass(status) {
    const value = String(status || "").toLowerCase();

    if (value === "running") {
        return "pill pill-ok";
    }

    if (value === "stopped" || value === "error") {
        return "pill pill-err";
    }

    return "pill pill-warn";
}

function setText(element, value) {
    if (element) {
        element.textContent = value;
    }
}

function setHtml(element, value) {
    if (element) {
        element.innerHTML = value;
    }
}

/* ─────────────────────────────────────────────
   Service selector / selected state
───────────────────────────────────────────── */
function setSelectedServiceByKey(serviceKey) {
    selectedServiceKey = serviceKey || "";

    if (serviceSelector) {
        serviceSelector.value = selectedServiceKey;
    }

    renderSelectedServiceOutput(currentServicesData);
    renderTabbedServices(currentServicesData);
}

function syncServiceSelector(services) {
    if (!serviceSelector) {
        return;
    }

    const previousValue = selectedServiceKey || serviceSelector.value || "";

    serviceSelector.innerHTML = '<option value="">Choose a service...</option>';

    services.forEach(service => {
        const key = getServiceKey(service);
        const option = document.createElement("option");

        option.value = key;
        option.textContent = service.name || key || "Unknown service";

        serviceSelector.appendChild(option);
    });

    const stillExists = services.some(service => getServiceKey(service) === previousValue);

    if (stillExists) {
        selectedServiceKey = previousValue;
    } else if (services.length > 0) {
        selectedServiceKey = getServiceKey(services[0]);
    } else {
        selectedServiceKey = "";
    }

    serviceSelector.value = selectedServiceKey;

    renderSelectedServiceOutput(services);
}

/* ─────────────────────────────────────────────
   Managed services strip counts
───────────────────────────────────────────── */
function updateManagedServicesCounts(services) {
    const serviceList = safeArray(services);

    const runningCount = serviceList.filter(service => {
        return String(service.status || "").toLowerCase() === "running";
    }).length;

    if (serviceCountPill) {
        serviceCountPill.textContent = `${serviceList.length} services`;
    }

    if (stripSvcCount) {
        stripSvcCount.textContent = String(serviceList.length);
    }

    if (stripSvcRunning) {
        stripSvcRunning.textContent = `${runningCount} running`;
    }
}

/* ─────────────────────────────────────────────
   Selected service detail panel
───────────────────────────────────────────── */
function setDetailButtonsEnabled(enabled) {
    const buttons = [
        svcStartBtn,
        svcStopBtn,
        svcRestartBtn,
        svcClearBtn,
        svcViewOutputBtn
    ];

    buttons.forEach(button => {
        if (button) {
            button.disabled = !enabled;
        }
    });
}

function clearSelectedServiceDetail() {
    setText(svcDetailName, "No service selected");
    setText(svcDetailMetaTop, "Select a service to view details.");

    if (svcDetailStatusPill) {
        svcDetailStatusPill.className = "pill pill-warn";
        svcDetailStatusPill.textContent = "No selection";
    }

    setText(svcDetailType, "—");
    setText(svcDetailPid, "—");
    setText(svcDetailUptime, "—");
    setText(svcDetailLogFile, "—");
    setText(svcDetailCwd, "—");
    setText(svcDetailCommand, "—");
    setText(svcDetailOutput, "Select a service to view output.");

    if (selectedServiceName) {
        selectedServiceName.textContent = "No service selected";
    }

    if (selectedServiceOutput) {
        selectedServiceOutput.textContent = "Select a service to view output.";
    }

    setDetailButtonsEnabled(false);
}

function renderSelectedServiceOutput(services) {
    const service = getSelectedService(services);

    if (!service) {
        if (selectedServiceName) {
            selectedServiceName.textContent = "No service selected";
        }

        if (selectedServiceOutput) {
            selectedServiceOutput.textContent = selectedServiceKey
                ? "Selected service is no longer available."
                : "Select a service to view output.";
        }

        if (svcDetailOutput) {
            svcDetailOutput.textContent = selectedServiceKey
                ? "Selected service is no longer available."
                : "Select a service to view output.";
        }

        return;
    }

    const outputText = fullOutputText(service);

    if (selectedServiceName) {
        selectedServiceName.textContent = `${service.name} (${service.status || "unknown"})`;
    }

    if (selectedServiceOutput) {
        selectedServiceOutput.textContent = outputText;
    }

    if (svcDetailOutput) {
        svcDetailOutput.textContent = outputText;
    }
}

function renderSelectedServiceDetail(service) {
    if (!service) {
        clearSelectedServiceDetail();
        return;
    }

    const status = service.status || "unknown";
    const type = service.service_type || "N/A";
    const pid = service.pid ?? "N/A";
    const uptime = formatUptime(service.uptime_seconds);
    const logFile = service.log_file || "N/A";
    const cwd = service.cwd || "N/A";
    const command = getServiceCommand(service);

    setText(svcDetailName, service.name || "Unknown service");
    setText(svcDetailMetaTop, `${status} · ${type}`);

    if (svcDetailStatusPill) {
        svcDetailStatusPill.className = getStatusPillClass(status);
        svcDetailStatusPill.textContent = String(status).toUpperCase();
    }

    setText(svcDetailType, type);
    setText(svcDetailPid, String(pid));
    setText(svcDetailUptime, uptime);
    setText(svcDetailLogFile, logFile);
    setText(svcDetailCwd, cwd);
    setText(svcDetailCommand, command);

    renderSelectedServiceOutput(currentServicesData);
    setDetailButtonsEnabled(true);
}

/* ─────────────────────────────────────────────
   Tab rendering
───────────────────────────────────────────── */
function hasTabbedServiceLayout() {
    return Boolean(serviceTabsList && svcDetailName && svcDetailOutput);
}

function renderServiceTabs(services) {
    if (!serviceTabsList) {
        return;
    }

    serviceTabsList.innerHTML = "";

    if (!services.length) {
        serviceTabsList.innerHTML = `
            <div class="svc-detail-line" style="padding:8px 10px;color:var(--txt-dim);">
                No services found.
            </div>
        `;
        return;
    }

    services.forEach(service => {
        const key = getServiceKey(service);
        const status = service.status || "unknown";
        const type = service.service_type || "N/A";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "svc-tab-btn" + (key === selectedServiceKey ? " active" : "");

        button.innerHTML = `
            <span class="${getStatusDotClass(status)}"></span>
            <span class="svc-tab-main">
                <div class="svc-tab-name">${escapeHtml(service.name || key || "Unknown service")}</div>
                <div class="svc-tab-sub">${escapeHtml(status)} · ${escapeHtml(type)}</div>
            </span>
        `;

        button.addEventListener("click", () => {
            setSelectedServiceByKey(key);
        });

        serviceTabsList.appendChild(button);
    });
}

function renderTabbedServices(services) {
    if (!hasTabbedServiceLayout()) {
        return;
    }

    const serviceList = safeArray(services);

    if (!serviceList.length) {
        selectedServiceKey = "";
        renderServiceTabs([]);
        clearSelectedServiceDetail();
        updateManagedServicesCounts(serviceList);
        return;
    }

    const selectedStillExists = serviceList.some(service => getServiceKey(service) === selectedServiceKey);

    if (!selectedServiceKey || !selectedStillExists) {
        selectedServiceKey = getServiceKey(serviceList[0]);
    }

    if (serviceSelector) {
        serviceSelector.value = selectedServiceKey;
    }

    renderServiceTabs(serviceList);
    renderSelectedServiceDetail(getSelectedService(serviceList));
    updateManagedServicesCounts(serviceList);
}

/* ─────────────────────────────────────────────
   Legacy card rendering
   Kept as compatibility fallback. If your new HTML hides
   #servicesContainer, this still gives old code something to use.
───────────────────────────────────────────── */
function renderLegacyServiceCards(services) {
    if (!servicesContainer) {
        return;
    }

    servicesContainer.innerHTML = "";

    services.forEach(service => {
        const card = document.createElement("div");
        const status = service.status || "stopped";
        const commandText = getServiceCommand(service);

        card.className = `service-card ${status}`;
        card.dataset.serviceName = service.name || "";
        card.dataset.serviceType = service.service_type || "";
        card.dataset.pid = service.pid ?? "";
        card.dataset.uptime = formatUptime(service.uptime_seconds);
        card.dataset.cwd = service.cwd || "";
        card.dataset.command = commandText;
        card.dataset.logFile = service.log_file || "";

        card.innerHTML = `
            <div class="service-header">
                <div class="service-title svc-name">${escapeHtml(service.name || "Unknown service")}</div>
                <div class="status-badge ${status === "running" ? "status-running" : "status-stopped"}">
                    ${escapeHtml(status)}
                </div>
            </div>

            <div class="service-meta">
                <div><span class="label">Type:</span> ${escapeHtml(service.service_type || "N/A")}</div>
                <div><span class="label">PID:</span> ${service.pid ?? "N/A"}</div>
                <div><span class="label">Uptime:</span> ${formatUptime(service.uptime_seconds)}</div>
                <div><span class="label">Working Dir:</span> <span class="mono">${escapeHtml(service.cwd ?? "N/A")}</span></div>
                <div><span class="label">Command:</span> <span class="mono">${escapeHtml(commandText || "N/A")}</span></div>
                <div><span class="label">Log File:</span> <span class="mono">${escapeHtml(service.log_file || "N/A")}</span></div>
            </div>

            <div class="service-actions">
                <button class="btn-start" data-action="start" data-service-name="${encodeServiceName(service.name)}">Start</button>
                <button class="btn-stop" data-action="stop" data-service-name="${encodeServiceName(service.name)}">Stop</button>
                <button class="btn-restart" data-action="restart" data-service-name="${encodeServiceName(service.name)}">Restart</button>
                <button class="btn-clear" data-action="clear" data-service-name="${encodeServiceName(service.name)}">Clear Output</button>
                <button class="btn-refresh" data-action="select" data-service-name="${encodeServiceName(service.name)}">View Output</button>
            </div>

            <div class="service-preview-row">
                <div class="service-preview-text">${escapeHtml(lastOutputLine(service))}</div>
            </div>
        `;

        servicesContainer.appendChild(card);
    });
}

function renderServices(services) {
    const serviceList = safeArray(services);

    renderLegacyServiceCards(serviceList);
    renderTabbedServices(serviceList);
    updateManagedServicesCounts(serviceList);
}

/* ─────────────────────────────────────────────
   GPU rendering
───────────────────────────────────────────── */
function renderGpuUsage(gpuPayload) {
    if (!gpuUsageBlock || !gpuBackendPill) {
        return;
    }

    if (!gpuPayload || gpuPayload.error) {
        gpuUsageBlock.innerHTML = `<div class="error-text">${escapeHtml(gpuPayload?.error || "GPU data unavailable.")}</div>`;
        gpuBackendPill.textContent = "GPU backend: unavailable";
        return;
    }

    gpuBackendPill.textContent = `GPU backend: ${gpuPayload.backend || "-"}`;

    const gpus = safeArray(gpuPayload.gpus);

    if (!gpus.length) {
        gpuUsageBlock.innerHTML = `<div class="muted">No GPUs detected.</div>`;
        return;
    }

    let html = `
        <table>
            <thead>
                <tr>
                    <th>GPU</th>
                    <th>Util</th>
                    <th>VRAM</th>
                    <th>Temp</th>
                    <th>Power</th>
                </tr>
            </thead>
            <tbody>
    `;

    gpus.forEach(gpu => {
        const util = `${gpu.utilization_gpu_pct ?? 0}%`;
        const vram = `${gpu.mem_used_gb ?? 0} / ${gpu.mem_total_gb ?? 0} GB`;
        const temp = gpu.temperature_c == null ? "-" : `${gpu.temperature_c} C`;
        const power = gpu.power_w == null ? "-" : `${gpu.power_w} W`;

        html += `
            <tr>
                <td>${escapeHtml(`${gpu.index} - ${gpu.name}`)}</td>
                <td>${escapeHtml(util)}</td>
                <td>${escapeHtml(vram)}</td>
                <td>${escapeHtml(temp)}</td>
                <td>${escapeHtml(power)}</td>
            </tr>
        `;
    });

    html += `</tbody></table>`;
    gpuUsageBlock.innerHTML = html;
}

function renderGpuStatus(insights) {
    if (!gpuStatusBlock || !gpuServicePill) {
        return;
    }

    if (!insights || !insights.service) {
        gpuStatusBlock.innerHTML = `<div class="error-text">GPU service snapshot unavailable.</div>`;
        gpuServicePill.textContent = "GPU service: unavailable";
        return;
    }

    const service = insights.service;
    gpuServicePill.textContent = `GPU service: ${service.status || "-"}`;

    let html = `
        <div><span class="label">Status:</span> ${escapeHtml(service.status || "N/A")}</div>
        <div><span class="label">PID:</span> ${service.pid ?? "N/A"}</div>
        <div><span class="label">Uptime:</span> ${escapeHtml(formatUptime(service.uptime_seconds))}</div>
        <div><span class="label">Working Dir:</span> <span class="mono">${escapeHtml(service.cwd || "N/A")}</span></div>
        <div><span class="label">Command:</span> <span class="mono">${escapeHtml(normalizeCommand(service.command))}</span></div>
    `;

    const statusData = insights.status_data;

    if (statusData && statusData.error) {
        html += `<div class="error-text" style="margin-top:8px;">${escapeHtml(statusData.error)}</div>`;
    } else if (statusData) {
        const modelCount = Object.keys(statusData.models || {}).length;

        html += `<div style="margin-top:8px;"><span class="label">GPU Count:</span> ${statusData.gpus ?? "N/A"}</div>`;
        html += `<div><span class="label">Loaded Models:</span> ${modelCount}</div>`;

        if (modelCount > 0) {
            html += `
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Kind</th>
                            <th>Device</th>
                            <th>Sharded</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            Object.keys(statusData.models).sort().forEach(name => {
                const model = statusData.models[name] || {};

                html += `
                    <tr>
                        <td>${escapeHtml(name)}</td>
                        <td>${escapeHtml(model.kind ?? "-")}</td>
                        <td>${escapeHtml(model.device ?? "-")}</td>
                        <td>${model.sharded ? "yes" : "no"}</td>
                    </tr>
                `;
            });

            html += `</tbody></table>`;
        }
    }

    gpuStatusBlock.innerHTML = html;
}

function renderGpuProcess(processPayload) {
    if (!gpuProcessBlock) {
        return;
    }

    if (!processPayload || processPayload.error) {
        gpuProcessBlock.innerHTML = `<div class="error-text">${escapeHtml(processPayload?.error || "Process data unavailable.")}</div>`;
        return;
    }

    gpuProcessBlock.innerHTML = `
        <div><span class="label">PID:</span> ${processPayload.pid ?? "N/A"}</div>
        <div><span class="label">CPU:</span> ${escapeHtml(String(processPayload.cpu_percent ?? 0))}%</div>
        <div><span class="label">RAM (RSS):</span> ${escapeHtml(String(processPayload.ram_used_gb ?? 0))} GB</div>
        <div><span class="label">RAM (VMS):</span> ${escapeHtml(String(processPayload.ram_vms_gb ?? 0))} GB</div>
        <div><span class="label">Threads:</span> ${processPayload.threads ?? "N/A"}</div>
        <div><span class="label">Create Time:</span> ${escapeHtml(String(processPayload.create_time ?? "N/A"))}</div>
    `;
}

/* ─────────────────────────────────────────────
   API calls
───────────────────────────────────────────── */
async function readJsonResponse(response) {
    const text = await response.text();

    if (!text) {
        return {};
    }

    try {
        return JSON.parse(text);
    } catch (error) {
        return {
            success: false,
            error: `Invalid JSON response: ${error.message}`,
            raw: text
        };
    }
}

async function fetchServices() {
    const response = await fetch("/api/services", {
        cache: "no-store"
    });

    const data = await readJsonResponse(response);

    if (!response.ok) {
        return {
            success: false,
            services: [],
            error: data.error || `HTTP ${response.status}`
        };
    }

    return data;
}

async function fetchGpuInsights() {
    const response = await fetch("/api/gpu-insights", {
        cache: "no-store"
    });

    const data = await readJsonResponse(response);

    if (!response.ok) {
        return {
            error: data.error || `HTTP ${response.status}`
        };
    }

    return data;
}

async function postServiceAction(encodedServiceName, actionName, actionLabel) {
    try {
        const response = await fetch(`/api/services/${encodedServiceName}/${actionName}`, {
            method: "POST"
        });

        const data = await readJsonResponse(response);

        showMessage(
            data.message || data.error || `${actionLabel} completed.`,
            Boolean(data.success)
        );

        await refreshAll();
    } catch (error) {
        showMessage(`${actionLabel} failed: ${error.message}`, false);
    }
}

async function startService(encodedServiceName) {
    await postServiceAction(encodedServiceName, "start", "Start");
}

async function stopService(encodedServiceName) {
    await postServiceAction(encodedServiceName, "stop", "Stop");
}

async function restartService(encodedServiceName) {
    await postServiceAction(encodedServiceName, "restart", "Restart");
}

async function clearOutput(encodedServiceName) {
    await postServiceAction(encodedServiceName, "clear-output", "Clear output");
}

async function runServiceActionByName(serviceName, action) {
    if (!serviceName) {
        showMessage("No service selected.", false);
        return;
    }

    const encodedServiceName = encodeServiceName(serviceName);

    if (action === "start") {
        await startService(encodedServiceName);
    } else if (action === "stop") {
        await stopService(encodedServiceName);
    } else if (action === "restart") {
        await restartService(encodedServiceName);
    } else if (action === "clear") {
        await clearOutput(encodedServiceName);
    } else if (action === "select") {
        setSelectedServiceByKey(serviceName);
    }
}

/* ─────────────────────────────────────────────
   Refresh flow
───────────────────────────────────────────── */
async function refreshAll() {
    const [servicesResult, gpuResult] = await Promise.allSettled([
        fetchServices(),
        fetchGpuInsights()
    ]);

    if (servicesResult.status === "fulfilled") {
        const servicesData = servicesResult.value;

        if (!servicesData.success) {
            showMessage(servicesData.error || "Failed to fetch services.", false);
            currentServicesData = [];
            renderServices([]);
            syncServiceSelector([]);
        } else {
            currentServicesData = safeArray(servicesData.services);
            renderServices(currentServicesData);
            syncServiceSelector(currentServicesData);
        }
    } else {
        showMessage("Failed to fetch services: " + servicesResult.reason.message, false);
        currentServicesData = [];
        renderServices([]);
        syncServiceSelector([]);
    }

    if (gpuResult.status === "fulfilled") {
        const gpuInsights = gpuResult.value;

        renderGpuUsage(gpuInsights?.gpu || { error: "GPU data unavailable." });
        renderGpuStatus(gpuInsights || {});
        renderGpuProcess(gpuInsights?.process || { error: "Process data unavailable." });
    } else {
        const errorMessage = gpuResult.reason?.message || "GPU insight request failed.";

        renderGpuUsage({ error: errorMessage });
        renderGpuStatus({});
        renderGpuProcess({ error: errorMessage });
    }
}

function startAutoRefresh() {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
    }

    const seconds = Math.max(1, parseInt(refreshSecondsInput?.value || "3", 10));

    autoRefreshTimer = setInterval(refreshAll, seconds * 1000);
}

/* ─────────────────────────────────────────────
   Event wiring
───────────────────────────────────────────── */
document.addEventListener("click", async event => {
    const button = event.target.closest("button[data-action]");

    if (!button) {
        return;
    }

    const encodedServiceName = button.dataset.serviceName || "";
    const action = button.dataset.action || "";
    const decodedServiceName = decodeURIComponent(encodedServiceName);

    if (action === "start") {
        await startService(encodedServiceName);
    } else if (action === "stop") {
        await stopService(encodedServiceName);
    } else if (action === "restart") {
        await restartService(encodedServiceName);
    } else if (action === "clear") {
        await clearOutput(encodedServiceName);
    } else if (action === "select") {
        setSelectedServiceByKey(decodedServiceName);
    }
});

if (refreshAllBtn) {
    refreshAllBtn.addEventListener("click", refreshAll);
}

if (refreshSecondsInput) {
    refreshSecondsInput.addEventListener("change", startAutoRefresh);
}

if (serviceSelector) {
    serviceSelector.addEventListener("change", function () {
        setSelectedServiceByKey(serviceSelector.value);
    });
}

if (svcStartBtn) {
    svcStartBtn.addEventListener("click", async function () {
        await runServiceActionByName(selectedServiceKey, "start");
    });
}

if (svcStopBtn) {
    svcStopBtn.addEventListener("click", async function () {
        await runServiceActionByName(selectedServiceKey, "stop");
    });
}

if (svcRestartBtn) {
    svcRestartBtn.addEventListener("click", async function () {
        await runServiceActionByName(selectedServiceKey, "restart");
    });
}

if (svcClearBtn) {
    svcClearBtn.addEventListener("click", async function () {
        await runServiceActionByName(selectedServiceKey, "clear");
    });
}

if (svcViewOutputBtn) {
    svcViewOutputBtn.addEventListener("click", function () {
        renderSelectedServiceOutput(currentServicesData);
    });
}

/* ─────────────────────────────────────────────
   Boot
───────────────────────────────────────────── */
clearSelectedServiceDetail();
refreshAll();
startAutoRefresh();