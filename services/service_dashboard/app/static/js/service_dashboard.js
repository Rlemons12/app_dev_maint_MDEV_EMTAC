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

let autoRefreshTimer = null;
let currentServicesData = [];
let selectedServiceKey = "";

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
    div.innerText = text ?? "";
    return div.innerHTML;
}

function getServiceKey(service) {
    return service?.name || "";
}

function safeArray(value) {
    return Array.isArray(value) ? value : [];
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

function setSelectedServiceByKey(serviceKey) {
    selectedServiceKey = serviceKey || "";
    if (serviceSelector) {
        serviceSelector.value = selectedServiceKey;
    }
    renderSelectedServiceOutput(currentServicesData);
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
        option.textContent = service.name;
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

function renderSelectedServiceOutput(services) {
    if (!selectedServiceName || !selectedServiceOutput) {
        return;
    }

    if (!selectedServiceKey) {
        selectedServiceName.textContent = "No service selected";
        selectedServiceOutput.textContent = "Select a service to view output.";
        return;
    }

    const service = services.find(item => getServiceKey(item) === selectedServiceKey);

    if (!service) {
        selectedServiceName.textContent = "No service selected";
        selectedServiceOutput.textContent = "Selected service is no longer available.";
        return;
    }

    selectedServiceName.textContent = `${service.name} (${service.status || "unknown"})`;
    selectedServiceOutput.textContent = fullOutputText(service);
}

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
        <div><span class="label">Command:</span> <span class="mono">${escapeHtml(service.command || "N/A")}</span></div>
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

async function fetchServices() {
    const response = await fetch("/api/services");
    return await response.json();
}

async function fetchGpuInsights() {
    const response = await fetch("/api/gpu-insights");
    return await response.json();
}

function renderServices(services) {
    if (!servicesContainer) {
        return;
    }

    servicesContainer.innerHTML = "";

    services.forEach(service => {
        const card = document.createElement("div");
        card.className = `service-card ${service.status || "stopped"}`;

        const commandText = service.service_type === "process"
            ? service.command
            : service.start_command;

        card.innerHTML = `
            <div class="service-header">
                <div class="service-title">${escapeHtml(service.name)}</div>
                <div class="status-badge ${service.status === "running" ? "status-running" : "status-stopped"}">
                    ${escapeHtml(service.status || "unknown")}
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
                <button class="btn-start" data-action="start" data-service-name="${encodeURIComponent(service.name)}">Start</button>
                <button class="btn-stop" data-action="stop" data-service-name="${encodeURIComponent(service.name)}">Stop</button>
                <button class="btn-restart" data-action="restart" data-service-name="${encodeURIComponent(service.name)}">Restart</button>
                <button class="btn-clear" data-action="clear" data-service-name="${encodeURIComponent(service.name)}">Clear Output</button>
                <button class="btn-refresh" data-action="select" data-service-name="${encodeURIComponent(service.name)}">View Output</button>
            </div>

            <div class="service-preview-row">
                <div class="service-preview-text">${escapeHtml(lastOutputLine(service))}</div>
            </div>
        `;

        servicesContainer.appendChild(card);
    });
}

async function postServiceAction(encodedServiceName, actionName, actionLabel) {
    try {
        const response = await fetch(`/api/services/${encodedServiceName}/${actionName}`, {
            method: "POST"
        });

        const data = await response.json();
        showMessage(data.message || `${actionLabel} completed.`, Boolean(data.success));
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
    try {
        const response = await fetch(`/api/services/${encodedServiceName}/clear-output`, {
            method: "POST"
        });

        const data = await response.json();
        showMessage(data.message || "Output cleared.", Boolean(data.success));
        await refreshAll();
    } catch (error) {
        showMessage(`Clear failed: ${error.message}`, false);
    }
}

async function refreshAll() {
    try {
        const [servicesData, gpuInsights] = await Promise.all([
            fetchServices(),
            fetchGpuInsights()
        ]);

        if (!servicesData.success) {
            showMessage("Failed to fetch services.", false);
        } else {
            currentServicesData = safeArray(servicesData.services);
            renderServices(currentServicesData);
            syncServiceSelector(currentServicesData);
        }

        renderGpuUsage(gpuInsights?.gpu || { error: "GPU data unavailable." });
        renderGpuStatus(gpuInsights || {});
        renderGpuProcess(gpuInsights?.process || { error: "Process data unavailable." });
    } catch (error) {
        showMessage("Refresh failed: " + error.message, false);
    }
}

function startAutoRefresh() {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
    }

    const seconds = Math.max(1, parseInt(refreshSecondsInput?.value || "3", 10));
    autoRefreshTimer = setInterval(refreshAll, seconds * 1000);
}

document.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) {
        return;
    }

    const encodedServiceName = button.dataset.serviceName;
    const action = button.dataset.action;

    if (action === "start") {
        await startService(encodedServiceName);
    } else if (action === "stop") {
        await stopService(encodedServiceName);
    } else if (action === "restart") {
        await restartService(encodedServiceName);
    } else if (action === "clear") {
        await clearOutput(encodedServiceName);
    } else if (action === "select") {
        selectedServiceKey = decodeURIComponent(encodedServiceName);
        if (serviceSelector) {
            serviceSelector.value = selectedServiceKey;
        }
        renderSelectedServiceOutput(currentServicesData);
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
        selectedServiceKey = serviceSelector.value;
        renderSelectedServiceOutput(currentServicesData);
    });
}

refreshAll();
startAutoRefresh();