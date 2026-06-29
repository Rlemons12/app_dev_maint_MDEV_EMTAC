const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const questionInput = document.querySelector("#questionInput");
const allowDestructive = document.querySelector("#allowDestructive");
const connectionSummary = document.querySelector("#connectionSummary");
const openGrafanaLink = document.querySelector("#openGrafanaLink");
const refreshToolsButton = document.querySelector("#refreshToolsButton");
const mcpPython = document.querySelector("#mcpPython");
const mcpScript = document.querySelector("#mcpScript");
const mcpTransport = document.querySelector("#mcpTransport");
const mcpHost = document.querySelector("#mcpHost");
const mcpPort = document.querySelector("#mcpPort");
const mcpStatus = document.querySelector("#mcpStatus");
const startMcpButton = document.querySelector("#startMcpButton");
const stopMcpButton = document.querySelector("#stopMcpButton");
const aiProvider = document.querySelector("#aiProvider");
const aiModel = document.querySelector("#aiModel");
const queryDatabase = document.querySelector("#queryDatabase");
const sqlInput = document.querySelector("#sqlInput");
const runQueryButton = document.querySelector("#runQueryButton");
const queryResult = document.querySelector("#queryResult");
const toolSelect = document.querySelector("#toolSelect");
const toolArguments = document.querySelector("#toolArguments");
const runToolButton = document.querySelector("#runToolButton");
const toolDetails = document.querySelector("#toolDetails");
const toolResult = document.querySelector("#toolResult");

let tools = [];

function renderJson(value) {
  return JSON.stringify(value, null, 2);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }

  return payload;
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  chatLog.append(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = busy ? "Working..." : label;
}

async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    connectionSummary.textContent =
      `${settings.postgres_host}:${settings.postgres_port} ` +
      `db=${settings.postgres_default_db} schema=${settings.postgres_default_schema} ` +
      `provider=${settings.ai_provider}`;
    queryDatabase.placeholder = settings.postgres_default_db || "default";
    mcpPython.value = settings.mcp_server_python || "";
    mcpScript.value = settings.mcp_server_script || "";
    openGrafanaLink.href = settings.grafana_url || "http://localhost:3000";
    aiProvider.value = settings.ai_provider || "openai";
    aiModel.value =
      aiProvider.value === "openai"
        ? settings.openai_model || ""
        : settings.hf_router_model || "";
  } catch (error) {
    connectionSummary.textContent = error.message;
  }
}

function renderMcpStatus(statusPayload) {
  if (statusPayload.running) {
    const endpoint = `${statusPayload.config.host || "127.0.0.1"}:${statusPayload.config.port || 8000}`;
    mcpStatus.textContent = `Running (pid ${statusPayload.pid}) ${endpoint}`;
  } else {
    mcpStatus.textContent = "Stopped";
  }
}

async function loadMcpStatus() {
  try {
    const payload = await api("/api/mcp-server/status");
    renderMcpStatus(payload);
  } catch (error) {
    mcpStatus.textContent = error.message;
  }
}

function renderToolDetails() {
  const selected = tools.find((tool) => tool.name === toolSelect.value);

  if (!selected) {
    toolDetails.textContent = "";
    return;
  }

  const schema = selected.input_schema || {};
  const description = selected.description || "No description.";
  const properties = schema.properties ? Object.keys(schema.properties) : [];
  const required = Array.isArray(schema.required) ? schema.required : [];

  toolDetails.innerHTML = `
    <div>${description}</div>
    <div><strong>Fields:</strong> ${properties.length ? properties.join(", ") : "none"}</div>
    <div><strong>Required:</strong> ${required.length ? required.join(", ") : "none"}</div>
  `;
}

async function loadTools() {
  refreshToolsButton.disabled = true;

  try {
    const payload = await api("/api/tools");
    tools = payload.tools || [];
    toolSelect.innerHTML = "";

    for (const tool of tools) {
      const option = document.createElement("option");
      option.value = tool.name;
      option.textContent = tool.name;
      toolSelect.append(option);
    }

    const listDatabases = tools.find((tool) => tool.name === "postgres_list_databases");
    toolSelect.value = listDatabases ? listDatabases.name : tools[0]?.name || "";
    toolArguments.value = "{}";
    renderToolDetails();
  } catch (error) {
    toolResult.textContent = error.message;
  } finally {
    refreshToolsButton.disabled = false;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  addMessage("user", question);
  questionInput.value = "";
  const sendButton = chatForm.querySelector("button");
  setBusy(sendButton, true, "Send");

  try {
    const payload = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        allow_destructive: allowDestructive.checked,
        provider: aiProvider.value || null,
        model: aiModel.value.trim() || null,
      }),
    });
    addMessage("assistant", payload.answer || "");
  } catch (error) {
    addMessage("error", error.message);
  } finally {
    setBusy(sendButton, false, "Send");
  }
});

startMcpButton.addEventListener("click", async () => {
  setBusy(startMcpButton, true, "Start MCP Server");
  try {
    const payload = await api("/api/mcp-server/start", {
      method: "POST",
      body: JSON.stringify({
        python_executable: mcpPython.value.trim() || null,
        server_script: mcpScript.value.trim() || null,
        transport: mcpTransport.value,
        host: mcpHost.value.trim() || "127.0.0.1",
        port: Number.parseInt(mcpPort.value, 10) || 8000,
      }),
    });
    renderMcpStatus({
      running: payload.status === "started" || payload.status === "already_running",
      pid: payload.pid || null,
      config: payload.config || {},
    });
  } catch (error) {
    mcpStatus.textContent = error.message;
  } finally {
    setBusy(startMcpButton, false, "Start MCP Server");
  }
});

stopMcpButton.addEventListener("click", async () => {
  setBusy(stopMcpButton, true, "Stop MCP Server");
  try {
    await api("/api/mcp-server/stop", {
      method: "POST",
      body: JSON.stringify({}),
    });
    await loadMcpStatus();
  } catch (error) {
    mcpStatus.textContent = error.message;
  } finally {
    setBusy(stopMcpButton, false, "Stop MCP Server");
  }
});

runQueryButton.addEventListener("click", async () => {
  const sql = sqlInput.value.trim();

  if (!sql) {
    return;
  }

  setBusy(runQueryButton, true, "Run Query");
  queryResult.textContent = "";

  try {
    const payload = await api("/api/read-query", {
      method: "POST",
      body: JSON.stringify({
        sql,
        database_name: queryDatabase.value.trim() || null,
      }),
    });
    queryResult.textContent = renderJson(payload.result);
  } catch (error) {
    queryResult.textContent = error.message;
  } finally {
    setBusy(runQueryButton, false, "Run Query");
  }
});

runToolButton.addEventListener("click", async () => {
  setBusy(runToolButton, true, "Run Tool");
  toolResult.textContent = "";

  try {
    const argumentsJson = toolArguments.value.trim() || "{}";
    const argumentsValue = JSON.parse(argumentsJson);

    const payload = await api("/api/tool", {
      method: "POST",
      body: JSON.stringify({
        name: toolSelect.value,
        arguments: argumentsValue,
      }),
    });
    toolResult.textContent = renderJson(payload.result);
  } catch (error) {
    toolResult.textContent = error.message;
  } finally {
    setBusy(runToolButton, false, "Run Tool");
  }
});

toolSelect.addEventListener("change", renderToolDetails);
refreshToolsButton.addEventListener("click", loadTools);

loadSettings();
loadMcpStatus();
loadTools();
