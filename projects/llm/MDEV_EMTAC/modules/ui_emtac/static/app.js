(function () {
    const state = {
        selectedProblemId: null,
        selectedSolutionId: null,
        selectedTaskId: null,
    };

    const ids = {
        area: document.getElementById("area"),
        equipmentGroup: document.getElementById("equipment-group"),
        model: document.getElementById("model"),
        location: document.getElementById("location"),
        assetNumber: document.getElementById("asset-number"),
        searchBtn: document.getElementById("search-btn"),
        resetBtn: document.getElementById("reset-btn"),
        problems: document.getElementById("problems"),
        solutions: document.getElementById("solutions"),
        tasks: document.getElementById("tasks"),
        taskDetail: document.getElementById("task-detail"),
    };

    if (!ids.area) {
        return;
    }

    async function fetchJson(url) {
        const response = await fetch(url, { headers: { "X-Requested-With": "XMLHttpRequest" } });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
    }

    function setOptions(select, items, placeholder) {
        select.innerHTML = "";
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = placeholder;
        select.appendChild(empty);
        items.forEach((item) => {
            const option = document.createElement("option");
            option.value = item.id;
            option.textContent = item.label;
            select.appendChild(option);
        });
        select.disabled = items.length === 0;
        select.value = "";
    }

    function clearNode(node, message) {
        node.classList.add("empty");
        node.innerHTML = message;
    }

    function escapeHtml(value) {
        return (value || "").replace(/[&<>"']/g, (char) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "'": "&#39;",
        }[char]));
    }

    function renderList(node, items, onClick, activeId, emptyMessage) {
        if (!items.length) {
            clearNode(node, emptyMessage);
            return;
        }
        node.classList.remove("empty");
        node.innerHTML = "";
        items.forEach((item) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `list-item${item.id === activeId ? " active" : ""}`;
            button.innerHTML = `<strong>${escapeHtml(item.name || item.label || `#${item.id}`)}</strong><small>${escapeHtml(item.description || "")}</small>`;
            button.addEventListener("click", () => onClick(item.id));
            node.appendChild(button);
        });
    }

    function renderTaskDetails(payload) {
        const task = payload.task || {};
        const position = payload.position || {};
        const tools = payload.tools || [];
        const parts = payload.parts || [];
        const documents = payload.documents || [];
        const images = payload.images || [];
        const drawings = payload.drawings || [];

        ids.taskDetail.classList.remove("empty");
        ids.taskDetail.innerHTML = `
            <section class="detail-section">
                <h3>${escapeHtml(task.name || "Task")}</h3>
                <p>${escapeHtml(task.description || task.instructions || "No task description available.")}</p>
            </section>
            <section class="detail-section">
                <h3>Position</h3>
                <p>${escapeHtml(position.name || "No linked position")}</p>
                <p>${escapeHtml(position.description || "")}</p>
            </section>
            <section class="detail-section">
                <h3>Suggested Tools</h3>
                ${tools.length ? `<div class="pill-list">${tools.map((tool) => `<span class="pill">${escapeHtml(tool.name || tool.tool_number || `Tool ${tool.id}`)}</span>`).join("")}</div>` : "<p>No tools found.</p>"}
            </section>
            <section class="detail-section">
                <h3>Parts</h3>
                ${parts.length ? `<div class="pill-list">${parts.map((part) => `<span class="pill">${escapeHtml(part.part_number || part.name || `Part ${part.id}`)}</span>`).join("")}</div>` : "<p>No parts found.</p>"}
            </section>
            <section class="detail-section">
                <h3>Documents</h3>
                ${documents.length ? documents.map((doc) => `<p>${escapeHtml(doc.title || doc.document_number || `Document ${doc.id}`)}</p>`).join("") : "<p>No documents found.</p>"}
            </section>
            <section class="detail-section">
                <h3>Drawings</h3>
                ${drawings.length ? drawings.map((drawing) => `<p>${escapeHtml(drawing.title || drawing.drawing_number || `Drawing ${drawing.id}`)}</p>`).join("") : "<p>No drawings found.</p>"}
            </section>
            <section class="detail-section">
                <h3>Images</h3>
                ${images.length ? `<div class="media-grid">${images.map((image) => `
                    <div class="media-card">
                        <img src="${encodeURI(image.file_url)}" alt="${escapeHtml(image.title || "Task image")}">
                        <strong>${escapeHtml(image.title || `Image ${image.id}`)}</strong>
                    </div>`).join("")}</div>` : "<p>No images found.</p>"}
            </section>
        `;
    }

    async function loadEquipmentGroups() {
        setOptions(ids.equipmentGroup, [], "Select group");
        setOptions(ids.model, [], "Select model");
        setOptions(ids.location, [], "Select location");
        setOptions(ids.assetNumber, [], "Select asset number");
        if (!ids.area.value) {
            return;
        }
        const items = await fetchJson(`/api/options/equipment-groups?area_id=${ids.area.value}`);
        setOptions(ids.equipmentGroup, items, "Select group");
    }

    async function loadModels() {
        setOptions(ids.model, [], "Select model");
        setOptions(ids.location, [], "Select location");
        setOptions(ids.assetNumber, [], "Select asset number");
        if (!ids.equipmentGroup.value) {
            return;
        }
        const items = await fetchJson(`/api/options/models?equipment_group_id=${ids.equipmentGroup.value}`);
        setOptions(ids.model, items, "Select model");
    }

    async function loadLocationAndAssets() {
        setOptions(ids.location, [], "Select location");
        setOptions(ids.assetNumber, [], "Select asset number");
        if (!ids.model.value) {
            return;
        }
        const [locations, assets] = await Promise.all([
            fetchJson(`/api/options/locations?model_id=${ids.model.value}`),
            fetchJson(`/api/options/asset-numbers?model_id=${ids.model.value}`),
        ]);
        setOptions(ids.location, locations, "Select location");
        setOptions(ids.assetNumber, assets, "Select asset number");
    }

    async function loadProblems() {
        const params = new URLSearchParams({
            area_id: ids.area.value,
            equipment_group_id: ids.equipmentGroup.value,
            model_id: ids.model.value,
            location_id: ids.location.value,
            asset_number_id: ids.assetNumber.value,
        });
        const items = await fetchJson(`/api/problems?${params.toString()}`);
        state.selectedProblemId = null;
        state.selectedSolutionId = null;
        state.selectedTaskId = null;
        renderList(ids.problems, items, selectProblem, state.selectedProblemId, "No problems found.");
        clearNode(ids.solutions, "Choose a problem.");
        clearNode(ids.tasks, "Choose a solution.");
        clearNode(ids.taskDetail, "Choose a task.");
    }

    async function selectProblem(problemId) {
        state.selectedProblemId = problemId;
        const items = await fetchJson(`/api/problems/${problemId}/solutions`);
        renderList(ids.solutions, items, selectSolution, state.selectedProblemId, "No solutions found.");
        clearNode(ids.tasks, "Choose a solution.");
        clearNode(ids.taskDetail, "Choose a task.");
    }

    async function selectSolution(solutionId) {
        state.selectedSolutionId = solutionId;
        const items = await fetchJson(`/api/solutions/${solutionId}/tasks`);
        renderList(ids.tasks, items, selectTask, state.selectedTaskId, "No tasks found.");
        clearNode(ids.taskDetail, "Choose a task.");
    }

    async function selectTask(taskId) {
        state.selectedTaskId = taskId;
        const payload = await fetchJson(`/api/tasks/${taskId}`);
        renderTaskDetails(payload);
    }

    function resetAll() {
        ids.area.value = "";
        setOptions(ids.equipmentGroup, [], "Select group");
        setOptions(ids.model, [], "Select model");
        setOptions(ids.location, [], "Select location");
        setOptions(ids.assetNumber, [], "Select asset number");
        clearNode(ids.problems, "Select filters to load problems.");
        clearNode(ids.solutions, "Choose a problem.");
        clearNode(ids.tasks, "Choose a solution.");
        clearNode(ids.taskDetail, "Choose a task.");
    }

    ids.area.addEventListener("change", () => loadEquipmentGroups().catch(console.error));
    ids.equipmentGroup.addEventListener("change", () => loadModels().catch(console.error));
    ids.model.addEventListener("change", () => loadLocationAndAssets().catch(console.error));
    ids.searchBtn.addEventListener("click", () => loadProblems().catch(console.error));
    ids.resetBtn.addEventListener("click", resetAll);
})();
