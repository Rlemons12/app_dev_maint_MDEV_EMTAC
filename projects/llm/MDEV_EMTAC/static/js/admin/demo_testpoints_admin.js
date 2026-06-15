(function () {
    "use strict";

    const API_BASE = "/admin/demo-testpoints/api";

    function qs(selector, root) {
        return (root || document).querySelector(selector);
    }

    function escapeHtml(value) {
        const div = document.createElement("div");
        div.textContent = value == null ? "" : String(value);
        return div.innerHTML;
    }

    function setMessage(message, isError) {
        const el = qs("#demoTestpointMessage");

        if (!el) {
            return;
        }

        el.textContent = message || "";
        el.style.color = isError ? "#ff6b6b" : "#39FF14";
    }

    function setTableMessage(message, isError) {
        const tbody = qs("#demoTestpointsTableBody");

        if (!tbody) {
            return;
        }

        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="color: ${isError ? "#ff6b6b" : "#eee"};">
                    ${escapeHtml(message)}
                </td>
            </tr>
        `;
    }

    function getPayload() {
        return {
            name: qs("#demoTestpointName") ? qs("#demoTestpointName").value.trim() : "",
            description: qs("#demoTestpointDescription") ? qs("#demoTestpointDescription").value.trim() : "",
            route_path: qs("#demoTestpointRoute") ? qs("#demoTestpointRoute").value.trim() : "",
            category: qs("#demoTestpointCategory") ? qs("#demoTestpointCategory").value.trim() || "General" : "General",
            sort_order: qs("#demoTestpointSortOrder") ? Number(qs("#demoTestpointSortOrder").value || 100) : 100,
            enabled: qs("#demoTestpointEnabled") ? qs("#demoTestpointEnabled").checked : true,
            tablet_visible: qs("#demoTestpointTabletVisible") ? qs("#demoTestpointTabletVisible").checked : true
        };
    }

    async function apiFetch(url, options) {
        const response = await fetch(url, options || {});
        const text = await response.text();

        let data = {};

        try {
            data = text ? JSON.parse(text) : {};
        } catch (error) {
            throw new Error("API did not return JSON. Response: " + text.slice(0, 200));
        }

        if (!response.ok || data.ok === false) {
            throw new Error(data.error || "Request failed with status " + response.status);
        }

        return data;
    }

    async function loadDemoTestpoints() {
        const tbody = qs("#demoTestpointsTableBody");

        if (!tbody) {
            console.warn("Demo Testpoints: table body not found.");
            return;
        }

        setTableMessage("Loading demo testpoints...", false);

        try {
            const data = await apiFetch(API_BASE);
            renderDemoTestpoints(data.testpoints || []);
        } catch (error) {
            console.error("Demo Testpoints load error:", error);
            setTableMessage("Unable to load demo testpoints: " + error.message, true);
        }
    }

    function renderDemoTestpoints(testpoints) {
        const tbody = qs("#demoTestpointsTableBody");

        if (!tbody) {
            return;
        }

        if (!testpoints.length) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6">
                        No demo testpoints configured.
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = testpoints.map(function (tp) {
            const enabledClass = tp.enabled ? "enabled" : "disabled";
            const enabledText = tp.enabled ? "Enabled" : "Disabled";

            const tabletBadge = tp.tablet_visible
                ? `<span class="demo-testpoint-status tablet">Tablet</span>`
                : "";

            return `
                <tr>
                    <td>
                        <strong>${escapeHtml(tp.name)}</strong>
                        <div class="demo-testpoint-description">
                            ${escapeHtml(tp.description || "")}
                        </div>
                    </td>

                    <td>
                        <a href="${escapeHtml(tp.route_path)}"
                           target="_blank"
                           class="demo-testpoint-route">
                            ${escapeHtml(tp.route_path)}
                        </a>
                    </td>

                    <td>${escapeHtml(tp.category || "General")}</td>

                    <td>
                        <span class="demo-testpoint-status ${enabledClass}">
                            ${enabledText}
                        </span>
                        ${tabletBadge}
                    </td>

                    <td>${escapeHtml(tp.sort_order)}</td>

                    <td>
                        <div class="demo-testpoint-actions">
                            <button
                                type="button"
                                class="btn btn-secondary"
                                data-demo-toggle="${escapeHtml(tp.id)}">
                                Toggle
                            </button>

                            <a
                                class="btn"
                                href="${escapeHtml(tp.route_path)}"
                                target="_blank">
                                Open
                            </a>

                            <button
                                type="button"
                                class="btn btn-danger"
                                data-demo-delete="${escapeHtml(tp.id)}">
                                Delete
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join("");

        tbody.querySelectorAll("[data-demo-toggle]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                toggleDemoTestpoint(btn.getAttribute("data-demo-toggle"));
            });
        });

        tbody.querySelectorAll("[data-demo-delete]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                deleteDemoTestpoint(btn.getAttribute("data-demo-delete"));
            });
        });
    }

    async function createDemoTestpoint(event) {
        event.preventDefault();

        const form = qs("#demoTestpointForm");

        if (!form) {
            return;
        }

        const payload = getPayload();

        if (!payload.name || !payload.route_path) {
            setMessage("Name and route path are required.", true);
            return;
        }

        try {
            setMessage("Saving demo testpoint...", false);

            await apiFetch(API_BASE, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            setMessage("Demo testpoint saved.", false);

            form.reset();

            if (qs("#demoTestpointEnabled")) {
                qs("#demoTestpointEnabled").checked = true;
            }

            if (qs("#demoTestpointTabletVisible")) {
                qs("#demoTestpointTabletVisible").checked = true;
            }

            if (qs("#demoTestpointSortOrder")) {
                qs("#demoTestpointSortOrder").value = 100;
            }

            await loadDemoTestpoints();
        } catch (error) {
            console.error("Demo Testpoints save error:", error);
            setMessage(error.message || "Unable to save demo testpoint.", true);
        }
    }

    async function toggleDemoTestpoint(testpointId) {
        if (!testpointId) {
            return;
        }

        try {
            await apiFetch(`${API_BASE}/${encodeURIComponent(testpointId)}/toggle`, {
                method: "POST"
            });

            await loadDemoTestpoints();
        } catch (error) {
            console.error("Demo Testpoints toggle error:", error);
            alert(error.message || "Unable to toggle demo testpoint.");
        }
    }

    async function deleteDemoTestpoint(testpointId) {
        if (!testpointId) {
            return;
        }

        if (!confirm("Delete this demo testpoint?")) {
            return;
        }

        try {
            await apiFetch(`${API_BASE}/${encodeURIComponent(testpointId)}`, {
                method: "DELETE"
            });

            await loadDemoTestpoints();
        } catch (error) {
            console.error("Demo Testpoints delete error:", error);
            alert(error.message || "Unable to delete demo testpoint.");
        }
    }

    function initDemoTestpointsAdmin() {
        const section = qs("#demo-testpoints");

        if (!section) {
            return;
        }

        const form = qs("#demoTestpointForm");

        if (form) {
            form.addEventListener("submit", createDemoTestpoint);
        }

        loadDemoTestpoints();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initDemoTestpointsAdmin);
    } else {
        initDemoTestpointsAdmin();
    }
})();