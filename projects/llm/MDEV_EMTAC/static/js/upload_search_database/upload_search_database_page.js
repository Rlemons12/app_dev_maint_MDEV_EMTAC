(function () {
    "use strict";

    const UPLOAD_FORM_IDS = new Set([
        "upload-image",
        "upload-document",
        "upload-powerpoint",
        "batch-upload"
    ]);

    const SEARCH_FORM_IDS = new Set([
        "search-documents",
        "search-drawings",
        "search-images",
        "compare-images",
        "search-powerpoints"
    ]);

    function byId(id) {
        return document.getElementById(id);
    }

    function setDisplay(element, value) {
        if (element) {
            element.style.display = value;
        }
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function isMobileView() {
        return window.innerWidth <= 768;
    }

    function isUploadFormId(formId) {
        return UPLOAD_FORM_IDS.has(formId);
    }

    function isSearchFormId(formId) {
        return SEARCH_FORM_IDS.has(formId);
    }

    function isKnownFormId(formId) {
        return isUploadFormId(formId) || isSearchFormId(formId);
    }

    function hideAllForms() {
        document.querySelectorAll(".form-container").forEach(function (form) {
            form.style.display = "none";
        });
    }

    function hideToolSections() {
        setDisplay(byId("upload-tools-section"), "none");
        setDisplay(byId("search-tools-section"), "none");
    }

    function showUploadSection() {
        setDisplay(byId("upload-tools-section"), "block");
        setDisplay(byId("search-tools-section"), "none");
    }

    function showSearchSection() {
        setDisplay(byId("search-tools-section"), "block");
        setDisplay(byId("upload-tools-section"), "none");
    }

    function showSectionForForm(formId) {
        if (isUploadFormId(formId)) {
            showUploadSection();
            return true;
        }

        if (isSearchFormId(formId)) {
            showSearchSection();
            return true;
        }

        hideToolSections();
        return false;
    }

    function hideResults() {
        setDisplay(byId("results"), "none");
    }

    function showResults() {
        setDisplay(byId("results"), "block");
    }

    function clearPreviousResults() {
        const ids = [
            "documents-list",
            "image-results",
            "image-compare-results",
            "drawing-results"
        ];

        ids.forEach(function (id) {
            const el = byId(id);
            if (el) {
                el.innerHTML = "";
            }
        });
    }

    function renderMessage(message, targetId) {
        clearPreviousResults();

        const target = byId(targetId || "documents-list");
        if (!target) {
            return;
        }

        target.innerHTML = `<li class="results-message">${escapeHtml(message)}</li>`;
        showResults();
    }

    function buildQueryString(formElement) {
        const formData = new FormData(formElement);
        return new URLSearchParams(formData).toString();
    }

    async function fetchJson(url) {
        const response = await fetch(url, {
            method: "GET",
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            }
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        return response.json();
    }

    function renderDocumentResults(data) {
        const target = byId("documents-list");
        if (!target) return;

        if (!data.documents || data.documents.length === 0) {
            renderMessage("No documents found.", "documents-list");
            return;
        }

        let html = "";
        data.documents.forEach(function (doc) {
            html += `
                <li>
                    <a href="/search_documents/view_document/${escapeHtml(doc.id)}" target="_blank" rel="noopener noreferrer">
                        ${escapeHtml(doc.title || "Untitled Document")}
                    </a>
                </li>
            `;
        });

        target.innerHTML = html;
        showResults();
    }

    function renderImageResults(data) {
        const target = byId("image-results");
        if (!target) return;

        if (!data.images || data.images.length === 0) {
            renderMessage("No images found.", "image-results");
            return;
        }

        let html = "";
        data.images.forEach(function (image) {
            const id = escapeHtml(image.id);
            const title = escapeHtml(image.title || "Untitled Image");
            const description = escapeHtml(image.description || "");

            html += `
                <li class="image-details">
                    <a href="/serve_image/${id}" target="_blank" rel="noopener noreferrer">
                        <img class="thumbnail" src="/serve_image/${id}" alt="${title}">
                    </a>
                    <div class="description">
                        <h3>${title}</h3>
                        <p>${description}</p>
                    </div>
                </li>
            `;
        });

        target.innerHTML = html;
        showResults();
    }

    function renderPowerpointResults(data) {
        const target = byId("documents-list");
        if (!target) return;

        if (!data.powerpoints || data.powerpoints.length === 0) {
            renderMessage("No PowerPoints found.", "documents-list");
            return;
        }

        let html = "";
        data.powerpoints.forEach(function (ppt) {
            html += `
                <li>
                    <a href="/powerpoints/view/${escapeHtml(ppt.id)}" target="_blank" rel="noopener noreferrer">
                        ${escapeHtml(ppt.title || "Untitled PowerPoint")}
                    </a>
                </li>
            `;
        });

        target.innerHTML = html;
        showResults();
    }

    function renderDrawingResults(data) {
        const target = byId("drawing-results");
        if (!target) return;

        if (!data.results || data.results.length === 0) {
            renderMessage("No drawings found.", "drawing-results");
            return;
        }

        let html = `<h3>Found ${escapeHtml(data.count || data.results.length)} drawing(s)</h3>`;

        data.results.forEach(function (drawing) {
            html += `
                <div class="drawing-result">
                    <h4>${escapeHtml(drawing.drw_name || "Unnamed Drawing")} (${escapeHtml(drawing.drw_number || "No Number")})</h4>
                    <div class="drawing-details">
                        <p><strong>ID:</strong> ${escapeHtml(drawing.id ?? "N/A")}</p>
                        <p><strong>Equipment:</strong> ${escapeHtml(drawing.drw_equipment_name || "N/A")}</p>
                        <p><strong>Revision:</strong> ${escapeHtml(drawing.drw_revision || "N/A")}</p>
                        <p><strong>Spare Part #:</strong> ${escapeHtml(drawing.drw_spare_part_number || "N/A")}</p>
                    </div>
                </div>
            `;
        });

        target.innerHTML = html;
        showResults();
    }

    function bindSearchForm(formId, endpoint, renderer, errorTargetId) {
        const form = byId(formId);
        if (!form) return;

        form.addEventListener("submit", async function (event) {
            event.preventDefault();
            clearPreviousResults();

            try {
                const query = buildQueryString(form);
                const data = await fetchJson(`${endpoint}?${query}`);

                if (data.error) {
                    renderMessage(data.error, errorTargetId);
                    return;
                }

                renderer(data);
            } catch (error) {
                console.error(`[upload_search_database_page] Error submitting ${formId}:`, error);
                renderMessage(`Error occurred: ${error.message}`, errorTargetId);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        console.log("[upload_search_database_page] DOMContentLoaded");

        hideAllForms();
        hideToolSections();
        hideResults();
        clearPreviousResults();

        bindSearchForm("search-documents-form", "/search_documents", renderDocumentResults, "documents-list");
        bindSearchForm("search-images-form", "/search_images", renderImageResults, "image-results");
        bindSearchForm("search-powerpoints-form", "/search_powerpoints", renderPowerpointResults, "documents-list");
        bindSearchForm("drawing-search-form", "/drawings/search", renderDrawingResults, "drawing-results");

        // Override AFTER base_sidebar.js has installed its version.
        window.showForm = function (formId) {
            console.log("[upload_search_database_page] showForm called with:", formId);

            if (!isKnownFormId(formId)) {
                console.warn("[upload_search_database_page] Unknown form id:", formId);
                hideAllForms();
                hideToolSections();
                hideResults();
                clearPreviousResults();
                return;
            }

            hideAllForms();
            hideToolSections();

            const sectionShown = showSectionForForm(formId);
            const formElement = byId(formId);

            if (!sectionShown || !formElement) {
                console.error("[upload_search_database_page] Form container not found:", formId);
                return;
            }

            formElement.style.display = "block";
            clearPreviousResults();
            hideResults();

            if (isMobileView()) {
                formElement.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        };

        console.log("[upload_search_database_page] window.showForm overridden");
    });
})();