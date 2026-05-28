(function () {
    "use strict";

    console.log("[upload_search_database_page] loaded - WebView-safe search links v6");

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
            const element = byId(id);

            if (element) {
                element.innerHTML = "";
            }
        });
    }

    function renderMessage(message, targetId) {
        clearPreviousResults();

        const target = byId(targetId || "documents-list");

        if (!target) {
            return;
        }

        const item = document.createElement("li");
        item.className = "results-message";
        item.textContent = message;

        target.appendChild(item);
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

    function createOpenButton(label, onClick) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "emtac-search-open-button";
        button.textContent = label || "Open";

        button.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();
            onClick();
        });

        return button;
    }

    function createSameWindowLink(label, url) {
    const link = document.createElement("a");
    link.className = "emtac-search-open-button emtac-search-open-link";
    link.textContent = label || "Open";
    link.href = url || "#";

    /*
     * Android WebView rule:
     * Never allow document/search result links to open a popup tab/window.
     */
    link.removeAttribute("target");
    link.removeAttribute("rel");

    link.setAttribute("target", "_self");
    link.setAttribute("data-emtac-same-window-link", "true");

    link.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (!url) {
            console.warn("[upload_search_database_page] Missing URL for link:", label);
            return;
        }

        let absoluteUrl = url;

        try {
            absoluteUrl = new URL(url, window.location.origin).href;
        } catch (err) {
            console.warn(
                "[upload_search_database_page] Could not normalize URL, using raw URL:",
                url,
                err
            );
        }

        console.log(
            "[upload_search_database_page] Opening same-window link:",
            absoluteUrl
        );

        /*
         * Force same-window navigation.
         * This avoids Android WebView popup handling and keeps EMTAC session/cookies.
         */
        try {
            window.location.assign(absoluteUrl);
        } catch (err) {
            window.location.href = absoluteUrl;
        }
    });

    return link;
}

    function ensureSearchResultStyles() {
        if (document.getElementById("emtac-search-result-styles")) {
            return;
        }

        const style = document.createElement("style");
        style.id = "emtac-search-result-styles";

        style.textContent = `
            .emtac-search-result-item {
                list-style: none;
                margin: 0 0 10px 0;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                background: rgba(0, 0, 0, 0.16);
            }

            .emtac-search-result-title {
                color: #eee;
                font-weight: 700;
                margin-bottom: 6px;
                overflow-wrap: anywhere;
            }

            .emtac-search-result-meta {
                color: #bbb;
                font-size: 12px;
                margin: 3px 0 8px 0;
                overflow-wrap: anywhere;
            }

            .emtac-search-open-button {
                display: inline-block;
                cursor: pointer;
                border: none;
                border-radius: 6px;
                padding: 7px 10px;
                margin-top: 6px;
                background: #39FF14;
                color: #111 !important;
                font-weight: 700;
                text-decoration: none !important;
                line-height: 1.2;
            }

            .emtac-search-open-button:hover {
                filter: brightness(1.08);
                text-decoration: none !important;
            }

            .emtac-search-open-link {
                text-align: center;
            }

            .emtac-search-image-card {
                display: grid;
                grid-template-columns: 120px minmax(0, 1fr);
                gap: 10px;
                align-items: start;
            }

            .emtac-search-image-button {
                cursor: pointer;
                border: 1px solid rgba(57, 255, 20, 0.35);
                border-radius: 8px;
                background: #000;
                padding: 6px;
            }

            .emtac-search-image-button img {
                display: block;
                width: 100%;
                height: 100px;
                object-fit: contain;
                background: #000;
            }

            .drawing-result {
                cursor: default;
            }

            @media (max-width: 700px) {
                .emtac-search-image-card {
                    grid-template-columns: 1fr;
                }

                .emtac-search-image-button img {
                    height: 180px;
                }

                .emtac-search-open-button {
                    width: 100%;
                }
            }
        `;

        document.head.appendChild(style);
    }

    function renderDocumentResults(data) {
        ensureSearchResultStyles();

        const target = byId("documents-list");

        if (!target) {
            return;
        }

        target.innerHTML = "";

        if (!data.documents || data.documents.length === 0) {
            renderMessage("No documents found.", "documents-list");
            return;
        }

        const fragment = document.createDocumentFragment();

        data.documents.forEach(function (doc) {
            const item = document.createElement("li");
            item.className = "emtac-search-result-item";

            const titleText = doc.title || "Untitled Document";
            const docId = doc.id;

            const title = document.createElement("div");
            title.className = "emtac-search-result-title";
            title.textContent = titleText;

            const meta = document.createElement("div");
            meta.className = "emtac-search-result-meta";
            meta.textContent = `Document ID: ${docId ?? "N/A"}`;

            const url = `/search_documents/view_document/${encodeURIComponent(docId)}`;

            /*
             * Use a real anchor for documents.
             * This avoids popup blocking, iframe blank-page issues, and JS navigation issues.
             */
            const openLink = createSameWindowLink("Open Document", url);

            item.appendChild(title);
            item.appendChild(meta);
            item.appendChild(openLink);
            fragment.appendChild(item);
        });

        target.appendChild(fragment);
        showResults();
    }

    function renderImageResults(data) {
        ensureSearchResultStyles();

        const target = byId("image-results");

        if (!target) {
            return;
        }

        target.innerHTML = "";

        if (!data.images || data.images.length === 0) {
            renderMessage("No images found.", "image-results");
            return;
        }

        const fragment = document.createDocumentFragment();

        data.images.forEach(function (image) {
            const id = image.id;
            const titleText = image.title || "Untitled Image";
            const descriptionText = image.description || "";
            const imageUrl = `/serve_image/${encodeURIComponent(id)}`;

            const item = document.createElement("li");
            item.className = "emtac-search-result-item image-details";

            const card = document.createElement("div");
            card.className = "emtac-search-image-card";

            const imageButton = document.createElement("button");
            imageButton.type = "button";
            imageButton.className = "emtac-search-image-button";
            imageButton.title = titleText;

            const img = document.createElement("img");
            img.className = "thumbnail";
            img.src = imageUrl;
            img.alt = titleText;
            img.loading = "lazy";

            imageButton.appendChild(img);

            imageButton.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();

                if (typeof window.openImageViewerInPage === "function") {
                    window.openImageViewerInPage(titleText, imageUrl);
                    return;
                }

                if (typeof window.openImageViewer === "function") {
                    window.openImageViewer(titleText, imageUrl);
                    return;
                }

                window.location.href = imageUrl;
            });

            const description = document.createElement("div");
            description.className = "description";

            const title = document.createElement("h3");
            title.textContent = titleText;

            const paragraph = document.createElement("p");
            paragraph.textContent = descriptionText;

            const openButton = createOpenButton("Open Image", function () {
                if (typeof window.openImageViewerInPage === "function") {
                    window.openImageViewerInPage(titleText, imageUrl);
                    return;
                }

                if (typeof window.openImageViewer === "function") {
                    window.openImageViewer(titleText, imageUrl);
                    return;
                }

                window.location.href = imageUrl;
            });

            description.appendChild(title);
            description.appendChild(paragraph);
            description.appendChild(openButton);

            card.appendChild(imageButton);
            card.appendChild(description);

            item.appendChild(card);
            fragment.appendChild(item);
        });

        target.appendChild(fragment);
        showResults();
    }

    function renderPowerpointResults(data) {
        ensureSearchResultStyles();

        const target = byId("documents-list");

        if (!target) {
            return;
        }

        target.innerHTML = "";

        if (!data.powerpoints || data.powerpoints.length === 0) {
            renderMessage("No PowerPoints found.", "documents-list");
            return;
        }

        const fragment = document.createDocumentFragment();

        data.powerpoints.forEach(function (ppt) {
            const item = document.createElement("li");
            item.className = "emtac-search-result-item";

            const titleText = ppt.title || "Untitled PowerPoint";
            const pptId = ppt.id;

            const title = document.createElement("div");
            title.className = "emtac-search-result-title";
            title.textContent = titleText;

            const meta = document.createElement("div");
            meta.className = "emtac-search-result-meta";
            meta.textContent = `PowerPoint ID: ${pptId ?? "N/A"}`;

            const url = `/powerpoints/view/${encodeURIComponent(pptId)}`;
            const openLink = createSameWindowLink("Open PowerPoint", url);

            item.appendChild(title);
            item.appendChild(meta);
            item.appendChild(openLink);
            fragment.appendChild(item);
        });

        target.appendChild(fragment);
        showResults();
    }

    function hasUsableDrawingFile(drawing) {
    const filePath = String(drawing?.file_path || "").trim().toLowerCase();

    if (!filePath) {
        return false;
    }

    const placeholderValues = new Set([
        "active_drawing_list_import",
        "none",
        "null",
        "n/a",
        "na"
    ]);

    if (placeholderValues.has(filePath)) {
        return false;
    }

    /*
     * Optional backend support:
     * If later the search payload includes file_exists: false,
     * this front end will respect it.
     */
    if (drawing.file_exists === false) {
        return false;
    }

    return true;
}

function createDisabledDrawingButton(label, reason) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "emtac-search-open-button emtac-search-open-button-disabled";
    button.textContent = label || "Unavailable";
    button.disabled = true;

    if (reason) {
        button.title = reason;
    }

    return button;
}

    function renderDrawingResults(data) {
    ensureSearchResultStyles();

    const target = byId("drawing-results");

    if (!target) {
        return;
    }

    target.innerHTML = "";

    if (!data.results || data.results.length === 0) {
        renderMessage("No drawings found.", "drawing-results");
        return;
    }

    const heading = document.createElement("h3");
    heading.textContent = `Found ${data.count || data.results.length} drawing(s)`;
    target.appendChild(heading);

    const fragment = document.createDocumentFragment();

    data.results.forEach(function (drawing) {
        const wrapper = document.createElement("div");
        wrapper.className = "drawing-result emtac-search-result-item";

        const drawingId = drawing.id;
        const hasFile = hasUsableDrawingFile(drawing);

        const title = document.createElement("h4");
        title.textContent = `${drawing.drw_name || "Unnamed Drawing"} (${drawing.drw_number || "No Number"})`;

        const details = document.createElement("div");
        details.className = "drawing-details";

        details.innerHTML = `
            <p><strong>ID:</strong> ${escapeHtml(drawing.id ?? "N/A")}</p>
            <p><strong>Equipment:</strong> ${escapeHtml(drawing.drw_equipment_name || "N/A")}</p>
            <p><strong>Revision:</strong> ${escapeHtml(drawing.drw_revision || "N/A")}</p>
            <p><strong>Spare Part #:</strong> ${escapeHtml(drawing.drw_spare_part_number || "N/A")}</p>
            <p><strong>File Path:</strong> ${escapeHtml(drawing.file_path || "N/A")}</p>
        `;

        const actions = document.createElement("div");
        actions.className = "emtac-search-result-actions";

        if (
            hasFile &&
            drawingId !== null &&
            drawingId !== undefined &&
            String(drawingId).trim() !== ""
        ) {
            const viewerUrl = `/drawings/print-viewer/${encodeURIComponent(drawingId)}`;
            const openViewerLink = createSameWindowLink("View Print", viewerUrl);
            actions.appendChild(openViewerLink);
        } else {
            const reason = "No drawing file is linked yet. The file_path is missing or still set to active_drawing_list_import.";
            const openDisabled = createDisabledDrawingButton("View Print", reason);

            const status = document.createElement("div");
            status.className = "emtac-search-file-status";
            status.textContent = "No print file linked yet. Run the drawing file sync or update file_path.";

            actions.appendChild(openDisabled);
            actions.appendChild(status);
        }

        wrapper.appendChild(title);
        wrapper.appendChild(details);
        wrapper.appendChild(actions);
        fragment.appendChild(wrapper);
    });

    target.appendChild(fragment);
    showResults();
}

    function bindSearchForm(formId, endpoint, renderer, errorTargetId) {
        const form = byId(formId);

        if (!form) {
            return;
        }

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
                formElement.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        };

        console.log("[upload_search_database_page] window.showForm overridden");
    });
})();