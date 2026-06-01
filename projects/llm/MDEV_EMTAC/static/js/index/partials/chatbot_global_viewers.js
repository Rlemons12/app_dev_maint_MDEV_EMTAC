// ============================================================
// EMTAC Global Viewers
// File: static/js/index/partials/chatbot_global_viewers.js
//
// WEBVIEW-SAFE DROP-IN REPLACEMENT
//
// Purpose:
// - Keep existing calls working:
//      window.openDocumentViewer(title, text)
//      window.openImageViewer(title, src)
//      window.openDrawingDetails(drawing)
//      window.openPartDetailsViewer(title, images, drawings)
//      window.openImageViewerInPage(title, src)
//      window.openPartDetailsInPage({ part, label, images, drawings, errorMessage })
//
// - Avoid window.open(...) completely.
// - Use fullscreen in-page overlays.
// - Android WebView Back button closes the active viewer.
// ============================================================

console.log("[EMTAC] chatbot_global_viewers.js loaded - WebView-safe in-page mode");

(function () {
    "use strict";

    // ------------------------------------------------------------
    // Shared Safe Helpers
    // ------------------------------------------------------------

    function escapeHtml(value) {
        return String(value || "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function safeText(value, fallback = "") {
        const text = value === null || value === undefined ? fallback : value;
        return String(text);
    }

    function safeUrl(value) {
        if (!value || typeof value !== "string") {
            return "";
        }

        const raw = value.trim();

        if (!raw) {
            return "";
        }

        if (raw.startsWith("/")) {
            return raw;
        }

        if (raw.startsWith("data:image/")) {
            return raw;
        }

        if (raw.startsWith("blob:")) {
            return raw;
        }

        try {
            const parsed = new URL(raw, window.location.origin);

            if (parsed.protocol === "http:" || parsed.protocol === "https:") {
                return parsed.href;
            }

            return "";
        } catch (error) {
            console.warn("[EMTAC] Invalid URL blocked:", raw, error);
            return "";
        }
    }

    function getImageSrc(img) {
        if (!img || typeof img !== "object") {
            return "";
        }

        return safeUrl(
            img.src ||
            img.file_path ||
            img.file_url ||
            img.image_url ||
            img.url ||
            img.href ||
            ""
        );
    }

    function getDrawingNumber(drawing) {
        if (!drawing || typeof drawing !== "object") {
            return "—";
        }

        return (
            drawing.drw_number ||
            drawing.drawing_number ||
            drawing.number ||
            drawing.drawing_no ||
            "—"
        );
    }

    function getDrawingName(drawing) {
        if (!drawing || typeof drawing !== "object") {
            return "";
        }

        return (
            drawing.drw_name ||
            drawing.name ||
            drawing.title ||
            drawing.description ||
            ""
        );
    }

    function getDrawingRevision(drawing) {
        if (!drawing || typeof drawing !== "object") {
            return "";
        }

        return (
            drawing.drw_revision ||
            drawing.revision ||
            drawing.rev ||
            ""
        );
    }

    function normalizeImages(images) {
        if (!Array.isArray(images)) {
            return [];
        }

        return images
            .map((img, index) => {
                if (typeof img === "string") {
                    const src = safeUrl(img);

                    if (!src) {
                        return null;
                    }

                    return {
                        src: src,
                        title: `Image ${index + 1}`,
                        description: ""
                    };
                }

                if (!img || typeof img !== "object") {
                    return null;
                }

                const src = getImageSrc(img);

                if (!src) {
                    return null;
                }

                return {
                    src: src,
                    title: safeText(
                        img.title ||
                        img.name ||
                        img.description ||
                        `Image ${index + 1}`
                    ),
                    description: safeText(
                        img.description ||
                        img.name ||
                        img.title ||
                        ""
                    )
                };
            })
            .filter(Boolean);
    }

    function normalizeDrawings(drawings) {
        if (!Array.isArray(drawings)) {
            return [];
        }

        return drawings
            .map((drawing) => {
                if (typeof drawing === "string") {
                    return {
                        number: drawing,
                        name: "",
                        revision: "",
                        url: "",
                        raw: drawing,
                    };
                }

                if (!drawing || typeof drawing !== "object") {
                    return null;
                }

                return {
                    number: getDrawingNumber(drawing),
                    name: getDrawingName(drawing),
                    revision: getDrawingRevision(drawing),
                    url:
                        drawing.url ||
                        drawing.file_url ||
                        drawing.file_path_url ||
                        drawing.web_url ||
                        drawing.href ||
                        "",
                    raw: drawing,
                };
            })
            .filter(Boolean);
    }

    function normalizeParts(parts) {
        if (!Array.isArray(parts)) {
            return [];
        }

        return parts
            .map((part) => {
                if (!part || typeof part !== "object") {
                    return null;
                }

                return {
                    part_number:
                        part.part_number ||
                        part.partNumber ||
                        part.number ||
                        "—",
                    name:
                        part.name ||
                        part.part_name ||
                        part.description ||
                        "",
                    raw: part,
                };
            })
            .filter(Boolean);
    }

    window.escapeHtml = window.escapeHtml || escapeHtml;

    // ------------------------------------------------------------
    // Style Injection
    // ------------------------------------------------------------

    function ensureGlobalViewerStyles() {
        if (document.getElementById("emtac-global-viewer-styles")) {
            return;
        }

        const style = document.createElement("style");
        style.id = "emtac-global-viewer-styles";

        style.textContent = `
            .emtac-viewer-open {
                overflow: hidden !important;
            }

            .emtac-viewer-overlay {
                position: fixed;
                inset: 0;
                z-index: 999999;
                background: rgba(0, 0, 0, 0.96);
                color: #eee;
                display: flex;
                flex-direction: column;
            }

            .emtac-viewer-panel {
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 0;
                width: 100%;
            }

            .emtac-viewer-header {
                flex: 0 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 12px;
                background: rgba(20, 20, 20, 0.98);
                border-bottom: 2px solid #39FF14;
            }

            .emtac-viewer-header h2 {
                margin: 0;
                color: #39FF14;
                font-size: 18px;
                line-height: 1.3;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .emtac-viewer-buttons {
                display: flex;
                gap: 8px;
                flex: 0 0 auto;
            }

            .emtac-viewer-button {
                cursor: pointer;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                background: #39FF14;
                color: #111;
                font-weight: 700;
            }

            .emtac-viewer-close {
                background: #ff4d4d;
                color: #fff;
            }

            .emtac-viewer-body {
                flex: 1 1 auto;
                min-height: 0;
                overflow: auto;
                padding: 14px;
            }

            .emtac-viewer-body pre {
                margin: 0;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                background: #151515;
                color: #eee;
                padding: 14px;
                border-radius: 8px;
                border: 1px solid rgba(57, 255, 20, 0.35);
                font-family: Consolas, Monaco, "Courier New", monospace;
                font-size: 14px;
                line-height: 1.5;
            }

            .emtac-image-viewer-body {
                flex: 1 1 auto;
                min-height: 0;
                overflow: hidden;
                position: relative;
                background: #000;
                cursor: grab;
                touch-action: none;
            }

            .emtac-image-viewer-img {
                position: absolute;
                top: 50%;
                left: 50%;
                max-width: none;
                max-height: none;
                transform-origin: center center;
                user-select: none;
                pointer-events: none;
            }

            .emtac-entity-tabs {
                display: flex;
                flex: 0 0 auto;
                background: #181818;
                border-bottom: 1px solid #333;
            }

            .emtac-entity-tab {
                padding: 10px 14px;
                cursor: pointer;
                border-right: 1px solid #333;
                color: #ccc;
                user-select: none;
                background: transparent;
                border-top: none;
                border-left: none;
                border-bottom: none;
            }

            .emtac-entity-tab.active {
                background: #222;
                color: #39FF14;
            }

            .emtac-entity-panel {
                display: none;
                flex: 1 1 auto;
                min-height: 0;
                overflow: auto;
                padding: 14px;
            }

            .emtac-entity-panel.active {
                display: block;
            }

            .emtac-entity-item {
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 8px;
                background: #151515;
                margin-bottom: 10px;
            }

            .emtac-entity-item strong {
                color: #39FF14;
            }

            .emtac-muted {
                color: #aaa;
                font-size: 12px;
                margin-top: 3px;
            }

            .emtac-image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                gap: 10px;
            }

            .emtac-image-card {
                display: flex;
                flex-direction: column;
                gap: 6px;
                background: #151515;
                border: 1px solid rgba(57, 255, 20, 0.25);
                border-radius: 8px;
                padding: 8px;
                cursor: pointer;
                color: #eee;
                text-align: left;
            }

            .emtac-image-card:hover {
                border-color: #39FF14;
                background: rgba(57, 255, 20, 0.08);
            }

            .emtac-image-card img {
                width: 100%;
                height: 120px;
                object-fit: contain;
                background: #000;
                border-radius: 6px;
            }

            .emtac-image-card-title {
                font-size: 12px;
                color: #ccc;
                overflow-wrap: anywhere;
            }

            .emtac-detail-grid {
                display: grid;
                grid-template-columns: 160px minmax(0, 1fr);
                gap: 8px 12px;
                padding: 12px;
                border: 1px solid rgba(57, 255, 20, 0.25);
                border-radius: 8px;
                background: #151515;
                margin-bottom: 14px;
            }

            .emtac-detail-label {
                color: #39FF14;
                font-weight: 700;
            }

            .emtac-detail-value {
                color: #eee;
                overflow-wrap: anywhere;
            }

            .emtac-warning {
                margin-bottom: 12px;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(255, 193, 7, 0.5);
                background: rgba(255, 193, 7, 0.12);
                color: #ffe08a;
            }

            @media (max-width: 700px) {
                .emtac-viewer-header {
                    align-items: stretch;
                    flex-direction: column;
                }

                .emtac-viewer-header h2 {
                    white-space: normal;
                }

                .emtac-viewer-buttons {
                    width: 100%;
                }

                .emtac-viewer-button {
                    flex: 1 1 auto;
                }

                .emtac-detail-grid {
                    grid-template-columns: 1fr;
                }

                .emtac-entity-tabs {
                    overflow-x: auto;
                }
            }
        `;

        document.head.appendChild(style);
    }

    // ------------------------------------------------------------
    // Overlay / History Helpers
    // ------------------------------------------------------------

    function removeOverlay(id) {
        const overlay = document.getElementById(id);

        if (overlay) {
            overlay.remove();
        }

        if (!document.querySelector(".emtac-viewer-overlay")) {
            document.body.classList.remove("emtac-viewer-open");
        }
    }

    function pushViewerHistory(stateKey) {
        try {
            const state = {};
            state[stateKey] = true;

            if (!window.history.state || window.history.state[stateKey] !== true) {
                window.history.pushState(
                    state,
                    "",
                    window.location.href
                );
            }
        } catch (err) {
            console.warn("[EMTAC] Could not push viewer history state:", err);
        }
    }

    function closeWithBack(stateKey) {
        try {
            if (window.history.state && window.history.state[stateKey] === true) {
                window.history.back();
            }
        } catch (err) {
            console.warn("[EMTAC] Could not go back from viewer:", err);
        }
    }

    // ============================================================
    // TEXT / DOCUMENT VIEWER
    // ============================================================

    function openTextViewer(title, text, stateKey = "emtacTextViewer") {
        ensureGlobalViewerStyles();
        removeOverlay("emtac-global-text-viewer-overlay");

        const overlay = document.createElement("div");
        overlay.id = "emtac-global-text-viewer-overlay";
        overlay.className = "emtac-viewer-overlay";

        const panel = document.createElement("div");
        panel.className = "emtac-viewer-panel";

        const header = document.createElement("div");
        header.className = "emtac-viewer-header";

        const heading = document.createElement("h2");
        heading.textContent = title || "Document";

        const buttons = document.createElement("div");
        buttons.className = "emtac-viewer-buttons";

        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.className = "emtac-viewer-button";
        copyButton.textContent = "Copy";

        copyButton.onclick = async () => {
            try {
                await navigator.clipboard.writeText(String(text || ""));
                copyButton.textContent = "Copied";
                setTimeout(() => {
                    copyButton.textContent = "Copy";
                }, 1200);
            } catch (err) {
                console.warn("[EMTAC] Clipboard copy failed:", err);
                copyButton.textContent = "Copy Failed";
                setTimeout(() => {
                    copyButton.textContent = "Copy";
                }, 1200);
            }
        };

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.className = "emtac-viewer-button emtac-viewer-close";
        closeButton.textContent = "Close";
        closeButton.onclick = () => {
            removeOverlay("emtac-global-text-viewer-overlay");
            closeWithBack(stateKey);
        };

        buttons.appendChild(copyButton);
        buttons.appendChild(closeButton);

        const body = document.createElement("div");
        body.className = "emtac-viewer-body";

        const pre = document.createElement("pre");
        pre.textContent = text || "";

        body.appendChild(pre);
        header.appendChild(heading);
        header.appendChild(buttons);
        panel.appendChild(header);
        panel.appendChild(body);
        overlay.appendChild(panel);

        document.body.appendChild(overlay);
        document.body.classList.add("emtac-viewer-open");

        pushViewerHistory(stateKey);
    }

    // ============================================================
    // IMAGE VIEWER
    // ============================================================

    function openGlobalImageViewerInPage(title, src) {
        ensureGlobalViewerStyles();

        const safeTitle = safeText(title, "Image");
        const safeSrc = safeUrl(src);

        if (!safeSrc) {
            console.warn("[EMTAC] Image viewer missing usable source:", src);
            return;
        }

        removeOverlay("emtac-global-image-viewer-overlay");

        const overlay = document.createElement("div");
        overlay.id = "emtac-global-image-viewer-overlay";
        overlay.className = "emtac-viewer-overlay";

        const panel = document.createElement("div");
        panel.className = "emtac-viewer-panel";

        const header = document.createElement("div");
        header.className = "emtac-viewer-header";

        const heading = document.createElement("h2");
        heading.textContent = safeTitle;

        const buttons = document.createElement("div");
        buttons.className = "emtac-viewer-buttons";

        const zoomInButton = document.createElement("button");
        zoomInButton.type = "button";
        zoomInButton.className = "emtac-viewer-button";
        zoomInButton.textContent = "+";

        const zoomOutButton = document.createElement("button");
        zoomOutButton.type = "button";
        zoomOutButton.className = "emtac-viewer-button";
        zoomOutButton.textContent = "−";

        const resetButton = document.createElement("button");
        resetButton.type = "button";
        resetButton.className = "emtac-viewer-button";
        resetButton.textContent = "Reset";

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.className = "emtac-viewer-button emtac-viewer-close";
        closeButton.textContent = "Close";

        buttons.appendChild(zoomInButton);
        buttons.appendChild(zoomOutButton);
        buttons.appendChild(resetButton);
        buttons.appendChild(closeButton);

        header.appendChild(heading);
        header.appendChild(buttons);

        const body = document.createElement("div");
        body.className = "emtac-image-viewer-body";

        const image = document.createElement("img");
        image.className = "emtac-image-viewer-img";
        image.src = safeSrc;
        image.alt = safeTitle;

        body.appendChild(image);
        panel.appendChild(header);
        panel.appendChild(body);
        overlay.appendChild(panel);

        document.body.appendChild(overlay);
        document.body.classList.add("emtac-viewer-open");

        let scale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let dragging = false;
        let startX = 0;
        let startY = 0;

        function updateTransform() {
            image.style.transform =
                `translate(-50%, -50%) translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
        }

        function resetView() {
            scale = 1;
            offsetX = 0;
            offsetY = 0;
            updateTransform();
        }

        zoomInButton.onclick = () => {
            scale *= 1.2;
            updateTransform();
        };

        zoomOutButton.onclick = () => {
            scale *= 0.8;
            updateTransform();
        };

        resetButton.onclick = resetView;

        closeButton.onclick = () => {
            removeOverlay("emtac-global-image-viewer-overlay");
            closeWithBack("emtacImageViewer");
        };

        body.addEventListener("mousedown", event => {
            dragging = true;
            body.style.cursor = "grabbing";
            startX = event.clientX - offsetX;
            startY = event.clientY - offsetY;
        });

        body.addEventListener("mousemove", event => {
            if (!dragging) {
                return;
            }

            offsetX = event.clientX - startX;
            offsetY = event.clientY - startY;
            updateTransform();
        });

        window.addEventListener("mouseup", () => {
            dragging = false;
            body.style.cursor = "grab";
        });

        body.addEventListener("wheel", event => {
            event.preventDefault();
            scale *= event.deltaY < 0 ? 1.1 : 0.9;
            updateTransform();
        }, { passive: false });

        body.addEventListener("touchstart", event => {
            if (event.touches.length !== 1) {
                return;
            }

            dragging = true;
            startX = event.touches[0].clientX - offsetX;
            startY = event.touches[0].clientY - offsetY;
        }, { passive: true });

        body.addEventListener("touchmove", event => {
            if (!dragging || event.touches.length !== 1) {
                return;
            }

            offsetX = event.touches[0].clientX - startX;
            offsetY = event.touches[0].clientY - startY;
            updateTransform();
        }, { passive: true });

        body.addEventListener("touchend", () => {
            dragging = false;
        });

        resetView();
        pushViewerHistory("emtacImageViewer");
    }

    // ============================================================
    // ENTITY VIEWER: Drawings / Parts
    // ============================================================

    function openEntityViewerInPage({
        title,
        activeTab,
        drawings,
        images,
        parts,
        part,
        errorMessage
    }) {
        ensureGlobalViewerStyles();
        removeOverlay("emtac-global-entity-viewer-overlay");

        const safeTitle = safeText(title, "Details");
        const normalizedImages = normalizeImages(images);
        const normalizedDrawings = normalizeDrawings(drawings);
        const normalizedParts = normalizeParts(parts);
        const safeActiveTab = activeTab || (
            normalizedImages.length ? "images" :
            normalizedDrawings.length ? "drawings" :
            "details"
        );

        const overlay = document.createElement("div");
        overlay.id = "emtac-global-entity-viewer-overlay";
        overlay.className = "emtac-viewer-overlay";

        const panel = document.createElement("div");
        panel.className = "emtac-viewer-panel";

        const header = document.createElement("div");
        header.className = "emtac-viewer-header";

        const heading = document.createElement("h2");
        heading.textContent = safeTitle;

        const buttons = document.createElement("div");
        buttons.className = "emtac-viewer-buttons";

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.className = "emtac-viewer-button emtac-viewer-close";
        closeButton.textContent = "Close";
        closeButton.onclick = () => {
            removeOverlay("emtac-global-entity-viewer-overlay");
            closeWithBack("emtacEntityViewer");
        };

        buttons.appendChild(closeButton);

        header.appendChild(heading);
        header.appendChild(buttons);

        const tabs = document.createElement("div");
        tabs.className = "emtac-entity-tabs";

        const detailsTab = buildTabButton("details", "Details");
        const imagesTab = buildTabButton("images", `Images (${normalizedImages.length})`);
        const drawingsTab = buildTabButton("drawings", `Drawings (${normalizedDrawings.length})`);
        const partsTab = buildTabButton("parts", `Parts (${normalizedParts.length})`);

        tabs.appendChild(detailsTab);
        tabs.appendChild(imagesTab);
        tabs.appendChild(drawingsTab);
        tabs.appendChild(partsTab);

        const detailsPanel = buildPanel("details");
        const imagesPanel = buildPanel("images");
        const drawingsPanel = buildPanel("drawings");
        const partsPanel = buildPanel("parts");

        if (errorMessage) {
            const warning = document.createElement("div");
            warning.className = "emtac-warning";
            warning.textContent = errorMessage;
            detailsPanel.appendChild(warning);
        }

        detailsPanel.appendChild(buildDetailsGrid(part || { title: safeTitle }, safeTitle));
        renderImagesPanel(imagesPanel, normalizedImages);
        renderDrawingsPanel(drawingsPanel, normalizedDrawings);
        renderPartsPanel(partsPanel, normalizedParts);

        panel.appendChild(header);
        panel.appendChild(tabs);
        panel.appendChild(detailsPanel);
        panel.appendChild(imagesPanel);
        panel.appendChild(drawingsPanel);
        panel.appendChild(partsPanel);
        overlay.appendChild(panel);

        document.body.appendChild(overlay);
        document.body.classList.add("emtac-viewer-open");

        function setActiveTab(tabName) {
            overlay.querySelectorAll(".emtac-entity-tab").forEach(tab => {
                tab.classList.toggle("active", tab.dataset.tab === tabName);
            });

            overlay.querySelectorAll(".emtac-entity-panel").forEach(panelNode => {
                panelNode.classList.toggle("active", panelNode.dataset.panel === tabName);
            });
        }

        function buildTabButton(tabName, label) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "emtac-entity-tab";
            button.dataset.tab = tabName;
            button.textContent = label;
            button.onclick = () => setActiveTab(tabName);
            return button;
        }

        function buildPanel(panelName) {
            const panelNode = document.createElement("div");
            panelNode.className = "emtac-entity-panel";
            panelNode.dataset.panel = panelName;
            return panelNode;
        }

        setActiveTab(safeActiveTab);
        pushViewerHistory("emtacEntityViewer");
    }

    function buildDetailsGrid(data, fallbackTitle) {
        const grid = document.createElement("div");
        grid.className = "emtac-detail-grid";

        addDetailRow(grid, "Title", fallbackTitle || "—");

        if (data && typeof data === "object") {
            addDetailRow(grid, "Name", data.name || data.part_name || data.drw_name || data.title || "—");
            addDetailRow(grid, "Number", data.part_number || data.partNumber || data.drw_number || data.number || "—");
            addDetailRow(grid, "Revision", data.drw_revision || data.revision || data.rev || "—");
            addDetailRow(grid, "Manufacturer", data.manufacturer || data.oem_mfg || data.oem || "—");
            addDetailRow(grid, "Model", data.model || "—");
            addDetailRow(grid, "Description", data.description || data.desc || "—");
        }

        return grid;
    }

    function addDetailRow(grid, label, value) {
        const labelDiv = document.createElement("div");
        labelDiv.className = "emtac-detail-label";
        labelDiv.textContent = label;

        const valueDiv = document.createElement("div");
        valueDiv.className = "emtac-detail-value";
        valueDiv.textContent = value == null || value === "" ? "—" : String(value);

        grid.appendChild(labelDiv);
        grid.appendChild(valueDiv);
    }

    function renderImagesPanel(panel, images) {
        if (!images.length) {
            panel.innerHTML = "<p>No images available.</p>";
            return;
        }

        const grid = document.createElement("div");
        grid.className = "emtac-image-grid";

        images.forEach((imageItem, index) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "emtac-image-card";

            const img = document.createElement("img");
            img.src = imageItem.src;
            img.alt = imageItem.title || `Image ${index + 1}`;
            img.loading = "lazy";

            const caption = document.createElement("div");
            caption.className = "emtac-image-card-title";
            caption.textContent = imageItem.title || imageItem.description || `Image ${index + 1}`;

            card.appendChild(img);
            card.appendChild(caption);

            card.onclick = () => {
                openGlobalImageViewerInPage(
                    imageItem.title || `Image ${index + 1}`,
                    imageItem.src
                );
            };

            grid.appendChild(card);
        });

        panel.appendChild(grid);
    }

    function renderDrawingsPanel(panel, drawings) {
        if (!drawings.length) {
            panel.innerHTML = "<p>No drawings available.</p>";
            return;
        }

        drawings.forEach(drawing => {
            const item = document.createElement("div");
            item.className = "emtac-entity-item";

            const strong = document.createElement("strong");
            strong.textContent = drawing.number || "—";

            item.appendChild(strong);

            if (drawing.name) {
                const name = document.createElement("div");
                name.textContent = drawing.name;
                item.appendChild(name);
            }

            if (drawing.revision) {
                const revision = document.createElement("div");
                revision.className = "emtac-muted";
                revision.textContent = `Rev: ${drawing.revision}`;
                item.appendChild(revision);
            }

            if (drawing.url) {
                const openButton = document.createElement("button");
                openButton.type = "button";
                openButton.className = "emtac-viewer-button";
                openButton.textContent = "Open File";
                openButton.style.marginTop = "8px";
                openButton.onclick = () => {
                    window.location.href = drawing.url;
                };
                item.appendChild(openButton);
            }

            panel.appendChild(item);
        });
    }

    function renderPartsPanel(panel, parts) {
        if (!parts.length) {
            panel.innerHTML = "<p>No parts available.</p>";
            return;
        }

        parts.forEach(part => {
            const item = document.createElement("div");
            item.className = "emtac-entity-item";

            const strong = document.createElement("strong");
            strong.textContent = part.part_number || "—";

            item.appendChild(strong);

            if (part.name) {
                const name = document.createElement("div");
                name.textContent = part.name;
                item.appendChild(name);
            }

            panel.appendChild(item);
        });
    }

    // ============================================================
    // Global Popstate / Escape Handling
    // ============================================================

    window.addEventListener("popstate", () => {
        removeOverlay("emtac-global-text-viewer-overlay");
        removeOverlay("emtac-global-image-viewer-overlay");
        removeOverlay("emtac-global-entity-viewer-overlay");
    });

    document.addEventListener("keydown", event => {
        if (event.key !== "Escape") {
            return;
        }

        const activeOverlay =
            document.getElementById("emtac-global-text-viewer-overlay") ||
            document.getElementById("emtac-global-image-viewer-overlay") ||
            document.getElementById("emtac-global-entity-viewer-overlay");

        if (activeOverlay) {
            event.preventDefault();
            activeOverlay.remove();
            document.body.classList.remove("emtac-viewer-open");
            window.history.back();
        }
    });

    // ============================================================
    // GLOBAL DOCUMENT VIEWER
    // ============================================================

    window.openDocumentViewer = function (title, text) {
        const safeTitle = safeText(title, "Document");
        const safeBody = safeText(text, "");

        if (typeof window.openChunkPopout === "function") {
            window.openChunkPopout(safeTitle, safeBody);
            return;
        }

        openTextViewer(safeTitle, safeBody, "emtacTextViewer");
    };

    // ============================================================
    // GLOBAL IMAGE VIEWER
    // ============================================================

    window.openImageViewerInPage = function (title, src) {
        openGlobalImageViewerInPage(title, src);
    };

    window.openImageViewer = function (title, src) {
        openGlobalImageViewerInPage(title, src);
    };

    // ============================================================
    // GLOBAL DRAWING DETAILS VIEWER
    // ============================================================

    window.openDrawingDetails = function (drawing) {
        if (!drawing || typeof drawing !== "object") {
            console.warn("[EMTAC] openDrawingDetails called without drawing object:", drawing);
            return;
        }

        if (typeof window.openDrawingDetailsInPage === "function") {
            window.openDrawingDetailsInPage(drawing);
            return;
        }

        const title =
            getDrawingNumber(drawing) !== "—"
                ? getDrawingNumber(drawing)
                : getDrawingName(drawing) || "Drawing Details";

        openEntityViewerInPage({
            title: title,
            activeTab: "drawings",
            drawings: [drawing],
            images: drawing.images || drawing.part_images || [],
            parts: drawing.spare_parts || drawing.parts || [],
            part: drawing,
            errorMessage: ""
        });
    };

    // ============================================================
    // GLOBAL PART DETAILS VIEWER
    // ============================================================

    window.openPartDetailsInPage = function ({
        part = {},
        label = "Part Details",
        images = [],
        drawings = [],
        errorMessage = ""
    } = {}) {
        openEntityViewerInPage({
            title: label || "Part Details",
            activeTab: Array.isArray(images) && images.length ? "images" : "details",
            drawings: drawings,
            images: images,
            parts: [],
            part: part,
            errorMessage: errorMessage
        });
    };

    window.openPartDetailsViewer = function (title, images = [], drawings = []) {
        openEntityViewerInPage({
            title: safeText(title, "Part Details"),
            activeTab: Array.isArray(images) && images.length ? "images" : "drawings",
            drawings: drawings,
            images: images,
            parts: [],
            part: {
                name: title,
                part_number: title
            },
            errorMessage: ""
        });
    };

    console.log("[EMTAC] Global viewer functions ready", {
        openDocumentViewer: typeof window.openDocumentViewer,
        openImageViewer: typeof window.openImageViewer,
        openImageViewerInPage: typeof window.openImageViewerInPage,
        openDrawingDetails: typeof window.openDrawingDetails,
        openPartDetailsViewer: typeof window.openPartDetailsViewer,
        openPartDetailsInPage: typeof window.openPartDetailsInPage,
    });
})();