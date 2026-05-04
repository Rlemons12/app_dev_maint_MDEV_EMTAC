// ============================================================
// EMTAC Global Viewers
// File: static/js/index/partials/chatbot_global_viewers.js
//
// Provides shared popup/viewer functions used by:
// - chatbot_display_documents.js
// - chatbot_display_thumbnails.js
// - chatbot_display_drawings.js
// - chatbot_display_parts.js
// ============================================================

console.log("[EMTAC] chatbot_global_viewers.js loaded");

// ------------------------------------------------------------
// Shared Safe Helpers
// ------------------------------------------------------------

(function () {
    "use strict";

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

        // Allow normal app-relative URLs.
        if (raw.startsWith("/")) {
            return raw;
        }

        // Allow browser-safe image/data URLs.
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

    function writePopup(win, html) {
        win.document.open();
        win.document.write(html);
        win.document.close();
    }

    function getImageSrc(img) {
        if (!img || typeof img !== "object") {
            return "";
        }

        return safeUrl(
            img.src ||
            img.file_path ||
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

    function buildBaseStyles() {
        return `
            body {
                margin: 0;
                background: #111;
                color: #eee;
                font-family: Arial, sans-serif;
            }

            header {
                background: #1c1c1c;
                color: #39FF14;
                padding: 10px 14px;
                border-bottom: 1px solid #333;
                font-weight: bold;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .content {
                padding: 14px;
            }

            .tabs {
                display: flex;
                background: #181818;
                border-bottom: 1px solid #333;
            }

            .tab {
                padding: 8px 14px;
                cursor: pointer;
                border-right: 1px solid #333;
                color: #ccc;
                user-select: none;
            }

            .tab.active {
                background: #222;
                color: #39FF14;
            }

            .panel {
                display: none;
                padding: 14px;
                height: calc(100vh - 96px);
                overflow-y: auto;
            }

            .panel.active {
                display: block;
            }

            .item {
                padding: 8px 0;
                border-bottom: 1px solid #333;
            }

            .item strong {
                color: #39FF14;
            }

            .muted {
                color: #aaa;
                font-size: 12px;
            }

            .image-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .image-card {
                border: 1px solid #333;
                background: #181818;
                padding: 8px;
                max-width: 260px;
            }

            .image-card img {
                max-width: 240px;
                max-height: 220px;
                display: block;
                border: 1px solid #333;
                background: #000;
            }

            .image-card-title {
                margin-top: 6px;
                font-size: 12px;
                color: #ccc;
                word-break: break-word;
            }

            a {
                color: #39FF14;
            }
        `;
    }

    function buildTabsScript() {
        return `
            <script>
                document.querySelectorAll(".tab").forEach(function (tab) {
                    tab.onclick = function () {
                        document.querySelectorAll(".tab").forEach(function (t) {
                            t.classList.remove("active");
                        });

                        document.querySelectorAll(".panel").forEach(function (p) {
                            p.classList.remove("active");
                        });

                        tab.classList.add("active");

                        var target = document.getElementById(tab.dataset.tab);
                        if (target) {
                            target.classList.add("active");
                        }
                    };
                });
            </script>
        `;
    }

    // Expose escapeHtml globally for existing renderer compatibility.
    window.escapeHtml = window.escapeHtml || escapeHtml;

    // ============================================================
    // GLOBAL DOCUMENT VIEWER
    // Called by:
    //   window.openDocumentViewer(title, text)
    // ============================================================

    let documentViewerWindow = null;

    window.openDocumentViewer = function (title, text) {
        const safeTitle = safeText(title, "Document");
        const safeBody = safeText(text, "");

        if (documentViewerWindow && !documentViewerWindow.closed) {
            documentViewerWindow.focus();
            documentViewerWindow.postMessage({
                type: "EMTAC_DOCUMENT_VIEWER_UPDATE",
                title: safeTitle,
                text: safeBody,
            }, "*");
            return;
        }

        documentViewerWindow = window.open(
            "",
            "EMTAC_DOCUMENT_VIEWER",
            "width=850,height=700,scrollbars=yes,resizable=yes"
        );

        if (!documentViewerWindow) {
            alert("Pop-up blocked. Please allow pop-ups for this site.");
            return;
        }

        const html = `
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(safeTitle)}</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
        }

        header {
            background: #1c1c1c;
            color: #39FF14;
            padding: 10px 14px;
            border-bottom: 1px solid #333;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        pre {
            padding: 16px;
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
            font-size: 13px;
            background: #111;
            color: #eee;
        }
    </style>
</head>
<body>
    <header id="viewer-title"></header>
    <pre id="viewer-text"></pre>

    <script>
        var titleEl = document.getElementById("viewer-title");
        var textEl = document.getElementById("viewer-text");

        function loadDocument(title, text) {
            titleEl.textContent = title || "Document";
            textEl.textContent = text || "";
            document.title = title || "Document";
        }

        window.addEventListener("message", function (event) {
            if (!event.data || event.data.type !== "EMTAC_DOCUMENT_VIEWER_UPDATE") {
                return;
            }

            loadDocument(event.data.title, event.data.text);
        });

        loadDocument(
            ${JSON.stringify(safeTitle)},
            ${JSON.stringify(safeBody)}
        );
    </script>
</body>
</html>
        `;

        writePopup(documentViewerWindow, html);
    };

    // ============================================================
    // GLOBAL IMAGE VIEWER
    // Called by:
    //   window.openImageViewer(title, src)
    // ============================================================

    let imageViewerWindow = null;

    window.openImageViewer = function (title, src) {
        const safeTitle = safeText(title, "Image");
        const safeSrc = safeUrl(src);

        if (!safeSrc) {
            console.warn("[EMTAC] openImageViewer called without usable src:", src);
            return;
        }

        if (imageViewerWindow && !imageViewerWindow.closed) {
            imageViewerWindow.focus();
            imageViewerWindow.postMessage({
                type: "EMTAC_IMAGE_VIEWER_UPDATE",
                title: safeTitle,
                src: safeSrc,
            }, "*");
            return;
        }

        imageViewerWindow = window.open(
            "",
            "EMTAC_IMAGE_VIEWER",
            "width=1100,height=800,scrollbars=no,resizable=yes"
        );

        if (!imageViewerWindow) {
            alert("Pop-up blocked. Please allow pop-ups for this site.");
            return;
        }

        const html = `
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(safeTitle)}</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }

        header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #1c1c1c;
            border-bottom: 1px solid #333;
        }

        header h2 {
            flex: 1;
            font-size: 14px;
            color: #39FF14;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        button {
            background: #2a2a2a;
            color: #eee;
            border: 1px solid #444;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }

        button:hover {
            background: #333;
        }

        #viewer {
            width: 100%;
            height: calc(100vh - 42px);
            overflow: hidden;
            cursor: grab;
            position: relative;
            background: #000;
        }

        #image {
            position: absolute;
            top: 50%;
            left: 50%;
            transform-origin: center center;
            user-select: none;
            pointer-events: none;
            max-width: none;
            max-height: none;
        }
    </style>
</head>
<body>
    <header>
        <h2 id="viewer-title"></h2>
        <button id="zoom-in">+</button>
        <button id="zoom-out">−</button>
        <button id="reset">Reset</button>
        <button id="open-tab">Open Tab</button>
    </header>

    <div id="viewer">
        <img id="image" alt="" />
    </div>

    <script>
        var scale = 1;
        var offsetX = 0;
        var offsetY = 0;
        var dragging = false;
        var startX = 0;
        var startY = 0;
        var currentSrc = "";

        var img = document.getElementById("image");
        var titleEl = document.getElementById("viewer-title");
        var viewer = document.getElementById("viewer");

        function updateTransform() {
            img.style.transform =
                "translate(-50%, -50%) translate(" +
                offsetX + "px," + offsetY + "px) scale(" + scale + ")";
        }

        function loadImage(title, src) {
            titleEl.textContent = title || "Image";
            document.title = title || "Image";
            currentSrc = src || "";
            img.src = currentSrc;
            scale = 1;
            offsetX = 0;
            offsetY = 0;
            updateTransform();
        }

        function zoom(factor) {
            scale = scale * factor;
            updateTransform();
        }

        function resetView() {
            scale = 1;
            offsetX = 0;
            offsetY = 0;
            updateTransform();
        }

        document.getElementById("zoom-in").onclick = function () {
            zoom(1.2);
        };

        document.getElementById("zoom-out").onclick = function () {
            zoom(0.8);
        };

        document.getElementById("reset").onclick = resetView;

        document.getElementById("open-tab").onclick = function () {
            if (currentSrc) {
                window.open(currentSrc, "_blank");
            }
        };

        viewer.addEventListener("mousedown", function (event) {
            dragging = true;
            viewer.style.cursor = "grabbing";
            startX = event.clientX - offsetX;
            startY = event.clientY - offsetY;
        });

        viewer.addEventListener("mousemove", function (event) {
            if (!dragging) {
                return;
            }

            offsetX = event.clientX - startX;
            offsetY = event.clientY - startY;
            updateTransform();
        });

        window.addEventListener("mouseup", function () {
            dragging = false;
            viewer.style.cursor = "grab";
        });

        viewer.addEventListener("wheel", function (event) {
            event.preventDefault();
            zoom(event.deltaY < 0 ? 1.1 : 0.9);
        }, { passive: false });

        window.addEventListener("message", function (event) {
            if (!event.data || event.data.type !== "EMTAC_IMAGE_VIEWER_UPDATE") {
                return;
            }

            loadImage(event.data.title, event.data.src);
        });

        loadImage(
            ${JSON.stringify(safeTitle)},
            ${JSON.stringify(safeSrc)}
        );
    </script>
</body>
</html>
        `;

        writePopup(imageViewerWindow, html);
    };

    // ============================================================
    // GLOBAL DRAWING DETAILS VIEWER
    // Called by:
    //   window.openDrawingDetails(drawing)
    // ============================================================

    let drawingDetailsWindow = null;

    window.openDrawingDetails = function (drawing) {
        if (!drawing || typeof drawing !== "object") {
            console.warn("[EMTAC] openDrawingDetails called without drawing object:", drawing);
            return;
        }

        const title = safeText(
            getDrawingNumber(drawing) !== "—"
                ? getDrawingNumber(drawing)
                : getDrawingName(drawing),
            "Drawing Details"
        );

        const images = normalizeImages(
            drawing.images ||
            drawing.part_images ||
            []
        );

        const parts = normalizeParts(
            drawing.spare_parts ||
            drawing.parts ||
            []
        );

        const drawings = normalizeDrawings([drawing]);

        if (drawingDetailsWindow && !drawingDetailsWindow.closed) {
            drawingDetailsWindow.focus();
            drawingDetailsWindow.postMessage({
                type: "EMTAC_DRAWING_DETAILS_UPDATE",
                title: title,
                drawings: drawings,
                images: images,
                parts: parts,
            }, "*");
            return;
        }

        drawingDetailsWindow = window.open(
            "",
            "EMTAC_DRAWING_DETAILS_VIEWER",
            "width=1100,height=780,scrollbars=yes,resizable=yes"
        );

        if (!drawingDetailsWindow) {
            alert("Pop-up blocked. Please allow pop-ups for this site.");
            return;
        }

        const html = buildEntityViewerHtml({
            title: title,
            activeTab: "drawings",
            drawings: drawings,
            images: images,
            parts: parts,
            messageType: "EMTAC_DRAWING_DETAILS_UPDATE",
        });

        writePopup(drawingDetailsWindow, html);
    };

    // ============================================================
    // GLOBAL PART DETAILS VIEWER
    // Called by:
    //   window.openPartDetailsViewer(title, images, drawings)
    // ============================================================

    let partDetailsWindow = null;

    window.openPartDetailsViewer = function (title, images = [], drawings = []) {
        const safeTitle = safeText(title, "Part Details");

        const normalizedImages = normalizeImages(images);
        const normalizedDrawings = normalizeDrawings(drawings);

        if (partDetailsWindow && !partDetailsWindow.closed) {
            partDetailsWindow.focus();
            partDetailsWindow.postMessage({
                type: "EMTAC_PART_DETAILS_UPDATE",
                title: safeTitle,
                drawings: normalizedDrawings,
                images: normalizedImages,
                parts: [],
            }, "*");
            return;
        }

        partDetailsWindow = window.open(
            "",
            "EMTAC_PART_DETAILS_VIEWER",
            "width=1100,height=780,scrollbars=yes,resizable=yes"
        );

        if (!partDetailsWindow) {
            alert("Pop-up blocked. Please allow pop-ups for this site.");
            return;
        }

        const html = buildEntityViewerHtml({
            title: safeTitle,
            activeTab: normalizedImages.length ? "images" : "drawings",
            drawings: normalizedDrawings,
            images: normalizedImages,
            parts: [],
            messageType: "EMTAC_PART_DETAILS_UPDATE",
        });

        writePopup(partDetailsWindow, html);
    };

    // ============================================================
    // Shared Entity Viewer HTML Builder
    // ============================================================

    function buildEntityViewerHtml({
        title,
        activeTab,
        drawings,
        images,
        parts,
        messageType,
    }) {
        const normalizedDrawings = normalizeDrawings(drawings);
        const normalizedImages = normalizeImages(images);
        const normalizedParts = normalizeParts(parts);

        const initialPayload = {
            title: safeText(title, "Details"),
            activeTab: activeTab || "drawings",
            drawings: normalizedDrawings,
            images: normalizedImages,
            parts: normalizedParts,
        };

        return `
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(initialPayload.title)}</title>
    <style>
        ${buildBaseStyles()}
    </style>
</head>
<body>
    <header id="viewer-title"></header>

    <div class="tabs">
        <div class="tab" data-tab="drawings">Drawings</div>
        <div class="tab" data-tab="images">Images</div>
        <div class="tab" data-tab="parts">Parts</div>
    </div>

    <div id="drawings" class="panel"></div>
    <div id="images" class="panel"></div>
    <div id="parts" class="panel"></div>

    <script>
        var expectedMessageType = ${JSON.stringify(messageType)};
        var titleEl = document.getElementById("viewer-title");
        var drawingsPanel = document.getElementById("drawings");
        var imagesPanel = document.getElementById("images");
        var partsPanel = document.getElementById("parts");

        function escapeChildHtml(value) {
            return String(value || "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");
        }

        function setActiveTab(tabName) {
            document.querySelectorAll(".tab").forEach(function (tab) {
                tab.classList.remove("active");
            });

            document.querySelectorAll(".panel").forEach(function (panel) {
                panel.classList.remove("active");
            });

            var tab = document.querySelector('.tab[data-tab="' + tabName + '"]');
            var panel = document.getElementById(tabName);

            if (tab && panel) {
                tab.classList.add("active");
                panel.classList.add("active");
            }
        }

        function render(payload) {
            payload = payload || {};

            var title = payload.title || "Details";
            var activeTab = payload.activeTab || "drawings";
            var drawings = Array.isArray(payload.drawings) ? payload.drawings : [];
            var images = Array.isArray(payload.images) ? payload.images : [];
            var parts = Array.isArray(payload.parts) ? payload.parts : [];

            titleEl.textContent = title;
            document.title = title;

            if (drawings.length) {
                drawingsPanel.innerHTML = drawings.map(function (drawing) {
                    var number = escapeChildHtml(drawing.number || "—");
                    var name = escapeChildHtml(drawing.name || "");
                    var revision = escapeChildHtml(drawing.revision || "");

                    return (
                        '<div class="item">' +
                            '<strong>' + number + '</strong>' +
                            (name ? '<div>' + name + '</div>' : '') +
                            (revision ? '<div class="muted">Rev: ' + revision + '</div>' : '') +
                        '</div>'
                    );
                }).join("");
            } else {
                drawingsPanel.innerHTML = "<p>No drawings available.</p>";
            }

            if (images.length) {
                imagesPanel.innerHTML =
                    '<div class="image-grid">' +
                    images.map(function (image) {
                        var src = escapeChildHtml(image.src || "");
                        var imageTitle = escapeChildHtml(image.title || src || "Image");

                        if (!src) {
                            return "";
                        }

                        return (
                            '<div class="image-card">' +
                                '<img src="' + src + '" alt="' + imageTitle + '" />' +
                                '<div class="image-card-title">' + imageTitle + '</div>' +
                            '</div>'
                        );
                    }).join("") +
                    '</div>';
            } else {
                imagesPanel.innerHTML = "<p>No images available.</p>";
            }

            if (parts.length) {
                partsPanel.innerHTML = parts.map(function (part) {
                    var partNumber = escapeChildHtml(part.part_number || "—");
                    var name = escapeChildHtml(part.name || "");

                    return (
                        '<div class="item">' +
                            '<strong>' + partNumber + '</strong>' +
                            (name ? '<div>' + name + '</div>' : '') +
                        '</div>'
                    );
                }).join("");
            } else {
                partsPanel.innerHTML = "<p>No parts available.</p>";
            }

            setActiveTab(activeTab);
        }

        document.querySelectorAll(".tab").forEach(function (tab) {
            tab.onclick = function () {
                setActiveTab(tab.dataset.tab);
            };
        });

        window.addEventListener("message", function (event) {
            if (!event.data || event.data.type !== expectedMessageType) {
                return;
            }

            render(event.data);
        });

        render(${JSON.stringify(initialPayload)});
    </script>
</body>
</html>
        `;
    }

    console.log("[EMTAC] Global viewer functions ready", {
        openDocumentViewer: typeof window.openDocumentViewer,
        openImageViewer: typeof window.openImageViewer,
        openDrawingDetails: typeof window.openDrawingDetails,
        openPartDetailsViewer: typeof window.openPartDetailsViewer,
    });
})();