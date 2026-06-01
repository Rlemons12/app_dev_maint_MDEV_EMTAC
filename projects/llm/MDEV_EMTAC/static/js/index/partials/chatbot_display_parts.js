// ============================================================
// Render Parts Panel
// WebView-safe version with fullscreen in-page part viewer
// Name first, Part Number second
// ============================================================

function renderParts(parts) {
    const section = document.getElementById("parts-container");

    if (!section) {
        console.warn("[renderParts] #parts-container not found");
        return;
    }

    section.replaceChildren();
    ensurePartViewerStyles();

    if (!Array.isArray(parts) || parts.length === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No parts found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
        return;
    }

    parts.forEach((part) => {
        const partId = part.id;
        const partNumber = part.part_number || part.partNumber || part.number;
        const name = part.name || part.part_name || "";

        if (!partId || (!partNumber && !name)) {
            console.warn("[renderParts] Invalid part object:", part);
            return;
        }

        const label = buildPartLabel(part);

        const item = document.createElement("div");
        item.className = "document-item part-item";

        const link = document.createElement("button");
        link.type = "button";
        link.className = "document-chunk-link part-link-button";
        link.textContent = label;

        link.addEventListener("click", async (event) => {
            event.preventDefault();
            event.stopPropagation();

            await openPartDetailsFromPart(part, label);
        });

        item.appendChild(link);
        section.appendChild(item);
    });
}


// ============================================================
// Data loading
// ============================================================

async function openPartDetailsFromPart(part, label) {
    const partId = part.id;

    let images = [];
    let drawings = [];
    let errorMessage = "";

    try {
        images = await fetchPartImages(partId);
    } catch (err) {
        console.error("[openPartDetailsFromPart] Failed to fetch images:", err);
        errorMessage += "Failed to fetch part images. ";
    }

    try {
        drawings = await fetchPartDrawings(partId);
    } catch (err) {
        console.error("[openPartDetailsFromPart] Failed to fetch drawings:", err);
        errorMessage += "Failed to fetch part drawings. ";
    }

    openPartDetailsInPage({
        part,
        label,
        images,
        drawings,
        errorMessage: errorMessage.trim()
    });
}

async function fetchPartImages(partId) {
    const response = await fetch(`/parts/${encodeURIComponent(partId)}/images`);

    if (!response.ok) {
        throw new Error(`Failed to fetch part images. Status=${response.status}`);
    }

    const data = await response.json();

    if (!Array.isArray(data.images)) {
        return [];
    }

    return data.images
        .map(normalizeImage)
        .filter(img => img.src);
}

async function fetchPartDrawings(partId) {
    const response = await fetch(`/parts/${encodeURIComponent(partId)}/drawings`);

    if (!response.ok) {
        console.warn("[fetchPartDrawings] No drawings returned for part:", partId);
        return [];
    }

    const data = await response.json();

    if (!Array.isArray(data.drawings)) {
        return [];
    }

    return data.drawings.map(normalizeDrawing);
}

function normalizeImage(img) {
    return {
        src: img.src || img.url || img.image_url || img.file_path || img.file_url || "",
        title: img.title || img.name || img.description || "Part Image",
        description: img.description || img.name || img.title || ""
    };
}

function normalizeDrawing(drawing) {
    return {
        ...drawing,
        title:
            drawing.drw_name ||
            drawing.title ||
            drawing.name ||
            drawing.drawing_name ||
            "Untitled Drawing",
        number:
            drawing.drw_number ||
            drawing.drawing_number ||
            drawing.number ||
            "",
        revision:
            drawing.drw_revision ||
            drawing.revision ||
            "",
        url:
            drawing.url ||
            drawing.file_url ||
            drawing.file_path_url ||
            drawing.web_url ||
            drawing.href ||
            ""
    };
}


// ============================================================
// Label helpers
// ============================================================

function buildPartLabel(part) {
    const partNumber = part.part_number || part.partNumber || part.number || "";
    const name = part.name || part.part_name || "";

    const drawingsInline =
        Array.isArray(part.drawings) && part.drawings.length
            ? ` [${part.drawings
                .map(d =>
                    d.drawing_number ||
                    d.drw_number ||
                    d.name ||
                    d.title ||
                    null
                )
                .filter(Boolean)
                .join(", ")}]`
            : "";

    if (name) {
        return `${name} — ${partNumber}${drawingsInline}`.trim();
    }

    return `${partNumber}${drawingsInline}`.trim();
}


// ============================================================
// Fullscreen In-Page Part Viewer
// ============================================================

function openPartDetailsInPage({ part, label, images, drawings, errorMessage }) {
    ensurePartViewerStyles();
    closePartDetailsInPage(false);

    const overlay = document.createElement("div");
    overlay.id = "emtac-part-viewer-overlay";
    overlay.className = "emtac-part-viewer-overlay";

    const panel = document.createElement("div");
    panel.className = "emtac-part-viewer-panel";

    const header = document.createElement("div");
    header.className = "emtac-part-viewer-header";

    const title = document.createElement("h2");
    title.textContent = label || "Part Details";

    const buttons = document.createElement("div");
    buttons.className = "emtac-part-viewer-buttons";

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "emtac-part-viewer-button emtac-part-viewer-close";
    closeButton.textContent = "Close";
    closeButton.onclick = () => closePartDetailsInPage(true);

    buttons.appendChild(closeButton);

    header.appendChild(title);
    header.appendChild(buttons);

    const body = document.createElement("div");
    body.className = "emtac-part-viewer-body";

    if (errorMessage) {
        const errorBox = document.createElement("div");
        errorBox.className = "emtac-part-viewer-warning";
        errorBox.textContent = errorMessage;
        body.appendChild(errorBox);
    }

    body.appendChild(buildPartDetailGrid(part));
    body.appendChild(buildImagesSection(images));
    body.appendChild(buildDrawingsSection(drawings));

    const rawDetails = document.createElement("details");
    rawDetails.className = "emtac-part-viewer-raw";

    const rawSummary = document.createElement("summary");
    rawSummary.textContent = "Raw Part Data";

    const rawPre = document.createElement("pre");
    rawPre.textContent = JSON.stringify(part, null, 2);

    rawDetails.appendChild(rawSummary);
    rawDetails.appendChild(rawPre);

    body.appendChild(rawDetails);

    panel.appendChild(header);
    panel.appendChild(body);
    overlay.appendChild(panel);

    document.body.appendChild(overlay);
    document.body.classList.add("emtac-part-viewer-open");

    try {
        if (!window.history.state || window.history.state.emtacPartViewer !== true) {
            window.history.pushState(
                { emtacPartViewer: true },
                "",
                window.location.href
            );
        }
    } catch (err) {
        console.warn("[openPartDetailsInPage] Could not push history state:", err);
    }
}

function closePartDetailsInPage(goBack) {
    const imageOverlay = document.getElementById("emtac-part-image-zoom-overlay");
    if (imageOverlay) {
        imageOverlay.remove();
    }

    const overlay = document.getElementById("emtac-part-viewer-overlay");
    if (overlay) {
        overlay.remove();
    }

    document.body.classList.remove("emtac-part-viewer-open");

    if (goBack === true) {
        try {
            if (window.history.state && window.history.state.emtacPartViewer === true) {
                window.history.back();
            }
        } catch (err) {
            console.warn("[closePartDetailsInPage] Could not go back:", err);
        }
    }
}


// ============================================================
// Viewer sections
// ============================================================

function buildPartDetailGrid(part) {
    const grid = document.createElement("div");
    grid.className = "emtac-part-detail-grid";

    addPartDetailRow(grid, "Part ID", part.id || "—");
    addPartDetailRow(grid, "Name", part.name || part.part_name || "—");
    addPartDetailRow(grid, "Part Number", part.part_number || part.partNumber || part.number || "—");
    addPartDetailRow(grid, "Manufacturer", part.manufacturer || part.oem_mfg || part.oem || "—");
    addPartDetailRow(grid, "Model", part.model || "—");
    addPartDetailRow(grid, "Description", part.description || part.desc || "—");

    return grid;
}

function addPartDetailRow(grid, label, value) {
    const labelDiv = document.createElement("div");
    labelDiv.className = "emtac-part-detail-label";
    labelDiv.textContent = label;

    const valueDiv = document.createElement("div");
    valueDiv.className = "emtac-part-detail-value";
    valueDiv.textContent = value == null || value === "" ? "—" : String(value);

    grid.appendChild(labelDiv);
    grid.appendChild(valueDiv);
}

function buildImagesSection(images) {
    const section = document.createElement("div");
    section.className = "emtac-part-viewer-section";

    const title = document.createElement("h3");
    title.textContent = "Images";
    section.appendChild(title);

    if (!Array.isArray(images) || images.length === 0) {
        const empty = document.createElement("p");
        empty.className = "emtac-part-empty";
        empty.textContent = "No images found for this part.";
        section.appendChild(empty);
        return section;
    }

    const grid = document.createElement("div");
    grid.className = "emtac-part-image-grid";

    images.forEach((img, index) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "emtac-part-image-card";

        const image = document.createElement("img");
        image.src = img.src;
        image.alt = img.title || `Part Image ${index + 1}`;
        image.loading = "lazy";

        const caption = document.createElement("div");
        caption.className = "emtac-part-image-caption";
        caption.textContent = img.title || img.description || `Image ${index + 1}`;

        card.appendChild(image);
        card.appendChild(caption);

        card.onclick = () => {
            openPartImageZoom(img);
        };

        grid.appendChild(card);
    });

    section.appendChild(grid);
    return section;
}

function buildDrawingsSection(drawings) {
    const section = document.createElement("div");
    section.className = "emtac-part-viewer-section";

    const title = document.createElement("h3");
    title.textContent = "Drawings";
    section.appendChild(title);

    if (!Array.isArray(drawings) || drawings.length === 0) {
        const empty = document.createElement("p");
        empty.className = "emtac-part-empty";
        empty.textContent = "No drawings found for this part.";
        section.appendChild(empty);
        return section;
    }

    const list = document.createElement("div");
    list.className = "emtac-part-drawing-list";

    drawings.forEach(drawing => {
        const row = document.createElement("div");
        row.className = "emtac-part-drawing-row";

        const info = document.createElement("div");
        info.className = "emtac-part-drawing-info";

        const drawingTitle = document.createElement("div");
        drawingTitle.className = "emtac-part-drawing-title";
        drawingTitle.textContent = drawing.title || "Untitled Drawing";

        const drawingMeta = document.createElement("div");
        drawingMeta.className = "emtac-part-drawing-meta";

        const metaParts = [];

        if (drawing.number) {
            metaParts.push(`No: ${drawing.number}`);
        }

        if (drawing.revision) {
            metaParts.push(`Rev: ${drawing.revision}`);
        }

        drawingMeta.textContent = metaParts.length ? metaParts.join(" | ") : "No drawing metadata.";

        info.appendChild(drawingTitle);
        info.appendChild(drawingMeta);

        row.appendChild(info);

        if (drawing.url) {
            const openButton = document.createElement("button");
            openButton.type = "button";
            openButton.className = "emtac-part-viewer-button";
            openButton.textContent = "Open File";
            openButton.onclick = () => {
                window.location.href = drawing.url;
            };

            row.appendChild(openButton);
        }

        list.appendChild(row);
    });

    section.appendChild(list);
    return section;
}


// ============================================================
// Image zoom overlay
// ============================================================

function openPartImageZoom(img) {
    const existing = document.getElementById("emtac-part-image-zoom-overlay");
    if (existing) {
        existing.remove();
    }

    const overlay = document.createElement("div");
    overlay.id = "emtac-part-image-zoom-overlay";
    overlay.className = "emtac-part-image-zoom-overlay";

    const panel = document.createElement("div");
    panel.className = "emtac-part-image-zoom-panel";

    const header = document.createElement("div");
    header.className = "emtac-part-image-zoom-header";

    const title = document.createElement("h3");
    title.textContent = img.title || img.description || "Part Image";

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "emtac-part-viewer-button emtac-part-viewer-close";
    closeButton.textContent = "Close Image";
    closeButton.onclick = () => overlay.remove();

    header.appendChild(title);
    header.appendChild(closeButton);

    const image = document.createElement("img");
    image.src = img.src;
    image.alt = img.title || "Part image";

    panel.appendChild(header);
    panel.appendChild(image);
    overlay.appendChild(panel);

    document.body.appendChild(overlay);

    overlay.addEventListener("click", event => {
        if (event.target === overlay) {
            overlay.remove();
        }
    });
}


// ============================================================
// Back button / Escape support
// ============================================================

window.addEventListener("popstate", () => {
    const imageOverlay = document.getElementById("emtac-part-image-zoom-overlay");
    if (imageOverlay) {
        imageOverlay.remove();
        return;
    }

    const overlay = document.getElementById("emtac-part-viewer-overlay");
    if (overlay) {
        closePartDetailsInPage(false);
    }
});

document.addEventListener("keydown", event => {
    if (event.key !== "Escape") {
        return;
    }

    const imageOverlay = document.getElementById("emtac-part-image-zoom-overlay");
    if (imageOverlay) {
        event.preventDefault();
        imageOverlay.remove();
        return;
    }

    const overlay = document.getElementById("emtac-part-viewer-overlay");
    if (overlay) {
        event.preventDefault();
        closePartDetailsInPage(true);
    }
});


// ============================================================
// Styles
// ============================================================

function ensurePartViewerStyles() {
    if (document.getElementById("emtac-part-viewer-styles")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "emtac-part-viewer-styles";

    style.textContent = `
        .part-item {
            margin-bottom: 10px;
        }

        .part-link-button {
            display: block;
            width: 100%;
            text-align: left;
            background: transparent;
            border: none;
            color: #4da6ff;
            cursor: pointer;
            text-decoration: underline;
            font-size: 13px;
            line-height: 1.4;
            padding: 6px;
            border-radius: 4px;
        }

        .part-link-button:hover {
            color: #9ccfff;
            background-color: rgba(77, 166, 255, 0.15);
        }

        .emtac-part-viewer-open {
            overflow: hidden;
        }

        .emtac-part-viewer-overlay {
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: rgba(0, 0, 0, 0.96);
            color: #eee;
            display: flex;
            flex-direction: column;
        }

        .emtac-part-viewer-panel {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
            width: 100%;
        }

        .emtac-part-viewer-header {
            flex: 0 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: rgba(20, 20, 20, 0.98);
            border-bottom: 2px solid #39FF14;
        }

        .emtac-part-viewer-header h2 {
            margin: 0;
            color: #39FF14;
            font-size: 18px;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .emtac-part-viewer-buttons {
            display: flex;
            gap: 8px;
            flex: 0 0 auto;
        }

        .emtac-part-viewer-button {
            cursor: pointer;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            background: #39FF14;
            color: #111;
            font-weight: 700;
            flex: 0 0 auto;
        }

        .emtac-part-viewer-close {
            background: #ff4d4d;
            color: #fff;
        }

        .emtac-part-viewer-body {
            flex: 1 1 auto;
            min-height: 0;
            overflow: auto;
            padding: 14px;
        }

        .emtac-part-viewer-warning {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 193, 7, 0.5);
            background: rgba(255, 193, 7, 0.12);
            color: #ffe08a;
        }

        .emtac-part-detail-grid {
            display: grid;
            grid-template-columns: 160px minmax(0, 1fr);
            gap: 8px 12px;
            padding: 12px;
            border: 1px solid rgba(57, 255, 20, 0.25);
            border-radius: 8px;
            background: #151515;
        }

        .emtac-part-detail-label {
            color: #39FF14;
            font-weight: 700;
        }

        .emtac-part-detail-value {
            color: #eee;
            overflow-wrap: anywhere;
        }

        .emtac-part-viewer-section {
            margin-top: 14px;
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
        }

        .emtac-part-viewer-section h3 {
            margin-top: 0;
            color: #39FF14;
        }

        .emtac-part-empty {
            color: #aaa;
            margin-bottom: 0;
        }

        .emtac-part-image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 10px;
        }

        .emtac-part-image-card {
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

        .emtac-part-image-card:hover {
            border-color: #39FF14;
            background: rgba(57, 255, 20, 0.08);
        }

        .emtac-part-image-card img {
            width: 100%;
            height: 110px;
            object-fit: contain;
            background: #000;
            border-radius: 6px;
        }

        .emtac-part-image-caption {
            font-size: 12px;
            color: #ccc;
            overflow-wrap: anywhere;
        }

        .emtac-part-drawing-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .emtac-part-drawing-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            background: #151515;
        }

        .emtac-part-drawing-info {
            min-width: 0;
        }

        .emtac-part-drawing-title {
            color: #eee;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .emtac-part-drawing-meta {
            margin-top: 3px;
            color: #aaa;
            font-size: 12px;
            overflow-wrap: anywhere;
        }

        .emtac-part-viewer-raw {
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            background: #151515;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .emtac-part-viewer-raw summary {
            cursor: pointer;
            color: #39FF14;
            font-weight: 700;
        }

        .emtac-part-viewer-raw pre {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            color: #eee;
        }

        .emtac-part-image-zoom-overlay {
            position: fixed;
            inset: 0;
            z-index: 1000000;
            background: rgba(0, 0, 0, 0.98);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
        }

        .emtac-part-image-zoom-panel {
            width: 100%;
            height: 100%;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }

        .emtac-part-image-zoom-header {
            flex: 0 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 8px;
            border-bottom: 2px solid #39FF14;
        }

        .emtac-part-image-zoom-header h3 {
            color: #39FF14;
            margin: 0;
            overflow-wrap: anywhere;
        }

        .emtac-part-image-zoom-panel img {
            flex: 1 1 auto;
            min-height: 0;
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            background: #000;
        }

        @media (max-width: 700px) {
            .emtac-part-viewer-header,
            .emtac-part-image-zoom-header {
                align-items: stretch;
                flex-direction: column;
            }

            .emtac-part-viewer-header h2 {
                white-space: normal;
            }

            .emtac-part-viewer-buttons {
                width: 100%;
            }

            .emtac-part-viewer-button {
                flex: 1 1 auto;
            }

            .emtac-part-detail-grid {
                grid-template-columns: 1fr;
            }

            .emtac-part-drawing-row {
                align-items: stretch;
                flex-direction: column;
            }
        }
    `;

    document.head.appendChild(style);
}


// ============================================================
// Global exports
// ============================================================

window.renderParts = renderParts;
window.openPartDetailsInPage = openPartDetailsInPage;
window.closePartDetailsInPage = closePartDetailsInPage;