// ============================================================
// Render Documents with INLINE toggle + FULLSCREEN IN-PAGE VIEWER
// Document-scoped conversation mode frontend support
// Tablet/WebView safe version
// ============================================================

console.log("[EMTAC] chatbot_display_documents.js DOCUMENT_SCOPE_2026_05_28_1 loaded");


// ============================================================
// Document Scope State
// ============================================================

const EMTAC_DOCUMENT_SCOPE_STORAGE_KEY = "emtac_active_document_scope";

let activeDocumentScope = restoreDocumentScopeFromSessionStorage();
syncDocumentScopeGlobals(activeDocumentScope);


// ============================================================
// Render Documents
// ============================================================

function displayDocuments(docs) {
    const section = document.getElementById("doc-links-section");

    if (!section) {
        console.warn("[displayDocuments] #doc-links-section not found");
        return;
    }

    section.replaceChildren();

    ensureChunkViewerStyles();
    ensureDocumentModeBanner();
    renderActiveDocumentModeBanner();

    if (!Array.isArray(docs) || docs.length === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No documents found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
        return;
    }

    let renderedDocumentCount = 0;
    let renderedChunkCount = 0;

    docs.forEach((doc, docIndex) => {
        inspectDocumentPayloadObject(doc, docIndex);

        if (!Array.isArray(doc.chunks) || doc.chunks.length === 0) {
            return;
        }

        const documentTitle = getDocumentDisplayName(doc, docIndex);
        const documentScope = buildDocumentScopeFromPayloadDocument(doc, documentTitle);

        const documentWrapper = document.createElement("div");
        documentWrapper.className = "document-item document-scope-item";

        // ---------- DOCUMENT TOOLBAR ----------
        const documentToolbar = document.createElement("div");
        documentToolbar.className = "document-scope-toolbar";

        const documentTitleEl = document.createElement("div");
        documentTitleEl.className = "document-scope-title";
        documentTitleEl.textContent = documentTitle;

        const askDocumentButton = document.createElement("button");
        askDocumentButton.type = "button";
        askDocumentButton.className = "document-scope-button";
        askDocumentButton.textContent = "Ask this document";

        if (!documentScope) {
            askDocumentButton.disabled = true;
            askDocumentButton.title = "Document mode unavailable because complete_document_id is missing.";
            askDocumentButton.classList.add("document-scope-button-disabled");
        } else {
            askDocumentButton.title = `Ask follow-up questions only about ${documentScope.document_name}`;

            askDocumentButton.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();

                const activatedScope = setActiveDocumentScope(documentScope);

                if (activatedScope) {
                    console.info("[EMTAC DOCUMENT MODE] Enabled:", activatedScope);
                }
            });
        }

        documentToolbar.appendChild(documentTitleEl);
        documentToolbar.appendChild(askDocumentButton);
        documentWrapper.appendChild(documentToolbar);

        // ---------- CHUNKS ----------
        doc.chunks.forEach((ch, chunkIndex) => {
            if (!ch || !ch.text) {
                return;
            }

            const chunkTitle =
                ch.title ||
                ch.chunk_title ||
                `${documentTitle} – Chunk ${chunkIndex + 1}`;

            const chunkWrapper = document.createElement("div");
            chunkWrapper.className = "document-chunk-item";

            // ---------- HEADER ROW ----------
            const header = document.createElement("div");
            header.className = "document-header-row";

            // ---------- INLINE TOGGLE LINK ----------
            const toggleLink = document.createElement("button");
            toggleLink.type = "button";
            toggleLink.className = "document-chunk-link";
            toggleLink.textContent = chunkTitle;

            // ---------- VIEW FULLSCREEN BUTTON ----------
            const popoutLink = document.createElement("button");
            popoutLink.type = "button";
            popoutLink.className = "document-popout-link";
            popoutLink.textContent = "Open";

            // ---------- HIDDEN INLINE CONTENT ----------
            const text = document.createElement("div");
            text.className = "document-chunk-text";
            text.textContent = String(ch.text || "").trim();
            text.style.display = "none";

            // ---------- INLINE TOGGLE ----------
            toggleLink.addEventListener("click", () => {
                const open = text.style.display === "block";
                text.style.display = open ? "none" : "block";
            });

            // ---------- FULLSCREEN IN-PAGE VIEWER ----------
            popoutLink.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();

                openChunkPopout(chunkTitle, ch.text);
            });

            header.appendChild(toggleLink);
            header.appendChild(popoutLink);

            chunkWrapper.appendChild(header);
            chunkWrapper.appendChild(text);

            documentWrapper.appendChild(chunkWrapper);
            renderedChunkCount += 1;
        });

        if (documentWrapper.childElementCount > 1) {
            section.appendChild(documentWrapper);
            renderedDocumentCount += 1;
        }
    });

    if (renderedDocumentCount === 0 || renderedChunkCount === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No document chunks found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
    }

    console.log("[EMTAC DOCUMENT MODE] Documents rendered:", {
        renderedDocumentCount,
        renderedChunkCount,
        activeDocumentScope: getActiveDocumentScope(),
    });
}


// ============================================================
// Document Scope Helpers
// ============================================================

function inspectDocumentPayloadObject(documentItem, index) {
    try {
        console.group(`[EMTAC DOCUMENT MODE INSPECT] Document ${index}`);

        console.log("Raw document object:", documentItem);

        console.table({
            id: documentItem?.id ?? null,
            document_id: documentItem?.document_id ?? null,
            documentId: documentItem?.documentId ?? null,
            complete_document_id: documentItem?.complete_document_id ?? null,
            completed_document_id: documentItem?.completed_document_id ?? null,
            completeDocumentId: documentItem?.completeDocumentId ?? null,
            completeDocumentID: documentItem?.completeDocumentID ?? null,
            document_name: documentItem?.document_name ?? null,
            name: documentItem?.name ?? null,
            title: documentItem?.title ?? null,
            file_name: documentItem?.file_name ?? null,
            file_path: documentItem?.file_path ?? null,
            url: documentItem?.url ?? null,
            file_url: documentItem?.file_url ?? null,
            display_url: documentItem?.display_url ?? null,
            source_table: documentItem?.source_table ?? null,
            source_id: documentItem?.source_id ?? null,
            chunks_count: Array.isArray(documentItem?.chunks) ? documentItem.chunks.length : 0,
            first_chunk_id: Array.isArray(documentItem?.chunks) && documentItem.chunks[0]
                ? documentItem.chunks[0].id ?? documentItem.chunks[0].chunk_id ?? null
                : null,
            first_chunk_complete_document_id: Array.isArray(documentItem?.chunks) && documentItem.chunks[0]
                ? documentItem.chunks[0].complete_document_id ?? null
                : null,
        });

        console.groupEnd();
    } catch (error) {
        console.warn("[EMTAC DOCUMENT MODE INSPECT] Failed to inspect document object.", error);
    }
}


function buildDocumentScopeFromPayloadDocument(documentItem, fallbackName) {
    if (!documentItem || typeof documentItem !== "object") {
        return null;
    }

    const firstChunk = Array.isArray(documentItem.chunks) && documentItem.chunks.length > 0
        ? documentItem.chunks[0]
        : null;

    const completeDocumentId = firstNonEmptyValue(
        documentItem.complete_document_id,
        documentItem.completed_document_id,
        documentItem.completeDocumentId,
        documentItem.completeDocumentID,
        documentItem.complete_document?.id,
        documentItem.completed_document?.id,
        documentItem.document?.complete_document_id,
        firstChunk?.complete_document_id,
        firstChunk?.completed_document_id,
        firstChunk?.completeDocumentId
    );

    const documentId = firstNonEmptyValue(
        documentItem.document_id,
        documentItem.documentId,
        documentItem.document?.id,
        firstChunk?.document_id,
        firstChunk?.documentId,
        firstChunk?.document?.id,
        documentItem.id
    );

    const documentName = firstNonEmptyValue(
        documentItem.document_name,
        documentItem.name,
        documentItem.title,
        documentItem.file_name,
        documentItem.filename,
        documentItem.original_filename,
        documentItem.file_path ? getFileNameFromPath(documentItem.file_path) : null,
        firstChunk?.document_name,
        firstChunk?.title,
        fallbackName,
        "Selected Document"
    );

    if (!completeDocumentId) {
        console.warn(
            "[EMTAC DOCUMENT MODE] Cannot build document_scope because complete_document_id is missing.",
            documentItem
        );

        return null;
    }

    return normalizeDocumentScope({
        enabled: true,
        scope_type: "complete_document",
        document_id: documentId || null,
        complete_document_id: completeDocumentId,
        document_name: documentName || "Selected Document",
    });
}


function normalizeDocumentScope(scope) {
    if (!scope || typeof scope !== "object") {
        return null;
    }

    const enabled = scope.enabled !== false;

    const scopeType = String(
        scope.scope_type ||
        scope.scopeType ||
        "complete_document"
    ).trim() || "complete_document";

    const completeDocumentId = firstNonEmptyValue(
        scope.complete_document_id,
        scope.completed_document_id,
        scope.completeDocumentId,
        scope.completeDocumentID
    );

    if (!enabled || scopeType !== "complete_document" || !completeDocumentId) {
        return null;
    }

    const documentId = firstNonEmptyValue(
        scope.document_id,
        scope.documentId,
        null
    );

    const documentName = String(
        firstNonEmptyValue(
            scope.document_name,
            scope.documentName,
            scope.name,
            scope.title,
            "Selected Document"
        )
    ).trim() || "Selected Document";

    return {
        enabled: true,
        scope_type: "complete_document",
        document_id: documentId || null,
        complete_document_id: completeDocumentId,
        document_name: documentName,
    };
}


function getDocumentDisplayName(documentItem, index) {
    if (!documentItem || typeof documentItem !== "object") {
        return `Document ${index + 1}`;
    }

    const firstChunk = Array.isArray(documentItem.chunks) && documentItem.chunks.length > 0
        ? documentItem.chunks[0]
        : null;

    const name = firstNonEmptyValue(
        documentItem.document_name,
        documentItem.name,
        documentItem.title,
        documentItem.file_name,
        documentItem.filename,
        documentItem.original_filename,
        documentItem.file_path ? getFileNameFromPath(documentItem.file_path) : null,
        firstChunk?.document_name,
        firstChunk?.title
    );

    if (name) {
        return String(name).trim();
    }

    const completeDocumentId = firstNonEmptyValue(
        documentItem.complete_document_id,
        documentItem.completed_document_id,
        documentItem.completeDocumentId,
        firstChunk?.complete_document_id
    );

    if (completeDocumentId) {
        return `Document ${completeDocumentId}`;
    }

    return `Document ${index + 1}`;
}


function firstNonEmptyValue(...values) {
    for (const value of values) {
        if (value === null || value === undefined) {
            continue;
        }

        if (typeof value === "string") {
            const trimmed = value.trim();

            if (trimmed) {
                return trimmed;
            }

            continue;
        }

        if (typeof value === "number") {
            if (Number.isFinite(value)) {
                return value;
            }

            continue;
        }

        if (typeof value === "boolean") {
            return value;
        }

        return value;
    }

    return null;
}


function getFileNameFromPath(filePath) {
    const value = String(filePath || "").trim();

    if (!value) {
        return "";
    }

    const normalized = value.replaceAll("\\", "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || value;
}


// ============================================================
// Document Scope Browser Storage
// ============================================================

function getActiveDocumentScope() {
    if (activeDocumentScope) {
        return activeDocumentScope;
    }

    const restoredScope = restoreDocumentScopeFromSessionStorage();

    if (restoredScope) {
        activeDocumentScope = restoredScope;
        syncDocumentScopeGlobals(activeDocumentScope);
        return activeDocumentScope;
    }

    return null;
}


function setActiveDocumentScope(scope) {
    const normalizedScope = normalizeDocumentScope(scope);

    if (!normalizedScope) {
        console.warn("[EMTAC DOCUMENT MODE] Invalid document scope. Cannot enable document mode.", scope);
        return null;
    }

    activeDocumentScope = normalizedScope;
    persistDocumentScopeToSessionStorage(activeDocumentScope);
    syncDocumentScopeGlobals(activeDocumentScope);
    renderActiveDocumentModeBanner();

    dispatchDocumentScopeChangedEvent(activeDocumentScope);

    return activeDocumentScope;
}


function clearActiveDocumentScope() {
    activeDocumentScope = null;
    persistDocumentScopeToSessionStorage(null);
    syncDocumentScopeGlobals(null);
    renderActiveDocumentModeBanner();

    dispatchDocumentScopeChangedEvent(null);

    console.info("[EMTAC DOCUMENT MODE] Cleared document scope.");
}


function restoreDocumentScopeFromSessionStorage() {
    try {
        const rawScope = sessionStorage.getItem(EMTAC_DOCUMENT_SCOPE_STORAGE_KEY);

        if (!rawScope) {
            return null;
        }

        const parsedScope = JSON.parse(rawScope);
        return normalizeDocumentScope(parsedScope);

    } catch (error) {
        console.debug("[EMTAC DOCUMENT MODE] Unable to restore document scope from sessionStorage.", error);
        return null;
    }
}


function persistDocumentScopeToSessionStorage(scope) {
    try {
        const normalizedScope = normalizeDocumentScope(scope);

        if (normalizedScope) {
            sessionStorage.setItem(
                EMTAC_DOCUMENT_SCOPE_STORAGE_KEY,
                JSON.stringify(normalizedScope)
            );
            return;
        }

        sessionStorage.removeItem(EMTAC_DOCUMENT_SCOPE_STORAGE_KEY);

    } catch (error) {
        console.debug("[EMTAC DOCUMENT MODE] Unable to persist document scope to sessionStorage.", error);
    }
}


function syncDocumentScopeGlobals(scope) {
    const normalizedScope = normalizeDocumentScope(scope);

    window.EMTAC_ACTIVE_DOCUMENT_SCOPE = normalizedScope;
    window.activeDocumentScope = normalizedScope;
    window.currentDocumentScope = normalizedScope;
}


function dispatchDocumentScopeChangedEvent(scope) {
    try {
        document.dispatchEvent(new CustomEvent("emtac:document-scope-changed", {
            detail: {
                document_scope: scope || null,
            },
        }));
    } catch (error) {
        console.debug("[EMTAC DOCUMENT MODE] Unable to dispatch document scope event.", error);
    }
}


// ============================================================
// Document Mode Banner
// ============================================================

function ensureDocumentModeBanner() {
    let banner = document.getElementById("emtac-document-mode-banner");

    if (banner) {
        return banner;
    }

    const section = document.getElementById("doc-links-section");

    banner = document.createElement("div");
    banner.id = "emtac-document-mode-banner";
    banner.className = "emtac-document-mode-banner";
    banner.style.display = "none";

    if (section && section.parentNode) {
        section.parentNode.insertBefore(banner, section);
    } else if (document.body) {
        document.body.prepend(banner);
    }

    return banner;
}


function renderActiveDocumentModeBanner() {
    const banner = ensureDocumentModeBanner();

    if (!banner) {
        return;
    }

    const scope = getActiveDocumentScope();

    banner.replaceChildren();

    if (!scope) {
        banner.style.display = "none";
        banner.classList.remove("emtac-document-mode-banner-active");
        return;
    }

    banner.style.display = "flex";
    banner.classList.add("emtac-document-mode-banner-active");

    const label = document.createElement("span");
    label.className = "emtac-document-mode-label";
    label.textContent = `Document Mode: ${scope.document_name}`;

    const meta = document.createElement("span");
    meta.className = "emtac-document-mode-meta";
    meta.textContent = `complete_document_id=${scope.complete_document_id}`;

    const exitButton = document.createElement("button");
    exitButton.type = "button";
    exitButton.className = "emtac-document-mode-exit";
    exitButton.textContent = "Exit Document Mode";

    exitButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();

        clearActiveDocumentScope();
    });

    banner.appendChild(label);
    banner.appendChild(meta);
    banner.appendChild(exitButton);
}


// ============================================================
// Fullscreen In-Page Chunk Viewer
// This avoids WebView popup/new-tab problems.
// Android Back button will close it because we push history state.
// ============================================================

function openChunkPopout(title, text) {
    ensureChunkViewerStyles();

    closeChunkPopout(false);

    const overlay = document.createElement("div");
    overlay.id = "emtac-doc-popout-overlay";
    overlay.className = "emtac-doc-popout-overlay";

    const panel = document.createElement("div");
    panel.className = "emtac-doc-popout-panel";

    const header = document.createElement("div");
    header.className = "emtac-doc-popout-header";

    const heading = document.createElement("h2");
    heading.textContent = title || "Document";

    const buttonRow = document.createElement("div");
    buttonRow.className = "emtac-doc-popout-buttons";

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.textContent = "Copy";
    copyButton.className = "emtac-doc-popout-button";

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.textContent = "Close";
    closeButton.className = "emtac-doc-popout-button emtac-doc-popout-close";

    const body = document.createElement("div");
    body.className = "emtac-doc-popout-body";

    const pre = document.createElement("pre");
    pre.textContent = String(text || "").trim();

    body.appendChild(pre);

    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(String(text || "").trim());
            copyButton.textContent = "Copied";

            setTimeout(() => {
                copyButton.textContent = "Copy";
            }, 1200);

        } catch (err) {
            console.warn("[openChunkPopout] Clipboard copy failed:", err);
            copyButton.textContent = "Copy failed";

            setTimeout(() => {
                copyButton.textContent = "Copy";
            }, 1200);
        }
    });

    closeButton.addEventListener("click", () => {
        closeChunkPopout(true);
    });

    buttonRow.appendChild(copyButton);
    buttonRow.appendChild(closeButton);

    header.appendChild(heading);
    header.appendChild(buttonRow);

    panel.appendChild(header);
    panel.appendChild(body);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    document.body.classList.add("emtac-doc-popout-open");

    // Add a browser/WebView history entry so Android Back closes the viewer.
    try {
        if (!window.history.state || window.history.state.emtacDocViewer !== true) {
            window.history.pushState(
                { emtacDocViewer: true },
                "",
                window.location.href
            );
        }
    } catch (err) {
        console.warn("[openChunkPopout] Could not push history state:", err);
    }
}


function closeChunkPopout(goBack) {
    const overlay = document.getElementById("emtac-doc-popout-overlay");

    if (overlay) {
        overlay.remove();
    }

    document.body.classList.remove("emtac-doc-popout-open");

    if (goBack === true) {
        try {
            if (window.history.state && window.history.state.emtacDocViewer === true) {
                window.history.back();
            }
        } catch (err) {
            console.warn("[closeChunkPopout] Could not go back:", err);
        }
    }
}


// ============================================================
// Back button / Escape support
// ============================================================

window.addEventListener("popstate", () => {
    const overlay = document.getElementById("emtac-doc-popout-overlay");

    if (overlay) {
        closeChunkPopout(false);
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        const overlay = document.getElementById("emtac-doc-popout-overlay");

        if (overlay) {
            event.preventDefault();
            closeChunkPopout(true);
        }
    }
});


// ============================================================
// Styles
// ============================================================

function ensureChunkViewerStyles() {
    if (document.getElementById("emtac-doc-popout-style")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "emtac-doc-popout-style";

    style.textContent = `
        .document-scope-item {
            border: 1px solid rgba(77, 166, 255, 0.25);
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.025);
        }

        .document-scope-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            padding: 6px 6px 8px 6px;
            margin-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .document-scope-title {
            flex: 1 1 auto;
            color: #e8f2ff;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.4;
            overflow-wrap: anywhere;
        }

        .document-scope-button {
            flex: 0 0 auto;
            cursor: pointer;
            font-size: 12px;
            color: #111;
            background: #ffd54f;
            border: none;
            border-radius: 5px;
            padding: 5px 8px;
            font-weight: 800;
        }

        .document-scope-button:hover {
            filter: brightness(1.08);
        }

        .document-scope-button-disabled,
        .document-scope-button:disabled {
            cursor: not-allowed;
            opacity: 0.45;
            filter: grayscale(0.8);
        }

        .document-chunk-item {
            margin-top: 4px;
        }

        .document-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }

        .document-chunk-link {
            flex: 1 1 auto;
            color: #4da6ff;
            background: transparent;
            border: none;
            cursor: pointer;
            text-decoration: underline;
            font-size: 13px;
            line-height: 1.4;
            padding: 6px;
            border-radius: 4px;
            text-align: left;
            transition: background-color 0.15s ease, color 0.15s ease;
        }

        .document-chunk-link:hover {
            color: #9ccfff;
            background-color: rgba(77, 166, 255, 0.15);
        }

        .document-popout-link {
            flex: 0 0 auto;
            cursor: pointer;
            font-size: 12px;
            color: #111;
            background: #39FF14;
            border: none;
            border-radius: 5px;
            padding: 5px 8px;
            font-weight: 700;
        }

        .document-popout-link:hover {
            filter: brightness(1.08);
        }

        .document-chunk-text {
            margin: 6px 0 10px 0;
            padding: 10px;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(77, 166, 255, 0.2);
            color: #ddd;
            font-size: 13px;
            line-height: 1.45;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .emtac-document-mode-banner {
            display: none;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            margin: 0 0 10px 0;
            border-radius: 8px;
            border: 1px solid rgba(255, 213, 79, 0.75);
            background: rgba(255, 213, 79, 0.12);
            color: #fff8d8;
            font-size: 13px;
            line-height: 1.35;
        }

        .emtac-document-mode-banner-active {
            display: flex;
        }

        .emtac-document-mode-label {
            flex: 1 1 auto;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .emtac-document-mode-meta {
            flex: 0 1 auto;
            font-size: 11px;
            opacity: 0.8;
            white-space: nowrap;
        }

        .emtac-document-mode-exit {
            flex: 0 0 auto;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            padding: 5px 8px;
            background: #ff4d4d;
            color: #fff;
            font-size: 12px;
            font-weight: 800;
        }

        .emtac-document-mode-exit:hover {
            filter: brightness(1.08);
        }

        .emtac-doc-popout-open {
            overflow: hidden;
        }

        .emtac-doc-popout-overlay {
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: rgba(0, 0, 0, 0.96);
            color: #eee;
            display: flex;
            flex-direction: column;
        }

        .emtac-doc-popout-panel {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
            width: 100%;
        }

        .emtac-doc-popout-header {
            flex: 0 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: rgba(20, 20, 20, 0.98);
            border-bottom: 2px solid #39FF14;
        }

        .emtac-doc-popout-header h2 {
            margin: 0;
            color: #39FF14;
            font-size: 18px;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .emtac-doc-popout-buttons {
            display: flex;
            gap: 8px;
            flex: 0 0 auto;
        }

        .emtac-doc-popout-button {
            cursor: pointer;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            background: #39FF14;
            color: #111;
            font-weight: 700;
        }

        .emtac-doc-popout-close {
            background: #ff4d4d;
            color: #fff;
        }

        .emtac-doc-popout-body {
            flex: 1 1 auto;
            min-height: 0;
            overflow: auto;
            padding: 14px;
        }

        .emtac-doc-popout-body pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
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

        @media (max-width: 700px) {
            .document-scope-toolbar {
                align-items: stretch;
                flex-direction: column;
            }

            .document-scope-button {
                width: 100%;
            }

            .emtac-document-mode-banner {
                align-items: stretch;
                flex-direction: column;
            }

            .emtac-document-mode-meta {
                white-space: normal;
            }

            .emtac-document-mode-exit {
                width: 100%;
            }

            .emtac-doc-popout-header {
                align-items: stretch;
                flex-direction: column;
            }

            .emtac-doc-popout-header h2 {
                white-space: normal;
            }

            .emtac-doc-popout-buttons {
                width: 100%;
            }

            .emtac-doc-popout-button {
                flex: 1 1 auto;
            }
        }
    `;

    document.head.appendChild(style);
}


// ============================================================
// Small helper kept for compatibility
// ============================================================

function escapeHtml(str) {
    return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}


// ============================================================
// DOM Ready
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    ensureChunkViewerStyles();
    ensureDocumentModeBanner();
    renderActiveDocumentModeBanner();

    console.log("[EMTAC DOCUMENT MODE] Initialized:", {
        activeDocumentScope: getActiveDocumentScope(),
    });
});


// ============================================================
// Expose functions globally for existing EMTAC scripts
// ============================================================

window.displayDocuments = displayDocuments;
window.openChunkPopout = openChunkPopout;
window.closeChunkPopout = closeChunkPopout;

window.inspectDocumentPayloadObject = inspectDocumentPayloadObject;
window.buildDocumentScopeFromPayloadDocument = buildDocumentScopeFromPayloadDocument;

window.getEMTACActiveDocumentScope = getActiveDocumentScope;
window.setEMTACActiveDocumentScope = setActiveDocumentScope;
window.clearEMTACActiveDocumentScope = clearActiveDocumentScope;

window.EMTAC_DOCUMENT_SCOPE_STORAGE_KEY = EMTAC_DOCUMENT_SCOPE_STORAGE_KEY;