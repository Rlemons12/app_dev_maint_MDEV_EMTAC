// ============================================================
// Render Documents with INLINE toggle + POPOUT window option
// ============================================================

function displayDocuments(docs) {
    const section = document.getElementById("doc-links-section");

    if (!section) {
        console.warn("[displayDocuments] #doc-links-section not found");
        return;
    }

    section.replaceChildren();

    if (!Array.isArray(docs) || docs.length === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No documents found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
        return;
    }

    docs.forEach(doc => {
        if (!Array.isArray(doc.chunks) || doc.chunks.length === 0) return;

        doc.chunks.forEach((ch, index) => {
            if (!ch?.text) return;

            const wrapper = document.createElement("div");
            wrapper.className = "document-item";

            /* ---------- HEADER ROW ---------- */
            const header = document.createElement("div");
            header.style.display = "flex";
            header.style.justifyContent = "space-between";
            header.style.alignItems = "center";
            header.style.gap = "8px";

            /* ---------- INLINE TOGGLE LINK ---------- */
            const toggleLink = document.createElement("div");
            toggleLink.className = "document-chunk-link";
            toggleLink.textContent =
                doc.title ||
                `Document ${doc.complete_document_id || ""} – Chunk ${index + 1}`;

            /* ---------- POPOUT LINK ---------- */
            const popoutLink = document.createElement("span");
            popoutLink.textContent = "Pop out";
            popoutLink.style.cursor = "pointer";
            popoutLink.style.fontSize = "12px";
            popoutLink.style.color = "#39FF14";
            popoutLink.style.textDecoration = "underline";

            /* ---------- HIDDEN CONTENT ---------- */
            const text = document.createElement("div");
            text.className = "document-chunk-text";
            text.textContent = ch.text.trim();

            /* ---------- INLINE TOGGLE ---------- */
            toggleLink.addEventListener("click", () => {
                const open = text.style.display === "block";
                text.style.display = open ? "none" : "block";
            });

            /* ---------- POPOUT HANDLER ---------- */
            popoutLink.addEventListener("click", (e) => {
                e.stopPropagation();
                window.openDocumentViewer(
                    toggleLink.textContent,
                    ch.text
                );
            });


            header.appendChild(toggleLink);
            header.appendChild(popoutLink);

            wrapper.appendChild(header);
            wrapper.appendChild(text);
            section.appendChild(wrapper);
        });
    });
}


// ============================================================
// Pop-out Chunk Window
// ============================================================

function openChunkPopout(title, text) {
    const win = window.open(
        "",
        "_blank",
        "width=700,height=600,scrollbars=yes,resizable=yes"
    );

    if (!win) {
        alert("Pop-up blocked. Please allow pop-ups for this site.");
        return;
    }

    win.document.write(`
        <html>
        <head>
            <title>${title}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #111;
                    color: #eee;
                    padding: 20px;
                    line-height: 1.5;
                }
                h2 {
                    color: #39FF14;
                    margin-top: 0;
                }
                pre {
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    background: #1c1c1c;
                    padding: 15px;
                    border-radius: 6px;
                    border: 1px solid #333;
                }
            </style>
        </head>
        <body>
            <h2>${title}</h2>
            <pre>${escapeHtml(text)}</pre>
        </body>
        </html>
    `);

    win.document.close();
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
