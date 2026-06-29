// ============================================================
// Render Documents + TOGGLEABLE Chunk Links (NO AUTO TEXT)
// DROP-IN REPLACEMENT
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
        const docWrapper = document.createElement("div");
        docWrapper.className = "document-item";

        // ------------------------------
        // Document Title
        // ------------------------------
        const title = document.createElement("div");
        title.className = "document-title";
        title.textContent =
            doc.title ||
            `Document #${doc.complete_document_id || doc.document_id || ""}`;

        docWrapper.appendChild(title);

        // ------------------------------
        // Chunks (LINK → TOGGLE TEXT)
        // ------------------------------
        if (Array.isArray(doc.chunks) && doc.chunks.length > 0) {
            const chunksDiv = document.createElement("div");
            chunksDiv.className = "document-chunks";

            doc.chunks.forEach((ch, index) => {
                if (!ch || !ch.text) return;

                // ---- LINK ----
                const chunkLink = document.createElement("a");
                chunkLink.href = "javascript:void(0)";
                chunkLink.className = "document-chunk-link";
                chunkLink.textContent = `View Chunk ${index + 1}`;

                // ---- HIDDEN TEXT ----
                const chunkText = document.createElement("div");
                chunkText.className = "document-chunk-text";
                chunkText.textContent = ch.text.trim();
                chunkText.style.display = "none";

                // Metadata (future-proof)
                chunkLink.dataset.chunkId = ch.chunk_id ?? "";
                chunkLink.dataset.documentId =
                    doc.complete_document_id ?? "";
                chunkLink.dataset.score = ch.score ?? "";

                // ---- TOGGLE BEHAVIOR ----
                chunkLink.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    const isOpen = chunkText.style.display === "block";

                    chunkText.style.display = isOpen ? "none" : "block";
                    chunkLink.textContent = isOpen
                        ? `View Chunk ${index + 1}`
                        : `Hide Chunk ${index + 1}`;

                    console.log("[DOC CHUNK TOGGLE]", {
                        document_id: doc.complete_document_id,
                        chunk_id: ch.chunk_id,
                        open: !isOpen
                    });
                });

                chunksDiv.appendChild(chunkLink);
                chunksDiv.appendChild(chunkText);
            });

            docWrapper.appendChild(chunksDiv);
        }

        section.appendChild(docWrapper);
    });
}
