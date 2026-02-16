// ============================================================
// Render Parts Panel
// Name first, Part Number second
// Clicking opens combined Images + Drawings popout
// ============================================================

function renderParts(parts) {
    const section = document.getElementById("parts-container");

    if (!section) {
        console.warn("[renderParts] #parts-container not found");
        return;
    }

    section.replaceChildren();

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

        // --------------------------------------------------
        // Inline drawing numbers (BEST EFFORT – may be empty)
        // --------------------------------------------------
        const drawingsInline =
            Array.isArray(part.drawings) && part.drawings.length
                ? ` [${part.drawings
                      .map(d =>
                          d.drawing_number ||
                          d.drw_number ||
                          d.name ||
                          null
                      )
                      .filter(Boolean)
                      .join(", ")}]`
                : "";

        const label = name
            ? `${name} — ${partNumber ?? ""}${drawingsInline}`.trim()
            : `${partNumber ?? ""}${drawingsInline}`.trim();

        const item = document.createElement("div");
        item.className = "document-item";

        const link = document.createElement("a");
        link.href = "javascript:void(0)";
        link.className = "document-chunk-link";
        link.textContent = label;

        // --------------------------------------------------
        // CLICK → OPEN PART DETAILS (Images + Drawings)
        // --------------------------------------------------
        link.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();

            try {
                // -----------------------------
                // Fetch images
                // -----------------------------
                const imgRes = await fetch(`/parts/${partId}/images`);
                if (!imgRes.ok) {
                    throw new Error("Failed to fetch part images");
                }

                const imgData = await imgRes.json();

                const images = Array.isArray(imgData.images)
                    ? imgData.images
                          .map(img => ({
                              src: img.src || img.file_path || img.url,
                              title: img.title || img.name || ""
                          }))
                          .filter(img => img.src)
                    : [];

                // -----------------------------
                // Fetch drawings (ON DEMAND)
                // -----------------------------
                let drawings = [];

                try {
                    const drwRes = await fetch(`/parts/${partId}/drawings`);
                    if (drwRes.ok) {
                        const drwData = await drwRes.json();
                        drawings = Array.isArray(drwData.drawings)
                            ? drwData.drawings
                            : [];
                    } else {
                        console.warn(
                            "[renderParts] No drawings returned for part:",
                            partId
                        );
                    }
                } catch (err) {
                    console.error(
                        "[renderParts] Failed to fetch drawings:",
                        err
                    );
                }

                console.log("[renderParts] images:", images);
                console.log("[renderParts] drawings:", drawings);

                // -----------------------------
                // Open viewer
                // -----------------------------
                if (images.length || drawings.length) {
                    window.openPartDetailsViewer(label, images, drawings);
                    return;
                }

                console.warn(
                    "[renderParts] No images or drawings for part:",
                    partId
                );

            } catch (err) {
                console.error("[renderParts] Error:", err);
            }
        });

        item.appendChild(link);
        section.appendChild(item);
    });
}
