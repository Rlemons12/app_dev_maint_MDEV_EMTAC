// ============================================================
// MDEV_EMTAC\static\js\index\partials\chatbot_display_thumbnails.js
// Render Images as DOCUMENT-STYLE LINKS
// Clicking link opens GLOBAL image viewer
//
// Rules:
// - Prefer web-safe image routes: /serve_image/<id>
// - Never prefer raw DB_IMAGES/file_path paths
// - Support array input OR { images: [...] }
// - Force section into 2-column image-link layout via class hooks
// ============================================================

function displayThumbnails(imagePanel) {
    const section = document.getElementById("thumbnails-section");

    if (!section) {
        console.warn("[displayThumbnails] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();

    // Add layout hooks so CSS can force 2 columns.
    section.classList.add("images-container", "image-results");

    // Normalize input:
    // 1. { images: [...] }
    // 2. [...]
    // 3. anything else -> []
    const images = Array.isArray(imagePanel?.images)
        ? imagePanel.images
        : Array.isArray(imagePanel)
            ? imagePanel
            : [];

    if (images.length === 0) {
        const empty = document.createElement("p");
        empty.textContent = "No images found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
        return;
    }

    images.forEach((img, index) => {
        if (!img || typeof img !== "object") {
            console.warn("[displayThumbnails] Invalid image item:", img);
            return;
        }

        const src = resolveImageUrl(img);

        if (!src) {
            console.warn("[displayThumbnails] Image missing usable src:", img);
            return;
        }

        const item = document.createElement("div");
        item.className = "document-item image-document-item";

        const link = document.createElement("a");
        link.href = "javascript:void(0)";
        link.className = "document-chunk-link image-document-link";
        link.textContent = img.title || img.name || `View Image ${index + 1}`;
        link.title = link.textContent;

        link.addEventListener("click", e => {
            e.preventDefault();
            e.stopPropagation();

            if (typeof window.openImageViewer !== "function") {
                console.error("[displayThumbnails] openImageViewer not available");
                return;
            }

            window.openImageViewer(
                img.title || img.name || `Image ${index + 1}`,
                src
            );
        });

        item.appendChild(link);
        section.appendChild(item);
    });
}


// ------------------------------------------------------------
// Image URL Resolver
// ------------------------------------------------------------

function resolveImageUrl(img) {
    /*
        Safe priority order:

        1. /serve_image/<id> built from img.id
        2. img.src if already web-safe
        3. img.url if already web-safe
        4. img.href if already web-safe
        5. convert old /images/<id> to /serve_image/<id>
        6. only use file_path if it is already /serve_image/<id>

        Important:
        We intentionally do NOT prefer raw DB_IMAGES paths.
    */

    if (!img || typeof img !== "object") {
        return "";
    }

    // Best route: build from image ID.
    if (img.id !== undefined && img.id !== null && String(img.id).trim() !== "") {
        const numericId = Number(img.id);

        if (Number.isInteger(numericId) && numericId > 0) {
            return `/serve_image/${numericId}`;
        }
    }

    const candidates = [
        img.src,
        img.url,
        img.href,
        img.file_path
    ];

    for (const candidate of candidates) {
        if (typeof candidate !== "string") {
            continue;
        }

        const value = candidate.trim();

        if (!value) {
            continue;
        }

        // Already correct.
        if (value.startsWith("/serve_image/")) {
            return value;
        }

        // Convert older route convention.
        if (value.startsWith("/images/")) {
            const maybeId = value.split("/").pop();

            if (/^\d+$/.test(maybeId)) {
                return `/serve_image/${maybeId}`;
            }
        }

        // Absolute same-host serve route.
        try {
            const parsed = new URL(value, window.location.origin);

            if (parsed.pathname.startsWith("/serve_image/")) {
                return parsed.pathname + parsed.search;
            }

            if (parsed.pathname.startsWith("/images/")) {
                const maybeId = parsed.pathname.split("/").pop();

                if (/^\d+$/.test(maybeId)) {
                    return `/serve_image/${maybeId}`;
                }
            }
        } catch (err) {
            // Ignore invalid URL values and keep checking.
        }
    }

    return "";
}