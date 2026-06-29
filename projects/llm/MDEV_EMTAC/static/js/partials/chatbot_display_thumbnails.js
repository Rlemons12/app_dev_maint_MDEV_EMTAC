// ============================================================
// MDEV_EMTAC\static\js\index\partials\chatbot_display_thumbnails.js
// Render Images as DOCUMENT-STYLE LINKS
//
// WebView-safe:
// - Does NOT use window.open()
// - Calls in-page image viewer when available
// - Falls back to window.openImageViewer(), which should also be WebView-safe
//
// Rules:
// - Prefer web-safe image routes: /serve_image/<id>
// - Never prefer raw DB_IMAGES/file_path paths
// - Support array input OR { images: [...] }
// ============================================================

console.log("[EMTAC] chatbot_display_thumbnails.js loaded - WebView-safe image renderer");

function displayThumbnails(imagePanel) {
    const section = document.getElementById("thumbnails-section");

    if (!section) {
        console.warn("[displayThumbnails] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();

    section.classList.add("images-container", "image-results");

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

    images.forEach((imgData, index) => {
        if (!imgData || typeof imgData !== "object") {
            console.warn("[displayThumbnails] Invalid image item:", imgData);
            return;
        }

        const src = resolveImageUrl(imgData);

        if (!src) {
            console.warn("[displayThumbnails] Image missing usable src:", imgData);
            return;
        }

        const title =
            imgData.title ||
            imgData.name ||
            imgData.description ||
            `View Image ${index + 1}`;

        const item = document.createElement("div");
        item.className = "document-item image-document-item";

        const imageLink = document.createElement("button");
        imageLink.type = "button";
        imageLink.className = "document-chunk-link image-document-link";
        imageLink.textContent = title;
        imageLink.title = title;

        imageLink.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();

            openThumbnailImageViewer(title, src);
        });

        item.appendChild(imageLink);
        section.appendChild(item);
    });
}


// ------------------------------------------------------------
// Open image viewer
// ------------------------------------------------------------

function openThumbnailImageViewer(title, src) {
    if (typeof window.openImageViewerInPage === "function") {
        window.openImageViewerInPage(title, src);
        return;
    }

    if (typeof window.openImageViewer === "function") {
        window.openImageViewer(title, src);
        return;
    }

    console.error("[openThumbnailImageViewer] No image viewer function available.", {
        title,
        src
    });
}


// ------------------------------------------------------------
// Image URL Resolver
// ------------------------------------------------------------

function resolveImageUrl(img) {
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
        img.image_url,
        img.file_url,
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

        // Absolute same-host route support.
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
            // Ignore invalid URL values.
        }
    }

    return "";
}


// ------------------------------------------------------------
// Global exports
// ------------------------------------------------------------

window.displayThumbnails = displayThumbnails;
window.resolveImageUrl = resolveImageUrl;
window.openThumbnailImageViewer = openThumbnailImageViewer;