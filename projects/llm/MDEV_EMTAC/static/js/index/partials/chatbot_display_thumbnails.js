// ============================================================
// MDEV_EMTAC\static\js\index\partials\chatbot_display_thumbnails.js
// Render Images as DOCUMENT-STYLE LINKS
// Clicking link opens GLOBAL image viewer
// ============================================================

function displayThumbnails(imagePanel) {
    const section = document.getElementById("thumbnails-section");

    if (!section) {
        console.warn("[displayThumbnails] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();

    // Normalize input (array OR { images: [...] })
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
        const src = img.file_path || img.src || img.url;
        if (!src) {
            console.warn("[displayThumbnails] Image missing src:", img);
            return;
        }

        const item = document.createElement("div");
        item.className = "document-item";

        const link = document.createElement("a");
        link.href = "javascript:void(0)";
        link.className = "document-chunk-link";
        link.textContent = img.title || `View Image ${index + 1}`;

        link.addEventListener("click", e => {
            e.preventDefault();
            e.stopPropagation();

            if (typeof window.openImageViewer !== "function") {
                console.error("[displayThumbnails] openImageViewer not available");
                return;
            }

            window.openImageViewer(
                img.title || `Image ${index + 1}`,
                src
            );
        });

        item.appendChild(link);
        section.appendChild(item);
    });
}
