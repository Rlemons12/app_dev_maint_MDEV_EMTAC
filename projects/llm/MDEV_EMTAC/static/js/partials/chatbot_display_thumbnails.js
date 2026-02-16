// ============================================================
// Render Images as DOCUMENT-STYLE LINKS (NO <img> ELEMENTS)
// Clicking link opens GLOBAL image viewer
// ============================================================

function displayThumbnails(imagePanel) {
    const section = document.getElementById("thumbnails-section");
    if (!section) {
        console.warn("[displayThumbnails] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();

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
        const src = imgData.src || imgData.url;
        if (!src) {
            console.warn("[displayThumbnails] Image missing src:", imgData);
            return;
        }

        const item = document.createElement("div");
        item.className = "document-item";

        const imageLink = document.createElement("a");
        imageLink.href = "javascript:void(0)";
        imageLink.className = "document-chunk-link";
        imageLink.textContent =
            imgData.title || `View Image ${index + 1}`;

        imageLink.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();

            window.openImageViewer(
                imgData.title || `Image ${index + 1}`,
                src
            );
        });

        item.appendChild(imageLink);
        section.appendChild(item);
    });
}
