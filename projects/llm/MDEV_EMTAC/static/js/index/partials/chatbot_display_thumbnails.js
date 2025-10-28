// Render Images into #thumbnails-section
function displayThumbnails(thumbnails) {
    const section = document.getElementById("thumbnails-section");
    section.innerHTML = "";

    thumbnails.forEach(th => {
        const anchor = document.createElement("a");
        anchor.href = th.src;
        anchor.setAttribute("data-full-src", th.src);
        anchor.target = "_blank";

        const img = document.createElement("img");
        img.src = th.thumbnail_src || th.src;
        img.alt = th.title;
        img.title = th.title;
        img.classList.add("thumbnail");

        anchor.appendChild(img);
        section.appendChild(anchor);
    });
}
