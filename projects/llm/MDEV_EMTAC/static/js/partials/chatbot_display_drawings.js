// Render Drawings into #drawing-section
function displayDrawings(drawings) {
    const section = document.getElementById("drawing-section");
    section.innerHTML = "";

    if (!Array.isArray(drawings) || drawings.length === 0) {
        section.innerHTML = "<p>No drawings found.</p>";
        return;
    }

    drawings.forEach(d => {
        const a = document.createElement("a");
        a.href = d.url;
        a.textContent = d.title || "Untitled Drawing";
        a.target = "_blank";
        a.rel = "noopener noreferrer";

        section.appendChild(a);
        section.appendChild(document.createElement("br"));
    });
}
