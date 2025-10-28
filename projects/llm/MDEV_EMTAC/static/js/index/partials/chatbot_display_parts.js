// Render Parts into #parts-container
function renderParts(parts) {
    const container = document.getElementById("parts-container");
    if (!container) return;
    console.log("renderParts received:", parts);

    container.innerHTML = ""; // Clear old

    if (!Array.isArray(parts) || parts.length === 0) {
        container.innerHTML = "<p>No parts found.</p>";
        return;
    }

    const ul = document.createElement("ul");
    ul.classList.add("parts-list");

    parts.forEach((part, idx) => {
        const li = document.createElement("li");
        li.classList.add("part-item");

        const title = part.title || part.name || `Part ${idx + 1}`;
        const desc = part.description || "";

        li.innerHTML = `<strong>${title}</strong>${desc ? `<br><small>${desc}</small>` : ""}`;
        ul.appendChild(li);
    });

    container.appendChild(ul);
}
