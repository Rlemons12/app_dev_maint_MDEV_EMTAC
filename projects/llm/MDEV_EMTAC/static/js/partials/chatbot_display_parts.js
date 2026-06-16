// Render Parts into #parts-container
function renderParts(partsPanel) {
    const container = document.getElementById("parts-container");
    if (!container) return;

    console.log("renderParts received:", partsPanel);

    container.innerHTML = ""; // Clear old

    // ----------------------------------
    // Normalize input (new + old support)
    // ----------------------------------
    const parts = Array.isArray(partsPanel?.parts)
        ? partsPanel.parts
        : Array.isArray(partsPanel)
            ? partsPanel
            : [];

    if (parts.length === 0) {
        container.innerHTML = "<p>No parts found.</p>";
        return;
    }

    const ul = document.createElement("ul");
    ul.classList.add("parts-list");

    parts.forEach((part, idx) => {
        const li = document.createElement("li");
        li.classList.add("part-item");

        const title =
            part.part_number ||
            part.name ||
            part.title ||
            `Part #${idx + 1}`;

        const desc =
            part.description ||
            part.long_description ||
            "";

        li.innerHTML = `
            <strong>${title}</strong>
            ${desc ? `<br><small>${desc}</small>` : ""}
        `;

        ul.appendChild(li);
    });

    container.appendChild(ul);
}
