// static/js/partials/chatbot_display_drawings.js

function renderDrawings(drawings) {
    const container = document.getElementById("drawing-section");
    if (!container) {
        console.warn("⚠️ No #drawing-section element found in DOM.");
        return;
    }

    // Clear existing content
    container.innerHTML = "";

    // Handle no results
    if (!Array.isArray(drawings) || drawings.length === 0) {
        container.innerHTML = "<p>No drawings found.</p>";
        return;
    }

    // Create scrollable list container
    const ul = document.createElement("ul");
    ul.classList.add("drawings-list");

    drawings.forEach((drawing, idx) => {
        const li = document.createElement("li");
        li.classList.add("drawing-item");

        // Default icon
        const icon = "📐"; // you can replace with <i class="fa fa-drafting-compass"></i> if using FontAwesome

        // Safe fallbacks for fields
        const equipName = drawing.drw_equipment_name || "Unknown Equipment";
        const drwName = drawing.drw_name || "Untitled Drawing";
        const drwNumber = drawing.drw_number || "N/A";
        const sparePart = drawing.drw_spare_part_number || "N/A";
        const url = drawing.url || "#";

        li.innerHTML = `
            <a href="${url}" target="_blank" rel="noopener noreferrer">
                <span class="drawing-icon">${icon}</span>
                <strong>${drwName}</strong>
            </a>
            <div class="drawing-meta">
                <span><b>Equipment:</b> ${equipName}</span><br>
                <span><b>Number:</b> ${drwNumber}</span><br>
                <span><b>Spare Part #:</b> ${sparePart}</span>
            </div>
        `;

        ul.appendChild(li);
    });

    container.appendChild(ul);
}

// Optional: Apply some CSS directly here for scrollability
const style = document.createElement("style");
style.textContent = `
    .drawings-list {
        list-style: none;
        padding: 0;
        margin: 0;
        max-height: 300px;       /* scroll cutoff */
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 6px;
    }
    .drawing-item {
        padding: 8px;
        border-bottom: 1px solid #eee;
    }
    .drawing-item:last-child {
        border-bottom: none;
    }
    .drawing-item a {
        text-decoration: none;
        color: #2c3e50;
        font-weight: bold;
    }
    .drawing-icon {
        margin-right: 6px;
    }
    .drawing-meta {
        font-size: 0.9em;
        color: #555;
        margin-top: 4px;
    }
`;
document.head.appendChild(style);
