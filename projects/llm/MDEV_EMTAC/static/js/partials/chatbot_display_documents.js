// Render Documents into #doc-links-section
function displayDocuments(docs) {
    const section = document.getElementById("doc-links-section");
    section.innerHTML = "";

    if (!Array.isArray(docs) || docs.length === 0) {
        section.innerHTML = "<p>No documents found.</p>";
        return;
    }

    docs.forEach(doc => {
        const a = document.createElement("a");
        a.href = doc.url;
        a.textContent = doc.title || "Untitled Document";
        a.target = "_blank";
        a.rel = "noopener noreferrer";

        section.appendChild(a);
        section.appendChild(document.createElement("br"));
    });
}
