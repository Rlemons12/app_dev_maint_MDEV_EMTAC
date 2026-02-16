// ===============================
// Chatbot Frontend Script
// ===============================

// Global debounce timeout for submissions
let submissionTimeout = null;

// Text-to-speech state
let isTextToSpeechEnabled = false;
let voiceSelect;

// ===============================
// Chatbot Frontend Script
// ===============================

let __isSubmitting = false;   // ✅ REQUIRED lock

if (typeof window.escapeHtml !== "function") {
    window.escapeHtml = function (str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    };
}

// ===============================
// Voice Setup
// ===============================
function populateVoiceList() {
    if (!('speechSynthesis' in window)) return;
    if (!voiceSelect) {
        voiceSelect = document.getElementById('voice-select');
        if (!voiceSelect) {
            console.warn("No #voice-select element found, skipping voice list population.");
            return;
        }
    }

    const voices = window.speechSynthesis.getVoices();
    voiceSelect.innerHTML = "";

    voices.forEach((voice, i) => {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `${voice.name} (${voice.lang})`;
        voiceSelect.appendChild(option);
    });
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = populateVoiceList;
    populateVoiceList();
}

// ===============================
// Text-to-Speech Functions
// ===============================
function toggleTextToSpeech() {
    isTextToSpeechEnabled = !isTextToSpeechEnabled;
    console.log("Text-to-Speech enabled:", isTextToSpeechEnabled);
}

function speakText(text) {
    if (!isTextToSpeechEnabled || !('speechSynthesis' in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    if (voiceSelect && voiceSelect.value) {
        const voices = window.speechSynthesis.getVoices();
        const selectedVoice = voices[voiceSelect.value];
        if (selectedVoice) utterance.voice = selectedVoice;
    }
    window.speechSynthesis.speak(utterance);
}

// ===============================
// Submit Question
// ===============================
function submitQuestion() {
    if (__isSubmitting) {
        console.warn("[submitQuestion] duplicate call blocked");
        return;
    }

    __isSubmitting = true;
    console.log("Submitting question...");

    const userId = document.getElementById('user_id')?.value || "anonymous";
    const area = document.getElementById('area')?.value || "";
    const userInput = document.getElementById('user_input')?.value || "";

    clearAllContainers();

    const inputField = document.getElementById('user_input');
    if (inputField) inputField.value = "";

    fetch('/chatbot/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, area, question: userInput })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received response from server:", data);
        console.log("blocks received:", data.blocks);

        const answerEl = document.getElementById('answer');
        const finalAnswer = data.answer || data.rag_answer || "No answer returned.";

        if (answerEl) answerEl.innerHTML = finalAnswer;

        if (data.blocks) {
            if (data.blocks?.["documents-container"]?.length) {
                const allParts = [];

                data.blocks["documents-container"].forEach(doc => {
                    if (Array.isArray(doc.parts)) {
                        allParts.push(...doc.parts);
                    }
                });

                if (allParts.length > 0) {
                    renderParts(allParts);
                } else {
                    console.warn("[submitQuestion] No parts found in documents-container");
}

            }
            if (data.blocks["images-container"]?.length) displayThumbnails(data.blocks["images-container"]);
            if (data.blocks["documents-container"]?.length) displayDocuments(data.blocks["documents-container"]);

        }


        // --------------------------------------------------
        // DRAWINGS PANEL (Area → Model → Asset navigation)
        // --------------------------------------------------
        if (typeof window.updateDrawingsPanelFromBlocks === "function") {
            window.updateDrawingsPanelFromBlocks(data.blocks);
        } else {
            console.warn(
                "[Drawings] updateDrawingsPanelFromBlocks not found — script load order issue"
            );
        }

        if (isTextToSpeechEnabled) {
            speakText(finalAnswer);
        }
    })
    .catch(err => console.error("Error:", err))
    .finally(() => {
        __isSubmitting = false;
    });
}



// ===============================
// Clear All Containers
// ===============================
function clearAllContainers() {
    const containers = ["parts-container", "thumbnails-section", "doc-links-section", "drawing-section"];
    containers.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = "";
    });
}

// ===============================
// Event Bindings
// ===============================
document.addEventListener("DOMContentLoaded", () => {


    const toggleBtn = document.getElementById("toggle-voice");
    if (toggleBtn) toggleBtn.addEventListener("click", toggleTextToSpeech);
    else console.log("No #toggle-voice button found, skipping voice binding.");
});


// ============================================================
// GLOBAL ENTITY DETAILS VIEWER (Drawings-first, reusable)
// ============================================================

let __entityDetailsViewerWindow = null;

window.openEntityDetailsViewer = function ({
    title = "Details",
    drawings = [],
    images = [],
    parts = []
}) {


    // --------------------------------------------
    // Reuse existing window
    // --------------------------------------------
    if (__entityDetailsViewerWindow && !__entityDetailsViewerWindow.closed) {
        __entityDetailsViewerWindow.focus();
        __entityDetailsViewerWindow.postMessage({ title, drawings, images, parts }, "*");
        return;
    }

    __entityDetailsViewerWindow = window.open(
        "",
        "EMTAC_ENTITY_DETAILS_VIEWER",
        "width=1200,height=750,resizable=yes,scrollbars=yes"
    );

    if (!__entityDetailsViewerWindow) {
        alert("Pop-up blocked. Please allow pop-ups.");
        return;
    }

    __entityDetailsViewerWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
<title>${escapeHtml(title)}</title>

<style>
body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
}
header {
    background: #1c1c1c;
    padding: 10px 14px;
    border-bottom: 1px solid #333;
    color: #39FF14;
    font-weight: bold;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tabs {
    display: flex;
    background: #181818;
    border-bottom: 1px solid #333;
}
.tab {
    padding: 8px 14px;
    cursor: pointer;
    border-right: 1px solid #333;
    color: #ccc;
}
.tab.active {
    background: #222;
    color: #39FF14;
}
.panel {
    display: none;
    padding: 14px;
    height: calc(100vh - 96px);
    overflow-y: auto;
}
.panel.active {
    display: block;
}
.drawing-item {
    padding: 8px 0;
    border-bottom: 1px solid #333;
}
.drawing-item strong {
    color: #39FF14;
}
.image-grid img {
    max-width: 200px;
    margin: 8px;
    border: 1px solid #333;
}
</style>
</head>

<body>

<header id="title"></header>

<div class="tabs">
    <div class="tab active" data-tab="drawings">Drawings</div>
    <div class="tab" data-tab="images">Images</div>
    <div class="tab" data-tab="parts">Parts</div>
</div>

<div id="drawings" class="panel active"></div>
<div id="images" class="panel"></div>
<div id="parts" class="panel"></div>


<script>
const titleEl = document.getElementById("title");
const drawingsPanel = document.getElementById("drawings");
const imagesPanel = document.getElementById("images");
const partsPanel = document.getElementById("parts");


function render({ title, drawings = [], images = [], parts = [] }) {
    titleEl.textContent = title;

    // -------- Drawings --------
    if (Array.isArray(drawings) && drawings.length) {
        drawingsPanel.innerHTML = drawings.map(d => {
            const num = d.drw_number || d.drawing_number || "-";
            const name = d.drw_name || d.name || "";
            const rev = d.drw_revision || d.revision || "";

            return (
                '<div class="drawing-item">' +
                    '<strong>' + num + '</strong>' +
                    (name ? '<div>' + name + '</div>' : '') +
                    (rev ? '<div>Rev: ' + rev + '</div>' : '') +
                '</div>'
            );
        }).join("");
    } else {
        drawingsPanel.innerHTML = "<p>No drawings available.</p>";
    }

    // -------- Images --------
    if (Array.isArray(images) && images.length) {
        imagesPanel.innerHTML =
            '<div class="image-grid">' +
            images.map(i => {
                const src = i.src || i.file_path || i.url;
                return src ? '<img src="' + src + '" />' : '';
            }).join("") +
            '</div>';
    } else {
        imagesPanel.innerHTML = "<p>No images available.</p>";
    }

    // -------- Parts --------
    if (Array.isArray(parts) && parts.length) {
        partsPanel.innerHTML = parts.map(p => {
            const pn = p.part_number || "—";
            const pname = p.name || "";

            return (
                '<div class="drawing-item">' +
                    '<strong>' + pn + '</strong>' +
                    (pname ? '<div>' + pname + '</div>' : '') +
                '</div>'
            );
        }).join("");
    } else {
        partsPanel.innerHTML = "<p>No parts available.</p>";
    }
}



// ---------------- Tabs ----------------
document.querySelectorAll(".tab").forEach(tab => {
    tab.onclick = () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.add("active");
    };
});

// ---------------- Message updates ----------------
window.addEventListener("message", e => {
    if (e.data?.title) render(e.data);
});

// Initial render
render({
    title: ${JSON.stringify(title)},
    drawings: ${JSON.stringify(drawings)},
    images: ${JSON.stringify(images)},
    parts: ${JSON.stringify(parts)}
});

</script>

</body>
</html>
    `);

    __entityDetailsViewerWindow.document.close();
};


// ============================================================
// DRAWING → ENTITY VIEWER ADAPTER
// ============================================================

window.openDrawingDetails = function (drawing) {
    if (!drawing) {
        console.warn("[openDrawingDetails] No drawing provided");
        return;
    }

    const drawings = [drawing];

    const images = Array.isArray(drawing.images)
        ? drawing.images
              .map(img => ({
                  src: img.src || img.file_path || img.url,
                  title: img.title || img.name || ""
              }))
              .filter(i => i.src)
        : [];

    // 🔑 DEFINE PARTS HERE (THIS WAS MISSING)
    const parts = Array.isArray(drawing.spare_parts)
        ? drawing.spare_parts.map(p => ({
              part_number: p.part_number,
              name: p.name
          }))
        : [];

    const title =
        drawing.drw_number ||
        drawing.drawing_number ||
        drawing.drw_name ||
        "Drawing Details";

    window.openEntityDetailsViewer({
        title,
        drawings,
        images,
        parts   // ✅ now defined
    });
};

// ============================================================
// GLOBAL IMAGE VIEWER (Single Reused Window)
// ============================================================

let __imageViewerWindow = null;

window.openImageViewer = function (title, src) {

    // --------------------------------------------------------
    // Reuse window if already open
    // --------------------------------------------------------
    if (__imageViewerWindow && !__imageViewerWindow.closed) {
        __imageViewerWindow.focus();
        __imageViewerWindow.postMessage({ title, src }, "*");
        return;
    }

    __imageViewerWindow = window.open(
        "",
        "EMTAC_IMAGE_VIEWER",
        "width=1000,height=750,scrollbars=no,resizable=yes"
    );

    if (!__imageViewerWindow) {
        alert("Pop-up blocked. Please allow pop-ups for this site.");
        return;
    }

    __imageViewerWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(title)}</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }

        header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: #1c1c1c;
            border-bottom: 1px solid #333;
        }

        header h2 {
            flex: 1;
            font-size: 14px;
            color: #39FF14;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        button {
            background: #2a2a2a;
            color: #eee;
            border: 1px solid #444;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }

        button:hover {
            background: #333;
        }

        #viewer {
            width: 100%;
            height: calc(100vh - 42px);
            overflow: hidden;
            cursor: grab;
            position: relative;
        }

        img {
    position: absolute;
    top: 50%;
    left: 50%;
    transform-origin: center center;
    user-select: none;
    pointer-events: none;

    /* 🔑 TRUE CENTERING */
    transform: translate(-50%, -50%);
}

    </style>
</head>
<body>

<header>
    <h2 id="title"></h2>
    <button onclick="zoom(1.2)">+</button>
    <button onclick="zoom(0.8)">−</button>
    <button onclick="resetView()">Reset</button>
    <button onclick="download()">Download</button>
</header>

<div id="viewer">
    <img id="image" />
</div>

<script>
    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let dragging = false;
    let startX, startY;

    const img = document.getElementById("image");
    const titleEl = document.getElementById("title");
    const viewer = document.getElementById("viewer");

    function updateTransform() {
    img.style.transform =
        "translate(-50%, -50%) translate(" +
        offsetX + "px," + offsetY + "px) scale(" + scale + ")";
}


    function loadImage(title, src) {
        titleEl.textContent = title;
        img.src = src;
        scale = 1;
        offsetX = 0;
        offsetY = 0;
        updateTransform();
    }

    function zoom(factor) {
        scale *= factor;
        updateTransform();
    }

    function resetView() {
        scale = 1;
        offsetX = 0;
        offsetY = 0;
        updateTransform();
    }

    function download() {
        const a = document.createElement("a");
        a.href = img.src;
        a.download = "";
        a.click();
    }

    viewer.addEventListener("mousedown", e => {
        dragging = true;
        viewer.style.cursor = "grabbing";
        startX = e.clientX - offsetX;
        startY = e.clientY - offsetY;
    });

    viewer.addEventListener("mousemove", e => {
        if (!dragging) return;
        offsetX = e.clientX - startX;
        offsetY = e.clientY - startY;
        updateTransform();
    });

    window.addEventListener("mouseup", () => {
        dragging = false;
        viewer.style.cursor = "grab";
    });

    viewer.addEventListener("wheel", e => {
        e.preventDefault();
        zoom(e.deltaY < 0 ? 1.1 : 0.9);
    }, { passive: false });

    window.addEventListener("message", e => {
        if (e.data?.src) {
            loadImage(e.data.title, e.data.src);
        }
    });

    // initial load
    loadImage(${JSON.stringify(title)}, ${JSON.stringify(src)});
</script>

</body>
</html>
    `);

    __imageViewerWindow.document.close();
};
// ============================================================
// GLOBAL DOCUMENT VIEWER (Single Reused Window)
// ============================================================

let __documentViewerWindow = null;

window.openDocumentViewer = function (title, text) {

    // --------------------------------------------
    // Reuse window if already open
    // --------------------------------------------
    if (__documentViewerWindow && !__documentViewerWindow.closed) {
        __documentViewerWindow.focus();
        __documentViewerWindow.postMessage({ title, text }, "*");
        return;
    }

    __documentViewerWindow = window.open(
        "",
        "EMTAC_DOCUMENT_VIEWER",
        "width=800,height=650,scrollbars=yes,resizable=yes"
    );

    if (!__documentViewerWindow) {
        alert("Pop-up blocked. Please allow pop-ups for this site.");
        return;
    }

    __documentViewerWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(title)}</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
        }

        header {
            padding: 10px 14px;
            background: #1c1c1c;
            border-bottom: 1px solid #333;
        }

        header h2 {
            margin: 0;
            font-size: 14px;
            color: #39FF14;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        pre {
            padding: 16px;
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
            font-size: 13px;
            background: #111;
        }
    </style>
</head>
<body>

<header>
    <h2 id="title"></h2>
</header>

<pre id="content"></pre>

<script>
    const titleEl = document.getElementById("title");
    const contentEl = document.getElementById("content");

    function loadDocument(title, text) {
        titleEl.textContent = title;
        contentEl.textContent = text;
    }

    window.addEventListener("message", e => {
        if (e.data?.text) {
            loadDocument(e.data.title, e.data.text);
        }
    });

    // initial load
    loadDocument(${JSON.stringify(title)}, ${JSON.stringify(text)});
</script>

</body>
</html>
    `);

    __documentViewerWindow.document.close();
};
// ============================================================
// GLOBAL PART IMAGES VIEWER (Single Reused Window)
// ============================================================

let __partImagesViewerWindow = null;

window.openPartImagesViewer = function (title, images) {

    if (!Array.isArray(images) || images.length === 0) {
        console.warn("[PartImagesViewer] No images to display");
        return;
    }

    // --------------------------------------------
    // Reuse existing window
    // --------------------------------------------
    if (__partImagesViewerWindow && !__partImagesViewerWindow.closed) {
        __partImagesViewerWindow.focus();
        __partImagesViewerWindow.postMessage({ title, images }, "*");
        return;
    }

    __partImagesViewerWindow = window.open(
        "",
        "EMTAC_PART_IMAGES_VIEWER",
        "width=1100,height=750,scrollbars=no,resizable=yes"
    );

    if (!__partImagesViewerWindow) {
        alert("Pop-up blocked. Please allow pop-ups for this site.");
        return;
    }

    __partImagesViewerWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
    <title>${escapeHtml(title)}</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        header {
            padding: 10px 14px;
            background: #1c1c1c;
            border-bottom: 1px solid #333;
            font-size: 14px;
            color: #39FF14;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* LEFT: THUMBNAILS */
        .sidebar {
            width: 260px;
            background: #181818;
            border-right: 1px solid #333;
            overflow-y: auto;
            padding: 8px;
        }

        .thumb {
            margin-bottom: 8px;
            cursor: pointer;
            border: 1px solid #333;
            padding: 4px;
            background: #111;
        }

        .thumb img {
            width: 100%;
            display: block;
        }

        .thumb:hover {
            border-color: #39FF14;
        }

        /* RIGHT: MAIN IMAGE */
        .viewer {
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #000;
        }

        .viewer img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 100%;
            max-height: 100%;
            user-select: none;
        }
    </style>
</head>
<body>

<header id="title"></header>

<div class="container">
    <div class="sidebar" id="thumbs"></div>
    <div class="viewer">
        <img id="mainImage" />
    </div>
</div>

<script>
    const titleEl = document.getElementById("title");
    const thumbsEl = document.getElementById("thumbs");
    const mainImage = document.getElementById("mainImage");

    function loadImages(title, images) {
        titleEl.textContent = title;
        thumbsEl.innerHTML = "";

        if (!images.length) return;

        // Load first image by default
        const first = images[0];
        mainImage.src = first.file_path || first.src || first.url || "";

        images.forEach(img => {
            const src = img.file_path || img.src || img.url;
            if (!src) return;

            const div = document.createElement("div");
            div.className = "thumb";

            const image = document.createElement("img");
            image.src = src;

            div.appendChild(image);

            div.addEventListener("click", () => {
                mainImage.src = src;
            });

            thumbsEl.appendChild(div);
        });
    }

    window.addEventListener("message", e => {
        if (e.data?.images) {
            loadImages(e.data.title, e.data.images);
        }
    });

    // Initial load
    loadImages(${JSON.stringify(title)}, ${JSON.stringify(images)});
</script>

</body>
</html>
    `);

    __partImagesViewerWindow.document.close();
};

window.openPartDetailsViewer = function (title, images = [], drawings = []) {

    const win = window.open(
        "",
        "EMTAC_PART_DETAILS_VIEWER",
        "width=1200,height=750,resizable=yes"
    );

    if (!win) {
        alert("Pop-up blocked. Please allow pop-ups.");
        return;
    }

    win.document.write(`
<!DOCTYPE html>
<html>
<head>
<title>${escapeHtml(title)}</title>
<style>
body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
}
header {
    background: #1c1c1c;
    padding: 10px;
    border-bottom: 1px solid #333;
    color: #39FF14;
}
.tabs {
    display: flex;
    background: #181818;
}
.tab {
    padding: 8px 12px;
    cursor: pointer;
    border-right: 1px solid #333;
}
.tab.active {
    background: #222;
    color: #39FF14;
}
.panel {
    display: none;
    padding: 12px;
    height: calc(100vh - 110px);
    overflow-y: auto;
}
.panel.active {
    display: block;
}
img {
    max-width: 200px;
    margin: 8px;
    border: 1px solid #333;
}
.drawing {
    padding: 8px;
    border-bottom: 1px solid #333;
}
</style>
</head>
<body>

<header>${escapeHtml(title)}</header>

<div class="tabs">
    <div class="tab active" data-tab="images">Images</div>
    <div class="tab" data-tab="drawings">Drawings</div>
</div>

<div class="panel active" id="images"></div>
<div class="panel" id="drawings"></div>

<script>
var images = ${JSON.stringify(images)};
var drawings = ${JSON.stringify(drawings)};

var imgPanel = document.getElementById("images");
var drwPanel = document.getElementById("drawings");

/* ---------------- Images ---------------- */
if (Array.isArray(images) && images.length) {
    imgPanel.innerHTML = images.map(function (i) {
    var src = i.src || "";
    var lower = src.toLowerCase();

    // Only render browser-safe image types
    if (!lower.match(/\\.(png|jpg|jpeg|gif|webp)$/)) {
        return (
            '<div style="margin:8px;color:#aaa;font-size:12px">' +
            'Unsupported image type:<br>' +
            '<a href="' + src + '" target="_blank" style="color:#39FF14">' +
            (i.title || src) +
            '</a></div>'
        );
    }

    return '<img src="' + src + '" alt="' + (i.title || '') + '" />';
}).join('');

} else {
    imgPanel.innerHTML = "<p>No images</p>";
}

/* ---------------- Drawings ---------------- */
if (Array.isArray(drawings) && drawings.length) {
    drwPanel.innerHTML = drawings.map(function (d) {

        // Case 1: drawing is already a string
        if (typeof d === "string") {
            return (
                '<div class="drawing">' +
                    '<strong>' + d + '</strong>' +
                '</div>'
            );
        }

        // Case 2: drawing is an object (various backends)
        var number =
            d.drw_number ||
            d.drawing_number ||
            d.number ||
            "-";

        var name =
            d.drw_name ||
            d.name ||
            "";

        var rev =
            d.drw_revision ||
            d.revision ||
            "";

        return (
            '<div class="drawing">' +
                '<strong>' + number + '</strong>' +
                (name ? '<div>' + name + '</div>' : '') +
                (rev ? '<div>Rev: ' + rev + '</div>' : '') +
            '</div>'
        );
    }).join("");
} else {
    drwPanel.innerHTML = "<p>No drawings</p>";
}


/* ---------------- Tabs ---------------- */
document.querySelectorAll(".tab").forEach(function (tab) {
    tab.onclick = function () {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.add("active");
    };
});
</script>

</body>
</html>
    `);

    win.document.close();
};


