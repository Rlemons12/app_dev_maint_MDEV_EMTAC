/* ============================================================
   EMTAC – Drawing Panel
   Tablet/WebView-safe version

   File:
     static/js/index/partials/chatbot_display_drawings.js

   Purpose:
   - Renders drawings on the chatbot/index page
   - Supports All / By Area / By Asset Number tabs
   - Opens drawing details in a fullscreen in-page overlay
   - Displays browser-previewable drawing PDFs inside the index overlay
   - Uses /drawings/file-status/<drawing_id> to check whether a file exists
   - Uses /drawings/file/<drawing_id> for inline PDF preview
   - Does NOT use the upload_search_database drawing print viewer page
   - Android/WebView Back button closes the active drawing viewer
============================================================ */

console.log("[EMTAC] chatbot_display_drawings.js loaded - index drawing renderer with inline PDF preview");


/* ------------------------------------------------------------
   STATE
------------------------------------------------------------ */

var drawingState = window.__EMTAC_DRAWING_STATE__ || {
  drawings: [],
  activeTab: "all",        // all | area | asset
  selectedArea: null,
  selectedAsset: null,
};

window.__EMTAC_DRAWING_STATE__ = drawingState;


/* ------------------------------------------------------------
   SAFE DOM
------------------------------------------------------------ */

function el(id) {
  return document.getElementById(id);
}

function clear(node) {
  if (node) {
    node.innerHTML = "";
  }
}

function safeText(value, fallback = "") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  return String(value);
}


/* ------------------------------------------------------------
   NATURAL SORT HELPERS
------------------------------------------------------------ */

var drawingNaturalSorter = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: "base",
});

function safeSortValue(value) {
  if (value === null || value === undefined || value === "") {
    return "zzzzzz";
  }

  return String(value).trim();
}

function compareNatural(a, b) {
  return drawingNaturalSorter.compare(
    safeSortValue(a),
    safeSortValue(b)
  );
}

function getDrawingNumberValue(drawing) {
  return (
    drawing?.drw_number ||
    drawing?.drawing_number ||
    drawing?.number ||
    drawing?.drawing_no ||
    ""
  );
}

function getDrawingTitleValue(drawing) {
  return (
    drawing?.drw_name ||
    drawing?.title ||
    drawing?.name ||
    ""
  );
}

function sortDrawingsForDisplay(drawings) {
  if (!Array.isArray(drawings)) {
    return [];
  }

  return [...drawings].sort((a, b) => {
    return (
      compareNatural(a?._area, b?._area) ||
      compareNatural(a?._asset, b?._asset) ||
      compareNatural(a?._model, b?._model) ||
      compareNatural(getDrawingNumberValue(a), getDrawingNumberValue(b)) ||
      compareNatural(getDrawingTitleValue(a), getDrawingTitleValue(b))
    );
  });
}


/* ------------------------------------------------------------
   DATA NORMALIZATION
------------------------------------------------------------ */

function flattenDrawingNavigation(nav) {
  if (!nav || !Array.isArray(nav.areas)) {
    return [];
  }

  const out = [];

  nav.areas.forEach(area => {
    (area.models || []).forEach(model => {
      (model.assets || []).forEach(asset => {
        (asset.drawings || []).forEach(drawing => {
          out.push(normalizeDrawing({
            ...drawing,
            _area: area.area_name || area.name || "Unknown Area",
            _model: model.model_name || model.name || "Unknown Model",
            _asset: asset.asset_name || asset.asset_number || asset.name || "Unknown Asset Number",
          }));
        });
      });
    });
  });

  return out;
}

function extractDrawingsFromDocuments(docs = []) {
  const all = [];

  docs.forEach(doc => {
    if (doc && doc.drawing_navigation) {
      all.push(...flattenDrawingNavigation(doc.drawing_navigation));
    }
  });

  return dedupeDrawings(all);
}

function normalizeDrawing(drawing) {
  if (!drawing || typeof drawing !== "object") {
    return null;
  }

  const area =
    drawing._area ||
    drawing.area ||
    drawing.area_name ||
    drawing.areaName ||
    drawing.location_area ||
    "Unknown Area";

  const model =
    drawing._model ||
    drawing.model ||
    drawing.model_name ||
    drawing.modelName ||
    "Unknown Model";

  const asset =
    drawing._asset ||
    drawing.asset_number ||
    drawing.assetNumber ||
    drawing.asset ||
    drawing.asset_name ||
    drawing.assetName ||
    drawing.asset_no ||
    "Unknown Asset Number";

  return {
    ...drawing,
    _area: area,
    _model: model,
    _asset: asset,
  };
}

function normalizeDrawingList(drawings = []) {
  if (!Array.isArray(drawings)) {
    return [];
  }

  return dedupeDrawings(
    drawings
      .map(normalizeDrawing)
      .filter(Boolean)
  );
}

function dedupeDrawings(list) {
  const seen = new Set();

  return list.filter(drawing => {
    /*
      Important:
      Do not dedupe only by drawing number.
      The same drawing can be associated with different areas/assets.
    */
    const keyParts = [
      drawing.id || "",
      drawing.drawing_id || "",
      drawing.drw_number || drawing.drawing_number || drawing.number || "",
      drawing._area || "",
      drawing._model || "",
      drawing._asset || "",
      drawing.url || drawing.file_url || drawing.file_path || "",
    ];

    const key = keyParts.join("|");

    if (seen.has(key)) {
      return false;
    }

    seen.add(key);
    return true;
  });
}


/* ------------------------------------------------------------
   FILTER HELPERS
------------------------------------------------------------ */

function getAreaCounts() {
  const counts = new Map();

  drawingState.drawings.forEach(drawing => {
    const key = drawing._area || "Unknown Area";
    counts.set(key, (counts.get(key) || 0) + 1);
  });

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => compareNatural(a.name, b.name));
}

function getAssetCounts() {
  const counts = new Map();

  drawingState.drawings.forEach(drawing => {
    const key = drawing._asset || "Unknown Asset Number";
    counts.set(key, (counts.get(key) || 0) + 1);
  });

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => compareNatural(a.name, b.name));
}

function getVisibleDrawings() {
  let visible = [];

  if (drawingState.activeTab === "area") {
    if (!drawingState.selectedArea) {
      return [];
    }

    visible = drawingState.drawings.filter(
      drawing => drawing._area === drawingState.selectedArea
    );

    return sortDrawingsForDisplay(visible);
  }

  if (drawingState.activeTab === "asset") {
    if (!drawingState.selectedAsset) {
      return [];
    }

    visible = drawingState.drawings.filter(
      drawing => drawing._asset === drawingState.selectedAsset
    );

    return sortDrawingsForDisplay(visible);
  }

  return sortDrawingsForDisplay(drawingState.drawings);
}

function validateCurrentSelections() {
  const areaNames = new Set(getAreaCounts().map(area => area.name));
  const assetNames = new Set(getAssetCounts().map(asset => asset.name));

  if (drawingState.selectedArea && !areaNames.has(drawingState.selectedArea)) {
    drawingState.selectedArea = null;
  }

  if (drawingState.selectedAsset && !assetNames.has(drawingState.selectedAsset)) {
    drawingState.selectedAsset = null;
  }
}


/* ------------------------------------------------------------
   ENTRY POINTS
------------------------------------------------------------ */

function renderDrawings(drawings) {
  drawingState.drawings = sortDrawingsForDisplay(
    normalizeDrawingList(drawings)
  );

  if (!["all", "area", "asset"].includes(drawingState.activeTab)) {
    drawingState.activeTab = "all";
  }

  validateCurrentSelections();
  renderAll();
}

function updateDrawingsPanelFromBlocks(blocks) {
  drawingState.drawings = sortDrawingsForDisplay(
    extractDrawingsFromDocuments(
      blocks?.["documents-container"] || []
    )
  );

  drawingState.activeTab = "all";
  drawingState.selectedArea = null;
  drawingState.selectedAsset = null;

  renderAll();
}


/* ------------------------------------------------------------
   MASTER RENDER
------------------------------------------------------------ */

function renderAll() {
  ensureDrawingViewerStyles();
  renderDrawingTabs();
  renderDrawingSelectors();
  renderDrawingList(getVisibleDrawings());
}


/* ------------------------------------------------------------
   TABS
------------------------------------------------------------ */

function renderDrawingTabs() {
  const bar = el("drawing-area-bar");

  if (!bar) {
    console.warn("[renderDrawingTabs] #drawing-area-bar not found");
    return;
  }

  clear(bar);
  bar.classList.add("drawing-tab-bar");

  const tabs = [
    {
      key: "all",
      label: `All Drawings (${drawingState.drawings.length})`,
    },
    {
      key: "area",
      label: "By Area",
    },
    {
      key: "asset",
      label: "By Asset Number",
    },
  ];

  tabs.forEach(tab => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "drawing-tab-button";
    button.textContent = tab.label;

    if (drawingState.activeTab === tab.key) {
      button.classList.add("active");
    }

    button.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();

      drawingState.activeTab = tab.key;

      if (tab.key !== "area") {
        drawingState.selectedArea = null;
      }

      if (tab.key !== "asset") {
        drawingState.selectedAsset = null;
      }

      renderAll();
    });

    bar.appendChild(button);
  });
}


/* ------------------------------------------------------------
   SELECTORS
------------------------------------------------------------ */

function renderDrawingSelectors() {
  const bar = el("drawing-model-bar");

  if (!bar) {
    console.warn("[renderDrawingSelectors] #drawing-model-bar not found");
    return;
  }

  clear(bar);
  bar.classList.add("drawing-selector-bar");

  if (drawingState.activeTab === "all") {
    const info = document.createElement("div");
    info.className = "drawing-filter-info";
    info.textContent = `Showing ${drawingState.drawings.length} drawing(s), sorted by Area → Asset Number → Model → Drawing Number.`;
    bar.appendChild(info);
    return;
  }

  if (drawingState.activeTab === "area") {
    renderAreaSelector(bar);
    return;
  }

  if (drawingState.activeTab === "asset") {
    renderAssetSelector(bar);
  }
}

function renderAreaSelector(container) {
  const areas = getAreaCounts();

  if (!areas.length) {
    const empty = document.createElement("div");
    empty.className = "drawing-filter-info";
    empty.textContent = "No areas found.";
    container.appendChild(empty);
    return;
  }

  const label = document.createElement("div");
  label.className = "drawing-filter-label";
  label.textContent = "Select Area:";
  container.appendChild(label);

  const chipWrap = document.createElement("div");
  chipWrap.className = "drawing-chip-wrap";

  areas.forEach(area => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "drawing-chip";
    button.textContent = `${area.name} (${area.count})`;

    if (drawingState.selectedArea === area.name) {
      button.classList.add("active");
    }

    button.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();

      drawingState.selectedArea =
        drawingState.selectedArea === area.name
          ? null
          : area.name;

      renderAll();
    });

    chipWrap.appendChild(button);
  });

  container.appendChild(chipWrap);

  if (!drawingState.selectedArea) {
    const prompt = document.createElement("div");
    prompt.className = "drawing-filter-prompt";
    prompt.textContent = "Choose an area to show matching drawings.";
    container.appendChild(prompt);
  }
}

function renderAssetSelector(container) {
  const assets = getAssetCounts();

  if (!assets.length) {
    const empty = document.createElement("div");
    empty.className = "drawing-filter-info";
    empty.textContent = "No asset numbers found.";
    container.appendChild(empty);
    return;
  }

  const label = document.createElement("div");
  label.className = "drawing-filter-label";
  label.textContent = "Select Asset Number:";
  container.appendChild(label);

  const chipWrap = document.createElement("div");
  chipWrap.className = "drawing-chip-wrap";

  assets.forEach(asset => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "drawing-chip";
    button.textContent = `${asset.name} (${asset.count})`;

    if (drawingState.selectedAsset === asset.name) {
      button.classList.add("active");
    }

    button.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();

      drawingState.selectedAsset =
        drawingState.selectedAsset === asset.name
          ? null
          : asset.name;

      renderAll();
    });

    chipWrap.appendChild(button);
  });

  container.appendChild(chipWrap);

  if (!drawingState.selectedAsset) {
    const prompt = document.createElement("div");
    prompt.className = "drawing-filter-prompt";
    prompt.textContent = "Choose an asset number to show matching drawings.";
    container.appendChild(prompt);
  }
}


/* ------------------------------------------------------------
   DRAWINGS LIST
------------------------------------------------------------ */

function renderDrawingList(drawings) {
  const box = el("drawing-section");

  if (!box) {
    console.warn("[renderDrawingList] #drawing-section not found");
    return;
  }

  clear(box);

  if (!Array.isArray(drawings) || drawings.length === 0) {
    const empty = document.createElement("p");

    if (drawingState.activeTab === "area" && !drawingState.selectedArea) {
      empty.textContent = "Select an area to view drawings.";
    } else if (drawingState.activeTab === "asset" && !drawingState.selectedAsset) {
      empty.textContent = "Select an asset number to view drawings.";
    } else {
      empty.textContent = "No drawings found.";
    }

    box.appendChild(empty);
    return;
  }

  const count = document.createElement("div");
  count.className = "drawing-result-count";

  if (drawingState.activeTab === "area" && drawingState.selectedArea) {
    count.textContent = `Showing ${drawings.length} drawing(s) for Area: ${drawingState.selectedArea}.`;
  } else if (drawingState.activeTab === "asset" && drawingState.selectedAsset) {
    count.textContent = `Showing ${drawings.length} drawing(s) for Asset Number: ${drawingState.selectedAsset}.`;
  } else {
    count.textContent = `Showing ${drawings.length} drawing(s).`;
  }

  box.appendChild(count);

  const ul = document.createElement("ul");
  ul.className = "drawings-list";

  drawings.forEach(drawing => {
    const li = document.createElement("li");
    li.className = "drawing-item";

    const link = document.createElement("button");
    link.type = "button";
    link.className = "drawing-link-button";

    const drawingTitle =
      drawing.drw_name ||
      drawing.title ||
      drawing.name ||
      "Untitled Drawing";

    link.innerHTML = `<strong>${escapeHtml(drawingTitle)}</strong>`;

    link.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      openDrawingDetailsInPage(drawing);
    };

    const meta = document.createElement("div");
    meta.className = "drawing-meta";

    meta.innerHTML = `
      <div><b>No:</b> ${escapeHtml(drawing.drw_number || drawing.drawing_number || drawing.number || "—")}</div>
      <div><b>Area:</b> ${escapeHtml(drawing._area || "—")}</div>
      <div><b>Model:</b> ${escapeHtml(drawing._model || "—")}</div>
      <div><b>Asset Number:</b> ${escapeHtml(drawing._asset || "—")}</div>
    `;

    if (Array.isArray(drawing.spare_parts) && drawing.spare_parts.length > 0) {
      meta.appendChild(buildSparePartsBlock(drawing.spare_parts));
    }

    li.appendChild(link);
    li.appendChild(meta);
    ul.appendChild(li);
  });

  box.appendChild(ul);
}

function buildSparePartsBlock(parts) {
  const sp = document.createElement("div");
  sp.className = "drawing-spares";

  const label = document.createElement("div");
  label.className = "spare-label";
  label.textContent = "Spare Parts:";
  sp.appendChild(label);

  parts.forEach(part => {
    const row = document.createElement("div");
    row.className = "spare-row";

    const partNumber = document.createElement("span");
    partNumber.className = "spare-part-number";
    partNumber.textContent = part.part_number || "—";

    const partName = document.createElement("span");
    partName.className = "spare-part-name";
    partName.textContent = part.name || "";

    row.appendChild(partNumber);
    row.appendChild(partName);
    sp.appendChild(row);
  });

  return sp;
}


/* ------------------------------------------------------------
   IN-PAGE DRAWING VIEWER
------------------------------------------------------------ */

function openDrawingDetailsInPage(drawing) {
  ensureDrawingViewerStyles();
  closeDrawingDetailsInPage(false);

  const overlay = document.createElement("div");
  overlay.id = "emtac-drawing-viewer-overlay";
  overlay.className = "emtac-drawing-viewer-overlay";

  const panel = document.createElement("div");
  panel.className = "emtac-drawing-viewer-panel";

  const header = document.createElement("div");
  header.className = "emtac-drawing-viewer-header";

  const title = document.createElement("h2");
  title.textContent =
    drawing?.drw_name ||
    drawing?.title ||
    drawing?.name ||
    "Drawing Details";

  const buttons = document.createElement("div");
  buttons.className = "emtac-drawing-viewer-buttons";

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "emtac-drawing-viewer-button emtac-drawing-viewer-close";
  closeButton.textContent = "Close";

  closeButton.onclick = () => {
    closeDrawingDetailsInPage(true);
  };

  buttons.appendChild(closeButton);

  header.appendChild(title);
  header.appendChild(buttons);

  const body = document.createElement("div");
  body.className = "emtac-drawing-viewer-body";

  body.appendChild(buildDrawingDetailGrid(drawing));

  const pdfSection = document.createElement("div");
  pdfSection.className = "emtac-drawing-pdf-section";

  const pdfTitle = document.createElement("h3");
  pdfTitle.textContent = "Drawing PDF";

  const pdfStatus = document.createElement("div");
  pdfStatus.className = "emtac-drawing-pdf-status";
  pdfStatus.textContent = "Checking drawing PDF...";

  const pdfFrameWrap = document.createElement("div");
  pdfFrameWrap.className = "emtac-drawing-pdf-frame-wrap";

  pdfSection.appendChild(pdfTitle);
  pdfSection.appendChild(pdfStatus);
  pdfSection.appendChild(pdfFrameWrap);
  body.appendChild(pdfSection);

  if (Array.isArray(drawing?.spare_parts) && drawing.spare_parts.length > 0) {
    const spareSection = document.createElement("div");
    spareSection.className = "emtac-drawing-viewer-section";

    const spareTitle = document.createElement("h3");
    spareTitle.textContent = "Spare Parts";

    spareSection.appendChild(spareTitle);
    spareSection.appendChild(buildSparePartsBlock(drawing.spare_parts));
    body.appendChild(spareSection);
  }

  const rawSection = document.createElement("details");
  rawSection.className = "emtac-drawing-viewer-raw";

  const rawSummary = document.createElement("summary");
  rawSummary.textContent = "Raw Drawing Data";

  const rawPre = document.createElement("pre");
  rawPre.textContent = JSON.stringify(drawing || {}, null, 2);

  rawSection.appendChild(rawSummary);
  rawSection.appendChild(rawPre);
  body.appendChild(rawSection);

  panel.appendChild(header);
  panel.appendChild(body);
  overlay.appendChild(panel);

  document.body.appendChild(overlay);
  document.body.classList.add("emtac-drawing-viewer-open");

  /*
    Load the PDF directly into the index/chatbot overlay.
    No EMTAC print button is passed here because the viewer should not
    expose a separate print control.
  */
  loadDrawingPdfIntoIndexViewer(drawing, pdfStatus, pdfFrameWrap);

  try {
    if (!window.history.state || window.history.state.emtacDrawingViewer !== true) {
      window.history.pushState(
        { emtacDrawingViewer: true },
        "",
        window.location.href
      );
    }
  } catch (err) {
    console.warn("[openDrawingDetailsInPage] Could not push history state:", err);
  }
}

function buildDrawingDetailGrid(drawing) {
  const grid = document.createElement("div");
  grid.className = "emtac-drawing-detail-grid";

  addDrawingDetailRow(grid, "Drawing Name", drawing?.drw_name || drawing?.title || drawing?.name || "—");
  addDrawingDetailRow(grid, "Drawing Number", drawing?.drw_number || drawing?.drawing_number || drawing?.number || "—");
  addDrawingDetailRow(grid, "Revision", drawing?.drw_revision || drawing?.revision || "—");
  addDrawingDetailRow(grid, "Equipment Name", drawing?.drw_equipment_name || drawing?.equipment_name || "—");
  addDrawingDetailRow(grid, "Area", drawing?._area || "—");
  addDrawingDetailRow(grid, "Model", drawing?._model || "—");
  addDrawingDetailRow(grid, "Asset Number", drawing?._asset || "—");
  addDrawingDetailRow(grid, "Spare Part Number", drawing?.drw_spare_part_number || "—");

  const fileUrl = getDrawingFileUrl(drawing);
  const drawingId = getDrawingId(drawing);

  addDrawingDetailRow(grid, "Drawing ID", drawingId || "—");
  addDrawingDetailRow(grid, "File URL", fileUrl || "—");

  return grid;
}

function addDrawingDetailRow(grid, label, value) {
  const labelDiv = document.createElement("div");
  labelDiv.className = "emtac-drawing-detail-label";
  labelDiv.textContent = label;

  const valueDiv = document.createElement("div");
  valueDiv.className = "emtac-drawing-detail-value";
  valueDiv.textContent = value == null || value === "" ? "—" : String(value);

  grid.appendChild(labelDiv);
  grid.appendChild(valueDiv);
}

function getDrawingId(drawing) {
  if (!drawing || typeof drawing !== "object") {
    return null;
  }

  return (
    drawing.id ||
    drawing.drawing_id ||
    drawing.drawingId ||
    drawing.drw_id ||
    null
  );
}

function getDrawingFileUrl(drawing) {
  if (!drawing || typeof drawing !== "object") {
    return null;
  }

  return (
    drawing.url ||
    drawing.file_url ||
    drawing.file_path_url ||
    drawing.web_url ||
    drawing.href ||
    null
  );
}

function getDrawingInlinePdfUrl(drawing) {
  const drawingId = getDrawingId(drawing);

  if (drawingId) {
    return `/drawings/file/${encodeURIComponent(drawingId)}`;
  }

  return getDrawingFileUrl(drawing);
}

function getDrawingFileStatusUrl(drawing) {
  const drawingId = getDrawingId(drawing);

  if (!drawingId) {
    return null;
  }

  return `/drawings/file-status/${encodeURIComponent(drawingId)}`;
}

async function loadDrawingPdfIntoIndexViewer(drawing, statusNode, frameWrapNode) {
  const inlinePdfUrl = getDrawingInlinePdfUrl(drawing);
  const statusUrl = getDrawingFileStatusUrl(drawing);

  if (!inlinePdfUrl) {
    showDrawingPdfWarning(
      statusNode,
      "No drawing PDF path is available for this drawing record."
    );
    return;
  }

  if (!statusUrl) {
    showDrawingPdfInfo(
      statusNode,
      "Drawing PDF preview loaded from the file URL provided by the search result."
    );

    renderDrawingPdfFrame(inlinePdfUrl, frameWrapNode);
    return;
  }

  try {
    const response = await fetch(statusUrl, {
      method: "GET",
      headers: {
        "Accept": "application/json"
      }
    });

    let data = null;

    try {
      data = await response.json();
    } catch (jsonError) {
      console.warn("[loadDrawingPdfIntoIndexViewer] File status response was not JSON:", jsonError);
    }

    if (!response.ok || !data || !data.success || !data.can_open_file) {
      const message =
        data?.error?.message ||
        data?.message ||
        "The database found the drawing record, but the drawing PDF could not be loaded.";

      showDrawingPdfWarning(statusNode, message);
      return;
    }

    if (data.can_preview === false) {
      showDrawingPdfWarning(
        statusNode,
        "The drawing file exists, but the browser cannot preview this file type."
      );
      return;
    }

    const fileUrl = data.file_url || inlinePdfUrl;

    showDrawingPdfInfo(statusNode, "Drawing PDF loaded.");
    renderDrawingPdfFrame(fileUrl, frameWrapNode);
  } catch (error) {
    console.warn("[loadDrawingPdfIntoIndexViewer] Could not check drawing file status:", error);

    showDrawingPdfWarning(
      statusNode,
      "Could not verify the drawing file. Attempting to load the PDF directly."
    );

    renderDrawingPdfFrame(inlinePdfUrl, frameWrapNode);
  }
}

function showDrawingPdfInfo(statusNode, message) {
  if (!statusNode) {
    return;
  }

  statusNode.textContent = message;
  statusNode.classList.remove("emtac-drawing-pdf-status-warning");
}

function showDrawingPdfWarning(statusNode, message) {
  if (!statusNode) {
    return;
  }

  statusNode.textContent = message;
  statusNode.classList.add("emtac-drawing-pdf-status-warning");
}

function renderDrawingPdfFrame(pdfUrl, frameWrapNode) {
  if (!frameWrapNode) {
    return null;
  }

  frameWrapNode.innerHTML = "";

  const iframe = document.createElement("iframe");
  iframe.className = "emtac-drawing-pdf-frame";
  iframe.src = buildPdfViewerUrlWithoutToolbar(pdfUrl);
  iframe.title = "Drawing PDF";
  iframe.loading = "lazy";

  iframe.addEventListener("load", () => {
    console.log("[EMTAC] Drawing PDF iframe loaded:", iframe.src);
  });

  frameWrapNode.appendChild(iframe);

  return iframe;
}

function buildPdfViewerUrlWithoutToolbar(pdfUrl) {
  if (!pdfUrl) {
    return "";
  }

  const cleanUrl = String(pdfUrl).split("#")[0];

  return `${cleanUrl}#toolbar=0&navpanes=0&scrollbar=1&view=FitH`;
}

function wireDrawingPrintButton(printButton, iframe, pdfUrl) {
  if (!printButton || !iframe) {
    return;
  }

  printButton.disabled = false;
  printButton.title = "Print the displayed drawing PDF.";

  printButton.onclick = () => {
    try {
      iframe.contentWindow.focus();
      iframe.contentWindow.print();
    } catch (error) {
      console.warn("[wireDrawingPrintButton] Browser blocked iframe print. Opening PDF directly:", error);
      window.location.href = pdfUrl;
    }
  };
}

function closeDrawingDetailsInPage(goBack) {
  const overlay = document.getElementById("emtac-drawing-viewer-overlay");

  if (overlay) {
    overlay.remove();
  }

  document.body.classList.remove("emtac-drawing-viewer-open");

  if (goBack === true) {
    try {
      if (window.history.state && window.history.state.emtacDrawingViewer === true) {
        window.history.back();
      }
    } catch (err) {
      console.warn("[closeDrawingDetailsInPage] Could not go back:", err);
    }
  }
}


/* ------------------------------------------------------------
   BACK BUTTON SUPPORT
------------------------------------------------------------ */

window.addEventListener("popstate", () => {
  const overlay = document.getElementById("emtac-drawing-viewer-overlay");

  if (overlay) {
    closeDrawingDetailsInPage(false);
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    const overlay = document.getElementById("emtac-drawing-viewer-overlay");

    if (overlay) {
      event.preventDefault();
      closeDrawingDetailsInPage(true);
    }
  }
});


/* ------------------------------------------------------------
   STYLES
------------------------------------------------------------ */

function ensureDrawingViewerStyles() {
  if (el("drawing-ui-styles")) {
    return;
  }

  const style = document.createElement("style");
  style.id = "drawing-ui-styles";

  style.textContent = `
    .drawing-tab-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }

    .drawing-tab-button {
      cursor: pointer;
      border: 1px solid rgba(57, 255, 20, 0.35);
      border-radius: 7px;
      padding: 7px 10px;
      background: rgba(0, 0, 0, 0.25);
      color: #ddd;
      font-weight: 700;
    }

    .drawing-tab-button.active {
      background: rgba(57, 255, 20, 0.20);
      border-color: #39FF14;
      color: #ffffff;
    }

    .drawing-selector-bar {
      margin-bottom: 10px;
    }

    .drawing-filter-info,
    .drawing-filter-prompt {
      color: #aaa;
      font-size: 12px;
      margin: 4px 0 8px 0;
    }

    .drawing-filter-label {
      color: #39FF14;
      font-weight: 700;
      margin: 6px 0;
    }

    .drawing-chip-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-height: 180px;
      overflow-y: auto;
      padding: 4px 0;
    }

    .drawing-chip {
      padding: 6px 10px;
      background: #1e1e1e;
      border: 1px solid #444;
      border-radius: 6px;
      color: #9fef00;
      cursor: pointer;
    }

    .drawing-chip.active {
      background: rgba(57, 255, 20, 0.18);
      border-color: #39FF14;
      color: #ffffff;
    }

    .drawing-result-count {
      color: #aaa;
      font-size: 12px;
      margin-bottom: 8px;
    }

    .drawings-list {
      list-style: none;
      padding-left: 0;
      margin: 0;
    }

    .drawing-item {
      padding: 8px;
      margin-bottom: 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      background: rgba(0, 0, 0, 0.16);
    }

    .drawing-link-button {
      display: block;
      width: 100%;
      text-align: left;
      color: #eee;
      background: transparent;
      border: none;
      padding: 6px 0;
      cursor: pointer;
      font-size: 14px;
    }

    .drawing-link-button:hover {
      color: #ffffff;
      text-decoration: underline;
    }

    .drawing-meta {
      margin-left: 8px;
      margin-top: 4px;
      font-size: 12px;
      color: #bbb;
    }

    .drawing-spares {
      margin-top: 6px;
      padding-left: 10px;
      border-left: 2px solid rgba(57, 255, 20, 0.4);
    }

    .spare-label {
      color: #39FF14;
      font-weight: bold;
      margin-bottom: 2px;
    }

    .spare-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: #ccc;
      padding: 2px 0;
    }

    .spare-row:hover {
      color: #fff;
    }

    .spare-part-number {
      font-weight: bold;
    }

    .emtac-drawing-viewer-open {
      overflow: hidden;
    }

    .emtac-drawing-viewer-overlay {
      position: fixed;
      inset: 0;
      z-index: 999999;
      background: rgba(0, 0, 0, 0.96);
      color: #eee;
      display: flex;
      flex-direction: column;
    }

    .emtac-drawing-viewer-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      width: 100%;
    }

    .emtac-drawing-viewer-header {
      flex: 0 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: rgba(20, 20, 20, 0.98);
      border-bottom: 2px solid #39FF14;
    }

    .emtac-drawing-viewer-header h2 {
      margin: 0;
      color: #39FF14;
      font-size: 18px;
      line-height: 1.3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .emtac-drawing-viewer-buttons {
      display: flex;
      gap: 8px;
      flex: 0 0 auto;
    }

    .emtac-drawing-viewer-button {
      cursor: pointer;
      border: none;
      border-radius: 6px;
      padding: 8px 12px;
      background: #39FF14;
      color: #111;
      font-weight: 700;
    }

    .emtac-drawing-viewer-button:hover:not(:disabled) {
      filter: brightness(1.1);
    }

    .emtac-drawing-viewer-button:disabled {
      cursor: not-allowed;
      background: #333333 !important;
      color: #777777 !important;
      opacity: 0.75;
    }

    .emtac-drawing-viewer-close {
      background: #ff4d4d;
      color: #fff;
    }

    .emtac-drawing-viewer-body {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 14px;
    }

    .emtac-drawing-detail-grid {
      display: grid;
      grid-template-columns: 160px minmax(0, 1fr);
      gap: 8px 12px;
      padding: 12px;
      border: 1px solid rgba(57, 255, 20, 0.25);
      border-radius: 8px;
      background: #151515;
    }

    .emtac-drawing-detail-label {
      color: #39FF14;
      font-weight: 700;
    }

    .emtac-drawing-detail-value {
      color: #eee;
      overflow-wrap: anywhere;
    }

    .emtac-drawing-pdf-section {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid rgba(57, 255, 20, 0.25);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.04);
    }

    .emtac-drawing-pdf-section h3 {
      margin-top: 0;
      margin-bottom: 8px;
      color: #39FF14;
    }

    .emtac-drawing-pdf-status {
      color: #cccccc;
      font-size: 13px;
      margin-bottom: 10px;
      overflow-wrap: anywhere;
    }

    .emtac-drawing-pdf-status-warning {
      color: #ffcc66;
    }

    .emtac-drawing-pdf-frame-wrap {
      width: 100%;
      min-height: 0;
    }

    .emtac-drawing-pdf-frame {
      width: 100%;
      height: 72vh;
      min-height: 520px;
      border: 1px solid rgba(57, 255, 20, 0.35);
      border-radius: 8px;
      background: #000000;
    }

    .emtac-drawing-viewer-section {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.04);
    }

    .emtac-drawing-viewer-section h3 {
      margin-top: 0;
      color: #39FF14;
    }

    .emtac-drawing-viewer-raw {
      margin-top: 14px;
      padding: 12px;
      border-radius: 8px;
      background: #151515;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .emtac-drawing-viewer-raw summary {
      cursor: pointer;
      color: #39FF14;
      font-weight: 700;
    }

    .emtac-drawing-viewer-raw pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #eee;
    }

    @media (max-width: 700px) {
      .emtac-drawing-viewer-header {
        align-items: stretch;
        flex-direction: column;
      }

      .emtac-drawing-viewer-header h2 {
        white-space: normal;
      }

      .emtac-drawing-viewer-buttons {
        width: 100%;
      }

      .emtac-drawing-viewer-button {
        flex: 1 1 auto;
      }

      .emtac-drawing-detail-grid {
        grid-template-columns: 1fr;
      }

      .emtac-drawing-pdf-frame {
        height: 68vh;
        min-height: 420px;
      }
    }
  `;

  document.head.appendChild(style);
}


/* ------------------------------------------------------------
   HELPERS
------------------------------------------------------------ */

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}


/* ------------------------------------------------------------
   GLOBAL EXPORTS
------------------------------------------------------------ */

window.renderDrawings = renderDrawings;
window.updateDrawingsPanelFromBlocks = updateDrawingsPanelFromBlocks;
window.openDrawingDetailsInPage = openDrawingDetailsInPage;
window.closeDrawingDetailsInPage = closeDrawingDetailsInPage;
window.renderAllDrawings = renderAll;