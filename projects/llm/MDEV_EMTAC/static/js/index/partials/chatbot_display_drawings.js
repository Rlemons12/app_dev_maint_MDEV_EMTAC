/* ============================================================
   EMTAC – Drawing Navigation UI (Area → Model → Asset)
   DROP-IN REPLACEMENT (HOVER SAFE, STATE-DRIVEN)
============================================================ */

/* ------------------------------------------------------------
   STATE
------------------------------------------------------------ */
const drawingState = {
  drawings: [],
  selectedArea: null,
  selectedModels: new Set(),
  selectedAssets: new Set(),
  dropdownPinned: false,
};

const CLICK_DELAY = 220;

/* ------------------------------------------------------------
   SAFE DOM
------------------------------------------------------------ */
const el = id => document.getElementById(id);
const clear = n => n && (n.innerHTML = "");

/* ------------------------------------------------------------
   DATA NORMALIZATION
------------------------------------------------------------ */
function flattenDrawingNavigation(nav) {
  if (!nav?.areas) return [];
  const out = [];

  nav.areas.forEach(a =>
    (a.models || []).forEach(m =>
      (m.assets || []).forEach(s =>
        (s.drawings || []).forEach(d =>
          out.push({
            ...d,
            _area: a.area_name,
            _model: m.model_name,
            _asset: s.asset_name,
          })
        )
      )
    )
  );

  return out;
}

function extractDrawingsFromDocuments(docs = []) {
  const all = [];
  docs.forEach(d => {
    if (d?.drawing_navigation) {
      all.push(...flattenDrawingNavigation(d.drawing_navigation));
    }
  });
  return dedupe(all);
}

function dedupe(list) {
  const seen = new Set();
  return list.filter(d => {
    const k = d.id || d.drw_number || d.url;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/* ------------------------------------------------------------
   SELECTION LOGIC (PRIORITY ORDER)
------------------------------------------------------------ */
function getVisibleDrawings() {
  const s = drawingState;

  if (s.selectedAssets.size) {
    return s.drawings.filter(d => s.selectedAssets.has(d._asset));
  }
  if (s.selectedModels.size) {
    return s.drawings.filter(d => s.selectedModels.has(d._model));
  }
  if (s.selectedArea) {
    return s.drawings.filter(d => d._area === s.selectedArea);
  }
  return [];
}

/* ------------------------------------------------------------
   DROPDOWN CORE (BODY-ATTACHED, FIXED)
------------------------------------------------------------ */
function showDropdown(anchor, rowsBuilder) {
  hideDropdown(true);

  const rect = anchor.getBoundingClientRect();
  const dd = document.createElement("div");
  dd.id = "asset-dropdown";
  dd.className = "asset-dropdown";
  dd.style.position = "fixed";
  dd.style.top = `${rect.bottom + 6}px`;
  dd.style.left = `${rect.left}px`;

  dd.onmouseenter = () => (drawingState.dropdownPinned = true);
  dd.onmouseleave = () => {
    drawingState.dropdownPinned = false;
    hideDropdown();
  };

  rowsBuilder(dd);
  document.body.appendChild(dd);
}

function hideDropdown(force = false) {
  if (!force && drawingState.dropdownPinned) return;
  document.getElementById("asset-dropdown")?.remove();
  drawingState.dropdownPinned = false;
}
let dropdownCloseTimer = null;

/* ------------------------------------------------------------
   AREA BAR (LEVEL 1)
------------------------------------------------------------ */
function renderAreaBar() {
  const bar = el("drawing-area-bar");
  if (!bar) return;
  clear(bar);

  [...new Set(drawingState.drawings.map(d => d._area))].forEach(area => {
    const b = document.createElement("button");
    b.className = "drawing-chip";
    b.textContent = area;

    /* ---------------- ACTIVE HIGHLIGHT ---------------- */
    if (drawingState.selectedArea === area) {
      b.classList.add("active");
    }

    /* ---------------- CLICK (select area) ---------------- */
    b.onclick = () => {
      drawingState.selectedArea = area;
      drawingState.selectedModels.clear();
      drawingState.selectedAssets.clear();
      hideDropdown(true);
      renderAll();
    };

    b.ondblclick = b.onclick;

    /* ---------------- HOVER (preview models) ---------------- */
    b.onmouseenter = () => {
      // cancel pending close
      if (dropdownCloseTimer) {
        clearTimeout(dropdownCloseTimer);
        dropdownCloseTimer = null;
      }

      showDropdown(b, dd => {
        [...new Set(
          drawingState.drawings
            .filter(d => d._area === area)
            .map(d => d._model)
        )].forEach(model => {
          const row = document.createElement("div");
          row.className = "asset-row clickable";
          row.textContent = model;

          /* -------- ACTIVE MODEL HIGHLIGHT -------- */
          if (drawingState.selectedModels.has(model)) {
            row.classList.add("active");
          }

          row.onclick = (e) => {
            e.stopPropagation(); // ⛔ prevent hover-close race

            // Lock context
            drawingState.selectedArea = area;

            // Reset deeper filters
            drawingState.selectedModels.clear();
            drawingState.selectedAssets.clear();

            // Apply model filter
            drawingState.selectedModels.add(model);

            hideDropdown(true);
            renderAll();
          };

          row.ondblclick = row.onclick;

          dd.appendChild(row);
        });
      });
    };

    /* ---------------- SAFE LEAVE (delayed close) ---------------- */
    b.onmouseleave = () => {
      dropdownCloseTimer = setTimeout(() => {
        if (!drawingState.dropdownPinned) {
          hideDropdown();
        }
      }, 180);
    };

    bar.appendChild(b);
  });
}

/* ------------------------------------------------------------
   MODEL BAR (LEVEL 2)
------------------------------------------------------------ */
function renderModelBar() {
  const bar = el("drawing-model-bar");
  if (!bar) return;
  clear(bar);

  if (!drawingState.selectedArea) return;

  const models = [...new Set(
    drawingState.drawings
      .filter(d => d._area === drawingState.selectedArea)
      .map(d => d._model)
  )];

  models.forEach(model => {
    const btn = document.createElement("button");
    btn.className = "drawing-chip model-chip";
    btn.textContent = model;

    /* ---------------- CLICK (select model) ---------------- */
    btn.onclick = e => {
      if (!e.ctrlKey && !e.shiftKey) {
        drawingState.selectedModels.clear();
      }
      drawingState.selectedModels.add(model);
      drawingState.selectedAssets.clear();
      hideDropdown(true);
      renderDrawings(getVisibleDrawings());
    };

    btn.ondblclick = btn.onclick;

    /* ---------------- HOVER (show assets) ---------------- */
    btn.onmouseenter = () => {
      // cancel any pending close
      if (dropdownCloseTimer) {
        clearTimeout(dropdownCloseTimer);
        dropdownCloseTimer = null;
      }

      showDropdown(btn, dd => {
        [...new Set(
          drawingState.drawings
            .filter(d => d._model === model)
            .map(d => d._asset)
        )].forEach(asset => {
          const row = document.createElement("label");
          row.className = "asset-row";

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = drawingState.selectedAssets.has(asset);

          cb.onchange = () => {
            cb.checked
              ? drawingState.selectedAssets.add(asset)
              : drawingState.selectedAssets.delete(asset);
            renderDrawings(getVisibleDrawings());
          };

          row.append(cb, document.createTextNode(asset));
          dd.appendChild(row);
        });

        // 🔑 KEEP DROPDOWN OPEN WHILE HOVERING IT
        dd.onmouseenter = () => {
          drawingState.dropdownPinned = true;
          if (dropdownCloseTimer) {
            clearTimeout(dropdownCloseTimer);
            dropdownCloseTimer = null;
          }
        };

        dd.onmouseleave = () => {
          drawingState.dropdownPinned = false;
          dropdownCloseTimer = setTimeout(() => {
            hideDropdown();
          }, 180);
        };
      });
    };

    /* ---------------- SAFE LEAVE (delayed close) ---------------- */
    btn.onmouseleave = () => {
      dropdownCloseTimer = setTimeout(() => {
        if (!drawingState.dropdownPinned) {
          hideDropdown();
        }
      }, 180);
    };

    bar.appendChild(btn);
  });
}

/* ------------------------------------------------------------
   DRAWINGS PANEL (LEVEL 4)
------------------------------------------------------------ */
function renderDrawings(drawings) {
  const box = el("drawing-section");
  if (!box) return;
  clear(box);

  if (!drawings.length) {
    box.innerHTML = "<p>No drawings found.</p>";
    return;
  }

  const ul = document.createElement("ul");
  ul.className = "drawings-list";

  drawings.forEach(d => {
    const li = document.createElement("li");
    li.className = "drawing-item";

    const link = document.createElement("a");
    link.href = "javascript:void(0)";
    link.innerHTML = `
      📐 <strong>${d.drw_name || "Untitled"}</strong>
    `;

    // 🔑 THIS IS THE IMPORTANT PART
    link.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      window.openDrawingDetails(d);
    };

    link.ondblclick = link.onclick;

    const meta = document.createElement("div");
meta.className = "drawing-meta";

meta.innerHTML = `<div><b>No:</b> ${d.drw_number || "—"}</div>`;

// --------------------------------------------------
// 🔧 Spare Parts (UNDER drawing number)
// --------------------------------------------------
if (Array.isArray(d.spare_parts) && d.spare_parts.length > 0) {
  const sp = document.createElement("div");
  sp.className = "drawing-spares";

  const label = document.createElement("div");
  label.className = "spare-label";
  label.textContent = "Spare Parts:";
  sp.appendChild(label);

  d.spare_parts.forEach(p => {
    const row = document.createElement("div");
    row.className = "spare-row";

    row.innerHTML = `
      <span class="spare-part-number">${p.part_number || "—"}</span>
      <span class="spare-part-name">${p.name || ""}</span>
    `;

    // (Optional future hook)
    // row.onclick = () => openEntityDetailsViewer({ title: p.part_number, parts: [p] });

    sp.appendChild(row);
  });

  meta.appendChild(sp);
}

li.appendChild(link);
li.appendChild(meta);
ul.appendChild(li);

  });

  box.appendChild(ul);
}

/* ------------------------------------------------------------
   ENTRY POINT
------------------------------------------------------------ */
window.updateDrawingsPanelFromBlocks = blocks => {
  drawingState.drawings = extractDrawingsFromDocuments(
    blocks?.["documents-container"] || []
  );

  drawingState.selectedArea = null;
  drawingState.selectedModels.clear();
  drawingState.selectedAssets.clear();
  hideDropdown(true);

  renderAll();
};

/* ------------------------------------------------------------
   MASTER RENDER
------------------------------------------------------------ */
function renderAll() {
  renderAreaBar();
  renderModelBar();
  renderDrawings(getVisibleDrawings());
}

/* ------------------------------------------------------------
   STYLES (SAFE)
------------------------------------------------------------ */
(() => {
  if (el("drawing-ui-styles")) return;
  const s = document.createElement("style");
  s.id = "drawing-ui-styles";
  s.textContent = `
    .drawing-chip {
      padding: 6px 10px;
      margin: 4px;
      background: #1e1e1e;
      border: 1px solid #444;
      border-radius: 6px;
      color: #9fef00;
      cursor: pointer;
    }
    .asset-dropdown {
      background: #111;
      border: 1px solid #39FF14;
      border-radius: 8px;
      padding: 6px;
      z-index: 99999;
      min-width: 220px;
    }
    .asset-row {
      display: flex;
      gap: 6px;
      padding: 4px;
      color: #ddd;
      cursor: pointer;
      white-space: nowrap;
    }
    .asset-row:hover {
      background: rgba(57,255,20,0.12);
    }
/* 🔑 DRAWING LINKS — NO BLUE */
    .drawing-item a {
      color: #eee;
      text-decoration: none;
}
.drawing-item a:hover {
  text-decoration: underline;
}
.drawing-meta {
  margin-left: 18px;
  margin-top: 4px;
  font-size: 12px;
  color: #bbb;
}

.drawing-spares {
  margin-top: 6px;
  padding-left: 10px;
  border-left: 2px solid rgba(57,255,20,0.4);
}

.spare-label {
  color: #39FF14;
  font-weight: bold;
  margin-bottom: 2px;
}

.spare-row {
  display: flex;
  gap: 6px;
  cursor: pointer;
  color: #ccc;
}

.spare-row:hover {
  color: #fff;
}

.spare-part-number {
  font-weight: bold;
}

  `;
  document.head.appendChild(s);
})();
