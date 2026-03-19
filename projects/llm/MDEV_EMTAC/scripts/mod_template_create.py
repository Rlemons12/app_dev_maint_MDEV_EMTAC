from __future__ import annotations

from pathlib import Path
from typing import Dict


BASE_PROJECT_DIR = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")
TEMPLATES_DIR = BASE_PROJECT_DIR / "templates" / "module_template_html" / "partials"
CSS_DIR = BASE_PROJECT_DIR / "static" / "css" / "module_template"
JS_DIR = BASE_PROJECT_DIR / "static" / "js" / "module_template"


PANEL_LAYOUTS_CSS = """/* static/css/module_template/panel_layouts.css
==================================
STANDARDIZED MODULE PANEL LAYOUTS
Shared layout system for module template partials
================================== */

/* ==================================
BASE SHELL
================================== */
.mt-layout {
    padding: 20px;
    height: 100%;
    overflow-y: auto;
    background-color: inherit;
    box-sizing: border-box;
}

.mt-layout-header {
    margin-bottom: 20px;
    text-align: center;
    border-bottom: 2px solid rgba(57, 255, 20, 0.3);
    padding-bottom: 15px;
}

.mt-layout-title {
    margin: 0 0 8px 0;
    color: #39FF14;
    font-size: 24px;
    font-weight: bold;
}

.mt-layout-description {
    margin: 0;
    color: #ccc;
    font-size: 14px;
    font-style: italic;
}

/* ==================================
GRID
================================== */
.mt-grid {
    display: grid;
    gap: 15px;
    height: calc(100% - 120px);
    min-height: 400px;
}

.mt-grid--2 {
    grid-template-columns: 1fr 1fr;
}

.mt-grid--3-bottom-span {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
}

.mt-grid--4 {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
}

.mt-grid--5-main-left {
    grid-template-columns: 2fr 1fr;
    grid-template-rows: repeat(4, 1fr);
}

.mt-grid--6 {
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: 1fr 1fr;
}

/* ==================================
PANELS
================================== */
.mt-panel {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.3s ease;
}

.mt-panel:hover {
    border-color: rgba(57, 255, 20, 0.5);
    box-shadow: 0 2px 8px rgba(57, 255, 20, 0.2);
}

.mt-panel--bottom-span {
    grid-column: 1 / -1;
}

.mt-panel--main {
    grid-row: 1 / -1;
}

.mt-panel-header {
    background-color: rgba(0, 0, 0, 0.3);
    padding: 12px 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}

.mt-panel-title {
    margin: 0;
    color: #39FF14;
    font-size: 16px;
    font-weight: bold;
    text-transform: uppercase;
}

.mt-panel-body {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
}

.mt-panel.is-collapsed .mt-panel-body {
    display: none;
}

/* ==================================
BUTTONS
================================== */
.mt-toggle-btn {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 6px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.3s ease;
}

.mt-toggle-btn:hover {
    background: #39FF14;
    color: black;
    border-color: #39FF14;
}

.mt-panel.is-collapsed .mt-toggle-btn {
    background: #ff4444;
    color: white;
    border-color: #ff4444;
}

/* ==================================
INNER CONTENT
================================== */
.mt-panel-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 100%;
}

.mt-item {
    background-color: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 12px;
    color: white;
    font-size: 14px;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

.mt-item:hover {
    background-color: rgba(57, 255, 20, 0.1);
    border-color: #39FF14;
}

.mt-item--small {
    padding: 8px;
    font-size: 12px;
}

.mt-item--wide {
    flex: 1;
}

.mt-item.is-active {
    background-color: rgba(57, 255, 20, 0.2);
    border-color: #39FF14;
}

/* ==================================
RESPONSIVE
================================== */
@media (max-width: 1024px) {
    .mt-grid--2,
    .mt-grid--4 {
        grid-template-columns: 1fr;
    }

    .mt-grid--3-bottom-span {
        grid-template-columns: 1fr;
        grid-template-rows: repeat(3, 1fr);
    }

    .mt-grid--3-bottom-span .mt-panel--bottom-span {
        grid-column: 1;
    }

    .mt-grid--6 {
        grid-template-columns: repeat(2, 1fr);
        grid-template-rows: repeat(3, 1fr);
    }
}

@media (max-width: 768px) {
    .mt-layout {
        padding: 10px;
    }

    .mt-grid {
        gap: 10px;
    }

    .mt-grid--5-main-left,
    .mt-grid--6 {
        grid-template-columns: 1fr;
        grid-template-rows: auto;
    }

    .mt-panel--main {
        grid-row: auto;
    }
}
"""


PANEL_LAYOUTS_JS = """/* static/js/module_template/panel_layouts.js
==================================
STANDARDIZED MODULE PANEL LAYOUTS JS
Shared behavior for module template partials
================================== */

document.addEventListener("DOMContentLoaded", () => {
    const layouts = document.querySelectorAll("[data-mt-layout]");

    layouts.forEach((layout) => {
        const items = layout.querySelectorAll(".mt-item");
        const toggles = layout.querySelectorAll(".mt-toggle-btn");

        items.forEach((item) => {
            item.addEventListener("click", () => {
                item.classList.add("is-active");
                window.setTimeout(() => {
                    item.classList.remove("is-active");
                }, 1000);
            });
        });

        toggles.forEach((button) => {
            button.addEventListener("click", () => {
                const panel = button.closest(".mt-panel");
                if (!panel) {
                    return;
                }
                panel.classList.toggle("is-collapsed");
            });
        });
    });
});
"""


LAYOUT_PANELS_2 = """<div class="mt-layout" data-mt-layout="panels-2">
    <div class="mt-layout-header">
        <h1 class="mt-layout-title">Two Panel Layout</h1>
        <p class="mt-layout-description">Two equal containers side by side</p>
    </div>

    <div class="mt-grid mt-grid--2">
        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Left Container</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Left Container">⚙️</button>
            </div>
            <div class="mt-panel-body">
                <p>two_container_template left content</p>
                <div class="mt-panel-content">
                    <div class="mt-item">two_container_template item 1</div>
                    <div class="mt-item">two_container_template item 2</div>
                    <div class="mt-item">two_container_template item 3</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Right Container</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Right Container">📊</button>
            </div>
            <div class="mt-panel-body">
                <p>two_container_template right content</p>
                <div class="mt-panel-content">
                    <div class="mt-item">two_container_template data A</div>
                    <div class="mt-item">two_container_template data B</div>
                    <div class="mt-item">two_container_template data C</div>
                </div>
            </div>
        </section>
    </div>
</div>
"""


LAYOUT_PANELS_3_BOTTOM_SPAN = """<div class="mt-layout" data-mt-layout="panels-3-bottom-span">
    <div class="mt-layout-header">
        <h1 class="mt-layout-title">Three Panel Layout</h1>
        <p class="mt-layout-description">Two containers on top, one spanning bottom</p>
    </div>

    <div class="mt-grid mt-grid--3-bottom-span">
        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Top Left</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Top Left">📈</button>
            </div>
            <div class="mt-panel-body">
                <p>three_container_template top left</p>
                <div class="mt-panel-content">
                    <div class="mt-item">three_container_template chart area</div>
                    <div class="mt-item">three_container_template statistics</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Top Right</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Top Right">🎯</button>
            </div>
            <div class="mt-panel-body">
                <p>three_container_template top right</p>
                <div class="mt-panel-content">
                    <div class="mt-item">three_container_template controls</div>
                    <div class="mt-item">three_container_template settings</div>
                </div>
            </div>
        </section>

        <section class="mt-panel mt-panel--bottom-span">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Bottom Container (Full Width)</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Bottom Container">📋</button>
            </div>
            <div class="mt-panel-body">
                <p>three_container_template bottom full width</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--wide">three_container_template data table / list view</div>
                    <div class="mt-item">three_container_template actions</div>
                </div>
            </div>
        </section>
    </div>
</div>
"""


LAYOUT_PANELS_4_GRID = """<div class="mt-layout" data-mt-layout="panels-4">
    <div class="mt-layout-header">
        <h1 class="mt-layout-title">Four Panel Grid</h1>
        <p class="mt-layout-description">Four equal containers in 2x2 grid</p>
    </div>

    <div class="mt-grid mt-grid--4">
        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Top Left</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Top Left">📊</button>
            </div>
            <div class="mt-panel-body">
                <p>four_container_template top left</p>
                <div class="mt-panel-content">
                    <div class="mt-item">four_container_template dashboard A</div>
                    <div class="mt-item mt-item--small">four_container_template metrics</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Top Right</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Top Right">📈</button>
            </div>
            <div class="mt-panel-body">
                <p>four_container_template top right</p>
                <div class="mt-panel-content">
                    <div class="mt-item">four_container_template dashboard B</div>
                    <div class="mt-item mt-item--small">four_container_template analytics</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Bottom Left</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Bottom Left">🔧</button>
            </div>
            <div class="mt-panel-body">
                <p>four_container_template bottom left</p>
                <div class="mt-panel-content">
                    <div class="mt-item">four_container_template tools panel</div>
                    <div class="mt-item mt-item--small">four_container_template controls</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Bottom Right</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Bottom Right">📋</button>
            </div>
            <div class="mt-panel-body">
                <p>four_container_template bottom right</p>
                <div class="mt-panel-content">
                    <div class="mt-item">four_container_template data view</div>
                    <div class="mt-item mt-item--small">four_container_template details</div>
                </div>
            </div>
        </section>
    </div>
</div>
"""


LAYOUT_PANELS_5_MAIN_LEFT = """<div class="mt-layout" data-mt-layout="panels-5-main-left">
    <div class="mt-layout-header">
        <h1 class="mt-layout-title">Five Panel Layout</h1>
        <p class="mt-layout-description">One large container with four smaller panels</p>
    </div>

    <div class="mt-grid mt-grid--5-main-left">
        <section class="mt-panel mt-panel--main">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Main Content Area</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Main Content Area">🎛️</button>
            </div>
            <div class="mt-panel-body">
                <p>five_container_template main content</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--wide">five_container_template primary interface</div>
                    <div class="mt-item">five_container_template secondary data</div>
                    <div class="mt-item">five_container_template additional info</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel 1</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel 1">📊</button>
            </div>
            <div class="mt-panel-body">
                <p>five_container_template panel 1</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--small">five_container_template widget A</div>
                    <div class="mt-item mt-item--small">five_container_template status</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel 2</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel 2">📈</button>
            </div>
            <div class="mt-panel-body">
                <p>five_container_template panel 2</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--small">five_container_template widget B</div>
                    <div class="mt-item mt-item--small">five_container_template metrics</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel 3</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel 3">🔧</button>
            </div>
            <div class="mt-panel-body">
                <p>five_container_template panel 3</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--small">five_container_template widget C</div>
                    <div class="mt-item mt-item--small">five_container_template tools</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel 4</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel 4">📋</button>
            </div>
            <div class="mt-panel-body">
                <p>five_container_template panel 4</p>
                <div class="mt-panel-content">
                    <div class="mt-item mt-item--small">five_container_template widget D</div>
                    <div class="mt-item mt-item--small">five_container_template actions</div>
                </div>
            </div>
        </section>
    </div>
</div>
"""


LAYOUT_PANELS_6_GRID = """<div class="mt-layout" data-mt-layout="panels-6">
    <div class="mt-layout-header">
        <h1 class="mt-layout-title">Six Panel Grid</h1>
        <p class="mt-layout-description">Six containers in 3x2 grid formation</p>
    </div>

    <div class="mt-grid mt-grid--6">
        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel A</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel A">📊</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel A</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data A</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel B</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel B">📈</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel B</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data B</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel C</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel C">🎯</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel C</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data C</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel D</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel D">🔧</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel D</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data D</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel E</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel E">📋</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel E</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data E</div>
                </div>
            </div>
        </section>

        <section class="mt-panel">
            <div class="mt-panel-header">
                <h3 class="mt-panel-title">Panel F</h3>
                <button class="mt-toggle-btn" type="button" aria-label="Toggle Panel F">⚙️</button>
            </div>
            <div class="mt-panel-body">
                <p>six_container_template panel F</p>
                <div class="mt-panel-content">
                    <div class="mt-item">six_container_template data F</div>
                </div>
            </div>
        </section>
    </div>
</div>
"""


def ensure_directories() -> None:
    """Create required output directories."""
    for directory in (TEMPLATES_DIR, CSS_DIR, JS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_file_map() -> Dict[Path, str]:
    """Return all files to create."""
    return {
        CSS_DIR / "panel_layouts.css": PANEL_LAYOUTS_CSS,
        JS_DIR / "panel_layouts.js": PANEL_LAYOUTS_JS,
        TEMPLATES_DIR / "layout_panels_2.html": LAYOUT_PANELS_2,
        TEMPLATES_DIR / "layout_panels_3_bottom_span.html": LAYOUT_PANELS_3_BOTTOM_SPAN,
        TEMPLATES_DIR / "layout_panels_4_grid.html": LAYOUT_PANELS_4_GRID,
        TEMPLATES_DIR / "layout_panels_5_main_left.html": LAYOUT_PANELS_5_MAIN_LEFT,
        TEMPLATES_DIR / "layout_panels_6_grid.html": LAYOUT_PANELS_6_GRID,
    }


def write_files(file_map: Dict[Path, str]) -> None:
    """Write all generated files to disk."""
    for file_path, content in file_map.items():
        file_path.write_text(content.strip() + "\\n", encoding="utf-8")
        print(f"Created: {file_path}")


def main() -> None:
    """Main entrypoint."""
    ensure_directories()
    file_map = build_file_map()
    write_files(file_map)
    print("\\nModule template layout files created successfully.")


if __name__ == "__main__":
    main()