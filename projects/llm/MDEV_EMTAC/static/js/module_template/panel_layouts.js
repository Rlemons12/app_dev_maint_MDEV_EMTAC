/* static/js/module_template/panel_layouts.js
==================================
STANDARDIZED MODULE PANEL LAYOUTS JS
UPGRADED SIDE-COLLAPSE VERSION
================================== */

document.addEventListener("DOMContentLoaded", () => {
    console.log("[PANELS] panel_layouts.js loaded");

    const layouts = document.querySelectorAll("[data-mt-layout]");
    console.log("[PANELS] layouts found:", layouts.length);

    function getTwoPanelGrid(layout) {
        return layout.querySelector(".mt-grid--2");
    }

    function getDirectPanels(grid) {
        if (!grid) {
            return [];
        }
        return Array.from(grid.children).filter((child) =>
            child.classList.contains("mt-panel")
        );
    }

    function isMobileStackedLayout() {
        return window.innerWidth <= 1024;
    }

    function clearTwoPanelStateClasses(grid) {
        if (!grid) {
            return;
        }

        grid.classList.remove(
            "mt-grid--left-collapsed",
            "mt-grid--right-collapsed",
            "mt-grid--both-collapsed"
        );
    }

    function updateToggleAccessibility(panel) {
        if (!panel) {
            return;
        }

        const button = panel.querySelector(".mt-toggle-btn");
        const body = panel.querySelector(".mt-panel-body");
        const isCollapsed = panel.classList.contains("is-collapsed");

        if (button) {
            button.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
            button.setAttribute(
                "aria-label",
                isCollapsed ? "Expand panel" : "Collapse panel"
            );
            button.title = isCollapsed ? "Expand panel" : "Collapse panel";
        }

        if (body && body.id && button) {
            button.setAttribute("aria-controls", body.id);
        }
    }

    function updateTwoPanelLayoutState(layout) {
        const grid = getTwoPanelGrid(layout);
        if (!grid) {
            return;
        }

        const panels = getDirectPanels(grid);
        if (panels.length < 2) {
            clearTwoPanelStateClasses(grid);
            return;
        }

        const leftPanel = panels[0];
        const rightPanel = panels[1];

        const leftCollapsed = leftPanel.classList.contains("is-collapsed");
        const rightCollapsed = rightPanel.classList.contains("is-collapsed");

        clearTwoPanelStateClasses(grid);

        if (isMobileStackedLayout()) {
            return;
        }

        if (leftCollapsed && rightCollapsed) {
            grid.classList.add("mt-grid--both-collapsed");
        } else if (leftCollapsed) {
            grid.classList.add("mt-grid--left-collapsed");
        } else if (rightCollapsed) {
            grid.classList.add("mt-grid--right-collapsed");
        }
    }

    function setPanelCollapsedState(layout, panel, shouldCollapse) {
        if (!panel) {
            return;
        }

        panel.classList.toggle("is-collapsed", shouldCollapse);
        updateToggleAccessibility(panel);
        updateTwoPanelLayoutState(layout);

        console.log("[PANELS] panel state updated", {
            panelId: panel.id || "(no id)",
            collapsed: shouldCollapse
        });
    }

    function togglePanelCollapsedState(layout, panel) {
        if (!panel) {
            return;
        }

        const isCurrentlyCollapsed = panel.classList.contains("is-collapsed");
        setPanelCollapsedState(layout, panel, !isCurrentlyCollapsed);
    }

    function bindItemClicks(layout) {
        const items = layout.querySelectorAll(".mt-item");

        items.forEach((item) => {
            if (item.dataset.mtItemBound === "true") {
                return;
            }

            item.dataset.mtItemBound = "true";

            item.addEventListener("click", () => {
                item.classList.add("is-active");

                window.setTimeout(() => {
                    item.classList.remove("is-active");
                }, 1000);
            });
        });
    }

    function bindPanelToggles(layout) {
        const toggles = layout.querySelectorAll(".mt-toggle-btn");

        toggles.forEach((button) => {
            if (button.dataset.mtToggleBound === "true") {
                return;
            }

            button.dataset.mtToggleBound = "true";

            const panel = button.closest(".mt-panel");
            if (!panel) {
                console.warn("[PANELS] toggle button has no parent panel", button);
                return;
            }

            updateToggleAccessibility(panel);

            button.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();

                console.log("[PANELS] toggle clicked", {
                    panelId: panel.id || "(no id)"
                });

                togglePanelCollapsedState(layout, panel);
            });
        });
    }

    function initializeLayout(layout) {
        bindItemClicks(layout);
        bindPanelToggles(layout);

        const panels = layout.querySelectorAll(".mt-panel");
        panels.forEach((panel) => {
            updateToggleAccessibility(panel);
        });

        updateTwoPanelLayoutState(layout);
    }

    layouts.forEach((layout) => {
        initializeLayout(layout);
    });

    let resizeTimer = null;

    window.addEventListener("resize", () => {
        window.clearTimeout(resizeTimer);

        resizeTimer = window.setTimeout(() => {
            layouts.forEach((layout) => {
                updateTwoPanelLayoutState(layout);
            });
        }, 100);
    });
});