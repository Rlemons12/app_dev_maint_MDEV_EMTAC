/* static/js/module_template/base_sidebar.js
==================================
EMTAC BASE SIDEBAR CONTROLLER
Handles:
- Desktop sidebar collapse
- Mobile sidebar open/close
- Persisted sidebar state
- ARIA state updates
================================== */

console.log("[SIDEBAR] base_sidebar.js loaded");

document.addEventListener("DOMContentLoaded", function () {
    const collapseButton = document.getElementById("sidebarCollapse");
    const sidebar = document.getElementById("mainSidebar");
    const content = document.querySelector(".content");

    const storageKey = "emtac_sidebar_collapsed";
    const mobileBreakpoint = 768;

    if (!collapseButton || !sidebar) {
        console.warn("[SIDEBAR] Required sidebar elements not found.");
        return;
    }

    function isMobileView() {
        return window.innerWidth <= mobileBreakpoint;
    }

    function setAriaExpanded(expanded) {
        collapseButton.setAttribute("aria-expanded", expanded ? "true" : "false");
    }

    function applyDesktopState(isCollapsed, persist = true) {
        if (isCollapsed) {
            document.body.classList.add("sidebar-is-collapsed");
            sidebar.classList.add("collapsed");

            if (content) {
                content.classList.add("sidebar-collapsed");
            }
        } else {
            document.body.classList.remove("sidebar-is-collapsed");
            sidebar.classList.remove("collapsed");

            if (content) {
                content.classList.remove("sidebar-collapsed");
            }
        }

        setAriaExpanded(!isCollapsed);

        if (persist) {
            localStorage.setItem(storageKey, isCollapsed ? "true" : "false");
        }

        console.log("[SIDEBAR] Desktop state applied:", {
            collapsed: isCollapsed,
            bodyClasses: document.body.className,
            sidebarClasses: sidebar.className,
            contentClasses: content ? content.className : "(no content element)"
        });
    }

    function applyMobileState(isOpen) {
        if (isOpen) {
            sidebar.classList.add("active");
            sidebar.classList.remove("collapsed");
            document.body.classList.remove("sidebar-is-collapsed");
        } else {
            sidebar.classList.remove("active");
            sidebar.classList.add("collapsed");
        }

        setAriaExpanded(isOpen);

        console.log("[SIDEBAR] Mobile state applied:", {
            open: isOpen,
            sidebarClasses: sidebar.className
        });
    }

    function initializeSidebarState() {
        const savedCollapsed = localStorage.getItem(storageKey) === "true";

        if (isMobileView()) {
            sidebar.classList.remove("active");
            sidebar.classList.add("collapsed");
            document.body.classList.remove("sidebar-is-collapsed");

            if (content) {
                content.classList.remove("sidebar-collapsed");
            }

            setAriaExpanded(false);

            console.log("[SIDEBAR] Initialized mobile state");
            return;
        }

        applyDesktopState(savedCollapsed, false);
        console.log("[SIDEBAR] Initialized desktop state from storage:", savedCollapsed);
    }

    collapseButton.addEventListener("click", function () {
        if (isMobileView()) {
            const isOpen = sidebar.classList.contains("active");
            applyMobileState(!isOpen);
            return;
        }

        const isCollapsed = document.body.classList.contains("sidebar-is-collapsed");
        applyDesktopState(!isCollapsed, true);
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && isMobileView() && sidebar.classList.contains("active")) {
            applyMobileState(false);
        }
    });

    window.addEventListener("resize", function () {
        initializeSidebarState();
    });

    initializeSidebarState();
});