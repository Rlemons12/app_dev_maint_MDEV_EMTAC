console.log("[SIDEBAR] base_sidebar.js loaded");
document.addEventListener("DOMContentLoaded", function () {
    const sidebarToggleBtn = document.getElementById("sidebarCollapse");
    const sidebar = document.getElementById("mainSidebar");
    const content = document.querySelector(".content");
    const toggleVoiceBtn = document.getElementById("toggle-voice");
    const toggleTextToSpeechBtn = document.getElementById("toggle-text-to-speech");

    function logSidebarState(context) {
        if (!sidebar) {
            console.warn(`[SIDEBAR] ${context} | sidebar not found`);
            return;
        }

        const computedStyle = window.getComputedStyle(sidebar);

        console.log(`[SIDEBAR] ${context}`);
        console.log("  window.innerWidth:", window.innerWidth);
        console.log("  isMobileView:", isMobileView());
        console.log("  body classes:", document.body.className);
        console.log("  sidebar classes:", sidebar.className);
        console.log("  sidebar transform:", computedStyle.transform);
        console.log("  sidebar width:", computedStyle.width);
        console.log("  sidebar left:", computedStyle.left);

        if (content) {
            const contentStyle = window.getComputedStyle(content);
            console.log("  content classes:", content.className);
            console.log("  content margin-left:", contentStyle.marginLeft);
            console.log("  content width:", contentStyle.width);
        } else {
            console.log("  content not found");
        }

        if (sidebarToggleBtn) {
            console.log("  hamburger aria-expanded:", sidebarToggleBtn.getAttribute("aria-expanded"));
        } else {
            console.log("  sidebar toggle button not found");
        }
    }

    function isMobileView() {
        return window.innerWidth <= 768;
    }

    function setDesktopCollapsedState(collapsed) {
        document.body.classList.toggle("sidebar-is-collapsed", collapsed);

        if (sidebarToggleBtn) {
            sidebarToggleBtn.setAttribute("aria-expanded", String(!collapsed));
        }

        logSidebarState(`setDesktopCollapsedState(${collapsed})`);
    }

    function setMobileSidebarState(open) {
        if (!sidebar) {
            console.warn("[SIDEBAR] setMobileSidebarState called but sidebar not found");
            return;
        }

        sidebar.classList.toggle("active", open);

        if (sidebarToggleBtn) {
            sidebarToggleBtn.setAttribute("aria-expanded", String(open));
        }

        logSidebarState(`setMobileSidebarState(${open})`);
    }

    function toggleSidebar() {
        console.log("[SIDEBAR] toggleSidebar called");

        if (!sidebar) {
            console.warn("[SIDEBAR] Sidebar not found");
            return;
        }

        if (isMobileView()) {
            const willOpen = !sidebar.classList.contains("active");
            console.log("[SIDEBAR] Mobile toggle | willOpen =", willOpen);
            setMobileSidebarState(willOpen);
            return;
        }

        const willCollapse = !document.body.classList.contains("sidebar-is-collapsed");
        console.log("[SIDEBAR] Desktop toggle | willCollapse =", willCollapse);
        setDesktopCollapsedState(willCollapse);
    }

    function resetSidebarForViewport() {
        console.log("[SIDEBAR] resetSidebarForViewport called");

        if (!sidebar) {
            console.warn("[SIDEBAR] resetSidebarForViewport | sidebar not found");
            return;
        }

        if (isMobileView()) {
            document.body.classList.remove("sidebar-is-collapsed");
            sidebar.classList.remove("active");

            if (sidebarToggleBtn) {
                sidebarToggleBtn.setAttribute("aria-expanded", "false");
            }

            logSidebarState("resetSidebarForViewport -> mobile");
        } else {
            sidebar.classList.remove("active");

            if (sidebarToggleBtn) {
                sidebarToggleBtn.setAttribute(
                    "aria-expanded",
                    String(!document.body.classList.contains("sidebar-is-collapsed"))
                );
            }

            logSidebarState("resetSidebarForViewport -> desktop");
        }
    }

    console.log("[SIDEBAR] DOMContentLoaded");
    console.log("[SIDEBAR] sidebarToggleBtn found:", !!sidebarToggleBtn);
    console.log("[SIDEBAR] sidebar found:", !!sidebar);
    console.log("[SIDEBAR] content found:", !!content);

    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener("click", function (event) {
            console.log("[SIDEBAR] hamburger click detected");
            event.preventDefault();
            event.stopPropagation();
            toggleSidebar();
        });
    } else {
        console.warn("[SIDEBAR] sidebarCollapse button not found");
    }

    document.addEventListener("click", function (event) {
        if (!isMobileView() || !sidebar) {
            return;
        }

        const clickedToggle = sidebarToggleBtn && (
            event.target === sidebarToggleBtn || sidebarToggleBtn.contains(event.target)
        );

        if (!sidebar.contains(event.target) && !clickedToggle) {
            console.log("[SIDEBAR] outside click on mobile -> closing sidebar");
            setMobileSidebarState(false);
        }
    });

    if (toggleVoiceBtn) {
        toggleVoiceBtn.addEventListener("click", function () {
            this.classList.toggle("active");
            console.log("[SIDEBAR] Voice toggle clicked");
        });
    }

    if (toggleTextToSpeechBtn) {
        toggleTextToSpeechBtn.addEventListener("click", function () {
            this.classList.toggle("active");
            console.log("[SIDEBAR] Text-to-speech toggle clicked");
        });
    }

    window.showForm = function (formId) {
        console.log("[SIDEBAR] showForm called with:", formId);

        const forms = document.querySelectorAll(".form-container");
        forms.forEach((form) => {
            form.style.display = "none";
        });

        const resultsContainer = document.getElementById("results-container");
        if (resultsContainer) {
            resultsContainer.style.display = formId.includes("search") ? "block" : "none";
        }

        const selectedForm = document.getElementById(formId);
        if (selectedForm) {
            selectedForm.style.display = "block";

            if (isMobileView()) {
                selectedForm.scrollIntoView({ behavior: "smooth" });
                setMobileSidebarState(false);
            }
        } else {
            console.warn("[SIDEBAR] showForm could not find form:", formId);
        }
    };

    initializeVoiceSelection();
    resetSidebarForViewport();

    window.addEventListener("resize", function () {
        console.log("[SIDEBAR] resize event");
        resetSidebarForViewport();
    });
});

function initializeVoiceSelection() {
    const voiceSelection = document.getElementById("voice-selection");

    if (!voiceSelection || !window.speechSynthesis) {
        console.log("[SIDEBAR] Voice selection not initialized");
        return;
    }

    function populateVoiceList() {
        const voices = window.speechSynthesis.getVoices();

        voiceSelection.innerHTML = "";

        const defaultOption = document.createElement("option");
        defaultOption.textContent = "Select Voice";
        defaultOption.value = "";
        voiceSelection.appendChild(defaultOption);

        voices.forEach((voice) => {
            const option = document.createElement("option");
            option.textContent = `${voice.name} (${voice.lang})`;
            option.value = voice.name;
            voiceSelection.appendChild(option);
        });

        const savedVoice = localStorage.getItem("selectedVoice");
        if (savedVoice) {
            for (let i = 0; i < voiceSelection.options.length; i += 1) {
                if (voiceSelection.options[i].value === savedVoice) {
                    voiceSelection.selectedIndex = i;
                    break;
                }
            }
        }

        console.log("[SIDEBAR] Voice list populated");
    }

    populateVoiceList();

    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = populateVoiceList;
    }

    voiceSelection.addEventListener("change", function () {
        if (this.value) {
            localStorage.setItem("selectedVoice", this.value);
            console.log("[SIDEBAR] Selected voice:", this.value);
        }
    });
}