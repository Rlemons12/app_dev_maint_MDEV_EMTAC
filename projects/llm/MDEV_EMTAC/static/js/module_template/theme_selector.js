console.log("[THEME] theme_selector.js loaded");

document.addEventListener("DOMContentLoaded", function () {
    const themeSelect = document.getElementById("theme-select");
    const themeLink = document.getElementById("theme-style");
    const themeContainer = document.querySelector(".sidebar-theme-selector");

    const storageKey = "emtac_selected_theme";

    if (!themeSelect || !themeLink || !themeContainer) {
        console.warn("[THEME] Theme selector setup skipped: required elements not found.");
        return;
    }

    const themeBasePath =
        themeContainer.dataset.themeBaseUrl || "/static/css/module_template/themes/";

    function filenameToThemeClass(themeFile) {
        if (!themeFile) {
            return "";
        }

        let name = themeFile.trim().toLowerCase();

        if (name.endsWith(".css")) {
            name = name.slice(0, -4);
        }

        return name.replace(/_/g, "-");
    }

    function getAvailableThemeClasses() {
        return Array.from(themeSelect.options)
            .map((option) => option.value)
            .filter((value) => value)
            .map((value) => filenameToThemeClass(value));
    }

    function clearExistingThemeClasses() {
        const themeClasses = getAvailableThemeClasses();

        themeClasses.forEach((className) => {
            document.documentElement.classList.remove(className);
            document.body.classList.remove(className);
        });
    }

    function clearTheme(persist = true) {
        themeLink.href = "";
        themeSelect.value = "";
        clearExistingThemeClasses();

        if (persist) {
            localStorage.removeItem(storageKey);
        }

        console.log("[THEME] Theme cleared");
    }

    function applyTheme(themeFile, persist = true) {
        if (!themeFile) {
            clearTheme(persist);
            return;
        }

        const optionExists = Array.from(themeSelect.options).some(
            (option) => option.value === themeFile
        );

        if (!optionExists) {
            console.warn("[THEME] Theme option not found:", themeFile);
            clearTheme(persist);
            return;
        }

        const themeClass = filenameToThemeClass(themeFile);

        clearExistingThemeClasses();

        if (themeClass) {
            document.documentElement.classList.add(themeClass);
            document.body.classList.add(themeClass);
        }

        themeLink.href = `${themeBasePath}${themeFile}`;
        themeSelect.value = themeFile;

        if (persist) {
            localStorage.setItem(storageKey, themeFile);
        }

        console.log("[THEME] Applied theme file:", themeFile);
        console.log("[THEME] Applied theme class:", themeClass);
        console.log("[THEME] Theme href:", themeLink.href);
        console.log("[THEME] HTML class list:", document.documentElement.className);
        console.log("[THEME] Body class list:", document.body.className);
    }

    themeSelect.addEventListener("change", function () {
        applyTheme(this.value, true);
    });

    const savedTheme = localStorage.getItem(storageKey) || "";

    if (savedTheme) {
        applyTheme(savedTheme, false);
    } else {
        clearTheme(false);
    }
});