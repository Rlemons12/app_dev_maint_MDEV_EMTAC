document.addEventListener("DOMContentLoaded", function () {
    const themeSelect = document.getElementById("theme-select");
    const themeLink = document.getElementById("theme-style");
    const themeContainer = document.querySelector(".sidebar-theme-selector");
    const storageKey = "selectedTheme";

    if (!themeSelect || !themeLink || !themeContainer) {
        console.warn("Theme selector setup skipped: required elements not found.");
        return;
    }

    const themeBasePath =
        themeContainer.dataset.themeBaseUrl || "/static/css/module_template/themes/";

    function filenameToBodyClass(themeFile) {
        if (!themeFile) {
            return "";
        }

        let name = themeFile.trim().toLowerCase();

        if (name.endsWith(".css")) {
            name = name.slice(0, -4);
        }

        return name.replace(/_/g, "-");
    }

    function getAvailableThemeBodyClasses() {
        return Array.from(themeSelect.options)
            .map((option) => option.value)
            .filter((value) => value)
            .map((value) => filenameToBodyClass(value));
    }

    function clearExistingThemeBodyClasses() {
        const themeClasses = getAvailableThemeBodyClasses();
        themeClasses.forEach((className) => {
            document.body.classList.remove(className);
        });
    }

    function clearTheme() {
        themeLink.href = "";
        themeSelect.value = "";
        clearExistingThemeBodyClasses();
        localStorage.removeItem(storageKey);
        console.log("Theme cleared");
    }

    function applyTheme(themeFile) {
        if (!themeFile) {
            clearTheme();
            return;
        }

        const optionExists = Array.from(themeSelect.options).some(
            (option) => option.value === themeFile
        );

        if (!optionExists) {
            console.warn("Theme option not found:", themeFile);
            clearTheme();
            return;
        }

        const bodyClass = filenameToBodyClass(themeFile);

        clearExistingThemeBodyClasses();
        if (bodyClass) {
            document.body.classList.add(bodyClass);
        }

        themeLink.href = `${themeBasePath}${themeFile}?v=${Date.now()}`;
        themeSelect.value = themeFile;
        localStorage.setItem(storageKey, themeFile);

        console.log("Applied theme file:", themeFile);
        console.log("Applied body class:", bodyClass);
        console.log("Theme href:", themeLink.href);
        console.log("Body class list:", document.body.className);
    }

    themeSelect.addEventListener("change", function () {
        applyTheme(this.value);
    });

    const savedTheme = localStorage.getItem(storageKey);
    if (savedTheme) {
        applyTheme(savedTheme);
    }
});