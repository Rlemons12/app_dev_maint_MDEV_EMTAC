(function () {
    "use strict";

    const TILE_SELECTS = [
        {
            id: "searchdocument_areaDropdown",
            label: "Area",
            emptyText: "All Areas",
            noOptionsText: "No Areas"
        },
        {
            id: "searchdocument_equipmentGroupDropdown",
            label: "Equipment Group",
            emptyText: "All Equipment Groups",
            noOptionsText: "Select Area First"
        },
        {
            id: "searchdocument_modelDropdown",
            label: "Model",
            emptyText: "All Models",
            noOptionsText: "Select Equipment First"
        },
        {
            id: "searchdocument_assetNumberDropdown",
            label: "Asset Number",
            emptyText: "All Assets",
            noOptionsText: "Select Model First"
        },
        {
            id: "searchdocument_locationDropdown",
            label: "Location",
            emptyText: "All Locations",
            noOptionsText: "Select Asset First"
        }
    ];

    function qs(selector, root) {
        return (root || document).querySelector(selector);
    }

    function qsa(selector, root) {
        return Array.from((root || document).querySelectorAll(selector));
    }

    function escapeHtml(value) {
        const div = document.createElement("div");
        div.textContent = value == null ? "" : String(value);
        return div.innerHTML;
    }

    function normalizeText(value) {
        return String(value || "")
            .replace(/\s+/g, " ")
            .trim()
            .toLowerCase();
    }

    function destroySelect2(selectEl) {
        if (!selectEl) {
            return;
        }

        try {
            if (
                window.jQuery &&
                window.jQuery.fn &&
                window.jQuery.fn.select2
            ) {
                const $select = window.jQuery(selectEl);

                if ($select.data("select2")) {
                    try {
                        $select.select2("close");
                    } catch (_) {
                        // ignore close errors
                    }

                    $select.select2("destroy");
                }
            }
        } catch (error) {
            console.warn("Demo tile picker: Select2 cleanup skipped.", error);
        }

        const parent = selectEl.parentElement;

        if (parent) {
            qsa(".select2, .select2-container", parent).forEach(function (el) {
                el.style.display = "none";
                el.setAttribute("data-demo-select2-hidden", "true");
            });
        }

        let next = selectEl.nextElementSibling;
        let count = 0;

        while (next && count < 5) {
            if (
                next.classList &&
                (
                    next.classList.contains("select2") ||
                    next.classList.contains("select2-container")
                )
            ) {
                next.style.display = "none";
                next.setAttribute("data-demo-select2-hidden", "true");
            }

            next = next.nextElementSibling;
            count += 1;
        }

        qsa(".select2-container--open, .select2-dropdown").forEach(function (el) {
            el.style.display = "none";
        });
    }

    function getOptions(selectEl, config) {
        const options = Array.from(selectEl.options)
            .map(function (option) {
                const rawText = String(option.textContent || "").trim();
                const rawValue = String(option.value || "").trim();

                return {
                    value: option.value,
                    text: rawText,
                    thumb: option.getAttribute("data-thumb") || "",
                    isEmpty: rawValue === "" || normalizeText(rawText).includes("select ")
                };
            })
            .filter(function (item) {
                return item.text;
            });

        return options.map(function (item) {
            if (item.isEmpty) {
                return {
                    value: item.value,
                    text: config.emptyText,
                    thumb: item.thumb,
                    isEmpty: true
                };
            }

            return item;
        });
    }

    function setSelectValue(selectEl, value) {
        selectEl.value = value;

        selectEl.dispatchEvent(new Event("input", {
            bubbles: true,
            cancelable: true
        }));

        selectEl.dispatchEvent(new Event("change", {
            bubbles: true,
            cancelable: true
        }));

        if (window.jQuery) {
            window.jQuery(selectEl).val(value).trigger("change");
        }
    }

    function findLabelForSelect(selectEl) {
        if (!selectEl || !selectEl.id) {
            return null;
        }

        return document.querySelector(`label[for="${selectEl.id}"]`);
    }

    function createDisabledTile(text) {
        const tile = document.createElement("div");
        tile.className = "document-demo-select-tile document-demo-select-tile--disabled";
        tile.innerHTML = `
            <span class="document-demo-select-tile-text">
                ${escapeHtml(text)}
            </span>
        `;
        return tile;
    }

    function createOptionTile(selectEl, item, config, renderTiles) {
        const isActive = String(selectEl.value) === String(item.value);

        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = "document-demo-select-tile";
        tile.dataset.value = item.value;
        tile.setAttribute("role", "radio");
        tile.setAttribute("aria-checked", isActive ? "true" : "false");

        if (isActive) {
            tile.classList.add("active");
        }

        if (item.thumb) {
            tile.classList.add("document-demo-select-tile--with-thumb");
            tile.innerHTML = `
                <span class="document-demo-select-tile-thumb">
                    <img src="${escapeHtml(item.thumb)}" alt="${escapeHtml(item.text)}">
                </span>
                <span class="document-demo-select-tile-caption">
                    ${escapeHtml(item.text)}
                </span>
            `;
        } else {
            tile.classList.add("document-demo-select-tile--text-only");
            tile.innerHTML = `
                <span class="document-demo-select-tile-text">
                    ${escapeHtml(item.text)}
                </span>
            `;
        }

        tile.addEventListener("click", function () {
            setSelectValue(selectEl, item.value);
            renderTiles();
        });

        return tile;
    }

    function tileizeSelect(config) {
        const selectEl = document.getElementById(config.id);

        if (!selectEl) {
            console.warn(`Demo tile picker: #${config.id} not found.`);
            return;
        }

        destroySelect2(selectEl);

        if (selectEl.dataset.demoTileInitialized === "true") {
            return;
        }

        selectEl.dataset.demoTileInitialized = "true";
        selectEl.classList.add("document-demo-hidden-select");
        selectEl.style.display = "none";

        const label = findLabelForSelect(selectEl);

        const picker = document.createElement("div");
        picker.className = "document-demo-select-tile-picker";
        picker.dataset.tilePickerFor = config.id;

        picker.innerHTML = `
            <div class="document-demo-select-tile-header">
                <span class="document-demo-select-tile-title">
                    ${escapeHtml(config.label)}
                </span>
            </div>

            <div
                class="document-demo-select-tile-grid"
                role="radiogroup"
                aria-label="${escapeHtml(config.label)}"
            ></div>
        `;

        if (label) {
            label.insertAdjacentElement("afterend", picker);
        } else {
            selectEl.insertAdjacentElement("beforebegin", picker);
        }

        const grid = qs(".document-demo-select-tile-grid", picker);

        function renderTiles() {
            destroySelect2(selectEl);

            const options = getOptions(selectEl, config);

            grid.innerHTML = "";

            if (!options.length) {
                grid.appendChild(createDisabledTile(config.noOptionsText));
                return;
            }

            options.forEach(function (item) {
                const tile = createOptionTile(selectEl, item, config, renderTiles);
                grid.appendChild(tile);
            });
        }

        renderTiles();

        selectEl.addEventListener("change", renderTiles);

        const observer = new MutationObserver(function () {
            renderTiles();
        });

        observer.observe(selectEl, {
            childList: true,
            subtree: true,
            attributes: true
        });
    }

    function initAllDocumentSearchTiles() {
        TILE_SELECTS.forEach(tileizeSelect);
    }

    function initFeedbackForm() {
        const form = qs("#documentSearchDemoFeedbackForm");
        const status = qs("#documentSearchDemoFeedbackStatus");

        if (!form) {
            return;
        }

        function getCheckedValue(name) {
            const checked = qs(`input[name="${name}"]:checked`, form);
            return checked ? checked.value : "";
        }

        form.addEventListener("submit", async function (event) {
            event.preventDefault();

            const feedbackUrl = form.getAttribute("data-feedback-url");

            if (!feedbackUrl) {
                if (status) {
                    status.textContent = "Feedback URL not configured.";
                }
                return;
            }

            const payload = {
                search_easy: getCheckedValue("search_easy"),
                results_clear: getCheckedValue("results_clear"),
                display_easy: getCheckedValue("display_easy"),
                user_role: qs("#documentDemoUserRole", form)?.value || "",
                what_worked: qs("#documentDemoWhatWorked", form)?.value || "",
                what_was_confusing: qs("#documentDemoConfusing", form)?.value || "",
                suggested_changes: qs("#documentDemoChanges", form)?.value || ""
            };

            try {
                if (status) {
                    status.textContent = "Saving...";
                }

                const response = await fetch(feedbackUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(function () {
                    return {};
                });

                if (!response.ok || data.ok === false) {
                    throw new Error(data.error || "Unable to save feedback.");
                }

                if (status) {
                    status.textContent = "Feedback saved.";
                }

                form.reset();
            } catch (error) {
                console.error(error);

                if (status) {
                    status.textContent = "Could not save feedback.";
                }
            }
        });
    }

    function init() {
        initAllDocumentSearchTiles();
        initFeedbackForm();

        /*
         * Existing search scripts populate dependent dropdowns after page load.
         * These retries make sure every dropdown gets tileized after those scripts run.
         */
        setTimeout(initAllDocumentSearchTiles, 250);
        setTimeout(initAllDocumentSearchTiles, 750);
        setTimeout(initAllDocumentSearchTiles, 1500);
        setTimeout(initAllDocumentSearchTiles, 2500);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();