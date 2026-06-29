// ============================================================
// Complete Document Search Dropdown Loader
// WebView-safe / reload-safe version
//
// Handles:
// - Area
// - Equipment Group
// - Model
// - Asset Number
// - Location
//
// This file only populates dropdowns.
// It does not open document/image/drawing links.
// ============================================================

(function () {
    "use strict";

    const LOG_PREFIX = "[CompleteDocumentDropdowns]";

    const SELECTORS = {
        area: "#searchdocument_areaDropdown",
        equipmentGroup: "#searchdocument_equipmentGroupDropdown",
        model: "#searchdocument_modelDropdown",
        assetNumber: "#searchdocument_assetNumberDropdown",
        location: "#searchdocument_locationDropdown",
    };

    const API_URL = "/get_completedocument_list_data_bp";

    let cachedDropdownData = null;

    function asArray(value) {
        return Array.isArray(value) ? value : [];
    }

    function getDropdownDataArray(data, key) {
        if (!data || typeof data !== "object") {
            return [];
        }

        return asArray(data[key]);
    }

    function resetDropdown($dropdown, placeholderText) {
        if (!$dropdown || !$dropdown.length) {
            return;
        }

        $dropdown.empty();
        $dropdown.append(
            $("<option>", {
                value: "",
                text: placeholderText || "Select...",
            })
        );
    }

    function appendOption($dropdown, value, text) {
        $dropdown.append(
            $("<option>", {
                value: value == null ? "" : String(value),
                text: text == null || String(text).trim() === "" ? "Unnamed" : String(text),
            })
        );
    }

    function initSelect2($dropdown, placeholderText) {
        if (!$dropdown || !$dropdown.length) {
            return;
        }

        if (!$.fn || typeof $.fn.select2 !== "function") {
            return;
        }

        try {
            if ($dropdown.hasClass("select2-hidden-accessible")) {
                $dropdown.select2("destroy");
            }

            $dropdown.select2({
                width: "100%",
                placeholder: placeholderText || "Select...",
                allowClear: true,
            });
        } catch (err) {
            console.warn(`${LOG_PREFIX} Select2 initialization failed:`, err);
        }
    }

    function initializeSelect2Controls() {
        initSelect2($(SELECTORS.area), "Select Area...");
        initSelect2($(SELECTORS.equipmentGroup), "Select Equipment Group...");
        initSelect2($(SELECTORS.model), "Select Model...");
        initSelect2($(SELECTORS.assetNumber), "Select Asset Number...");
        initSelect2($(SELECTORS.location), "Select Location...");
    }

    function populateAreas(data) {
        const $areaDropdown = $(SELECTORS.area);

        resetDropdown($areaDropdown, "Select Area...");

        getDropdownDataArray(data, "areas").forEach(area => {
            appendOption(
                $areaDropdown,
                area.id,
                area.name
            );
        });

        initSelect2($areaDropdown, "Select Area...");
    }

    function populateEquipmentGroupsForArea(areaId) {
        const data = cachedDropdownData || {};
        const $equipmentGroupDropdown = $(SELECTORS.equipmentGroup);

        resetDropdown($equipmentGroupDropdown, "Select Equipment Group...");

        getDropdownDataArray(data, "equipment_groups").forEach(group => {
            if (String(group.area_id) === String(areaId)) {
                appendOption(
                    $equipmentGroupDropdown,
                    group.id,
                    group.name
                );
            }
        });

        initSelect2($equipmentGroupDropdown, "Select Equipment Group...");

        // Reset child dropdowns when parent changes.
        populateModelsForEquipmentGroup("");
    }

    function populateModelsForEquipmentGroup(equipmentGroupId) {
        const data = cachedDropdownData || {};
        const $modelDropdown = $(SELECTORS.model);

        resetDropdown($modelDropdown, "Select Model...");

        getDropdownDataArray(data, "models").forEach(model => {
            if (String(model.equipment_group_id) === String(equipmentGroupId)) {
                appendOption(
                    $modelDropdown,
                    model.id,
                    model.name
                );
            }
        });

        initSelect2($modelDropdown, "Select Model...");

        // Reset child dropdowns when parent changes.
        populateAssetNumbersForModel("");
        populateLocationsForModel("");
    }

    function populateAssetNumbersForModel(modelId) {
        const data = cachedDropdownData || {};
        const $assetNumberDropdown = $(SELECTORS.assetNumber);

        resetDropdown($assetNumberDropdown, "Select Asset Number...");

        getDropdownDataArray(data, "asset_numbers").forEach(assetNumber => {
            if (String(assetNumber.model_id) === String(modelId)) {
                appendOption(
                    $assetNumberDropdown,
                    assetNumber.id,
                    assetNumber.number || assetNumber.name
                );
            }
        });

        initSelect2($assetNumberDropdown, "Select Asset Number...");
    }

    function populateLocationsForModel(modelId) {
        const data = cachedDropdownData || {};
        const $locationDropdown = $(SELECTORS.location);

        resetDropdown($locationDropdown, "Select Location...");

        getDropdownDataArray(data, "locations").forEach(location => {
            if (String(location.model_id) === String(modelId)) {
                appendOption(
                    $locationDropdown,
                    location.id,
                    location.name
                );
            }
        });

        initSelect2($locationDropdown, "Select Location...");
    }

    function bindDropdownEvents() {
        const $areaDropdown = $(SELECTORS.area);
        const $equipmentGroupDropdown = $(SELECTORS.equipmentGroup);
        const $modelDropdown = $(SELECTORS.model);

        /*
         * Namespace the handlers so repeated script loads do not stack duplicate
         * change events.
         */
        $areaDropdown
            .off("change.emtacCompleteDocument")
            .on("change.emtacCompleteDocument", function () {
                const selectedAreaId = $(this).val();
                populateEquipmentGroupsForArea(selectedAreaId);
            });

        $equipmentGroupDropdown
            .off("change.emtacCompleteDocument")
            .on("change.emtacCompleteDocument", function () {
                const selectedGroupId = $(this).val();
                populateModelsForEquipmentGroup(selectedGroupId);
            });

        $modelDropdown
            .off("change.emtacCompleteDocument")
            .on("change.emtacCompleteDocument", function () {
                const selectedModelId = $(this).val();
                populateAssetNumbersForModel(selectedModelId);
                populateLocationsForModel(selectedModelId);
            });
    }

    function clearAllDropdowns() {
        resetDropdown($(SELECTORS.area), "Select Area...");
        resetDropdown($(SELECTORS.equipmentGroup), "Select Equipment Group...");
        resetDropdown($(SELECTORS.model), "Select Model...");
        resetDropdown($(SELECTORS.assetNumber), "Select Asset Number...");
        resetDropdown($(SELECTORS.location), "Select Location...");
        initializeSelect2Controls();
    }

    function populateCompleteDocumentDropdowns() {
        console.log(`${LOG_PREFIX} Loading dropdown data...`);

        clearAllDropdowns();

        $.ajax({
            url: API_URL,
            type: "GET",
            cache: false,
            success: function (data) {
                cachedDropdownData = data || {};

                console.log(`${LOG_PREFIX} Dropdown data loaded.`, {
                    areas: getDropdownDataArray(cachedDropdownData, "areas").length,
                    equipmentGroups: getDropdownDataArray(cachedDropdownData, "equipment_groups").length,
                    models: getDropdownDataArray(cachedDropdownData, "models").length,
                    assetNumbers: getDropdownDataArray(cachedDropdownData, "asset_numbers").length,
                    locations: getDropdownDataArray(cachedDropdownData, "locations").length,
                });

                bindDropdownEvents();
                populateAreas(cachedDropdownData);

                // Keep children empty until the user selects a real area.
                populateEquipmentGroupsForArea("");
            },
            error: function (xhr, status, error) {
                console.error(`${LOG_PREFIX} Error fetching dropdown data:`, {
                    status,
                    error,
                    responseText: xhr && xhr.responseText,
                });
            },
        });
    }

    $(document).ready(function () {
        console.log(`${LOG_PREFIX} Document ready. Initializing dropdowns.`);
        populateCompleteDocumentDropdowns();
    });

    window.populateCompleteDocumentDropdowns = populateCompleteDocumentDropdowns;
})();