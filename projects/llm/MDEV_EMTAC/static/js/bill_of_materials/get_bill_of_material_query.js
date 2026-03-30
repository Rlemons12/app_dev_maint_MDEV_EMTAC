/**
 * static/js/bill_of_materials/get_bill_of_material_query.js
 * Bill of Materials - Advanced Search Functionality
 * Native select version (Select2 removed)
 */

// Use a namespace to prevent global conflicts
window.BOMAdvancedSearch = window.BOMAdvancedSearch || {};

// Only define constants if they don't already exist
if (!window.BOMAdvancedSearch.DROPDOWN_IDS) {
    window.BOMAdvancedSearch.DROPDOWN_IDS = {
        area: "filter_areaDropdown",
        equipmentGroup: "filter_equipmentGroupDropdown",
        model: "filter_modelDropdown",
        assetNumber: "filter_assetNumberDropdown",
        location: "filter_locationDropdown"
    };
}

// API endpoint for fetching dropdown data
window.BOMAdvancedSearch.DATA_ENDPOINT =
    `${window.location.origin}/bill_of_materials/get_parts_position_data`;

/**
 * Main function to populate all dropdowns in the advanced search form
 */
window.populateDropdownsForPartsPosition = function() {
    if (sessionStorage.getItem("clearAdvancedForm") === "true") {
        sessionStorage.removeItem("clearAdvancedForm");
        window.BOMAdvancedSearch.clearAdvancedForm();
    }

    window.BOMAdvancedSearch.logDebug("Starting dropdown population process");
    window.BOMAdvancedSearch.showLoadingIndicators();

    $.ajax({
        url: window.BOMAdvancedSearch.DATA_ENDPOINT,
        type: "GET",
        dataType: "json",
        beforeSend: function() {
            window.BOMAdvancedSearch.logDebug(
                "Sending AJAX request to: " + window.BOMAdvancedSearch.DATA_ENDPOINT
            );
        },
        success: function(data) {
            window.BOMAdvancedSearch.logDebug("Received data:", data);

            if (!window.BOMAdvancedSearch.validateData(data)) {
                window.BOMAdvancedSearch.showError(
                    "Invalid data received from server. Please try again."
                );
                window.BOMAdvancedSearch.hideLoadingIndicators();
                return;
            }

            window.BOMAdvancedSearch.populateAreaDropdown(data.areas);
            window.BOMAdvancedSearch.setupDropdownEvents(data);
            window.BOMAdvancedSearch.ensureElementsVisible();
            window.BOMAdvancedSearch.hideLoadingIndicators();

            window.BOMAdvancedSearch.logDebug(
                "Dropdown population completed successfully"
            );
        },
        error: function(xhr, status, error) {
            window.BOMAdvancedSearch.logDebug(
                "AJAX Error: " + status + " - " + error
            );
            console.error("Error details:", xhr.responseText);

            window.BOMAdvancedSearch.showError(
                "Failed to load dropdown data. Please try again."
            );
            window.BOMAdvancedSearch.hideLoadingIndicators();
        }
    });
};

/**
 * Reset all form values in the advanced search form
 */
window.BOMAdvancedSearch.clearAdvancedForm = function() {
    window.BOMAdvancedSearch.logDebug("Clearing advanced search form");

    const formSelects = $("#advancedSearchForm select, #advancedSearchFilterForm select");

    formSelects.each(function() {
        const select = $(this);
        select.find("option:not(:first)").remove();
        select.val("");
    });

    window.BOMAdvancedSearch.logDebug("Advanced form cleared");
};

/**
 * Validate that the received data contains all required properties
 */
window.BOMAdvancedSearch.validateData = function(data) {
    if (!data) {
        return false;
    }

    const requiredProperties = [
        "areas",
        "equipment_groups",
        "models",
        "asset_numbers",
        "locations"
    ];

    let isValid = true;

    requiredProperties.forEach(function(prop) {
        if (!data[prop] || !Array.isArray(data[prop])) {
            window.BOMAdvancedSearch.logDebug(`Missing or invalid property: ${prop}`);
            isValid = false;
        }
    });

    return isValid;
};

/**
 * Helper: reset a dropdown to a placeholder option
 */
window.BOMAdvancedSearch.resetDropdownToPlaceholder = function(dropdown, placeholderText) {
    dropdown.empty();
    dropdown.append(
        $("<option></option>")
            .attr("value", "")
            .text(placeholderText)
    );
    dropdown.val("");
};

/**
 * Populate the Area dropdown with options
 */
window.BOMAdvancedSearch.populateAreaDropdown = function(areas) {
    const areaDropdown = $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.area);

    window.BOMAdvancedSearch.resetDropdownToPlaceholder(
        areaDropdown,
        "Select an area..."
    );

    if (areas && areas.length > 0) {
        $.each(areas, function(index, area) {
            areaDropdown.append(
                $("<option></option>")
                    .attr("value", area.id)
                    .text(area.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${areas.length} area options`);
    } else {
        window.BOMAdvancedSearch.logDebug("No areas available to populate dropdown");
    }
};

/**
 * Set up event handlers for all dropdown changes
 */
window.BOMAdvancedSearch.setupDropdownEvents = function(data) {
    window.BOMAdvancedSearch.logDebug("Setting up dropdown cascade events");

    $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.area)
        .off("change")
        .on("change", function() {
            const selectedAreaId = $(this).val();
            window.BOMAdvancedSearch.logDebug(`Area changed to: ${selectedAreaId}`);

            window.BOMAdvancedSearch.populateEquipmentGroupDropdown(
                data.equipment_groups,
                selectedAreaId
            );

            window.BOMAdvancedSearch.clearDropdowns([
                window.BOMAdvancedSearch.DROPDOWN_IDS.model,
                window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber,
                window.BOMAdvancedSearch.DROPDOWN_IDS.location
            ]);
        });

    $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.equipmentGroup)
        .off("change")
        .on("change", function() {
            const selectedGroupId = $(this).val();
            window.BOMAdvancedSearch.logDebug(
                `Equipment group changed to: ${selectedGroupId}`
            );

            window.BOMAdvancedSearch.populateModelDropdown(
                data.models,
                selectedGroupId
            );

            window.BOMAdvancedSearch.clearDropdowns([
                window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber,
                window.BOMAdvancedSearch.DROPDOWN_IDS.location
            ]);
        });

    $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.model)
        .off("change")
        .on("change", function() {
            const selectedModelId = $(this).val();
            window.BOMAdvancedSearch.logDebug(`Model changed to: ${selectedModelId}`);

            window.BOMAdvancedSearch.populateAssetNumberDropdown(
                data.asset_numbers,
                selectedModelId
            );
            window.BOMAdvancedSearch.populateLocationDropdown(
                data.locations,
                selectedModelId
            );
        });

    $("#resetFilterBtn")
        .off("click")
        .on("click", function(e) {
            e.preventDefault();
            window.BOMAdvancedSearch.resetAllDropdowns();
        });
};

/**
 * Populate the Equipment Group dropdown based on selected Area
 */
window.BOMAdvancedSearch.populateEquipmentGroupDropdown = function(groups, selectedAreaId) {
    const dropdown = $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.equipmentGroup);

    window.BOMAdvancedSearch.resetDropdownToPlaceholder(
        dropdown,
        "Select an equipment group..."
    );

    if (!selectedAreaId) {
        return;
    }

    const filteredGroups = groups.filter(function(group) {
        return String(group.area_id) === String(selectedAreaId);
    });

    if (filteredGroups.length > 0) {
        $.each(filteredGroups, function(index, group) {
            dropdown.append(
                $("<option></option>")
                    .attr("value", group.id)
                    .text(group.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(
            `Added ${filteredGroups.length} equipment group options for area ${selectedAreaId}`
        );
    } else {
        window.BOMAdvancedSearch.logDebug(
            `No equipment groups available for area ${selectedAreaId}`
        );
    }
};

/**
 * Populate the Model dropdown based on selected Equipment Group
 */
window.BOMAdvancedSearch.populateModelDropdown = function(models, selectedGroupId) {
    const dropdown = $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.model);

    window.BOMAdvancedSearch.resetDropdownToPlaceholder(
        dropdown,
        "Select a model..."
    );

    if (!selectedGroupId) {
        return;
    }

    const filteredModels = models.filter(function(model) {
        return String(model.equipment_group_id) === String(selectedGroupId);
    });

    if (filteredModels.length > 0) {
        $.each(filteredModels, function(index, model) {
            dropdown.append(
                $("<option></option>")
                    .attr("value", model.id)
                    .text(model.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(
            `Added ${filteredModels.length} model options for equipment group ${selectedGroupId}`
        );
    } else {
        window.BOMAdvancedSearch.logDebug(
            `No models available for equipment group ${selectedGroupId}`
        );
    }
};

/**
 * Populate the Asset Number dropdown based on selected Model
 */
window.BOMAdvancedSearch.populateAssetNumberDropdown = function(assetNumbers, selectedModelId) {
    const dropdown = $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber);

    window.BOMAdvancedSearch.resetDropdownToPlaceholder(
        dropdown,
        "Select an asset number..."
    );

    if (!selectedModelId) {
        return;
    }

    const filteredAssets = assetNumbers.filter(function(asset) {
        return String(asset.model_id) === String(selectedModelId);
    });

    if (filteredAssets.length > 0) {
        $.each(filteredAssets, function(index, asset) {
            dropdown.append(
                $("<option></option>")
                    .attr("value", asset.id)
                    .text(asset.number)
            );
        });
        window.BOMAdvancedSearch.logDebug(
            `Added ${filteredAssets.length} asset number options for model ${selectedModelId}`
        );
    } else {
        window.BOMAdvancedSearch.logDebug(
            `No asset numbers available for model ${selectedModelId}`
        );
    }
};

/**
 * Populate the Location dropdown based on selected Model
 */
window.BOMAdvancedSearch.populateLocationDropdown = function(locations, selectedModelId) {
    const dropdown = $("#" + window.BOMAdvancedSearch.DROPDOWN_IDS.location);

    window.BOMAdvancedSearch.resetDropdownToPlaceholder(
        dropdown,
        "Select a location..."
    );

    if (!selectedModelId) {
        return;
    }

    const filteredLocations = locations.filter(function(location) {
        return String(location.model_id) === String(selectedModelId);
    });

    if (filteredLocations.length > 0) {
        $.each(filteredLocations, function(index, location) {
            dropdown.append(
                $("<option></option>")
                    .attr("value", location.id)
                    .text(location.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(
            `Added ${filteredLocations.length} location options for model ${selectedModelId}`
        );
    } else {
        window.BOMAdvancedSearch.logDebug(
            `No locations available for model ${selectedModelId}`
        );
    }
};

/**
 * Select2 removed intentionally.
 * Keep helper so the rest of the page logic does not break.
 */
window.BOMAdvancedSearch.safeSelect2Initialize = function(dropdown) {
    return;
};

/**
 * Clear multiple dropdowns
 */
window.BOMAdvancedSearch.clearDropdowns = function(dropdownIds) {
    const labels = {
        filter_modelDropdown: "Select a model...",
        filter_assetNumberDropdown: "Select an asset number...",
        filter_locationDropdown: "Select a location...",
        filter_equipmentGroupDropdown: "Select an equipment group..."
    };

    dropdownIds.forEach(function(id) {
        const dropdown = $("#" + id);
        window.BOMAdvancedSearch.resetDropdownToPlaceholder(
            dropdown,
            labels[id] || "Select..."
        );
    });
};

/**
 * Reset all dropdowns to their initial state
 */
window.BOMAdvancedSearch.resetAllDropdowns = function() {
    window.BOMAdvancedSearch.logDebug("Resetting all dropdowns");

    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(function(id) {
        const dropdown = $("#" + id);
        dropdown.empty();
    });

    window.populateDropdownsForPartsPosition();
};

/**
 * Select2 removed intentionally.
 */
window.BOMAdvancedSearch.initializeSelect2 = function() {
    window.BOMAdvancedSearch.logDebug(
        "Select2 disabled - using native select controls"
    );
};

/**
 * Ensure all form elements are visible
 */
window.BOMAdvancedSearch.ensureElementsVisible = function() {
    $("#advancedSearchForm, #advancedSearchFilterForm").css({
        display: "block",
        visibility: "visible"
    });

    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(function(id) {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = "block";
            element.style.visibility = "visible";
            element.style.opacity = "1";
        } else {
            window.BOMAdvancedSearch.logDebug(
                `WARNING: Element with ID ${id} not found!`
            );
        }
    });
};

/**
 * Show loading indicators for all dropdowns
 */
window.BOMAdvancedSearch.showLoadingIndicators = function() {
    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(function(id) {
        const dropdown = $("#" + id);
        dropdown.empty();
        dropdown.append(
            $("<option></option>")
                .attr("value", "")
                .text("Loading...")
        );
        dropdown.prop("disabled", true);
        dropdown.val("");
    });
};

/**
 * Hide loading indicators for all dropdowns
 */
window.BOMAdvancedSearch.hideLoadingIndicators = function() {
    const labels = {
        filter_areaDropdown: "Select an area...",
        filter_equipmentGroupDropdown: "Select an equipment group...",
        filter_modelDropdown: "Select a model...",
        filter_assetNumberDropdown: "Select an asset number...",
        filter_locationDropdown: "Select a location..."
    };

    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(function(id) {
        const dropdown = $("#" + id);
        if (dropdown.find("option").length === 0) {
            dropdown.append(
                $("<option></option>")
                    .attr("value", "")
                    .text(labels[id] || "Select...")
            );
        } else {
            dropdown.find("option:first").text(labels[id] || "Select...");
        }
        dropdown.prop("disabled", false);
        dropdown.val("");
    });
};

/**
 * Display an error message to the user
 */
window.BOMAdvancedSearch.showError = function(message) {
    if ($("#searchFormError").length === 0) {
        const formContainer = $("#advancedSearchForm, #advancedSearchFilterForm").first();
        formContainer.prepend(
            $("<div id='searchFormError'></div>").css({
                "background-color": "#f8d7da",
                color: "#721c24",
                padding: "10px",
                "margin-bottom": "15px",
                border: "1px solid #f5c6cb",
                "border-radius": "4px"
            })
        );
    }

    $("#searchFormError").text(message).show();
    window.BOMAdvancedSearch.logDebug("ERROR: " + message);
};

/**
 * Log debug information to console and debug info area
 */
window.BOMAdvancedSearch.logDebug = function() {
    return;
};

/**
 * Toggle debug information visibility
 */
window.BOMAdvancedSearch.toggleDebugInfo = function() {
    return;
};

/**
 * Function to ensure results stay visible
 */
window.BOMAdvancedSearch.ensureResultsStayVisible = function() {
    window.BOMAdvancedSearch.logDebug("Using standalone results page approach");

    localStorage.removeItem("expectingResults");
    localStorage.removeItem("lastSearchTime");

    window.BOMAdvancedSearch.logDebug(
        "Cleared result flags for standalone page"
    );
};

/**
 * Global clear function that works for both forms
 */
window.clearAdvancedForm = function() {
    console.log("clearAdvancedForm called");

    const advFilterForm = document.getElementById("advancedSearchFilterForm");
    const advContentForm = document.getElementById("advancedSearchFormContent");

    console.log("Forms found:", {
        advFilterForm: !!advFilterForm,
        advContentForm: !!advContentForm
    });

    const formToReset = advFilterForm || (advContentForm ? advContentForm.form : null);

    if (formToReset) {
        console.log("Resetting form:", formToReset.id);
        formToReset.reset();
        window.BOMAdvancedSearch.resetAllDropdowns();
    } else {
        console.warn("No form found to reset!");
    }
};

// Override any document location changes after search
var originalAssign = window.location.assign;
window.location.assign = function(url) {
    console.log("Location change attempted to:", url);

    if (
        localStorage.getItem("expectingResults") === "true" &&
        !url.includes("view_bill_of_material")
    ) {
        console.log("Blocking redirect that would hide results");
        return;
    }

    originalAssign.call(window.location, url);
};



// Initialize when document is ready
$(document).ready(function() {
    window.BOMAdvancedSearch.logDebug("Document ready - Advanced Search JS loaded");

    if (window.jQuery) {
        window.BOMAdvancedSearch.logDebug("jQuery is available");
        window.BOMAdvancedSearch.logDebug("Using native select controls");
    } else {
        console.error("jQuery is not available - this should never happen");
    }

    const isResultsPage =
        window.location.pathname.includes("view_bill_of_material") ||
        window.location.pathname.includes("bill_of_material_results") ||
        window.location.pathname.includes("/create_bill_of_material");

    if (isResultsPage) {
        window.BOMAdvancedSearch.logDebug(
            "We're on the results page - minimal initialization"
        );
        sessionStorage.removeItem("clearAdvancedForm");
        return;
    }

    let advancedForm = $("#advancedSearchFilterForm");

    window.BOMAdvancedSearch.logDebug("Found form count: " + advancedForm.length);

    if (advancedForm.length > 1) {
        window.BOMAdvancedSearch.logDebug(
            "WARNING: Found multiple forms with the same ID!"
        );

        advancedForm.each(function(idx) {
            window.BOMAdvancedSearch.logDebug(
                `Form ${idx} parent: ${$(this).parent().attr("id") || "unknown"}`
            );
        });

        advancedForm = advancedForm.eq(0);
    }

    if (advancedForm.length) {
        window.BOMAdvancedSearch.logDebug(
            "Using search form ID: " + advancedForm.attr("id")
        );

        advancedForm.off("submit");

        advancedForm.on("submit", function(event) {
            window.BOMAdvancedSearch.logDebug("Form submitted: " + this.id);

            if ($(this).attr("method").toUpperCase() !== "POST") {
                window.BOMAdvancedSearch.logDebug(
                    "Form method is not POST! Correcting..."
                );
                $(this).attr("method", "POST");
            }

            let hasValue = false;
            $(this).find("select").each(function() {
                if ($(this).val()) {
                    hasValue = true;
                    return false;
                }
            });

            if (!hasValue) {
                window.BOMAdvancedSearch.logDebug("No values selected in form");
                window.BOMAdvancedSearch.showError(
                    "Please select at least one search criteria"
                );
                event.preventDefault();
                return false;
            }

            const formData = $(this).serialize();
            window.BOMAdvancedSearch.logDebug("Form data: " + formData);
            window.BOMAdvancedSearch.logDebug("Form action: " + $(this).attr("action"));
            window.BOMAdvancedSearch.logDebug("Form method: " + $(this).attr("method"));

            sessionStorage.setItem("clearAdvancedForm", "true");
            return true;
        });
    } else {
        window.BOMAdvancedSearch.logDebug(
            "Warning: Advanced search form not found!"
        );
    }

    $(".back-to-form")
        .off("click")
        .on("click", function(e) {
            window.BOMAdvancedSearch.logDebug(
                "Back to search clicked, setting clear flag"
            );
            e.preventDefault();
            sessionStorage.setItem("clearAdvancedForm", "true");
            window.location.href = $(this).attr("href");
        });

    if (sessionStorage.getItem("clearAdvancedForm") === "true") {
        window.BOMAdvancedSearch.logDebug("Clear flag found on page load");
    }

    const advancedFormContainer = document.getElementById("advancedSearchForm");
    if (
        advancedFormContainer &&
        advancedFormContainer.style.display !== "none" &&
        getComputedStyle(advancedFormContainer).display !== "none"
    ) {
        window.BOMAdvancedSearch.logDebug(
            "Advanced form is visible on page load, populating dropdowns"
        );
        window.populateDropdownsForPartsPosition();
    } else {
        window.BOMAdvancedSearch.logDebug("Advanced form is hidden on page load");
    }


});