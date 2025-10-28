/**
 * static/js/bill_of_materials/get_bill_of_material_query.js
 * Enhanced Bill of Materials - Advanced Search Functionality
 * This script handles the population of dropdown menus in the advanced search form.
 */

// Use a namespace to prevent global conflicts
window.BOMAdvancedSearch = window.BOMAdvancedSearch || {};

// Only define constants if they don't already exist
if (!window.BOMAdvancedSearch.DROPDOWN_IDS) {
    window.BOMAdvancedSearch.DROPDOWN_IDS = {
        area: 'filter_areaDropdown',
        equipmentGroup: 'filter_equipmentGroupDropdown',
        model: 'filter_modelDropdown',
        assetNumber: 'filter_assetNumberDropdown',
        location: 'filter_locationDropdown'
    };
}

// API endpoint for fetching dropdown data
window.BOMAdvancedSearch.DATA_ENDPOINT = '/get_parts_position_data';

/**
 * Main function to populate all dropdowns in the advanced search form
 */
window.populateDropdownsForPartsPosition = function() {
    // Check for form clearing flag first
    if (sessionStorage.getItem('clearAdvancedForm') === 'true') {
        sessionStorage.removeItem('clearAdvancedForm');
        window.BOMAdvancedSearch.clearAdvancedForm();
    }

    // Display debug info
    window.BOMAdvancedSearch.logDebug('Starting dropdown population process');
    window.BOMAdvancedSearch.showLoadingIndicators();

    // Make AJAX request to fetch data
    $.ajax({
        url: window.BOMAdvancedSearch.DATA_ENDPOINT,
        type: 'GET',
        dataType: 'json',
        beforeSend: function() {
            window.BOMAdvancedSearch.logDebug('Sending AJAX request to: ' + window.BOMAdvancedSearch.DATA_ENDPOINT);
        },
        success: function(data) {
            window.BOMAdvancedSearch.logDebug('Received data:', data);

            // Validate received data
            if (!window.BOMAdvancedSearch.validateData(data)) {
                window.BOMAdvancedSearch.showError('Invalid data received from server. Please try again.');
                return;
            }

            // Populate the area dropdown first
            window.BOMAdvancedSearch.populateAreaDropdown(data.areas);

            // Set up cascade event handlers for all dropdowns
            window.BOMAdvancedSearch.setupDropdownEvents(data);

            // Initialize Select2 if available
            window.BOMAdvancedSearch.initializeSelect2();

            // Ensure all elements are visible
            window.BOMAdvancedSearch.ensureElementsVisible();

            // Remove loading indicators
            window.BOMAdvancedSearch.hideLoadingIndicators();

            window.BOMAdvancedSearch.logDebug('Dropdown population completed successfully');
        },
        error: function(xhr, status, error) {
            window.BOMAdvancedSearch.logDebug('AJAX Error: ' + status + ' - ' + error);
            console.error('Error details:', xhr.responseText);

            window.BOMAdvancedSearch.showError('Failed to load dropdown data. Please try again.');
            window.BOMAdvancedSearch.hideLoadingIndicators();
        }
    });
};

/**
 * Reset all form values in the advanced search form
 */
window.BOMAdvancedSearch.clearAdvancedForm = function() {
    window.BOMAdvancedSearch.logDebug("Clearing advanced search form");

    // Get all select elements in both possible form versions
    const formSelects = $('#advancedSearchForm select, #advancedSearchFilterForm select');

    // Reset each select to empty value
    formSelects.each(function() {
        const select = $(this);

        // Clear all options except the first one
        select.find('option:not(:first)').remove();

        // Reset to first option
        select.val('');

        // If using Select2, trigger change event
        if ($.fn.select2 && select.data('select2')) {
            try {
                select.val('').trigger('change');
            } catch(e) {
                console.warn("Error resetting Select2:", e);
            }
        }
    });

    window.BOMAdvancedSearch.logDebug("Advanced form cleared");
};

/**
 * Validate that the received data contains all required properties
 */
window.BOMAdvancedSearch.validateData = function(data) {
    if (!data) return false;

    const requiredProperties = ['areas', 'equipment_groups', 'models', 'asset_numbers', 'locations'];
    let isValid = true;

    requiredProperties.forEach(prop => {
        if (!data[prop] || !Array.isArray(data[prop])) {
            window.BOMAdvancedSearch.logDebug(`Missing or invalid property: ${prop}`);
            isValid = false;
        }
    });

    return isValid;
};

/**
 * Populate the Area dropdown with options
 */
window.BOMAdvancedSearch.populateAreaDropdown = function(areas) {
    const areaDropdown = $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.area);

    // Clear existing options except the first one
    areaDropdown.find('option:not(:first)').remove();

    // Add each area as an option
    if (areas && areas.length > 0) {
        $.each(areas, function(index, area) {
            areaDropdown.append(
                $('<option></option>')
                    .attr('value', area.id)
                    .text(area.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${areas.length} area options`);
    } else {
        window.BOMAdvancedSearch.logDebug('No areas available to populate dropdown');
    }
};

/**
 * Set up event handlers for all dropdown changes
 */
window.BOMAdvancedSearch.setupDropdownEvents = function(data) {
    window.BOMAdvancedSearch.logDebug('Setting up dropdown cascade events');

    // Area dropdown change event
    $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.area).off('change').on('change', function() {
        const selectedAreaId = $(this).val();
        window.BOMAdvancedSearch.logDebug(`Area changed to: ${selectedAreaId}`);

        window.BOMAdvancedSearch.populateEquipmentGroupDropdown(data.equipment_groups, selectedAreaId);

        // Clear subsequent dropdowns
        window.BOMAdvancedSearch.clearDropdowns([
            window.BOMAdvancedSearch.DROPDOWN_IDS.model,
            window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber,
            window.BOMAdvancedSearch.DROPDOWN_IDS.location
        ]);
    });

    // Equipment group dropdown change event
    $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.equipmentGroup).off('change').on('change', function() {
        const selectedGroupId = $(this).val();
        window.BOMAdvancedSearch.logDebug(`Equipment group changed to: ${selectedGroupId}`);

        window.BOMAdvancedSearch.populateModelDropdown(data.models, selectedGroupId);

        // Clear subsequent dropdowns
        window.BOMAdvancedSearch.clearDropdowns([
            window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber,
            window.BOMAdvancedSearch.DROPDOWN_IDS.location
        ]);
    });

    // Model dropdown change event
    $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.model).off('change').on('change', function() {
        const selectedModelId = $(this).val();
        window.BOMAdvancedSearch.logDebug(`Model changed to: ${selectedModelId}`);

        window.BOMAdvancedSearch.populateAssetNumberDropdown(data.asset_numbers, selectedModelId);
        window.BOMAdvancedSearch.populateLocationDropdown(data.locations, selectedModelId);
    });

    // Reset button event handler
    $('#resetFilterBtn').off('click').on('click', function() {
        window.BOMAdvancedSearch.resetAllDropdowns();
    });
};

/**
 * Populate the Equipment Group dropdown based on selected Area
 */
window.BOMAdvancedSearch.populateEquipmentGroupDropdown = function(groups, selectedAreaId) {
    const dropdown = $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.equipmentGroup);

    // Clear existing options except the first one
    dropdown.find('option:not(:first)').remove();

    if (!selectedAreaId) {
        return;
    }

    // Filter and add equipment groups for the selected area
    const filteredGroups = groups.filter(group => group.area_id == selectedAreaId);

    if (filteredGroups.length > 0) {
        $.each(filteredGroups, function(index, group) {
            dropdown.append(
                $('<option></option>')
                    .attr('value', group.id)
                    .text(group.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${filteredGroups.length} equipment group options for area ${selectedAreaId}`);
    } else {
        window.BOMAdvancedSearch.logDebug(`No equipment groups available for area ${selectedAreaId}`);
    }

    // Reinitialize Select2 if available
    window.BOMAdvancedSearch.safeSelect2Initialize(dropdown);
};

/**
 * Populate the Model dropdown based on selected Equipment Group
 */
window.BOMAdvancedSearch.populateModelDropdown = function(models, selectedGroupId) {
    const dropdown = $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.model);

    // Clear existing options except the first one
    dropdown.find('option:not(:first)').remove();

    if (!selectedGroupId) {
        return;
    }

    // Filter and add models for the selected equipment group
    const filteredModels = models.filter(model => model.equipment_group_id == selectedGroupId);

    if (filteredModels.length > 0) {
        $.each(filteredModels, function(index, model) {
            dropdown.append(
                $('<option></option>')
                    .attr('value', model.id)
                    .text(model.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${filteredModels.length} model options for equipment group ${selectedGroupId}`);
    } else {
        window.BOMAdvancedSearch.logDebug(`No models available for equipment group ${selectedGroupId}`);
    }

    // Reinitialize Select2 if available
    window.BOMAdvancedSearch.safeSelect2Initialize(dropdown);
};

/**
 * Populate the Asset Number dropdown based on selected Model
 */
window.BOMAdvancedSearch.populateAssetNumberDropdown = function(assetNumbers, selectedModelId) {
    const dropdown = $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.assetNumber);

    // Clear existing options except the first one
    dropdown.find('option:not(:first)').remove();

    if (!selectedModelId) {
        return;
    }

    // Filter and add asset numbers for the selected model
    const filteredAssets = assetNumbers.filter(asset => asset.model_id == selectedModelId);

    if (filteredAssets.length > 0) {
        $.each(filteredAssets, function(index, asset) {
            dropdown.append(
                $('<option></option>')
                    .attr('value', asset.id)
                    .text(asset.number)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${filteredAssets.length} asset number options for model ${selectedModelId}`);
    } else {
        window.BOMAdvancedSearch.logDebug(`No asset numbers available for model ${selectedModelId}`);
    }

    // Reinitialize Select2 if available
    window.BOMAdvancedSearch.safeSelect2Initialize(dropdown);
};

/**
 * Populate the Location dropdown based on selected Model
 */
window.BOMAdvancedSearch.populateLocationDropdown = function(locations, selectedModelId) {
    const dropdown = $('#' + window.BOMAdvancedSearch.DROPDOWN_IDS.location);

    // Clear existing options except the first one
    dropdown.find('option:not(:first)').remove();

    if (!selectedModelId) {
        return;
    }

    // Filter and add locations for the selected model
    const filteredLocations = locations.filter(location => location.model_id == selectedModelId);

    if (filteredLocations.length > 0) {
        $.each(filteredLocations, function(index, location) {
            dropdown.append(
                $('<option></option>')
                    .attr('value', location.id)
                    .text(location.name)
            );
        });
        window.BOMAdvancedSearch.logDebug(`Added ${filteredLocations.length} location options for model ${selectedModelId}`);
    } else {
        window.BOMAdvancedSearch.logDebug(`No locations available for model ${selectedModelId}`);
    }

    // Reinitialize Select2 if available
    window.BOMAdvancedSearch.safeSelect2Initialize(dropdown);
};

/**
 * Safely initialize Select2 on a dropdown
 */
window.BOMAdvancedSearch.safeSelect2Initialize = function(dropdown) {
    if (!$.fn.select2) {
        return;
    }

    try {
        // Make sure dropdown exists and is in the DOM
        if (dropdown.length === 0 || !document.getElementById(dropdown.attr('id'))) {
            window.BOMAdvancedSearch.logDebug(`Skip Select2 init for non-existent element: ${dropdown.attr('id')}`);
            return;
        }

        // Check if Select2 is already initialized
        if (dropdown.data('select2')) {
            // Safely destroy existing instance
            try {
                dropdown.select2('destroy');
            } catch (e) {
                // Ignore errors from destroy
            }
        }

        // Initialize Select2
        dropdown.select2({
            width: '100%',
            dropdownParent: $('#advancedSearchForm')
        });
    } catch (e) {
        console.warn('Error initializing Select2:', e);
    }
};

/**
 * Clear multiple dropdowns
 */
window.BOMAdvancedSearch.clearDropdowns = function(dropdownIds) {
    dropdownIds.forEach(id => {
        const dropdown = $('#' + id);
        dropdown.find('option:not(:first)').remove();

        // Reset Select2 if available
        if ($.fn.select2 && dropdown.data('select2')) {
            try {
                dropdown.val('').trigger('change');
            } catch (e) {
                console.warn(`Error resetting Select2 for ${id}:`, e);
            }
        }
    });
};

/**
 * Reset all dropdowns to their initial state
 */
window.BOMAdvancedSearch.resetAllDropdowns = function() {
    window.BOMAdvancedSearch.logDebug('Resetting all dropdowns');

    // Reset each dropdown to its default state
    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(id => {
        const dropdown = $('#' + id);
        dropdown.find('option:not(:first)').remove();
        dropdown.val('');

        // Reset Select2 if available
        if ($.fn.select2 && dropdown.data('select2')) {
            try {
                dropdown.val('').trigger('change');
            } catch (e) {
                console.warn(`Error resetting Select2 for ${id}:`, e);
            }
        }
    });

    // Re-populate the area dropdown to return to initial state
    window.populateDropdownsForPartsPosition();
};

/**
 * Initialize Select2 for all dropdowns
 */
window.BOMAdvancedSearch.initializeSelect2 = function() {
    if ($.fn.select2) {
        try {
            window.BOMAdvancedSearch.logDebug('Initializing Select2 for all dropdowns');

            // Initialize Select2 for each dropdown
            Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(id => {
                const dropdown = $('#' + id);
                window.BOMAdvancedSearch.safeSelect2Initialize(dropdown);
            });

            window.BOMAdvancedSearch.logDebug('Select2 initialization complete');
        } catch (e) {
            console.error('Error initializing Select2:', e);
        }
    } else {
        window.BOMAdvancedSearch.logDebug('Select2 library not available');
    }
};

/**
 * Ensure all form elements are visible
 */
window.BOMAdvancedSearch.ensureElementsVisible = function() {
    // Make sure the form container is visible
    $('#advancedSearchForm, #advancedSearchFilterForm').css({
        'display': 'block',
        'visibility': 'visible'
    });

    // Make sure all select elements are visible
    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'block';
            element.style.visibility = 'visible';
            element.style.opacity = '1';
        } else {
            window.BOMAdvancedSearch.logDebug(`WARNING: Element with ID ${id} not found!`);
        }
    });

    // The rest of your function...
};

/**
 * Show loading indicators for all dropdowns
 */
window.BOMAdvancedSearch.showLoadingIndicators = function() {
    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(id => {
        const dropdown = $('#' + id);
        dropdown.find('option:not(:first)').remove();
        dropdown.find('option:first').text('Loading...');
        dropdown.prop('disabled', true);
    });
};

/**
 * Hide loading indicators for all dropdowns
 */
window.BOMAdvancedSearch.hideLoadingIndicators = function() {
    Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS).forEach(id => {
        const dropdown = $('#' + id);
        const defaultText = id.replace('filter_', '').replace('Dropdown', '');
        dropdown.find('option:first').text(`Select a ${defaultText}...`);
        dropdown.prop('disabled', false);
    });
};

/**
 * Display an error message to the user
 */
window.BOMAdvancedSearch.showError = function(message) {
    // Create error element if it doesn't exist
    if ($('#searchFormError').length === 0) {
        const formContainer = $('#advancedSearchForm, #advancedSearchFilterForm').first();
        formContainer.prepend(
            $('<div id="searchFormError"></div>')
                .css({
                    'background-color': '#f8d7da',
                    'color': '#721c24',
                    'padding': '10px',
                    'margin-bottom': '15px',
                    'border': '1px solid #f5c6cb',
                    'border-radius': '4px'
                })
        );
    }

    // Display the error message
    $('#searchFormError').text(message).show();

    // Log the error
    window.BOMAdvancedSearch.logDebug('ERROR: ' + message);
};

/**
 * Log debug information to console and debug info area
 */
window.BOMAdvancedSearch.logDebug = function(message, data) {
    // Log to console
    if (data) {
        console.log('[BOM Debug]', message, data);
    } else {
        console.log('[BOM Debug]', message);
    }

    // Log to debug info area if it exists
    const debugArea = $('#searchDebugInfo');
    if (debugArea.length > 0) {
        const timestamp = new Date().toISOString().substring(11, 23);
        let logMessage = `[${timestamp}] ${message}`;

        if (data) {
            try {
                logMessage += '\n' + JSON.stringify(data, null, 2);
            } catch (e) {
                logMessage += '\n[Object cannot be stringified]';
            }
        }

        debugArea.append($('<div></div>').text(logMessage));

        // Auto-scroll to bottom
        debugArea.scrollTop(debugArea[0].scrollHeight);
    }
};

/**
 * Toggle debug information visibility
 */
window.BOMAdvancedSearch.toggleDebugInfo = function() {
    const debugArea = $('#searchDebugInfo');
    debugArea.toggle();
};

/**
 * Function to ensure results stay visible
 */
window.BOMAdvancedSearch.ensureResultsStayVisible = function() {
    window.BOMAdvancedSearch.logDebug("Using standalone results page approach");

    // Since we're now using a separate page for results, we just need to clear the flags
    // No need to manipulate DOM elements as results are on another page
    localStorage.removeItem('expectingResults');
    localStorage.removeItem('lastSearchTime');

    window.BOMAdvancedSearch.logDebug("Cleared result flags for standalone page");

    // No need for setTimeout checks since we're using a full page redirect
};

/**
 * Global clear function that works for both forms
 */
window.clearAdvancedForm = function() {
    console.log("clearAdvancedForm called");

    // Check for both possible form IDs
    const advFilterForm = document.getElementById('advancedSearchFilterForm');
    const advContentForm = document.getElementById('advancedSearchFormContent');

    console.log("Forms found:", {
        advFilterForm: !!advFilterForm,
        advContentForm: !!advContentForm
    });

    // Find the form that currently exists
    const formToReset = advFilterForm || (advContentForm ? advContentForm.form : null);
    if (formToReset) {
        console.log("Resetting form:", formToReset.id);
        formToReset.reset();

        // Also clear all selects to default state
        window.BOMAdvancedSearch.resetAllDropdowns();
    } else {
        console.warn("No form found to reset!");
    }
};

// Override any document location changes after search
var originalAssign = window.location.assign;
window.location.assign = function(url) {
    console.log("Location change attempted to:", url);

    // If we're expecting results, only allow redirect to view_bill_of_material
    if (localStorage.getItem('expectingResults') === 'true' &&
        !url.includes('view_bill_of_material')) {
        console.log("Blocking redirect that would hide results");
        return;
    }

    // Otherwise proceed with normal behavior
    originalAssign.call(window.location, url);
};

/**
 * Add a debug function for diagnosing form state issues
 */
window.debugForms = function() {
    console.group("Advanced Search Form Debug Info");

    // Check for both possible forms
    const advancedSearchForm = document.getElementById('advancedSearchForm');
    const advancedSearchFilterForm = document.getElementById('advancedSearchFilterForm');
    const advancedSearchFormContent = document.getElementById('advancedSearchFormContent');

    console.log("Form containers found:", {
        advancedSearchForm: !!advancedSearchForm,
        advancedSearchFilterForm: !!advancedSearchFilterForm,
        advancedSearchFormContent: !!advancedSearchFormContent
    });

    // Check all form attributes
    if (advancedSearchFilterForm) {
        console.log("advancedSearchFilterForm attributes:", {
            method: advancedSearchFilterForm.method,
            action: advancedSearchFilterForm.action,
            id: advancedSearchFilterForm.id,
            display: getComputedStyle(advancedSearchFilterForm).display,
            visible: advancedSearchFilterForm.offsetParent !== null
        });
    }

    if (advancedSearchFormContent) {
        console.log("advancedSearchFormContent attributes:", {
            method: advancedSearchFormContent.method,
            action: advancedSearchFormContent.action,
            id: advancedSearchFormContent.id,
            display: getComputedStyle(advancedSearchFormContent).display,
            visible: advancedSearchFormContent.offsetParent !== null
        });
    }

    // Check dropdown elements
    const dropdownIds = Object.values(window.BOMAdvancedSearch.DROPDOWN_IDS);

    const dropdownsInfo = {};

    dropdownIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            dropdownsInfo[id] = {
                exists: true,
                options: element.options.length,
                value: element.value,
                display: getComputedStyle(element).display,
                visible: element.offsetParent !== null
            };
        } else {
            dropdownsInfo[id] = { exists: false };
        }
    });

    console.log("Dropdown elements:", dropdownsInfo);

    console.groupEnd();

    return "Form debug completed - see console for details";
};

// Initialize when document is ready - ONE document.ready handler
$(document).ready(function() {
    window.BOMAdvancedSearch.logDebug('Document ready - Advanced Search JS loaded');

    // 1. Check jQuery and Select2 availability
    if (window.jQuery) {
        window.BOMAdvancedSearch.logDebug('jQuery is available');

        if ($.fn.select2) {
            window.BOMAdvancedSearch.logDebug('Select2 is available');
        } else {
            window.BOMAdvancedSearch.logDebug('WARNING: Select2 library is not available');
        }
    } else {
        console.error('jQuery is not available - this should never happen');
    }

    // 2. Determine what page we're on
    const isResultsPage = window.location.pathname.includes('view_bill_of_material') ||
                         window.location.pathname.includes('bill_of_material_results') ||
                         window.location.pathname.includes('/create_bill_of_material');

    if (isResultsPage) {
        window.BOMAdvancedSearch.logDebug("We're on the results page - minimal initialization");
        // Clear any leftover flags
        sessionStorage.removeItem('clearAdvancedForm');
        // Skip form initialization on results page
        return;
    }

    // 3. Find the form with a SINGLE selector - don't use the comma selector that can find duplicates
    var advancedForm = $('#advancedSearchFilterForm');

    // Debug what we found
    window.BOMAdvancedSearch.logDebug("Found form count: " + advancedForm.length);
    if (advancedForm.length > 1) {
        window.BOMAdvancedSearch.logDebug("WARNING: Found multiple forms with the same ID!");
        // Log each form's parent for troubleshooting
        advancedForm.each(function(idx) {
            window.BOMAdvancedSearch.logDebug(`Form ${idx} parent: ${$(this).parent().attr('id') || 'unknown'}`);
        });
        // Keep only the first one to avoid duplicates
        advancedForm = advancedForm.eq(0);
    }

    if (advancedForm.length) {
        window.BOMAdvancedSearch.logDebug("Using search form ID: " + advancedForm.attr('id'));

        // Remove any existing handlers to prevent duplicates
        advancedForm.off('submit');

        // Add submit handler
        advancedForm.on('submit', function(event) {
            window.BOMAdvancedSearch.logDebug("Form submitted: " + this.id);

            // Verify form method
            if ($(this).attr('method').toUpperCase() !== 'POST') {
                window.BOMAdvancedSearch.logDebug("Form method is not POST! Correcting...");
                $(this).attr('method', 'POST');
            }

            // Verify form has values
            let hasValue = false;
            $(this).find('select').each(function() {
                if ($(this).val()) {
                    hasValue = true;
                    return false; // Break the loop
                }
            });

            if (!hasValue) {
                window.BOMAdvancedSearch.logDebug("No values selected in form");
                window.BOMAdvancedSearch.showError('Please select at least one search criteria');
                event.preventDefault();
                return false;
            }

            // Get form data for debugging
            const formData = $(this).serialize();
            window.BOMAdvancedSearch.logDebug("Form data: " + formData);
            window.BOMAdvancedSearch.logDebug("Form action: " + $(this).attr('action'));
            window.BOMAdvancedSearch.logDebug("Form method: " + $(this).attr('method'));

            // Set flag to clear form on next view
            sessionStorage.setItem('clearAdvancedForm', 'true');

            // Let the form submit normally
            return true;
        });
    } else {
        window.BOMAdvancedSearch.logDebug("Warning: Advanced search form not found!");
    }

    // 4. Add handler for "Back to Search" links
    $('.back-to-form').off('click').on('click', function(e) {
        window.BOMAdvancedSearch.logDebug("Back to search clicked, setting clear flag");
        // Prevent default link behavior
        e.preventDefault();
        // Set the clear flag
        sessionStorage.setItem('clearAdvancedForm', 'true');
        // Navigate to the target URL
        window.location.href = $(this).attr('href');
    });

    // 5. Check for the clear flag when page loads
    if (sessionStorage.getItem('clearAdvancedForm') === 'true') {
        window.BOMAdvancedSearch.logDebug("Clear flag found on page load");
        // Will be processed in populateDropdownsForPartsPosition
        // Don't remove the flag yet - it will be used by the populate function
    }

    // 6. Only initialize the form when it's visible
    const advancedFormContainer = document.getElementById('advancedSearchForm');
    if (advancedFormContainer && (advancedFormContainer.style.display !== 'none' && getComputedStyle(advancedFormContainer).display !== 'none')) {
        window.BOMAdvancedSearch.logDebug('Advanced form is visible on page load, populating dropdowns');
        window.populateDropdownsForPartsPosition();
    } else {
        window.BOMAdvancedSearch.logDebug('Advanced form is hidden on page load');
    }

    // 7. Add debug toggle (only in development)
    if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
        // Only add if it doesn't already exist
        if ($('#toggleDebugBtn').length === 0) {
            $('<button type="button" id="toggleDebugBtn">Toggle Debug Info</button>')
                .css({
                    'position': 'fixed',
                    'bottom': '10px',
                    'right': '10px',
                    'z-index': '9999',
                    'background': '#333',
                    'color': '#fff',
                    'border': 'none',
                    'padding': '5px 10px',
                    'cursor': 'pointer'
                })
                .click(window.BOMAdvancedSearch.toggleDebugInfo)
                .appendTo('body');
        }
        // Show debug area in development environment
        $('#searchDebugInfo').show();
    }
});