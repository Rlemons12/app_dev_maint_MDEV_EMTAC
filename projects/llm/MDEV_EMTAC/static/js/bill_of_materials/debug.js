/**
 * BOM Dropdown Debugging Script
 * This script helps diagnose and fix issues with dropdowns in the Bill of Materials search interface
 *
 * How to use:
 * 1. Copy this entire script
 * 2. Open your browser's developer console (F12 or right-click > Inspect > Console)
 * 3. Paste and press Enter to run
 * 4. Check the console for diagnostic information
 * 5. The script will attempt to fix the dropdown issues automatically
 */

// Create a namespace for our debugging tools
window.BOMDebugger = {
    // Configuration
    config: {
        endpointUrl: '/get_parts_position_data',
        dropdownIds: {
            area: 'filter_areaDropdown',
            equipmentGroup: 'filter_equipmentGroupDropdown',
            model: 'filter_modelDropdown',
            assetNumber: 'filter_assetNumberDropdown',
            location: 'filter_locationDropdown'
        },
        bomDropdownIds: {
            area: 'bom_areaDropdown',
            equipmentGroup: 'bom_equipmentGroupDropdown',
            model: 'bom_modelDropdown',
            assetNumber: 'bom_assetNumberDropdown',
            location: 'bom_locationDropdown'
        }
    },

    // Logging with timestamp
    log: function(message, data) {
        const timestamp = new Date().toISOString().substr(11, 8);
        console.log(`[${timestamp}] [BOMDebugger] ${message}`);
        if (data !== undefined) {
            console.log(data);
        }
    },

    // Error logging
    error: function(message, error) {
        const timestamp = new Date().toISOString().substr(11, 8);
        console.error(`[${timestamp}] [BOMDebugger] ERROR: ${message}`);
        if (error) {
            console.error(error);
        }
    },

    // Check if jQuery and Select2 are available
    checkDependencies: function() {
        this.log("Checking dependencies...");

        if (typeof jQuery === 'undefined') {
            this.error("jQuery is not available!");
            return false;
        }

        this.log("jQuery version: " + jQuery.fn.jquery);

        if (typeof jQuery.fn.select2 === 'undefined') {
            this.log("Select2 is not available - some enhanced features will be disabled");
        } else {
            this.log("Select2 is available");
        }

        return true;
    },

    // Check for the existence of DOM elements
    checkElements: function() {
        this.log("Checking for required DOM elements...");

        // Check for the form container
        const formExists = $('#advancedSearchForm').length > 0;
        this.log(`Advanced search form exists: ${formExists}`);

        if (!formExists) {
            this.error("The advanced search form is missing from the DOM!");
            return false;
        }

        // Check for each dropdown
        let allFound = true;
        for (const [key, id] of Object.entries(this.config.dropdownIds)) {
            const elementExists = $(`#${id}`).length > 0;
            this.log(`${key} dropdown (#${id}) exists: ${elementExists}`);
            if (!elementExists) {
                allFound = false;
            }
        }

        // Check for BOM dropdowns
        this.log("Checking BOM-specific dropdowns...");
        let bomDropdownsFound = true;
        for (const [key, id] of Object.entries(this.config.bomDropdownIds)) {
            const elementExists = $(`#${id}`).length > 0;
            this.log(`BOM ${key} dropdown (#${id}) exists: ${elementExists}`);
            if (!elementExists) {
                bomDropdownsFound = false;
            }
        }

        return { allFound, bomDropdownsFound };
    },

    // Make a test AJAX request to fetch dropdown data
    testDataFetch: function() {
        this.log(`Testing data fetch from ${this.config.endpointUrl}...`);

        return new Promise((resolve, reject) => {
            $.ajax({
                url: this.config.endpointUrl,
                type: 'GET',
                dataType: 'json',
                success: (data) => {
                    this.log("Data received successfully:", data);

                    // Validate the data structure
                    const requiredProperties = ['areas', 'equipment_groups', 'models', 'asset_numbers', 'locations'];
                    let isValid = true;
                    let missingProps = [];

                    requiredProperties.forEach(prop => {
                        if (!data[prop] || !Array.isArray(data[prop])) {
                            isValid = false;
                            missingProps.push(prop);
                        }
                    });

                    if (!isValid) {
                        this.error(`Invalid data structure. Missing or invalid properties: ${missingProps.join(', ')}`);
                        reject(`Invalid data structure. Missing: ${missingProps.join(', ')}`);
                    } else {
                        this.log("Data structure is valid");
                        resolve(data);
                    }
                },
                error: (xhr, status, error) => {
                    this.error(`AJAX request failed: ${status} - ${error}`);
                    reject(error);
                }
            });
        });
    },

    // Force the form to be visible
    forceFormVisibility: function() {
        this.log("Forcing advanced search form to be visible...");

        const form = $('#advancedSearchForm');
        if (form.length > 0) {
            const wasHidden = form.is(':hidden');
            this.log(`Form was hidden: ${wasHidden}`);

            // Apply direct styles to force visibility
            form.css({
                'display': 'block',
                'visibility': 'visible',
                'opacity': 1,
                'height': 'auto',
                'overflow': 'visible',
                'position': 'relative',
                'z-index': 1000
            });

            this.log("Applied direct styles to force form visibility");
            return wasHidden;
        } else {
            this.error("Could not find advanced search form!");
            return false;
        }
    },

    // Force dropdowns to be visible
    forceDropdownVisibility: function() {
        this.log("Forcing dropdowns to be visible...");

        for (const [key, id] of Object.entries(this.config.dropdownIds)) {
            const dropdown = $(`#${id}`);
            if (dropdown.length > 0) {
                dropdown.css({
                    'display': 'block',
                    'visibility': 'visible',
                    'opacity': 1,
                    'width': '100%',
                    'max-width': '600px',
                    'height': 'auto',
                    'position': 'relative',
                    'z-index': 999
                });

                this.log(`Applied direct styles to ${key} dropdown (#${id})`);
            }
        }
    },

    // Directly populate the area dropdown with data
    populateAreaDropdown: function(areas) {
        const areaDropdownId = this.config.dropdownIds.area;
        const areaDropdown = $(`#${areaDropdownId}`);

        if (areaDropdown.length === 0) {
            this.error(`Area dropdown (#${areaDropdownId}) not found!`);
            return false;
        }

        this.log(`Populating area dropdown (#${areaDropdownId}) with ${areas.length} items...`);

        // Clear existing options except the first one
        areaDropdown.find('option:not(:first)').remove();

        // Add each area as an option
        if (areas && areas.length > 0) {
            $.each(areas, (index, area) => {
                areaDropdown.append(
                    $('<option></option>')
                        .attr('value', area.id)
                        .text(area.name)
                );
            });

            this.log(`Added ${areas.length} area options`);
            return true;
        } else {
            this.log("No areas available to populate dropdown");
            return false;
        }
    },

    // Set up cascade events for the dropdowns
    setupDropdownEvents: function(data) {
        this.log("Setting up dropdown cascade events...");

        const { dropdownIds } = this.config;

        // Area dropdown change event
        $(`#${dropdownIds.area}`).off('change').on('change', function() {
            const selectedAreaId = $(this).val();
            console.log(`Area dropdown changed to: ${selectedAreaId}`);

            // Get equipment group dropdown
            const equipmentGroupDropdown = $(`#${dropdownIds.equipmentGroup}`);

            // Clear existing options except the first one
            equipmentGroupDropdown.find('option:not(:first)').remove();

            if (!selectedAreaId) {
                return;
            }

            // Filter and add equipment groups for the selected area
            const filteredGroups = data.equipment_groups.filter(group =>
                group.area_id == selectedAreaId
            );

            if (filteredGroups.length > 0) {
                $.each(filteredGroups, function(index, group) {
                    equipmentGroupDropdown.append(
                        $('<option></option>')
                            .attr('value', group.id)
                            .text(group.name)
                    );
                });
                console.log(`Added ${filteredGroups.length} equipment group options`);
            } else {
                console.log(`No equipment groups available for area ${selectedAreaId}`);
            }

            // Trigger change to cascade updates
            equipmentGroupDropdown.trigger('change');
        });

        // Equipment group dropdown change event
        $(`#${dropdownIds.equipmentGroup}`).off('change').on('change', function() {
            const selectedGroupId = $(this).val();
            console.log(`Equipment group dropdown changed to: ${selectedGroupId}`);

            // Get model dropdown
            const modelDropdown = $(`#${dropdownIds.model}`);

            // Clear existing options except the first one
            modelDropdown.find('option:not(:first)').remove();

            if (!selectedGroupId) {
                return;
            }

            // Filter and add models for the selected equipment group
            const filteredModels = data.models.filter(model =>
                model.equipment_group_id == selectedGroupId
            );

            if (filteredModels.length > 0) {
                $.each(filteredModels, function(index, model) {
                    modelDropdown.append(
                        $('<option></option>')
                            .attr('value', model.id)
                            .text(model.name)
                    );
                });
                console.log(`Added ${filteredModels.length} model options`);
            } else {
                console.log(`No models available for equipment group ${selectedGroupId}`);
            }

            // Trigger change to cascade updates
            modelDropdown.trigger('change');
        });

        // Model dropdown change event
        $(`#${dropdownIds.model}`).off('change').on('change', function() {
            const selectedModelId = $(this).val();
            console.log(`Model dropdown changed to: ${selectedModelId}`);

            // Get asset number and location dropdowns
            const assetNumberDropdown = $(`#${dropdownIds.assetNumber}`);
            const locationDropdown = $(`#${dropdownIds.location}`);

            // Clear existing options except the first one
            assetNumberDropdown.find('option:not(:first)').remove();
            locationDropdown.find('option:not(:first)').remove();

            if (!selectedModelId) {
                return;
            }

            // Filter and add asset numbers for the selected model
            const filteredAssets = data.asset_numbers.filter(asset =>
                asset.model_id == selectedModelId
            );

            if (filteredAssets.length > 0) {
                $.each(filteredAssets, function(index, asset) {
                    assetNumberDropdown.append(
                        $('<option></option>')
                            .attr('value', asset.id)
                            .text(asset.number)
                    );
                });
                console.log(`Added ${filteredAssets.length} asset number options`);
            } else {
                console.log(`No asset numbers available for model ${selectedModelId}`);
            }

            // Filter and add locations for the selected model
            const filteredLocations = data.locations.filter(location =>
                location.model_id == selectedModelId
            );

            if (filteredLocations.length > 0) {
                $.each(filteredLocations, function(index, location) {
                    locationDropdown.append(
                        $('<option></option>')
                            .attr('value', location.id)
                            .text(location.name)
                    );
                });
                console.log(`Added ${filteredLocations.length} location options`);
            } else {
                console.log(`No locations available for model ${selectedModelId}`);
            }

            // Initialize Select2 if available
            if ($.fn.select2) {
                try {
                    assetNumberDropdown.select2({
                        width: '100%'
                    });
                    locationDropdown.select2({
                        width: '100%'
                    });
                } catch (e) {
                    console.warn('Error initializing Select2:', e);
                }
            }
        });

        this.log("Dropdown cascade events set up successfully");
        return true;
    },

    // Run all fixes
    runAllFixes: function() {
        this.log("Running all fixes...");

        if (!this.checkDependencies()) {
            this.error("Critical dependencies missing, cannot proceed with fixes");
            return false;
        }

        const { allFound, bomDropdownsFound } = this.checkElements();
        if (!allFound) {
            this.error("Some required elements are missing, fixes may not work completely");
        }

        // Force form and dropdown visibility
        this.forceFormVisibility();
        this.forceDropdownVisibility();

        // Test data fetch and populate dropdowns
        this.testDataFetch()
            .then((data) => {
                // Populate area dropdown
                const areaPopulated = this.populateAreaDropdown(data.areas);
                if (areaPopulated) {
                    // Set up cascade events
                    this.setupDropdownEvents(data);

                    // Trigger change on area dropdown to populate the equipment group dropdown
                    const areaDropdown = $(`#${this.config.dropdownIds.area}`);
                    if (areaDropdown.val()) {
                        areaDropdown.trigger('change');
                    }

                    this.log("Dropdowns should now be populated and working");
                }

                // Also fix BOM-specific dropdowns if they exist
                if (bomDropdownsFound) {
                    this.log("Fixing BOM-specific dropdowns...");

                    // Re-run the populateDropdownsBOM function
                    if (typeof window.populateDropdownsBOM === 'function') {
                        window.populateDropdownsBOM();
                        this.log("Called populateDropdownsBOM() function");
                    } else {
                        this.error("populateDropdownsBOM function not found");
                    }
                }

                this.log("All fixes have been applied successfully");

                // Final check - is the form actually visible now?
                const formVisible = $('#advancedSearchForm').is(':visible');
                this.log(`Final check - form is visible: ${formVisible}`);

                // Add a debug button for the user to manually toggle the form
                if (!$('#debugToggleFormBtn').length) {
                    $('body').append(`
                        <button id="debugToggleFormBtn" 
                                style="position: fixed; top: 10px; right: 10px; z-index: 9999; 
                                       background: #dc3545; color: white; padding: 5px 10px; 
                                       border: none; border-radius: 3px; cursor: pointer;">
                            Toggle Advanced Search Form
                        </button>
                    `);

                    $('#debugToggleFormBtn').on('click', () => {
                        const form = $('#advancedSearchForm');
                        if (form.is(':visible')) {
                            form.hide();
                            this.log("Form manually hidden");
                        } else {
                            form.show();
                            this.log("Form manually shown");
                        }
                    });

                    this.log("Added debug toggle button in top-right corner");
                }
            })
            .catch((error) => {
                this.error("Failed to fetch dropdown data", error);
            });

        return true;
    }
};

// Run the debugger
BOMDebugger.log("BOM Dropdown Debugger started");

// Add a button to the page for manual running
$('body').append(`
    <button id="runBOMDebugger" 
            style="position: fixed; bottom: 10px; right: 10px; z-index: 9999; 
                   background: #007bff; color: white; padding: 5px 10px; 
                   border: none; border-radius: 3px; cursor: pointer;">
        Fix BOM Dropdowns
    </button>
`);

$('#runBOMDebugger').on('click', function() {
    BOMDebugger.runAllFixes();
});

// Automatically run the fixes
BOMDebugger.runAllFixes();