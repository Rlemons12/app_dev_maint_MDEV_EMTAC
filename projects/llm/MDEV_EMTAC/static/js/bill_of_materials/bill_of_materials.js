// Function to fetch data and populate dropdowns for the BOM form
function populateDropdownsBOM() {
    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#bom_areaDropdown'), dataKey: 'areas' },
        { element: $('#bom_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#bom_modelDropdown'), dataKey: 'models' },
        { element: $('#bom_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#bom_locationDropdown'), dataKey: 'locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_bom_list_data', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            // Populate areas dropdown
            var areaDropdown = $('#bom_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            areaDropdown.append('<option value="">Select...</option>');
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#bom_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options
                equipmentGroupDropdown.append('<option value="">Select...</option>');

                // Populate equipment group dropdown with associated groups based on selected area
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change(); // Trigger change event for equipment group dropdown
            });

            // Event listener for equipment group dropdown change
            $('#bom_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#bom_modelDropdown');
                modelDropdown.empty(); // Clear existing options
                modelDropdown.append('<option value="">Select...</option>');

                // Populate model dropdown with associated models based on selected equipment group
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change(); // Trigger change event for model dropdown
            });

            // Event listener for model dropdown change
            $('#bom_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#bom_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options
                assetNumberDropdown.append('<option value="">Select...</option>');

                // Populate asset number dropdown with associated asset numbers based on selected model
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for asset number dropdown

                // Populate location dropdown with associated locations based on selected model
                var locationDropdown = $('#bom_locationDropdown');
                locationDropdown.empty(); // Clear existing options
                locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });

                // Only initialize Select2 if it's available
                if ($.fn.select2) {
                    try {
                        locationDropdown.select2();
                        assetNumberDropdown.select2();
                        areaDropdown.select2();
                    } catch (e) {
                        console.warn("Select2 initialization failed:", e);
                    }
                }
            });

            // Call change event to populate equipment group dropdown initially based on the default selected area
            areaDropdown.change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

// Add logic to allow users to manually type in Asset Number and Location
$(document).ready(function() {
    $('#bom_assetNumberInput').on('input', function() {
        var assetNumber = $(this).val();
        if (assetNumber.length > 1) {
            $('#bom_assetNumberDropdown').prop('disabled', true); // Disable dropdown when typing in manually
        } else {
            $('#bom_assetNumberDropdown').prop('disabled', false); // Enable dropdown if input is cleared
        }
    });

    $('#bom_locationInput').on('input', function() {
        var location = $(this).val();
        if (location.length > 1) {
            $('#bom_locationDropdown').prop('disabled', true); // Disable dropdown when typing in manually
        } else {
            $('#bom_locationDropdown').prop('disabled', false); // Enable dropdown if input is cleared
        }
    });

    // Call the function to populate dropdowns when the page loads
    populateDropdownsBOM();
});

// AJAX‑ify the pagination links so we preserve the partial=true flag
$(document).on('click', '#results-container .pagination a', function(e) {
    e.preventDefault();

    // Get the href and force partial rendering
    const href = $(this).attr('href');
    const url  = href + (href.includes('partial=') ? '' : '&partial=true');

    // Load *only* the results partial into our container
    $('#results-container').load(url, function() {
        console.log('Loaded page via AJAX:', url);
        // Re‑invoke any post‑load logic, e.g. showing the results container
        if (window.BOMAdvancedSearch && window.BOMAdvancedSearch.ensureResultsStayVisible) {
            window.BOMAdvancedSearch.ensureResultsStayVisible();
        }
    });
});

// AJAX‑ify the Items‑per‑page dropdown
$(document).on('change', '#results-container #per_page', function(e) {
    e.preventDefault();

    // Read the new page size and reset to the first page
    const perPage = $(this).val();
    const action = $(this).closest('form').attr('action');
    const url    = `${action}?per_page=${perPage}&index=0&partial=true`;

    // Load only the results partial
    $('#results-container').load(url, function() {
        console.log('Loaded per_page via AJAX:', url);

        // After loading, re‑show results (hide the forms)
        if (window.BOMAdvancedSearch && window.BOMAdvancedSearch.ensureResultsStayVisible) {
            window.BOMAdvancedSearch.ensureResultsStayVisible();
        }
    });
});
