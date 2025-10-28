function populateTroubleshootingGuideDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var documentDropdowns = [
        { element: $('#tsg_areaDropdown'), dataKey: 'areas' },
        { element: $('#tsg_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#tsg_modelDropdown'), dataKey: 'models' },
        { element: $('#tsg_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#tsg_locationDropdown'), dataKey: 'locations' },
        { element: $('#tsg_documentSearchDropdown'), dataKey: 'documents' },
        { element: $('#tsg_problemImageSearchDropdown'), dataKey: 'images' },
        { element: $('#tsg_solutionImageSearchDropdown'), dataKey: 'images' },
        { element: $('#tsg_drawingSearchDropdown'), dataKey: 'drawings' },
        { element: $('#tsg_partSearchDropdown'), dataKey: 'parts' }
    ];

    // AJAX request to fetch data for Troubleshooting Guide form dropdowns
    $.ajax({
        url: '/get_troubleshooting_guide_data_bp', // URL to fetch data from Flask route
        type: 'GET',
        success: function(data) {
            // Populate dropdowns
            documentDropdowns.forEach(function(dropdown) {
                var dropdownElement = dropdown.element;
                var dataKey = dropdown.dataKey;
                dropdownElement.empty(); // Clear existing options
                dropdownElement.append('<option value="">Select...</option>'); // Add a placeholder option
                $.each(data[dataKey], function(index, item) {
                    dropdownElement.append('<option value="' + item.id + '">' + (item.name || item.title || item.number || item.part_number) + '</option>');
                });
            });

            // Initialize Select2 for image search dropdown
            $('#tsg_problemImageSearchDropdown').select2();
            $('#tsg_solutionImageSearchDropdown').select2();

            // Populate Equipment Group dropdown initially based on the default selected Area
            $('#tsg_areaDropdown').change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#tsg_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options
                equipmentGroupDropdown.append('<option value="">Select...</option>'); // Add a placeholder option
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change(); // Trigger change event for Equipment Group dropdown
            });

            // Populate Model dropdown initially based on the default selected Equipment Group
            $('#tsg_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#tsg_modelDropdown');
                modelDropdown.empty(); // Clear existing options
                modelDropdown.append('<option value="">Select...</option>'); // Add a placeholder option
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change(); // Trigger change event for Model dropdown
            });

            // Populate Asset Number and Location dropdowns initially based on the default selected Model
            $('#tsg_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#tsg_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options
                assetNumberDropdown.append('<option value="">Select...</option>'); // Add a placeholder option
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for Asset Number dropdown

                var locationDropdown = $('#tsg_locationDropdown');
                locationDropdown.empty(); // Clear existing options
                locationDropdown.append('<option value="">Select...</option>'); // Add a placeholder option
                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
            });

            // Trigger initial change event to populate the dropdowns
            $('#tsg_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateTroubleshootingGuideDropdowns();
});
