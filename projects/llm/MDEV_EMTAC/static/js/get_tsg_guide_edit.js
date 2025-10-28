function populateDropdownsForPartsPosition() {
    // Define an array of dropdown elements along with their corresponding data keys
    var partDropdowns = [
        { element: $('#tsg_edit_areaDropdown'), dataKey: 'areas' },
        { element: $('#tsg_edit_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#tsg_edit_modelDropdown'), dataKey: 'models' },
        { element: $('#tsg_edit_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#tsg_edit_locationDropdown'), dataKey: 'locations' }
    ];

    // Clear the previous search results
    $('#searchResults').empty();

    // AJAX request to fetch data for dropdowns
    $.ajax({
        url: '/get_list_data', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            // Populate Area dropdown
            var areaDropdown = $('#tsg_edit_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            areaDropdown.append('<option value="">Select...</option>');

            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for Area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#tsg_edit_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options
                equipmentGroupDropdown.append('<option value="">Select...</option>');

                // Populate Equipment Group dropdown based on selected Area
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change(); // Trigger change event for Equipment Group dropdown
            });

            // Event listener for Equipment Group dropdown change
            $('#tsg_edit_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#tsg_edit_modelDropdown');
                modelDropdown.empty(); // Clear existing options
                modelDropdown.append('<option value="">Select...</option>');

                // Populate Model dropdown based on selected Equipment Group
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change(); // Trigger change event for Model dropdown
            });

            // Event listener for Model dropdown change
            $('#tsg_edit_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#tsg_edit_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options
                assetNumberDropdown.append('<option value="">Select...</option>');

                // Populate Asset Number dropdown based on selected Model
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for Asset Number dropdown

                // Populate Location dropdown based on selected Asset Number
                var locationDropdown = $('#tsg_edit_locationDropdown');
                locationDropdown.empty(); // Clear existing options
                locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });

                // Initialize Select2 or any other necessary actions
                areaDropdown.select2(), equipmentGroupDropdown.select2(), modelDropdown.select2(), assetNumberDropdown.select2(), locationDropdown.select2();
            });

            // Trigger change event to initially populate dependent dropdowns
            $('#tsg_edit_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateDropdownsForPartsPosition();
});
