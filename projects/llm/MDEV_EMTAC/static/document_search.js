function populateCompleteDocumentDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var documentDropdowns = [
        { element: $('#searchdocument_areaDropdown'), dataKey: 'searchdocument_area' },
        { element: $('#searchdocument_equipmentGroupDropdown'), dataKey: 'searchdocument_equipmentgroup' },
        { element: $('#searchdocument_modelDropdown'), dataKey: 'searchdocument_model' },
        { element: $('#searchdocument_assetNumberDropdown'), dataKey: 'searchdocument_asset_number' },
        { element: $('#searchdocument_locationDropdown'), dataKey: 'searchdocument_location' }
    ];

    // AJAX request to fetch data for CompleteDocument form dropdowns
    $.ajax({
        url: '/get_completedocument_list_data_bp', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            // Populate areas dropdown
            var areaDropdown = $('#searchdocument_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#searchdocument_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options

                // Add a placeholder option
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
            $('#searchdocument_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#searchdocument_modelDropdown');
                modelDropdown.empty(); // Clear existing options

                // Add a placeholder option
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
            $('#searchdocument_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#searchdocument_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options

                // Add a placeholder option
                assetNumberDropdown.append('<option value="">Select...</option>');

                // Populate asset number dropdown with associated asset numbers based on selected model
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for asset number dropdown

                // Populate location dropdown with associated locations based on selected model
                var locationDropdown = $('#searchdocument_locationDropdown');
                locationDropdown.empty(); // Clear existing options

                // Add a placeholder option
                locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
                // Initialize Select2 or any other necessary actions

                locationDropdown.select2(), assetNumberDropdown.select2(), areaDropdown.select2();
            });

            // Call change event to populate equipment group dropdown initially based on the default selected area
            $('#searchdocument_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

$(document).ready(function() {
    console.log("Document ready. Calling populateCompleteDocumentDropdowns()");
    populateCompleteDocumentDropdowns(); // Corrected function name
});
