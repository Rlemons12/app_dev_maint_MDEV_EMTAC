function populateSearchImageDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var imageDropdowns = [
        { element: $('#tsg_searchimage_areaDropdown'), dataKey: 'tsg_searchimage_area' },
        { element: $('#tsg_searchimage_equipmentGroupDropdown'), dataKey: 'tsg_searchimage_equipment_group' },
        { element: $('#tsg_searchimage_modelDropdown'), dataKey: 'tsg_searchimage_model' },
        { element: $('#tsg_searchimage_assetNumberDropdown'), dataKey: 'tsg_searchimage_asset_number' },
        { element: $('#tsg_searchimage_locationDropdown'), dataKey: 'tsg_searchimage_location' }
    ];

    // AJAX request to fetch data for Image form dropdowns
    $.ajax({
        url: '/get_tsg_search_image_list_data_bp', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            // Populate areas dropdown
            var areaDropdown = $('#tsg_searchimage_areaDropdown');
            areaDropdown.empty(); // Clear existing options
			
			// Add an empty option
			areaDropdown.append('<option value="">Select...</option>');

			
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#tsg_searchimage_equipmentGroupDropdown');
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
            $('#tsg_searchimage_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#tsg_searchimage_modelDropdown');
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
            $('#tsg_searchimage_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#tsg_searchimage_assetNumberDropdown');
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
                var locationDropdown = $('#tsg_searchimage_locationDropdown');
                locationDropdown.empty(); // Clear existing options

                // Add a placeholder option
                locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
                // Initialize Select2 or any other necessary actions

                // Initialize Select2 or any other necessary actions
				areaDropdown.select2();
				equipmentGroupDropdown.select2();
				modelDropdown.select2();
				assetNumberDropdown.select2();
				locationDropdown.select2();

            });

            // Call change event to populate equipment group dropdown initially based on the default selected area
            $('#tsg_searchimage_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateSearchImageDropdowns()
});
