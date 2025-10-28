// upload_image.js - Handles image upload form dropdowns
function populateImageUploadDropdowns() {
    console.log('Populating image upload dropdowns...');

    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#image_areaDropdown'), dataKey: 'areas' },
        { element: $('#image_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#image_modelDropdown'), dataKey: 'models' },
        { element: $('#image_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#image_locationDropdown'), dataKey: 'locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_image_list_data', // URL to fetch data from
        type: 'GET',
        success: function(data) {
            console.log('Image upload data received:', data);

            // Populate areas dropdown
            var areaDropdown = $('#image_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            // Add an empty option
            areaDropdown.append('<option value="">Select Area...</option>');

            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#image_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options

                // Add a placeholder option
                equipmentGroupDropdown.append('<option value="">Select Equipment Group...</option>');

                if (selectedAreaId) {
                    // Populate equipment group dropdown with associated groups based on selected area
                    $.each(data['equipment_groups'], function(index, group) {
                        if (group.area_id == selectedAreaId) {
                            equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                        }
                    });
                }
                equipmentGroupDropdown.change(); // Trigger change event for equipment group dropdown
            });

            // Event listener for equipment group dropdown change
            $('#image_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#image_modelDropdown');
                modelDropdown.empty(); // Clear existing options

                // Add a placeholder option
                modelDropdown.append('<option value="">Select Model...</option>');

                if (selectedGroupId) {
                    // Populate model dropdown with associated models based on selected equipment group
                    $.each(data['models'], function(index, model) {
                        if (model.equipment_group_id == selectedGroupId) {
                            modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                        }
                    });
                }
                modelDropdown.change(); // Trigger change event for model dropdown
            });

            // Event listener for model dropdown change
            $('#image_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#image_assetNumberDropdown');
                var locationDropdown = $('#image_locationDropdown');

                assetNumberDropdown.empty(); // Clear existing options
                locationDropdown.empty(); // Clear existing options

                // Add placeholder options
                assetNumberDropdown.append('<option value="">Select Asset Number...</option>');
                locationDropdown.append('<option value="">Select Location...</option>');

                if (selectedModelId) {
                    // Populate asset number dropdown with associated asset numbers based on selected model
                    $.each(data['asset_numbers'], function(index, assetNumber) {
                        if (assetNumber.model_id == selectedModelId) {
                            assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                        }
                    });

                    // Populate location dropdown with associated locations based on selected model
                    $.each(data['locations'], function(index, location) {
                        if (location.model_id == selectedModelId) {
                            locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                        }
                    });
                }

                // Initialize Select2 for better UX
                areaDropdown.select2({placeholder: "Select Area"});
                equipmentGroupDropdown.select2({placeholder: "Select Equipment Group"});
                modelDropdown.select2({placeholder: "Select Model"});
                assetNumberDropdown.select2({placeholder: "Select Asset Number"});
                locationDropdown.select2({placeholder: "Select Location"});
            });

            // Initialize Select2 for area dropdown
            areaDropdown.select2({placeholder: "Select Area"});

            console.log('Image upload dropdowns populated successfully');
        },
        error: function(xhr, status, error) {
            console.error('Error fetching image upload data:', error);
            console.error('Status:', status);
            console.error('Response:', xhr.responseText);
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    console.log('Image upload JS ready');
    populateImageUploadDropdowns();
});