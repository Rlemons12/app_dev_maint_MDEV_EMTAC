// Function to fetch data and populate dropdowns
function populateDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#areaDropdown'), dataKey: 'areas' },
        { element: $('#equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#modelDropdown'), dataKey: 'models' },
        { element: $('#assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#locationDropdown'), dataKey: 'locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_list_data', // URL to fetch data from (replace with your server-side route)
        type: 'GET',
        success: function(data) {
            // Populate areas dropdown
            var areaDropdown = $('#areaDropdown');
            areaDropdown.empty(); // Clear existing options
			// Add an empty option
			areaDropdown.append('<option value="">Select...</option>');

            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
			areaDropdown.change(function() {
				var selectedAreaId = $(this).val();
				var equipmentGroupDropdown = $('#equipmentGroupDropdown');
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
			$('#equipmentGroupDropdown').change(function() {
				var selectedGroupId = $(this).val();
				var modelDropdown = $('#modelDropdown');
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
            $('#modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#assetNumberDropdown');
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
                var locationDropdown = $('#locationDropdown');
                locationDropdown.empty(); // Clear existing options
				
				// Add a placeholder option
				locationDropdown.append('<option value="">Select...</option>');
				
                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
                // Initialize Select2 or any other necessary actions
                locationDropdown.select2(),areaDropdown.select2(); 
            });

            // Call change event to populate equipment group dropdown initially based on the default selected area
            areaDropdown.change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateDropdowns();
});
