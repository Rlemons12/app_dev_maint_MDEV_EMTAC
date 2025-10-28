function populateTroubleshootingGuideDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var documentDropdowns = [
        { element: $('#tsg_areaDropdown'), dataKey: 'tsg_areas' },
        { element: $('#tsg_equipmentGroupDropdown'), dataKey: 'tsg_equipment_groups' },
        { element: $('#tsg_modelDropdown'), dataKey: 'tsg_models' },
        { element: $('#tsg_assetNumberDropdown'), dataKey: 'tsg_asset_numbers' },
        { element: $('#tsg_locationDropdown'), dataKey: 'tsg_locations' },
        { element: $('#tsg_documentSearchDropdown'), dataKey: 'tsg_document_search' },
        { element: $('#tsg_imageSearchDropdown'), dataKey: 'tsg_image_search' }, // Add image search dropdown
        { element: $('#tsg_problemImageSearchDropdown'), dataKey: 'tsg_problem_images' }, // Add problem image search dropdown
        { element: $('#tsg_solutionImageSearchDropdown'), dataKey: 'tsg_solution_images' }, // Add solution image search dropdown
        
    ];
    
    // AJAX request to fetch data for Troubleshooting Guide form dropdowns
    $.ajax({
        url: '/get_troubleshooting_guide_data_bp', // URL to fetch data from Flask route
        type: 'GET',
        success: function(data) {
            // Populate Area dropdown
            var areaDropdown = $('#tsg_areaDropdown');
            areaDropdown.empty(); // Clear existing options
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for Area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#tsg_equipmentGroupDropdown');
                equipmentGroupDropdown.empty(); // Clear existing options
                // Add a placeholder option
                equipmentGroupDropdown.append('<option value="">Select...</option>');
                // Populate Equipment Group dropdown with associated groups based on selected Area
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change(); // Trigger change event for Equipment Group dropdown
            });

            // Event listener for Equipment Group dropdown change
            $('#tsg_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#tsg_modelDropdown');
                modelDropdown.empty(); // Clear existing options
                // Add a placeholder option
                modelDropdown.append('<option value="">Select...</option>');
                // Populate Model dropdown with associated models based on selected Equipment Group
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change(); // Trigger change event for Model dropdown
            });

            // Event listener for Model dropdown change
            $('#tsg_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#tsg_assetNumberDropdown');
                assetNumberDropdown.empty(); // Clear existing options
                // Add a placeholder option
                assetNumberDropdown.append('<option value="">Select...</option>');
                // Populate Asset Number dropdown with associated asset numbers based on selected Model
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change(); // Trigger change event for Asset Number dropdown

                // Populate Location dropdown with associated locations based on selected Model
                var locationDropdown = $('#tsg_locationDropdown');
                locationDropdown.empty(); // Clear existing options
                // Add a placeholder option
                locationDropdown.append('<option value="">Select...</option>');
                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
                
               // Populate Document Search dropdown
				var documentSearchDropdown = $('#tsg_documentSearchDropdown');
				documentSearchDropdown.empty(); // Clear existing options
				// Add a placeholder option
				documentSearchDropdown.append('<option value="">None</option>');
				// Add options for each document
				$.each(data['documents'], function(index, document) {
					documentSearchDropdown.append('<option value="' + document.id + '">' + document.title + '</option>');
				});

				// Initialize Select2 for document search dropdown
				documentSearchDropdown.select2({
					placeholder: 'Search for documents...',
					allowClear: true, // Allow clearing the selection
					multiple: true, // Enable multiple selection
					width: '100%' // Set the width of the dropdown
				});

				
				
				// Populate Image Search dropdown
				var imageSearchDropdown = $('#tsg_imageSearchDropdown');
				imageSearchDropdown.empty(); // Clear existing options
				// Add a placeholder option
				imageSearchDropdown.append('<option value="">None</option>');
				// Add options for each image
				$.each(data['images'], function(index, image) {
					imageSearchDropdown.append('<option value="' + image.id + '">' + image.title + '</option>');
				});
				
				// Populate Problem Image Search dropdown
				var problemImageSearchDropdown = $('#tsg_problemImageSearchDropdown');
				problemImageSearchDropdown.empty(); // Clear existing options
				// Add a placeholder option
				problemImageSearchDropdown.append('<option value="">None</option>');
				// Add options for each image
				$.each(data['images'], function(index, image) {
					problemImageSearchDropdown.append('<option value="' + image.id + '">' + image.title + '</option>');
				});

				// Initialize Select2 for problem image search dropdown with a search bar
				problemImageSearchDropdown.select2({
					placeholder: 'Search for problem images...',
					allowClear: true, // Allow clearing the selection
					width: '100%' // Set the width of the dropdown
				});

				// Populate Solution Image Search dropdown
				var solutionImageSearchDropdown = $('#tsg_solutionImageSearchDropdown');
				solutionImageSearchDropdown.empty(); // Clear existing options
				// Add a placeholder option
				solutionImageSearchDropdown.append('<option value="">None</option>');
				// Add options for each image
				$.each(data['images'], function(index, image) {
					solutionImageSearchDropdown.append('<option value="' + image.id + '">' + image.title + '</option>');
				});

				// Initialize Select2 for solution image search dropdown with a search bar
				solutionImageSearchDropdown.select2({
					placeholder: 'Search for solution images...',
					allowClear: true, // Allow clearing the selection
					width: '100%' // Set the width of the dropdown
				});


				// Initialize Select2 for image search dropdown
				imageSearchDropdown.select2({
					placeholder: 'Search for images...',
					allowClear: true, // Allow clearing the selection
					multiple: true, // Enable multiple selection
					width: '100%' // Set the width of the dropdown
				});

				// Call the function to populate parts dropdown
				populatePartsDropdown(data['parts']);

				// Call the function to populate drawings dropdown
				populateDrawingsDropdown(data['drawing']);

                // Initialize Select2 or any other necessary actions
                imageSearchDropdown.select2(); // Assuming you're using Select2 for dropdown styling
            });

            // Call change event to populate Equipment Group dropdown initially based on the default selected Area
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
