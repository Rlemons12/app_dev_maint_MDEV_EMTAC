// Function to fetch data and populate dropdowns for the image form
function populateDropdownsimage() {
    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#image_areaDropdown'), dataKey: 'areas' },
        { element: $('#image_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#image_modelDropdown'), dataKey: 'models' },
        { element: $('#image_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#image_locationDropdown'), dataKey: 'locations' },
        // Added new dropdown elements
        { element: $('#image_subassemblyDropdown'), dataKey: 'subassemblies' },
        { element: $('#image_componentAssemblyDropdown'), dataKey: 'component_assemblies' },
        { element: $('#image_assemblyViewDropdown'), dataKey: 'assembly_views' },
        { element: $('#image_siteLocationDropdown'), dataKey: 'site_locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_image_list_data',
        type: 'GET',
        success: function(data) {
            // Populate site locations dropdown (independent of hierarchy)
            var siteLocationDropdown = $('#image_siteLocationDropdown');
            siteLocationDropdown.empty();
            siteLocationDropdown.append('<option value="">Select...</option>');
            $.each(data['site_locations'], function(index, site) {
                siteLocationDropdown.append('<option value="' + site.id + '">' + site.name + '</option>');
            });
            if (typeof siteLocationDropdown.select2 === 'function') {
                siteLocationDropdown.select2();
            }

            // Populate areas dropdown
            var areaDropdown = $('#image_areaDropdown');
            areaDropdown.empty(); // Clear existing options

            // Add a placeholder option
            areaDropdown.append('<option value="">Select...</option>');

            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            // Event listener for area dropdown change
            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#image_equipmentGroupDropdown');
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
            $('#image_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#image_modelDropdown');
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
            $('#image_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#image_assetNumberDropdown');
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
                var locationDropdown = $('#image_locationDropdown');
                locationDropdown.empty(); // Clear existing options

				// Add a placeholder option
				locationDropdown.append('<option value="">Select...</option>');

                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });

                locationDropdown.change(); // Trigger change event for location dropdown
            });

            // NEW: Event listener for location dropdown change
            $('#image_locationDropdown').change(function() {
                var selectedLocationId = $(this).val();
                var subassemblyDropdown = $('#image_subassemblyDropdown');
                subassemblyDropdown.empty(); // Clear existing options

                // Add a placeholder option
                subassemblyDropdown.append('<option value="">Select...</option>');

                // Populate subassembly dropdown with associated subassemblies based on selected location
                $.each(data['subassemblies'], function(index, subassembly) {
                    if (subassembly.location_id == selectedLocationId) {
                        subassemblyDropdown.append('<option value="' + subassembly.id + '">' + subassembly.name + '</option>');
                    }
                });
                subassemblyDropdown.change(); // Trigger change event for subassembly dropdown
            });

            // NEW: Event listener for subassembly dropdown change
            $('#image_subassemblyDropdown').change(function() {
                var selectedSubassemblyId = $(this).val();
                var componentAssemblyDropdown = $('#image_componentAssemblyDropdown');
                componentAssemblyDropdown.empty(); // Clear existing options

                // Add a placeholder option
                componentAssemblyDropdown.append('<option value="">Select...</option>');

                // Populate component assembly dropdown with associated component assemblies based on selected subassembly
                $.each(data['component_assemblies'], function(index, compAssembly) {
                    if (compAssembly.subassembly_id == selectedSubassemblyId) {
                        componentAssemblyDropdown.append('<option value="' + compAssembly.id + '">' + compAssembly.name + '</option>');
                    }
                });
                componentAssemblyDropdown.change(); // Trigger change event for component assembly dropdown
            });

            // NEW: Event listener for component assembly dropdown change
            $('#image_componentAssemblyDropdown').change(function() {
                var selectedComponentAssemblyId = $(this).val();
                var assemblyViewDropdown = $('#image_assemblyViewDropdown');
                assemblyViewDropdown.empty(); // Clear existing options

                // Add a placeholder option
                assemblyViewDropdown.append('<option value="">Select...</option>');

                // Populate assembly view dropdown with associated assembly views based on selected component assembly
                $.each(data['assembly_views'], function(index, assView) {
                    if (assView.component_assembly_id == selectedComponentAssemblyId) {
                        assemblyViewDropdown.append('<option value="' + assView.id + '">' + assView.name + '</option>');
                    }
                });
            });

            // Initialize Select2 for all dropdowns
            if (typeof areaDropdown.select2 === 'function') {
                dropdowns.forEach(function(dropdown) {
                    dropdown.element.select2();
                });
            }

            // Call change event to populate equipment group dropdown initially based on the default selected area
            areaDropdown.change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
            alert('Failed to load equipment data. Please try refreshing the page.');
        }
    });
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateDropdownsimage();
});