function populateTroubleshootingGuideDropdowns() {
    var documentDropdowns = [
        { element: $('#tsg_areaDropdown'), dataKey: 'areas' },
        { element: $('#tsg_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#tsg_modelDropdown'), dataKey: 'models' },
        { element: $('#tsg_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#tsg_locationDropdown'), dataKey: 'locations' },
        { element: $('#tsg_documentSearchDropdown'), dataKey: 'documents' },
        { element: $('#tsg_problemImageSearchDropdown'), dataKey: 'images' },
        { element: $('#tsg_solutionImageSearchDropdown'), dataKey: 'images' }
    ];

    $.ajax({
        url: '/get_troubleshooting_guide_data_bp',
        type: 'GET',
        success: function(data) {
            documentDropdowns.forEach(function(dropdown) {
                var dropdownElement = dropdown.element;
                var dataKey = dropdown.dataKey;
                dropdownElement.empty();
                dropdownElement.append('<option value="">Select...</option>');
                $.each(data[dataKey], function(index, item) {
                    dropdownElement.append('<option value="' + item.id + '">' + (item.name || item.title || item.number || item.part_number) + '</option>');
                });
            });

            $('#tsg_problemImageSearchDropdown').select2();
            $('#tsg_solutionImageSearchDropdown').select2();

            $('#tsg_areaDropdown').change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#tsg_equipmentGroupDropdown');
                equipmentGroupDropdown.empty();
                equipmentGroupDropdown.append('<option value="">Select...</option>');
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change();
            });

            $('#tsg_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#tsg_modelDropdown');
                modelDropdown.empty();
                modelDropdown.append('<option value="">Select...</option>');
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change();
            });

            $('#tsg_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#tsg_assetNumberDropdown');
                assetNumberDropdown.empty();
                assetNumberDropdown.append('<option value="">Select...</option>');
                $.each(data['asset_numbers'], function(index, assetNumber) {
                    if (assetNumber.model_id == selectedModelId) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    }
                });
                assetNumberDropdown.change();

                var locationDropdown = $('#tsg_locationDropdown');
                locationDropdown.empty();
                locationDropdown.append('<option value="">Select...</option>');
                $.each(data['locations'], function(index, location) {
                    if (location.model_id == selectedModelId) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    }
                });
            });

            $('#tsg_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

$(document).ready(function() {
    populateTroubleshootingGuideDropdowns();
});
