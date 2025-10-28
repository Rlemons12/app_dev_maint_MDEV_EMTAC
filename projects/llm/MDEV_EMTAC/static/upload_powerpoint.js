function populateUploadPowerPointDropdowns() {
    var powerPointDropdowns = [
        { element: $('#uploadpowerpoint_areaDropdown'), dataKey: 'areas' },
        { element: $('#uploadpowerpoint_equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#uploadpowerpoint_modelDropdown'), dataKey: 'models' },
        { element: $('#uploadpowerpoint_assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#uploadpowerpoint_locationDropdown'), dataKey: 'locations' }
    ];

    $.ajax({
        url: '/get_powerpoint_list_data_bp',
        type: 'GET',
        success: function(data) {
            populateDropdown($('#uploadpowerpoint_areaDropdown'), data['areas'], 'id', 'name');

            $('#uploadpowerpoint_areaDropdown').change(function() {
                var selectedAreaId = $(this).val();
                var filteredEquipmentGroups = data['equipment_groups'].filter(group => group.area_id == selectedAreaId);
                populateDropdown($('#uploadpowerpoint_equipmentGroupDropdown'), filteredEquipmentGroups, 'id', 'name');
                $('#uploadpowerpoint_equipmentGroupDropdown').change();
            });

            $('#uploadpowerpoint_equipmentGroupDropdown').change(function() {
                var selectedGroupId = $(this).val();
                var filteredModels = data['models'].filter(model => model.equipment_group_id == selectedGroupId);
                populateDropdown($('#uploadpowerpoint_modelDropdown'), filteredModels, 'id', 'name');
                $('#uploadpowerpoint_modelDropdown').change();
            });

            $('#uploadpowerpoint_modelDropdown').change(function() {
                var selectedModelId = $(this).val();
                var filteredAssetNumbers = data['asset_numbers'].filter(asset => asset.model_id == selectedModelId);
                populateDropdown($('#uploadpowerpoint_assetNumberDropdown'), filteredAssetNumbers, 'id', 'number');

                var filteredLocations = data['locations'].filter(location => location.model_id == selectedModelId);
                populateDropdown($('#uploadpowerpoint_locationDropdown'), filteredLocations, 'id', 'name');
            });

            $('#uploadpowerpoint_areaDropdown').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

function populateDropdown(dropdown, items, valueKey, textKey) {
    dropdown.empty();
    dropdown.append('<option value="">Select...</option>');
    $.each(items, function(index, item) {
        dropdown.append('<option value="' + item[valueKey] + '">' + item[textKey] + '</option>');
    });
}

$(document).ready(function() {
    populateUploadPowerPointDropdowns();
});
