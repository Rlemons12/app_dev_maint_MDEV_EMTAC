function populateSearchTroubleshootingGuideDropdowns() {
    var imageDropdowns = [
        { element: $('#problemArea'), dataKey: 'areas' },
        { element: $('#problemEquipmentGroup'), dataKey: 'equipment_groups' },
        { element: $('#problemModel'), dataKey: 'models' },
        { element: $('#problemAssetNumber'), dataKey: 'asset_numbers' },
        { element: $('#problemLocation'), dataKey: 'locations' },
        { element: $('#problemTitle'), dataKey: 'problems' },
        { element: $('#problemSiteLocation'), dataKey: 'site_locations' }
    ];

    $.ajax({
        url: '/get_search_troubleshooting_guide_data_bp',
        type: 'GET',
        success: function(data) {
            var areaDropdown = $('#problemArea');
            areaDropdown.empty();
            $.each(data['areas'], function(index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + '</option>');
            });

            areaDropdown.change(function() {
                var selectedAreaId = $(this).val();
                var equipmentGroupDropdown = $('#problemEquipmentGroup');
                equipmentGroupDropdown.empty();
                equipmentGroupDropdown.append('<option value="">Select...</option>');
                $.each(data['equipment_groups'], function(index, group) {
                    if (group.area_id == selectedAreaId) {
                        equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                    }
                });
                equipmentGroupDropdown.change();
            });

            $('#problemEquipmentGroup').change(function() {
                var selectedGroupId = $(this).val();
                var modelDropdown = $('#problemModel');
                modelDropdown.empty();
                modelDropdown.append('<option value="">Select...</option>');
                $.each(data['models'], function(index, model) {
                    if (model.equipment_group_id == selectedGroupId) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    }
                });
                modelDropdown.change();
            });

            $('#problemModel').change(function() {
                var selectedModelId = $(this).val();
                var assetNumberDropdown = $('#problemAssetNumber');
                var locationDropdown = $('#problemLocation');
                var titleDropdown = $('#problemTitle');
                var siteLocationDropdown = $('#problemSiteLocation');

                assetNumberDropdown.empty();
                locationDropdown.empty();
                titleDropdown.empty();
                siteLocationDropdown.empty();

                assetNumberDropdown.append('<option value="">Select...</option>');
                locationDropdown.append('<option value="">Select...</option>');
                titleDropdown.append('<option value="">Select...</option>');
                siteLocationDropdown.append('<option value="">Select...</option>');

                var filteredAssetNumbers = data['asset_numbers'].filter(function(assetNumber) {
                    return assetNumber.model_id == selectedModelId;
                });

                $.each(filteredAssetNumbers, function(index, assetNumber) {
                    assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                });

                var filteredLocations = data['locations'].filter(function(location) {
                    return location.model_id == selectedModelId;
                });

                $.each(filteredLocations, function(index, location) {
                    locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                });

                var filteredProblems = data['problems'].filter(function(problem) {
                    return problem.model_id == selectedModelId;
                });

                $.each(filteredProblems, function(index, problem) {
                    titleDropdown.append('<option value="' + problem.id + '">' + problem.name + '</option>');
                });

                $.each(data['site_locations'], function(index, siteLocation) {
                    siteLocationDropdown.append('<option value="' + siteLocation.id + '">' + siteLocation.title + '</option>');
                });
            });

            $('#problemArea').change();
        },
        error: function(xhr, status, error) {
            console.error('Error fetching data:', error);
        }
    });
}

$(document).ready(function() {
    populateSearchTroubleshootingGuideDropdowns();
});
