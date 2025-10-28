$(document).ready(function () {
    // Function to toggle visibility of new Site Location fields (legacy approach)
    function toggleNewSiteLocationFields(selectedValue) {
        console.log(`Toggling Site Location Fields based on selected value: ${selectedValue}`);
        if (selectedValue === 'new') {
            $('#newSiteLocationFields').removeClass('d-none');
            $('#new_siteLocation_title').prop('required', true);
            $('#new_siteLocation_room_number').prop('required', true);
        } else {
            $('#newSiteLocationFields').addClass('d-none');
            $('#new_siteLocation_title').prop('required', false);
            $('#new_siteLocation_room_number').prop('required', false);
        }
    }

    // Utility Function to Reset Fields (excluding Site Location)
    function resetField(field, placeholder) {
        // Exclude Site Location Dropdown from being reset
        if (field.attr('id') === 'new_pst_siteLocationDropdown') {
            return; // Do not reset Site Location Dropdown
        }
        console.log(`Resetting field. Placeholder: ${placeholder}`);
        field.empty().append('<option value="">' + placeholder + '</option>');
        field.prop('disabled', true);
        field.val(null).trigger('change'); // Reset Select2 if applicable
    }

    // Function to display Bootstrap alerts
    function showAlert(message, category) {
        var alertHtml = `
            <div class="alert alert-${category} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        $('#alertContainer').html(alertHtml);

        // Auto-dismiss after 5 seconds
        setTimeout(function() {
            $('.alert').alert('close');
        }, 5000);
    }

    // Initialize modals for new entities
    initializeModals();

    // Fetch Site Locations on page load
    fetchSiteLocations();

    // Handle Site Location Dropdown Change
    $('#new_pst_siteLocationDropdown').on('change', function () {
        var selectedValue = $(this).val();
        console.log(`Site Location Dropdown changed to: ${selectedValue}`);

        if (selectedValue === 'new') {
            // Show modal for new site location
            showNewSiteLocationModal('new_pst_');
        } else {
            // Hide the legacy new site location fields if they exist
            $('#newSiteLocationFields').addClass('d-none');
            $('#new_siteLocation_title').prop('required', false);
            $('#new_siteLocation_room_number').prop('required', false);
        }
    });

    // Fetch Equipment Groups when Area is selected
    $('#new_pst_areaDropdown').on('change', function () {
        var areaId = $(this).val();
        console.log(`Area Dropdown changed to: ${areaId}`);
        if (areaId) {
            // Reset subsequent dropdowns
            resetField($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_equipment_groups',
                data: { area_id: areaId },
                success: function (response) {
                    console.log('Received Equipment Groups:', response);
                    var equipmentGroupDropdown = $('#new_pst_equipmentGroupDropdown');
                    equipmentGroupDropdown.empty();
                    equipmentGroupDropdown.append('<option value="">Select Equipment Group</option>');
                    $.each(response, function (index, equipmentGroup) {
                        equipmentGroupDropdown.append('<option value="' + equipmentGroup.id + '">' + equipmentGroup.name + '</option>');
                    });
                    equipmentGroupDropdown.append('<option value="new">New Equipment Group...</option>');
                    equipmentGroupDropdown.prop('disabled', false);
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Equipment Groups:', error);
                    showAlert('Error fetching Equipment Groups.', 'danger');
                }
            });
        } else {
            // Reset Equipment Group Dropdown and subsequent fields
            resetField($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
        }
    });

    // Handle Equipment Group change
    $('#new_pst_equipmentGroupDropdown').on('change', function () {
        var equipmentGroupId = $(this).val();
        console.log(`Equipment Group Dropdown changed to: ${equipmentGroupId}`);

        if (equipmentGroupId === 'new') {
            // Show the equipment group modal
            showNewEquipmentGroupModal('new_pst_');
        } else if (equipmentGroupId) {
            // Reset subsequent dropdowns
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_models',
                data: { equipment_group_id: equipmentGroupId },
                success: function (response) {
                    console.log('Received Models:', response);
                    var modelDropdown = $('#new_pst_modelDropdown');
                    modelDropdown.empty();
                    modelDropdown.append('<option value="">Select Model</option>');
                    $.each(response, function (index, model) {
                        modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                    });
                    modelDropdown.append('<option value="new">New Model...</option>');
                    modelDropdown.prop('disabled', false);
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Models:', error);
                    showAlert('Error fetching Models.', 'danger');
                }
            });
        } else {
            // Reset Model Dropdown and subsequent fields
            resetField($('#new_pst_modelDropdown'), 'Select Model');
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
        }
    });

    // Handle Model change
    $('#new_pst_modelDropdown').on('change', function () {
        var modelId = $(this).val();
        console.log(`Model Dropdown changed to: ${modelId}`);

        if (modelId === 'new') {
            // Show the model modal
            showNewModelModal('new_pst_');
        } else if (modelId) {
            // Reset Asset Number and Location Dropdowns
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here

            // Fetch Asset Numbers
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_asset_numbers',
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Asset Numbers:', response);
                    var assetNumberDropdown = $('#new_pst_assetNumberDropdown');
                    assetNumberDropdown.empty();
                    assetNumberDropdown.append('<option value="">Select Asset Number</option>');
                    $.each(response, function (index, assetNumber) {
                        assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                    });
                    assetNumberDropdown.append('<option value="new">New Asset Number...</option>');
                    assetNumberDropdown.prop('disabled', false);
                    assetNumberDropdown.val(null).trigger('change');
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Asset Numbers:', error);
                    showAlert('Error fetching Asset Numbers.', 'danger');
                }
            });

            // Fetch Locations
            $.ajax({
                type: 'GET',
                url: '/pst_troubleshoot_new_entry/get_locations',
                data: { model_id: modelId },
                success: function (response) {
                    console.log('Received Locations:', response);
                    var locationDropdown = $('#new_pst_locationDropdown');
                    locationDropdown.empty();
                    locationDropdown.append('<option value="">Select Location</option>');
                    $.each(response, function (index, location) {
                        locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                    });
                    // Add "New Location" option
                    locationDropdown.append('<option value="new">New Location...</option>');
                    locationDropdown.prop('disabled', false);
                    locationDropdown.val(null).trigger('change');
                },
                error: function (xhr, status, error) {
                    console.error('Error fetching Locations:', error);
                    showAlert('Error fetching Locations.', 'danger');
                }
            });
        } else {
            // Reset Asset Number and Location Dropdowns
            resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
            resetField($('#new_pst_locationDropdown'), 'Select Location');
            // Site Location is independent, no need to reset it here
        }
    });

    // Handle Asset Number change
    $('#new_pst_assetNumberDropdown').on('change', function() {
        var assetNumberValue = $(this).val();

        if (assetNumberValue === 'new') {
            // Show the asset number modal
            showNewAssetNumberModal('new_pst_');
        }
    });

    // Handle Location change
    $('#new_pst_locationDropdown').on('change', function() {
        var locationValue = $(this).val();

        if (locationValue === 'new') {
            // Show the location modal
            showNewLocationModal('new_pst_');
        }
    });

    function fetchSiteLocations() {
        console.log('Fetching all Site Locations on page load.');

        $.ajax({
            type: 'GET',
            url: '/pst_troubleshoot_new_entry/get_site_locations',
            success: function (response) {
                console.log('Received Site Locations:', response);
                var siteLocationDropdown = $('#new_pst_siteLocationDropdown');

                siteLocationDropdown.empty().append('<option value="">Select Site Location</option>');

                if (response.length === 0) {
                    siteLocationDropdown.append('<option value="">No Site Locations Available</option>');
                } else {
                    $.each(response, function (index, siteLocation) {
                        var optionText = siteLocation.title + ' - Room ' + siteLocation.room_number;
                        siteLocationDropdown.append('<option value="' + siteLocation.id + '">' + optionText + '</option>');
                    });
                }

                // Append "New Site Location" option
                siteLocationDropdown.append('<option value="new">New Site Location...</option>');
                siteLocationDropdown.prop('disabled', false);  // Enable dropdown if disabled

                // Initialize or refresh Select2
                siteLocationDropdown.select2({
                    placeholder: 'Select Site Location or type "New..."',
                    allowClear: true
                });

                console.log('Select2 initialized for Site Location Dropdown after options loaded.');
            },
            error: function (xhr, status, error) {
                console.error('Error fetching Site Locations:', error);
                showAlert('Error fetching Site Locations.', 'danger');
            }
        });
    }

    // Handle Form Submission via AJAX
    $('#newProblemForm').on('submit', function (e) {
        e.preventDefault(); // Prevent default form submission
        console.log('New Problem Form submitted.');

        // Validate required fields before submission
        var problemName = $('#problemName').val(); // Corrected ID
        var problemDescription = $('#problemDescription').val(); // Corrected ID
        var areaId = $('#new_pst_areaDropdown').val();
        var equipmentGroupId = $('#new_pst_equipmentGroupDropdown').val();
        var modelId = $('#new_pst_modelDropdown').val();

        if (!problemName || !problemDescription || !areaId || !equipmentGroupId || !modelId) {
            showAlert('Name, Description, Area, Equipment Group, and Model are required.', 'warning');
            return;
        }

        // Proceed with AJAX submission since all required fields are filled
        var formData = $(this).serialize(); // Serialize form data
        console.log('Form Data:', formData);

        $.ajax({
            type: 'POST',
            url: '/pst_troubleshoot_new_entry/create_problem', //
            data: formData,
            success: function (response) {
                console.log('Create Problem Response:', response);
                if (response.success) {
                    showAlert(response.message, 'success');
                    // Clear the form inputs after successful creation
                    $('#newProblemForm')[0].reset(); // Correct form ID
                    // Reset dynamically populated dropdowns
                    resetField($('#new_pst_equipmentGroupDropdown'), 'Select Equipment Group');
                    resetField($('#new_pst_modelDropdown'), 'Select Model');
                    resetField($('#new_pst_assetNumberDropdown'), 'Select Asset Number');
                    resetField($('#new_pst_locationDropdown'), 'Select Location');
                } else {
                    showAlert(response.message, 'warning');
                }
            },
            error: function (xhr, status, error) {
                console.error('Error creating Problem:', error);
                var errorMessage = 'An error occurred while creating the problem.';
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessage = xhr.responseJSON.message;
                }
                showAlert('Error: ' + errorMessage, 'danger');
            }
        });
    });

    // Initialize modals for new entities
    function initializeModals() {
        // Equipment Group Modal
        $('body').append(`
            <div class="modal fade" id="newEquipmentGroupModal" tabindex="-1" aria-labelledby="newEquipmentGroupModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="newEquipmentGroupModalLabel">Create New Equipment Group</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="newEquipmentGroupForm">
                                <input type="hidden" id="formPrefix" name="formPrefix" value="">
                                <input type="hidden" id="newEquipmentGroupAreaId" name="area_id" value="">
                                
                                <div class="mb-3">
                                    <label for="newEquipmentGroupName" class="form-label">Name:</label>
                                    <input type="text" class="form-control" id="newEquipmentGroupName" name="name" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newEquipmentGroupDescription" class="form-label">Description:</label>
                                    <textarea class="form-control" id="newEquipmentGroupDescription" name="description" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveNewEquipmentGroupBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `);

        // Model Modal
        $('body').append(`
            <div class="modal fade" id="newModelModal" tabindex="-1" aria-labelledby="newModelModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="newModelModalLabel">Create New Model</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="newModelForm">
                                <input type="hidden" id="modelFormPrefix" name="formPrefix" value="">
                                <input type="hidden" id="newModelEquipmentGroupId" name="equipment_group_id" value="">
                                
                                <div class="mb-3">
                                    <label for="newModelName" class="form-label">Name:</label>
                                    <input type="text" class="form-control" id="newModelName" name="name" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newModelDescription" class="form-label">Description:</label>
                                    <textarea class="form-control" id="newModelDescription" name="description" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveNewModelBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `);

        // Asset Number Modal
        $('body').append(`
            <div class="modal fade" id="newAssetNumberModal" tabindex="-1" aria-labelledby="newAssetNumberModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="newAssetNumberModalLabel">Create New Asset Number</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="newAssetNumberForm">
                                <input type="hidden" id="assetNumberFormPrefix" name="formPrefix" value="">
                                <input type="hidden" id="newAssetNumberModelId" name="model_id" value="">
                                
                                <div class="mb-3">
                                    <label for="newAssetNumberNumber" class="form-label">Number:</label>
                                    <input type="text" class="form-control" id="newAssetNumberNumber" name="number" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newAssetNumberDescription" class="form-label">Description:</label>
                                    <textarea class="form-control" id="newAssetNumberDescription" name="description" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveNewAssetNumberBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `);

        // Location Modal
        $('body').append(`
            <div class="modal fade" id="newLocationModal" tabindex="-1" aria-labelledby="newLocationModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="newLocationModalLabel">Create New Location</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="newLocationForm">
                                <input type="hidden" id="locationFormPrefix" name="formPrefix" value="">
                                <input type="hidden" id="newLocationModelId" name="model_id" value="">
                                
                                <div class="mb-3">
                                    <label for="newLocationName" class="form-label">Name:</label>
                                    <input type="text" class="form-control" id="newLocationName" name="name" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newLocationDescription" class="form-label">Description:</label>
                                    <textarea class="form-control" id="newLocationDescription" name="description" rows="3"></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveNewLocationBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `);

        // Site Location Modal
        $('body').append(`
            <div class="modal fade" id="newSiteLocationModal" tabindex="-1" aria-labelledby="newSiteLocationModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="newSiteLocationModalLabel">Create New Site Location</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="newSiteLocationForm">
                                <div class="mb-3">
                                    <label for="newSiteLocationTitle" class="form-label">Title:</label>
                                    <input type="text" class="form-control" id="newSiteLocationTitle" name="title" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newSiteLocationRoomNumber" class="form-label">Room Number:</label>
                                    <input type="text" class="form-control" id="newSiteLocationRoomNumber" name="room_number" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="newSiteLocationSiteArea" class="form-label">Site Area:</label>
                                    <input type="text" class="form-control" id="newSiteLocationSiteArea" name="site_area" value="Default Area">
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveNewSiteLocationBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `);
    }

    // Function to show the new equipment group modal
    function showNewEquipmentGroupModal(formPrefix) {
        const areaDropdownId = `#${formPrefix}areaDropdown`;
        const areaId = $(areaDropdownId).val();

        $('#formPrefix').val(formPrefix);
        $('#newEquipmentGroupAreaId').val(areaId);

        const newEquipmentGroupModal = new bootstrap.Modal(document.getElementById('newEquipmentGroupModal'));
        newEquipmentGroupModal.show();
    }

    // Function to show the new model modal
    function showNewModelModal(formPrefix) {
        const equipmentGroupDropdownId = `#${formPrefix}equipmentGroupDropdown`;
        const equipmentGroupId = $(equipmentGroupDropdownId).val();

        $('#modelFormPrefix').val(formPrefix);
        $('#newModelEquipmentGroupId').val(equipmentGroupId);

        const newModelModal = new bootstrap.Modal(document.getElementById('newModelModal'));
        newModelModal.show();
    }

    // Function to show the new asset number modal
    function showNewAssetNumberModal(formPrefix) {
        const modelDropdownId = `#${formPrefix}modelDropdown`;
        const modelId = $(modelDropdownId).val();

        $('#assetNumberFormPrefix').val(formPrefix);
        $('#newAssetNumberModelId').val(modelId);

        const newAssetNumberModal = new bootstrap.Modal(document.getElementById('newAssetNumberModal'));
        newAssetNumberModal.show();
    }

    // Function to show the new location modal
    function showNewLocationModal(formPrefix) {
        const modelDropdownId = `#${formPrefix}modelDropdown`;
        const modelId = $(modelDropdownId).val();

        $('#locationFormPrefix').val(formPrefix);
        $('#newLocationModelId').val(modelId);

        const newLocationModal = new bootstrap.Modal(document.getElementById('newLocationModal'));
        newLocationModal.show();
    }

    // Function to show the new site location modal
    function showNewSiteLocationModal(formPrefix) {
        // For site location, we don't need to track form prefix or parent entity
        const newSiteLocationModal = new bootstrap.Modal(document.getElementById('newSiteLocationModal'));
        newSiteLocationModal.show();
    }

    // Handle save button clicks for all modals
    $('#saveNewEquipmentGroupBtn').on('click', function() {
        console.log('Saving new Equipment Group...');
        const formData = $('#newEquipmentGroupForm').serialize();
        const formPrefix = $('#formPrefix').val();

        $.ajax({
            url: '/pst_troubleshoot_new_entry/create_equipment_group',
            type: 'POST',
            data: formData,
            success: function(response) {
                console.log('Equipment Group creation response:', response);
                if (response.success) {
                    // Close the modal
                    $('#newEquipmentGroupModal').modal('hide');

                    // Get the equipment group dropdown
                    const equipmentDropdownId = `#${formPrefix}equipmentGroupDropdown`;
                    const $equipmentDropdown = $(equipmentDropdownId);

                    // Add the new equipment group to the dropdown and select it
                    const newOption = new Option(response.equipment_group.name, response.equipment_group.id, true, true);
                    $equipmentDropdown.append(newOption).trigger('change');

                    // Show success message
                    showAlert(response.message, 'success');
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                console.error('Error creating Equipment Group:', error);
                const errorMessage = xhr.responseJSON?.message || 'An error occurred while creating the equipment group.';
                showAlert(errorMessage, 'danger');
            }
        });
    });

    $('#saveNewModelBtn').on('click', function() {
        console.log('Saving new Model...');
        const formData = $('#newModelForm').serialize();
        const formPrefix = $('#modelFormPrefix').val();

        $.ajax({
            url: '/pst_troubleshoot_new_entry/create_model',
            type: 'POST',
            data: formData,
            success: function(response) {
                console.log('Model creation response:', response);
                if (response.success) {
                    // Close the modal
                    $('#newModelModal').modal('hide');

                    // Get the model dropdown
                    const modelDropdownId = `#${formPrefix}modelDropdown`;
                    const $modelDropdown = $(modelDropdownId);

                    // Add the new model to the dropdown and select it
                    const newOption = new Option(response.model.name, response.model.id, true, true);
                    $modelDropdown.append(newOption).trigger('change');

                    // Show success message
                    showAlert(response.message, 'success');
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                console.error('Error creating Model:', error);
                const errorMessage = xhr.responseJSON?.message || 'An error occurred while creating the model.';
                showAlert(errorMessage, 'danger');
            }
        });
    });

    $('#saveNewAssetNumberBtn').on('click', function() {
        console.log('Saving new Asset Number...');
        const formData = $('#newAssetNumberForm').serialize();
        const formPrefix = $('#assetNumberFormPrefix').val();

        $.ajax({
            url: '/pst_troubleshoot_new_entry/create_asset_number',
            type: 'POST',
            data: formData,
            success: function(response) {
                console.log('Asset Number creation response:', response);
                if (response.success) {
                    // Close the modal
                    $('#newAssetNumberModal').modal('hide');

                    // Get the asset number dropdown
                    const assetNumberDropdownId = `#${formPrefix}assetNumberDropdown`;
                    const $assetNumberDropdown = $(assetNumberDropdownId);

                    // Add the new asset number to the dropdown and select it
                    const newOption = new Option(response.asset_number.number, response.asset_number.id, true, true);
                    $assetNumberDropdown.append(newOption).trigger('change');

                    // Show success message
                    showAlert(response.message, 'success');
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                console.error('Error creating Asset Number:', error);
                const errorMessage = xhr.responseJSON?.message || 'An error occurred while creating the asset number.';
                showAlert(errorMessage, 'danger');
            }
        });
    });

    // Handler for the save button in the Location modal
    $('#saveNewLocationBtn').on('click', function() {
        console.log('Saving new Location...');
        const formData = $('#newLocationForm').serialize();
        const formPrefix = $('#locationFormPrefix').val();

        $.ajax({
            url: '/pst_troubleshoot_new_entry/create_location',
            type: 'POST',
            data: formData,
            success: function(response) {
                console.log('Location creation response:', response);
                if (response.success) {
                    // Close the modal
                    $('#newLocationModal').modal('hide');

                    // Get the location dropdown
                    const locationDropdownId = `#${formPrefix}locationDropdown`;
                    const $locationDropdown = $(locationDropdownId);

                    // Add the new location to the dropdown and select it
                    const newOption = new Option(response.location.name, response.location.id, true, true);
                    $locationDropdown.append(newOption).trigger('change');

                    // Show success message
                    showAlert(response.message, 'success');
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                console.error('Error creating Location:', error);
                const errorMessage = xhr.responseJSON?.message || 'An error occurred while creating the location.';
                showAlert(errorMessage, 'danger');
            }
        });
    });

    // Handler for the save button in the Site Location modal
    $('#saveNewSiteLocationBtn').on('click', function() {
        console.log('Saving new Site Location...');
        const formData = $('#newSiteLocationForm').serialize();

        $.ajax({
            url: '/pst_troubleshoot_new_entry/create_site_location',
            type: 'POST',
            data: formData,
            success: function(response) {
                console.log('Site Location creation response:', response);
                if (response.success) {
                    // Close the modal
                    $('#newSiteLocationModal').modal('hide');

                    // Get the site location dropdown
                    const $siteLocationDropdown = $('#new_pst_siteLocationDropdown');

                    // Create the formatted option text
                    const optionText = response.site_location.title + ' - Room ' + response.site_location.room_number;

                    // Add the new site location to the dropdown and select it
                    const newOption = new Option(optionText, response.site_location.id, true, true);
                    $siteLocationDropdown.append(newOption).trigger('change');

                    // Show success message
                    showAlert(response.message, 'success');
                } else {
                    showAlert(response.message, 'danger');
                }
            },
            error: function(xhr, status, error) {
                console.error('Error creating Site Location:', error);
                const errorMessage = xhr.responseJSON?.message || 'An error occurred while creating the site location.';
                showAlert(errorMessage, 'danger');
            }
        });
    });
});