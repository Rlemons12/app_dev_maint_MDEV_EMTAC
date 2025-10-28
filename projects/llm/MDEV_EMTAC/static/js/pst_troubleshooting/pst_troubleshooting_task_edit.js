// pst_troubleshooting_task_edit.js
document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // === 1. Define Backend Endpoint URLs ===
    const ENDPOINTS = {
    tasks: {
        details: '/pst_troubleshooting_guide_edit_update/task_details/',
        update: '/pst_troubleshooting_guide_edit_update/update_task',
        savePosition: '/pst_troubleshooting_guide_edit_update/save_position',
        searchDocuments: '/pst_troubleshooting_guide_edit_update/search_documents',
        saveDocuments: '/pst_troubleshooting_guide_edit_update/save_task_documents',
        searchDrawings: '/pst_troubleshooting_guide_edit_update/search_drawings',
        saveDrawings: '/pst_troubleshooting_guide_edit_update/save_task_drawings',
        searchParts: '/pst_troubleshooting_guide_edit_update/search_parts', // New endpoint for part search
        saveParts: '/pst_troubleshooting_guide_edit_update/save_task_parts', // Placeholder for saving parts
        searchImages: '/pst_troubleshooting_guide_edit_update/search_images', // Added line for image search
        saveImages: '/pst_troubleshooting_guide_edit_update/save_task_images',
        removePosition: '/pst_troubleshooting_guide_edit_update/remove_position', // Corrected line
        searchTool: '/pst_troubleshooting_guide_edit_update/search_tools', // searches for tools
        saveTool: '/pst_troubleshooting_guide_edit_update/save_task_tools', //saves tools
        removePart: '/pst_troubleshooting_guide_edit_update/remove_task_part', // New endpoint for removing a part
        removeDrawing: '/pst_troubleshooting_guide_edit_update/remove_task_drawing', // New endpoint for removing a drawing
        removeImage: '/pst_troubleshooting_guide_edit_update/remove_task_image', // New endpoint for removing an image
        removeDocument: '/pst_troubleshooting_guide_edit_update/remove_task_document', // New endpoint for removing a document
        removeTool: '/pst_troubleshooting_guide_edit_update/remove_task_tools' // remove Tool

    }
};

    // === 2. Initialize Global State Object ===
    window.AppState = window.AppState || {};
    window.AppState.currentTaskId = null;
    window.AppState.currentSolutionId = null;

    // === 3. Event Delegation for Save and Remove Position Buttons ===
    const positionsContainer = document.getElementById('pst_task_edit_positions_container');
    if (positionsContainer) {
        positionsContainer.addEventListener('click', (event) => {
            if (event.target && event.target.matches('.savePositionBtn')) {
                const positionSection = event.target.closest('.position-section');
                const index = Array.from(positionsContainer.children).indexOf(positionSection);
                console.log(`Delegated Save Position button clicked for index ${index}`);
                savePosition(positionSection, index);
            }

            if (event.target && event.target.matches('.removePositionBtn')) {
                const positionSection = event.target.closest('.position-section');
                const index = Array.from(positionsContainer.children).indexOf(positionSection);
                console.log("Delegated Remove Position button clicked for index", index);
                handleRemovePosition(positionSection, index);
            }
        });
        console.log("Attached delegated event listener to positions container.");
    } else {
        console.warn("Positions container with ID 'pst_task_edit_positions_container' not found.");
    }

    // === 3. Initialize Select2 for Document Search ===
    // Initialize Select2 for Document Search with empty placeholder for selected items
    $('#pst_task_edit_task_documents').select2({
        placeholder: 'Select or search for documents',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchDocuments, // Ensure this is defined in ENDPOINTS
            dataType: 'json',
            delay: 250,
            data: params => ({ query: params.term }),
            processResults: data => ({
                results: data.map(doc => ({
                    id: doc.id,
                    text: doc.text // Ensure backend provides `id` and `text` fields
                }))
            }),
            cache: true
        }
    });

    // Listen for change events on the select2 element
    $('#pst_task_edit_task_documents').on('change', updateSelectedDocumentsDisplay);

    // === 4. Save Selected Documents ===
    async function saveSelectedDocuments() {
        const selectedDocumentIds = $('#pst_task_edit_task_documents').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save documents.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            document_ids: selectedDocumentIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveDocuments, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Documents saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save documents.', 'danger');
            }
        } catch (error) {
            console.error('Error saving documents:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving documents.', 'danger');
        }
    }

    const saveDocumentsBtn = document.getElementById('saveDocumentsBtn');
    if (saveDocumentsBtn) {
        saveDocumentsBtn.addEventListener('click', saveSelectedDocuments);
    } else {
        console.warn("Save Documents button with ID 'saveDocumentsBtn' not found.");
    }

    async function savePosition(positionSection, index) {
    const areaDropdown = positionSection.querySelector('.areaDropdown');
    const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
    const modelDropdown = positionSection.querySelector('.modelDropdown');
    const assetNumberInput = positionSection.querySelector('.assetNumberInput');
    const locationInput = positionSection.querySelector('.locationInput');
    const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
    // Select new dropdown elements
    const assemblyDropdown = positionSection.querySelector('.assembliesDropdown');
    const subassemblyDropdown = positionSection.querySelector('.subassembliesDropdown');
    const assemblyViewDropdown = positionSection.querySelector('.assemblyViewsDropdown');

    if (!areaDropdown.value || !equipmentGroupDropdown.value || !modelDropdown.value) {
        SolutionTaskCommon.showAlert('Please fill in all required fields before saving.', 'warning');
        return;
    }

    if (!window.AppState.currentTaskId || !window.AppState.currentSolutionId) {
        SolutionTaskCommon.showAlert('Task and Solution must be selected before saving a position.', 'warning');
        return;
    }

    // Construct the position data object with existing fields
    const positionData = {
        area_id: parseInt(areaDropdown.value, 10) || null,
        equipment_group_id: parseInt(equipmentGroupDropdown.value, 10) || null,
        model_id: parseInt(modelDropdown.value, 10) || null,
        asset_number_id: parseInt(assetNumberInput.value.trim(), 10) || null, // Converted to integer
        location_id: parseInt(locationInput.value.trim(), 10) || null,       // Converted to integer
        site_location_id: parseInt(siteLocationDropdown.value, 10) || null,
        assembly_id: parseInt(assemblyDropdown.value, 10) || null,
        subassembly_id: parseInt(subassemblyDropdown.value, 10) || null,
        assembly_view_id: parseInt(assemblyViewDropdown.value, 10) || null
    };


    const saveBtn = positionSection.querySelector('.savePositionBtn');
    let originalBtnText = '';
    if (saveBtn) {
        saveBtn.disabled = true;
        originalBtnText = saveBtn.textContent;
        saveBtn.textContent = 'Saving...';
    }

    try {
        const response = await fetch(ENDPOINTS.tasks.savePosition, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: window.AppState.currentTaskId,
                solution_id: window.AppState.currentSolutionId,
                position_data: positionData
            })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Update the data-position-id attribute with the new position_id
            const newPositionId = data.position_id;
            positionSection.setAttribute('data-position-id', newPositionId);

            SolutionTaskCommon.showAlert('Position saved successfully.', 'success');
            console.log(`Position saved with ID: ${newPositionId}`);
        } else {
            SolutionTaskCommon.showAlert(data.error || 'Failed to save position.', 'danger');
            console.error('Failed to save position:', data.error || data.message);
        }
    } catch (error) {
        console.error('Error saving position:', error);
        SolutionTaskCommon.showAlert('An error occurred while saving the position.', 'danger');
    } finally {
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = originalBtnText;
        }
    }
}

    // === 6. Task Handling Functions ===
    async function saveTaskDetails() {
        const taskId = window.AppState.currentTaskId;
        const taskNameInput = document.getElementById('pst_task_edit_task_name');
        const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
        const positionsData = collectPositionsData();

        const updatedTaskData = {
            task_id: taskId,
            name: taskNameInput.value.trim(),
            description: taskDescriptionTextarea.value.trim(),
            positions: positionsData
        };

        const saveTaskBtn = document.getElementById('saveTaskBtn');
        if (saveTaskBtn) {
            saveTaskBtn.disabled = true;
            saveTaskBtn.textContent = 'Saving...';
        }

        const response = await fetchWithHandling(ENDPOINTS.tasks.update, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedTaskData)
        });

        if (response.status === 'success') {
            SolutionTaskCommon.showAlert('Task updated successfully.', 'success');
        } else {
            SolutionTaskCommon.showAlert('Failed to update task.', 'danger');
        }

        if (saveTaskBtn) {
            saveTaskBtn.disabled = false;
            saveTaskBtn.textContent = 'Save Task';
        }
    }

    function collectPositionsData() {
        const positionsContainer = document.getElementById('pst_task_edit_positions_container');
        const positionsData = [];

        // Loop over all position sections (each representing one position entry)
        positionsContainer.querySelectorAll('.position-section').forEach(section => {
        // Retrieve base dropdown and input elements
        const areaDropdown = section.querySelector('.areaDropdown');
        const equipmentGroupDropdown = section.querySelector('.equipmentGroupDropdown');
        const modelDropdown = section.querySelector('.modelDropdown');
        const assetNumberInput = section.querySelector('.assetNumberInput');
        const locationInput = section.querySelector('.locationInput');
        const siteLocationDropdown = section.querySelector('.siteLocationDropdown');

        // Retrieve additional dropdowns for assemblies and views
        const assemblyDropdown = section.querySelector('.assemblyDropdown');
        const subassemblyDropdown = section.querySelector('.subassemblyDropdown');
        const assemblyViewDropdown = section.querySelector('.assemblyViewDropdown');

        // Parse integer values for base identifiers
        const areaId = parseInt(areaDropdown.value, 10) || null;
        const equipmentGroupId = parseInt(equipmentGroupDropdown.value, 10) || null;
        const modelId = parseInt(modelDropdown.value, 10) || null;

        // Adjust these based on backend expectations
        const assetNumberId = assetNumberInput.value.trim() ? parseInt(assetNumberInput.value.trim(), 10) : null;
        const locationId = locationInput.value.trim() ? parseInt(locationInput.value.trim(), 10) : null;
        const siteLocationId = parseInt(siteLocationDropdown.value, 10) || null;

        // Parse new dropdowns for assemblies
        const assemblyId = assemblyDropdown.value.trim() ? parseInt(assemblyDropdown.value.trim(), 10) : null;
        const subassemblyId = subassemblyDropdown.value.trim() ? parseInt(subassemblyDropdown.value.trim(), 10) : null;
        const assemblyViewId = assemblyViewDropdown.value.trim() ? parseInt(assemblyViewDropdown.value.trim(), 10) : null;

        // Optional: Validate parsed integer values and warn if any are invalid
        if (assetNumberInput.value.trim() && isNaN(assetNumberId)) {
            console.warn(`Invalid Asset Number ID in position section.`);
        }
        if (locationInput.value.trim() && isNaN(locationId)) {
            console.warn(`Invalid Location ID in position section.`);
        }
        if (assemblyDropdown.value.trim() && isNaN(assemblyId)) {
            console.warn(`Invalid Assembly ID in position section.`);
        }
        if (subassemblyDropdown.value.trim() && isNaN(subassemblyId)) {
            console.warn(`Invalid Subassembly ID in position section.`);
        }
        if (assemblyViewDropdown.value.trim() && isNaN(assemblyViewId)) {
            console.warn(`Invalid Assembly View ID in position section.`);
        }

        // Construct the payload object; update keys if your backend requires a different name.
        const positionData = {
            area_id: areaId,
            equipment_group_id: equipmentGroupId,
            model_id: modelId,
            asset_number_id: assetNumberId,
            location_id: locationId,
            site_location_id: siteLocationId,

            // New fields for assembly information.
            // If your backend now expects "component_assembly_id" instead of "subassembly_id", you can change the key here.
            assembly_id: assemblyId,
            subassembly_id: subassemblyId,
            assembly_view_id: assemblyViewId
        };

        console.log('Collected Position Data:', positionData);
        positionsData.push(positionData);
    });

    return positionsData;
}


    async function fetchWithHandling(url, options = {}) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error('Fetch error:', error);
            throw error;
        }
    }

    // === 7. Initialize Event Listeners ===
    function initializeEventListeners() {
    const saveTaskBtn = document.getElementById('saveTaskBtn');
    if (saveTaskBtn) {
        saveTaskBtn.addEventListener('click', saveTaskDetails);
    }

    // Add event listener for Update Task Details button
    const updateTaskDetailsBtn = document.getElementById('updateTaskDetailsBtn');
    if (updateTaskDetailsBtn) {
        updateTaskDetailsBtn.addEventListener('click', updateTaskDetails);
    } else {
        console.warn("Update Task Details button with ID 'updateTaskDetailsBtn' not found.");
    }

    async function updateTaskDetails() {
    const taskId = window.AppState.currentTaskId;
    const taskNameInput = document.getElementById('pst_task_edit_task_name');
    const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
    const taskName = taskNameInput.value.trim();
    const taskDescription = taskDescriptionTextarea.value.trim();

    if (!taskId) {
        SolutionTaskCommon.showAlert('No task selected to update.', 'warning');
        return;
    }

    if (!taskName) {
        SolutionTaskCommon.showAlert('Task name cannot be empty.', 'warning');
        return;
    }

    const payload = {
        task_id: taskId,
        name: taskName,
        description: taskDescription
    };

    const updateBtn = document.getElementById('updateTaskDetailsBtn');
    let originalBtnText = '';
    if (updateBtn) {
        updateBtn.disabled = true;
        originalBtnText = updateBtn.textContent;
        updateBtn.textContent = 'Updating...';
    }

    try {
        const response = await fetch(ENDPOINTS.tasks.update, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            SolutionTaskCommon.showAlert('Task updated successfully.', 'success');
        } else {
            SolutionTaskCommon.showAlert(data.error || 'Failed to update task.', 'danger');
            console.error('Failed to update task:', data.error || data.message);
        }
    } catch (error) {
        console.error('Error updating task:', error);
        SolutionTaskCommon.showAlert('An error occurred while updating the task.', 'danger');
    } finally {
        if (updateBtn) {
            updateBtn.disabled = false;
            updateBtn.textContent = originalBtnText;
        }
    }
}

        // Ensure the "Save Documents" button has the correct event listener
        const saveDocumentsBtn = document.getElementById('saveDocumentsBtn');
        if (saveDocumentsBtn) {
            saveDocumentsBtn.addEventListener('click', saveSelectedDocuments);
        } else {
            console.warn("Save Documents button with ID 'saveDocumentsBtn' not found.");
        }
    }

    // Initialize Select2 for Drawing Search
    $('#pst_task_edit_task_drawings').select2({
        placeholder: 'Select or search for drawings',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchDrawings, // Define this endpoint in the ENDPOINTS object
            dataType: 'json',
            delay: 250,
            data: params => ({ q: params.term }), // Use 'q' as defined in the route
            processResults: data => ({
                results: data // Directly use the array of drawing objects with 'id' and 'text'
            }),
            cache: true
        }
    });

    async function saveSelectedDrawings() {
        const selectedDrawingIds = $('#pst_task_edit_task_drawings').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save drawings.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            drawing_ids: selectedDrawingIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveDrawings, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Drawings saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save drawings.', 'danger');
            }
        } catch (error) {
            console.error('Error saving drawings:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving drawings.', 'danger');
        }
    }

    // Ensure the "Save Drawings" button has the correct event listener
    const saveDrawingsBtn = document.getElementById('saveDrawingsBtn');
    if (saveDrawingsBtn) {
        saveDrawingsBtn.addEventListener('click', saveSelectedDrawings);
    } else {
        console.warn("Save Drawings button with ID 'saveDrawingsBtn' not found.");
    }

    // Initialize Select2 for Part Search
    $('#pst_task_edit_task_parts').select2({
        placeholder: 'Select or search for parts',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchParts,
            dataType: 'json',
            delay: 250,
            data: params => ({ q: params.term }),
            processResults: data => {
                console.log("Received part data:", data);
                return { results: data };
            },
            cache: true
        }
    });

    async function saveSelectedParts() {
        const selectedPartIds = $('#pst_task_edit_task_parts').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save parts.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            part_ids: selectedPartIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveParts, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Parts saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save parts.', 'danger');
            }
        } catch (error) {
            console.error('Error saving parts:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving parts.', 'danger');
        }
    }

    const savePartsBtn = document.getElementById('savePartsBtn');
    if (savePartsBtn) {
    console.log('Save Parts button found.'); // Confirm button existence
    savePartsBtn.addEventListener('click', saveSelectedParts);
} else {
    console.warn('Save Parts button with ID "savePartsBtn" not found.');
}

    $('#pst_task_edit_task_images').select2({
    placeholder: 'Select or search for images',
    allowClear: true,
    ajax: {
        url: ENDPOINTS.tasks.searchImages, // Ensure this is defined in the ENDPOINTS object
        dataType: 'json',
        delay: 250,
        data: params => ({ q: params.term }), // Use 'q' as defined in the route
        processResults: data => ({
            results: data.map(image => ({
                id: image.id,
                text: `${image.title} - ${image.description}`
            }))
        }),
        cache: true
    }
});

    async function saveSelectedImages() {
        const selectedImageIds = $('#pst_task_edit_task_images').val();
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save images.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            image_ids: selectedImageIds
        };

        try {
            const response = await fetch(ENDPOINTS.tasks.saveImages, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Images saved successfully.', 'success');
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save images.', 'danger');
            }
        } catch (error) {
            console.error('Error saving images:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving images.', 'danger');
        }
    }

    // Inside your initializeEventListeners function
    const saveImagesBtn = document.getElementById('saveImagesBtn');
    if (saveImagesBtn) {
        saveImagesBtn.addEventListener('click', saveSelectedImages);
    } else {
        console.warn("Save Images button with ID 'saveImagesBtn' not found.");
    }

    //===8
    // Function to update selected parts display with Remove buttons
    /**
     * Function to update the selected parts display with Remove buttons
     */
    function updateSelectedPartsDisplay() {
        const selectedPartIds = $('#pst_task_edit_task_parts').val();
        const selectedPartsContainer = $('#pst_task_edit_selected_parts');
        const taskId = window.AppState.currentTaskId;

        // Clear the container first
        selectedPartsContainer.empty();

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to manage parts.', 'warning');
            return;
        }

        selectedPartIds.forEach(id => {
            const partText = $('#pst_task_edit_task_parts option[value="' + id + '"]').text();
            const partDiv = $('<div class="selected-item d-flex align-items-center"></div>')
                .text(partText)
                .append(
                    $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                        .on('click', function () {
                            // Confirm removal
                            const confirmDeletion = confirm('Are you sure you want to remove this part?');
                            if (!confirmDeletion) return;

                            // Optimistically remove the item from the UI
                            partDiv.remove();

                            // Call the backend to remove the association
                            removeTaskPart(taskId, id);
                        })
                );
            selectedPartsContainer.append(partDiv);
        });
    }

    /**
     * Function to remove a part from a task by calling the backend endpoint
     * @param {number} taskId - The ID of the task
     * @param {number} partId - The ID of the part to remove
     */
    async function removeTaskPart(taskId, partId) {
        try {
            console.log(`Attempting to remove Part ID ${partId} from Task ID ${taskId}`);
            const response = await fetch(ENDPOINTS.tasks.removePart, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, part_id: partId })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                SolutionTaskCommon.showAlert('Part removed successfully.', 'success');
                console.log(`Part ID ${partId} removed from Task ID ${taskId}.`);
            } else {
                // Re-add the part to the UI if removal failed
                SolutionTaskCommon.showAlert(data.message || 'Failed to remove part.', 'danger');
                console.error('Failed to remove part:', data.message);
                updateSelectedPartsDisplay();
            }
        } catch (error) {
            // Re-add the part to the UI if an error occurred
            SolutionTaskCommon.showAlert('An error occurred while removing the part.', 'danger');
            console.error('Error removing part:', error);
            updateSelectedPartsDisplay();
        }
    }

    // Event listener for Select2 change event on parts
    $('#pst_task_edit_task_parts').on('change', updateSelectedPartsDisplay);


    // Event listener for Select2 change event on parts
    $('#pst_task_edit_task_parts').on('change', updateSelectedPartsDisplay);

    /**
     * Function to update the selected drawings display with Remove buttons
     */
    function updateSelectedDrawingsDisplay() {
        const selectedDrawingIds = $('#pst_task_edit_task_drawings').val();
        const selectedDrawingsContainer = $('#pst_task_edit_selected_drawings');
        const taskId = window.AppState.currentTaskId;

        // Clear the container first
        selectedDrawingsContainer.empty();

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to manage drawings.', 'warning');
            return;
        }

        selectedDrawingIds.forEach(id => {
            const drawingText = $('#pst_task_edit_task_drawings option[value="' + id + '"]').text();
            const drawingDiv = $('<div class="selected-item d-flex align-items-center"></div>')
                .text(drawingText)
                .append(
                    $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                        .on('click', function () {
                            // Confirm removal
                            const confirmDeletion = confirm('Are you sure you want to remove this drawing?');
                            if (!confirmDeletion) return;

                            // Optimistically remove the item from the UI
                            drawingDiv.remove();

                            // Call the backend to remove the association
                            removeTaskDrawing(taskId, id);
                        })
                );
            selectedDrawingsContainer.append(drawingDiv);
        });
    }

    /**
     * Function to remove a drawing from a task by calling the backend endpoint
     * @param {number} taskId - The ID of the task
     * @param {number} drawingId - The ID of the drawing to remove
     */
    async function removeTaskDrawing(taskId, drawingId) {
        try {
            console.log(`Attempting to remove Drawing ID ${drawingId} from Task ID ${taskId}`);
            const response = await fetch(ENDPOINTS.tasks.removeDrawing, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, drawing_id: drawingId })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                SolutionTaskCommon.showAlert('Drawing removed successfully.', 'success');
                console.log(`Drawing ID ${drawingId} removed from Task ID ${taskId}.`);
            } else {
                // Re-add the drawing to the UI if removal failed
                SolutionTaskCommon.showAlert(data.message || 'Failed to remove drawing.', 'danger');
                console.error('Failed to remove drawing:', data.message);
                updateSelectedDrawingsDisplay();
            }
        } catch (error) {
            // Re-add the drawing to the UI if an error occurred
            SolutionTaskCommon.showAlert('An error occurred while removing the drawing.', 'danger');
            console.error('Error removing drawing:', error);
            updateSelectedDrawingsDisplay();
        }
    }

    // Event listener for Select2 change event on drawings
    $('#pst_task_edit_task_drawings').on('change', updateSelectedDrawingsDisplay);

    // Event listener for Select2 change event on drawings
    $('#pst_task_edit_task_drawings').on('change', updateSelectedDrawingsDisplay);

    /**
     * Function to update the selected images display with Remove buttons
     */
    function updateSelectedImagesDisplay() {
        const selectedImageIds = $('#pst_task_edit_task_images').val();
        const selectedImagesContainer = $('#pst_task_edit_selected_images');
        const taskId = window.AppState.currentTaskId;

        // Clear the container first
        selectedImagesContainer.empty();

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to manage images.', 'warning');
            return;
        }

        selectedImageIds.forEach(id => {
            const imageText = $('#pst_task_edit_task_images option[value="' + id + '"]').text();
            const imageDiv = $('<div class="selected-item d-flex align-items-center"></div>')
                .text(imageText)
                .append(
                    $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                        .on('click', function () {
                            // Confirm removal
                            const confirmDeletion = confirm('Are you sure you want to remove this image?');
                            if (!confirmDeletion) return;

                            // Optimistically remove the item from the UI
                            imageDiv.remove();

                            // Call the backend to remove the association
                            removeTaskImage(taskId, id);
                        })
                );
            selectedImagesContainer.append(imageDiv);
        });
    }

    /**
     * Function to remove an image from a task by calling the backend endpoint
     * @param {number} taskId - The ID of the task
     * @param {number} imageId - The ID of the image to remove
     */
    async function removeTaskImage(taskId, imageId) {
        try {
            console.log(`Attempting to remove Image ID ${imageId} from Task ID ${taskId}`);
            const response = await fetch(ENDPOINTS.tasks.removeImage, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, image_id: imageId })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                SolutionTaskCommon.showAlert('Image removed successfully.', 'success');
                console.log(`Image ID ${imageId} removed from Task ID ${taskId}.`);
            } else {
                // Re-add the image to the UI if removal failed
                SolutionTaskCommon.showAlert(data.message || 'Failed to remove image.', 'danger');
                console.error('Failed to remove image:', data.message);
                // Optionally, you can reload the display to reflect the actual state
                updateSelectedImagesDisplay();
            }
        } catch (error) {
            // Re-add the image to the UI if an error occurred
            SolutionTaskCommon.showAlert('An error occurred while removing the image.', 'danger');
            console.error('Error removing image:', error);
            updateSelectedImagesDisplay();
        }
    }

    // Event listener for Select2 change event on images
    $('#pst_task_edit_task_images').on('change', updateSelectedImagesDisplay);

    // Event listener for Select2 change event on images
    $('#pst_task_edit_task_images').on('change', updateSelectedImagesDisplay);

    // Function to update the selected documents display with Remove buttons
    function updateSelectedDocumentsDisplay() {
        const selectedDocumentIds = $('#pst_task_edit_task_documents').val() || [];
        const selectedDocumentsContainer = $('#pst_task_edit_selected_documents');
        const taskId = window.AppState.currentTaskId;

        // Clear the container first
        selectedDocumentsContainer.empty();

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to manage documents.', 'warning');
            return;
        }

        // Fetch selected options and add them to the custom display container
        selectedDocumentIds.forEach(id => {
            const documentText = $('#pst_task_edit_task_documents option[value="' + id + '"]').text();
            const documentDiv = $('<div class="selected-item d-flex align-items-center"></div>')
                .text(documentText)
                .append(
                    $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                        .on('click', function () {
                            // Confirm removal
                            const confirmDeletion = confirm('Are you sure you want to remove this document?');
                            if (!confirmDeletion) return;

                            // Optimistically remove the item from the UI
                            documentDiv.remove();

                            // Remove the item from Select2 and update the display
                            const updatedSelection = $('#pst_task_edit_task_documents').val().filter(val => val !== id);
                            $('#pst_task_edit_task_documents').val(updatedSelection).trigger('change');

                            // Call the backend to remove the association
                            removeTaskDocument(taskId, id);
                        })
                );
            selectedDocumentsContainer.append(documentDiv);
        });
    }

    // Listen for change events on the select2 element
    $('#pst_task_edit_task_images').on('change', updateSelectedImagesDisplay);

    // === 8. Initialize Select2 for Tools ===
    // Initialize Select2 for Tool Search
    $('#pst_task_edit_task_tools').select2({
        placeholder: 'Select or search for tools',
        allowClear: true,
        ajax: {
            url: ENDPOINTS.tasks.searchTool, // Ensure this endpoint is correctly defined in ENDPOINTS
            dataType: 'json',
            delay: 250,
            data: params => ({ q: params.term }), // 'q' is the query parameter expected by the backend
            processResults: data => ({
                results: data.map(tool => ({
                    id: tool.id,
                    text: `${tool.name} (${tool.type || 'No Type'})` // Customize as needed
                }))
            }),
            cache: true
        }
    });

    // Listen for change events on the Select2 element
    $('#pst_task_edit_task_tools').on('change', updateSelectedToolsDisplay);

    /**
     * Function to update the selected tools display with Remove buttons
     */
    function updateSelectedToolsDisplay() {
        const selectedToolIds = $('#pst_task_edit_task_tools').val() || [];
        const selectedToolsContainer = $('#pst_task_edit_selected_tools');
        const taskId = window.AppState.currentTaskId;

        // Clear the container first
        selectedToolsContainer.empty();

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to manage tools.', 'warning');
            return;
        }

        selectedToolIds.forEach(id => {
            const toolText = $('#pst_task_edit_task_tools option[value="' + id + '"]').text();
            const toolDiv = $('<div class="selected-item d-flex align-items-center mb-2"></div>')
                .text(toolText)
                .append(
                    $('<button type="button" class="btn btn-sm btn-danger ms-2">Remove</button>')
                        .on('click', function () {
                            // Confirm removal
                            const confirmDeletion = confirm('Are you sure you want to remove this tool?');
                            if (!confirmDeletion) return;

                            // Optimistically remove the item from the UI
                            toolDiv.remove();

                            // Call the backend to remove the association
                            removeTaskTool(taskId, id);
                        })
                );
            selectedToolsContainer.append(toolDiv);
        });
    }

    /**
     * Function to save selected tools for a task
     */
    async function saveSelectedTools() {
        const selectedToolIds = $('#pst_task_edit_task_tools').val(); // Array of selected tool IDs
        const taskId = window.AppState.currentTaskId;

        if (!taskId) {
            SolutionTaskCommon.showAlert('No task selected to save tools.', 'warning');
            return;
        }

        const payload = {
            task_id: taskId,
            tool_ids: selectedToolIds
        };

        // Get CSRF token if using Flask-WTF
        const csrfToken = $('input[name="csrf_token"]').val();
        if (csrfToken) {
            payload.csrf_token = csrfToken;
        }

        try {
            const response = await fetch(ENDPOINTS.tasks.saveTool, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken // Include CSRF token in headers if needed
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Tools saved successfully.', 'success');
                console.log('Tools saved:', data);
                updateSelectedToolsDisplay(); // Refresh the display
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to save tools.', 'danger');
                console.error('Failed to save tools:', data.message);
            }
        } catch (error) {
            console.error('Error saving tools:', error);
            SolutionTaskCommon.showAlert('An error occurred while saving tools.', 'danger');
        }
    }

    // Attach event listener to the Save Tools button
    const saveToolsBtn = document.getElementById('saveToolsBtn');
    if (saveToolsBtn) {
        saveToolsBtn.addEventListener('click', saveSelectedTools);
    } else {
        console.warn("Save Tools button with ID 'saveToolsBtn' not found.");
    }

    /**
     * Function to remove a tool from a task by calling the backend endpoint
     * @param {number} taskId - The ID of the task
     * @param {number} toolId - The ID of the tool to remove
     */
    async function removeTaskTool(taskId, toolId) {
        const payload = {
            task_id: taskId,
            tool_id: toolId
        };

        // Get CSRF token if using Flask-WTF
        const csrfToken = $('input[name="csrf_token"]').val();
        if (csrfToken) {
            payload.csrf_token = csrfToken;
        }

        try {
            const response = await fetch(ENDPOINTS.tasks.removeTool, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken // Include CSRF token in headers if needed
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.status === 'success') {
                SolutionTaskCommon.showAlert('Tool removed successfully.', 'success');
                console.log(`Tool ID ${toolId} removed from Task ID ${taskId}.`);
                updateSelectedToolsDisplay(); // Refresh the display
            } else {
                SolutionTaskCommon.showAlert(data.message || 'Failed to remove tool.', 'danger');
                console.error('Failed to remove tool:', data.message);
                updateSelectedToolsDisplay(); // Re-render to reflect actual state
            }
        } catch (error) {
            console.error('Error removing tool:', error);
            SolutionTaskCommon.showAlert('An error occurred while removing the tool.', 'danger');
            updateSelectedToolsDisplay(); // Re-render to reflect actual state
        }
    }

    /**
     * Function to remove a document from a task by calling the backend endpoint
     * @param {number} taskId - The ID of the task
     * @param {number} documentId - The ID of the document to remove
     */
    async function removeTaskDocument(taskId, documentId) {
        try {
            console.log(`Attempting to remove Document ID ${documentId} from Task ID ${taskId}`);
            const response = await fetch(ENDPOINTS.tasks.removeDocument, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: taskId, document_id: documentId })
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                SolutionTaskCommon.showAlert('Document removed successfully.', 'success');
                console.log(`Document ID ${documentId} removed from Task ID ${taskId}.`);
            } else {
                // Re-add the document to the UI if removal failed
                SolutionTaskCommon.showAlert(data.message || 'Failed to remove document.', 'danger');
                console.error('Failed to remove document:', data.message);
                updateSelectedDocumentsDisplay();
            }
        } catch (error) {
            // Re-add the document to the UI if an error occurred
            SolutionTaskCommon.showAlert('An error occurred while removing the document.', 'danger');
            console.error('Error removing document:', error);
            updateSelectedDocumentsDisplay();
        }
    }

    // Event listener for Select2 change event on documents
    $('#pst_task_edit_task_documents').on('change', updateSelectedDocumentsDisplay);

    // === 13. Remove Position Function ===
    async function handleRemovePosition(positionSection, index) {
        const positionId = positionSection.getAttribute('data-position-id');
        const taskId = window.AppState.currentTaskId; // Ensure this is correctly set
        const removeBtn = positionSection.querySelector('.removePositionBtn');

        // Confirm deletion with the user
        const confirmDeletion = confirm('Are you sure you want to remove this position?');
        if (!confirmDeletion) return;

        let originalBtnText = '';
        if (removeBtn) {
            // Disable the Remove button to prevent multiple clicks
            removeBtn.disabled = true;
            originalBtnText = removeBtn.textContent;
            removeBtn.textContent = 'Removing...';
        }

        // Check if positionId is a temporary ID
        if (positionId && !positionId.startsWith('temp-')) {
            // The position exists in the backend; proceed to delete the association
            try {
                // Make a POST request to remove the position association
                const response = await fetch(ENDPOINTS.tasks.removePosition, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ task_id: taskId, position_id: positionId })
                });

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    // Remove the position from the DOM
                    positionSection.remove();
                    SolutionTaskCommon.showAlert('Position removed successfully.', 'success');
                    console.log(`Removed position section with index ${index} and position_id ${positionId}`);
                } else {
                    // Handle errors returned by the backend
                    SolutionTaskCommon.showAlert(data.error || 'Failed to remove position.', 'danger');
                    console.error('Failed to remove position:', data.error || data.message);
                    if (removeBtn) {
                        // Re-enable the Remove button and restore original text
                        removeBtn.disabled = false;
                        removeBtn.textContent = originalBtnText;
                    }
                }
            } catch (error) {
                // Handle network or unexpected errors
                console.error('Error removing position:', error);
                SolutionTaskCommon.showAlert('An error occurred while removing the position.', 'danger');
                if (removeBtn) {
                    // Re-enable the Remove button and restore original text
                    removeBtn.disabled = false;
                    removeBtn.textContent = originalBtnText;
                }
            }
        } else {
            // If position_id is a temporary ID, remove from DOM without backend call
            positionSection.remove();
            SolutionTaskCommon.showAlert('Position removed successfully.', 'info');
            console.log(`Removed position section with index ${index} without backend association.`);
            if (removeBtn) {
                // Re-enable the Remove button and restore original text
                removeBtn.disabled = false;
                removeBtn.textContent = originalBtnText;
            }
        }
    }

        initializeEventListeners();
    });