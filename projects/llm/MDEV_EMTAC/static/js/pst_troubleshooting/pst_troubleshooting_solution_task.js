// pst_troubleshooting_solution_task_edit.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // === 1. Define Backend Endpoint URLs ===
    const ENDPOINTS = {
        solutions: {
            get: '/pst_troubleshooting_solution/get_solutions/',
            add: '/pst_troubleshooting_solution/add_solution/',
            remove: '/pst_troubleshooting_solution/remove_solutions/',
        },
        tasks: {
            get: '/pst_troubleshooting_solution/get_tasks/',
            add: '/pst_troubleshooting_solution/add_task/',
            remove: '/pst_troubleshooting_solution/remove_task/',
            details: '/pst_troubleshooting_task/get_task_details/',
            update: '/pst_troubleshooting_task/update_task_details/',
        },
        initialData: {
        areas: '/pst_troubleshooting_guide_edit_update/get_areas',
        equipmentGroups: '/pst_troubleshooting_guide_edit_update/get_equipment_groups',
        models: '/pst_troubleshooting_guide_edit_update/get_models',
        assetNumbers: '/pst_troubleshooting_guide_edit_update/get_asset_numbers',
        locations: '/pst_troubleshooting_guide_edit_update/get_locations',
        siteLocations: '/pst_troubleshooting_guide_edit_update/get_site_locations',
        subassemblies: '/pst_troubleshooting_guide_edit_update/get_subassemblies',
        componetassemblies: '/pst_troubleshooting_guide_edit_update/component_assemblies',
        assemblyViews: '/pst_troubleshooting_guide_edit_update/get_assembly_views',
        },
        associations: { // Updated to include tools
    image: {
        remove: '/pst_troubleshooting_guide_edit_update/remove_task_image',
        // Add other image-related endpoints if necessary
    },
    part: {
        remove: '/pst_troubleshooting_guide_edit_update/remove_task_part',
        // Add other part-related endpoints if necessary
    },
    document: {
        remove: '/pst_troubleshooting_guide_edit_update/remove_task_document',
        // Add other document-related endpoints if necessary
    },
    drawing: {
        remove: '/pst_troubleshooting_guide_edit_update/remove_task_drawing',
        // Add other drawing-related endpoints if necessary
    },
    tool: {
        remove: '/pst_troubleshooting_guide_edit_update/remove_task_tools',
        // Add other tool-related endpoints if necessary
    }
}
    };

    // === 2. Namespace for Common Functions ===
    window.SolutionTaskCommon = {
    /**
     * Show alert messages to the user
     * @param {string} message - The message to display
     * @param {string} category - Bootstrap alert category (e.g., 'success', 'danger')
     */
    showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.setAttribute('role', 'alert');
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            alertContainer.appendChild(alertDiv);
            // Automatically remove the alert after 5 seconds
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        } else {
            console.warn("Alert container with ID 'alertContainer' not found.");
        }
    },

    /**
     * Populate a dropdown (select element) with data
     * @param {HTMLElement} dropdown - The select element
     * @param {Array} data - The array of data objects to populate
     * @param {string} placeholder - The placeholder text for the first option
     */
    populateDropdown(dropdown, data, placeholder) {
        if (!dropdown) {
            console.warn(`Dropdown element not provided.`);
            return;
        }

        // Clear existing options
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;

        if (!Array.isArray(data)) {
            console.error(`Expected an array for dropdown, but received:`, data);
            this.showAlert(`Invalid data format for dropdown.`, 'danger');
            dropdown.disabled = true;
            return;
        }

        // Populate new options
        data.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.textContent = this.getDropdownText(item);
            dropdown.appendChild(option);
        });

        // Disable the dropdown if no data is available
        dropdown.disabled = data.length === 0;
    },

    /**
     * Determine the display text for a dropdown option based on available properties
     * @param {Object} item - The data object
     * @returns {string} - The display text
     */
    getDropdownText(item) {
        if (item.name) {
            return item.name;
        } else if (item.title && item.room_number) {
            return `${item.title} - Room ${item.room_number}`;
        } else if (item.number) { // For Asset Numbers
            return item.number;
        } else {
            return `ID: ${item.id}`;
        }
    },

    /**
 * Update the display box for selected items (images, parts, drawings, documents)
 * @param {string} displayBoxId - The ID of the display container
 * @param {Array} items - The array of selected items
 * @param {string} itemType - The type of items ('image', 'part', 'drawing', 'document')
 */
updateSelectedDisplay(displayBoxId, items, itemType) {
    console.log(`Function called: updateSelectedDisplay(displayBoxId='${displayBoxId}', itemType='${itemType}', items=`, items, `)`);

    const displayBox = document.getElementById(displayBoxId);
    if (!displayBox) {
        console.warn(`Element with ID '${displayBoxId}' not found.`);
        return;
    }
    console.log(`Clearing existing items in display box with ID '${displayBoxId}'.`);
    displayBox.innerHTML = ''; // Clear existing items

    if (!Array.isArray(items) || items.length === 0) {
        console.log('No items to display. Setting display box text to "No items selected."');
        displayBox.textContent = 'No items selected.';
        return;
    }

    // Retrieve the current task ID from AppState
    const taskId = window.AppState?.currentTaskId;
    console.log(`Retrieved taskId from AppState: ${taskId}`);

    if (!taskId) {
        SolutionTaskCommon.showAlert('No task selected. Unable to remove item.', 'warning');
        console.warn('No task ID available in AppState.');
        return;
    }

    items.forEach((item, index) => {
        console.log(`Processing item ${index + 1}/${items.length}:`, item);

        const itemDiv = document.createElement('div');
        itemDiv.classList.add('selected-item', 'd-flex', 'align-items-center', 'mb-1');

        let itemText = '';
        switch (itemType) {
            case 'part':
                itemDiv.textContent = `${item.part_number || 'N/A'} - ${item.name || 'N/A'}`;
                console.log(`Formatted part text: '${itemText}'`);
                break;
            case 'drawing':
                itemDiv.textContent = `${item.drw_number || 'N/A'} - ${item.drw_name || 'N/A'}`;
                console.log(`Formatted drawing text: '${itemText}'`);
                break;
            case 'image':
                itemDiv.textContent = `${item.title || 'N/A'} - ${item.description || 'N/A'}`;
                console.log(`Formatted image text: '${itemText}'`);
                break;
            case 'document':
                itemDiv.textContent = `${item.title || 'N/A'}`;
                console.log(`Formatted document text: '${itemText}'`);
                break;
            case 'tool':
                itemDiv.textContent = `${item.name || 'N/A'} ${item.type ? `(${item.type})` : ''}`;
                console.log(`Formatted tool text: '${itemText}'`);
                break;
            default:
                itemText = `Unknown item type: ${itemType}`;
                console.warn(`Unknown item type encountered: '${itemType}'`);
        }

        // Create remove button with data attributes
        const removeButton = document.createElement('button');
        removeButton.classList.add('btn', 'btn-sm', 'btn-outline-danger', 'ms-2', 'remove-item-btn');
        removeButton.textContent = 'Remove';
        removeButton.setAttribute('data-item-id', item.id);
        removeButton.setAttribute('data-item-type', itemType);
        itemDiv.appendChild(removeButton);
        displayBox.appendChild(itemDiv);
    });

    // Attach a single event listener using event delegation
    displayBox.addEventListener('click', async (event) => {
        if (event.target && event.target.matches('.remove-item-btn')) {
            const btn = event.target;
            const itemId = btn.getAttribute('data-item-id');
            const currentItemType = btn.getAttribute('data-item-type');

            if (!itemId || !currentItemType) {
                console.warn('Remove button missing data attributes.');
                return;
            }

            // Retrieve the current task ID from AppState
            const taskId = window.AppState?.currentTaskId;

            if (!taskId) {
                SolutionTaskCommon.showAlert('No task selected. Unable to remove item.', 'warning');
                console.warn('No task ID available in AppState.');
                return;
            }

            try {
                // Show a loading alert
                SolutionTaskCommon.showAlert(`Removing ${currentItemType}...`, 'info');

                // Access the correct remove endpoint
                const removeEndpoint = ENDPOINTS.associations[currentItemType]?.remove;

                if (!removeEndpoint) {
                    SolutionTaskCommon.showAlert(`No remove endpoint defined for ${currentItemType}.`, 'danger');
                    console.error(`Remove endpoint not found for itemType: ${currentItemType}`);
                    return;
                }

                // Make an API call to remove the item on the server
                const response = await fetch(removeEndpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_id: taskId, [`${currentItemType}_id`]: itemId }),
                });

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    // Remove the item from the local array
                    const itemIndex = items.findIndex(selectedItem => selectedItem.id === parseInt(itemId, 10));
                    if (itemIndex > -1) items.splice(itemIndex, 1);
                    // Update the display
                    SolutionTaskCommon.updateSelectedDisplay(displayBoxId, items, itemType);
                    // Show success alert
                    SolutionTaskCommon.showAlert(`${SolutionTaskCommon.capitalizeFirstLetter(currentItemType)} removed successfully.`, 'success');
                } else {
                    // Handle server-side errors
                    SolutionTaskCommon.showAlert(data.error || `Failed to remove ${currentItemType}.`, 'danger');
                    console.error(`Failed to remove ${currentItemType}:`, data.error || data.message);
                }
            } catch (error) {
                // Handle network or unexpected errors
                SolutionTaskCommon.showAlert(`An error occurred while removing the ${currentItemType}.`, 'danger');
                console.error(`Error removing ${currentItemType}:`, error);
            }
        }
    }, { once: false });
},

    /**
     * Clear input fields by their element references
     * @param  {...HTMLElement} elements - The input/select elements to clear
     */
    clearInputFields(...elements) {
        elements.forEach(element => {
            if (element) {
                if (element.tagName === 'SELECT') {
                    element.selectedIndex = 0;
                } else {
                    element.value = '';
                }
            } else {
                console.warn(`One of the provided elements is undefined.`);
            }
        });
    },

    /**
     * Clear the Edit Task form
     */
    clearEditTaskForm() {
        // Clear Task Name and Description
        const taskNameInput = document.getElementById('pst_task_edit_task_name');
        const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');
        this.clearInputFields(taskNameInput, taskDescriptionTextarea);

        // Clear Positions Container
        const positionsContainer = document.getElementById('pst_task_edit_positions_container');
        if (positionsContainer) {
            positionsContainer.innerHTML = '';
        } else {
            console.warn("Element with ID 'pst_task_edit_positions_container' not found.");
        }

        // Clear selected items displays
        ['pst_task_edit_selected_images', 'pst_task_edit_selected_parts', 'pst_task_edit_selected_drawings'].forEach(id => {
            const displayBox = document.getElementById(id);
            if (displayBox) {
                displayBox.innerHTML = '';
            } else {
                console.warn(`Element with ID '${id}' not found.`);
            }
        });
    },

    /**
     * Capitalize the first letter of a string
     * @param {string} string - The string to capitalize
     * @returns {string} - The capitalized string
     */
    capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    },

    /**
    * Fetch initial site locations independently
    * @returns {Promise<Array>} - Array of site location objects
    */
async fetchInitialSiteLocations() {
    try {
        const url = `${ENDPOINTS.initialData.siteLocations}`;
        const data = await fetchWithHandling(url);
        console.log(`Received Site Locations:`, data);
        // Since data is an array directly, check if data is an array
        return Array.isArray(data) ? data : [];
    } catch (error) {
        console.error(`Error fetching Site Locations:`, error);
        return [];
    }
},


    /**
     * Fetch initial areas
     * @returns {Promise<Array>} - Array of area objects
     */
    async fetchInitialAreas() {
        try {
            const data = await fetchWithHandling(ENDPOINTS.initialData.areas);
            return data.areas || [];
        } catch (error) {
            return [];
        }
    },

    /**
     * Fetch initial equipment groups based on area ID
     * @param {string|number} areaId
     * @returns {Promise<Array>} - Array of equipment group objects
     */
    async fetchInitialEquipmentGroups(areaId) {
        try {
            const url = `${ENDPOINTS.initialData.equipmentGroups}?area_id=${encodeURIComponent(areaId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Equipment Groups:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Equipment Groups for Area ID ${areaId}:`, error);
            return [];
        }
    },

    /**
     * Fetch initial models based on equipment group ID
     * @param {string|number} equipmentGroupId
     * @returns {Promise<Array>} - Array of model objects
     */
    async fetchInitialModels(equipmentGroupId) {
        try {
            const url = `${ENDPOINTS.initialData.models}?equipment_group_id=${encodeURIComponent(equipmentGroupId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Models:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Models for Equipment Group ID ${equipmentGroupId}:`, error);
            return [];
        }
    },

    /**
     * Fetch initial asset numbers based on model ID
     * @param {string|number} modelId
     * @returns {Promise<Array>} - Array of asset number objects
     */
    async fetchInitialAssetNumbers(modelId) {
        try {
            const url = `${ENDPOINTS.initialData.assetNumbers}?model_id=${encodeURIComponent(modelId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Asset Numbers:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Asset Numbers for Model ID ${modelId}:`, error);
            return [];
        }
    },

    /**
 * Fetch initial subassemblies based on Location ID
 * @param {string|number} locationId - The ID of the selected Location
 * @returns {Promise<Array>} - Array of assembly objects
 */
    async fetchInitialAssembliesByLocation(locationId) {
    try {
        // Build the request URL
        const url = `${ENDPOINTS.initialData.subassemblies}?location_id=${encodeURIComponent(locationId)}`;

        // Use the fetchWithHandling helper for consistency
        // This will handle the fetch, parse JSON, check response.ok, etc.
        const data = await fetchWithHandling(url);

        console.log(`Received subassemblies:`, data);

        // If your endpoint returns an array directly (like your models endpoint):
        //   return Array.isArray(data) ? data : [];
        //
        // If your endpoint wraps the array (like { subassemblies: [...] } ):
        //   return Array.isArray(data.subassemblies) ? data.subassemblies : [];

        // Assuming your endpoint returns a simple array (like [ {id, name}, ... ])
        return Array.isArray(data) ? data : [];
    } catch (error) {
        console.error(`Error fetching subassemblies for Location ID ${locationId}:`, error);
        // Return an empty array if an error occurs
        return [];
    }
},

    /**
     * Fetch initial locations based on model ID
     * @param {string|number} modelId
     * @returns {Promise<Array>} - Array of location objects
     */
    async fetchInitialLocations(modelId) {
        try {
            const url = `${ENDPOINTS.initialData.locations}?model_id=${encodeURIComponent(modelId)}`;
            const data = await fetchWithHandling(url);
            console.log(`Received Locations:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Locations for Model ID ${modelId}:`, error);
            return [];
        }
    },

    /**
     * Fetch initial componetassemblies based on assembly ID
     * @param {string|number} assemblyId
     * @returns {Promise<Array>} - Array of subassembly objects
     */
    async fetchInitialSubassemblies(assemblyId) {
        try {
            // Build the request URL
            const url = `${ENDPOINTS.initialData.componetassemblies}?assembly_id=${encodeURIComponent(assemblyId)}`;
            // Use the fetchWithHandling helper for consistency
            const data = await fetchWithHandling(url);
            console.log(`Received Subassemblies:`, data);
            return Array.isArray(data) ? data : [];
        } catch (error) {
            console.error(`Error fetching Subassemblies for Assembly ID ${assemblyId}:`, error);
            return [];
        }
    },

    /**
     * Fetch initial assembly views based on subassembly ID
     * @param {string|number} subassemblyId
     * @returns {Promise<Array>} - Array of assembly view objects
     */
    async fetchInitialAssemblyViews(subassemblyId) {
        try {
            // Build the request URL
            const url = `${ENDPOINTS.initialData.assemblyViews}?subassembly_id=${encodeURIComponent(subassemblyId)}`;
            // Use the fetchWithHandling helper for consistency
            const data = await fetchWithHandling(url);
            console.log(`Received Assembly Views:`, data);
            return Array.isArray(data) ? data : [];

        } catch (error) {
            console.error(`Error fetching Assembly Views for Subassembly ID ${subassemblyId}:`, error);
            return [];
        }
    },

};

    // === 3. Helper Function for Fetch with Error Handling ===
    async function fetchWithHandling(url, options = {}) {
        try {
            console.log(`Fetching URL: ${url}`);
            const response = await fetch(url, options);
            const data = await response.json();
            console.log(`Received data from ${url}:`, data);
            if (!response.ok) {
                throw new Error(data.error || 'Unknown error');
            }
            return data;
        } catch (error) {
            SolutionTaskCommon.showAlert(error.message || 'Unknown error', 'danger');
            console.error(`Error fetching ${url}:`, error);
            throw error;
        }
    }

    /**
     * Fetch and display solutions for a specific problem
     * @param {number|string} problemId - The ID of the current problem
     */
    async function fetchSolutions(problemId) {
        window.AppState.currentProblemId = problemId;
        SolutionTaskCommon.showAlert('Loading solutions...', 'info');
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.solutions.get}${encodeURIComponent(problemId)}`);

            // Verify data structure and solutions array
            const solutions = data.solutions || data; // Check if solutions are under data.solutions or directly under data
            if (!Array.isArray(solutions)) {
                console.error("Expected an array for solutions, but received:", solutions);
                return; // Exit if solutions is not an array
            }

            // Populate solutions dropdown
            populateSolutionsDropdown(solutions);

            // Update problem name if available
            if (data.problem_name) {
                updateProblemName(data.problem_name);
            }

            // Activate the solution tab
            activateTab('solution-tab');

            console.log("Data received from fetchSolutions:", solutions);
        } catch (error) {
            console.error("Error fetching solutions:", error);
            // Error handling is already managed in fetchWithHandling
        }
    }

    /**
 * Populate solutions dropdown
 * @param {Array} solutions - Array of solution objects
 */
    function populateSolutionsDropdown(solutions) {
    const solutionsDropdown = document.getElementById('existing_solutions');

    // Validate that solutions is an array
    if (!Array.isArray(solutions)) {
        console.error("Expected an array for solutions, but received:", solutions);
        return; // Exit function if data is not in the expected format
    }

    // Use the SolutionTaskCommon helper function to populate the dropdown
    SolutionTaskCommon.populateDropdown(solutionsDropdown, solutions, 'Select Solution');
    console.log("Solutions dropdown populated with:", solutions);
}

    /**
     * Update problem name header
     * @param {string} problemName - The name of the problem
     */
    function updateProblemName(problemName) {
        const header = document.getElementById('selected-problem-name');
        if (header) {
            header.textContent = `Problem Solutions for: ${problemName}`;
        } else {
            console.warn("Element with ID 'selected-problem-name' not found.");
        }
    }

    /**
     * Activate a specific tab
     * @param {string} tabId - The ID of the tab to activate
     */
    function activateTab(tabId) {
        const tabLink = document.getElementById(tabId);
        if (tabLink) {
            new bootstrap.Tab(tabLink).show();
        } else {
            console.warn(`Tab with ID '${tabId}' not found.`);
        }
    }

    /**
     * Add a new solution for a problem
     * @param {number|string} problemId - The ID of the current problem
     * @param {string} solutionName - The name of the new solution
     * @param {string} solutionDescription - The description of the new solution
     */
    async function addNewSolution(problemId, solutionName, solutionDescription) {
        if (!problemId) {
            SolutionTaskCommon.showAlert('No problem selected.', 'warning');
            return;
        }

        try {
            // Wait for the server to confirm the solution is added
            const response = await fetchWithHandling(ENDPOINTS.solutions.add, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem_id: problemId, name: solutionName, description: solutionDescription })
            });

            // Confirm that the response is successful and contains the expected data
            if (response && response.message === 'Solution added successfully.' && response.solution) {
                SolutionTaskCommon.showAlert('Solution added successfully.', 'success');

                // After confirming the solution is saved, fetch the updated solutions list
                await fetchSolutions(problemId);

                // Clear input fields after successful addition
                const nameInput = document.getElementById('new_solution_name');
                const descInput = document.getElementById('new_solution_description');
                SolutionTaskCommon.clearInputFields(nameInput, descInput);
            } else {
                console.error("Unexpected response format:", response);
                SolutionTaskCommon.showAlert('Failed to add the solution. Unexpected response from the server.', 'warning');
            }

        } catch (error) {
            console.error("Error in addNewSolution:", error);
            SolutionTaskCommon.showAlert('An error occurred while adding the solution.', 'danger');
        }
    }

    /**
     * Remove selected solutions from a problem
     * @param {number|string} problemId - The ID of the current problem
     * @param {Array} solutionIds - Array of solution IDs to remove
     */
    async function removeSolutions(problemId, solutionIds) {
        if (!problemId || !Array.isArray(solutionIds) || solutionIds.length === 0) {
            SolutionTaskCommon.showAlert('No solutions selected for removal.', 'warning');
            return;
        }
        try {
            await fetchWithHandling(ENDPOINTS.solutions.remove, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem_id: problemId, solution_ids: solutionIds })
            });
            SolutionTaskCommon.showAlert('Selected solutions removed successfully.', 'success');
            await fetchSolutions(problemId);
        } catch (error) {
            // Error handling is already managed in fetchWithHandling
        }
    }

    /**
     * Fetch tasks for a selected solution and populate the task dropdown
     * @param {number|string} solutionId - The ID of the selected solution
     */
    async function fetchTasksForSolution(solutionId) {
        if (!solutionId) {
            console.warn("Invalid solutionId passed to fetchTasksForSolution.");
            return;
        }
        SolutionTaskCommon.showAlert('Loading tasks...', 'info');
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.tasks.get}${encodeURIComponent(solutionId)}`);
            populateTasksDropdown(data.tasks);
            activateTab('task-tab');
        } catch (error) {
            // Error handling is already managed in fetchWithHandling
        }
    }

    /**
     * Populate tasks in a dropdown list
     * @param {Array} tasks - Array of task objects
     */
    function populateTasksDropdown(tasks) {
        const tasksDropdown = document.getElementById('existing_tasks');
        SolutionTaskCommon.populateDropdown(tasksDropdown, tasks, 'Select Task');


    }

    // Single-click event listener for selecting a task
    document.getElementById('existing_tasks').addEventListener('click', (event) => {
        window.AppState.currentTaskId = event.target.value; // Set the selected task ID for possible removal
        console.log(`Task selected with ID: ${window.AppState.currentTaskId}`);
    });

    // Double-click event listener for editing a task
    document.getElementById('existing_tasks').addEventListener('dblclick', (event) => {
        const taskId = event.target.value; // Get the ID of the double-clicked task
        if (taskId) {
            openTaskDetails(taskId); // Call function to open the edit form
        }
    });

    async function openTaskDetails(taskId) {
        // Assign the task ID to the global AppState
        window.AppState.currentTaskId = taskId;
        console.log(`openTaskDetails called with taskId: ${window.AppState.currentTaskId}, solutionId: ${window.AppState.currentSolutionId}`);
        try {
            const data = await fetchWithHandling(`${ENDPOINTS.tasks.details}${encodeURIComponent(taskId)}`);

            // Populate the form with task data if found
            if (data && data.task) {
                await populateEditTaskForm(data.task);
                activateTab('edit-task-tab'); // Switch to the edit tab
                console.log(`Editing task with ID: ${taskId}`);
            } else {
                SolutionTaskCommon.clearEditTaskForm(); // Clear form if task is not found
                SolutionTaskCommon.showAlert('Task not found.', 'warning');
            }
        } catch (error) {
            console.error(`Error opening task details for task ID ${taskId}:`, error);
        }
    }

     /**
     * Add a new task to a solution
     * @param {number|string} solutionId - The ID of the current solution
     * @param {string} name - The name of the new task
     * @param {string} description - The description of the new task
     */
    async function addNewTask(solutionId = window.currentSolutionId, name, description) {
        // Confirm solution ID is valid
        console.log("Adding Task:", { solutionId, name, description });

        if (!solutionId) {
            SolutionTaskCommon.showAlert('No solution selected.', 'warning');
            return;
        }
        try {
            await fetchWithHandling(ENDPOINTS.tasks.add, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ solution_id: solutionId, name, description })
            });
            SolutionTaskCommon.showAlert('Task added successfully.', 'success');

            // Fetch updated tasks list for this solution
            await fetchTasksForSolution(solutionId);

            // Clear input fields
            const nameInput = document.getElementById('new_task_name');
            const descInput = document.getElementById('new_task_description');
            SolutionTaskCommon.clearInputFields(nameInput, descInput);
        } catch (error) {
            console.error("Error in addNewTask:", error);
            SolutionTaskCommon.showAlert('An error occurred while adding the task.', 'danger');
        }
    }

/**
 * Populate the Edit Task form with task data
 * @param {Object} task - The task data object
 */
async function populateEditTaskForm(task) {
    // Populate main Task Name and Description
    const taskNameInput = document.getElementById('pst_task_edit_task_name');
    const taskDescriptionTextarea = document.getElementById('pst_task_edit_task_description');

    if (taskNameInput) taskNameInput.value = task.name || '';
    if (taskDescriptionTextarea) taskDescriptionTextarea.value = task.description || '';

    // Populate Positions
    const positionsContainer = document.getElementById('pst_task_edit_positions_container');
    if (positionsContainer) {
        positionsContainer.innerHTML = ''; // Clear existing positions
        if (task.positions?.length) {
            for (let i = 0; i < task.positions.length; i++) {
                await addPosition(task.positions[i], i);
            }
        } else {
            SolutionTaskCommon.showAlert('No positions associated with this task.', 'info');
        }
    } else {
        console.warn("Element with ID 'pst_task_edit_positions_container' not found.");
    }

    // Populate associated images, parts, drawings, documents and tools
    console.log('Updating selected images:', task.associations.images);
    SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_images', task.associations?.images || [], 'image');
    console.log('Updating selected parts:', task.associations.parts);
    SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_parts', task.associations?.parts || [], 'part');
    console.log('Updating selected drawings:', task.associations.drawings);
    SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_drawings', task.associations?.drawings || [], 'drawing');
    console.log('Updating selected documents:', task.associations.completeDocuments);
    SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_documents', task.associations?.completeDocuments || [], 'document');
    console.log('Updating selected tools:', task.associations.tools);
    SolutionTaskCommon.updateSelectedDisplay('pst_task_edit_selected_tools', task.associations?.tools || [], 'tool');
}

    /**
 * Add a new position section
 * @param {Object} positionData - The data for the position (optional)
 * @param {number} index - The index of the position (for unique IDs)
 */
    async function addPosition(positionData = null, index = 0) {
    console.log(`\n=== addPosition called with index: ${index} and positionData:`, positionData);

    const container = document.getElementById('pst_task_edit_positions_container');
    const template = document.getElementById('position-template');

    if (!container || !template) {
        console.warn("Positions container or template not found.");
        return;
    }
    console.log("Positions container and template found.");
    console.log('Template innerHTML:', template.innerHTML);

    const clone = template.content.cloneNode(true);
    const positionSection = clone.querySelector('.position-section');

    if (!positionSection) {
        console.error("Position section element not found in the template.");
        return;
    }
    console.log("Cloned position section from template.");

    // Generate a unique ID. Using index as uniqueId here; alternatively, use a more robust unique identifier.
    const uniqueId = `${Date.now()}_${index}`; // e.g., "1701234567890_0"
    console.log(`Generated uniqueId: ${uniqueId}`);

    // Assign unique IDs to all select elements, including subassemblies, componetassemblies, and assemblyViews
    const elementsToId = [
        'areaDropdown',
        'equipmentGroupDropdown',
        'modelDropdown',
        'assetNumberInput',
        'locationInput',
        'siteLocationDropdown',
        'assembliesDropdown',
        'subassembliesDropdown',
        'assemblyViewsDropdown'
    ];

    elementsToId.forEach(elementClass => {
        const element = positionSection.querySelector(`.${elementClass}`);
        if (element) {
            const baseId = element.id; // e.g., 'assembliesDropdown_{{uniqueId}}'
            const newId = baseId.replace('__UNIQUE__', uniqueId);
            element.id = newId;
            console.log(`Assigned ID "${newId}" to element with class "${elementClass}".`);

            // Update corresponding label's 'for' attribute
            const label = positionSection.querySelector(`label[for="${baseId}"]`);
            if (label) {
                label.setAttribute('for', newId);
                console.log(`Updated label 'for' attribute to "${newId}" for "${elementClass}".`);
            } else {
                console.warn(`Label for "${elementClass}" not found.`);
            }

            // Additional log to verify the updated element's ID
            console.log(`Element after ID assignment:`, element);
        } else {
            console.warn(`Element with class "${elementClass}" not found in position section.`);
        }
    });

    // Assign unique ID to the Remove Position button if necessary
    const removeBtn = positionSection.querySelector('.removePositionBtn');
    if (removeBtn) {
        const baseId = removeBtn.id; // e.g., 'removePositionBtn_{{uniqueId}}'
        const newId = baseId.replace('{{uniqueId}}', uniqueId);
        removeBtn.id = newId;
        console.log(`Assigned new id to Remove Button: ${removeBtn.id}`);
    } else {
        console.warn("Remove Position button with class 'removePositionBtn' not found.");
    }

    try {
        // If positionData is provided, populate the fields
        if (positionData) {
            console.log('Position Data:', positionData);
            console.log("Position data provided. Populating fields with existing data.");
            await populatePositionFields(positionSection, positionData, uniqueId);
            // **Set the data-position-id attribute to the actual position_id from backend**
            if (positionData.position_id) {
                positionSection.setAttribute('data-position-id', positionData.position_id);
                console.log(`Set data-position-id to "${positionData.position_id}".`);
            } else {
                console.warn("positionData provided but position_id is missing.");
                positionSection.setAttribute('data-position-id', `temp-${Date.now()}`);
                console.log(`Assigned temporary data-position-id "temp-${Date.now()}" due to missing position_id.`);
            }
        } else {
            console.log("No position data provided. Initializing fields for a new position.");
            // **For new positions, set a temporary ID**
            await initializeNewPosition(positionSection, uniqueId);
            const tempId = `temp-${Date.now()}`;
            positionSection.setAttribute('data-position-id', tempId);
            console.log(`Set temporary data-position-id to "${tempId}".`);
        }

        // Append the cloned and populated position section to the container
        container.appendChild(clone);
        console.log(`Appended position section with uniqueId ${uniqueId} to the container.`);

        // Log the current state of the container
        console.log(`Current container HTML:`, container.innerHTML);
    } catch (error) {
        console.error(`Error in addPosition for uniqueId ${uniqueId}:`, error);
    }

    console.log(`=== addPosition completed for uniqueId: ${uniqueId} ===\n`);
}

    /**
 * Populate position fields based on provided data
 * @param {HTMLElement} positionSection - The position section element
 * @param {Object} positionData - The data for the position
 * @param {number} index - The index of the position
 */
    async function populatePositionFields(positionSection, positionData, index) {
    console.log(`\n--- populatePositionFields called for positionSection index: ${index} ---`);
    console.log('Position Data:', positionData);

    // === 1. Populate Area Dropdown ===
    const areaDropdown = positionSection.querySelector('.areaDropdown');
    if (areaDropdown) {
        try {
            console.log('Attempting to fetch areas...');
            const areas = await window.SolutionTaskCommon.fetchInitialAreas();
            console.log('Fetched areas:', areas);

            window.SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            console.log(`Dropdown options after populating Area Dropdown (ID: ${areaDropdown.id}):`, areaDropdown.options);

            areaDropdown.value = positionData.area_id || '';
            areaDropdown.disabled = false;
            console.log(`Set Area Dropdown (ID: ${areaDropdown.id}) to value: ${positionData.area_id}`);
        } catch (error) {
            console.error('Error fetching Areas:', error);
            window.SolutionTaskCommon.showAlert('Failed to load areas.', 'danger');
            window.SolutionTaskCommon.populateDropdown(areaDropdown, [], 'Select Area');
            areaDropdown.disabled = true;
            console.log(`Area Dropdown (ID: ${areaDropdown.id}) has been disabled due to error.`);
        }
    } else {
        console.warn('Area Dropdown element not found in position section.');
    }

    // === 2. Populate Equipment Group Dropdown based on Area ===
    const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
    if (equipmentGroupDropdown && positionData.area_id) {
        try {
            console.log(`Attempting to fetch Equipment Groups for Area ID: ${positionData.area_id}`);
            const equipmentGroups = await window.SolutionTaskCommon.fetchInitialEquipmentGroups(positionData.area_id);
            console.log('Fetched Equipment Groups:', equipmentGroups);

            window.SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, equipmentGroups, 'Select Equipment Group');
            console.log(`Dropdown options after populating Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}):`, equipmentGroupDropdown.options);

            // Validate equipment_group_id exists in fetched equipmentGroups
            const isValidId = equipmentGroups.some(group => String(group.id) === String(positionData.equipment_group_id));
            if (isValidId) {
                equipmentGroupDropdown.value = positionData.equipment_group_id;
                console.log(`Set Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) to value: ${positionData.equipment_group_id}`);
            } else {
                console.warn(`Invalid Equipment Group ID: ${positionData.equipment_group_id} for Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id})`);
                equipmentGroupDropdown.value = '';
            }
            equipmentGroupDropdown.disabled = false;
            console.log(`Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) has been enabled.`);
        } catch (error) {
            console.error('Error fetching Equipment Groups:', error);
            window.SolutionTaskCommon.showAlert('Failed to load equipment groups.', 'danger');
            window.SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, [], 'Select Equipment Group');
            equipmentGroupDropdown.disabled = true;
            console.log(`Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) has been disabled due to error.`);
        }
    } else if (!positionData.area_id) {
        console.warn('Area ID is missing in positionData; cannot populate Equipment Group Dropdown.');
    } else {
        console.warn('Equipment Group Dropdown element not found in position section.');
    }

    // === 3. Populate Model Dropdown based on Equipment Group ===
    const modelDropdown = positionSection.querySelector('.modelDropdown');
    if (modelDropdown && positionData.equipment_group_id) {
        try {
            console.log(`Attempting to fetch Models for Equipment Group ID: ${positionData.equipment_group_id}`);
            const models = await window.SolutionTaskCommon.fetchInitialModels(positionData.equipment_group_id);
            console.log('Fetched Models:', models);

            window.SolutionTaskCommon.populateDropdown(modelDropdown, models, 'Select Model');
            console.log(`Dropdown options after populating Model Dropdown (ID: ${modelDropdown.id}):`, modelDropdown.options);

            const isValidModelId = models.some(model => String(model.id) === String(positionData.model_id));
            if (isValidModelId) {
                modelDropdown.value = positionData.model_id;
                console.log(`Set Model Dropdown (ID: ${modelDropdown.id}) to value: ${positionData.model_id}`);
            } else {
                console.warn(`Invalid Model ID: ${positionData.model_id} for Model Dropdown (ID: ${modelDropdown.id})`);
                modelDropdown.value = '';
            }
            modelDropdown.disabled = false;
            console.log(`Model Dropdown (ID: ${modelDropdown.id}) has been enabled.`);
        } catch (error) {
            console.error('Error fetching Models:', error);
            window.SolutionTaskCommon.showAlert('Failed to load models.', 'danger');
            window.SolutionTaskCommon.populateDropdown(modelDropdown, [], 'Select Model');
            modelDropdown.disabled = true;
            console.log(`Model Dropdown (ID: ${modelDropdown.id}) has been disabled due to error.`);
        }
    } else if (!positionData.equipment_group_id) {
        console.warn('Equipment Group ID is missing in positionData; cannot populate Model Dropdown.');
    } else {
        console.warn('Model Dropdown element not found in position section.');
    }

    // === 4. Populate Asset Number based on Model ===
    const assetNumberInput = positionSection.querySelector('.assetNumberInput');
    if (assetNumberInput && positionData.model_id) {
        try {
            console.log(`Attempting to fetch Asset Numbers for Model ID: ${positionData.model_id}`);
            const assetNumbers = await window.SolutionTaskCommon.fetchInitialAssetNumbers(positionData.model_id);
            console.log('Fetched Asset Numbers:', assetNumbers);

            window.SolutionTaskCommon.populateDropdown(assetNumberInput, assetNumbers, 'Select Asset Number');
            console.log(`Dropdown options after populating Asset Number Input (ID: ${assetNumberInput.id}):`, assetNumberInput.options);

            const isValidAssetNumber = assetNumbers.some(asset => String(asset.id) === String(positionData.asset_number));
            const matchingAsset = assetNumbers.find(asset => String(asset.number) === String(positionData.asset_number));
            if (matchingAsset) {
                assetNumberInput.value = matchingAsset.id;
                console.log(`Set Asset Number Input (ID: ${assetNumberInput.id}) to value: ${matchingAsset.id}`);
            } else {
                console.warn(`Invalid Asset Number: ${positionData.asset_number} for Asset Number Input (ID: ${assetNumberInput.id})`);
                assetNumberInput.value = '';
            }
            assetNumberInput.disabled = false;
            console.log(`Asset Number Input (ID: ${assetNumberInput.id}) has been enabled.`);
        } catch (error) {
            console.error('Error fetching Asset Numbers:', error);
            window.SolutionTaskCommon.showAlert('Failed to load asset numbers.', 'danger');
            window.SolutionTaskCommon.populateDropdown(assetNumberInput, [], 'Select Asset Number');
            assetNumberInput.disabled = true;
            console.log(`Asset Number Input (ID: ${assetNumberInput.id}) has been disabled due to error.`);
        }
    } else if (!positionData.model_id) {
        console.warn('Model ID is missing in positionData; cannot populate Asset Number Input.');
    } else {
        console.warn('Asset Number Input element not found in position section.');
    }

    // === 5. Populate Location based on Model ===
    const locationInput = positionSection.querySelector('.locationInput');
    if (locationInput && positionData.model_id) {
        try {
            console.log(`Attempting to fetch Locations for Model ID: ${positionData.model_id}`);
            const locations = await window.SolutionTaskCommon.fetchInitialLocations(positionData.model_id);
            console.log('Fetched Locations:', locations);

            window.SolutionTaskCommon.populateDropdown(locationInput, locations, 'Select Location');
            console.log(`Dropdown options after populating Location Input (ID: ${locationInput.id}):`, locationInput.options);

            const locationId = positionData.location_id; // Use the correct property
            const matchingLocation = locations.find(location => String(location.id) === String(locationId));
            console.log(`Matching Location for ID ${locationId}:`, matchingLocation);

            if (matchingLocation) {
                locationInput.value = matchingLocation.id;
                console.log(`Set Location Input (ID: ${locationInput.id}) to value: ${matchingLocation.id}`);
            } else {
                console.warn(`Invalid Location ID: ${locationId} for Location Input (ID: ${locationInput.id})`);
                locationInput.value = '';
            }
            locationInput.disabled = false;
            console.log(`Location Input (ID: ${locationInput.id}) has been enabled.`);
        } catch (error) {
            console.error('Error fetching Locations:', error);
            window.SolutionTaskCommon.showAlert('Failed to load locations.', 'danger');
            window.SolutionTaskCommon.populateDropdown(locationInput, [], 'Select Location');
            locationInput.disabled = true;
            console.log(`Location Input (ID: ${locationInput.id}) has been disabled due to error.`);
        }
    } else if (!positionData.model_id) {
        console.warn('Model ID is missing in positionData; cannot populate Location Input.');
    } else {
        console.warn('Location Input element not found in position section.');
    }

    // === 6. Populate Assemblies ===
    const assembliesDropdown = positionSection.querySelector('.assembliesDropdown');
    if (assembliesDropdown) {
        try {
            console.log('Attempting to fetch Assemblies...');
            // **Important:** Ensure that you pass the required locationId to fetchInitialAssembliesByLocation
            const locationId = positionData.location_id;
            if (!locationId) {
                console.warn('Location ID is missing; cannot fetch Assemblies.');
                throw new Error('Location ID is required to fetch Assemblies.');
            }

            console.log(`Fetching Assemblies for Location ID: ${locationId}`);
            const subassemblies = await SolutionTaskCommon.fetchInitialAssembliesByLocation(locationId);
            console.log('Fetched Assemblies:', subassemblies);

            window.SolutionTaskCommon.populateDropdown(assembliesDropdown, subassemblies, 'Select Subassembly');
            console.log(`Dropdown options after populating Assemblies Dropdown (ID: ${assembliesDropdown.id}):`, assembliesDropdown.options);

            // Validate assembly_id exists in fetched subassemblies
            const isValidAssemblyId = subassemblies.some(assembly => String(assembly.id) === String(positionData.assembly_id));
            if (isValidAssemblyId) {
                assembliesDropdown.value = positionData.assembly_id;
                console.log(`Set Assemblies Dropdown (ID: ${assembliesDropdown.id}) to value: ${positionData.assembly_id}`);
            } else {
                console.warn(`Invalid Assembly ID: ${positionData.assembly_id} for Assemblies Dropdown (ID: ${assembliesDropdown.id})`);
                assembliesDropdown.value = '';
            }
            console.log("Assemblies returned:", subassemblies);
            console.log("subassemblies.length:", subassemblies.length);
            //assembliesDropdown.disabled = subassemblies.length === 0;
            console.log('async function populatePositionFields disabled')
            console.log(`Assemblies Dropdown (ID: ${assembliesDropdown.id}) has been ${subassemblies.length === 0 ? 'disabled' : 'enabled'}.`);


            console.log(`Assemblies Dropdown (ID: ${assembliesDropdown.id}) has been ${subassemblies.length === 0 ? 'disabled' : 'enabled'}.`);
        } catch (error) {
            console.error('Error fetching Assemblies:', error);
            window.SolutionTaskCommon.showAlert('Failed to load subassemblies.', 'danger');
            window.SolutionTaskCommon.populateDropdown(assembliesDropdown, [], 'Select Subassembly');
            //assembliesDropdown.disabled = true;
            console.log('async function populatePositionFields disabled')
            console.log(`Assemblies Dropdown (ID: ${assembliesDropdown.id}) has been disabled due to error.`);
        }
    } else {
        console.warn('Assemblies Dropdown element not found in position section.');
    }

    // === 7. Populate Subassemblies based on selected Subassembly ===
    const subassembliesDropdown = positionSection.querySelector('.subassembliesDropdown');
    if (subassembliesDropdown && positionData.assembly_id) {
        try {
            console.log(`Attempting to fetch Subassemblies for Assembly ID: ${positionData.assembly_id}`);
            const componetassemblies = await SolutionTaskCommon.fetchInitialSubassemblies(positionData.assembly_id);
            console.log('Fetched Subassemblies:', componetassemblies);

            window.SolutionTaskCommon.populateDropdown(subassembliesDropdown, componetassemblies, 'Select Subassembly');
            console.log(`Dropdown options after populating Subassemblies Dropdown (ID: ${subassembliesDropdown.id}):`, subassembliesDropdown.options);

            subassembliesDropdown.value = positionData.subassembly_id || '';
            subassembliesDropdown.disabled = false;
            console.log(`Set Subassemblies Dropdown (ID: ${subassembliesDropdown.id}) to value: ${positionData.subassembly_id}`);
        } catch (error) {
            console.error('Error fetching Subassemblies:', error);
            window.SolutionTaskCommon.showAlert('Failed to load componetassemblies.', 'danger');
            window.SolutionTaskCommon.populateDropdown(subassembliesDropdown, [], 'Select Subassembly');
            subassembliesDropdown.disabled = true;
            console.log(`Subassemblies Dropdown (ID: ${subassembliesDropdown.id}) has been disabled due to error.`);
        }
    } else if (!positionData.assembly_id) {
        console.warn('Subassembly ID is missing in positionData; cannot populate Subassemblies Dropdown.');
    } else {
        console.warn('Subassemblies Dropdown element not found in position section.');
    }

    // === 8. Populate Subassembly Views based on selected Subassembly ===
    const assemblyViewsDropdown = positionSection.querySelector('.assemblyViewsDropdown');
    if (assemblyViewsDropdown && positionData.subassembly_id) {
        try {
            console.log(`Attempting to fetch Assembly Views for Subassembly ID: ${positionData.subassembly_id}`);
            const assemblyViews = await SolutionTaskCommon.fetchInitialAssemblyViews(positionData.subassembly_id);
            console.log('Fetched Subassembly Views:', assemblyViews);

            window.SolutionTaskCommon.populateDropdown(assemblyViewsDropdown, assemblyViews, 'Select Subassembly View');
            console.log(`Dropdown options after populating Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id}):`, assemblyViewsDropdown.options);

            assemblyViewsDropdown.value = positionData.assembly_view_id || '';
            assemblyViewsDropdown.disabled = false;
            console.log(`Set Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id}) to value: ${positionData.assembly_view_id}`);
        } catch (error) {
            console.error('Error fetching Subassembly Views:', error);
            window.SolutionTaskCommon.showAlert('Failed to load assembly views.', 'danger');
            window.SolutionTaskCommon.populateDropdown(assemblyViewsDropdown, [], 'Select Subassembly View');
            assemblyViewsDropdown.disabled = true;
            console.log(`Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id}) has been disabled due to error.`);
        }
    } else if (!positionData.subassembly_id) {
        console.warn('Subassembly ID is missing in positionData; cannot populate Subassembly Views Dropdown.');
    } else {
        console.warn('Subassembly Views Dropdown element not found in position section.');
    }

    // === 9. Populate Site Location Dropdown Independently ===
    const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
    if (siteLocationDropdown) {
        try {
            console.log('Attempting to fetch Site Locations...');
            const siteLocations = await window.SolutionTaskCommon.fetchInitialSiteLocations();
            console.log('Fetched Site Locations:', siteLocations);

            window.SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
            console.log(`Dropdown options after populating Site Location Dropdown (ID: ${siteLocationDropdown.id}):`, siteLocationDropdown.options);

            const siteLocationId = positionData.site_location_id; // Use the correct property
            console.log(`Site Location ID from Position Data: ${siteLocationId}`);

            const matchingSiteLocation = siteLocations.find(location => String(location.id) === String(siteLocationId));
            console.log('Matching Site Location:', matchingSiteLocation);

            if (matchingSiteLocation) {
                siteLocationDropdown.value = matchingSiteLocation.id;
                console.log(`Set Site Location Dropdown (ID: ${siteLocationDropdown.id}) to value: ${matchingSiteLocation.id}`);
            } else {
                console.warn(`Invalid Site Location ID: ${siteLocationId} for Site Location Dropdown (ID: ${siteLocationDropdown.id})`);
                siteLocationDropdown.value = '';
            }

            // Ensure the dropdown remains enabled regardless of data
            siteLocationDropdown.disabled = siteLocations.length === 0;
            console.log(`Site Location Dropdown (ID: ${siteLocationDropdown.id}) has been ${siteLocationDropdown.disabled ? 'disabled' : 'enabled'}.`);
        } catch (error) {
            console.error('Error fetching Site Locations:', error);

            // Show an alert to inform the user about the failure
            window.SolutionTaskCommon.showAlert('Failed to load site locations.', 'danger');

            // Clear existing options and set to default
            window.SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
            console.log(`Dropdown options after clearing Site Location Dropdown (ID: ${siteLocationDropdown.id}):`, siteLocationDropdown.options);

            // Ensure the dropdown remains enabled even if there's an error
            siteLocationDropdown.disabled = siteLocations.length === 0;
            console.log(`Site Location Dropdown (ID: ${siteLocationDropdown.id}) has been ${siteLocationDropdown.disabled ? 'disabled' : 'enabled'}.`);
        }
    } else {
        console.warn('Site Location Dropdown element not found in position section.');
    }

    // === 10. Add Event Listeners for Dynamic Dropdown Dependencies ===
    addPositionEventListeners(positionSection, index);
    console.log(`--- populatePositionFields completed for positionSection index: ${index} ---\n`);
}

    /**
 * Initialize a new position with default dropdown states
 * @param {HTMLElement} positionSection - The position section element
 * @param {number} index - The index of the position
 */
    async function initializeNewPosition(positionSection, index) {
    console.log(`\n--- initializeNewPosition called for positionSection index: ${index} ---`);

    // === 1. Populate Area Dropdown ===
    const areaDropdown = positionSection.querySelector('.areaDropdown');
    if (areaDropdown) {
        try {
            console.log('Attempting to fetch areas...');
            const areas = await window.SolutionTaskCommon.fetchInitialAreas();
            console.log('Fetched areas:', areas);

            window.SolutionTaskCommon.populateDropdown(areaDropdown, areas, 'Select Area');
            console.log(`Dropdown options after populating Area Dropdown (ID: ${areaDropdown.id}):`, Array.from(areaDropdown.options).map(opt => ({ value: opt.value, text: opt.text })));

            areaDropdown.disabled = false;
            console.log(`Area Dropdown (ID: ${areaDropdown.id}) has been enabled.`);
        } catch (error) {
            console.error('Error initializing Area Dropdown:', error);
            window.SolutionTaskCommon.showAlert('Failed to load areas.', 'danger');
            window.SolutionTaskCommon.populateDropdown(areaDropdown, [], 'Select Area');
            areaDropdown.disabled = true;
            console.log(`Area Dropdown (ID: ${areaDropdown.id}) has been disabled due to error.`);
        }
    } else {
        console.warn('Area Dropdown element not found in position section.');
    }

    // === 2. Populate Equipment Group Dropdown ===
    const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
    if (equipmentGroupDropdown) {
        try {
            window.SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, [], 'Select Equipment Group');
            equipmentGroupDropdown.disabled = true;
            console.log(`Initialized and disabled Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id})`);
        } catch (error) {
            console.error('Error initializing Equipment Group Dropdown:', error);
        }
    } else {
        console.warn('Equipment Group Dropdown element not found in position section.');
    }

    // === 3. Populate Model Dropdown ===
    const modelDropdown = positionSection.querySelector('.modelDropdown');
    if (modelDropdown) {
        try {
            window.SolutionTaskCommon.populateDropdown(modelDropdown, [], 'Select Model');
            modelDropdown.disabled = true;
            console.log(`Initialized and disabled Model Dropdown (ID: ${modelDropdown.id})`);
        } catch (error) {
            console.error('Error initializing Model Dropdown:', error);
        }
    } else {
        console.warn('Model Dropdown element not found in position section.');
    }

    // === 4. Populate Asset Number Dropdown ===
    const assetNumberInput = positionSection.querySelector('.assetNumberInput');
    if (assetNumberInput) {
        try {
            window.SolutionTaskCommon.populateDropdown(assetNumberInput, [], 'Select Asset Number');
            assetNumberInput.disabled = true;
            console.log(`Initialized and disabled Asset Number Dropdown (ID: ${assetNumberInput.id})`);
        } catch (error) {
            console.error('Error initializing Asset Number Dropdown:', error);
        }
    } else {
        console.warn('Asset Number Input element not found in position section.');
    }

    // === 5. Populate Location Dropdown ===
    const locationInput = positionSection.querySelector('.locationInput');
    if (locationInput) {
        try {
            window.SolutionTaskCommon.populateDropdown(locationInput, [], 'Select Location');
            locationInput.disabled = true;
            console.log(`Initialized and disabled Location Dropdown (ID: ${locationInput.id})`);
        } catch (error) {
            console.error('Error initializing Location Dropdown:', error);
        }
    } else {
        console.warn('Location Input element not found in position section.');
    }

    // === 6. Populate Assemblies Dropdown ===
    const assembliesDropdown = positionSection.querySelector('.assembliesDropdown');
    if (assembliesDropdown) {
        try {
            window.SolutionTaskCommon.populateDropdown(assembliesDropdown, [], 'Select Subassembly');
            //assembliesDropdown.disabled = true;
            console.log(`Initialized and disabled Assemblies Dropdown (ID: ${assembliesDropdown.id})`);
        } catch (error) {
            console.error('Error initializing Assemblies Dropdown:', error);
        }
    } else {
        console.warn('Assemblies Dropdown element not found in position section.');
    }

    // === 7. Populate Subassemblies Dropdown ===
    const subassembliesDropdown = positionSection.querySelector('.subassembliesDropdown');
    if (subassembliesDropdown) {
        try {
            window.SolutionTaskCommon.populateDropdown(subassembliesDropdown, [], 'Select Subassembly');
            subassembliesDropdown.disabled = true;
            console.log(`Initialized and disabled Subassemblies Dropdown (ID: ${subassembliesDropdown.id})`);
        } catch (error) {
            console.error('Error initializing Subassemblies Dropdown:', error);
        }
    } else {
        console.warn('Subassemblies Dropdown element not found in position section.');
    }

    // === 8. Populate Subassembly Views Dropdown ===
    const assemblyViewsDropdown = positionSection.querySelector('.assemblyViewsDropdown');
    if (assemblyViewsDropdown) {
        try {
            window.SolutionTaskCommon.populateDropdown(assemblyViewsDropdown, [], 'Select Subassembly View');
            assemblyViewsDropdown.disabled = true;
            console.log(`Initialized and disabled Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id})`);
        } catch (error) {
            console.error('Error initializing Subassembly Views Dropdown:', error);
        }
    } else {
        console.warn('Subassembly Views Dropdown element not found in position section.');
    }

    // === 9. Populate Site Location Dropdown Independently ===
    const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');
    if (siteLocationDropdown) {
        try {
            console.log('Attempting to fetch Site Locations...');
            const siteLocations = await window.SolutionTaskCommon.fetchInitialSiteLocations();
            console.log('Fetched Site Locations:', siteLocations);

            window.SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
            console.log(`Dropdown options after populating Site Location Dropdown (ID: ${siteLocationDropdown.id}):`, Array.from(siteLocationDropdown.options).map(opt => ({ value: opt.value, text: opt.text })));

            // Ensure the dropdown remains enabled regardless of data
            siteLocationDropdown.disabled = siteLocations.length === 0;
            console.log(`Site Location Dropdown (ID: ${siteLocationDropdown.id}) is ${siteLocationDropdown.disabled ? 'disabled' : 'enabled'}.`);
        } catch (error) {
            console.error('Error fetching Site Locations:', error);

            // Show an alert to inform the user about the failure
            window.SolutionTaskCommon.showAlert('Failed to load site locations.', 'danger');

            // Clear existing options and set to default
            window.SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
            console.log(`Dropdown options after clearing Site Location Dropdown (ID: ${siteLocationDropdown.id}):`, Array.from(siteLocationDropdown.options).map(opt => ({ value: opt.value, text: opt.text })));

            // Ensure the dropdown remains disabled even if there's an error
            siteLocationDropdown.disabled = true;
            console.log(`Site Location Dropdown (ID: ${siteLocationDropdown.id}) has been disabled due to error.`);
        }
    } else {
        console.warn('Site Location Dropdown element not found in position section.');
    }

    // === 10. Add Event Listeners for Dynamic Dropdown Dependencies ===
    try {
        addPositionEventListeners(positionSection, index);
        console.log(`Added event listeners for positionSection index: ${index}.`);
    } catch (error) {
        console.error(`Error adding event listeners for positionSection index: ${index}:`, error);
    }

    console.log(`--- initializeNewPosition completed for positionSection index: ${index} ---\n`);
}

    /**
 * Add event listeners for dynamic dropdown dependencies within a position section
 * @param {HTMLElement} positionSection - The position section element
 * @param {number} index - The index of the position
 */
    function addPositionEventListeners(positionSection, index) {
    // === Existing Dropdowns ===
    const areaDropdown = positionSection.querySelector('.areaDropdown');
    const equipmentGroupDropdown = positionSection.querySelector('.equipmentGroupDropdown');
    const modelDropdown = positionSection.querySelector('.modelDropdown');
    const assetNumberInput = positionSection.querySelector('.assetNumberInput');
    const locationInput = positionSection.querySelector('.locationInput');
    const siteLocationDropdown = positionSection.querySelector('.siteLocationDropdown');

    // === New Dropdowns ===
    const assembliesDropdown = positionSection.querySelector('.assembliesDropdown');
    const subassembliesDropdown = positionSection.querySelector('.subassembliesDropdown');
    const assemblyViewsDropdown = positionSection.querySelector('.assemblyViewsDropdown');

    /**
     * Helper function to populate the Site Location dropdown
     * Now independent of other dropdowns
     */
    async function populateSiteLocationDropdown() {
        SolutionTaskCommon.showAlert('Loading site locations...', 'info');
        try {
            const siteLocations = await SolutionTaskCommon.fetchInitialSiteLocations();
            SolutionTaskCommon.populateDropdown(siteLocationDropdown, siteLocations, 'Select Site Location');
            siteLocationDropdown.disabled = siteLocations.length === 0;
            console.log(`Populated Site Location Dropdown (ID: ${siteLocationDropdown.id}) with Site Locations.`);
        } catch (error) {
            console.error('Error fetching site locations:', error);
            SolutionTaskCommon.showAlert('Failed to load site locations.', 'danger');
            SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
            siteLocationDropdown.disabled = true;
        }
    }

    // === Event Listeners for Existing Dropdowns ===

    if (areaDropdown) {
        areaDropdown.addEventListener('change', async () => {
            const selectedAreaId = areaDropdown.value;
            console.log(`Area Dropdown (ID: ${areaDropdown.id}) changed to: ${selectedAreaId}`);
            if (selectedAreaId) {
                try {
                    const equipmentGroups = await SolutionTaskCommon.fetchInitialEquipmentGroups(selectedAreaId);
                    SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, equipmentGroups, 'Select Equipment Group');
                    equipmentGroupDropdown.disabled = equipmentGroups.length === 0;
                    console.log(`Populated Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) with Equipment Groups for Area ID ${selectedAreaId}`);

                    // Reset and disable dependent dropdowns except Site Location
                    /**resetDropdowns([
                        modelDropdown,
                        assetNumberInput,
                        locationInput,
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]); */

                } catch (error) {
                    console.error(`Error fetching Equipment Groups for Area ID ${selectedAreaId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load equipment groups.', 'danger');
                    SolutionTaskCommon.populateDropdown(equipmentGroupDropdown, [], 'Select Equipment Group');
                    equipmentGroupDropdown.disabled = true;

                    // Reset and disable dependent dropdowns except Site Location
                    /** resetDropdowns([
                        modelDropdown,
                        assetNumberInput,
                        locationInput,
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]); */
                }
            } else {
                // Reset and disable dependent dropdowns except Site Location
                /** resetDropdowns([
                    equipmentGroupDropdown,
                    modelDropdown,
                    assetNumberInput,
                    locationInput,
                    //assembliesDropdown,
                    subassembliesDropdown,
                    assemblyViewsDropdown
                ]); */
            }
        });
    }

    if (equipmentGroupDropdown) {
        equipmentGroupDropdown.addEventListener('change', async () => {
            const selectedEquipmentGroupId = equipmentGroupDropdown.value;
            console.log(`Equipment Group Dropdown (ID: ${equipmentGroupDropdown.id}) changed to: ${selectedEquipmentGroupId}`);
            if (selectedEquipmentGroupId) {
                try {
                    const models = await SolutionTaskCommon.fetchInitialModels(selectedEquipmentGroupId);
                    SolutionTaskCommon.populateDropdown(modelDropdown, models, 'Select Model');
                    modelDropdown.disabled = models.length === 0;
                    console.log(`Populated Model Dropdown (ID: ${modelDropdown.id}) with Models for Equipment Group ID ${selectedEquipmentGroupId}`);

                    // Reset and disable dependent dropdowns except Site Location
                    /** resetDropdowns([
                        assetNumberInput,
                        locationInput,
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]); */
                } catch (error) {
                    console.error(`Error fetching Models for Equipment Group ID ${selectedEquipmentGroupId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load models.', 'danger');
                    SolutionTaskCommon.populateDropdown(modelDropdown, [], 'Select Model');
                    modelDropdown.disabled = true;

                    // Reset and disable dependent dropdowns except Site Location
                    /** resetDropdowns([
                        assetNumberInput,
                        locationInput,
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]); */
                }
            } else {
                // Reset and disable dependent dropdowns except Site Location
                /** resetDropdowns([
                    modelDropdown,
                    assetNumberInput,
                    locationInput,
                    //assembliesDropdown,
                    subassembliesDropdown,
                    assemblyViewsDropdown
                ]); */
            }
        });
    }

    if (modelDropdown) {
        modelDropdown.addEventListener('change', async () => {
            const selectedModelId = modelDropdown.value;
            console.log(`Model Dropdown (ID: ${modelDropdown.id}) changed to: ${selectedModelId}`);
            if (selectedModelId) {
                SolutionTaskCommon.showAlert('Loading asset numbers and locations...', 'info');
                try {
                    const [assetNumbers, locations] = await Promise.all([
                        SolutionTaskCommon.fetchInitialAssetNumbers(selectedModelId),
                        SolutionTaskCommon.fetchInitialLocations(selectedModelId)
                    ]);
                    SolutionTaskCommon.populateDropdown(assetNumberInput, assetNumbers, 'Select Asset Number');
                    assetNumberInput.disabled = assetNumbers.length === 0;
                    console.log(`Populated Asset Number Dropdown (ID: ${assetNumberInput.id}) with Asset Numbers for Model ID ${selectedModelId}`);

                    SolutionTaskCommon.populateDropdown(locationInput, locations, 'Select Location');
                    locationInput.disabled = locations.length === 0;
                    console.log(`Populated Location Dropdown (ID: ${locationInput.id}) with Locations for Model ID ${selectedModelId}`);

                    // Reset and disable Subassembly-related Dropdowns
                    /** resetDropdowns([
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]);*/
                } catch (error) {
                    console.error(`Error fetching Asset Numbers or Locations for Model ID ${selectedModelId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load asset numbers or locations.', 'danger');
                    /** resetDropdowns([
                        assetNumberInput,
                        locationInput,
                        //assembliesDropdown,
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]);*/
                }
            } else {
                // Reset and disable dependent dropdowns except Site Location
                /** resetDropdowns([
                    assetNumberInput,
                    locationInput,
                    //assembliesDropdown,
                    subassembliesDropdown,
                    assemblyViewsDropdown
                ]); */
            }
        });
    }

    if (assetNumberInput) {
        assetNumberInput.addEventListener('change', () => {
            const selectedAssetNumber = assetNumberInput.value;
            console.log(`Asset Number Dropdown (ID: ${assetNumberInput.id}) changed to: ${selectedAssetNumber}`);
            // Since Site Location is independent, no action needed here
            // Optionally, you can clear Site Location if required
            // SolutionTaskCommon.populateDropdown(siteLocationDropdown, [], 'Select Site Location');
            // siteLocationDropdown.disabled = true;
        });
    }

    if (locationInput) {
        locationInput.addEventListener('change', async () => {
            const selectedLocationId = locationInput.value;
            console.log(`Location Input (ID: ${locationInput.id}) changed to: ${selectedLocationId}`);
            if (selectedLocationId) {
                try {
                    const subassemblies = await SolutionTaskCommon.fetchInitialAssembliesByLocation(selectedLocationId);
                    SolutionTaskCommon.populateDropdown(assembliesDropdown, subassemblies, 'Select Subassembly');
                    console.log(`assembliesDropdown subassemblies.length = ${subassemblies.length}`);
                    //assembliesDropdown.disabled = subassemblies.length === 0;

                    console.log(`Populated Assemblies Dropdown (ID: ${assembliesDropdown.id}) with Assemblies for Location ID ${selectedLocationId}`);

                    // Reset and disable Subassemblies and Subassembly Views Dropdowns
                    /** resetDropdowns([
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]);*/
                } catch (error) {
                    console.error(`Error fetching Assemblies for Location ID ${selectedLocationId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load subassemblies.', 'danger');
                    SolutionTaskCommon.populateDropdown(assembliesDropdown, [], 'Select Subassembly');
                    //assembliesDropdown.disabled = true;

                    // Reset and disable Subassemblies and Subassembly Views Dropdowns
                    /** resetDropdowns([
                        subassembliesDropdown,
                        assemblyViewsDropdown
                    ]); */
                }
            } else {
                // Reset and disable Assemblies, Subassemblies, and Subassembly Views Dropdowns
                /** resetDropdowns([
                    //assembliesDropdown,
                    subassembliesDropdown,
                    assemblyViewsDropdown
                ]);*/
            }
        });
    }

    // === Event Listeners for New Dropdowns ===

    if (assembliesDropdown) {
        assembliesDropdown.addEventListener('change', async () => {
            const selectedAssemblyId = assembliesDropdown.value;
            console.log(`Assemblies Dropdown (ID: ${assembliesDropdown.id}) changed to: ${selectedAssemblyId}`);
            if (selectedAssemblyId) {
                try {
                    const componetassemblies = await SolutionTaskCommon.fetchInitialSubassemblies(selectedAssemblyId);
                    SolutionTaskCommon.populateDropdown(subassembliesDropdown, componetassemblies, 'Select Subassembly');
                    subassembliesDropdown.disabled = componetassemblies.length === 0;
                    console.log(`Populated Subassemblies Dropdown (ID: ${subassembliesDropdown.id}) with Subassemblies for Assembly ID ${selectedAssemblyId}`);

                    // Reset and disable Subassembly Views Dropdown
                    /** resetDropdowns([assemblyViewsDropdown]);*/
                } catch (error) {
                    console.error(`Error fetching Subassemblies for Assembly ID ${selectedAssemblyId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load componetassemblies.', 'danger');
                    SolutionTaskCommon.populateDropdown(subassembliesDropdown, [], 'Select Subassembly');
                    subassembliesDropdown.disabled = true;

                    // Reset and disable Subassembly Views Dropdown
                    /** resetDropdowns([assemblyViewsDropdown]); */
                }
            } else {
                // Reset and disable Subassemblies and Subassembly Views Dropdowns
                /** resetDropdowns([subassembliesDropdown, assemblyViewsDropdown]); */
            }
        });
    }

    if (subassembliesDropdown) {
        subassembliesDropdown.addEventListener('change', async () => {
            const selectedSubassemblyId = subassembliesDropdown.value;
            console.log(`Subassemblies Dropdown (ID: ${subassembliesDropdown.id}) changed to: ${selectedSubassemblyId}`);
            if (selectedSubassemblyId) {
                try {
                    const assemblyViews = await SolutionTaskCommon.fetchInitialAssemblyViews(selectedSubassemblyId);
                    SolutionTaskCommon.populateDropdown(assemblyViewsDropdown, assemblyViews, 'Select Subassembly View');
                    assemblyViewsDropdown.disabled = assemblyViews.length === 0;
                    console.log(`Populated Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id}) with Assembly Views for Subassembly ID ${selectedSubassemblyId}`);
                } catch (error) {
                    console.error(`Error fetching Assembly Views for Subassembly ID ${selectedSubassemblyId}:`, error);
                    SolutionTaskCommon.showAlert('Failed to load assembly views.', 'danger');
                    SolutionTaskCommon.populateDropdown(assemblyViewsDropdown, [], 'Select Subassembly View');
                    assemblyViewsDropdown.disabled = true;
                }
            } else {
                // Reset and disable Subassembly Views Dropdown
                /** resetDropdowns([assemblyViewsDropdown]); */
            }
        });
    }

    if (assemblyViewsDropdown) {
        assemblyViewsDropdown.addEventListener('change', () => {
            const selectedAssemblyViewId = assemblyViewsDropdown.value;
            console.log(`Assembly Views Dropdown (ID: ${assemblyViewsDropdown.id}) changed to: ${selectedAssemblyViewId}`);
            // Add any additional logic if needed when assembly view changes
            // For example, fetching more details based on the selected assembly view
        });
    }

    // === Remove Position Button ===
    const removeBtn = positionSection.querySelector('.removePositionBtn');
    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            positionSection.remove();
            console.log(`Removed position section with index ${index}`);
        });
    }
}

    /**
 * Reset and disable multiple dropdowns
 * @param {Array} elements - Array of dropdown/select elements to reset
    */
    function resetDropdowns(elements) {
    elements.forEach(element => {
        if (!element) {
            console.warn('Attempted to reset a dropdown that does not exist.');
            return;
        }

        // 1. Skip if we want to exclude assembliesDropdown
        if (element.classList.contains('assembliesDropdown')) {
            console.log(`Skipping reset for assembliesDropdown (ID: ${element.id}).`);
            return;
        }

        // 2. Otherwise, proceed as before
        const elementClass = Array.from(element.classList)
            .find(cls => cls.endsWith('Dropdown') || cls.endsWith('Input'));

        if (!elementClass) {
            console.warn(`No Dropdown/Input class found for element with ID: ${element.id}`);
            return;
        }

        const baseName = elementClass.replace('Dropdown', '').replace('Input', '');
        const placeholderText = `Select ${SolutionTaskCommon.capitalizeFirstLetter(baseName)}`;

        // This is the normal "clear" step
        SolutionTaskCommon.populateDropdown(element, [], placeholderText);
        element.disabled = true;
        console.log(`Reset and disabled ${elementClass} (ID: ${element.id})`);
    });
}


    // === 5. Event Listeners ===

    /**
     * Initialize event listeners for the form
     */
    function initializeEventListeners() {
        const addSolutionBtn = document.getElementById('addSolutionBtn');
        if (addSolutionBtn) {
            addSolutionBtn.addEventListener('click', () => {
                const nameInput = document.getElementById('new_solution_name');
                const descInput = document.getElementById('new_solution_description');
                const name = nameInput.value.trim();
                const description = descInput?.value.trim();

                // Retrieve window.AppState.currentProblemId from sessionStorage
                const problemId = parseInt(sessionStorage.getItem('window.AppState.currentProblemId'), 10);
                console.log('Add Solution Clicked. name:', name, 'window.AppState.currentProblemId:', problemId);

                if (name && problemId) {
                    addNewSolution(problemId, name, description);
                } else {
                    SolutionTaskCommon.showAlert('Solution name cannot be empty or no problem selected.', 'warning');
                }
            });
        }

        // Updated Solutions Dropdown Click and Double-Click
        const existingSolutions = document.getElementById('existing_solutions');
        if (existingSolutions) {
            // Single-click selection using 'click' event
            existingSolutions.addEventListener('click', (event) => {
                const solutionId = parseInt(event.target.value, 10); // Ensure solutionId is an integer
                console.log(`Solutions Dropdown clicked on: ${solutionId}`);

                if (solutionId) {
                    // Update the globally accessible window.AppState.currentSolutionId in window.AppState
                    window.AppState.currentSolutionId = solutionId;
                    console.log(`Updated Current Solution ID: ${window.AppState.currentSolutionId}`);

                    // Highlight the selected solution (optional visual feedback)
                    highlightSelectedOption(existingSolutions, solutionId);
                    // Note: No additional logic like fetching tasks is triggered here
                } else {
                    // Reset globally accessible window.AppState.currentSolutionId in window.AppState
                    window.AppState.currentSolutionId = null;
                    console.log(`Cleared Current Solution ID`);

                    // Optionally, clear related UI elements
                    const tasksDropdown = document.getElementById('existing_tasks');
                    if (tasksDropdown) {
                        SolutionTaskCommon.populateDropdown(tasksDropdown, [], 'Select Task');
                        tasksDropdown.disabled = true;
                        console.log(`Cleared Tasks Dropdown`);
                    }
                    SolutionTaskCommon.clearEditTaskForm();
                }
            });

            // Double-click activation using 'dblclick' event
            existingSolutions.addEventListener('dblclick', (event) => {
                const selectedOption = existingSolutions.options[existingSolutions.selectedIndex];
                const solutionId = selectedOption ? selectedOption.value : null; // Get the value of the selected option

                if (solutionId) {
                    openSolutionDetails(solutionId); // Function to activate/edit the solution
                }
            });
        } else {
            console.warn("Solutions dropdown with ID 'existing_solutions' not found.");
        }

        // Add event listener for Remove Solution button
        const removeSolutionsBtn = document.getElementById('removeSolutionsBtn');
        if (removeSolutionsBtn) {
            removeSolutionsBtn.addEventListener('click', () => {
                const solutionsDropdown = document.getElementById('existing_solutions');
                const selectedSolutionId = solutionsDropdown.value;

                // Check if a solution is selected
                if (!selectedSolutionId) {
                    SolutionTaskCommon.showAlert('Please select a solution to remove.', 'warning');
                    return;
                }

                // Confirm removal
                const confirmDelete = confirm('Are you sure you want to remove the selected solution?');
                if (!confirmDelete) return;

                // **Add console log to check window.AppState.currentProblemId**
                console.log('Remove Solution Clicked - window.AppState.currentProblemId:', window.AppState.currentProblemId);

                // Retrieve window.AppState.currentProblemId from AppState
                const problemId = window.AppState.currentProblemId;
                if (!problemId) {
                    SolutionTaskCommon.showAlert('No problem selected.', 'warning');
                    return;
                }

                // Call the removeSolutions function
                removeSolutions(problemId, [selectedSolutionId]);
            });
        } else {
            console.warn("Remove Solution button with ID 'removeSolutionsBtn' not found.");
        }

        /**
         * Function to highlight the selected option in the dropdown
         * @param {HTMLElement} dropdown - The select element
         * @param {number} selectedId - The ID of the selected solution
         */
        function highlightSelectedOption(dropdown, selectedId) {
            // Remove 'active' class from all options
            for (let i = 0; i < dropdown.options.length; i++) {
                dropdown.options[i].classList.remove('active');
            }

            // Add 'active' class to the selected option
            const selectedOption = dropdown.querySelector(`option[value="${selectedId}"]`);
            if (selectedOption) {
                selectedOption.classList.add('active');
                console.log(`Highlighted Solution ID: ${selectedId}`);
            }
        }

        /**
         * Function to open solution details, fetch tasks, and activate the task tab
         * @param {number|string} solutionId - The ID of the selected solution
         */
        function openSolutionDetails(solutionId) {
            console.log(`Opening solution details for Solution ID: ${solutionId}`);

            // Update the globally accessible window.AppState.currentSolutionId
            window.AppState.currentSolutionId = parseInt(solutionId, 10);
            console.log(`Updated Current Solution ID in AppState: ${window.AppState.currentSolutionId}`);

            // Fetch tasks for the selected solution
            fetchTasksForSolution(solutionId);

            // Activate the task tab
            activateTab('task-tab');
        }

        // Add Task Button
        const addTaskBtn = document.getElementById('addTaskBtn');
        if (addTaskBtn) {
            addTaskBtn.addEventListener('click', () => {
                const nameInput = document.getElementById('new_task_name');
                const descInput = document.getElementById('new_task_description');
                const name = nameInput.value.trim();
                const description = descInput?.value.trim();

                // Use window.AppState.currentSolutionId for consistency
                const solutionId = window.AppState.currentSolutionId;
                console.log('Task Name:', name, 'Current Solution ID:', solutionId);

                if (name && solutionId) {
                    addNewTask(solutionId, name, description);
                } else {
                    SolutionTaskCommon.showAlert('Task name cannot be empty or no solution selected.', 'warning');
                }
            });
        }

        // Add Position Button
        const addPositionBtn = document.getElementById('addPositionBtn');
        if (addPositionBtn) {
            addPositionBtn.addEventListener('click', () => {
                const positionsContainer = document.getElementById('pst_task_edit_positions_container');
                const currentIndex = positionsContainer ? positionsContainer.children.length : 0;
                console.log(`Adding new position with index: ${currentIndex}`);
                addPosition(null, currentIndex);
            });
        }

        // Event listener for Remove Task button
        document.getElementById('removeTaskBtn').addEventListener('click', () => {
            const tasksDropdown = document.getElementById('existing_tasks');
            window.AppState.currentTaskId = tasksDropdown.value;

            // Check if a task is selected
            if (!window.AppState.currentTaskId) {
                SolutionTaskCommon.showAlert('Please select a task to remove.', 'warning');
                return;
            }

            // Confirm removal
            const confirmDelete = confirm('Are you sure you want to remove the selected task?');
            if (!confirmDelete) return;

            // Call the remove task function
            removeTask(window.AppState.currentTaskId);
        });

        // Function to remove the task
        function removeTask(taskId) {
            const solutionId = window.AppState.currentSolutionId; // Retrieve the current solution ID
            if (!solutionId) {
                SolutionTaskCommon.showAlert('No solution selected.', 'warning');
                return;
            }

            console.log('removeTask called with taskId:', taskId, 'and solutionId:', solutionId);

            fetch('/pst_troubleshooting_solution/remove_task/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ task_id: taskId, solution_id: solutionId })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errData => {
                        console.error('Server error:', errData);
                        SolutionTaskCommon.showAlert(errData.error || 'Failed to remove task', 'danger');
                        throw new Error(errData.error || 'Failed to remove task');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    SolutionTaskCommon.showAlert(data.message, 'success');
                    fetchTasksForSolution(solutionId); // Refresh tasks for the solution
                } else {
                    SolutionTaskCommon.showAlert(data.error || 'Failed to remove task', 'danger');
                }
            })
            .catch(error => console.error('Error removing task:', error));
        }
    }

    // === 7. Initialize Event Listeners ===
    initializeEventListeners();

    // === 8. Optional: Fetch and Display Initial Data ===
    // You can call fetchSolutions with a default problem ID or based on user selection
    // Example:
    // fetchSolutions(initialProblemId);
});