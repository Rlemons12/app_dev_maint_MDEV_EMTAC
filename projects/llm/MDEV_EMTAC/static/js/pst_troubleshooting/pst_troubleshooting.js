// Consolidated Script: pst_troubleshooting.js

document.addEventListener('DOMContentLoaded', () => {
    // Explicitly set each endpoint URL
    const GET_EQUIPMENT_GROUPS_URL = '/pst_troubleshooting/get_equipment_groups';
    const GET_MODELS_URL = '/pst_troubleshooting/get_models';
    const GET_ASSET_NUMBERS_URL = '/pst_troubleshooting/get_asset_numbers';
    const GET_LOCATIONS_URL = '/pst_troubleshooting/get_locations';
    const GET_SITE_LOCATIONS_URL = '/pst_troubleshooting/get_site_locations';
    const SEARCH_PROBLEMS_URL = '/pst_troubleshooting_position_update/search_problems';
    const GET_PROBLEM_DETAILS_URL = '/pst_troubleshooting_position_update/get_problem_details/';
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';
    const DELETE_PROBLEM_URL = '/pst_troubleshooting_solution/delete_problem';

    // Dropdown elements
    const dropdowns = {
        area: document.getElementById('pst_areaDropdown'),
        equipmentGroup: document.getElementById('pst_equipmentGroupDropdown'),
        model: document.getElementById('pst_modelDropdown'),
        assetNumber: document.getElementById('pst_assetNumberDropdown'),
        location: document.getElementById('pst_locationDropdown'),
        siteLocation: document.getElementById('pst_siteLocationDropdown')
    };

    // Initialize AppState if it doesn't already exist
    window.AppState = window.AppState || {};

    // Ensure properties exist without overwriting existing values
    if (typeof window.AppState.currentProblemId === 'undefined') {
        window.AppState.currentProblemId = null;
    }
    if (typeof window.AppState.currentSolutionId === 'undefined') {
        window.AppState.currentSolutionId = null;
    }
    if (typeof window.AppState.selectedTaskId === 'undefined') {
        window.AppState.selectedTaskId = null;
    }

console.log('Initialized AppState in pst_troubleshooting.js:', window.AppState);


    console.log("JavaScript loaded and ready");

    // Fetch and populate Site Locations when the page loads
    fetchData(GET_SITE_LOCATIONS_URL)
        .then(data => populateDropdown(dropdowns.siteLocation, data, 'Select Site Location', item => `${item.title} - Room ${item.room_number}`))
        .catch(error => console.error('Error fetching site locations:', error));

    // Event listeners for dropdowns
    dropdowns.area.addEventListener('change', () => {
        const areaId = dropdowns.area.value;
        if (areaId) {
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${areaId}`)
                .then(data => populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group', item => item.name))
                .catch(error => console.error('Error fetching equipment groups:', error));

            // Reset dependent dropdowns
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        } else {
            resetAllDropdowns();
        }
    });

    dropdowns.equipmentGroup.addEventListener('change', () => {
        const equipmentGroupId = dropdowns.equipmentGroup.value;
        if (equipmentGroupId) {
            fetchData(`${GET_MODELS_URL}?equipment_group_id=${equipmentGroupId}`)
                .then(data => populateDropdown(dropdowns.model, data, 'Select Model', item => item.name))
                .catch(error => console.error('Error fetching models:', error));

            // Reset dependent dropdowns
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        }
    });

    dropdowns.model.addEventListener('change', () => {
        const modelId = dropdowns.model.value;
        if (modelId) {
            fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${modelId}`)
                .then(data => populateDropdown(dropdowns.assetNumber, data, 'Select Asset Number', item => item.number))
                .catch(error => console.error('Error fetching asset numbers:', error));

            fetchData(`${GET_LOCATIONS_URL}?model_id=${modelId}`)
                .then(data => populateDropdown(dropdowns.location, data, 'Select Location', item => item.name))
                .catch(error => console.error('Error fetching locations:', error));

            // Do not touch the siteLocation dropdown here
        } else {
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
        }
    });

    // Search button event listener
document.getElementById('searchProblemByPositionBtn').addEventListener('click', () => {
    const assetNumberDropdown = dropdowns.assetNumber;
    const locationDropdown = dropdowns.location;

    // Get asset number text if available
    let assetNumberText = '';
    if (assetNumberDropdown.value) {
        const selectedOption = assetNumberDropdown.options[assetNumberDropdown.selectedIndex];
        assetNumberText = selectedOption.text.trim();
    }

    // Get location text if available
    let locationText = '';
    if (locationDropdown.value) {
        const selectedOption = locationDropdown.options[locationDropdown.selectedIndex];
        locationText = selectedOption.text.trim();
    }

    const searchParams = {
        area_id: dropdowns.area.value,
        equipment_group_id: dropdowns.equipmentGroup.value,
        model_id: dropdowns.model.value,
        asset_number: assetNumberText,  // Use text, not ID
        location: locationText,         // Use text, not ID
        site_location_id: dropdowns.siteLocation.value
    };

    if (Object.values(searchParams).some(value => value)) {
        fetch(`${SEARCH_PROBLEMS_URL}?${new URLSearchParams(searchParams)}`)
            .then(response => response.json())
            .then(data => displaySearchResults(data))
            .catch(error => console.error('Error searching for problems:', error));
    } else {
        alert('Please enter at least one search criterion.');
    }
});

    /**
     * Display search results with Update and Edit Solutions buttons
     * @param {Array} problems - Array of problem objects
     */
function displaySearchResults(problems) {
    const resultsList = document.getElementById('pst_positionResultsList');
    resultsList.innerHTML = '';
    document.getElementById('pst_searchResults').style.display = problems.length ? 'block' : 'none';

    if (problems.length > 0) {
        window.AppState.currentProblemId = problems[0].id;
        sessionStorage.setItem('window.AppState.currentProblemId', window.AppState.currentProblemId);  // Store in sessionStorage
        console.log('Updated Current Problem ID:', window.AppState.currentProblemId);
    } else {
        window.AppState.currentProblemId = null;
        sessionStorage.removeItem('window.AppState.currentProblemId');  // Remove from sessionStorage if null
        console.log('No problems found. Resetting Current Problem ID:', window.AppState.currentProblemId);
    }

    problems.forEach(problem => {
        const listItem = document.createElement('li');
        listItem.classList.add('list-group-item');
        listItem.innerHTML = `
            <strong>${problem.name}</strong> - ${problem.description}
            <button class="btn btn-sm btn-warning float-end ms-2 update-problem-btn" data-problem-id="${problem.id}">Update Problem Position</button>
            <button class="btn btn-sm btn-info float-end ms-2 edit-solutions-btn" data-problem-id="${problem.id}">Edit Related Solutions</button>
            <button class="btn btn-sm btn-danger float-end ms-2 delete-problem-btn" data-problem-id="${problem.id}">Delete Problem</button>
            <br>
            <!--<small>Area: ${problem.area}</small><br>
            <small>Equipment Group: ${problem.equipment_group}</small><br>
            <small>Model: ${problem.model}</small><br>
            <small>Asset Number: ${problem.asset_number}</small><br>
            <small>Location: ${problem.location}</small><br>
            <small>Site Location: ${problem.site_location}</small>-->
        `;
        resultsList.appendChild(listItem);
    });

    resultsList.addEventListener('click', (event) => {
        if (event.target.classList.contains('update-problem-btn')) {
            const problemId = event.target.dataset.problemId;
            console.log("Update Problem Position button clicked for problem ID:", problemId);
            fetchProblemDetails(problemId);
        }
        if (event.target.classList.contains('edit-solutions-btn')) {
            const problemId = event.target.dataset.problemId;
            console.log("Edit Related Solutions button clicked for problem ID:", problemId);
            window.AppState.currentProblemId = problemId;
            sessionStorage.setItem('window.AppState.currentProblemId', window.AppState.currentProblemId);  // Update sessionStorage here too
            editRelatedSolutions(problemId);
        }
        if (event.target.classList.contains('delete-problem-btn')) {
            const problemId = event.target.dataset.problemId;
            console.log("Delete Problem button clicked for problem ID:", problemId);
            confirmAndDeleteProblem(problemId);
        }
    });
}

/**
 * Confirm with the user and delete the problem if confirmed
 * @param {number|string} problemId - The ID of the problem to delete
 */
function confirmAndDeleteProblem(problemId) {
    console.log(`confirmAndDeleteProblem called with problemId: ${problemId}`);
    const confirmation = confirm('Are you sure you want to delete this problem? This action cannot be undone.');
    if (confirmation) {
        deleteProblem(problemId);
    }
}
/**
 * Delete a problem by making an API call to the backend
 * @param {number|string} problemId - The ID of the problem to delete
 * @returns {Promise<boolean>} - Returns true if deletion was successful, else false
 */
async function deleteProblem(problemId) {
    try {
        console.log(`Initiating deletion for Problem ID: ${problemId}`);

        // Show a loading message or spinner to inform the user
        SolutionTaskCommon.showAlert('Deleting problem...', 'info');

        // Prepare the payload, ensuring problem_id is a number
        const payload = { problem_id: Number(problemId) };
        console.log("Sending payload:", payload);

        // Define the API endpoint without trailing slash
        const DELETE_PROBLEM_URL = '/pst_troubleshooting_solution/delete_problem';

        // Make the API call to delete the problem
        const response = await fetch(DELETE_PROBLEM_URL, {
            method: 'POST', // Ensure this matches the backend expectation
            headers: {
                'Content-Type': 'application/json' // Critical for JSON parsing
            },
            body: JSON.stringify(payload) // Correctly stringify the JSON
        });

        console.log(`Received response status: ${response.status}`);

        // Check if response is JSON
        const contentType = response.headers.get('Content-Type');
        let data;
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
            console.log("Received response data:", data);
        } else {
            data = { status: 'error', error: 'Unexpected response format.' };
            console.warn("Unexpected Content-Type. Response is not JSON.");
        }

        if (response.ok && data.status === 'success') {
            // Show success alert
            SolutionTaskCommon.showAlert('Problem deleted successfully.', 'success');
            console.log(`Problem ID ${problemId} deleted successfully.`);

            // Update the UI by removing the deleted problem
            removeProblemFromUI(problemId);

            return true;
        } else {
            // Handle server-side errors
            SolutionTaskCommon.showAlert(data.error || 'Failed to delete the problem.', 'danger');
            console.error('Failed to delete problem:', data.error || data.message);
            return false;
        }
    } catch (error) {
        // Handle network or unexpected errors
        SolutionTaskCommon.showAlert('An error occurred while deleting the problem.', 'danger');
        console.error('Error deleting problem:', error);
        return false;
    }
}

    // Fetch and display problem details in the update form
    function fetchProblemDetails(problemId) {
        fetchData(`${GET_PROBLEM_DETAILS_URL}${problemId}`)
            .then(data => populateUpdateForm(data))
            .catch(error => console.error('Error fetching problem details:', error));
    }

    function populateUpdateForm(data) {
        if (!data || !data.problem || !data.position) {
            console.error('Invalid data received for problem details:', data);
            alert('Error loading problem details.');
            return;
        }

        document.getElementById('update_problem_id').value = data.problem.id || '';
        document.getElementById('update_problem_name').value = data.problem.name || '';
        document.getElementById('update_problem_description').value = data.problem.description || '';

        const updateDropdowns = {
    area: document.getElementById('update_pst_areaDropdown'),
    equipmentGroup: document.getElementById('update_pst_equipmentGroupDropdown'),
    model: document.getElementById('update_pst_modelDropdown'),
    assetNumber: document.getElementById('update_pst_assetNumberDropdown'), // Changed from assetNumberInput
    location: document.getElementById('update_pst_locationDropdown'),       // Changed from locationInput
    siteLocation: document.getElementById('update_pst_siteLocationDropdown')
};

        if (updateDropdowns.area && data.position.area_id) {
            updateDropdowns.area.value = data.position.area_id;
            updateDropdowns.area.disabled = false;

            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(data.position.area_id)}`)
                .then(equipmentGroups => {
                    populateDropdown(updateDropdowns.equipmentGroup, equipmentGroups, 'Select Equipment Group', item => item.name);
                    updateDropdowns.equipmentGroup.value = data.position.equipment_group_id || '';

                    return fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(data.position.equipment_group_id)}`);
                })
                .then(models => {
    // Only proceed if model dropdown exists
    if (updateDropdowns.model) {
        populateDropdown(updateDropdowns.model, models, 'Select Model', item => item.name);
        updateDropdowns.model.value = data.position.model_id || '';
        updateDropdowns.model.disabled = false;
    } else {
        console.warn('Model dropdown not found in the DOM');
    }

    // Only enable equipment group if it exists
    if (updateDropdowns.equipmentGroup) {
        updateDropdowns.equipmentGroup.disabled = false;
    }

    // Now that we have model ID, fetch asset numbers for this model
    if (data.position.model_id) {
        console.log(`Fetching asset numbers for model ID: ${data.position.model_id}`);
        return fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${encodeURIComponent(data.position.model_id)}`);
    } else {
        console.warn('No model ID available to fetch asset numbers');
        return Promise.resolve([]);  // Return empty array to continue chain
    }
})
.then(assetNumbers => {
    // Populate asset number dropdown if it exists
    if (updateDropdowns.assetNumber) {
        console.log(`Populating asset number dropdown with ${assetNumbers.length} options`);
        populateDropdown(updateDropdowns.assetNumber, assetNumbers, 'Select Asset Number', item => item.number);

        // If we have asset_number_id in the data, select it
        if (data.position.asset_number_id) {
            updateDropdowns.assetNumber.value = data.position.asset_number_id;
        } else if (data.position.asset_number) {
            // Try to find asset by number as fallback
            const matchingAsset = assetNumbers.find(a => a.number === data.position.asset_number);
            if (matchingAsset) {
                updateDropdowns.assetNumber.value = matchingAsset.id;
            }
        }
        updateDropdowns.assetNumber.disabled = false;
    } else {
        console.warn('Asset Number dropdown not found in the DOM');
    }

    // Now fetch locations for this model
    if (data.position.model_id) {
        console.log(`Fetching locations for model ID: ${data.position.model_id}`);
        return fetchData(`${GET_LOCATIONS_URL}?model_id=${encodeURIComponent(data.position.model_id)}`);
    } else {
        console.warn('No model ID available to fetch locations');
        return Promise.resolve([]);  // Return empty array to continue chain
    }
})
.then(locations => {
    // Populate location dropdown if it exists
    if (updateDropdowns.location) {
        console.log(`Populating location dropdown with ${locations.length} options`);
        populateDropdown(updateDropdowns.location, locations, 'Select Location', item => item.name);

        // If we have location_id in the data, select it
        if (data.position.location_id) {
            updateDropdowns.location.value = data.position.location_id;
        } else if (data.position.location) {
            // Try to find location by name as fallback
            const matchingLocation = locations.find(l => l.name === data.position.location);
            if (matchingLocation) {
                updateDropdowns.location.value = matchingLocation.id;
            }
        }
        updateDropdowns.location.disabled = false;
    } else {
        console.warn('Location dropdown not found in the DOM');
    }

    // Now we can fetch site locations with all params
    const siteLocationParams = new URLSearchParams({
        model_id: data.position.model_id || '',
        area_id: data.position.area_id || '',
        equipment_group_id: data.position.equipment_group_id || ''
        // We could add asset_number_id and location_id here too if they're needed for filtering
    }).toString();

    if (updateDropdowns.siteLocation) {
        console.log(`Fetching site locations with params: ${siteLocationParams}`);
        return fetchData(`${GET_SITE_LOCATIONS_URL}?${siteLocationParams}`);
    } else {
        console.warn('Site Location dropdown not found');
        return Promise.resolve([]);
    }
})
.then(siteLocations => {
    // Populate site location dropdown if it exists
    if (updateDropdowns.siteLocation && Array.isArray(siteLocations)) {
        console.log(`Populating site location dropdown with ${siteLocations.length} options`);
        populateDropdown(updateDropdowns.siteLocation, siteLocations, 'Select Site Location',
            item => `${item.title} - Room ${item.room_number}`);

        if (data.position.site_location_id) {
            updateDropdowns.siteLocation.value = data.position.site_location_id;
        }
        updateDropdowns.siteLocation.disabled = false;
    } else {
        console.warn('Site Location dropdown not found or invalid data format');
    }
})
                .catch(error => console.error('Error fetching equipment groups or models:', error));
        } else {
            console.error('Area ID is missing in position data.');
            alert('Area information is missing in problem details.');
        }

        const updateSection = document.getElementById('pst_updateProblemSection');
        if (updateSection) {
            updateSection.style.display = 'block';
        }
    }

    function editRelatedSolutions(problemId) {
        window.AppState.currentProblemId = problemId;  // Set the current problem ID here
        console.log('Current Problem ID updated to:', window.AppState.currentProblemId);
        fetchData(`${GET_SOLUTIONS_URL}${problemId}`)
            .then(data => {
                if (!data || !Array.isArray(data) || data.length === 0) {
                    console.log(`No related solutions found for problem ID: ${problemId}`);
                }
                populateSolutionsTab(data);
                document.getElementById('solution-tab').click();
            })
            .catch(error => console.error('Error fetching related solutions:', error));
    }


    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        solutionsDropdown.innerHTML = '';

        // Check if solutions is an array, otherwise assign an empty array
        if (!Array.isArray(solutions)) {
            solutions = [];
        }

        // Populate the dropdown with existing solutions
        solutions.forEach(solution => {
            solutionsDropdown.innerHTML += `<option value="${solution.id}">${solution.name} - ${solution.description}</option>`;
        });

        // Optionally, disable the dropdown if there are no solutions
        if (solutions.length === 0) {
            solutionsDropdown.disabled = true;
        } else {
            solutionsDropdown.disabled = false;
        }
    }


    // Utility functions
    function populateDropdown(dropdown, data, placeholder, displayTextFunc) {
        dropdown.innerHTML = `<option value="">${placeholder}</option>`;
        data.forEach(item => {
            const displayText = displayTextFunc(item);
            dropdown.innerHTML += `<option value="${item.id}">${displayText}</option>`;
        });
        dropdown.disabled = false;
    }

    function resetDropdown(dropdown, placeholder) {
        if (dropdown !== dropdowns.siteLocation) { // Do not reset Site Location dropdown
            dropdown.innerHTML = `<option value="">${placeholder}</option>`;
            dropdown.disabled = true;
        }
    }

    function resetAllDropdowns() {
        resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
        resetDropdown(dropdowns.model, 'Select Model');
        resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
        resetDropdown(dropdowns.location, 'Select Location');
        // Site Location dropdown is independent; do not reset here
    }
});

function fetchData(url) {
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                // Check if response is JSON
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                } else {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
            }
            return response.json();
        });
}

