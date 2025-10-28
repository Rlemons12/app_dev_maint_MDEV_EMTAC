// static/js/pst_troubleshooting_position_update.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Explicitly set each endpoint URL
    const GET_EQUIPMENT_GROUPS_URL = '/pst_troubleshooting/get_equipment_groups';
    const GET_MODELS_URL = '/pst_troubleshooting/get_models';
    const GET_ASSET_NUMBERS_URL = '/pst_troubleshooting/get_asset_numbers';
    const GET_LOCATIONS_URL = '/pst_troubleshooting/get_locations';
    const GET_SITE_LOCATIONS_URL = '/pst_troubleshooting/get_site_locations';
    const SEARCH_PROBLEMS_URL = '/pst_troubleshooting_position_update/search_problems';
    const GET_PROBLEM_DETAILS_URL = '/pst_troubleshooting_position_update/get_problem/';
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';

    // Dropdown elements
    const dropdowns = {
        area: document.getElementById('pst_areaDropdown'),
        equipmentGroup: document.getElementById('pst_equipmentGroupDropdown'),
        model: document.getElementById('pst_modelDropdown'),
        assetNumber: document.getElementById('pst_assetNumberDropdown'), // Changed from assetNumberInput
        location: document.getElementById('pst_locationDropdown'), // Changed from locationInput
        siteLocation: document.getElementById('pst_siteLocationDropdown')
    };

    console.log("JavaScript loaded and ready");

    // Initialize Select2 for Site Location Dropdown
    if (dropdowns.siteLocation) {
        $('#pst_siteLocationDropdown').select2({
            placeholder: 'Select Site Location or type "New..."',
            allowClear: true
        });
    }

    // Event listeners for dropdowns
    if (dropdowns.area) {
        dropdowns.area.addEventListener('change', handleAreaChange);
    }

    if (dropdowns.equipmentGroup) {
        dropdowns.equipmentGroup.addEventListener('change', handleEquipmentGroupChange);
    }

    if (dropdowns.model) {
        dropdowns.model.addEventListener('change', handleModelChange);
    }

    // Search button event listener
    const searchButton = document.getElementById('searchProblemByPositionBtn');
    if (searchButton) {
        searchButton.addEventListener('click', handleSearchButtonClick);
    }

    // Event delegation for dynamically generated buttons
    const resultsList = document.getElementById('pst_positionResultsList');
    if (resultsList) {
        resultsList.addEventListener('click', handleResultsListClick);
    }

    // Add event listener for the update problem form submission
    const updateProblemForm = document.getElementById('updateProblemForm');
    if (updateProblemForm) {
        updateProblemForm.addEventListener('submit', handleFormSubmit);
    }

    // Function to handle form submission
    function handleFormSubmit(event) {
        // Get the dropdown elements
        const assetNumberDropdown = document.getElementById('update_pst_assetNumberDropdown');
        const locationDropdown = document.getElementById('update_pst_locationDropdown');

        // Extract the text values from the selected options
        let assetNumberText = '';
        if (assetNumberDropdown && assetNumberDropdown.selectedIndex > 0) {
            assetNumberText = assetNumberDropdown.options[assetNumberDropdown.selectedIndex].text;
        }

        let locationText = '';
        if (locationDropdown && locationDropdown.selectedIndex > 0) {
            locationText = locationDropdown.options[locationDropdown.selectedIndex].text;
        }

        console.log(`Form submission: Asset Number text = "${assetNumberText}", Location text = "${locationText}"`);

        // Create hidden fields to send these text values
        const assetNumberInput = document.createElement('input');
        assetNumberInput.type = 'hidden';
        assetNumberInput.name = 'asset_number';  // Match what the backend expects
        assetNumberInput.value = assetNumberText;
        this.appendChild(assetNumberInput);

        const locationInput = document.createElement('input');
        locationInput.type = 'hidden';
        locationInput.name = 'location';  // Match what the backend expects
        locationInput.value = locationText;
        this.appendChild(locationInput);

        // Form will now submit with both the selected IDs and the text values
        // No need to preventDefault() - let the form submit normally
    }

    // Function to handle Area change
    function handleAreaChange() {
        const areaId = dropdowns.area.value;
        if (areaId) {
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(areaId)}`)
                .then(data => populateDropdown(dropdowns.equipmentGroup, data, 'Select Equipment Group'))
                .catch(error => console.error('Error fetching equipment groups:', error));
        } else {
            resetAllDropdowns();
        }
    }

    // Function to handle Equipment Group change
    function handleEquipmentGroupChange() {
        const equipmentGroupId = dropdowns.equipmentGroup.value;
        if (equipmentGroupId) {
            fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(equipmentGroupId)}`)
                .then(data => populateDropdown(dropdowns.model, data, 'Select Model'))
                .catch(error => console.error('Error fetching models:', error));
        } else {
            resetDropdown(dropdowns.model, 'Select Model');
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number'); // Reset asset number dropdown
            resetDropdown(dropdowns.location, 'Select Location'); // Reset location dropdown
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    }

    // Function to handle Model change
    function handleModelChange() {
        const modelId = dropdowns.model.value;
        if (modelId) {
            // Fetch and populate asset numbers for the selected model
            fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${encodeURIComponent(modelId)}`)
                .then(data => {
                    console.log("Asset numbers loaded:", data);
                    populateDropdown(dropdowns.assetNumber, data, 'Select Asset Number', item => item.number);
                })
                .catch(error => console.error('Error fetching asset numbers:', error));

            // Fetch and populate locations for the selected model
            fetchData(`${GET_LOCATIONS_URL}?model_id=${encodeURIComponent(modelId)}`)
                .then(data => {
                    console.log("Locations loaded:", data);
                    populateDropdown(dropdowns.location, data, 'Select Location', item => item.name);
                })
                .catch(error => console.error('Error fetching locations:', error));

            // Do not touch the siteLocation dropdown here
        } else {
            resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
            resetDropdown(dropdowns.location, 'Select Location');
            resetDropdown(dropdowns.siteLocation, 'Select Site Location');
        }
    }

    // Function to handle Search button click
    function handleSearchButtonClick() {
        // Get text from the selected options (not the ID values)
        let assetNumberText = '';
        if (dropdowns.assetNumber && dropdowns.assetNumber.selectedIndex > 0) {
            assetNumberText = dropdowns.assetNumber.options[dropdowns.assetNumber.selectedIndex].text;
        }

        let locationText = '';
        if (dropdowns.location && dropdowns.location.selectedIndex > 0) {
            locationText = dropdowns.location.options[dropdowns.location.selectedIndex].text;
        }

        const searchParams = {
            area_id: dropdowns.area.value,
            equipment_group_id: dropdowns.equipmentGroup.value,
            model_id: dropdowns.model.value,
            asset_number: assetNumberText,  // Use the displayed text, not the ID value
            location: locationText,         // Use the displayed text, not the ID value
            site_location_id: dropdowns.siteLocation.value
        };

        console.log("Search parameters:", searchParams);

        // Check if at least one search criterion is provided
        const hasCriteria = Object.values(searchParams).some(value => value && value !== '');

        if (hasCriteria) {
            const queryString = new URLSearchParams(searchParams).toString();
            fetchData(`${SEARCH_PROBLEMS_URL}?${queryString}`)
                .then(data => displaySearchResults(data))
                .catch(error => console.error('Error searching for problems:', error));
        } else {
            alert('Please enter at least one search criterion.');
        }
    }

    // Function to handle click events in the search results list
    function handleResultsListClick(event) {
        if (event.target.classList.contains('update-problem-btn')) {
            const problemId = event.target.dataset.problemId;
            fetchProblemDetails(problemId);
        } else if (event.target.classList.contains('edit-solutions-btn')) {
            const problemId = event.target.dataset.problemId;
            editRelatedSolutions(problemId);
        }
    }

    // Fetch data with error handling
    function fetchData(url) {
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Unknown error occurred.');
                    });
                }
                return response.json();
            });
    }

    // Display search results
    function displaySearchResults(problems) {
        if (!Array.isArray(problems)) {
            console.error('Invalid data format received for search results.');
            return;
        }

        resultsList.innerHTML = '';
        document.getElementById('pst_searchResults').style.display = problems.length ? 'block' : 'none';

        problems.forEach(problem => {
            const listItem = document.createElement('li');
            listItem.classList.add('list-group-item');

            listItem.innerHTML = `
                <div>
                    <strong>${problem.name}</strong> - ${problem.description}
                </div>
                <div class="mt-2">
                    <button class="btn btn-sm btn-warning update-problem-btn" data-problem-id="${problem.id}">
                        Update Problem Position
                    </button>
                    <button class="btn btn-sm btn-info edit-solutions-btn" data-problem-id="${problem.id}">
                        Edit Related Solutions
                    </button>
                </div>
            `;
            resultsList.appendChild(listItem);
        });
    }

    // Fetch and display problem details in the update form
    function fetchProblemDetails(problemId) {
        if (!problemId) {
            console.error('Invalid problem ID:', problemId);
            return;
        }

        fetchData(`${GET_PROBLEM_DETAILS_URL}${encodeURIComponent(problemId)}`)
            .then(data => populateUpdateForm(data))
            .catch(error => console.error('Error fetching problem details:', error));
    }

    // Populate Update Form
    function populateUpdateForm(data) {
        // Log incoming data for debugging
        console.log("Populating update form with data:", data);

        // References to update form elements - update to use dropdown elements
        const updateDropdowns = {
            area: document.getElementById('update_pst_areaDropdown'),
            equipmentGroup: document.getElementById('update_pst_equipmentGroupDropdown'),
            model: document.getElementById('update_pst_modelDropdown'),
            assetNumber: document.getElementById('update_pst_assetNumberDropdown'), // Changed to dropdown
            location: document.getElementById('update_pst_locationDropdown'), // Changed to dropdown
            siteLocation: document.getElementById('update_pst_siteLocationDropdown')
        };

        // Log which elements were found/not found
        Object.entries(updateDropdowns).forEach(([key, element]) => {
            console.log(`Element ${key} found: ${element !== null}`);
        });

        if (!data || !data.problem || !data.position) {
            console.error('Problem data is undefined or null.');
            alert('Error loading problem details.');
            return;
        }

        // Set hidden problem ID and other fields with null checks
        const problemIdField = document.getElementById('update_problem_id');
        if (problemIdField) problemIdField.value = data.problem.id || '';

        const problemNameField = document.getElementById('update_problem_name');
        if (problemNameField) problemNameField.value = data.problem.name || '';

        const problemDescField = document.getElementById('update_problem_description');
        if (problemDescField) problemDescField.value = data.problem.description || '';

        // Populate and set the Area dropdown - with defensive null checks
        if (updateDropdowns.area && data.position.area_id) {
            updateDropdowns.area.value = data.position.area_id;
            updateDropdowns.area.disabled = false;

            // Fetch equipment groups based on the area
            fetchData(`${GET_EQUIPMENT_GROUPS_URL}?area_id=${encodeURIComponent(data.position.area_id)}`)
                .then(equipmentGroups => {
                    // Only proceed if equipment group dropdown exists
                    if (!updateDropdowns.equipmentGroup) {
                        console.warn('Equipment Group dropdown not found in DOM');
                        return Promise.reject('Equipment Group dropdown not found');
                    }

                    populateDropdown(updateDropdowns.equipmentGroup, equipmentGroups, 'Select Equipment Group');

                    // Check if equipment_group_id exists before setting value
                    if (data.position.equipment_group_id) {
                        updateDropdowns.equipmentGroup.value = data.position.equipment_group_id;
                        updateDropdowns.equipmentGroup.disabled = false;
                    }

                    // Only proceed to fetch models if we have an equipment group ID
                    if (data.position.equipment_group_id) {
                        return fetchData(`${GET_MODELS_URL}?equipment_group_id=${encodeURIComponent(data.position.equipment_group_id)}`);
                    } else {
                        return Promise.reject('No equipment group ID available');
                    }
                })
                .then(models => {
                    // Only proceed if model dropdown exists
                    if (!updateDropdowns.model) {
                        console.warn('Model dropdown not found in DOM');
                        return Promise.reject('Model dropdown not found');
                    }

                    populateDropdown(updateDropdowns.model, models, 'Select Model');

                    // Check if model_id exists before setting value
                    if (data.position.model_id) {
                        updateDropdowns.model.value = data.position.model_id;
                        updateDropdowns.model.disabled = false;
                    }

                    // Only proceed to fetch asset numbers if we have a model ID
                    if (data.position.model_id) {
                        return fetchData(`${GET_ASSET_NUMBERS_URL}?model_id=${encodeURIComponent(data.position.model_id)}`);
                    } else {
                        return Promise.reject('No model ID available');
                    }
                })
                .then(assetNumbers => {
                    // Populate asset number dropdown
                    if (updateDropdowns.assetNumber) {
                        console.log(`Populating asset number dropdown with ${assetNumbers.length} options`);
                        populateDropdown(updateDropdowns.assetNumber, assetNumbers, 'Select Asset Number', item => item.number);

                        // If we have asset_number_id, use that to select
                        if (data.position.asset_number_id) {
                            updateDropdowns.assetNumber.value = data.position.asset_number_id;
                        }
                        // Otherwise, try to find matching asset by number
                        else if (data.position.asset_number) {
                            // Look for option text that matches the asset number
                            for (let i = 0; i < updateDropdowns.assetNumber.options.length; i++) {
                                if (updateDropdowns.assetNumber.options[i].text === data.position.asset_number) {
                                    updateDropdowns.assetNumber.selectedIndex = i;
                                    break;
                                }
                            }
                        }
                        updateDropdowns.assetNumber.disabled = false;
                    }

                    // Fetch locations for the selected model
                    if (data.position.model_id) {
                        return fetchData(`${GET_LOCATIONS_URL}?model_id=${encodeURIComponent(data.position.model_id)}`);
                    } else {
                        return Promise.reject('No model ID available for locations');
                    }
                })
                .then(locations => {
                    // Populate location dropdown
                    if (updateDropdowns.location) {
                        console.log(`Populating location dropdown with ${locations.length} options`);
                        populateDropdown(updateDropdowns.location, locations, 'Select Location', item => item.name);

                        // If we have location_id, use that to select
                        if (data.position.location_id) {
                            updateDropdowns.location.value = data.position.location_id;
                        }
                        // Otherwise, try to find matching location by name
                        else if (data.position.location) {
                            // Look for option text that matches the location name
                            for (let i = 0; i < updateDropdowns.location.options.length; i++) {
                                if (updateDropdowns.location.options[i].text === data.position.location) {
                                    updateDropdowns.location.selectedIndex = i;
                                    break;
                                }
                            }
                        }
                        updateDropdowns.location.disabled = false;
                    }

                    // Prepare parameters for site locations fetch
                    if (updateDropdowns.siteLocation) {
                        // Build parameters object, only including those that exist
                        const params = {};
                        if (data.position.model_id) params.model_id = data.position.model_id;
                        if (data.position.area_id) params.area_id = data.position.area_id;
                        if (data.position.equipment_group_id) params.equipment_group_id = data.position.equipment_group_id;

                        // Only fetch if we have at least one parameter
                        if (Object.keys(params).length > 0) {
                            const siteLocationParams = new URLSearchParams(params).toString();
                            console.log(`Fetching site locations with params: ${siteLocationParams}`);
                            return fetchData(`${GET_SITE_LOCATIONS_URL}?${siteLocationParams}`);
                        } else {
                            // Fall back to fetching all site locations if no params
                            console.log('Fetching all site locations (no specific params)');
                            return fetchData(GET_SITE_LOCATIONS_URL);
                        }
                    } else {
                        return Promise.reject('Site location dropdown not found');
                    }
                })
                .then(siteLocations => {
                    if (Array.isArray(siteLocations) && updateDropdowns.siteLocation) {
                        populateDropdown(updateDropdowns.siteLocation, siteLocations, 'Select Site Location');

                        // Only set site location value if it exists
                        if (data.position.site_location_id) {
                            updateDropdowns.siteLocation.value = data.position.site_location_id;
                        }
                        updateDropdowns.siteLocation.disabled = false;
                    } else {
                        console.error('Invalid site locations data format or dropdown not found');
                    }
                })
                .catch(error => {
                    // Only log real errors, not our control flow rejections
                    if (typeof error === 'string' &&
                        (error.includes('dropdown not found') ||
                         error.includes('No model ID available') ||
                         error.includes('No equipment group ID available'))) {
                        console.warn(error); // Just a warning for expected issues
                    } else {
                        console.error('Error in the update form data loading chain:', error);
                    }
                });
        } else {
            console.error('Area dropdown not found or Area ID is missing in position data.');
            alert('Area information is missing or area dropdown not found.');
        }

        // Show the update problem section
        const updateSection = document.getElementById('pst_updateProblemSection');
        if (updateSection) {
            updateSection.style.display = 'block';
        }
    }

    // Fetch and populate Solutions Tab with related solutions for the selected problem
    function editRelatedSolutions(problemId) {
        if (!problemId) {
            console.error('Invalid problem ID:', problemId);
            return;
        }

        fetchData(`${GET_SOLUTIONS_URL}${encodeURIComponent(problemId)}`)
            .then(data => {
                populateSolutionsTab(data);
                // Switch to the Solutions tab after populating
                const solutionTab = document.getElementById('solution-tab');
                if (solutionTab) {
                    solutionTab.click();
                }
            })
            .catch(error => console.error('Error fetching related solutions:', error));
    }

    // Populate Solutions Tab
    function populateSolutionsTab(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        if (!solutionsDropdown) {
            console.error('Solutions dropdown element not found.');
            return;
        }

        solutionsDropdown.innerHTML = ''; // Clear previous options

        if (Array.isArray(solutions)) {
            solutions.forEach(solution => {
                const option = document.createElement('option');
                option.value = solution.id;
                option.textContent = `${solution.name} - ${solution.description}`;
                solutionsDropdown.appendChild(option);
            });
        } else {
            console.error('Invalid solutions data format.');
        }
    }

    // Utility functions
    function populateDropdown(dropdown, data, placeholder, displayTextFunc) {
        if (!dropdown) {
            console.error('Dropdown element is undefined.');
            return;
        }

        dropdown.innerHTML = `<option value="">${placeholder}</option>`;

        if (Array.isArray(data)) {
            data.forEach(item => {
                // Use the provided displayTextFunc if available, otherwise use default
                let displayText = displayTextFunc ? displayTextFunc(item) :
                    (item.name || `${item.title} - Room ${item.room_number}`);

                const option = document.createElement('option');
                option.value = item.id;
                option.textContent = displayText;
                dropdown.appendChild(option);
            });
            dropdown.disabled = false;
        } else {
            console.error('Data provided to populateDropdown is not an array:', data);
        }
    }

    function resetDropdown(dropdown, placeholder) {
        if (dropdown) {
            dropdown.innerHTML = `<option value="">${placeholder}</option>`;
            dropdown.disabled = true;
        }
    }

    function resetAllDropdowns() {
        resetDropdown(dropdowns.equipmentGroup, 'Select Equipment Group');
        resetDropdown(dropdowns.model, 'Select Model');
        resetDropdown(dropdowns.assetNumber, 'Select Asset Number');
        resetDropdown(dropdowns.location, 'Select Location');
        resetDropdown(dropdowns.siteLocation, 'Select Site Location');
    }
});