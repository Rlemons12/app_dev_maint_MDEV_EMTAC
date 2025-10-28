// static/js/tool/tool_manufacturer.js

console.log('tool_manufacturer.js loaded');

document.addEventListener('DOMContentLoaded', () => {
    initializeManufacturerManagement();
});

/**
 * Initializes the manufacturer management functionality by setting up event listeners.
 */
function initializeManufacturerManagement() {
    // Forms
    const addForm = document.getElementById('manufacturer_form');
    const editForm = document.getElementById('edit_manufacturer_form');
    const deleteForm = document.getElementById('delete_manufacturer_form');

    // Containers
    const feedbackContainer = document.getElementById('feedback-container');
    const manufacturersContainer = document.getElementById('manufacturers-results-container');

    // Initialize Add Manufacturer Form Submission
    if (addForm) {
        addForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Prevent default form submission

            // Clear previous error messages
            clearFormErrors('add');

            // Gather form data
            const formData = new FormData(addForm);

            try {
                // Fetch CSRF token from the form
                const csrfToken = addForm.querySelector('input[name="csrf_token"]').value;

                const response = await fetch(`/tool/tool_manufacturer/add`, { // Correct endpoint
                    method: 'POST',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrfToken // Include CSRF token
                    },
                    body: formData,
                    credentials: 'same-origin' // Include cookies
                });

                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const data = await response.json();
                    console.log('Add Response Data:', data); // Debugging

                    if (response.ok) {
                        if (data.success) {
                            displayFeedbackMessage('success', data.message || 'Manufacturer added successfully.');
                            addForm.reset();
                            // Optionally, refresh the manufacturers list
                            fetchManufacturers(1);
                        } else {
                            displayFeedbackMessage('danger', data.message || 'An unexpected error occurred.');
                        }
                    } else {
                        if (data.errors) {
                            displayFormErrors('add', data.errors);
                        } else {
                            displayFeedbackMessage('danger', data.message || 'An error occurred while adding the manufacturer.');
                        }
                    }
                } else {
                    throw new Error('Received non-JSON response');
                }
            } catch (error) {
                console.error('Error submitting add manufacturer:', error);
                displayFeedbackMessage('danger', 'An error occurred while adding the manufacturer. Please try again later.');
            }
        });
    }

    // Initialize Edit Manufacturer Form Submission
    if (editForm) {
        editForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Prevent default form submission

            // Clear previous error messages
            clearFormErrors('edit');

            // Gather form data
            const formData = new FormData(editForm);
            const manufacturerId = formData.get('manufacturer_id');

            try {
                // Fetch CSRF token from the form
                const csrfToken = editForm.querySelector('input[name="csrf_token"]').value;

                const response = await fetch(`/tool/tool_manufacturer/edit_manufacturer/${manufacturerId}`, { // Correct endpoint
                    method: 'POST',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrfToken // Include CSRF token
                    },
                    body: formData,
                    credentials: 'same-origin' // Include cookies
                });

                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const data = await response.json();
                    console.log('Edit Response Data:', data); // Debugging

                    if (response.ok) {
                        if (data.success) {
                            displayFeedbackMessage('success', data.message || 'Manufacturer updated successfully.');
                            editForm.reset();
                            // Optionally, refresh the manufacturers list
                            fetchManufacturers(1);
                            // Optionally, switch to the Add tab
                            switchToTab('add-tab');
                        } else {
                            displayFeedbackMessage('danger', data.message || 'An unexpected error occurred.');
                        }
                    } else {
                        if (data.errors) {
                            displayFormErrors('edit', data.errors);
                        } else {
                            displayFeedbackMessage('danger', data.message || 'An error occurred while updating the manufacturer.');
                        }
                    }
                } else {
                    throw new Error('Received non-JSON response');
                }
            } catch (error) {
                console.error('Error submitting edit manufacturer:', error);
                displayFeedbackMessage('danger', 'An error occurred while updating the manufacturer. Please try again later.');
            }
        });
    }

    // Initialize Delete Manufacturer Form Submission
    if (deleteForm) {
        deleteForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Prevent default form submission

            // Clear previous error messages
            clearFormErrors('delete');

            // Gather form data
            const formData = new FormData(deleteForm);
            const manufacturerId = formData.get('manufacturer_id');

            try {
                // Fetch CSRF token from the form
                const csrfToken = deleteForm.querySelector('input[name="csrf_token"]').value;

                const response = await fetch(`/tool/tool_manufacturer/delete`, { // Correct endpoint
                    method: 'POST',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrfToken // Include CSRF token
                    },
                    body: new URLSearchParams({
                        'manufacturer_id': manufacturerId
                    }),
                    credentials: 'same-origin' // Include cookies
                });

                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const data = await response.json();
                    console.log('Delete Response Data:', data); // Debugging

                    if (response.ok) {
                        if (data.success) {
                            displayFeedbackMessage('success', data.message || 'Manufacturer deleted successfully.');
                            deleteForm.reset();
                            // Optionally, refresh the manufacturers list
                            fetchManufacturers(1);
                            // Optionally, switch to the Add tab
                            switchToTab('add-tab');
                        } else {
                            displayFeedbackMessage('danger', data.message || 'An unexpected error occurred.');
                        }
                    } else {
                        displayFeedbackMessage('danger', data.message || 'An error occurred while deleting the manufacturer.');
                    }
                } else {
                    throw new Error('Received non-JSON response');
                }
            } catch (error) {
                console.error('Error submitting delete manufacturer:', error);
                displayFeedbackMessage('danger', 'An error occurred while deleting the manufacturer. Please try again later.');
            }
        });
    }

    // Handle Edit and Delete Button Clicks in the Manufacturers List
    if (manufacturersContainer) {
        // Handle Edit Button Click
        manufacturersContainer.addEventListener('click', async function (event) {
            const target = event.target.closest('.edit-manufacturer');
            if (!target) return;

            const manufacturerId = target.getAttribute('data-id');
            await populateEditForm(manufacturerId);
            switchToTab('edit-tab');
        });

        // Handle Delete Button Click
        manufacturersContainer.addEventListener('click', function (event) {
            const target = event.target.closest('.delete-manufacturer');
            if (!target) return;

            const manufacturerId = target.getAttribute('data-id');
            populateDeleteForm(manufacturerId);
            switchToTab('delete-tab');
        });
    }
}

/**
 * Fetches and populates the Edit Manufacturer form with existing data.
 * @param {number} manufacturerId - The ID of the manufacturer to edit.
 */
async function populateEditForm(manufacturerId) {
    try {
        const response = await fetch(`/tool/tool_manufacturer/get/${manufacturerId}`, { // Define this endpoint in Flask
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            credentials: 'same-origin'
        });

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            console.log('Populate Edit Form Response:', data);

            if (response.ok && data.success) {
                const manufacturer = data.manufacturer;
                document.getElementById('edit-manufacturer-id').value = manufacturer.id;
                document.getElementById('edit-manufacturer-name').value = manufacturer.name;
                document.getElementById('edit-manufacturer-description').value = manufacturer.description || '';
                document.getElementById('edit-manufacturer-country').value = manufacturer.country || '';
                document.getElementById('edit-manufacturer-website').value = manufacturer.website || '';
            } else {
                displayFeedbackMessage('danger', data.message || 'Failed to fetch manufacturer details.');
            }
        } else {
            throw new Error('Received non-JSON response');
        }
    } catch (error) {
        console.error('Error fetching manufacturer details:', error);
        displayFeedbackMessage('danger', 'An error occurred while fetching manufacturer details.');
    }
}

/**
 * Populates the Delete Manufacturer form with existing data.
 * @param {number} manufacturerId - The ID of the manufacturer to delete.
 */
function populateDeleteForm(manufacturerId) {
    // Find the manufacturer details from the table
    const row = document.querySelector(`button.delete-manufacturer[data-id="${manufacturerId}"]`).closest('tr');
    if (!row) {
        console.error('Manufacturer row not found for deletion.');
        displayFeedbackMessage('danger', 'Manufacturer not found.');
        return;
    }

    const name = row.querySelector('td:nth-child(1)').textContent;
    const deleteForm = document.getElementById('delete_manufacturer_form');

    if (deleteForm) {
        deleteForm.querySelector('#delete-manufacturer-id').value = manufacturerId;
        deleteForm.querySelector('#delete-manufacturer-name').value = name;
    }
}

/**
 * Fetches and displays manufacturers for a given page.
 * @param {number} page - The page number to fetch.
 */
async function fetchManufacturers(page = 1) {
    try {
        const response = await fetch(`/tool/get_tool_manufacturers?${new URLSearchParams({ page: page, per_page: 20 })}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            credentials: 'same-origin' // Include cookies
        });

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Manufacturer list endpoint not found. Please contact support.');
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Received non-JSON response');
        }

        const data = await response.json();
        console.log('Fetch Manufacturers Response:', data); // Debugging

        if (data.success) {
            updateManufacturersTable(data.manufacturers);
            updatePaginationControls(data.page, data.total_pages);
        } else {
            displayFeedbackMessage('danger', data.message || 'Failed to fetch manufacturers.');
        }
    } catch (error) {
        console.error('Error fetching manufacturers:', error);
        displayFeedbackMessage('danger', `An error occurred while fetching manufacturers: ${error.message}`);
    }
}

/**
 * Updates the manufacturers table with fetched data.
 * @param {Array} manufacturers - Array of manufacturer objects.
 */
function updateManufacturersTable(manufacturers) {
    const tbody = document.querySelector('#manufacturers-results-container table tbody');
    if (!tbody) {
        console.error("Manufacturers table body not found.");
        return;
    }

    tbody.innerHTML = ''; // Clear existing rows

    manufacturers.forEach(manufacturer => {
        const tr = document.createElement('tr');

        tr.innerHTML = `
            <td>${escapeHTML(manufacturer.name)}</td>
            <td>${escapeHTML(manufacturer.description) || 'N/A'}</td>
            <td>${escapeHTML(manufacturer.country)}</td>
            <td><a href="${escapeHTML(manufacturer.website)}" target="_blank">${escapeHTML(manufacturer.website)}</a></td>
            <td>
                <button class="btn btn-sm btn-warning edit-manufacturer" data-id="${manufacturer.id}">Edit</button>
                <button class="btn btn-sm btn-danger delete-manufacturer" data-id="${manufacturer.id}">Delete</button>
            </td>
        `;

        tbody.appendChild(tr);
    });
}

/**
 * Updates the pagination controls based on current page and total pages.
 * @param {number} currentPage - The current page number.
 * @param {number} totalPages - The total number of pages.
 */
function updatePaginationControls(currentPage, totalPages) {
    const paginationNav = document.querySelector('#manufacturers-results-container nav');
    if (paginationNav) {
        paginationNav.remove(); // Remove existing pagination
    }

    if (totalPages <= 1) return; // No need for pagination

    const container = document.getElementById('manufacturers-results-container');
    const paginationHTML = `
        <nav aria-label="Manufacturers pages">
            <ul class="pagination justify-content-center">
                ${currentPage > 1 ? `
                    <li class="page-item">
                        <a class="page-link" href="#" data-page="${currentPage - 1}" aria-label="Previous">
                            <span aria-hidden="true">&laquo; Previous</span>
                        </a>
                    </li>
                ` : `
                    <li class="page-item disabled">
                        <span class="page-link" aria-label="Previous">
                            <span aria-hidden="true">&laquo; Previous</span>
                        </span>
                    </li>
                `}

                <li class="page-item disabled">
                    <span class="page-link">
                        Page ${currentPage} of ${totalPages}
                    </span>
                </li>

                ${currentPage < totalPages ? `
                    <li class="page-item">
                        <a class="page-link" href="#" data-page="${currentPage + 1}" aria-label="Next">
                            <span aria-hidden="true">Next &raquo;</span>
                        </a>
                    </li>
                ` : `
                    <li class="page-item disabled">
                        <span class="page-link" aria-label="Next">
                            <span aria-hidden="true">Next &raquo;</span>
                        </span>
                    </li>
                `}
            </ul>
        </nav>
    `;

    container.insertAdjacentHTML('beforeend', paginationHTML);
}

/**
 * Displays feedback messages to the user.
 * @param {string} type - Type of message ('success', 'danger', etc.).
 * @param {string} message - The message content.
 */
function displayFeedbackMessage(type, message) {
    const feedbackContainer = document.getElementById('feedback-container');
    if (!feedbackContainer) {
        console.error("Feedback container element not found.");
        return;
    }

    feedbackContainer.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${escapeHTML(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
}

/**
 * Displays form validation errors.
 * @param {string} formType - The form type ('add', 'edit', 'delete').
 * @param {Object} errors - An object containing field-specific error messages.
 */
function displayFormErrors(formType, errors) {
    for (const [field, messages] of Object.entries(errors)) {
        const errorDiv = document.getElementById(`${formType}-error-${field}`);
        if (errorDiv) {
            errorDiv.textContent = messages.join(', ');
            errorDiv.previousElementSibling.classList.add('is-invalid');
        }
    }
}

/**
 * Clears all form validation error messages.
 * @param {string} formType - The form type ('add', 'edit', 'delete').
 */
function clearFormErrors(formType) {
    const errorElements = document.querySelectorAll(`#${formType}_form .invalid-feedback`);
    errorElements.forEach(el => {
        el.textContent = '';
        el.previousElementSibling.classList.remove('is-invalid');
    });
}

/**
 * Escapes HTML to prevent XSS attacks.
 * @param {string} unsafe - The string to escape.
 * @returns {string} - The escaped string.
 */
function escapeHTML(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Switches to a specific tab programmatically.
 * @param {string} tabId - The ID of the tab button to activate.
 */
function switchToTab(tabId) {
    const tabButton = document.getElementById(tabId);
    if (tabButton) {
        const tab = new bootstrap.Tab(tabButton);
        tab.show();
    }
}
