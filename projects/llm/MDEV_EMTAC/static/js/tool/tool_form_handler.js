// static/js/tool/tool_form_handler.js

document.addEventListener('DOMContentLoaded', () => {
    initializeToolForm();
});

function initializeToolForm() {
    const toolForm = document.getElementById('tool_add_form');
    if (!toolForm) {
        console.warn('Tool form not found.');
        return;
    }

    toolForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData(toolForm);

        // Append 'submit_tool' to FormData
        const submitButton = toolForm.querySelector('input[name="submit_tool"], button[name="submit_tool"]');
        if (submitButton) {
            formData.append('submit_tool', submitButton.value);
            console.log('submit_tool appended:', submitButton.value);
        } else {
            console.warn('Submit button with name "submit_tool" not found.');
        }

        try {
            const response = await fetch(toolForm.action, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const contentType = response.headers.get('Content-Type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new TypeError('Expected JSON response');
            }

            const result = await response.json();

            if (response.ok && result.success) {
                displayFormSuccess(result.message);
                toolForm.reset();
            } else {
                // If errors are returned, display them
                if (result.errors) {
                    // Aggregate error messages
                    let errorMessages = '';
                    for (const field in result.errors) {
                        if (result.errors.hasOwnProperty(field)) {
                            result.errors[field].forEach(error => {
                                errorMessages += `<strong>${field}:</strong> ${error}<br>`;
                            });
                        }
                    }
                    displayFormError(errorMessages || result.message || 'An error occurred while submitting the form.');
                } else {
                    displayFormError(result.message || 'An error occurred while submitting the form.');
                }
            }
        } catch (error) {
            console.error('Error submitting tool form:', error);
            displayFormError('An error occurred while submitting the form. Please try again.');
        }
    });
}

function displayFormError(message) {
    const formContainer = document.querySelector('.form-container');
    if (formContainer) {
        // Remove existing alerts
        const existingAlerts = formContainer.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        const newAlert = document.createElement('div');
        newAlert.className = 'alert alert-danger alert-dismissible fade show';
        newAlert.role = 'alert';
        newAlert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        formContainer.prepend(newAlert);
    } else {
        console.error('Form container not found. Cannot display error message.');
    }
}

function displayFormSuccess(message) {
    const formContainer = document.querySelector('.form-container');
    if (formContainer) {
        // Remove existing alerts
        const existingAlerts = formContainer.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        const newAlert = document.createElement('div');
        newAlert.className = 'alert alert-success alert-dismissible fade show';
        newAlert.role = 'alert';
        newAlert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        formContainer.prepend(newAlert);
    } else {
        console.error('Form container not found. Cannot display success message.');
    }
}
