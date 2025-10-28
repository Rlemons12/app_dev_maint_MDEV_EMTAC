// User Role-Based Button Access Control

document.addEventListener('DOMContentLoaded', function() {
    // Get the current user level from the page
    // This assumes the user level is displayed on the page in an element with a specific class or ID
    // Adjust this selector to match your actual DOM structure
    const userLevelElement = document.querySelector('.user-level') || document.querySelector('#user-level');
    let userLevel = '';

    if (userLevelElement) {
        userLevel = userLevelElement.textContent.trim();
    }

    // Check if user is Level_III or admin
    const isAuthorized = userLevel.includes('LEVEL_III') ||
                          userLevel.includes('LEVEL III') ||
                          userLevel.includes('Level III') ||
                          userLevel.includes('admin') ||
                          userLevel.includes('ADMIN');

    // If not authorized, disable all submit buttons
    if (!isAuthorized) {
        // Find all submit buttons
        const submitButtons = document.querySelectorAll('button[type="submit"], input[type="submit"]');

        // Add submit prevention to all forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function(event) {
                if (!isAuthorized) {
                    event.preventDefault();
                    alert('Only Level III and Admin users can submit forms.');
                    return false;
                }
            });
        });

        // Disable buttons and add visual cue
        submitButtons.forEach(button => {
            // Disable the button
            button.setAttribute('disabled', 'disabled');

            // Add styling to show it's disabled
            button.classList.add('btn-disabled');

            // Add tooltip explaining why it's disabled
            button.title = 'Only Level III and Admin users can perform this action';

            // Store original text if it's a button element
            if (button.tagName === 'BUTTON') {
                button.dataset.originalText = button.textContent;
                button.textContent += ' (Level III+ Only)';
            }
        });

        // Also disable action buttons that aren't submit buttons but perform similar actions
        const actionButtons = document.querySelectorAll(
            '#addSolutionBtn, #removeSolutionsBtn, #addTaskBtn, #removeTaskBtn, ' +
            '#updateTaskDetailsBtn, #savePositionBtn, #addPositionBtn, ' +
            '#savePartsBtn, #saveDrawingsBtn, #saveDocumentsBtn, #saveImagesBtn, ' +
            '#saveToolsBtn'
        );

        actionButtons.forEach(button => {
            // Disable the button
            button.setAttribute('disabled', 'disabled');

            // Add styling to show it's disabled
            button.classList.add('btn-disabled');

            // Add tooltip explaining why it's disabled
            button.title = 'Only Level III and Admin users can perform this action';

            // Store original text
            button.dataset.originalText = button.textContent;
            button.textContent += ' (Level III+ Only)';

            // Add click prevention
            button.addEventListener('click', function(event) {
                if (!isAuthorized) {
                    event.preventDefault();
                    event.stopPropagation();
                    alert('Only Level III and Admin users can perform this action.');
                    return false;
                }
            });
        });

        // Add a notification at the top of the page
        const container = document.querySelector('.container');
        if (container) {
            const notification = document.createElement('div');
            notification.className = 'alert alert-warning';
            notification.innerHTML = '<strong>Note:</strong> You are viewing in read-only mode. Only Level III and Admin users can make changes.';
            container.insertBefore(notification, container.firstChild);
        }
    }
});