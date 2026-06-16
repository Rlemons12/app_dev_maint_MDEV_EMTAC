// static/js/pst_troubleshooting/role_base_button_acc_ctrl.js
// User Role-Based Button Access Control
// Tour-safe version

document.addEventListener('DOMContentLoaded', function () {
    const userLevelElement =
        document.querySelector('.user-level') ||
        document.querySelector('#user-level');

    let userLevel = '';
    if (userLevelElement) {
        userLevel = userLevelElement.textContent.trim();
    }

    const isAuthorized =
        userLevel.includes('LEVEL_III') ||
        userLevel.includes('LEVEL III') ||
        userLevel.includes('Level III') ||
        userLevel.includes('admin') ||
        userLevel.includes('ADMIN');

    function isTourActive() {
        return document.body.classList.contains('tour-active');
    }

    function showRestrictedAlert() {
        alert('Only Level III and Admin users can perform this action.');
    }

    function markRestricted(button) {
        if (!button) return;

        button.classList.add('btn-disabled', 'restricted-action');
        button.setAttribute('data-role-restricted', 'true');
        button.title = 'Only Level III and Admin users can perform this action';

        if (button.tagName === 'BUTTON' || button.tagName === 'INPUT') {
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.textContent || button.value || '';
            }

            const currentText = button.tagName === 'BUTTON'
                ? button.textContent
                : button.value;

            if (currentText && !currentText.includes('(Level III+ Only)')) {
                if (button.tagName === 'BUTTON') {
                    button.textContent = `${currentText} (Level III+ Only)`;
                } else {
                    button.value = `${currentText} (Level III+ Only)`;
                }
            }
        }

        // IMPORTANT:
        // Do NOT set disabled=true here.
        // Disabled buttons cannot be clicked by the tour.
        button.setAttribute('aria-disabled', 'true');
    }

    function interceptRestrictedAction(element) {
        if (!element) return;

        element.addEventListener(
            'click',
            function (event) {
                if (isAuthorized) {
                    return;
                }

                // During the tour, allow the click to happen so Intro.js can demonstrate the UI.
                if (isTourActive()) {
                    return;
                }

                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                showRestrictedAlert();
                return false;
            },
            true
        );
    }

    function interceptRestrictedSubmit(form) {
        if (!form) return;

        form.addEventListener(
            'submit',
            function (event) {
                if (isAuthorized) {
                    return;
                }

                // Allow form interaction during the tour without submitting
                if (isTourActive()) {
                    event.preventDefault();
                    return false;
                }

                event.preventDefault();
                event.stopPropagation();
                showRestrictedAlert();
                return false;
            },
            true
        );
    }

    if (!isAuthorized) {
        // Intercept all forms
        document.querySelectorAll('form').forEach(form => {
            interceptRestrictedSubmit(form);
        });

        // Restrict all submit buttons, but do not disable them
        const submitButtons = document.querySelectorAll(
            'button[type="submit"], input[type="submit"]'
        );

        submitButtons.forEach(button => {
            markRestricted(button);
            interceptRestrictedAction(button);
        });

        // Restrict specific action buttons, but do not disable them
        const actionButtons = document.querySelectorAll(
            '#addSolutionBtn, #removeSolutionsBtn, #addTaskBtn, #removeTaskBtn, ' +
            '#updateTaskDetailsBtn, #savePositionBtn, #addPositionBtn, ' +
            '#savePartsBtn, #saveDrawingsBtn, #saveDocumentsBtn, #saveImagesBtn, ' +
            '#saveToolsBtn'
        );

        actionButtons.forEach(button => {
            markRestricted(button);
            interceptRestrictedAction(button);
        });

        const container = document.querySelector('.container');
        if (container && !document.querySelector('.user-level-notification')) {
            const notification = document.createElement('div');
            notification.className = 'alert alert-warning user-level-notification';
            notification.innerHTML =
                '<strong>Note:</strong> You are viewing in read-only mode. Only Level III and Admin users can make changes.';
            container.insertBefore(notification, container.firstChild);
        }
    }
});