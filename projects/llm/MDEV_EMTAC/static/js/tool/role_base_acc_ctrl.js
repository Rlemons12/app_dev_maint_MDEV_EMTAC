// Role-Based Access Control for Tool Templates
document.addEventListener('DOMContentLoaded', function () {

    let userLevel = '';

    // ==================================
    // GET USER LEVEL
    // ==================================
    const userLevelElement = document.querySelector('#tool-user-level');

    if (userLevelElement) {
        userLevel = (userLevelElement.getAttribute('data-user-level') || '')
            .toUpperCase()
            .trim();

        console.log("[RBAC] User level from data attribute:", userLevel);
    }

    // Fallback: sidebar text (legacy support)
    if (!userLevel) {
        const userLevelText = document.querySelector('.user-level, .user-info');
        if (userLevelText) {
            const text = userLevelText.textContent.toUpperCase();

            if (text.includes('ADMIN')) {
                userLevel = 'ADMIN';
            } else if (text.includes('LEVEL III') || text.includes('LEVEL 3')) {
                userLevel = 'LEVEL_III';
            }
        }
    }

    // ==================================
    // AUTHORIZATION (FIXED)
    // ==================================
    const isAdmin = userLevel === 'ADMIN';
    const isLevelThree = userLevel === 'LEVEL_III';
    const isAuthorized = isAdmin || isLevelThree;

    console.log("[RBAC] Final authorization:", isAuthorized);

    // ==================================
    // DEBUG (optional)
    // ==================================
    const debugElement = document.createElement('div');
    debugElement.style.position = 'fixed';
    debugElement.style.bottom = '10px';
    debugElement.style.right = '10px';
    debugElement.style.padding = '5px';
    debugElement.style.background = 'rgba(0,0,0,0.7)';
    debugElement.style.color = '#fff';
    debugElement.style.zIndex = '9999';
    debugElement.style.fontSize = '12px';
    debugElement.innerHTML = `Level="${userLevel}", Auth=${isAuthorized}`;
    document.body.appendChild(debugElement);

    // ==================================
    // MAIN CONTROL
    // ==================================
    if (!isAuthorized) {
        restrictCategoryTemplate();
        restrictManufacturerTemplate();
        restrictToolSearchEntryTemplate();
        restrictSearchToolTemplate();

        addNotificationBanner();

        const searchTab = document.querySelector('.tab-item[data-tab="search-tool-tab"]');
        if (searchTab && !searchTab.classList.contains('active')) {
            searchTab.click();
        }
    } else {
        removeLocks();
    }

    // ==================================
    // NOTIFICATION
    // ==================================
    function addNotificationBanner() {

        const slot = document.querySelector('#tool-role-notification-slot');

        if (!slot) {
            console.warn('[RBAC] Notification slot not found');
            return;
        }

        if (slot.querySelector('.user-level-notification')) {
            return;
        }

        const notification = document.createElement('div');
        notification.className = 'alert alert-warning user-level-notification';
        notification.innerHTML =
            '<strong>Note:</strong> You have limited access. Only Level III and Admin users can add or edit content.';

        slot.appendChild(notification);
    }

    // ==================================
    // REMOVE LOCKS
    // ==================================
    function removeLocks() {

        document.querySelectorAll('.tab-item.disabled-tab').forEach(tab => {
            tab.classList.remove('disabled-tab');
            tab.removeAttribute('title');
        });

        document.querySelectorAll('.btn-disabled').forEach(button => {
            button.removeAttribute('disabled');
            button.classList.remove('btn-disabled');
            button.removeAttribute('title');
        });

        document.querySelectorAll('.read-only-form').forEach(form => {
            form.classList.remove('read-only-form');
        });

        document.querySelectorAll('.user-level-notification').forEach(n => n.remove());
    }

    // ==================================
    // RESTRICTIONS
    // ==================================
    function restrictCategoryTemplate() {
        document.querySelectorAll('#add_category_form, #edit_category_form, #delete_category_form')
            .forEach(disableForm);

        document.querySelectorAll('.edit-category, .delete-category')
            .forEach(button => lockButton(button));
    }

    function restrictManufacturerTemplate() {
        document.querySelectorAll('#add_manufacturer_form, #edit_manufacturer_form, #delete_manufacturer_form')
            .forEach(disableForm);

        document.querySelectorAll('.edit-manufacturer, .delete-manufacturer')
            .forEach(button => lockButton(button));
    }

    function restrictToolSearchEntryTemplate() {

        const toolAddForm = document.querySelector('#tool_add_form');
        if (toolAddForm) disableForm(toolAddForm);

        const restrictedTabs = document.querySelectorAll(
            '.tab-item[data-tab="add-tool-tab"], ' +
            '.tab-item[data-tab="tool-manufacturer-tab"], ' +
            '.tab-item[data-tab="tool-category-tab"]'
        );

        restrictedTabs.forEach(tab => {
            tab.classList.add('disabled-tab');
            tab.title = 'Only Level III and Admin users can access this tab';

            tab.addEventListener('click', function (event) {
                if (!isAuthorized) {
                    event.preventDefault();
                    event.stopPropagation();
                    alert('Only Level III and Admin users can access this tab');
                }
            }, true);
        });
    }

    function restrictSearchToolTemplate() {

        document.querySelectorAll('.btn-primary, .btn-success, .btn-danger, .btn-warning')
            .forEach(button => {

                const text = button.textContent.toLowerCase();
                const id = (button.id || '').toLowerCase();

                if (text.includes('search') || id.includes('search')) return;

                lockButton(button);
            });
    }

    function disableForm(form) {
        if (!form) return;

        form.classList.add('read-only-form');

        form.querySelectorAll('button[type="submit"], input[type="submit"]')
            .forEach(button => lockButton(button));

        form.addEventListener('submit', function (event) {
            if (!isAuthorized) {
                event.preventDefault();
                alert('Only Level III and Admin users can submit this form');
            }
        });
    }

    function lockButton(button) {
        if (!button) return;

        button.setAttribute('disabled', 'disabled');
        button.classList.add('btn-disabled');
        button.title = 'Only Level III and Admin users can perform this action';
    }

});