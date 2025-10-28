// Role-Based Access Control for Tool Templates
document.addEventListener('DOMContentLoaded', function() {
    // Get the user level from multiple possible sources
    let userLevel = '';

    // Method 1: Check data attribute
    const userLevelElement = document.querySelector('[data-user-level]');
    if (userLevelElement) {
        userLevel = userLevelElement.getAttribute('data-user-level');
        console.log("User level from data attribute:", userLevel);
    }

    // Method 2: Check for visible user level display in the sidebar
    const userLevelText = document.querySelector('.user-level, .user-info');
    if (userLevelText) {
        const levelText = userLevelText.textContent;
        console.log("Found user level text:", levelText);

        // Check if it contains "ADMIN" or "Level III"
        if (levelText.includes('ADMIN') || levelText.toUpperCase().includes('ADMIN')) {
            userLevel = 'admin';
            console.log("Setting user level to admin based on sidebar text");
        } else if (levelText.includes('Level III') || levelText.includes('LEVEL III') || levelText.includes('level 3')) {
            userLevel = 'level_3';
            console.log("Setting user level to level_3 based on sidebar text");
        }
    }

    // Check if user is authorized (case-insensitive)
    const isAdmin = userLevel.toLowerCase() === 'admin' || userLevel === 'ADMIN';
    const isLevelThree = userLevel.toLowerCase() === 'level_3' ||
                         userLevel.toLowerCase() === 'level3' ||
                         userLevel.toLowerCase() === 'level 3';

    // Final authorization check (including general for testing)
    const isAuthorized = isAdmin || isLevelThree;

    console.log("Final authorization result:", isAuthorized);

    // DEBUGGING: Display authorization status on page for troubleshooting
    const debugElement = document.createElement('div');
    debugElement.style.position = 'fixed';
    debugElement.style.bottom = '10px';
    debugElement.style.right = '10px';
    debugElement.style.padding = '5px';
    debugElement.style.background = 'rgba(0,0,0,0.7)';
    debugElement.style.color = '#fff';
    debugElement.style.zIndex = '9999';
    debugElement.style.fontSize = '12px';
    debugElement.innerHTML = `Debug: Level="${userLevel}", Auth=${isAuthorized}`;
    document.body.appendChild(debugElement);

    // If not authorized, restrict access to add/edit functions
    if (!isAuthorized) {
        console.log("User is not authorized for add/edit operations");

        // 1. Handle Tool Category Template
        restrictCategoryTemplate();

        // 2. Handle Tool Manufacturer Template
        restrictManufacturerTemplate();

        // 3. Handle Tool Search Entry Template
        restrictToolSearchEntryTemplate();

        // 4. Handle Search Tool Template (only disable submit buttons, not search)
        restrictSearchToolTemplate();

        // Add notification banner
        addNotificationBanner();

        // Force switch to search tab by default
        const searchTab = document.querySelector('.tab-item[data-tab="search-tool-tab"]');
        if (searchTab) {
            // Simulate click on search tab if it's not already active
            if (!searchTab.classList.contains('active')) {
                searchTab.click();
            }
        }
    } else {
        console.log("User is authorized for add/edit operations");
        // Remove any existing locks on UI elements for authorized users
        removeLocks();
    }

    // Function to remove locks for authorized users
    function removeLocks() {
        // Remove locks from tabs
        const lockedTabs = document.querySelectorAll('.tab-item.disabled-tab');
        lockedTabs.forEach(tab => {
            tab.classList.remove('disabled-tab');
            tab.removeAttribute('title');
        });

        // Remove locks from buttons
        const disabledButtons = document.querySelectorAll('.btn-disabled');
        disabledButtons.forEach(button => {
            button.removeAttribute('disabled');
            button.classList.remove('btn-disabled');
            button.removeAttribute('title');
        });

        // Remove read-only class from forms
        const readOnlyForms = document.querySelectorAll('.read-only-form');
        readOnlyForms.forEach(form => {
            form.classList.remove('read-only-form');
        });

        // Remove any notification banners
        const notifications = document.querySelectorAll('.user-level-notification');
        notifications.forEach(notification => {
            notification.remove();
        });
    }

    // Function to restrict Tool Category Template
    function restrictCategoryTemplate() {
        // Disable add/edit/delete category forms
        const categoryForms = document.querySelectorAll('#add_category_form, #edit_category_form, #delete_category_form');
        categoryForms.forEach(form => {
            disableForm(form);
        });

        // Disable edit/delete buttons in the table
        const categoryButtons = document.querySelectorAll('.edit-category, .delete-category');
        categoryButtons.forEach(button => {
            button.setAttribute('disabled', 'disabled');
            button.classList.add('btn-disabled');
            button.title = 'Only Level III and Admin users can perform this action';
        });
    }

    // Function to restrict Tool Manufacturer Template
    function restrictManufacturerTemplate() {
        // Disable add/edit/delete manufacturer forms
        const manufacturerForms = document.querySelectorAll('#add_manufacturer_form, #edit_manufacturer_form, #delete_manufacturer_form');
        manufacturerForms.forEach(form => {
            disableForm(form);
        });

        // Disable edit/delete buttons in the table
        const manufacturerButtons = document.querySelectorAll('.edit-manufacturer, .delete-manufacturer');
        manufacturerButtons.forEach(button => {
            button.setAttribute('disabled', 'disabled');
            button.classList.add('btn-disabled');
            button.title = 'Only Level III and Admin users can perform this action';
        });
    }

    // Function to restrict Tool Search Entry Template
    function restrictToolSearchEntryTemplate() {
        // Disable tool add form
        const toolAddForm = document.querySelector('#tool_add_form');
        if (toolAddForm) {
            disableForm(toolAddForm);
        }

        // If tabs are present, disable certain tabs but leave search available
        const restrictedTabs = document.querySelectorAll('.tab-item[data-tab="add-tool-tab"], .tab-item[data-tab="tool-manufacturer-tab"], .tab-item[data-tab="tool-category-tab"]');
        restrictedTabs.forEach(tab => {
            tab.classList.add('disabled-tab');
            tab.title = 'Only Level III and Admin users can access this tab';

            // Prevent clicking on restricted tabs
            tab.addEventListener('click', function(event) {
                if (!isAuthorized) {
                    event.preventDefault();
                    event.stopPropagation();
                    alert('Only Level III and Admin users can access this tab');
                    return false;
                }
            }, true);
        });
    }

    // Function to restrict Search Tool Template (only disable submit buttons, not search)
    function restrictSearchToolTemplate() {
        // Allow search but disable any add/edit buttons
        const addEditButtons = document.querySelectorAll('.btn-primary:not([type="search"]), .btn-success, .btn-danger, .btn-warning');
        addEditButtons.forEach(button => {
            // Skip search buttons
            if (button.textContent.toLowerCase().includes('search') ||
                button.id.toLowerCase().includes('search') ||
                button.name.toLowerCase().includes('search')) {
                return;
            }

            // Disable non-search action buttons
            button.setAttribute('disabled', 'disabled');
            button.classList.add('btn-disabled');
            button.title = 'Only Level III and Admin users can perform this action';
        });
    }

    // Helper function to disable a form
    function disableForm(form) {
        if (!form) return;

        // Disable submit buttons
        const submitButtons = form.querySelectorAll('button[type="submit"], input[type="submit"]');
        submitButtons.forEach(button => {
            button.setAttribute('disabled', 'disabled');
            button.classList.add('btn-disabled');
            button.title = 'Only Level III and Admin users can submit this form';
        });

        // Add form submit prevention
        form.addEventListener('submit', function(event) {
            if (!isAuthorized) {
                event.preventDefault();
                alert('Only Level III and Admin users can submit this form');
                return false;
            }
        });

        // Optional: Add visual indication that form is read-only
        form.classList.add('read-only-form');
    }

    // Add notification banner to inform users of restricted access
    function addNotificationBanner() {
        const container = document.querySelector('.container') || document.querySelector('.main-container') || document.body;

        if (container) {
            const notification = document.createElement('div');
            notification.className = 'alert alert-warning user-level-notification';
            notification.innerHTML = '<strong>Note:</strong> You have limited access. Only Level III and Admin users can add or edit content.';

            // Insert at the beginning of the container
            if (container.firstChild) {
                container.insertBefore(notification, container.firstChild);
            } else {
                container.appendChild(notification);
            }
        }
    }
});