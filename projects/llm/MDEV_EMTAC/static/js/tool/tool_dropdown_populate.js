// static/js/tool/tool_dropdown_populate.js

// Define constants for Flask API endpoints
const GET_TOOL_POSITIONS_URL = '/tool/get_tool_positions';
const GET_TOOL_PACKAGES_URL = '/tool/get_tool_packages';
const GET_TOOL_MANUFACTURERS_URL = '/tool/get_tool_manufacturers';
const GET_TOOL_CATEGORIES_URL = '/tool/get_tool_categories';

/**
 * Function to populate dropdowns dynamically.
 * @param {string} url - The API endpoint to fetch data from.
 * @param {HTMLElement} selectElement - The select DOM element to populate.
 * @param {string} defaultOptionText - The placeholder text for the select field.
 */
async function populateDropdown(url, selectElement, defaultOptionText) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Error fetching data from ${url}: ${response.statusText}`);
        }
        const data = await response.json();

        selectElement.innerHTML = ''; // Clear existing options

        // Add default placeholder option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = defaultOptionText;
        defaultOption.disabled = true;
        defaultOption.selected = true;
        selectElement.appendChild(defaultOption);

        // Determine data key and set items accordingly
        let items = [];
        if (Array.isArray(data)) {
            // For endpoints returning flat arrays (positions, manufacturers, packages)
            items = data;
        } else if (data.categories) {
            // For categories which may have subcategories
            items = data.categories;
        } else {
            // Handle other potential data structures
            console.warn(`Unhandled data structure from ${url}`);
            items = [];
        }

        // Populate select element with options
        items.forEach(item => addOption(selectElement, item));

        // Initialize Select2 if applicable
        initializeSelect2(selectElement, defaultOptionText);

    } catch (error) {
        console.error('Error fetching data:', error);
        displayFetchError(selectElement, 'Failed to load options. Please try again.');
    }
}

/**
 * Recursive function to add options and subcategories to a select element.
 * @param {HTMLElement} selectElement - The select DOM element to populate.
 * @param {Object} item - The current item to add as an option.
 * @param {number} depth - The current depth for nested subcategories.
 */
function addOption(selectElement, item, depth = 0) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${'--'.repeat(depth)} ${item.name}`;
    selectElement.appendChild(option);

    // If the item has subcategories, add them recursively
    if (item.subcategories && Array.isArray(item.subcategories) && item.subcategories.length > 0) {
        item.subcategories.forEach(subItem => addOption(selectElement, subItem, depth + 1));
    }
}

/**
 * Function to display a fetch error within the select element.
 * @param {HTMLElement} selectElement - The select DOM element to display the error in.
 * @param {string} message - The error message to display.
 */
function displayFetchError(selectElement, message) {
    const errorOption = document.createElement('option');
    errorOption.value = '';
    errorOption.textContent = message;
    errorOption.disabled = true;
    errorOption.selected = true;
    selectElement.appendChild(errorOption);
}

/**
 * Function to initialize Select2 on a select element.
 * @param {HTMLElement} selectElement - The select DOM element to enhance with Select2.
 * @param {string} placeholder - The placeholder text for Select2.
 */
function initializeSelect2(selectElement, placeholder) {
    if ($(selectElement).hasClass('select2-hidden-accessible')) {
        // If Select2 is already initialized, destroy it first to avoid duplicates
        $(selectElement).select2('destroy');
    }

    $(selectElement).select2({
        placeholder: placeholder,
        allowClear: true,
        width: '100%' // Ensures Select2 uses the full width of the parent container
    });
}

/**
 * Initialize all select elements on the page.
 */
document.addEventListener('DOMContentLoaded', () => {
    const categorySelect = document.getElementById('tool_category');
    const manufacturerSelect = document.getElementById('tool_manufacturer');
    const positionSelect = document.getElementById('tool_position');
    const packageSelect = document.getElementById('tool_package');

    if (categorySelect) {
        populateDropdown(GET_TOOL_CATEGORIES_URL, categorySelect, "Select Category");
    }

    if (manufacturerSelect) {
        populateDropdown(GET_TOOL_MANUFACTURERS_URL, manufacturerSelect, "Select Manufacturer");
    }

    if (positionSelect) {
        populateDropdown(GET_TOOL_POSITIONS_URL, positionSelect, "Select Position");
    }

    if (packageSelect) {
        populateDropdown(GET_TOOL_PACKAGES_URL, packageSelect, "Select Package");
    }
});
