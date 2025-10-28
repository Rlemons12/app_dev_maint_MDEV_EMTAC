// drawing_search.js - JavaScript for the drawing search form

document.addEventListener('DOMContentLoaded', function() {
    // Get references to form elements
    const searchForm = document.getElementById('drawing-search-form');
    const fieldsSelect = document.getElementById('fields');
    const resetButton = searchForm.querySelector('button[type="reset"]');

    // Convert fields to a comma-separated list before submission
    searchForm.addEventListener('submit', function(e) {
        // Prevent submission if no search criteria are provided
        const searchText = document.getElementById('search_text').value.trim();
        const drawingId = document.getElementById('drawing_id').value.trim();
        const equipmentName = document.getElementById('drw_equipment_name').value.trim();
        const drawingNumber = document.getElementById('drw_number').value.trim();
        const drawingName = document.getElementById('drw_name').value.trim();
        const revision = document.getElementById('drw_revision').value.trim();
        const sparePartNumber = document.getElementById('drw_spare_part_number').value.trim();
        const filePath = document.getElementById('file_path').value.trim();

        // Check if at least one search criterion is provided
        if (!searchText && !drawingId && !equipmentName && !drawingNumber &&
            !drawingName && !revision && !sparePartNumber && !filePath) {
            e.preventDefault();
            alert('Please provide at least one search criterion');
            return;
        }

        // Handle multiple select fields
        if (fieldsSelect) {
            const selectedFields = Array.from(fieldsSelect.selectedOptions).map(option => option.value);

            if (selectedFields.length > 0) {
                // Create a hidden input with the comma-separated list
                const hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.name = 'fields';
                hiddenInput.value = selectedFields.join(',');
                this.appendChild(hiddenInput);

                // Disable the original select to prevent it from being submitted with multiple values
                fieldsSelect.disabled = true;
            }
        }
    });

    // Re-enable fields after form submission to avoid issues with back button
    if (searchForm) {
        window.addEventListener('pageshow', function(event) {
            if (event.persisted || (window.performance &&
                window.performance.navigation.type === window.performance.navigation.TYPE_BACK_FORWARD)) {
                fieldsSelect.disabled = false;
            }
        });
    }

    // Custom reset handler to ensure all fields are properly reset
    if (resetButton) {
        resetButton.addEventListener('click', function(e) {
            // Reset the multiple select
            if (fieldsSelect) {
                Array.from(fieldsSelect.options).forEach(option => {
                    option.selected = false;
                });
            }

            // Reset checkboxes
            document.getElementById('exact_match').checked = false;
            document.getElementById('include_part_images').checked = false;

            // Reset text and number inputs
            document.getElementById('search_text').value = '';
            document.getElementById('drawing_id').value = '';
            document.getElementById('drw_equipment_name').value = '';
            document.getElementById('drw_number').value = '';
            document.getElementById('drw_name').value = '';
            document.getElementById('drw_revision').value = '';
            document.getElementById('drw_spare_part_number').value = '';
            document.getElementById('file_path').value = '';
            document.getElementById('limit').value = '100';

            // Prevent the default reset behavior since we're handling it manually
            e.preventDefault();
        });
    }

    // Set default selections for fields if search_text is provided
    const searchTextInput = document.getElementById('search_text');
    searchTextInput.addEventListener('input', function() {
        if (this.value.trim() !== '' && fieldsSelect.selectedOptions.length === 0) {
            // Default selections if user enters search text but doesn't select fields
            Array.from(fieldsSelect.options).forEach(option => {
                if (['drw_equipment_name', 'drw_number', 'drw_name'].includes(option.value)) {
                    option.selected = true;
                }
            });
        }
    });

    // Real-time validation for the limit field
    const limitInput = document.getElementById('limit');
    limitInput.addEventListener('input', function() {
        const value = parseInt(this.value, 10);
        if (isNaN(value) || value < 1) {
            this.value = 1;
        } else if (value > 1000) {
            this.value = 1000;
        }
    });
});

// Helper function to update the form with URL parameters (for when returning to the page)
function populateFormFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);

    // Populate text and number fields
    document.getElementById('search_text').value = urlParams.get('search_text') || '';
    document.getElementById('drawing_id').value = urlParams.get('drawing_id') || '';
    document.getElementById('drw_equipment_name').value = urlParams.get('drw_equipment_name') || '';
    document.getElementById('drw_number').value = urlParams.get('drw_number') || '';
    document.getElementById('drw_name').value = urlParams.get('drw_name') || '';
    document.getElementById('drw_revision').value = urlParams.get('drw_revision') || '';
    document.getElementById('drw_spare_part_number').value = urlParams.get('drw_spare_part_number') || '';
    document.getElementById('file_path').value = urlParams.get('file_path') || '';
    document.getElementById('limit').value = urlParams.get('limit') || '100';

    // Populate checkboxes
    document.getElementById('exact_match').checked = urlParams.get('exact_match') === 'true';
    document.getElementById('include_part_images').checked = urlParams.get('include_part_images') === 'true';

    // Populate multi-select fields
    const fieldsParam = urlParams.get('fields');
    if (fieldsParam) {
        const fieldsArray = fieldsParam.split(',');
        const fieldsSelect = document.getElementById('fields');

        Array.from(fieldsSelect.options).forEach(option => {
            option.selected = fieldsArray.includes(option.value);
        });
    }
}

// Call the function when the page loads
window.addEventListener('DOMContentLoaded', populateFormFromUrl);