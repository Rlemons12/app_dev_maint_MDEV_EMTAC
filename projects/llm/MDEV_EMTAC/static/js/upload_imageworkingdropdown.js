// Function to fetch data and populate dropdowns
function populateDropdowns() {
    // Define an array of dropdown elements along with their corresponding data keys
    var dropdowns = [
        { element: $('#areaDropdown'), dataKey: 'areas' },
        { element: $('#equipmentGroupDropdown'), dataKey: 'equipment_groups' },
        { element: $('#modelDropdown'), dataKey: 'models' },
        { element: $('#assetNumberDropdown'), dataKey: 'asset_numbers' },
        { element: $('#locationDropdown'), dataKey: 'locations' }
    ];

    // AJAX request to fetch data
    $.ajax({
        url: '/get_list_data', // URL to fetch data from (replace with your server-side route)
        type: 'GET',
        success: function(data) {
            // Populate dropdowns
        $.each(dropdowns, function(index, dropdown) {
            var dropdownElement = dropdown.element;
            dropdownElement.empty(); // Clear existing options
            // Populate dropdown with data
            $.each(data[dropdown.dataKey], function(index, item) {
                // Check if the dropdown is for asset numbers
                if (dropdown.dataKey === 'asset_numbers') {
                    dropdownElement.append('<option value="' + item.number + '">' + item.number + '</option>');
                } else {
                    dropdownElement.append('<option value="' + item.name + '">' + item.name + '</option>');
                }
            });
            dropdownElement.select2(); // Initialize Select2
        });
    },
    error: function(xhr, status, error) {
        console.error('Error fetching data:', error);
    }
});
}

// Call the function to populate dropdowns when the page loads
$(document).ready(function() {
    populateDropdowns();
});
