// Function to handle form submission
function submitForm(event) {
    event.preventDefault(); // Prevent the default form submission behavior
    
    // Get form data
    var formData = {
        drw_equipment_name: $('#drw_equipment_name').val(),
        drw_number: $('#drw_number').val(),
        drw_name: $('#drw_name').val(),
        drw_revision: $('#drw_revision').val(),
        drw_spare_part_number: $('#drw_spare_part_number').val()
    };
    
    // Send AJAX request to Flask route
    $.ajax({
        url: '/tsg_search_drawing',
        type: 'GET',
        data: formData,
        success: function(response) {
            // Handle successful response
            console.log(response); // Log the response to the console
            
            // Update HTML content with drawing data
            var drawingList = $('#drawingList');
            drawingList.empty(); // Clear previous results

            // Iterate through the response and append drawing details to the drawingList div
            response.forEach(function(drawing) {
                drawingList.append(
                    '<div>' +
                    '<p><strong>Equipment Name:</strong> ' + drawing.equipment_name + '</p>' +
                    '<p><strong>Drawing Number:</strong> ' + drawing.number + '</p>' +
                    '<p><strong>Drawing Name:</strong> ' + drawing.name + '</p>' +
                    '<p><strong>Revision:</strong> ' + drawing.revision + '</p>' +
                    '<p><strong>Spare Part Number:</strong> ' + drawing.spare_part_number + '</p>' +
                    '</div><hr>'
                );
            });
        },
        error: function(xhr, status, error) {
            // Handle error
            console.error('Error:', error);
        }
    });
}

// Event listener for form submission
$('#searchForm').submit(submitForm);
