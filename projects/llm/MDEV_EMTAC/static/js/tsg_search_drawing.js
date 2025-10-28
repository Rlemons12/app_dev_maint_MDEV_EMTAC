$(document).ready(function() {
    $('#drawingSearchForm').submit(function(event) {
        event.preventDefault(); // Prevent the default form submission

        // Get form values
        var equipmentName = $('#drw_equipment_name').val();
        var drawingNumber = $('#drw_number').val();
        var drawingName = $('#drw_name').val();
        var revision = $('#drw_revision').val();
        var sparePartNumber = $('#drw_spare_part_number').val();

        // Send the data to the server using AJAX
        $.ajax({
            url: '/tsg_search_drawing',
            type: 'GET',
            data: {
                drw_equipment_name: equipmentName,
                drw_number: drawingNumber,
                drw_name: drawingName,
                drw_revision: revision,
                drw_spare_part_number: sparePartNumber
            },
            success: function(data) {
                // Handle the response data
                var searchResults = $('#searchResults');
                searchResults.empty(); // Clear previous results

                if (data.length === 0) {
                    searchResults.append('<p>No drawings found.</p>');
                } else {
                    var resultsTable = '<table><tr><th>Equipment Name</th><th>Drawing Number</th><th>Drawing Name</th><th>Revision</th><th>Spare Part Number</th></tr>';
                    $.each(data, function(index, drawing) {
                        resultsTable += '<tr>';
                        resultsTable += '<td>' + drawing.equipment_name + '</td>';
                        resultsTable += '<td>' + drawing.number + '</td>';
                        resultsTable += '<td>' + drawing.name + '</td>';
                        resultsTable += '<td>' + drawing.revision + '</td>';
                        resultsTable += '<td>' + drawing.spare_part_number + '</td>';
                        resultsTable += '</tr>';
                    });
                    resultsTable += '</table>';
                    searchResults.append(resultsTable);
                }
            },
            error: function(xhr, status, error) {
                console.error('Error searching drawings:', error);
            }
        });
    });
});
