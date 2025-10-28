// Update the tsghandleSearchResults function to populate the text box with problem and solution, display associated images, and generate hyperlinks for associated documents
function tsg_handleSearchResults(formData) {
    $.ajax({
        type: 'GET',
        url: '/search_problem_solution',
        data: formData,
        dataType: 'html', // Change the data type to HTML since the server returns HTML content
        success: function(response) {
            // Handle the response and update the HTML content with the search results
            $('#answer-section').html(response); // Update the answer section with the HTML content
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
}

// Call the function to handle search results when the form is submitted
$(document).on('submit', 'form[action="/search_problem_solution"]', function(event) {
    event.preventDefault(); // Prevent the default form submission
    
    // Get the form data
    var formData = $(this).serialize();
    
    // Call the function to handle search results
    tsg_handleSearchResults(formData);
});
