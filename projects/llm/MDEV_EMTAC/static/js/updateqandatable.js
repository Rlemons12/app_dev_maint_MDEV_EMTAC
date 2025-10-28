document.addEventListener('DOMContentLoaded', function() {
    // Get the submit button by its ID
    var submitButton = document.getElementById('submit_comment_rating');

    // Add an event listener to the submit button
    submitButton.addEventListener('click', function() {
        // Get the values of the rating and comment
        var rating = document.getElementById('rating').value;
        var comment = document.getElementById('comment').value;

        // Call a function to update the qanda table with the rating and comment
        updateQandATable(rating, comment);
    });

    // Function to update the qanda table with the rating and comment
    function updateQandATable(rating, comment) {
        // Make an AJAX request to update the qanda table
        fetch('/chatbot/update_qanda', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                rating: rating,
                comment: comment
            })
        })
        .then(response => {
            if (response.ok) {
                console.log('Q&A table updated successfully.');
                // Optionally, you can display a success message to the user
            } else {
                console.error('Failed to update Q&A table.');
                // Optionally, you can display an error message to the user
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Optionally, you can display an error message to the user
        });
    }
});
