// Function to display thumbnails for parts images
function displayPartsThumbnails(thumbnails) {
    const thumbnailsSection = document.getElementById("parts-thumbnails-section");
    thumbnailsSection.innerHTML = ""; // Clear existing thumbnails

    thumbnails.forEach(thumbnail => {
        // Create an anchor element for each thumbnail
        const anchor = document.createElement("a");
        anchor.href = thumbnail.src; // Set the link destination
        anchor.setAttribute("data-full-src", thumbnail.src); // Set the data-full-src attribute

        // Create an image element for each thumbnail
        const img = document.createElement("img");
        img.src = thumbnail.thumbnail_src; // Set the image source
        img.alt = thumbnail.title; // Set the alternative text
        img.title = thumbnail.title; // Set the image title

        // Add the 'thumbnail' class for styling
        img.classList.add("thumbnail");

        // Append the image to the anchor element
        anchor.appendChild(img);

        // Append the anchor element to the thumbnails section
        thumbnailsSection.appendChild(anchor);
    });
}

// Update the search form submission to prevent the default behavior and fetch images for parts
$(document).on('submit', 'form[action="/tsg_search_images_parts"]', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Get the form data
    var formData = $(this).serialize();

    // Send an AJAX request to fetch images for parts
    $.ajax({
        url: '/tsg_search_images_parts', // Updated URL for parts images
        method: 'GET',
        data: formData,
        success: function(response) {
            // Extract thumbnails from the response
            var thumbnails = response.thumbnails;

            // Call the function to display thumbnails for parts
            displayPartsThumbnails(thumbnails);
        },
        error: function(xhr, status, error) {
            // Handle errors if any
            console.error('Error:', error);
        }
    });
});
