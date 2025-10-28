// Function to display thumbnails
function displayThumbnails(thumbnails) {
    const thumbnailsSection = document.getElementById("thumbnails-section");
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
