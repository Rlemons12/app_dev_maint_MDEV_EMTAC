// JavaScript code to display hoverable links for relevant documents

// Assume doc_links is an array containing the hoverable links for relevant documents

// Get the thumbnails section element
var thumbnailsSection = document.getElementById('thumbnails-section');

// Clear any existing content in the thumbnails section
thumbnailsSection.innerHTML = '';

// Loop through each hoverable link and create an anchor element for it
doc_links.forEach(function(link) {
    // Create a new anchor element
    var anchor = document.createElement('a');

    // Set the href attribute to the document link
    anchor.href = link.url;

    // Set the target attribute to '_blank' to open the link in a new tab
    anchor.target = '_blank';

    // Set the text content of the anchor element to the document title
    anchor.textContent = link.title;

    // Append the anchor element to the thumbnails section
    thumbnailsSection.appendChild(anchor);

    // Add a line break after each anchor element
    thumbnailsSection.appendChild(document.createElement('br'));
});
