document.addEventListener('DOMContentLoaded', function () {

    // Function to search for images
    function searchImages() {
        const searchInputElement = document.getElementById('images-search');
        const searchInput = searchInputElement.value.trim();
        const suggestionBox = document.getElementById('image-suggestion-box');

        console.log('Search input:', searchInput);

        if (!searchInput) {
            suggestionBox.innerHTML = '';
            suggestionBox.style.cssText = 'display: none !important;';
            return;
        }

        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();

        const fetchUrl = `/pda_search_images?query=${encodeURIComponent(searchInput)}&t=${timestamp}`;
        console.log('Fetching URL:', fetchUrl);

        fetch(fetchUrl, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log('Received response:', response);
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            suggestionBox.innerHTML = '';  // Clear previous results

            if (Array.isArray(data) && data.length > 0) {
                data.forEach((image, index) => {
                    console.log(`Processing image ${index}:`, image);
                    const imageEntry = document.createElement('div');
                    imageEntry.className = 'suggestion-item';

                    // Display both title and description
                    imageEntry.innerHTML = `
                        <div>
                            <strong>Title:</strong> ${image.title}<br>
                            <strong>Description:</strong> ${image.description}
                        </div>
                    `;

                    imageEntry.addEventListener('click', function () {
                        addImageToPosition(image.id, image.title);
                        suggestionBox.style.cssText = 'display: none !important;';
                        searchInputElement.value = '';
                    });
                    suggestionBox.appendChild(imageEntry);
                });

                // Apply aggressive styling to make dropdown visible
                suggestionBox.style.cssText = 'display: block !important; z-index: 999999 !important; visibility: visible !important; opacity: 1 !important; position: absolute !important; top: 100% !important; left: 0 !important; width: 100% !important; background-color: rgba(0, 0, 0, 0.95) !important; border: 3px solid yellow !important; color: yellow !important; max-height: 300px !important; overflow-y: auto !important;';

                // Force a browser reflow/repaint
                const forceRepaint = suggestionBox.offsetHeight;

                console.log('Setting image suggestion box display to visible:', suggestionBox);
            } else {
                console.log('No images found for search input:', searchInput);
                suggestionBox.innerHTML = '<p>No images found.</p>';

                // Apply aggressive styling to make dropdown visible
                suggestionBox.style.cssText = 'display: block !important; z-index: 999999 !important; visibility: visible !important; opacity: 1 !important; position: absolute !important; top: 100% !important; left: 0 !important; width: 100% !important; background-color: rgba(0, 0, 0, 0.95) !important; border: 3px solid yellow !important; color: yellow !important; max-height: 300px !important; overflow-y: auto !important;';

                // Force a browser reflow/repaint
                const forceRepaint = suggestionBox.offsetHeight;

                console.log('Setting image suggestion box display to visible (no results):', suggestionBox);
            }
        })
        .catch(error => {
            alert('Error searching images: ' + error.message);
            console.error('Error searching images:', error);
        });
    }

    // Function to add image to position from search result
    function addImageToPosition(imageId, imageName) {
        const positionId = document.getElementById('position_id').value;
        console.log(`Adding image ${imageId} to position ${positionId}`);

        if (!imageId || !positionId) {
            alert('Image ID and Position ID are required.');
            console.error('Missing imageId or positionId:', { imageId, positionId });
            return;
        }

        fetch('/add_image_to_position', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_id: imageId, position_id: positionId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);
                console.log(`Image '${imageName}' added to position successfully.`);

                // Add the image to existing images list
                const existingImagesList = document.getElementById('existing-images-list');
                const newImageEntry = document.createElement('div');
                newImageEntry.className = 'existing-image';

                // Create span for image info
                const imageNameSpan = document.createElement('span');
                imageNameSpan.textContent = `Title: ${imageName}`;

                // Create remove button
                const removeButton = document.createElement('button');
                removeButton.type = 'button';
                removeButton.textContent = 'Remove';
                removeButton.className = 'remove-existing-image-button';
                removeButton.setAttribute('data-image-id', imageId);

                // Append to newImageEntry
                newImageEntry.appendChild(imageNameSpan);
                newImageEntry.appendChild(removeButton);

                existingImagesList.appendChild(newImageEntry);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
                console.error(`Error adding image '${imageName}' to position:`, data);
            }
        })
        .catch(error => {
            alert('Error adding image: ' + error.message);
            console.error('Error adding image:', error);
        });
    }

    // Function to remove an existing image
    function removeExistingImage(button, imageId, positionId) {
        if (confirm('Are you sure you want to remove this image?')) {
            fetch('/remove_image_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_id: imageId, position_id: positionId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
                    button.parentNode.remove();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Failed to remove image: ' + error.message);
                console.error('Failed to remove image:', error);
            });
        }
    }

    // Add event listener to the search input
    const imagesSearchInput = document.getElementById('images-search');
    if (imagesSearchInput) {
        imagesSearchInput.addEventListener('keyup', searchImages);
    } else {
        console.error('Images Search Input not found.');
    }

    // Add event listener to remove buttons
    document.addEventListener('click', function(event) {
        if (event.target && event.target.matches('.remove-existing-image-button')) {
            const imageId = event.target.getAttribute('data-image-id');
            const positionId = document.getElementById('position_id').value;
            removeExistingImage(event.target, imageId, positionId);
        }
    });

    // Add event listener for "Add Another Image" button
    const addImageButton = document.getElementById('add-image-button');
    if (addImageButton) {
        addImageButton.addEventListener('click', function() {
            // Add image entry logic here
        });
    }

    // Add event listener for "Submit New Images" button
    const submitImagesButton = document.getElementById('submit-images-button');
    if (submitImagesButton) {
        submitImagesButton.addEventListener('click', function() {
            // Submit new images logic here
        });
    }
});