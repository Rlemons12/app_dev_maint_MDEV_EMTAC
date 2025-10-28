document.getElementById('commentPopupLink').addEventListener('click', function() {
    document.getElementById('commentPopup').style.display = 'block';
});

document.getElementById('closePopup').addEventListener('click', function() {
    document.getElementById('commentPopup').style.display = 'none';
});

document.getElementById('commentForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('comment', document.getElementById('comment').value);
    formData.append('page_url', window.location.href);

    // Check if a file was uploaded
    const uploadedFile = document.querySelector('input[type="file"]').files[0];
    if (uploadedFile) {
        formData.append('screenshot', uploadedFile);
    }

    // Check if there is a pasted image (base64)
    const base64Image = document.getElementById('imageData').value;
    if (base64Image) {
        formData.append('imageData', base64Image);  // Add base64 image data to form
    }

    fetch('/submit-comment', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert('Comment submitted successfully');
        document.getElementById('commentPopup').style.display = 'none';
    })
    .catch(error => {
        alert('Error submitting comment');
        console.error('Error:', error);
    });
});

// Handle paste events for images
document.getElementById('comment').addEventListener('paste', function(event) {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") === 0) {
            const file = items[i].getAsFile();
            const reader = new FileReader();
            reader.onload = function(event) {
                // Store the base64 image in a hidden input field
                document.getElementById('imageData').value = event.target.result;

                // Show feedback that the image was added
                document.getElementById('pastedImageFeedback').style.display = 'block';

                // Optionally show a preview of the pasted image
                document.getElementById('pastedImagePreview').src = event.target.result;
                document.getElementById('pastedImagePreview').style.display = 'block';
            };
            reader.readAsDataURL(file);  // Convert image to base64
        }
    }
});
