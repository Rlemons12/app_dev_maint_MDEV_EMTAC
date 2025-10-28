document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed for image comparison.');

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-compare-form');
    const uploadedImagePreview = document.getElementById('uploaded-image-preview');
    const uploadedImage = document.getElementById('uploaded-image');

    if (!dropZone || !fileInput || !uploadForm || !uploadedImagePreview || !uploadedImage) {
        console.error('Required elements not found.');
        return;
    }

    console.log('Required elements found. Setting up event listeners.');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        console.log('Drag over event.');
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        console.log('Drag leave event.');
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        console.log('Drop event.');
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            console.log('Files dropped:', files);
            fileInput.files = files;
            displayUploadedImage(files[0]);
            console.log('Submitting form after file drop.');
            handleSubmit();
        } else {
            console.log('No files dropped.');
        }
    });

    dropZone.addEventListener('click', () => {
        console.log('Drop zone clicked.');
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            console.log('File input changed:', fileInput.files);
            displayUploadedImage(fileInput.files[0]);
            console.log('Submitting form after file input change.');
            handleSubmit();
        } else {
            console.log('No files selected.');
        }
    });

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the default form submission
        console.log('Form submit event.');
        handleSubmit();
    });

    function displayUploadedImage(file) {
        // Display the uploaded image in the preview area
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImagePreview.style.display = 'block';

            // Add a label to make it clear this is the query image
            dropZone.innerHTML = 'Image uploaded: ' + file.name;
            console.log('Image displayed:', file.name);
        };
        reader.readAsDataURL(file);
    }

    function handleSubmit() {
        const formData = new FormData(uploadForm);
        fetch('/upload_and_compare', { // AJAX request
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Response received from server.');
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Error from server:', data.error);
                showResults(`<p>${data.error}</p>`);
            } else {
                console.log('Rendering comparison results.');
                renderComparisonResults(data.image_similarity_search);
            }
        })
        .catch(error => {
            console.error('Error during fetch operation:', error);
            showResults(`<p>An error occurred while processing the image.</p>`);
        });
    }

    function renderComparisonResults(results) {
        let imagesList = '<h3 style="color: yellow; margin-top: 30px;">Similar Images</h3>';
        results.forEach(image => {
            console.log('Rendering result for image:', image);
            imagesList += `<div class="image-details">
                            <a href="/uploads/${image.file_path}" target="_blank">
                                <img class="thumbnail" src="/uploads/${image.file_path}" alt="${image.title}">
                            </a>
                            <div class="description">
                                <h2>${image.title}</h2>
                                <p>${image.description}</p>
                                <p>Similarity: ${image.similarity.toFixed(2)}</p>
                            </div>
                            <div style="clear: both;"></div>
                          </div>`;
        });
        console.log('Comparison results prepared.');
        document.getElementById('results').style.display = 'block';
        document.getElementById('image-compare-results').innerHTML = imagesList;
        document.getElementById('documents-list').innerHTML = ''; // Clear document results
    }

    function showResults(content) {
        console.log('Showing results.');
        document.getElementById('results').style.display = 'block';
        document.getElementById('documents-list').innerHTML = content;
        document.getElementById('image-results').innerHTML = ''; // Clear image results
        document.getElementById('image-compare-results').innerHTML = ''; // Clear image compare results
    }
});