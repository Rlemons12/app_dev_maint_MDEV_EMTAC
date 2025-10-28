document.addEventListener('DOMContentLoaded', function() {
    // Drag and drop functionality
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        fileInput.files = files;
        // Trigger form submission if required
        document.getElementById('upload-compare-form').submit();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            // Show file name or trigger form submission
            dropZone.textContent = fileInput.files[0].name;
            document.getElementById('upload-compare-form').submit();
        }
    });
});
