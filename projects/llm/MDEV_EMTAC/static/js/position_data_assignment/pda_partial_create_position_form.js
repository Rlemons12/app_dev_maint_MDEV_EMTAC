document.addEventListener('DOMContentLoaded', function () {
    /**
     * Toggle between the existing entry (dropdown) and the new entry input.
     * Assumes that for a given field, there are two elements:
     * - An "existing" container with id: <fieldName>_existing
     * - A "new" container with id: <fieldName>_new
     *
     * @param {string} fieldName - The base name of the field (e.g., "area").
     */
    function toggleNewEntry(fieldName) {
        const existingDiv = document.getElementById(fieldName + "_existing");
        const newDiv = document.getElementById(fieldName + "_new");
        if (!existingDiv || !newDiv) {
            console.warn(`Elements for field ${fieldName} not found.`);
            return;
        }

        // Toggle visibility: if the existing div is visible, hide it and show the new entry div.
        if (existingDiv.style.display === "none" || newDiv.style.display === "block") {
            // Hide new entry and show existing dropdown
            existingDiv.style.display = "block";
            newDiv.style.display = "none";
            // Optionally clear the new input(s)
            const inputs = newDiv.querySelectorAll("input, textarea");
            inputs.forEach(input => input.value = "");
        } else {
            // Hide existing dropdown and show new entry
            existingDiv.style.display = "none";
            newDiv.style.display = "block";
        }
    }

    /**
     * Optionally, handle form submission.
     * This simple handler shows a loading spinner when the form is submitted.
     * You can extend this to perform an AJAX submission if needed.
     */
    const form = document.getElementById('positionForm');
    if (form) {
        form.addEventListener('submit', function () {
            const spinner = document.getElementById('loadingSpinner');
            if (spinner) {
                spinner.style.display = "block";
            }
            // If you want to use AJAX, you would prevent the default here and use fetch.
            // For a standard submission, let the browser handle it.
        });
    } else {
        console.warn('Form with id "positionForm" not found.');
    }

    // Expose the toggleNewEntry function globally if you need to call it from inline HTML.
    window.toggleNewEntry = toggleNewEntry;
});
