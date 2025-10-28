$(document).ready(function() {
    console.log("Document search debug initialized");

    // Check if elements exist
    console.log("Search input exists:", $('#search-documents').length > 0);
    console.log("Suggestion box exists:", $('#document-suggestion-box').length > 0);

    // Add test data to verify rendering works
    function testSuggestionBoxRendering() {
        const suggestionBox = $('#document-suggestion-box');
        suggestionBox.empty();

        const testData = [
            { id: 999, title: "Test Document", rev: "1.0" }
        ];

        testData.forEach(doc => {
            const docEntry = $(`
                <div class="suggestion-item" data-document-id="${doc.id}">
                    <strong>Title:</strong> ${doc.title}<br>
                    <strong>Revision:</strong> ${doc.rev}
                </div>
            `);
            suggestionBox.append(docEntry);
        });

        suggestionBox.show();
        console.log("Test rendering complete - check if suggestion box is visible");
    }

    // Bind a click event to test rendering
    $('#search-documents').on('click', function() {
        console.log("Search input clicked");
        testSuggestionBoxRendering();
    });

    // Rebind the input event with debug logging
    $('#search-documents').on('input', function() {
        const searchInput = $(this).val().trim();
        console.log("Input detected:", searchInput);

        clearTimeout(window.searchDebounceTimer);
        window.searchDebounceTimer = setTimeout(function() {
            console.log("Debounce timer fired, searching for:", searchInput);
            // Your normal search function call
        }, 300);
    });
});