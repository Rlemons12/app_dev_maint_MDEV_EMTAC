// Function to search for problems to edit
function searchForProblem() {
    // Capture the search query and the dropdown values
    var searchQuery = $('#searchProblem').val();
    var areaId = $('#tsg_edit_areaDropdown').val();
    var equipmentGroupId = $('#tsg_edit_equipmentGroupDropdown').val();
    var modelId = $('#tsg_edit_modelDropdown').val();
    var assetNumberId = $('#tsg_edit_assetNumberDropdown').val();
    var locationId = $('#tsg_edit_locationDropdown').val();

    // Show a loading spinner (optional)
    $('#searchResults').html('<li>Loading...</li>');

    $.ajax({
        url: '/get_troubleshooting_guide_edit_data',  // The backend route
        method: 'GET',
        data: {
            query: searchQuery,               // Problem title or ID
            area: areaId,                     // Area ID from dropdown
            equipment_group: equipmentGroupId, // Equipment Group ID from dropdown
            model: modelId,                   // Model ID from dropdown
            asset_number: assetNumberId,       // Asset Number ID from dropdown
            location: locationId              // Location ID from dropdown
        },
        success: function(data) {
            // Clear previous results
            $('#searchResults').empty();

            // Populate the search results with problems
            if (data.problems.length > 0) {
                data.problems.forEach(function(problem) {
                    $('#searchResults').append(
                        '<li onclick="loadProblemSolution(' + problem.id + ')">' + problem.name + '</li>'
                    );
                });
            } else {
                $('#searchResults').append('<li>No results found</li>');
            }
        },
        error: function(xhr) {
            alert('Failed to search for problem: ' + xhr.responseText);
        }
    });
}

// Function to load the selected problem into the form
function loadProblemSolution(problemId) {
    // Disable the form to prevent multiple requests
    $('#editProblemSolutionForm button').prop('disabled', true);

    $.ajax({
        url: '/get_problem_solution_data/' + problemId,
        method: 'GET',
        success: function(data) {
            // Show the form for editing
            $('#editProblemSolutionForm').show();

            // Populate the form fields with the fetched data
            $('#edit_problem_id').val(data.problem.id);
            $('#edit_problem_name').val(data.problem.name);
            $('#edit_problem_description').val(data.problem.description);
            $('#edit_solution_id').val(data.solution.id);
            $('#edit_solution_description').val(data.solution.description);

            // Clear and populate the Associated Problem Images dropdown
            $('#edit_problem_imageDropdown').empty();
            data.problem.images.forEach(function(image) {
                $('#edit_problem_imageDropdown').append('<option value="' + image.id + '" selected>' + image.title + '</option>');
            });

            // Clear and populate the Associated Solution Images dropdown
            $('#edit_solution_imageDropdown').empty();
            data.solution.images.forEach(function(image) {
                $('#edit_solution_imageDropdown').append('<option value="' + image.id + '" selected>' + image.title + '</option>');
            });

            // Clear and populate the Associated Documents dropdown
            $('#edit_documentDropdown').empty();
            data.problem.documents.forEach(function(document) {
                $('#edit_documentDropdown').append('<option value="' + document.id + '" selected>' + document.title + '</option>');
            });

            // Clear and populate the Associated Parts dropdown
            $('#edit_partDropdown').empty();
            data.problem.parts.forEach(function(part) {
                $('#edit_partDropdown').append('<option value="' + part.id + '" selected>' + part.name + '</option>');
            });

            // Clear and populate the Associated Drawing Numbers dropdown
            $('#edit_drawingdropdown').empty();
            data.problem.drawings.forEach(function(drawing) {
                $('#edit_drawingdropdown').append('<option value="' + drawing.id + '" selected>' + drawing.number + '</option>');
            });

            // Enable the form after data is loaded
            $('#editProblemSolutionForm button').prop('disabled', false);
        },
        error: function(xhr) {
            alert('Failed to load problem/solution data: ' + xhr.responseText);
            $('#editProblemSolutionForm button').prop('disabled', false);
        }
    });
}
// Initialize Select2 for image search
$('#tsg_imageSearchDropdown').select2({
    ajax: {
        url: '/search_images',  // Endpoint for searching problem images
        dataType: 'json',
        delay: 250,
        data: function(params) {
            return { q: params.term };  // Search term entered by the user
        },
        processResults: function(data) {
            return {
                results: data.map(function(image) {
                    return {
                        id: image.id,
                        text: image.title  // Display image title
                    };
                })
            };
        },
        cache: true
    },
    minimumInputLength: 2,
    placeholder: 'Search for a problem image...'
});

// Add selected image from the search dropdown to the edit_problem_imageDropdown
$('#addImageButton').on('click', function() {
    var selectedImages = $('#tsg_imageSearchDropdown').select2('data');  // Get selected image data

    // Loop through selected images and add them to the edit_problem_imageDropdown
    selectedImages.forEach(function(image) {
        if ($('#edit_problem_imageDropdown option[value="' + image.id + '"]').length === 0) {
            var newOption = $('<option>', {
                value: image.id,
                text: image.text,  // Display image title
                selected: true  // Mark as selected when added
            });
            $('#edit_problem_imageDropdown').append(newOption);
        }
    });

    // Clear the search dropdown after adding the image
    $('#tsg_imageSearchDropdown').val(null).trigger('change');
});

// Remove selected images from the edit_problem_imageDropdown
$('#removeImageButton').on('click', function() {
    $('#edit_problem_imageDropdown option:selected').each(function() {
        $(this).remove();  // Remove the selected option
    });
});

// Ensure all images in the edit_problem_imageDropdown are selected before form submission
$('#editProblemSolutionForm').on('submit', function() {
    $('#edit_problem_imageDropdown option').prop('selected', true);  // Select all options before submission
});
// Initialize Select2 for solution image search
$('#tsg_solutionImageSearchDropdown').select2({
    ajax: {
        url: '/search_solution_images',  // Endpoint for searching solution images
        dataType: 'json',
        delay: 250,
        data: function(params) {
            return { q: params.term };  // Search term entered by the user
        },
        processResults: function(data) {
            return {
                results: data.map(function(image) {
                    return {
                        id: image.id,
                        text: image.title  // Display image title
                    };
                })
            };
        },
        cache: true
    },
    minimumInputLength: 2,
    placeholder: 'Search for a solution image...'
});

// Add selected solution image from the search dropdown to the edit_solution_imageDropdown
$('#addSolutionImageButton').on('click', function() {
    var selectedSolutionImages = $('#tsg_solutionImageSearchDropdown').select2('data');  // Get selected solution image data

    // Loop through selected solution images and add them to the edit_solution_imageDropdown
    selectedSolutionImages.forEach(function(image) {
        if ($('#edit_solution_imageDropdown option[value="' + image.id + '"]').length === 0) {
            var newOption = $('<option>', {
                value: image.id,
                text: image.text,  // Display image title
                selected: true  // Mark as selected when added
            });
            $('#edit_solution_imageDropdown').append(newOption);
        }
    });

    // Clear the search dropdown after adding the solution image
    $('#tsg_solutionImageSearchDropdown').val(null).trigger('change');
});

// Remove selected solution images from the edit_solution_imageDropdown
$('#removeSolutionImageButton').on('click', function() {
    $('#edit_solution_imageDropdown option:selected').each(function() {
        $(this).remove();  // Remove the selected option
    });
});

// Ensure all solution images in the edit_solution_imageDropdown are selected before form submission
$('#editProblemSolutionForm').on('submit', function() {
    $('#edit_solution_imageDropdown option').prop('selected', true);  // Select all options before submission
});

// Function to search for additional documents
$('#search_document').on('input', function () {
    var searchQuery = $(this).val();
    if (searchQuery.length > 2) {  // Trigger search only if input length > 2
        // Disable the search field while fetching data
        $('#search_document').prop('disabled', true);

        $.ajax({
            url: '/search_documents',  // Backend route for searching documents
            method: 'GET',
            data: { query: searchQuery },
            success: function (data) {
                // Clear previous search results
                $('#documentSearchResults').empty();

                // Populate the search results with documents
                if (data.documents.length > 0) {
                    data.documents.forEach(function (document) {
                        $('#documentSearchResults').append(
                            '<li><input type="checkbox" value="' + document.id + '">' + document.title + '</li>'
                        );
                    });
                } else {
                    $('#documentSearchResults').append('<li>No documents found</li>');
                }

                // Enable the search field after data is loaded
                $('#search_document').prop('disabled', false);
            },
            error: function (xhr) {
                alert('Error searching for documents: ' + xhr.responseText);
                $('#search_document').prop('disabled', false);
            }
        });
    }
});

// Function to add selected documents from search results to the dropdown
$('#addDocumentButton').on('click', function () {
    $('#documentSearchResults input:checked').each(function () {
        var docId = $(this).val();
        var docTitle = $(this).parent().text().trim();

        // Log the document being added
        console.log('Adding Document:', docId, docTitle);

        // Check if the document is already in the dropdown
        if ($('#edit_documentDropdown option[value="' + docId + '"]').length === 0) {
            // Add the document to the dropdown using jQuery's $('<option>')
            var newOption = $('<option>', {
                value: docId,
                text: docTitle,
                selected: true
            });
            $('#edit_documentDropdown').append(newOption);
            console.log('Document added:', docId);
        } else {
            console.log('Document already exists in dropdown:', docId);
        }
    });

    // Clear the search results after adding
    $('#documentSearchResults').empty();
});

// Function to remove selected documents from the dropdown
$('#deleteDocumentButton').on('click', function () {
    // Check if any document is selected for deletion
    if ($('#edit_documentDropdown option:selected').length > 0) {
        // Confirm with the user before deleting
        if (confirm('Are you sure you want to delete the selected document(s)?')) {
            $('#edit_documentDropdown option:selected').each(function () {
                var docId = $(this).val();
                var docTitle = $(this).text();
                console.log('Removing Document:', docId, docTitle);
                $(this).remove();  // Remove the selected option
            });
        }
    } else {
        alert('Please select at least one document to delete.');
    }
});

$(document).ready(function() {
    // Initialize Select2 for part search
    $('#tsg_partSearchDropdown').select2({
        placeholder: 'Search for part number...',
        minimumInputLength: 2,
        ajax: {
            url: '/search_parts',  // Endpoint for searching parts
            dataType: 'json',
            delay: 250,
            data: function(params) {
                return { q: params.term };  // Search term entered by the user
            },
            processResults: function(data) {
                return {
                    results: data.map(function(part) {
                        return {
                            id: part.id,
                            text: part.name  // Include part number if desired
                        };
                    })
                };
            },
            cache: true
        }
    });

    // Add selected parts from search dropdown to the edit_partDropdown
    $('#addPartButton').on('click', function() {
        var selectedParts = $('#tsg_partSearchDropdown').select2('data');

        selectedParts.forEach(function(part) {
            if ($('#edit_partDropdown option[value="' + part.id + '"]').length === 0) {
                var newOption = $('<option>', {
                    value: part.id,
                    text: part.text,
                    selected: true
                });
                $('#edit_partDropdown').append(newOption);
            }
        });

        // Clear the search dropdown after adding the part
        $('#tsg_partSearchDropdown').val(null).trigger('change');
    });

    // Remove selected parts from the edit_partDropdown
    $('#removePartButton').on('click', function() {
        $('#edit_partDropdown option:selected').each(function() {
            $(this).remove();
        });
    });

    // Ensure all parts are selected before form submission
    $('#editProblemSolutionForm').on('submit', function() {
        $('#edit_partDropdown option').prop('selected', true);
    });
});


// Initialize Select2 for drawing search
$('#tsg_drawingSearchDropdown').select2({
    ajax: {
        url: '/search_drawings',  // Endpoint for searching drawings
        dataType: 'json',
        delay: 250,
        data: function(params) {
            return { q: params.term };  // Search term entered by the user
        },
        processResults: function(data) {
            return {
                results: data.map(function(drawing) {
                    return {
                        id: drawing.id,
                        text: drawing.name  // Display drawing number and name
                    };
                })
            };
        },
        cache: true
    },
    minimumInputLength: 2,
    placeholder: 'Search for a drawing...'
});

// Add selected drawing from the search dropdown to the edit_drawingdropdown
$('#addDrawingButton').on('click', function() {
    var selectedDrawings = $('#tsg_drawingSearchDropdown').select2('data');  // Get selected drawing data

    // Loop through selected drawings and add them to the edit_drawingdropdown
    selectedDrawings.forEach(function(drawing) {
        if ($('#edit_drawingdropdown option[value="' + drawing.id + '"]').length === 0) {
            $('#edit_drawingdropdown').append(
                $('<option>', {
                    value: drawing.id,
                    text: drawing.text,  // Display drawing number and name
                    selected: true  // Mark as selected when added
                })
            );
        }
    });

    // Clear the search dropdown after adding the drawing
    $('#tsg_drawingSearchDropdown').val(null).trigger('change');
});

// Remove selected drawings from the edit_drawingdropdown
$('#removeDrawingButton').on('click', function() {
    $('#edit_drawingdropdown option:selected').each(function() {
        $(this).remove();  // Remove the selected option
    });
});

// Ensure all drawings in the edit_drawingdropdown are selected before form submission
$('#editProblemSolutionForm').on('submit', function() {
    $('#edit_drawingdropdown option').prop('selected', true);  // Select all options before submission
});
