$(document).ready(function() {

    // Add a search results container to the edit-part form if it doesn't exist
    if ($('#edit-part #search-results-container').length === 0) {
        $('#edit-part').append('<div id="search-results-container" style="margin-top: 20px; background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 5px;"></div>');
        console.log("Added search-results-container to edit-part form");
    }



// Enhanced table styling function with debugging and direct DOM manipulation
function addTableStyling() {
    console.log("Running enhanced table styling function");

    // First, remove any existing styles to avoid conflicts
    $('#edit-part-table-styles').remove();

    // Define super-specific styles with !important flags
    const tableStyles = `
        /* Table container */
        #edit-part #search-results-container {
            background-color: transparent !important;
            color: white !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 9999 !important;
        }
        
        /* Table element */
        #edit-part #search-results-container table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin-top: 15px !important;
            color: white !important;
            border: 1px solid #444 !important;
            background-color: transparent !important;
            display: table !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Table headers */
        #edit-part #search-results-container th {
            background-color: rgba(0, 0, 40, 0.7) !important;
            padding: 10px !important;
            text-align: left !important;
            border-bottom: 2px solid #555 !important;
            color: white !important;
            font-weight: bold !important;
            display: table-cell !important;
            visibility: visible !important;
        }
        
        /* Table cells */
        #edit-part #search-results-container td {
            padding: 10px !important;
            border-bottom: 1px solid #444 !important;
            color: white !important;
            display: table-cell !important;
            visibility: visible !important;
        }
        
        /* Table rows */
        #edit-part #search-results-container tr {
            display: table-row !important;
            visibility: visible !important;
        }
        
        /* Alternating row colors */
        #edit-part #search-results-container tbody tr:nth-child(odd) {
            background-color: rgba(0, 0, 0, 0.3) !important;
        }
        
        #edit-part #search-results-container tbody tr:nth-child(even) {
            background-color: rgba(3, 46, 102, 0.3) !important;
        }
        
        /* Edit buttons */
        #edit-part #search-results-container .edit-part-btn {
            background-color: #007bff !important;
            color: white !important;
            border: none !important;
            padding: 5px 10px !important;
            border-radius: 3px !important;
            cursor: pointer !important;
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        #edit-part #search-results-container .edit-part-btn:hover {
            background-color: #0056b3 !important;
        }

        /* Search results paragraph */
        #edit-part #search-results-container p {
            color: yellow !important;
            background-color: rgba(3, 46, 102, 0.3) !important;
            padding: 10px !important;
            border-radius: 5px !important;
            margin-bottom: 15px !important;
            border-left: 4px solid #007bff !important;
            display: block !important;
            visibility: visible !important;
        }
    `;

    // Add the styles to the page with a new ID
    $('head').append('<style id="edit-part-table-styles">' + tableStyles + '</style>');
    console.log("Added enhanced styles to head");

    // Force direct DOM manipulation with a slight delay to ensure the table has rendered
    setTimeout(function() {
        console.log("Starting direct DOM manipulation");

        // Make table container visible
        $('#edit-part #search-results-container').css({
            'background-color': 'transparent',
            'color': 'white',
            'display': 'block',
            'visibility': 'visible',
            'opacity': '1',
            'z-index': '9999'
        });

        // Debug - log what's in the container
        console.log("Container contents:", $('#edit-part #search-results-container').html());

        // Force table to be visible
        $('#edit-part #search-results-container table').css({
            'width': '100%',
            'border-collapse': 'collapse',
            'color': 'white',
            'border': '1px solid #444',
            'display': 'table',
            'visibility': 'visible',
            'opacity': '1',
            'background-color': 'transparent'
        });

        // Table headers
        $('#edit-part #search-results-container th').css({
            'background-color': 'rgba(0, 0, 40, 0.7)',
            'padding': '10px',
            'text-align': 'left',
            'border-bottom': '2px solid #555',
            'color': 'white',
            'font-weight': 'bold',
            'display': 'table-cell',
            'visibility': 'visible'
        });

        // Table cells
        $('#edit-part #search-results-container td').css({
            'padding': '10px',
            'border-bottom': '1px solid #444',
            'color': 'white',
            'display': 'table-cell',
            'visibility': 'visible'
        });

        // Table rows
        $('#edit-part #search-results-container tr').css({
            'display': 'table-row',
            'visibility': 'visible'
        });

        // Updated row backgrounds with better contrast for white text
        $('#edit-part #search-results-container table tr:nth-child(odd)').css({
            'background-color': 'rgba(20, 20, 40, 0.9)',
            'color': 'white'
        });

        $('#edit-part #search-results-container table tr:nth-child(even)').css({
            'background-color': 'rgba(3, 46, 102, 0.9)',
            'color': 'white'
        });

        // Also try direct attribute setting with !important
        $('#edit-part #search-results-container table tr:nth-child(odd)').attr('style', 'background-color: rgba(20, 20, 40, 0.9) !important; color: white !important;');
        $('#edit-part #search-results-container table tr:nth-child(even)').attr('style', 'background-color: rgba(3, 46, 102, 0.9) !important; color: white !important;');

        // Ensure all text in cells is white
        $('#edit-part #search-results-container table td').css({
            'color': 'white'
        }).attr('style', 'color: white !important;');

        // Edit buttons
        $('#edit-part #search-results-container .edit-part-btn').css({
            'background-color': '#007bff',
            'color': 'white',
            'border': 'none',
            'padding': '5px 10px',
            'border-radius': '3px',
            'cursor': 'pointer',
            'display': 'inline-block',
            'visibility': 'visible',
            'opacity': '1'
        });

        // Caption paragraph
        $('#edit-part #search-results-container p').first().css({
            'color': 'yellow',
            'background-color': 'rgba(3, 46, 102, 0.3)',
            'padding': '10px',
            'border-radius': '5px',
            'margin-bottom': '15px',
            'border-left': '4px solid #007bff',
            'display': 'block',
            'visibility': 'visible'
        });

        // Force a repaint by toggling a class
        $('#edit-part #search-results-container').addClass('force-repaint');
        setTimeout(function() {
            $('#edit-part #search-results-container').removeClass('force-repaint');
        }, 50);

        console.log("Direct DOM manipulation completed");

        // Add extra debug info
        $('#edit-part #search-results-container').append('<div style="margin-top:20px; color:yellow; font-size:12px;">Table styling applied at: ' + new Date().toLocaleTimeString() + '</div>');
    }, 300);

    // Add additional debug button
    if ($('#debug-table-btn').length === 0) {
        $('#edit-part').append('<button id="debug-table-btn" style="position:fixed; bottom:10px; right:10px; z-index:10000; background:#ff0000; color:white; padding:5px 10px; border:none; border-radius:3px;">Debug Table</button>');

        $(document).on('click', '#debug-table-btn', function() {
            console.log("Debug button clicked");

            // Force table styling again
            addTableStyling();

            // Check if table exists and output contents
            const tableExists = $('#edit-part #search-results-container table').length > 0;
            console.log("Table exists:", tableExists);

            if (tableExists) {
                console.log("Table HTML:", $('#edit-part #search-results-container table').prop('outerHTML'));
                console.log("Table rows:", $('#edit-part #search-results-container tr').length);
                console.log("Table cells:", $('#edit-part #search-results-container td').length);
            }

            // Try an extreme approach - clone the table and replace it
            const tableHTML = $('#edit-part #search-results-container table').prop('outerHTML');
            if (tableHTML) {
                $('#edit-part #search-results-container table').remove();
                $('#edit-part #search-results-container').append(tableHTML);
                console.log("Table cloned and replaced");
            }

            // Re-apply styling
            addTableStyling();
        });
    }
}

    // Add table styling immediately
    addTableStyling();

    // Toggle advanced search
    $('#toggleFormBtn').on('click', function() {
        $('#advancedSearchForm').toggle();
    });

    // AJAX handler for search results - part search
    $('#search-part-form').on('submit', function(e) {
        e.preventDefault();
        let query = $('#search_query').val();

        // Check which form container is currently active
        let inEditPart = $('#edit-part').is(':visible') || window.location.hash === '#edit-part';

        console.log("Part search form submitted. In edit part:", inEditPart);

        if (inEditPart) {
            // We're in the edit part section

            // Hide the edit form container
            $('#edit-part-form').hide();

            // Show loading indicator in the search results container within edit-part
            $('#edit-part #search-results-container').html('<p style="color:white;">Loading...</p>').show();

            // Make sure edit part container stays visible
            $('#edit-part').show();

            // Hide the main results container
            $('#results-container').hide();

            $.ajax({
                url: '/search_part_ajax',
                type: 'GET',
                data: { search_query: query },
                success: function(html) {
                    // Update the search results container within edit part
                    $('#edit-part #search-results-container').html(html).show();
                    console.log("Search results loaded for edit part");

                    // Apply styling to the newly added table
                    addTableStyling();

                    // Make sure edit part container stays visible
                    $('#edit-part').show();
                },
                error: function() {
                    $('#edit-part #search-results-container').html('<p style="color:red;">An error occurred.</p>');
                }
            });
        } else {
            // Standard search results - use the main results container

            // Show loading indicator in the results container
            $('#searchResults').html('<p style="color:white;">Loading...</p>');

            // Hide the edit form container
            $('#edit-part-form').hide();

            // Show the results container
            $('#results-container').show();

            $.ajax({
                url: '/search_part_ajax',
                type: 'GET',
                data: { search_query: query },
                success: function(html) {
                    // Update the results container
                    $('#searchResults').html(html);

                    // Make sure results container is visible
                    $('#results-container').show();

                    // Hide all form containers in the main content
                    $('.form-container').hide();
                },
                error: function() {
                    $('#searchResults').html('<p style="color:red;">An error occurred.</p>');
                    $('#results-container').show();
                }
            });
        }
    });

    // AJAX handler for BOM advanced search form
    $('#advancedSearchForm form').on('submit', function(e) {
        e.preventDefault();

        // Show loading indicator in the results container
        $('#searchResults').html('<p style="color:white;">Loading BOM results...</p>');

        // Show the results container
        $('#results-container').show();

        // Hide all form containers in the main content
        $('.form-container').hide();

        // Get form data and send request
        var formData = new FormData(this);

        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(html) {
                // Update the results container
                $('#searchResults').html(html);
                $('#results-container').show();
            },
            error: function() {
                $('#searchResults').html('<p style="color:red;">An error occurred during BOM search.</p>');
                $('#results-container').show();
            }
        });
    });

    // AJAX handler for BOM general search form
    $('#generalSearch form').on('submit', function(e) {
        e.preventDefault();

        // Show loading indicator in the results container
        $('#searchResults').html('<p style="color:white;">Loading BOM results...</p>');

        // Show the results container
        $('#results-container').show();

        // Hide all form containers in the main content
        $('.form-container').hide();

        // Get form data and send request
        var formData = new FormData(this);

        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(html) {
                // Update the results container
                $('#searchResults').html(html);
                $('#results-container').show();
            },
            error: function() {
                $('#searchResults').html('<p style="color:red;">An error occurred during BOM search.</p>');
                $('#results-container').show();
            }
        });
    });

    // Handle edit button clicks from search results
    $(document).on('click', '.edit-part-btn', function() {
        var partId = $(this).data('part-id');
        if (!partId) {
            console.error('No part ID found');
            return;
        }

        // Set hash to edit-part to maintain correct context
        if (history.pushState) {
            history.pushState(null, null, '#edit-part');
        } else {
            location.hash = '#edit-part';
        }

        // Hide the results container
        $('#results-container').hide();

        // Hide the advanced search form
        $('#advancedSearchForm').hide();

        // Show the edit part container
        $('#edit-part').show();

        // Show loading indicator in the edit form container
        $('#edit-part-form').html('<p style="color:white;">Loading edit form...</p>').show();

        $.ajax({
            url: '/edit_part_ajax/' + partId,
            type: 'GET',
            cache: false,
            success: function(data) {
                // Update the edit form container
                $('#edit-part-form').html(data);

                // Add AJAX flag to the form
                var form = $('#edit-part-form form');
                if (form.length > 0) {
                    form.append('<input type="hidden" name="ajax" value="true">');
                    form.attr('id', 'ajax-edit-form');

                    // Add a cancel button if not present
                    if ($('#cancel-edit-btn').length === 0) {
                        form.find('button[type="submit"], input[type="submit"]').after(
                            '<button type="button" class="btn btn-secondary ml-2" id="cancel-edit-btn">Cancel</button>'
                        );
                    }
                }

                // Scroll to the edit form
                $('html, body').animate({
                    scrollTop: $('#edit-part-form').offset().top - 50
                }, 500);
            },
            error: function(xhr, status, error) {
                console.error('Error loading edit form:', error);
                $('#edit-part-form').html(
                    '<div class="alert alert-danger">Error loading part details: ' + error + '</div>'
                );
            }
        });
    });

    // Handle edit form submission
    $(document).on('submit', '#ajax-edit-form', function(e) {
        e.preventDefault();
        var form = $(this);
        var formData = new FormData(this);

        $.ajax({
            url: form.attr('action'),
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#edit-part-form').prepend(
                    '<div class="alert alert-success">Part updated successfully!</div>'
                );

                // Update search results if there was a previous search
                let query = $('#search_query').val();
                if (query) {
                    $.ajax({
                        url: '/search_part_ajax',
                        type: 'GET',
                        data: { search_query: query },
                        success: function(data) {
                            // Update the appropriate results container for edit part
                            $('#edit-part #search-results-container').html(data);

                            // Apply styling to the updated table
                            addTableStyling();
                        }
                    });
                }
            },
            error: function(xhr) {
                var errorMessage = xhr.responseJSON?.message || 'Error updating part';
                $('#edit-part-form').prepend(
                    '<div class="alert alert-danger">' + errorMessage + '</div>'
                );
            }
        });
    });

    // Handle cancel button - return to search results
    $(document).on('click', '#cancel-edit-btn', function() {
        // Hide edit form
        $('#edit-part-form').html('').hide();

        // Show the edit part form
        $('#edit-part').show();

        // If there are search results, show them
        if ($('#edit-part #search-results-container').children().length > 0) {
            $('#edit-part #search-results-container').show();
        }
    });

    // Handle sidebar navigation links with hash changes
    function handleHashChange() {
        var hash = window.location.hash.substring(1);
        if (hash) {
            // Hide all containers
            $('.form-container').hide();
            $('#results-container').hide();
            $('#edit-part-form').hide();

            // Hide advanced search form
            $('#advancedSearchForm').hide();

            // Show the requested container
            $('#' + hash).show();

            // If we're switching to edit-part, make sure its search results container is visible if it has content
            if (hash === 'edit-part' && $('#edit-part #search-results-container').children().length > 0) {
                $('#edit-part #search-results-container').show();
            }
        } else {
            // Default to showing search-bill-of-materials
            $('.form-container').hide();
            $('#results-container').hide();
            $('#edit-part-form').hide();
            $('#search-bill-of-materials').show();
        }
    }

    // Initial hash handling
    handleHashChange();

    // Listen for hash changes
    $(window).on('hashchange', handleHashChange);
});