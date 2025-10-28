$(document).ready(function () {
    // ToolManagement Module
    window.ToolManagement = {
        init: function () {
            console.log('%c[ToolManagement] Initializing module.', 'color: green; font-weight: bold;');
            try {
                this.cacheElements();
                this.bindEvents();

                // Try to find position ID on initialization
                this.positionId = this.findPositionId();
                console.log('[ToolManagement] positionId (from initialization):', this.positionId);

                // If valid, fetch associated tools immediately
                if (this.positionId) {
                    this.fetchAssociatedTools(this.positionId);
                } else {
                    console.warn('[ToolManagement] No position_id found during initialization.');
                }

                console.log('%c[ToolManagement] Module initialized successfully.', 'color: green;');
            } catch (error) {
                console.error('%c[ToolManagement] Initialization failed:', 'color: red; font-weight: bold;', error);
            }
        },

        // Find position ID using multiple methods
        findPositionId: function () {
            console.log('[ToolManagement] Searching for position ID with multiple methods');

            var positionId = '';

            // Method 1: From tool_position_id element
            if ($('#tool_position_id').length > 0) {
                positionId = $('#tool_position_id').val();
                console.log('[ToolManagement] Method 1 (tool_position_id):', positionId);
            }

            // Method 2: From position_id element
            if (!positionId && $('#position_id').length > 0) {
                positionId = $('#position_id').val();
                console.log('[ToolManagement] Method 2 (position_id):', positionId);
            }

            // Method 3: From URL parameter
            if (!positionId) {
                var urlParams = new URLSearchParams(window.location.search);
                positionId = urlParams.get('position_id');
                console.log('[ToolManagement] Method 3 (URL param):', positionId);
            }

            // Method 4: From global variable if it exists
            if (!positionId && typeof window.currentPositionId !== 'undefined') {
                positionId = window.currentPositionId;
                console.log('[ToolManagement] Method 4 (global variable):', positionId);
            }

            // Method 5: From localStorage
            if (!positionId && localStorage.getItem('currentPositionId')) {
                positionId = localStorage.getItem('currentPositionId');
                console.log('[ToolManagement] Method 5 (localStorage):', positionId);
            }

            // Method 6: From any element with data-position-id attribute
            if (!positionId && $('[data-position-id]').length > 0) {
                positionId = $('[data-position-id]').first().data('position-id');
                console.log('[ToolManagement] Method 6 (data attribute):', positionId);
            }

            console.log('[ToolManagement] Final position ID found:', positionId);
            return positionId;
        },

        // Method to set position ID
        setPositionId: function (id) {
            if (id) {
                this.positionId = id;
                console.log('[ToolManagement] Manually set position ID:', id);

                // Store in form elements
                if ($('#position_id').length > 0) {
                    $('#position_id').val(id);
                }
                if ($('#tool_position_id').length > 0) {
                    $('#tool_position_id').val(id);
                }

                // Store in global variable
                window.currentPositionId = id;

                // Store in localStorage for persistence
                localStorage.setItem('currentPositionId', id);

                return true;
            }
            return false;
        },

        cacheElements: function () {
            console.log('%c[ToolManagement] Caching DOM elements...', 'color: blue;');

            // Sub-tabs
            this.$subTabs = $('.sub-tab-item');
            this.$subTabContents = $('.sub-tab-content');

            // "Associated Tools" sub-tab
            this.$associatedToolsTableBody = $('#associatedToolsTableBody');
            this.$goToAddToolsBtn = $('#goToAddToolsBtn');

            // Buttons/Forms in the Add/Search Tools sub-tab
            this.$searchButton = $('#toolSearchBtn');
            this.$searchForm = $('#toolManagementForm');
            this.$searchResults = $('#toolSearchResults');

            console.log('%c[ToolManagement] Elements cached:', 'color: blue;', {
                subTabs: this.$subTabs.length,
                subTabContents: this.$subTabContents.length,
                searchButton: this.$searchButton.length,
                searchForm: this.$searchForm.length,
                searchResults: this.$searchResults.length
            });
        },

        bindEvents: function () {
            var self = this;
            console.log('%c[ToolManagement] Binding event listeners...', 'color: blue;');

            // Sub-tab click
            self.$subTabs.on('click', function () {
                var targetTab = $(this).data('subtab');
                console.log('%c[ToolManagement] Sub-tab clicked.', 'color: purple;', {
                    clickedTab: $(this).text(),
                    targetTabID: targetTab
                });

                self.$subTabs.removeClass('active');
                self.$subTabContents.removeClass('active').hide();

                // Activate clicked tab
                $(this).addClass('active');
                $('#' + targetTab).addClass('active').fadeIn();
            });

            // "Go to Add Tools" button
            self.$goToAddToolsBtn.on('click', function () {
                console.log('%c[ToolManagement] Going to Add/Search Tools sub-tab.', 'color: purple;');
                self.$subTabs.removeClass('active');
                self.$subTabContents.removeClass('active').hide();

                self.$subTabs.filter('[data-subtab="add-search-tools"]').addClass('active');
                $('#add-search-tools').addClass('active').fadeIn();
            });

            // Search button
            self.$searchButton.on('click', function (e) {
                console.log('%c[ToolManagement] Search button clicked.', 'color: purple;');
                e.preventDefault();
                self.performSearch();
            });

            // Search form "Enter" submission
            self.$searchForm.on('submit', function (e) {
                console.log('%c[ToolManagement] Search form submitted (Enter key).', 'color: purple;');
                e.preventDefault();
                self.performSearch();
            });

            // Listen for clicks on "Add to Position" buttons in the search results
            // We use event delegation because the table is generated dynamically.
            $(document).on('click', '.btn-add-tool', function () {
                var toolId = $(this).data('tool-id');
                console.log('[ToolManagement] "Add to Position" clicked for toolId:', toolId);
                self.addToolToPosition(toolId);
            });

            // Add this new event binding for tool search input
            $('#tool-search').on('keyup', function() {
                console.log('[ToolManagement] Tool search input changed');
                self.searchTools();
            });
        },

        // ------------------------------------
        // FETCH ASSOCIATED TOOLS
        // ------------------------------------
        fetchAssociatedTools: function (positionId) {
            var self = this;
            console.log('[ToolManagement] Fetching associated tools for position:', positionId);

            $.ajax({
                url: '/pda_get_tools_by_position',
                method: 'GET',
                data: { position_id: positionId },
                dataType: 'json',
                success: function (response) {
                    console.log('[ToolManagement] Associated tools response:', response);
                    if (response && response.tools) {
                        self.renderAssociatedTools(response.tools);
                    } else {
                        console.warn('[ToolManagement] No "tools" key in response');
                        self.$associatedToolsTableBody.html('<tr><td colspan="6">No associated tools found.</td></tr>');
                    }
                },
                error: function (xhr, status, error) {
                    console.error('[ToolManagement] Error fetching associated tools:', error);
                }
            });
        },

        renderAssociatedTools: function (tools) {
    console.log('[ToolManagement] Rendering associated tools:', tools);

    // Use the existing-tools-list element instead of the non-existent associatedToolsTableBody
    var $container = $('#existing-tools-list');

    // Clear existing tools
    $container.empty();

    if (!tools || tools.length === 0) {
        $container.html('<div class="no-tools-message">No associated tools found.</div>');
        return;
    }

    // Build tool entries to match the HTML structure in position_data_assignment_tool_management.html
    tools.forEach(function (tool) {
        var toolHtml = `
            <div class="existing-tool" id="tool-${tool.id}">
                <div class="tool-info">
                    <span class="tool-details">
                        Tool ID: ${tool.id}, 
                        Name: ${tool.name || 'N/A'}, 
                        Type: ${tool.type || 'N/A'}, 
                        Material: ${tool.material || 'N/A'}
                    </span>
                    <button type="button" class="remove-existing-tool-button"
                            data-tool-id="${tool.id}"
                            data-position-id="${window.ToolManagement.positionId}">Remove</button>
                </div>
            </div>
        `;
        $container.append(toolHtml);
    });

    // Bind events for remove buttons
    this.bindRemoveToolEvents();
},

// Add this new function to handle the remove tool functionality
bindRemoveToolEvents: function() {
    var self = this;
    $('.remove-existing-tool-button').off('click').on('click', function() {
        var toolId = $(this).data('tool-id');
        var positionId = $(this).data('position-id');

        if (confirm('Are you sure you want to remove this tool from the position?')) {
            self.removeToolFromPosition(toolId, positionId);
        }
    });
},

// Add this new function to remove tools
removeToolFromPosition: function(toolId, positionId) {
    var self = this;

    $.ajax({
        url: '/pda_remove_tool_from_position',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            tool_id: toolId,
            position_id: positionId
        }),
        success: function(response) {
            console.log('[ToolManagement] Tool removed successfully:', response);

            // Remove the tool element from the DOM
            $('#tool-' + toolId).fadeOut(300, function() {
                $(this).remove();

                // Check if there are no tools left
                if ($('#existing-tools-list').children().length === 0) {
                    $('#existing-tools-list').html('<div class="no-tools-message">No associated tools found.</div>');
                }
            });

            // Show success message
            alert(response.message || 'Tool removed successfully.');
        },
        error: function(xhr, status, error) {
            console.error('[ToolManagement] Error removing tool:', error);
            alert('Failed to remove tool: ' + error);
        }
    });
},

        // ------------------------------------
        // SEARCH TOOLS
        // ------------------------------------
        performSearch: function () {
            var self = this;
            console.log('[ToolManagement] Initiating tool search...');
            var formData = self.$searchForm.serialize();
            console.log('[ToolManagement] Serialized form data:', formData);

            // Basic check
            if (!formData || formData.trim() === '') {
                console.warn('[ToolManagement] Warning: No form data to submit.');
                self.$searchResults.html('<p>Please enter search criteria.</p>');
                return;
            }

            // Show loading
            self.$searchButton.prop('disabled', true);
            self.$searchResults.html('<p>Loading...</p>');

            // Send form data to /pda_search_tools
            $.ajax({
                url: self.$searchForm.attr('action') || '/pda_search_tools',
                method: 'POST',
                data: formData,
                dataType: 'json',
                success: function (response) {
                    console.log('[ToolManagement] Search response:', response);
                    if (response && Array.isArray(response.tools)) {
                        if (response.tools.length > 0) {
                            self.renderSearchResults(response.tools);
                        } else {
                            self.$searchResults.html('<p>No tools found matching the criteria.</p>');
                        }
                    } else {
                        console.warn('[ToolManagement] Unexpected response format:', response);
                        self.$searchResults.html('<p>Unexpected response from the server.</p>');
                    }
                },
                error: function (xhr, status, error) {
                    console.error('[ToolManagement] AJAX request failed:', {
                        status: status,
                        error: error,
                        response: xhr.responseText
                    });
                    self.$searchResults.html('<p>An error occurred while searching for tools.</p>');
                },
                complete: function () {
                    self.$searchButton.prop('disabled', false);
                }
            });
        },

        renderSearchResults: function (tools) {
            console.log('[ToolManagement] Rendering search results:', tools);

            if (!tools || tools.length === 0) {
                this.$searchResults.html('<p>No tools found matching the criteria.</p>');
                return;
            }

            var html = '<table class="table table-bordered">';
            html += '<thead><tr>' +
                    '<th>ID</th><th>Name</th><th>Size</th><th>Type</th><th>Material</th><th>Category</th><th>Manufacturer</th><th>Actions</th>' +
                    '</tr></thead><tbody>';

            tools.forEach(function (tool) {
                html += '<tr>';
                html += '<td>' + tool.id + '</td>';
                html += '<td>' + (tool.name || '') + '</td>';
                html += '<td>' + (tool.size || 'N/A') + '</td>';
                html += '<td>' + (tool.type || 'N/A') + '</td>';
                html += '<td>' + (tool.material || 'N/A') + '</td>';
                html += '<td>' + (tool.tool_category || 'N/A') + '</td>';
                html += '<td>' + (tool.tool_manufacturer || 'N/A') + '</td>';
                html += '<td><button class="btn-add-tool" data-tool-id="' + tool.id + '">Add to Position</button></td>';
                html += '</tr>';
            });

            html += '</tbody></table>';
            this.$searchResults.html(html);
        },

        // ------------------------------------
        // ADD TOOL TO POSITION
        // ------------------------------------
        addToolToPosition: function (toolId) {
            var self = this;

            // Get position ID with our robust method
            var positionId = this.positionId;

            // If we don't have a cached position ID, try to find it again
            if (!positionId) {
                positionId = this.findPositionId();

                // If we found it, cache it
                if (positionId) {
                    this.setPositionId(positionId);
                }
            }

            console.log('[ToolManagement] addToolToPosition called. toolId:', toolId, 'positionId:', positionId);

            if (!toolId || !positionId) {
                alert('Missing tool ID or position ID. Please select a position first.');
                return;
            }

            // AJAX call to add tool to position
            $.ajax({
                url: '/pda_add_tool_to_position',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    tool_id: toolId,
                    position_id: positionId
                }),
                success: function (resp) {
                    console.log('[ToolManagement] addToolToPosition success:', resp);

                    // Show a success message
                    alert(resp.message || 'Tool added successfully!');

                    // Re-fetch associated tools
                    self.fetchAssociatedTools(positionId);

                    // Switch tabs if needed
                    if (self.$subTabs && self.$subTabs.length > 0) {
                        self.$subTabs.removeClass('active');
                        self.$subTabContents.removeClass('active').hide();
                        self.$subTabs.filter('[data-subtab="associated-tools"]').addClass('active');
                        $('#associated-tools').addClass('active').fadeIn();
                    }
                },
                error: function (xhr, status, error) {
                    console.error('[ToolManagement] addToolToPosition error:', error);

                    // Handle specific error codes
                    if (xhr.status === 409) {
                        // 409 CONFLICT - Tool already associated
                        alert('This tool is already associated with this position.');
                    } else {
                        // Other errors
                        alert('Failed to add tool: ' + error);
                    }
                }
            });
        },

        // ------------------------------------
        // SEARCH TOOLS DROPDOWN
        // ------------------------------------
        searchTools: function() {
            var self = this;
            var searchInput = $('#tool-search').val().trim();
            var suggestionBox = $('#tool-suggestion-box');

            console.log('[ToolManagement] Search input:', searchInput);

            if (!searchInput) {
                suggestionBox.html('');
                suggestionBox.css('display', 'none');
                return;
            }

            // Add timestamp to prevent caching
            var timestamp = new Date().getTime();

            $.ajax({
                url: '/pda_search_tools',
                method: 'POST',
                data: {
                    tool_name: searchInput,
                    tool_size: '',
                    tool_type: '',
                    tool_material: '',
                    t: timestamp
                },
                dataType: 'json',
                success: function(response) {
                    console.log('[ToolManagement] Received data:', response);
                    suggestionBox.html('');

                    // Extract tools from the response
                    var tools = response.tools || [];

                    if (tools.length > 0) {
                        tools.forEach(function(tool) {
                            var toolEntry = $('<div class="suggestion-item"></div>');
                            toolEntry.html(
                                '<div>' +
                                '<strong>Name:</strong> ' + (tool.name || '') + '<br>' +
                                '<strong>Category:</strong> ' + (tool.tool_category || '') + '<br>' +
                                '<strong>Manufacturer:</strong> ' + (tool.tool_manufacturer || '') +
                                '</div>'
                            );

                            toolEntry.on('click', function() {
                                self.addToolToPosition(tool.id);
                                suggestionBox.css('display', 'none');
                                $('#tool-search').val('');
                            });

                            suggestionBox.append(toolEntry);
                        });

                        // Apply aggressive styling to make dropdown visible
                        suggestionBox.attr('style',
                            'display: block !important; z-index: 999999 !important; ' +
                            'visibility: visible !important; opacity: 1 !important; ' +
                            'position: absolute !important; top: 100% !important; ' +
                            'left: 0 !important; width: 100% !important; ' +
                            'background-color: rgba(0, 0, 0, 0.95) !important; ' +
                            'border: 3px solid yellow !important; color: yellow !important; ' +
                            'max-height: 300px !important; overflow-y: auto !important;'
                        );
                        console.log('[ToolManagement] Set suggestion box to visible');
                    } else {
                        console.log('[ToolManagement] No tools found for search input:', searchInput);
                        suggestionBox.html('<p style="padding: 10px; margin: 0;">No tools found.</p>');
                        suggestionBox.attr('style',
                            'display: block !important; z-index: 999999 !important; ' +
                            'visibility: visible !important; opacity: 1 !important; ' +
                            'position: absolute !important; top: 100% !important; ' +
                            'left: 0 !important; width: 100% !important; ' +
                            'background-color: rgba(0, 0, 0, 0.95) !important; ' +
                            'border: 3px solid yellow !important; color: yellow !important; ' +
                            'max-height: 300px !important; overflow-y: auto !important;'
                        );
                    }
                },
                error: function(xhr, status, error) {
                    console.error('[ToolManagement] Error searching tools:', error);
                    console.error('[ToolManagement] Status code:', xhr.status);
                    console.error('[ToolManagement] Response text:', xhr.responseText);
                    alert('Error searching tools: ' + error);
                }
            });
        }
    };

    // Initialize the ToolManagement module
    ToolManagement.init();

    // Listen for position ID changes from other scripts
    $(document).on('positionIdChanged', function(e, newPositionId) {
        if (window.ToolManagement) {
            window.ToolManagement.setPositionId(newPositionId);
        }
    });

    // Additional code: Check if position ID is set somewhere else on the page
    // This can capture position IDs set by other scripts after our module loads
    setInterval(function() {
        if (window.ToolManagement && !window.ToolManagement.positionId) {
            var newPositionId = window.ToolManagement.findPositionId();
            if (newPositionId) {
                window.ToolManagement.setPositionId(newPositionId);
            }
        }
    }, 2000); // Check every 2 seconds
});