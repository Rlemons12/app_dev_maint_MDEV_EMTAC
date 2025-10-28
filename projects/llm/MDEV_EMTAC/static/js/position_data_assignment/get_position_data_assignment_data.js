    // static/js/position_data_assignment/get_position_data_assignment_data.js
    // ============================
    // Define Your API Endpoints Using Flask's url_for
    // ============================
    var getEquipmentGroupsUrl = "{{ url_for('position_data_assignment_bp.get_equipment_groups') }}";
    var getModelsUrl = "{{ url_for('position_data_assignment_bp.get_models') }}";
    var getAssetNumbersUrl = "{{ url_for('position_data_assignment_bp.get_asset_numbers') }}";
    var getLocationsUrl = "{{ url_for('position_data_assignment_bp.get_locations') }}";
    var getSiteLocationsUrl = "{{ url_for('position_data_assignment_bp.get_site_locations') }}";
    var getPositionsUrl = "{{ url_for('position_data_assignment_bp.get_positions') }}";
    var removeImageFromPositionUrl = "{{ url_for('position_data_assignment_bp.remove_image_from_position') }}";
    // New Routes for Subassembly, ComponentAssembly, and AssemblyView
    var getSubassembliesUrl = "{{ url_for('position_data_assignment_bp.get_subassemblies') }}";
    var getComponentAssembliesUrl = "{{ url_for('position_data_assignment_bp.get_component_assemblies') }}";
    var getAssemblyViewsUrl = "{{ url_for('position_data_assignment_bp.get_assembly_views') }}";


    $(document).ready(function() {
        // ============================
        // Function to Reset and Disable Dropdowns
        // ============================
        function resetDropdowns(selectors) {
            selectors.forEach(function(selector) {
                $(selector).prop('disabled', true).html('<option value="">Select...</option>');
            });
        }

        // ============================
        // Preload Form for Updates if position_id is Provided in URL
        // ============================
        function preloadForm(positionId) {
            if (positionId) {
                $.ajax({
                    url: getPositionsUrl, // Fetch the position data
                    method: 'GET',
                    data: { position_id: positionId },
                    success: function(data) {
                        if (data.position) {
                            $('#pda_areaDropdown').val(data.position.area_id).trigger('change');
                            $('#pda_equipmentGroupDropdown').val(data.position.equipment_group_id).prop('disabled', false);
                            $('#pda_modelDropdown').val(data.position.model_id).prop('disabled', false);
                            $('#pda_assetNumberDropdown').val(data.position.asset_number_id).prop('disabled', false);
                            $('#pda_locationDropdown').val(data.position.location_id).prop('disabled', false).trigger('change'); // Trigger change to load assemblies
                            $('#pda_siteLocation').val(data.position.site_location_id).prop('disabled', false);
                            $('#pda_assemblyDropdown').val(data.position.assembly_id).trigger('change');
                            $('#pda_subAssemblyDropdown').val(data.position.subassembly_id).prop('disabled', false);
                            $('#pda_assemblyViewDropdown').val(data.position.assembly_view_id).prop('disabled', false);
                        }
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching position data:", error);
                        alert("An error occurred while fetching the position.");
                    }
                });
            }
        }

        // ============================
        // Check if position_id is Available in the URL to Preload Form
        // ============================
        var positionId = new URLSearchParams(window.location.search).get('position_id');
        if (positionId) {
            preloadForm(positionId);
        }

        // ============================
        // Area Change Event
        // ============================
        $('#pda_areaDropdown').change(function() {
            var areaId = $(this).val();
            if (areaId) {
                $('#pda_equipmentGroupDropdown').prop('disabled', false);
                // Fetch Equipment Groups
                $.ajax({
                    url: getEquipmentGroupsUrl,
                    method: 'GET',
                    data: { area_id: areaId },
                    success: function(data) {
                        var equipmentGroupDropdown = $('#pda_equipmentGroupDropdown');
                        equipmentGroupDropdown.empty();
                        equipmentGroupDropdown.append('<option value="">Select Equipment Group</option>');
                        $.each(data, function(index, group) {
                            equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching equipment groups:", error);
                        alert("An error occurred while fetching equipment groups.");
                    }
                });
            } else {
                resetDropdowns(['#pda_equipmentGroupDropdown', '#pda_modelDropdown', '#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
            }
        });

        // ============================
        // Equipment Group Change Event
        // ============================
        $('#pda_equipmentGroupDropdown').change(function() {
            var equipmentGroupId = $(this).val();
            if (equipmentGroupId) {
                $('#pda_modelDropdown').prop('disabled', false);
                // Fetch Models
                $.ajax({
                    url: getModelsUrl,
                    method: 'GET',
                    data: { equipment_group_id: equipmentGroupId },
                    success: function(data) {
                        var modelDropdown = $('#pda_modelDropdown');
                        modelDropdown.empty();
                        modelDropdown.append('<option value="">Select Model</option>');
                        $.each(data, function(index, model) {
                            modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching models:", error);
                        alert("An error occurred while fetching models.");
                    }
                });
            } else {
                resetDropdowns(['#pda_modelDropdown', '#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
            }
        });

        // ============================
        // Model Change Event
        // ============================
        $('#pda_modelDropdown').change(function() {
            var modelId = $(this).val();
            if (modelId) {
                $('#pda_assetNumberDropdown').prop('disabled', false);
                $('#pda_locationDropdown').prop('disabled', false);
                // Fetch Asset Numbers
                $.ajax({
                    url: getAssetNumbersUrl,
                    method: 'GET',
                    data: { model_id: modelId },
                    success: function(data) {
                        var assetNumberDropdown = $('#pda_assetNumberDropdown');
                        assetNumberDropdown.empty();
                        assetNumberDropdown.append('<option value="">Select Asset Number</option>');
                        $.each(data, function(index, assetNumber) {
                            assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching asset numbers:", error);
                        alert("An error occurred while fetching asset numbers.");
                    }
                });
                // Fetch Locations
                $.ajax({
                    url: getLocationsUrl,
                    method: 'GET',
                    data: { model_id: modelId },
                    success: function(data) {
                        var locationDropdown = $('#pda_locationDropdown');
                        locationDropdown.empty();
                        locationDropdown.append('<option value="">Select Location</option>');
                        $.each(data, function(index, location) {
                            locationDropdown.append('<option value="' + location.id + '">' + location.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching locations:", error);
                        alert("An error occurred while fetching locations.");
                    }
                });
            } else {
                resetDropdowns(['#pda_assetNumberDropdown', '#pda_locationDropdown', '#pda_siteLocation', '#pda_position']);
            }
        });

        // ============================
        // Location Dropdown Change Event
        // ============================
        $('#pda_locationDropdown').change(function() {
            var locationId = $(this).val();
            console.log("Location dropdown changed. New value:", locationId);

            if (locationId) {
                console.log("Enabling the Subassembly dropdown.");
                $('#pda_subassemblyDropdown').prop('disabled', false);

                console.log("Initiating AJAX call to fetch subassemblies for location_id:", locationId);
                $.ajax({
                    url: getSubassembliesUrl,
                    method: 'GET',
                    data: { location_id: locationId },
                    success: function(data) {
                        console.log("Subassemblies data received:", data);
                        var subassemblyDropdown = $('#pda_subassemblyDropdown');
                        subassemblyDropdown.empty();
                        subassemblyDropdown.append('<option value="">Select Subassembly</option>');

                        $.each(data, function(index, subassembly) {
                            console.log("Adding subassembly at index " + index + ":", subassembly);
                            subassemblyDropdown.append('<option value="' + subassembly.id + '">' + subassembly.name + '</option>');
                        });

                        console.log("Resetting and disabling dependent dropdowns: Component Assembly and Assembly View.");
                        resetDropdowns(['#pda_componentAssemblyDropdown', '#pda_assemblyViewDropdown']);
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching subassemblies. Status:", status, "Error:", error);
                        alert("An error occurred while fetching subassemblies.");
                    }
                });
            } else {
                console.log("No location selected. Resetting Subassembly, Component Assembly, and Assembly View dropdowns.");
                resetDropdowns(['#pda_subassemblyDropdown', '#pda_componentAssemblyDropdown', '#pda_assemblyViewDropdown']);
            }
        });

        // ============================
        // Subassembly Change Event
        // ============================
        $('#pda_subassemblyDropdown').change(function() {
            var subassemblyId = $(this).val();
            console.log("Subassembly dropdown changed. New value:", subassemblyId);

            if (subassemblyId) {
                console.log("Enabling the Component Assembly dropdown.");
                $('#pda_componentAssemblyDropdown').prop('disabled', false);

                console.log("Initiating AJAX call to fetch component assemblies for subassembly_id:", subassemblyId);
                $.ajax({
                    url: getComponentAssembliesUrl,
                    method: 'GET',
                    data: { subassembly_id: subassemblyId },
                    success: function(data) {
                        console.log("Component assemblies data received:", data);
                        var componentAssemblyDropdown = $('#pda_componentAssemblyDropdown');
                        componentAssemblyDropdown.empty();
                        componentAssemblyDropdown.append('<option value="">Select Component Assembly</option>');

                        $.each(data, function(index, componentAssembly) {
                            console.log("Adding component assembly at index " + index + ":", componentAssembly);
                            componentAssemblyDropdown.append('<option value="' + componentAssembly.id + '">' + componentAssembly.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching component assemblies. Status:", status, "Error:", error);
                        alert("An error occurred while fetching component assemblies.");
                    }
                });
            } else {
                console.log("No subassembly selected. Resetting Component Assembly and Assembly View dropdowns.");
                resetDropdowns(['#pda_componentAssemblyDropdown', '#pda_assemblyViewDropdown']);
            }
        });

        // ============================
        // Component Assembly Change Event
        // ============================
        $('#pda_componentAssemblyDropdown').change(function() {
            var componentAssemblyId = $(this).val();
            console.log("Component Assembly dropdown changed. New value:", componentAssemblyId);

            if (componentAssemblyId) {
                console.log("Enabling the Assembly View dropdown.");
                $('#pda_assemblyViewDropdown').prop('disabled', false);

                console.log("Initiating AJAX call to fetch assembly views for component_assembly_id:", componentAssemblyId);
                $.ajax({
                    url: getAssemblyViewsUrl,
                    method: 'GET',
                    data: { component_assembly_id: componentAssemblyId },
                    success: function(data) {
                        console.log("Assembly views data received:", data);
                        var assemblyViewDropdown = $('#pda_assemblyViewDropdown');
                        assemblyViewDropdown.empty();
                        assemblyViewDropdown.append('<option value="">Select Assembly View</option>');

                        $.each(data, function(index, assemblyView) {
                            console.log("Adding assembly view at index " + index + ":", assemblyView);
                            assemblyViewDropdown.append('<option value="' + assemblyView.id + '">' + assemblyView.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching assembly views. Status:", status, "Error:", error);
                        alert("An error occurred while fetching assembly views.");
                    }
                });
            } else {
                console.log("No component assembly selected. Resetting Assembly View dropdown.");
                resetDropdowns(['#pda_assemblyViewDropdown']);
            }
        });

        // ============================
        // Site Location Change Event
        // ============================
        $('#pda_siteLocation').change(function() {
            var siteLocationId = $(this).val();
            var assetNumberId = $('#pda_assetNumberDropdown').val();
            var locationId = $('#pda_locationDropdown').val();
            var modelId = $('#pda_modelDropdown').val(); // Include model_id if necessary

            if (siteLocationId || assetNumberId || locationId || modelId) { // Check if any relevant field is filled
                $('#pda_position').prop('disabled', false); // Ensure the Position Dropdown is enabled
                // Fetch Positions without requiring all fields
                $.ajax({
                    url: getPositionsUrl,
                    method: 'GET',
                    data: {
                        site_location_id: siteLocationId || '',
                        asset_number_id: assetNumberId || '',
                        location_id: locationId || '',
                        model_id: modelId || '' // Include model_id if necessary
                    },
                    success: function(data) {
                        var positionDropdown = $('#pda_position');
                        positionDropdown.empty();
                        positionDropdown.append('<option value="">Select Position</option>');
                        $.each(data, function(index, position) {
                            positionDropdown.append('<option value="' + position.id + '">' + position.name + '</option>');
                        });
                    },
                    error: function(xhr, status, error) {
                        console.error("Error fetching positions:", error);
                        alert("An error occurred while fetching positions.");
                    }
                });
            } else {
                resetDropdowns(['#pda_position']);
            }
        });

        // ============================
        // Allow Manual Input for Asset Number and Location
        // ============================
        $('#pda_assetNumberInput').on('input', function() {
            var assetNumber = $(this).val();
            if (assetNumber.length > 1) {
                $('#pda_assetNumberDropdown').prop('disabled', true);
            } else {
                $('#pda_assetNumberDropdown').prop('disabled', false);
            }
        });

        $('#pda_locationInput').on('input', function() {
            var location = $(this).val();
            if (location.length > 1) {
                $('#pda_locationDropdown').prop('disabled', true);
            } else {
                $('#pda_locationDropdown').prop('disabled', false);
            }
        });

        // ============================
        // Functions to Manage Parts Entries
        // ============================
        function addPartEntry() {
            var container = document.getElementById('parts-container');
            var newEntry = document.createElement('div');
            newEntry.className = 'part-entry';
            newEntry.innerHTML = `
                <label>Part Number:</label>
                <input type="text" name="part_numbers[]" required>
                <button type="button" onclick="removePartEntry(this)">Remove</button>
            `;
            container.appendChild(newEntry);
        }

        function removePartEntry(button) {
            button.parentElement.remove();
        }

       // ============================
    // Search Position Button Click Event
    // ============================
    $('#searchPositionBtn').click(function () {
        $.ajax({
            url: getPositionsUrl, // Ensure this variable is defined with the correct URL
            method: 'GET',
            data: $('#searchPositionForm').serialize(),
            success: function (data) {
                clearAllSections(); // Clear previous results
    
                if (data && Array.isArray(data)) {
                    data.forEach(function (position) {
                        console.log('Position Data:', position);
    
                        // Populate position details on the page
                        setPositionDetails(position); // Ensure this function sets the #position_id
    
                        // Render associated entities
                        renderParts(position.parts);
                        renderImages(position.images);
                        renderDocuments(position.documents);
                        renderDrawings(position.drawings);
    
                        // After setting the position details, fetch associated tools
                        var positionId = $('#position_id').val();
                        if (positionId) {
                            // Call the global ToolManagement's fetchAssociatedTools method
                            window.ToolManagement.fetchAssociatedTools(positionId);
                        }
                    });
                } else {
                    alert("No positions found.");
                }
            },
            error: function (xhr, status, error) {
                console.error("Error fetching positions:", error);
                alert("An error occurred while fetching positions.");
            }
        });
    });


        // ============================
        // Function to Clear All Sections Before Rendering
        // ============================
        function clearAllSections() {
            $('#existing-parts-list, #existing-images-list, #existing-documents-list, #existing-drawings-list').empty();
            $('#position_display, #edit_areaDropdown, #edit_equipmentGroupDropdown, #edit_modelDropdown, #edit_assetNumberDropdown, #edit_locationDropdown, #edit_area_name, #edit_area_description, #edit_model_name, #edit_model_description, #edit_assetNumber, #edit_assetNumber_description, #edit_siteLocationDropdown, #edit_siteLocation_title, #edit_siteLocation_roomNumber').val('');
        }

        // ============================
        // Function to Set Position Details
        // ============================
        function setPositionDetails(position) {
            const positionId = position.position_id || position.id;
            if (positionId) {
                $('#position_display, #position_id').val(positionId);
                console.log('Position ID set:', positionId);
            }

            if (position.area) {
                $('#edit_areaDropdown').val(position.area.id);
                $('#edit_area_name').val(position.area.name);
                $('#edit_area_description').val(position.area.description);
            }

            if (position.equipment_group) {
                $('#edit_equipmentGroupDropdown').val(position.equipment_group.id);
            }

            if (position.model) {
                $('#edit_modelDropdown').val(position.model.id);
                $('#edit_model_name').val(position.model.name);
                $('#edit_model_description').val(position.model.description);
            }

            if (position.asset_number) {
                $('#edit_assetNumberDropdown').val(position.asset_number.id);
                $('#edit_assetNumber').val(position.asset_number.number);
                $('#edit_assetNumber_description').val(position.asset_number.description);
            }

            if (position.location) {
                $('#edit_locationDropdown').val(position.location.id);
                $('#edit_location_name').val(position.location.name);
                $('#edit_location_description').val(position.location.description || '');
            }

            if (position.site_location) {
                $('#edit_siteLocationDropdown').val(position.site_location.id);
                $('#edit_siteLocation_title').val(position.site_location.title);
                $('#edit_siteLocation_roomNumber').val(position.site_location.room_number);
            }
        }

        // ============================
        // Function to Render Parts
        // ============================
        function renderParts(parts) {
            const partsList = $('#existing-parts-list');
            partsList.empty();

            if (!parts || parts.length === 0) {
                partsList.append('<p>No parts available.</p>');
                return;
            }

            parts.forEach(function (part) {
                console.log('Rendering part:', part); // Debugging log
                const partEntry = $(`
                    <div class="existing-part" id="part-${part.part_id}">
                        <span>Part Number: ${escapeHtml(part.part_number)}, Name: ${escapeHtml(part.name)}</span>
                        <button type="button" class="remove-existing-part-button" data-part-id="${part.part_id}">Remove</button>
                    </div>
                `);
                partsList.append(partEntry);
            });
        }

        // ============================
        // Function to Render Images
        // ============================
        function renderImages(images) {
            const imagesList = $('#existing-images-list');
            imagesList.empty();

            if (!images || images.length === 0) {
                imagesList.append('<p>No images available.</p>');
                return;
            }

            images.forEach(function (image) {
                const safeTitle = escapeHtml(image.title || 'N/A');
                const safeDescription = escapeHtml(image.description || 'No description available');

                imagesList.append(`
                    <div class="existing-image" id="image-${image.image_id}">
                        <span>Title: ${safeTitle}, Description: ${safeDescription}</span>
                        <button type="button" class="remove-existing-image-button" data-image-id="${image.image_id}">Remove</button>
                    </div>
                `);
            });
        }

        // ============================
        // Function to Render Documents
        // ============================
        function renderDocuments(documents) {
            const documentsList = $('#existing-documents-list');
            documentsList.empty();

            if (!documents || documents.length === 0) {
                documentsList.append('<p>No documents available.</p>');
                return;
            }

            documents.forEach(function (doc) {
                const safeTitle = escapeHtml(doc.title || 'N/A');
                const safeRev = escapeHtml(doc.rev || 'N/A');

                documentsList.append(`
                    <div class="existing-document" id="document-${doc.document_id}">
                        <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                        <button type="button" class="remove-existing-document-button" data-document-id="${doc.document_id}">Remove</button>
                    </div>
                `);
            });
        }

        // ============================
        // Function to Render Drawings
        // ============================
        function renderDrawings(drawings) {
            const drawingsList = $('#existing-drawings-list');
            drawingsList.empty();

            if (!drawings || drawings.length === 0) {
                drawingsList.append('<p>No drawings available.</p>');
                return;
            }

            drawings.forEach(function (drawing) {
                drawingsList.append(`
                    <div class="existing-drawing" id="drawing-${drawing.drawing_id}">
                        <span>Drawing Name: ${escapeHtml(drawing.drw_name)}, Number: ${escapeHtml(drawing.drw_number)}</span>
                        <button type="button" class="remove-existing-drawing-button" data-drawing-id="${drawing.drawing_id}">Remove</button>
                    </div>
                `);
            });
        }

        // ============================
        // Utility Function to Escape HTML to Prevent XSS
        // ============================
        function escapeHtml(text) {
            return $('<div>').text(text).html();
        }

        // ============================
        // Functions to Paginate and Filter Parts, Images, and Drawings
        // ============================
        let allParts = [], allImages = [], allDrawings = [], allDocuments = [];
        let currentPartsPage = 1, currentImagesPage = 1, currentDrawingsPage = 1, currentDocumentsPage = 1;
        const partsPerPage = 10, imagesPerPage = 10, drawingsPerPage = 10, documentsPerPage = 10;

        // ============================
        // Function to Render Parts for the Current Page
        // ============================
        function renderPartsPage(page = 1) {
            const startIndex = (page - 1) * partsPerPage;
            const endIndex = startIndex + partsPerPage;
            const currentParts = allParts.slice(startIndex, endIndex);

            const partsList = document.getElementById('existing-parts-list');
            partsList.innerHTML = '';  // Clear the list

            currentParts.forEach(part => {
                const partItem = document.createElement('p');
                partItem.textContent = `Part Number: ${part.part_number}, Name: ${part.name}`;
                partsList.appendChild(partItem);
            });

            renderPartsPagination(page);
        }

        // ============================
        // Function to Render Pagination Controls for Parts
        // ============================
        function renderPartsPagination(page) {
            const totalPages = Math.ceil(allParts.length / partsPerPage);
            const paginationContainer = document.getElementById('parts-pagination');
            paginationContainer.innerHTML = '';

            if (totalPages > 1) {
                for (let i = 1; i <= totalPages; i++) {
                    const pageButton = document.createElement('button');
                    pageButton.textContent = i;
                    pageButton.disabled = i === page;
                    pageButton.onclick = () => renderPartsPage(i);
                    paginationContainer.appendChild(pageButton);
                }
            }
        }

        // ============================
        // Filter Parts Based on Search Input
        // ============================
        function filterParts() {
            const searchTerm = document.getElementById('search-parts').value.toLowerCase();
            allParts = allParts.filter(part =>
                part.part_number.toLowerCase().includes(searchTerm) ||
                part.name.toLowerCase().includes(searchTerm)
            );
            renderPartsPage(1);
        }

        // ============================
        // Function to Render Documents for the Current Page
        // ============================
        function renderDocumentsPage(page = 1) {
            const startIndex = (page - 1) * documentsPerPage;
            const endIndex = startIndex + documentsPerPage;
            const currentDocuments = allDocuments.slice(startIndex, endIndex);

            const documentsList = document.getElementById('existing-documents-list');
            documentsList.innerHTML = '';  // Clear the list

            currentDocuments.forEach(doc => {
                const safeTitle = doc.title ? escapeHtml(doc.title) : 'N/A';
                const safeRev = doc.rev ? escapeHtml(doc.rev) : 'N/A';

                const docEntry = document.createElement('div');
                docEntry.className = 'existing-document';
                docEntry.id = `document-${doc.document_id}`;
                docEntry.innerHTML = `
                    <span>Title: ${safeTitle}, Revision: ${safeRev}</span>
                    <button type="button" class="remove-existing-document-button" data-document-id="${doc.document_id}">Remove</button>
                `;
                documentsList.appendChild(docEntry);
            });

            renderDocumentsPagination(page);
        }

        // ============================
        // Function to Render Pagination Controls for Documents
        // ============================
        function renderDocumentsPagination(page) {
            const totalPages = Math.ceil(allDocuments.length / documentsPerPage);
            const paginationContainer = document.getElementById('documents-pagination');
            paginationContainer.innerHTML = '';

            if (totalPages > 1) {
                for (let i = 1; i <= totalPages; i++) {
                    const pageButton = document.createElement('button');
                    pageButton.textContent = i;
                    pageButton.disabled = i === page;
                    pageButton.onclick = () => renderDocumentsPage(i);
                    paginationContainer.appendChild(pageButton);
                }
            }
        }

        // ============================
        // Function to Render Images for the Current Page
        // ============================
        function renderImagesPage(page = 1) {
            const startIndex = (page - 1) * imagesPerPage;
            const endIndex = startIndex + imagesPerPage;
            const currentImages = allImages.slice(startIndex, endIndex);

            const imagesList = document.getElementById('existing-images-list');
            imagesList.innerHTML = '';  // Clear the list

            currentImages.forEach(image => {
                const safeTitle = escapeHtml(image.title || 'N/A');
                const safeDescription = escapeHtml(image.description || 'No description available');

                const imageEntry = document.createElement('div');
                imageEntry.className = 'existing-image';
                imageEntry.id = `image-${image.image_id}`;
                imageEntry.innerHTML = `
                    <span>Title: ${safeTitle}, Description: ${safeDescription}</span>
                    <button type="button" class="remove-existing-image-button" data-image-id="${image.image_id}">Remove</button>
                `;
                imagesList.appendChild(imageEntry);
            });

            renderImagesPagination(page);
        }

        // ============================
        // Function to Render Pagination Controls for Images
        // ============================
        function renderImagesPagination(page) {
            const totalPages = Math.ceil(allImages.length / imagesPerPage);
            const paginationContainer = document.getElementById('images-pagination');
            paginationContainer.innerHTML = '';

            if (totalPages > 1) {
                for (let i = 1; i <= totalPages; i++) {
                    const pageButton = document.createElement('button');
                    pageButton.textContent = i;
                    pageButton.disabled = i === page;
                    pageButton.onclick = () => renderImagesPage(i);
                    paginationContainer.appendChild(pageButton);
                }
            }
        }

        // ============================
        // Function to Render Drawings for the Current Page
        // ============================
        function renderDrawingsPage(page = 1) {
            const startIndex = (page - 1) * drawingsPerPage;
            const endIndex = startIndex + drawingsPerPage;
            const currentDrawings = allDrawings.slice(startIndex, endIndex);

            const drawingsList = document.getElementById('existing-drawings-list');
            drawingsList.innerHTML = '';  // Clear the list

            currentDrawings.forEach(drawing => {
                const drawingEntry = document.createElement('div');
                drawingEntry.className = 'existing-drawing';
                drawingEntry.id = `drawing-${drawing.drawing_id}`;
                drawingEntry.innerHTML = `
                    <span>Drawing Name: ${escapeHtml(drawing.drw_name)}, Number: ${escapeHtml(drawing.drw_number)}</span>
                    <button type="button" class="remove-existing-drawing-button" data-drawing-id="${drawing.drawing_id}">Remove</button>
                `;
                drawingsList.appendChild(drawingEntry);
            });

            renderDrawingsPagination(page);
        }

        // ============================
        // Function to Render Pagination Controls for Drawings
        // ============================
        function renderDrawingsPagination(page) {
            const totalPages = Math.ceil(allDrawings.length / drawingsPerPage);
            const paginationContainer = document.getElementById('drawings-pagination');
            paginationContainer.innerHTML = '';

            if (totalPages > 1) {
                for (let i = 1; i <= totalPages; i++) {
                    const pageButton = document.createElement('button');
                    pageButton.textContent = i;
                    pageButton.disabled = i === page;
                    pageButton.onclick = () => renderDrawingsPage(i);
                    paginationContainer.appendChild(pageButton);
                }
            }
        }

        // ============================
        // Simulate Loading of Parts, Images, and Drawings (Replace with Actual AJAX Requests)
        // ============================
        document.addEventListener('DOMContentLoaded', function() {
            // Replace these sample data arrays with actual AJAX requests if needed
            const samplePartsData = [
                // Example:
                // { part_id: 1, part_number: 'PN-001', name: 'Part One' },
                // { part_id: 2, part_number: 'PN-002', name: 'Part Two' },
            ];
            const sampleImagesData = [
                // Example:
                // { image_id: 1, title: 'Image One', description: 'Description One' },
            ];
            const sampleDrawingsData = [
                // Example:
                // { drawing_id: 1, drw_name: 'Drawing One', drw_number: 'DWG-001' },
            ];

            allParts = samplePartsData;
            allImages = sampleImagesData;
            allDrawings = sampleDrawingsData;

            renderPartsPage(1);
            renderImagesPage(1);
            renderDrawingsPage(1);
        });

        // ============================
        // Event Delegation for Removing Existing Images
        // ============================
        $('#existing-images-list').on('click', '.remove-existing-image-button', function() {
            const imageId = $(this).data('image-id');
            const positionId = $('#position_id').val();

            $.ajax({
                url: removeImageFromPositionUrl, // Updated to use removeImageFromPositionUrl
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ image_id: imageId, position_id: positionId }),
                success: function(response) {
                    // Remove the image element from the DOM
                    $(`#image-${imageId}`).remove();
                    console.log(`Removed image ID ${imageId} from position ID ${positionId}`);
                },
                error: function(xhr, status, error) {
                    console.error(`Error removing image ID ${imageId}:`, error);
                    alert('An error occurred while removing the image.');
                }
            });
        });

        // ============================
        // Event Delegation for Removing Existing Parts
        // ============================
        $('#existing-parts-list').on('click', '.remove-existing-part-button', function() {
            const partId = $(this).data('part-id');
            const positionId = $('#position_id').val();

            $.ajax({
                url: "{{ url_for('position_data_assignment_bp.remove_part_from_position') }}", // Define this route in your Flask app
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ part_id: partId, position_id: positionId }),
                success: function(response) {
                    // Remove the part element from the DOM
                    $(`#part-${partId}`).remove();
                    console.log(`Removed part ID ${partId} from position ID ${positionId}`);
                },
                error: function(xhr, status, error) {
                    console.error(`Error removing part ID ${partId}:`, error);
                    alert('An error occurred while removing the part.');
                }
            });
        });

        // ============================
        // Event Delegation for Removing Existing Documents
        // ============================
        $('#existing-documents-list').on('click', '.remove-existing-document-button', function() {
            const documentId = $(this).data('document-id');
            const positionId = $('#position_id').val();

            $.ajax({
                url: "{{ url_for('position_data_assignment_bp.remove_document_from_position') }}", // Define this route in your Flask app
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ document_id: documentId, position_id: positionId }),
                success: function(response) {
                    // Remove the document element from the DOM
                    $(`#document-${documentId}`).remove();
                    console.log(`Removed document ID ${documentId} from position ID ${positionId}`);
                },
                error: function(xhr, status, error) {
                    console.error(`Error removing document ID ${documentId}:`, error);
                    alert('An error occurred while removing the document.');
                }
            });
        });

        // ============================
        // Event Delegation for Removing Existing Drawings
        // ============================
        $('#existing-drawings-list').on('click', '.remove-existing-drawing-button', function() {
            const drawingId = $(this).data('drawing-id');
            const positionId = $('#position_id').val();

            $.ajax({
                url: "{{ url_for('position_data_assignment_bp.remove_drawing_from_position') }}", // Define this route in your Flask app
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ drawing_id: drawingId, position_id: positionId }),
                success: function(response) {
                    // Remove the drawing element from the DOM
                    $(`#drawing-${drawingId}`).remove();
                    console.log(`Removed drawing ID ${drawingId} from position ID ${positionId}`);
                },
                error: function(xhr, status, error) {
                    console.error(`Error removing drawing ID ${drawingId}:`, error);
                    alert('An error occurred while removing the drawing.');
                }
            });
        });

        // ============================
        // Additional Event Handlers or Functions (If Any)
        // ============================
        // Add any additional functionalities or event handlers below as needed.

    });

