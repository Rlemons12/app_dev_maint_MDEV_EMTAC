document.addEventListener('DOMContentLoaded', function () {

    // Function to search for drawings
    function searchDrawings() {
        const searchInputElement = document.getElementById('search-drawings');
        const searchInput = searchInputElement.value.trim();
        const suggestionBox = document.getElementById('drawing-suggestion-box');

        if (!searchInput) {
            suggestionBox.innerHTML = '';
            suggestionBox.style.display = 'none';
            return;
        }

        const timestamp = new Date().getTime();
        const fetchUrl = `/pda_search_drawings?query=${encodeURIComponent(searchInput)}&t=${timestamp}`;

        fetch(fetchUrl)
            .then(response => response.json())
            .then(data => {
    suggestionBox.innerHTML = '';
    if (Array.isArray(data) && data.length > 0) {
        data.forEach(drawing => {
            const drawingEntry = document.createElement('div');
            drawingEntry.className = 'suggestion-item';
            drawingEntry.innerHTML = `
                <div>
                    <strong>Name:</strong> ${drawing.drw_name || ''}<br>
                    <strong>Number:</strong> ${drawing.drw_number || ''}<br>
                    <strong>Equipment Name:</strong> ${drawing.drw_equipment_name || ''}<br>
                    <strong>Revision:</strong> ${drawing.drw_revision || ''}<br>
                    <strong>Spare Part Number:</strong> ${drawing.drw_spare_part_number || ''}
                </div>
            `;
            drawingEntry.addEventListener('click', function () {
                addDrawingToPosition(drawing.id, drawing.drw_name);
                suggestionBox.setAttribute('style', 'display: none !important;');
                searchInputElement.value = '';
            });
            suggestionBox.appendChild(drawingEntry);
        });
        suggestionBox.setAttribute('style', 'display: block !important; z-index: 9999 !important;');
        console.log('Setting drawing suggestion box display to visible:', suggestionBox);
    } else {
        suggestionBox.innerHTML = '<p>No drawings found.</p>';
        suggestionBox.setAttribute('style', 'display: block !important; z-index: 9999 !important;');
        console.log('Setting drawing suggestion box display to visible (no results):', suggestionBox);
    }
})
            .catch(error => {
                alert('Error searching drawings: ' + error.message);
                console.error('Error searching drawings:', error);
            });
    }

    // Function to add drawing to position
    function addDrawingToPosition(drawingId, drawingName) {
        const positionIdElement = document.getElementById('position_id');
        const positionId = positionIdElement ? positionIdElement.value.trim() : null;
        console.log(`Attempting to add Drawing ID: ${drawingId} to Position ID: ${positionId}`);

        if (!drawingId || !positionId) {
            alert('Drawing ID and Position ID are required.');
            console.error('Missing drawingId or positionId:', { drawingId, positionId });
            return;
        }

        fetch('/pda_add_drawing_to_position', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ drawing_id: drawingId, position_id: positionId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);
                if (data.drawing_id) {
                    const existingDrawingsList = document.getElementById('existing-drawings-list');
                    const drawingEntry = document.createElement('div');
                    drawingEntry.className = 'existing-drawing';

                    const drawingNameSpan = document.createElement('span');
                    drawingNameSpan.textContent = `Name: ${drawingName}`;

                    const removeButton = document.createElement('button');
                    removeButton.type = 'button';
                    removeButton.textContent = 'Remove';
                    removeButton.className = 'remove-existing-drawing-button';
                    removeButton.setAttribute('data-drawing-id', drawingId);

                    drawingEntry.appendChild(drawingNameSpan);
                    drawingEntry.appendChild(removeButton);

                    existingDrawingsList.appendChild(drawingEntry);
                }
            }
        })
        .catch(error => {
            alert('Error adding drawing: ' + error.message);
            console.error('Error adding drawing:', error);
        });
    }

    // Function to remove drawing from position
    function removeDrawingFromPosition(drawingId, positionId, drawingEntry) {
        if (confirm('Are you sure you want to remove this drawing?')) {
            fetch('/pda_remove_drawing_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ drawing_id: drawingId, position_id: positionId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
                    drawingEntry.remove();
                }
            })
            .catch(error => {
                alert('Error removing drawing: ' + error.message);
                console.error('Error removing drawing:', error);
            });
        }
    }

    // Function to handle drawing uploads
    function handleDrawingUpload() {
        const drwEquipmentName = document.getElementById('drw_equipment_name').value.trim();
        const drwNumber = document.getElementById('drw_number').value.trim();
        const drwName = document.getElementById('drw_name').value.trim();
        const drwRevision = document.getElementById('drw_revision').value.trim();
        const drwSparePartNumber = document.getElementById('drw_spare_part_number').value.trim();
        const positionIdElement = document.getElementById('position_id');
        const positionId = positionIdElement ? positionIdElement.value.trim() : null;
        const fileInput = document.getElementById('drawings-upload');
        const file = fileInput.files[0];

        if (!drwName || !file) {
            alert('Please provide a drawing name and select a file.');
            return;
        }

        if (!positionId) {
            alert('Position ID is missing. Cannot upload drawing.');
            console.error('Missing positionId:', positionId);
            return;
        }

        const formData = new FormData();
        formData.append('drw_equipment_name', drwEquipmentName);
        formData.append('drw_number', drwNumber);
        formData.append('drw_name', drwName);
        formData.append('drw_revision', drwRevision);
        formData.append('drw_spare_part_number', drwSparePartNumber);
        formData.append('position_id', positionId);
        formData.append('file', file);

        fetch('/pda_create_and_add_drawing', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);
                if (data.drawing_id) {
                    const existingDrawingsList = document.getElementById('existing-drawings-list');
                    const drawingEntry = document.createElement('div');
                    drawingEntry.className = 'existing-drawing';

                    const drawingNameSpan = document.createElement('span');
                    drawingNameSpan.textContent = `Name: ${data.drw_name}`;

                    const removeButton = document.createElement('button');
                    removeButton.type = 'button';
                    removeButton.textContent = 'Remove';
                    removeButton.className = 'remove-existing-drawing-button';
                    removeButton.setAttribute('data-drawing-id', data.drawing_id);

                    drawingEntry.appendChild(drawingNameSpan);
                    drawingEntry.appendChild(removeButton);

                    existingDrawingsList.appendChild(drawingEntry);

                    // Clear input fields
                    document.getElementById('drw_equipment_name').value = '';
                    document.getElementById('drw_number').value = '';
                    document.getElementById('drw_name').value = '';
                    document.getElementById('drw_revision').value = '';
                    document.getElementById('drw_spare_part_number').value = '';
                    fileInput.value = '';
                }
            }
        })
        .catch(error => {
            alert('Error uploading drawing: ' + error.message);
            console.error('Error uploading drawing:', error);
        });
    }

    // Event listeners
    const searchDrawingsInput = document.getElementById('search-drawings');
    if (searchDrawingsInput) {
        searchDrawingsInput.addEventListener('keyup', searchDrawings);
    } else {
        console.error('Search input element with id "search-drawings" not found.');
    }

    const uploadDrawingButton = document.getElementById('upload-drawing-button');
    if (uploadDrawingButton) {
        uploadDrawingButton.addEventListener('click', handleDrawingUpload);
    } else {
        console.error('Upload button with id "upload-drawing-button" not found.');
    }

    // Use event delegation for remove buttons
    const existingDrawingsList = document.getElementById('existing-drawings-list');
    existingDrawingsList.addEventListener('click', function(event) {
        if (event.target && event.target.matches('button.remove-existing-drawing-button')) {
            const drawingId = event.target.getAttribute('data-drawing-id');
            const positionIdElement = document.getElementById('position_id');
            const positionId = positionIdElement ? positionIdElement.value.trim() : null;
            removeDrawingFromPosition(drawingId, positionId, event.target.parentNode);
        }
    });

});
