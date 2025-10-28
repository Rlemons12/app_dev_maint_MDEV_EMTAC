document.addEventListener('DOMContentLoaded', function () {
    // Function to search for parts
    function searchParts() {
        const searchInputElement = document.getElementById('parts-search');
        const searchInput = searchInputElement.value.trim();
        const suggestionBox = document.getElementById('parts-suggestion-box');

        console.log('Parts search input:', searchInput);

        if (!searchInput) {
            suggestionBox.innerHTML = '';
            suggestionBox.style.cssText = 'display: none !important;';
            return;
        }

        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        const fetchUrl = `/pda_search_parts?query=${encodeURIComponent(searchInput)}&t=${timestamp}`;
        console.log('Fetching URL:', fetchUrl);

        fetch(fetchUrl, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log('Received response:', response);
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            suggestionBox.innerHTML = '';  // Clear previous results

            if (Array.isArray(data) && data.length > 0) {
                data.forEach((part, index) => {
                    console.log(`Processing part ${index}:`, part);
                    const partEntry = document.createElement('div');
                    partEntry.className = 'suggestion-item';

                    // Display part details
                    partEntry.innerHTML = `
                        <div>
                            <strong>Part Number:</strong> ${part.part_number || ''}<br>
                            <strong>Name:</strong> ${part.name || ''}<br>
                            <strong>Manufacturer:</strong> ${part.manufacturer || ''}
                        </div>
                    `;

                    partEntry.addEventListener('click', function () {
                        addPartToPosition(part.id, part.part_number, part.name);
                        suggestionBox.style.cssText = 'display: none !important;';
                        searchInputElement.value = '';
                    });
                    suggestionBox.appendChild(partEntry);
                });

                // Apply aggressive styling to make dropdown visible
                suggestionBox.style.cssText = 'display: block !important; z-index: 999999 !important; visibility: visible !important; opacity: 1 !important; position: absolute !important; top: 100% !important; left: 0 !important; width: 100% !important; background-color: rgba(0, 0, 0, 0.95) !important; border: 3px solid yellow !important; color: yellow !important; max-height: 300px !important; overflow-y: auto !important;';

                // Force a browser reflow/repaint
                const forceRepaint = suggestionBox.offsetHeight;

                console.log('Setting parts suggestion box display to visible:', suggestionBox);
            } else {
                console.log('No parts found for search input:', searchInput);
                suggestionBox.innerHTML = '<p>No parts found.</p>';

                // Apply aggressive styling to make dropdown visible
                suggestionBox.style.cssText = 'display: block !important; z-index: 999999 !important; visibility: visible !important; opacity: 1 !important; position: absolute !important; top: 100% !important; left: 0 !important; width: 100% !important; background-color: rgba(0, 0, 0, 0.95) !important; border: 3px solid yellow !important; color: yellow !important; max-height: 300px !important; overflow-y: auto !important;';

                // Force a browser reflow/repaint
                const forceRepaint = suggestionBox.offsetHeight;

                console.log('Setting parts suggestion box display to visible (no results):', suggestionBox);
            }
        })
        .catch(error => {
            alert('Error searching parts: ' + error.message);
            console.error('Error searching parts:', error);
        });
    }

    // Function to add part to position
    function addPartToPosition(partId, partNumber, partName) {
        const positionId = document.getElementById('position_id').value;
        console.log(`Adding part ${partId} (${partNumber}) to position ${positionId}`);

        if (!partId || !positionId) {
            alert('Part ID and Position ID are required.');
            console.error('Missing partId or positionId:', { partId, positionId });
            return;
        }

        fetch('/add_part_to_position', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ part_id: partId, position_id: positionId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);

                // Add the part to existing parts list
                const existingPartsList = document.getElementById('existing-parts-list');
                const newPartEntry = document.createElement('div');
                newPartEntry.className = 'existing-part';
                newPartEntry.id = `part-${partId}`;

                // Create span for part info
                const partInfoSpan = document.createElement('span');
                partInfoSpan.textContent = `Part Number: ${partNumber}, Name: ${partName}`;

                // Create remove button
                const removeButton = document.createElement('button');
                removeButton.type = 'button';
                removeButton.textContent = 'Remove';
                removeButton.className = 'remove-existing-part-button';
                removeButton.setAttribute('data-part-id', partId);
                removeButton.setAttribute('data-position-id', positionId);

                // Append to newPartEntry
                newPartEntry.appendChild(partInfoSpan);
                newPartEntry.appendChild(removeButton);

                existingPartsList.appendChild(newPartEntry);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            alert('Error adding part: ' + error.message);
            console.error('Error adding part:', error);
        });
    }

    // Function to remove part from position
    function removePartFromPosition(button, partId, positionId) {
        if (confirm('Are you sure you want to remove this part?')) {
            fetch('/pda_remove_part_from_position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ part_id: partId, position_id: positionId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
                    // Remove the part from the UI
                    button.closest('.existing-part').remove();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                alert('Error removing part: ' + error.message);
                console.error('Error removing part:', error);
            });
        }
    }

    // Add event listener to search input
    const partsSearchInput = document.getElementById('parts-search');
    if (partsSearchInput) {
        partsSearchInput.addEventListener('keyup', searchParts);
    } else {
        console.error('Parts Search Input not found.');
    }

    // Add event listeners to remove buttons
    document.addEventListener('click', function(event) {
        if (event.target && event.target.classList.contains('remove-existing-part-button')) {
            const partId = event.target.getAttribute('data-part-id');
            const positionId = event.target.getAttribute('data-position-id');
            removePartFromPosition(event.target, partId, positionId);
        }
    });

    // Add event listener for "Add Another Part" button
    const addPartButton = document.getElementById('add-part-button');
    if (addPartButton) {
        addPartButton.addEventListener('click', function() {
            // Add new part entry
            const newPartsContainer = document.getElementById('new-parts-container');
            const partEntry = document.createElement('div');
            partEntry.className = 'part-entry';

            partEntry.innerHTML = `
                <label for="part-number">Part Number:</label>
                <input type="text" class="part-number" required>
                <label for="part-name">Part Name:</label>
                <input type="text" class="part-name" required>
                <label for="part-manufacturer">Manufacturer:</label>
                <input type="text" class="part-manufacturer">
                <button type="button" class="remove-part-entry">Remove</button>
            `;

            // Add event listener to remove button
            const removeButton = partEntry.querySelector('.remove-part-entry');
            removeButton.addEventListener('click', function() {
                partEntry.remove();
            });

            newPartsContainer.appendChild(partEntry);
        });
    }

    // Add event listener for "Submit New Parts" button
    const submitPartsButton = document.getElementById('submit-parts-button');
    if (submitPartsButton) {
        submitPartsButton.addEventListener('click', function() {
            const partEntries = document.querySelectorAll('.part-entry');
            const positionId = document.getElementById('position_id').value;
            const partsData = [];

            partEntries.forEach(entry => {
                const partNumber = entry.querySelector('.part-number').value.trim();
                const partName = entry.querySelector('.part-name').value.trim();
                const manufacturer = entry.querySelector('.part-manufacturer').value.trim();

                if (partNumber && partName) {
                    partsData.push({
                        part_number: partNumber,
                        name: partName,
                        manufacturer: manufacturer,
                        position_id: positionId
                    });
                }
            });

            if (partsData.length > 0) {
                fetch('/pda_create_and_add_parts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ parts: partsData })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.message) {
                        alert(data.message);
                        // Clear the new parts container
                        document.getElementById('new-parts-container').innerHTML = '';

                        // Refresh the existing parts list if needed
                        if (data.added_parts && Array.isArray(data.added_parts)) {
                            const existingPartsList = document.getElementById('existing-parts-list');

                            data.added_parts.forEach(part => {
                                const newPartEntry = document.createElement('div');
                                newPartEntry.className = 'existing-part';
                                newPartEntry.id = `part-${part.id}`;

                                newPartEntry.innerHTML = `
                                    <span>Part Number: ${part.part_number}, Name: ${part.name}</span>
                                    <button type="button" class="remove-existing-part-button" 
                                            data-part-id="${part.id}" data-position-id="${positionId}">Remove</button>
                                `;

                                existingPartsList.appendChild(newPartEntry);
                            });
                        }
                    } else {
                        alert('Error: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    alert('Error submitting parts: ' + error.message);
                    console.error('Error submitting parts:', error);
                });
            } else {
                alert('Please add at least one part with part number and name.');
            }
        });
    }
});