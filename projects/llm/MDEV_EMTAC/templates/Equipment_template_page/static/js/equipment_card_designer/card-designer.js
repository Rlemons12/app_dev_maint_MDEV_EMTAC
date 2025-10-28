// Fixed version of card-designer.js
document.addEventListener('DOMContentLoaded', function() {
    console.log('Card Designer JS loaded');

    // Tab switching functionality
    setupTabs();

    // Setup real-time preview updates
    setupFormListeners();

    // Initialize the preview with default values
    updateAllPreviewFields();
});

// Tab switching functionality
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const formSections = document.querySelectorAll('.form-section');

    // Basic Info tab is active by default
    document.getElementById('basic-info').classList.add('active');

    tabButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();

            // Remove active class from all buttons and sections
            tabButtons.forEach(btn => btn.classList.remove('active'));
            formSections.forEach(section => section.classList.remove('active'));

            // Add active class to clicked button
            this.classList.add('active');

            // Show corresponding section
            const tabId = this.getAttribute('data-tab');
            const section = document.getElementById(tabId);
            if (section) {
                section.classList.add('active');
            }
        });
    });
}

// Setup form listeners for real-time preview
function setupFormListeners() {
    // Basic Info
    attachInputListener('equipment-name', updateEquipmentName);
    attachInputListener('equipment-id', updateEquipmentId);
    attachInputListener('asset-number', updateAssetNumber);
    attachInputListener('manufacturer', updateManufacturer);
    attachInputListener('model-number', updateModelNumber);
    attachInputListener('serial-number', updateSerialNumber);
    attachInputListener('year', updateYearOfManufacture);

    // Specifications
    attachInputListener('site-facility', updateSiteFacility);
    attachInputListener('area', updateArea);
    attachInputListener('line-section', updateLineSection);
    attachInputListener('location-code', updateLocationCode);

    // Custom fields
    setupCustomFields();

    // Appearance
    setupIconSelection();
    attachInputListener('card-color', updateCardColor, 'input');
    setupImageUpload();

    // Notes
    attachInputListener('maintenance-notes', updateMaintenanceNotes);
    attachInputListener('maintenance-schedule', updateMaintenanceSchedule, 'change');

    // Export buttons
    document.getElementById('save-template')?.addEventListener('click', saveAsTemplate);
    document.getElementById('export-pdf')?.addEventListener('click', exportAsPdf);
    document.getElementById('export-html')?.addEventListener('click', exportAsHtml);
}

// Helper function to attach input listeners
function attachInputListener(elementId, updateFunction, eventType = 'input') {
    const element = document.getElementById(elementId);
    if (element) {
        element.addEventListener(eventType, updateFunction);
    }
}

// Update preview functions
function updateEquipmentName() {
    const value = document.getElementById('equipment-name').value || 'Equipment Name';
    document.getElementById('previewTitle').textContent = value;
}

function updateEquipmentId() {
    const value = document.getElementById('equipment-id').value || '---';
    document.getElementById('previewEquipmentId').textContent = `Equipment ID: ${value}`;
}

function updateAssetNumber() {
    const value = document.getElementById('asset-number').value || '---';
    document.getElementById('previewAssetNumber').textContent = value;
}

function updateManufacturer() {
    const value = document.getElementById('manufacturer').value || '---';
    document.getElementById('previewManufacturer').textContent = value;
}

function updateModelNumber() {
    const value = document.getElementById('model-number').value || '---';
    document.getElementById('previewModelNumber').textContent = value;
}

function updateSerialNumber() {
    const value = document.getElementById('serial-number').value || '---';
    document.getElementById('previewSerialNumber').textContent = value;
}

function updateYearOfManufacture() {
    const value = document.getElementById('year').value || '---';
    document.getElementById('previewYear').textContent = value;
}

function updateSiteFacility() {
    const value = document.getElementById('site-facility').value || '---';
    document.getElementById('previewSiteFacility').textContent = value;
}

function updateArea() {
    const value = document.getElementById('area').value || '---';
    document.getElementById('previewArea').textContent = value;
}

function updateLineSection() {
    const value = document.getElementById('line-section').value || '---';
    document.getElementById('previewLineSection').textContent = value;
}

function updateLocationCode() {
    const value = document.getElementById('location-code').value || '---';
    document.getElementById('previewLocationCode').textContent = value;
}

// Set up custom fields functionality
function setupCustomFields() {
    const addFieldButton = document.getElementById('add-field');
    if (addFieldButton) {
        addFieldButton.addEventListener('click', addCustomField);
    }
}

// Add a custom field to the form
function addCustomField() {
    const container = document.getElementById('custom-fields-container');
    if (!container) return;

    const fieldId = Date.now(); // Use timestamp as unique ID

    // Create the custom field row
    const fieldRow = document.createElement('div');
    fieldRow.className = 'custom-field-row';
    fieldRow.dataset.fieldId = fieldId;

    // Create label input
    const labelInput = document.createElement('input');
    labelInput.type = 'text';
    labelInput.className = 'form-control';
    labelInput.placeholder = 'Field Name';
    labelInput.addEventListener('input', updateCustomField);

    // Create value input
    const valueInput = document.createElement('input');
    valueInput.type = 'text';
    valueInput.className = 'form-control';
    valueInput.placeholder = 'Value';
    valueInput.addEventListener('input', updateCustomField);

    // Create remove button
    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'remove-field';
    removeButton.textContent = '×';
    removeButton.addEventListener('click', function() {
        removeCustomField(fieldId);
    });

    // Assemble and add to container
    fieldRow.appendChild(labelInput);
    fieldRow.appendChild(valueInput);
    fieldRow.appendChild(removeButton);
    container.appendChild(fieldRow);
}

// Update a custom field in the preview
function updateCustomField(event) {
    const row = event.target.closest('.custom-field-row');
    if (!row) return;

    const fieldId = row.dataset.fieldId;
    const inputs = row.querySelectorAll('input');
    if (inputs.length < 2) return;

    const label = inputs[0].value;
    const value = inputs[1].value || '---';

    // Check if field already exists in preview
    const tableId = 'custom-field-' + fieldId;
    let tableRow = document.getElementById(tableId);

    if (!tableRow && label) {
        // Create new row in preview table
        tableRow = document.createElement('tr');
        tableRow.id = tableId;

        const thCell = document.createElement('th');
        thCell.textContent = label;

        const tdCell = document.createElement('td');
        tdCell.textContent = value;

        tableRow.appendChild(thCell);
        tableRow.appendChild(tdCell);

        document.getElementById('previewInfoTable').appendChild(tableRow);
    } else if (tableRow) {
        // Update existing row
        const cells = tableRow.children;
        if (cells.length >= 2) {
            cells[0].textContent = label;
            cells[1].textContent = value;
        }

        // Remove row if label is empty
        if (!label) {
            tableRow.remove();
        }
    }
}

// Remove a custom field
function removeCustomField(fieldId) {
    // Remove from form
    const fieldRow = document.querySelector(`.custom-field-row[data-field-id="${fieldId}"]`);
    if (fieldRow) {
        fieldRow.remove();
    }

    // Remove from preview
    const tableRow = document.getElementById('custom-field-' + fieldId);
    if (tableRow) {
        tableRow.remove();
    }
}

// Set up icon selection functionality
function setupIconSelection() {
    const iconOptions = document.querySelectorAll('.icon-option');

    // Select first icon by default
    if (iconOptions.length > 0) {
        iconOptions[0].classList.add('selected');
    }

    iconOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from all options
            iconOptions.forEach(opt => opt.classList.remove('selected'));

            // Add selected class to clicked option
            this.classList.add('selected');

            // Update preview
            const icon = this.getAttribute('data-icon');
            document.getElementById('previewIcon').textContent = icon;
        });
    });
}

// Update card accent color
function updateCardColor() {
    const color = document.getElementById('card-color').value;
    document.getElementById('previewIcon').style.backgroundColor = color;
}

// Set up image upload functionality
function setupImageUpload() {
    const imageInput = document.getElementById('equipment-image');
    if (imageInput) {
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                const imageContainer = document.getElementById('previewImageContainer');

                // Create image element
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'equipment-image';
                img.alt = document.getElementById('equipment-name').value || 'Equipment Image';

                // Clear container and add new image
                imageContainer.innerHTML = '';
                imageContainer.className = 'equipment-image';
                imageContainer.appendChild(img);
            };

            reader.readAsDataURL(file);
        });
    }
}

// Update maintenance notes
function updateMaintenanceNotes() {
    const notes = document.getElementById('maintenance-notes').value;
    const schedule = document.getElementById('maintenance-schedule').value;
    updateMaintenanceInfo(notes, schedule);
}

// Update maintenance schedule
function updateMaintenanceSchedule() {
    const notes = document.getElementById('maintenance-notes').value;
    const schedule = document.getElementById('maintenance-schedule').value;
    updateMaintenanceInfo(notes, schedule);
}

// Helper function to update maintenance info in preview
function updateMaintenanceInfo(notes, schedule) {
    const notesContainer = document.getElementById('previewNotes');
    const notesContent = document.getElementById('previewMaintenanceNotes');

    let displayText = notes || 'No maintenance notes available.';

    if (schedule && schedule !== '') {
        if (notes) {
            displayText += ` Scheduled maintenance: ${schedule}.`;
        } else {
            displayText = `Scheduled maintenance: ${schedule}.`;
        }
    }

    notesContent.textContent = displayText;

    // Show/hide notes section
    if (notes || schedule) {
        notesContainer.style.display = 'block';
    } else {
        notesContainer.style.display = 'none';
    }
}

// Save as template
function saveAsTemplate() {
    const data = {
        name: document.getElementById('equipment-name').value || 'Unnamed Equipment',
        id: document.getElementById('equipment-id').value,
        assetNumber: document.getElementById('asset-number').value,
        manufacturer: document.getElementById('manufacturer').value,
        modelNumber: document.getElementById('model-number').value,
        serialNumber: document.getElementById('serial-number').value,
        year: document.getElementById('year').value,
        siteFacility: document.getElementById('site-facility').value,
        area: document.getElementById('area').value,
        lineSection: document.getElementById('line-section').value,
        locationCode: document.getElementById('location-code').value,
        color: document.getElementById('card-color').value,
        maintenanceNotes: document.getElementById('maintenance-notes').value,
        maintenanceSchedule: document.getElementById('maintenance-schedule').value,
        icon: document.querySelector('.icon-option.selected')?.getAttribute('data-icon') || '🔧',
        // Custom fields and image would require special handling
    };

    // Save to localStorage
    const templateName = data.name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const key = 'equipment_template_' + templateName + '_' + Date.now();
    localStorage.setItem(key, JSON.stringify(data));

    alert('Template saved successfully!');
}

// Export as PDF
function exportAsPdf() {
    alert('PDF export functionality would require additional libraries or server-side processing.');
}

// Export as HTML
function exportAsHtml() {
    // Get the preview card
    const card = document.getElementById('previewCard');
    if (!card) return;

    // Create a clone to modify
    const cardClone = card.cloneNode(true);

    // Create HTML document
    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>${document.getElementById('equipment-name').value || 'Equipment Card'}</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f3f5f7;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }
        
        .equipment-card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            width: 100%;
            max-width: 680px;
            padding: 32px 38px 28px 38px;
            display: flex;
            flex-direction: column;
        }
        
        .equipment-header {
            border-bottom: 1px solid #e3e5e8;
            margin-bottom: 22px;
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 18px;
        }
        
        .equipment-icon {
            width: 48px;
            height: 48px;
            background: ${document.getElementById('card-color').value || '#d8ebfa'};
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
        }
        
        .equipment-title {
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin: 0;
        }
        
        .equipment-id {
            font-size: 0.95rem;
            color: #5b7a9a;
            margin-top: -15px;
            margin-bottom: 20px;
            padding-left: 66px;
        }
        
        .equipment-image-container {
            width: 370px;
            height: 260px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 26px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .equipment-image {
            width: 100%;
            height: 100%;
            border-radius: 12px;
            object-fit: contain;
            background: #e0e0e0;
            display: block;
        }
        
        .equipment-image-default {
            width: 100%;
            height: 100%;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #c1c1c1;
            border: 2px dashed #b0b0b0;
            color: #aaa;
            font-size: 2.1rem;
            font-weight: 600;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 18px;
        }
        
        .info-table th, .info-table td {
            text-align: left;
            padding: 9px 7px;
        }
        
        .info-table th {
            color: #5b7a9a;
            font-weight: 600;
            width: 170px;
            background: #f6f8fb;
        }
        
        .info-table tr:nth-child(even) td {
            background: #f9fafc;
        }
        
        .equipment-notes {
            margin-top: 20px;
            background: #f8f8e7;
            border-left: 4px solid #d5c752;
            padding: 15px 18px;
            border-radius: 8px;
            color: #706800;
        }
    </style>
</head>
<body>
    ${cardClone.outerHTML}
</body>
</html>
    `;

    // Create download
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${document.getElementById('equipment-name').value || 'equipment-card'}.html`;
    a.click();

    // Clean up
    URL.revokeObjectURL(url);
}

// Initialize all preview fields
function updateAllPreviewFields() {
    updateEquipmentName();
    updateEquipmentId();
    updateAssetNumber();
    updateManufacturer();
    updateModelNumber();
    updateSerialNumber();
    updateYearOfManufacture();
    updateSiteFacility();
    updateArea();
    updateLineSection();
    updateLocationCode();
    updateCardColor();
    updateMaintenanceNotes();
}