// static/js/tool/tool_search.js

document.addEventListener('DOMContentLoaded', () => {
    initializeToolSearch();
    
    // Add CSS styles for tool images dynamically
    addToolImageStyles();
});

/**
 * Initializes the tool search functionality by setting up event listeners.
 */
function initializeToolSearch() {
    const searchForm = document.getElementById('tool_search_form');
    const resultsContainer = document.getElementById('search-results-container');

    if (!searchForm) {
        console.warn('Search form not found.');
        return;
    }

    if (!resultsContainer) {
        console.warn('Search results container not found.');
        return;
    }

    // Optional: Add a loading spinner element
    const loadingSpinner = createLoadingSpinner();
    resultsContainer.appendChild(loadingSpinner);

    searchForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // Prevent default form submission

        // Show the loading spinner
        loadingSpinner.style.display = 'block';
        resultsContainer.innerHTML = ''; // Clear existing results
        resultsContainer.appendChild(loadingSpinner);

        // Get form data and convert field names to match backend expectations
        const formData = new FormData(searchForm);
        const searchParams = new URLSearchParams();

        // Map form field names to expected backend parameter names
        const toolName = formData.get('tool_name');
        if (toolName) searchParams.append('name', toolName);

        const toolMaterial = formData.get('tool_material');
        if (toolMaterial) searchParams.append('material', toolMaterial);

        const toolCategoryId = formData.get('tool_category_id');
        if (toolCategoryId) searchParams.append('category_id', toolCategoryId);

        const toolManufacturerId = formData.get('tool_manufacturer_id');
        if (toolManufacturerId) searchParams.append('manufacturer_id', toolManufacturerId);

        // Add CSRF token if present
        const csrfToken = formData.get('csrf_token');
        if (csrfToken) searchParams.append('csrf_token', csrfToken);

        // Add pagination parameters
        searchParams.append('page', 1);
        searchParams.append('per_page', 10);

        try {
            // Try both possible endpoint URLs
            let response;
            let url;

            // First try with tool prefix (most likely based on your logs)
            try {
                url = `/tool/tools_search?${searchParams.toString()}`;
                console.log(`Attempting to fetch from: ${url}`);
                response = await fetch(url, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status ${response.status}`);
                }
                
                // Store the successful URL for pagination
                window.lastSuccessfulToolSearchUrl = url;
            } catch (error) {
                console.log(`First attempt failed with: ${error.message}`);
                // Try without tool prefix as fallback
                url = `/tools_search?${searchParams.toString()}`;
                console.log(`Attempting to fetch from: ${url}`);
                response = await fetch(url, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status ${response.status}`);
                }
                
                // Store the successful URL for pagination
                window.lastSuccessfulToolSearchUrl = url;
            }

            const data = await response.json();
            console.log('Received data:', data);

            // Hide the loading spinner
            loadingSpinner.style.display = 'none';

            // Process response data
            displaySearchResults(data.tools, resultsContainer);
            displayPaginationControls(data.total, data.page, data.per_page, searchParams);
        } catch (error) {
            console.error('Error fetching tools:', error);
            loadingSpinner.style.display = 'none';
            displayErrorMessage(resultsContainer, 'An error occurred while searching for tools. Please try again later.');
        }
    });
}

/**
 * Creates a loading spinner element.
 * @returns {HTMLElement} - The loading spinner element.
 */
function createLoadingSpinner() {
    const spinnerDiv = document.createElement('div');
    spinnerDiv.className = 'd-flex justify-content-center my-3';
    spinnerDiv.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    spinnerDiv.style.display = 'none'; // Hidden by default
    return spinnerDiv;
}

/**
 * Formats the image URL to use the serve_image route.
 * @param {Object} img - The image object with id and file_path.
 * @returns {string} - The URL to access the image.
 */
function formatImageUrl(img) {
    if (!img || !img.id) return '';

    // Use the serve_image route with the image ID
    return `/serve_image/${img.id}`;
}

/**
 * Displays the search results in the designated container.
 * @param {Array} tools - Array of tool objects returned from the server.
 * @param {HTMLElement} container - The DOM element to display the results.
 */
function displaySearchResults(tools, container) {
    if (!Array.isArray(tools) || tools.length === 0) {
        container.innerHTML = '<p>No tools found matching your criteria.</p>';
        return;
    }

    // Create a table to display tools
    const table = document.createElement('table');
    table.className = 'table table-bordered table-striped';

    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Tool Name</th>
            <th>Size</th>
            <th>Type</th>
            <th>Material</th>
            <th>Category</th>
            <th>Manufacturer</th>
            <th>Description</th>
            <th>Images</th>
        </tr>
    `;
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    tools.forEach(tool => {
        const tr = document.createElement('tr');

        // Generate the image gallery HTML
        const imageGalleryHtml = generateImageGallery(tool);

        tr.innerHTML = `
            <td>${escapeHTML(tool.name)}</td>
            <td>${escapeHTML(tool.size) || 'N/A'}</td>
            <td>${escapeHTML(tool.type) || 'N/A'}</td>
            <td>${escapeHTML(tool.material) || 'N/A'}</td>
            <td>${escapeHTML(tool.category) || 'N/A'}</td>
            <td>${escapeHTML(tool.manufacturer) || 'N/A'}</td>
            <td>${escapeHTML(tool.description) || 'N/A'}</td>
            <td>${imageGalleryHtml}</td>
        `;

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    container.appendChild(table);

    // Initialize image previews after adding to DOM
    initializeImagePreviews();
}

/**
 * Generates HTML for an image gallery for a tool.
 * @param {Object} tool - The tool object containing image data.
 * @returns {string} - HTML for the image gallery.
 */
function generateImageGallery(tool) {
    // Check if the tool has images
    if (!tool.images || tool.images.length === 0) {
        return `<div class="no-image-placeholder">
                    <span class="material-icons">image_not_supported</span>
                    <span>No images</span>
                </div>`;
    }

    // If there's only one image, display it directly
    if (tool.images.length === 1) {
        const img = tool.images[0];
        return generateSingleImageHtml(img, tool.name);
    }

    // If there are multiple images, create a thumbnail gallery
    return generateImageGalleryHtml(tool.images, tool.name);
}

/**
 * Generates HTML for a single image.
 * @param {Object} img - The image object.
 * @param {string} toolName - The name of the tool.
 * @returns {string} - HTML for the single image.
 */
function generateSingleImageHtml(img, toolName) {
    const imageUrl = formatImageUrl(img);

    return `
        <div class="tool-image-container">
            <img src="${escapeAttribute(imageUrl)}" 
                 alt="${escapeAttribute(img.title || toolName)}" 
                 title="${escapeAttribute(img.description || '')}"
                 class="tool-image preview-image"
                 data-image-id="${escapeAttribute(img.id)}"
                 data-image-title="${escapeAttribute(img.title || toolName)}"
                 data-image-description="${escapeAttribute(img.description || '')}"
                 data-image-path="${escapeAttribute(imageUrl)}"
                 onerror="this.onerror=null; this.src='/static/img/no-image.png';">
        </div>
    `;
}

/**
 * Generates HTML for a thumbnail gallery.
 * @param {Array} images - Array of image objects.
 * @param {string} toolName - The name of the tool.
 * @returns {string} - HTML for the thumbnail gallery.
 */
function generateImageGalleryHtml(images, toolName) {
    const thumbnails = images.map((img, index) => {
        const imageUrl = formatImageUrl(img);

        return `
        <div class="tool-thumbnail-container">
            <img src="${escapeAttribute(imageUrl)}" 
                 alt="${escapeAttribute(img.title || toolName)}" 
                 title="${escapeAttribute(img.description || '')}"
                 class="tool-thumbnail preview-image"
                 data-image-id="${escapeAttribute(img.id)}"
                 data-image-title="${escapeAttribute(img.title || toolName)}"
                 data-image-description="${escapeAttribute(img.description || '')}"
                 data-image-path="${escapeAttribute(imageUrl)}"
                 onerror="this.onerror=null; this.src='/static/img/no-image.png';">
            <span class="thumbnail-number">${index + 1}</span>
        </div>
        `;
    }).join('');
    
    return `
        <div class="tool-gallery-container">
            ${thumbnails}
            <span class="image-count-badge">${images.length} images</span>
        </div>
    `;
}

/**
 * Initializes click events for image previews.
 */
function initializeImagePreviews() {
    // Add click event to all preview-image elements
    document.querySelectorAll('.preview-image').forEach(image => {
        image.addEventListener('click', function() {
            const imageId = this.getAttribute('data-image-id');
            const imageTitle = this.getAttribute('data-image-title');
            const imageDescription = this.getAttribute('data-image-description');
            const imagePath = this.getAttribute('data-image-path');
            
            showImagePreviewModal(imageId, imageTitle, imageDescription, imagePath);
        });
    });
}

/**
 * Creates and displays a modal for image preview.
 * @param {string} imageId - The ID of the image.
 * @param {string} imageTitle - The title of the image.
 * @param {string} imageDescription - The description of the image.
 * @param {string} imagePath - The path to the image file.
 */
function showImagePreviewModal(imageId, imageTitle, imageDescription, imagePath) {
    // Remove any existing modals
    const existingModal = document.querySelector('.modal-overlay');
    if (existingModal) {
        document.body.removeChild(existingModal);
    }
    
    // Create modal elements
    const modalOverlay = document.createElement('div');
    modalOverlay.className = 'modal-overlay';
    
    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';
    
    // Create close button
    const closeButton = document.createElement('button');
    closeButton.className = 'modal-close-button';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', () => {
        document.body.removeChild(modalOverlay);
    });
    
    // Create image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'modal-image-container';
    
    // Create image element
    const image = document.createElement('img');
    // Use the correct image path that was passed from the thumbnail/image
    image.src = imagePath;
    image.alt = imageTitle;
    image.className = 'modal-image';
    image.onerror = function() {
        this.onerror = null;
        this.src = '/static/img/no-image.png';
    };
    
    // Create image info container
    const imageInfo = document.createElement('div');
    imageInfo.className = 'modal-image-info';
    
    // Create image title
    const title = document.createElement('h3');
    title.textContent = imageTitle;
    title.className = 'modal-image-title';
    
    // Create image description
    const description = document.createElement('p');
    description.textContent = imageDescription || 'No description available';
    description.className = 'modal-image-description';
    
    // Add elements to the DOM
    imageInfo.appendChild(title);
    imageInfo.appendChild(description);
    
    imageContainer.appendChild(image);
    
    modalContent.appendChild(closeButton);
    modalContent.appendChild(imageContainer);
    modalContent.appendChild(imageInfo);
    
    modalOverlay.appendChild(modalContent);
    
    document.body.appendChild(modalOverlay);
    
    // Close modal when clicking outside the content
    modalOverlay.addEventListener('click', (event) => {
        if (event.target === modalOverlay) {
            document.body.removeChild(modalOverlay);
        }
    });
    
    // Close modal with escape key
    document.addEventListener('keydown', function escapeListener(event) {
        if (event.key === 'Escape') {
            if (document.body.contains(modalOverlay)) {
                document.body.removeChild(modalOverlay);
            }
            document.removeEventListener('keydown', escapeListener);
        }
    });
}

/**
 * Displays pagination controls based on the total number of tools.
 * @param {number} total - Total number of matching tools.
 * @param {number} currentPage - Current page number.
 * @param {number} perPage - Number of tools per page.
 * @param {URLSearchParams} searchParams - The search parameters.
 */
function displayPaginationControls(total, currentPage, perPage, searchParams) {
    const container = document.getElementById('search-results-container');
    const totalPages = Math.ceil(total / perPage);

    if (totalPages <= 1) return; // No need for pagination

    const paginationNav = document.createElement('nav');
    paginationNav.setAttribute('aria-label', 'Search results pages');

    const ul = document.createElement('ul');
    ul.className = 'pagination justify-content-center';

    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Previous" data-page="${currentPage - 1}">
            <span aria-hidden="true">&laquo; Previous</span>
        </a>
    `;
    ul.appendChild(prevLi);

    // Current page indicator
    const currentLi = document.createElement('li');
    currentLi.className = 'page-item disabled';
    currentLi.innerHTML = `
        <span class="page-link">
            Page ${currentPage} of ${totalPages}
        </span>
    `;
    ul.appendChild(currentLi);

    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Next" data-page="${currentPage + 1}">
            <span aria-hidden="true">Next &raquo;</span>
        </a>
    `;
    ul.appendChild(nextLi);

    paginationNav.appendChild(ul);
    container.appendChild(paginationNav);

    // Store the working URL to use for pagination
    const workingUrl = window.lastSuccessfulToolSearchUrl || '/tool/tools_search';

    // Add event listeners to pagination links
    paginationNav.addEventListener('click', async function(event) {
        event.preventDefault();
        const target = event.target.closest('a');
        if (!target) return;

        const selectedPage = parseInt(target.getAttribute('data-page'));
        if (isNaN(selectedPage) || selectedPage < 1 || selectedPage > totalPages) return;

        // Optional: Show loading spinner or disable pagination
        const loadingSpinner = createLoadingSpinner();
        container.innerHTML = ''; // Clear existing results
        container.appendChild(loadingSpinner);
        loadingSpinner.style.display = 'block';

        try {
            // Update page number in search params
            searchParams.set('page', selectedPage);

            const response = await fetch(`${workingUrl}?${searchParams.toString()}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            // Clear existing results and pagination
            container.innerHTML = '';

            displaySearchResults(data.tools, container);
            displayPaginationControls(data.total, data.page, data.per_page, searchParams);
        } catch (error) {
            console.error('Error fetching tools:', error);
            displayErrorMessage(container, 'An error occurred while fetching the next page. Please try again later.');
        } finally {
            // Hide the loading spinner if it's still in the DOM
            if (loadingSpinner.parentNode === container) {
                loadingSpinner.style.display = 'none';
            }
        }
    });
}

/**
 * Adds CSS styles for tool images to the document.
 */
function addToolImageStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        /* Tool Image Styles */
        .tool-image-container {
            width: 120px;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
            background-color: #f8f9fa;
        }

        .tool-image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .tool-image:hover {
            transform: scale(1.05);
        }

        .tool-gallery-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-width: 250px;
            position: relative;
        }

        .tool-thumbnail-container {
            width: 70px;
            height: 70px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            background-color: #f8f9fa;
        }

        .tool-thumbnail {
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .tool-thumbnail:hover {
            transform: scale(1.1);
        }

        .thumbnail-number {
            position: absolute;
            bottom: 0;
            right: 0;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            font-size: 10px;
            padding: 2px 4px;
            border-top-left-radius: 4px;
        }

        .image-count-badge {
            position: absolute;
            top: -10px;
            right: -10px;
            background-color: #0d6efd;
            color: white;
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: bold;
        }

        .no-image-placeholder {
            width: 100px;
            height: 100px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1px dashed #dee2e6;
            border-radius: 4px;
            color: #6c757d;
            font-size: 0.8rem;
            background-color: #f8f9fa;
        }

        .no-image-placeholder span.material-icons {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }

        .modal-content {
            background-color: white;
            border-radius: 8px;
            width: 90%;
            max-width: 800px;
            max-height: 90vh;
            overflow: auto;
            position: relative;
            display: flex;
            flex-direction: column;
        }

        .modal-close-button {
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 24px;
            background: none;
            border: none;
            color: #333;
            cursor: pointer;
            z-index: 10;
        }

        .modal-image-container {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f8f9fa;
            padding: 20px;
            max-height: 70vh;
        }

        .modal-image {
            max-width: 100%;
            max-height: 60vh;
            object-fit: contain;
        }

        .modal-image-info {
            padding: 15px 20px;
            border-top: 1px solid #dee2e6;
        }

        .modal-image-title {
            margin: 0 0 10px 0;
            font-size: 1.2rem;
            color: #212529;
        }

        .modal-image-description {
            margin: 0;
            color: #6c757d;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .tool-gallery-container {
                max-width: 190px;
            }
            
            .tool-thumbnail-container {
                width: 55px;
                height: 55px;
            }
            
            .modal-content {
                width: 95%;
            }
        }
    `;
    
    document.head.appendChild(styleElement);
}

/**
 * Displays an error message within the results container.
 * @param {HTMLElement} container - The DOM element to display the error message.
 * @param {string} message - The error message to display.
 */
function displayErrorMessage(container, message) {
    container.innerHTML = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${escapeHTML(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
}

/**
 * Escapes HTML to prevent XSS attacks.
 * @param {string} unsafe - The string to escape.
 * @returns {string} - The escaped string.
 */
function escapeHTML(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Escapes attribute values to prevent XSS attacks.
 * @param {string} unsafe - The attribute value to escape.
 * @returns {string} - The escaped attribute value.
 */
function escapeAttribute(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}