// static/js/tutorial/tour.js
// Tablet-friendly Intro.js tour with safer Bootstrap tab/collapse handling
// Includes tour-active body class so restricted controls can still be tapped during the tour

document.addEventListener('DOMContentLoaded', function () {
    const tourButton = document.getElementById('startTourBtn');
    if (!tourButton) {
        return;
    }

    let originalSearchResultsHTML = null;
    let originalSearchResultsDisplay = null;
    let activeTour = null;

    function getElement(id) {
        return document.getElementById(id);
    }

    function setTourActive(isActive) {
        document.body.classList.toggle('tour-active', !!isActive);
    }

    function getCollapseInstance(id) {
        const el = getElement(id);
        if (!el || typeof bootstrap === 'undefined' || !bootstrap.Collapse) {
            return null;
        }
        return bootstrap.Collapse.getInstance(el) || new bootstrap.Collapse(el, { toggle: false });
    }

    function showCollapse(id) {
        const collapse = getCollapseInstance(id);
        if (collapse) {
            collapse.show();
        }
    }

    function hideCollapse(id) {
        const collapse = getCollapseInstance(id);
        if (collapse) {
            collapse.hide();
        }
    }

    function isShown(id) {
        const el = getElement(id);
        return !!(el && el.classList.contains('show'));
    }

    function showBootstrapTab(tabElement) {
        if (!tabElement) {
            return;
        }

        if (typeof bootstrap !== 'undefined' && bootstrap.Tab) {
            const tab = bootstrap.Tab.getOrCreateInstance(tabElement);
            tab.show();
            return;
        }

        tabElement.click();
    }

    function safeScrollIntoView(element, delay = 150, block = 'center') {
        if (!element) {
            return;
        }

        window.setTimeout(() => {
            element.scrollIntoView({
                behavior: 'smooth',
                block: block
            });
        }, delay);
    }

    function refreshTour(delay = 120) {
        if (!activeTour) {
            return;
        }

        window.setTimeout(() => {
            try {
                activeTour.refresh();
            } catch (error) {
                console.warn('[TOUR] refresh failed:', error);
            }
        }, delay);
    }

    function restoreSearchResults() {
        const searchResults = getElement('pst_searchResults');
        if (!searchResults) {
            return;
        }

        if (originalSearchResultsHTML !== null) {
            searchResults.innerHTML = originalSearchResultsHTML;
        }

        if (originalSearchResultsDisplay !== null) {
            searchResults.style.display = originalSearchResultsDisplay;
        }
    }

    function hideSearchResults() {
        const searchResults = getElement('pst_searchResults');
        if (searchResults) {
            searchResults.style.display = 'none';
        }
    }

    function showSearchResultsPlaceholder() {
        const searchResults = getElement('pst_searchResults');
        if (!searchResults) {
            return;
        }

        if (originalSearchResultsHTML === null) {
            originalSearchResultsHTML = searchResults.innerHTML;
        }
        if (originalSearchResultsDisplay === null) {
            originalSearchResultsDisplay = searchResults.style.display;
        }

        const resultsList = getElement('pst_positionResultsList');
        if (resultsList) {
            resultsList.innerHTML = `
                <li class="list-group-item">
                    <strong>Example Problem</strong> - This is a placeholder problem to demonstrate available actions.
                    <button class="btn btn-sm btn-warning float-end ms-2 update-problem-btn" data-problem-id="placeholder">Update Problem Position</button>
                    <button class="btn btn-sm btn-info float-end ms-2 edit-solutions-btn" data-problem-id="placeholder">Edit Related Solutions</button>
                    <button class="btn btn-sm btn-danger float-end ms-2 delete-problem-btn" data-problem-id="placeholder">Delete Problem</button>
                </li>
            `;
        }

        searchResults.style.display = 'block';
        refreshTour(150);
    }

    function closeProblemAccordions() {
        hideCollapse('collapseSearchProblem');
        hideCollapse('collapseNewProblem');
        refreshTour(250);
    }

    function ensureSearchAccordionOpen() {
        if (!isShown('collapseSearchProblem')) {
            showCollapse('collapseSearchProblem');
            refreshTour(250);
        }
    }

    function ensureNewProblemAccordionOpen() {
        if (!isShown('collapseNewProblem')) {
            showCollapse('collapseNewProblem');
            refreshTour(250);
        }
    }

    function buildDynamicPositionFieldSteps(steps) {
        const dynamicSelectors = [
            ['.areaDropdown', "For each position, first select the <strong>Area</strong> where the equipment is located. This is the highest level in the equipment hierarchy."],
            ['.equipmentGroupDropdown', "Next, select the <strong>Equipment Group</strong>. This dropdown populates based on your Area selection."],
            ['.modelDropdown', "Then select the equipment <strong>Model</strong>. This dropdown populates based on your Equipment Group selection."],
            ['.assetNumberInput', "Select an <strong>Asset Number</strong> if applicable. This identifies a specific piece of equipment within the model type."],
            ['.locationInput', "Select a <strong>Location</strong> to specify where on the equipment this task takes place."],
            ['.assembliesDropdown', "If applicable, select a <strong>Subassembly</strong> to specify which subcomponent the task involves."],
            ['.subassembliesDropdown', "Further refine the location by selecting a <strong>Component Assembly</strong> within the subassembly."],
            ['.assemblyViewsDropdown', "If needed, select an <strong>Assembly View</strong> to specify a particular view or configuration of the component."],
            ['.siteLocationDropdown', "Select a <strong>Site Location</strong> to indicate the physical location where this equipment is installed."],
            ['.savePositionBtn', "After configuring the position details, <strong>CLICK</strong> this button to save the position information."]
        ];

        dynamicSelectors.forEach(([selector, intro]) => {
            const el = document.querySelector(selector);
            if (el) {
                steps.push({
                    element: el,
                    intro: intro,
                    position: 'right'
                });
            }
        });
    }

    function defineTourSteps() {
        const steps = [];

        const container = document.querySelector('.container');
        const problemTab = getElement('problem-tab');
        const solutionTab = getElement('solution-tab');
        const taskTab = getElement('task-tab');
        const editTaskTab = getElement('edit-task-tab');
        const newProblemForm = getElement('newProblemForm');

        if (container) {
            steps.push({
                element: container,
                intro: "<div class='intro-left-text'>Welcome to the PST Troubleshooting System! This tour will guide you through each part of the interface, starting with the Problem tab.</div>",
                position: 'auto'
            });
        }

        if (problemTab) {
            steps.push({
                element: problemTab,
                intro: "Start here in the <strong>Problem tab</strong>. This is where you search for existing problems or create new ones.",
                position: 'right',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(problemTab);
                        refreshTour(150);
                    }, 100);
                }
            });
        }

        const problemAccordion = getElement('problemAccordion');
        if (problemAccordion) {
            steps.push({
                element: problemAccordion,
                intro: "This accordion contains two sections: one for searching existing problems and another for creating new problems.",
                position: 'right'
            });
        }

        const searchAccordionHeader = getElement('headingSearchProblem');
        if (searchAccordionHeader) {
            steps.push({
                element: searchAccordionHeader,
                intro: "<strong>Click </strong> this section to expand the <strong>Search Problem by Position</strong> form.",
                position: 'right',
                onShow: function () {
                    closeProblemAccordions();
                    hideSearchResults();
                }
            });
        }

        const searchForm = getElement('searchProblemByPositionForm');
        if (searchForm) {
            steps.push({
                element: searchForm,
                intro: "Use this form to search for existing problems based on equipment details.",
                position: 'right',
                onShow: function () {
                    ensureSearchAccordionOpen();
                    safeScrollIntoView(searchForm, 250);
                }
            });
        }

        const areaDropdown = getElement('pst_areaDropdown');
        if (areaDropdown) {
            steps.push({
                element: areaDropdown,
                intro: "First, select an <strong>Area</strong> from this dropdown. This is required to start the search process.",
                position: 'right'
            });
        }

        const equipmentGroupDropdown = getElement('pst_equipmentGroupDropdown');
        if (equipmentGroupDropdown) {
            steps.push({
                element: equipmentGroupDropdown,
                intro: "Next, select an <strong>Equipment Group</strong>. This dropdown becomes available after selecting an Area.",
                position: 'right'
            });
        }

        const modelDropdown = getElement('pst_modelDropdown');
        if (modelDropdown) {
            steps.push({
                element: modelDropdown,
                intro: "Then select a <strong>Model</strong>. This dropdown becomes available after selecting an Equipment Group.",
                position: 'right'
            });
        }

        const assetNumberDropdown = getElement('pst_assetNumberDropdown');
        if (assetNumberDropdown) {
            steps.push({
                element: assetNumberDropdown,
                intro: "The <strong>Asset Number</strong> dropdown becomes available after selecting a Model. This field is optional for searching.",
                position: 'right'
            });
        }

        const locationDropdown = getElement('pst_locationDropdown');
        if (locationDropdown) {
            steps.push({
                element: locationDropdown,
                intro: "The <strong>Location</strong> dropdown becomes available after selecting a Model. This field is optional for searching.",
                position: 'right'
            });
        }

        const siteLocationDropdown = getElement('pst_siteLocationDropdown');
        if (siteLocationDropdown) {
            steps.push({
                element: siteLocationDropdown,
                intro: "Select a <strong>Site Location</strong> if needed. You can also create a new Site Location by selecting 'New Site Location...'",
                position: 'right'
            });
        }

        const searchButton = getElement('searchProblemByPositionBtn');
        if (searchButton) {
            steps.push({
                element: searchButton,
                intro: "After filling out the search criteria, <strong>CLICK </strong> this button to find matching problems.",
                position: 'right',
                onShow: function () {
                    ensureSearchAccordionOpen();
                    refreshTour(200);
                }
            });
        }

        const searchResults = getElement('pst_searchResults');
        if (searchResults) {
            steps.push({
                element: searchResults,
                intro: "After searching, the results will appear here. You can then update an existing problem, edit its solutions, or delete it using these buttons.",
                position: 'top',
                onShow: function () {
                    closeProblemAccordions();
                    window.setTimeout(() => {
                        showSearchResultsPlaceholder();
                        safeScrollIntoView(searchResults, 100, 'center');
                    }, 250);
                }
            });
        }

        const newProblemHeader = getElement('headingNewProblem');
        if (newProblemHeader) {
            steps.push({
                element: newProblemHeader,
                intro: "If you can't find an existing problem in the search results, <strong>CLICK </strong>here to expand the <strong>New Problem Form</strong>.",
                position: 'right',
                onShow: function () {
                    hideSearchResults();
                    ensureNewProblemAccordionOpen();
                    safeScrollIntoView(newProblemForm, 350);
                }
            });
        }

        if (newProblemForm) {
            steps.push({
                element: newProblemForm,
                intro: "Use this form to create a new problem when one doesn't already exist.",
                position: 'left',
                onShow: function () {
                    ensureNewProblemAccordionOpen();
                    safeScrollIntoView(newProblemForm, 250);
                }
            });
        }

        const problemName = getElement('problemName');
        if (problemName) {
            steps.push({
                element: problemName,
                intro: "Enter a descriptive <strong>Name</strong> for the problem. This should be concise but informative.",
                position: 'right'
            });
        }

        const problemDescription = getElement('problemDescription');
        if (problemDescription) {
            steps.push({
                element: problemDescription,
                intro: "Provide a detailed <strong>Description</strong> of the problem, including any relevant symptoms or conditions.",
                position: 'right'
            });
        }

        const newAreaDropdown = getElement('new_pst_areaDropdown');
        if (newAreaDropdown) {
            steps.push({
                element: newAreaDropdown,
                intro: "Select the <strong>Area</strong> where the problem occurs.",
                position: 'right'
            });
        }

        const newEquipmentGroupDropdown = getElement('new_pst_equipmentGroupDropdown');
        if (newEquipmentGroupDropdown) {
            steps.push({
                element: newEquipmentGroupDropdown,
                intro: "Select the <strong>Equipment Group</strong> associated with the problem.",
                position: 'right'
            });
        }

        if (newProblemForm) {
            const createButton = newProblemForm.querySelector('button[type="submit"]');
            if (createButton) {
                steps.push({
                    element: createButton,
                    intro: "After filling out all required fields, <strong>CLICK </strong>this button to create the new problem.",
                    position: 'right'
                });
            }
        }

        if (solutionTab) {
            steps.push({
                element: solutionTab,
                intro: "After finding or creating a problem, <strong> CLICK </strong> on the <strong>Solutions tab</strong> to manage solutions.",
                position: 'right',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(solutionTab);
                        restoreSearchResults();
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const selectedProblemName = getElement('selected-problem-name');
        if (selectedProblemName) {
            steps.push({
                element: selectedProblemName,
                intro: "This header shows which problem you're currently working with. It will display the name of the selected problem.",
                position: 'right'
            });
        }

        const existingSolutions = getElement('existing_solutions');
        if (existingSolutions) {
            steps.push({
                element: existingSolutions,
                intro: "This list shows all existing solutions for the selected problem. You can:<br>• <strong>Click</strong> on a solution to select it<br>• <strong>Double-click</strong> a solution to view its tasks<br>• Hold <strong>Ctrl</strong> or <strong>Shift</strong> to select multiple solutions for removal",
                position: 'right'
            });
        }

        const newSolutionName = getElement('new_solution_name');
        if (newSolutionName) {
            steps.push({
                element: newSolutionName,
                intro: "Enter a descriptive <strong>name</strong> for your new solution here. Choose a clear, concise name that describes the approach to solving the problem.",
                position: 'right'
            });
        }

        const newSolutionDescription = getElement('new_solution_description');
        if (newSolutionDescription) {
            steps.push({
                element: newSolutionDescription,
                intro: "Provide a detailed <strong>description</strong> of your solution here. Include any important context or constraints that apply to this solution.",
                position: 'right'
            });
        }

        const addSolutionBtn = getElement('addSolutionBtn');
        if (addSolutionBtn) {
            steps.push({
                element: addSolutionBtn,
                intro: "After entering a name and description, <strong>CLICK </strong>this button to <strong>add the new solution</strong> to the selected problem.",
                position: 'right'
            });
        }

        const removeSolutionsBtn = getElement('removeSolutionsBtn');
        if (removeSolutionsBtn) {
            steps.push({
                element: removeSolutionsBtn,
                intro: "Select one or more solutions from the list above, then <strong>CLICK </strong>this button to <strong>remove</strong> them. You'll be prompted to confirm the deletion.",
                position: 'left'
            });
        }

        if (taskTab) {
            steps.push({
                element: taskTab,
                intro: "Once you've selected a solution, <strong>CLICK </strong>on the <strong>Tasks tab</strong> to manage tasks for that solution.",
                position: 'right',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(taskTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const existingTasks = getElement('existing_tasks');
        if (existingTasks) {
            steps.push({
                element: existingTasks,
                intro: "This list shows all tasks for the selected solution. Double-<strong>CLICK </strong>a task to edit it in detail.",
                position: 'right'
            });
        }

        const newTaskName = getElement('new_task_name');
        if (newTaskName) {
            steps.push({
                element: newTaskName,
                intro: "Add new tasks by providing a name and description, then clicking 'Add Task'.",
                position: 'right'
            });
        }

        if (editTaskTab) {
            steps.push({
                element: editTaskTab,
                intro: "The <strong>Edit Task tab</strong> is where you configure all aspects of a task. You'll access this after creating a task or selecting an existing one to edit.",
                position: 'right',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(editTaskTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const taskName = getElement('pst_task_edit_task_name');
        if (taskName) {
            steps.push({
                element: taskName,
                intro: "Enter a clear, concise <strong>Task Name</strong> that describes what this task accomplishes. Good names help technicians quickly understand the task's purpose.",
                position: 'right'
            });
        }

        const taskDescription = getElement('pst_task_edit_task_description');
        if (taskDescription) {
            steps.push({
                element: taskDescription,
                intro: "Provide a detailed <strong>Task Description</strong> with step-by-step instructions or important context. Be thorough but clear in your explanation.",
                position: 'right'
            });
        }

        const updateTaskBtn = getElement('updateTaskDetailsBtn');
        if (updateTaskBtn) {
            steps.push({
                element: updateTaskBtn,
                intro: "After editing the name or description, <strong>CLICK</strong> this button to save those changes.",
                position: 'right'
            });
        }

        const editTaskSubTabs = getElement('editTaskSubTabs');
        if (editTaskSubTabs) {
            steps.push({
                element: editTaskSubTabs,
                intro: "These sub-tabs organize different types of information for your task. You'll need to work through each tab to fully configure the task.",
                position: 'bottom'
            });
        }

        const taskDetailsTab = getElement('task-details-tab');
        if (taskDetailsTab) {
            steps.push({
                element: taskDetailsTab,
                intro: "The <strong>Task Details</strong> tab is where you define position information - the specific equipment locations this task applies to.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(taskDetailsTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const addPositionBtn = getElement('addPositionBtn');
        if (addPositionBtn) {
            steps.push({
                element: addPositionBtn,
                intro: "<strong>CLICK</strong> this button to add a new position. You can add multiple positions if the task applies to different equipment locations.",
                position: 'right',
                onShow: function () {
                    const positionsContainer = getElement('positionsContainer');
                    if (positionsContainer && positionsContainer.children.length === 0) {
                        window.setTimeout(() => {
                            addPositionBtn.click();
                            refreshTour(250);
                        }, 120);
                    }
                }
            });
        }

        buildDynamicPositionFieldSteps(steps);

        const imagesTab = getElement('task-images-tab');
        if (imagesTab) {
            steps.push({
                element: imagesTab,
                intro: "The <strong>Images</strong> tab allows you to associate relevant images with this task, such as equipment photos or visual guides.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(imagesTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const imagesSelect = getElement('pst_task_edit_task_images');
        if (imagesSelect) {
            steps.push({
                element: imagesSelect,
                intro: "This searchable dropdown lets you find and select images. You can type to search and select multiple images for the task.",
                position: 'right'
            });
        }

        const saveImagesBtn = getElement('saveImagesBtn');
        if (saveImagesBtn) {
            steps.push({
                element: saveImagesBtn,
                intro: "After selecting images, <strong>CLICK</strong> this button to save the image associations to the task.",
                position: 'right'
            });
        }

        const selectedImages = getElement('pst_task_edit_selected_images');
        if (selectedImages) {
            steps.push({
                element: selectedImages,
                intro: "This area displays all images currently associated with the task. Each image has a 'Remove' button to delete associations if needed.",
                position: 'right'
            });
        }

        const partsTab = getElement('task-parts-tab');
        if (partsTab) {
            steps.push({
                element: partsTab,
                intro: "The <strong>Parts</strong> tab lets you associate parts that are needed or affected by this task. This helps technicians know which parts to have on hand.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(partsTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const partsSelect = getElement('pst_task_edit_task_parts');
        if (partsSelect) {
            steps.push({
                element: partsSelect,
                intro: "This searchable dropdown lets you find and select parts by part number or name. You can select multiple parts for the task.",
                position: 'right'
            });
        }

        const savePartsBtn = getElement('savePartsBtn');
        if (savePartsBtn) {
            steps.push({
                element: savePartsBtn,
                intro: "After selecting parts, <strong>CLICK</strong> this button to save the part associations to the task.",
                position: 'right'
            });
        }

        const selectedParts = getElement('pst_task_edit_selected_parts');
        if (selectedParts) {
            steps.push({
                element: selectedParts,
                intro: "This area displays all parts currently associated with the task, showing part numbers and names for quick reference.",
                position: 'right'
            });
        }

        const drawingsTab = getElement('task-drawings-tab');
        if (drawingsTab) {
            steps.push({
                element: drawingsTab,
                intro: "The <strong>Drawings</strong> tab lets you associate technical drawings that are relevant to completing this task.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(drawingsTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const drawingsSelect = getElement('pst_task_edit_task_drawings');
        if (drawingsSelect) {
            steps.push({
                element: drawingsSelect,
                intro: "This searchable dropdown lets you find and select technical drawings by number or name. You can select multiple drawings.",
                position: 'right'
            });
        }

        const saveDrawingsBtn = getElement('saveDrawingsBtn');
        if (saveDrawingsBtn) {
            steps.push({
                element: saveDrawingsBtn,
                intro: "After selecting drawings, <strong>CLICK</strong> this button to save the drawing associations to the task.",
                position: 'right'
            });
        }

        const selectedDrawings = getElement('pst_task_edit_selected_drawings');
        if (selectedDrawings) {
            steps.push({
                element: selectedDrawings,
                intro: "This area displays all drawings currently associated with the task, showing drawing numbers and names for quick reference.",
                position: 'right'
            });
        }

        const documentsTab = getElement('task-documents-tab');
        if (documentsTab) {
            steps.push({
                element: documentsTab,
                intro: "The <strong>Documents</strong> tab lets you associate reference documents needed for completing this task, such as manuals or procedures.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(documentsTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const documentsSelect = getElement('pst_task_edit_task_documents');
        if (documentsSelect) {
            steps.push({
                element: documentsSelect,
                intro: "This searchable dropdown lets you find and select documents by title or content. You can select multiple documents.",
                position: 'right'
            });
        }

        const saveDocumentsBtn = getElement('saveDocumentsBtn');
        if (saveDocumentsBtn) {
            steps.push({
                element: saveDocumentsBtn,
                intro: "After selecting documents, <strong>CLICK</strong> this button to save the document associations to the task.",
                position: 'right'
            });
        }

        const selectedDocuments = getElement('pst_task_edit_selected_documents');
        if (selectedDocuments) {
            steps.push({
                element: selectedDocuments,
                intro: "This area displays all documents currently associated with the task, showing document titles for quick reference.",
                position: 'right'
            });
        }

        const toolsTab = getElement('task-tools-tab');
        if (toolsTab) {
            steps.push({
                element: toolsTab,
                intro: "The <strong>Tools</strong> tab lets you specify what tools are required to complete this task. This helps technicians prepare properly.",
                position: 'bottom',
                onShow: function () {
                    window.setTimeout(() => {
                        showBootstrapTab(toolsTab);
                        refreshTour(200);
                    }, 120);
                }
            });
        }

        const toolsSelect = getElement('pst_task_edit_task_tools');
        if (toolsSelect) {
            steps.push({
                element: toolsSelect,
                intro: "This searchable dropdown lets you find and select tools by name or type. You can select multiple tools that are needed for this task.",
                position: 'right'
            });
        }

        const saveToolsBtn = getElement('saveToolsBtn');
        if (saveToolsBtn) {
            steps.push({
                element: saveToolsBtn,
                intro: "After selecting tools, <strong>CLICK</strong> this button to save the tool associations to the task.",
                position: 'right'
            });
        }

        const selectedTools = getElement('pst_task_edit_selected_tools');
        if (selectedTools) {
            steps.push({
                element: selectedTools,
                intro: "This area displays all tools currently associated with the task, showing tool names and types for quick reference.",
                position: 'right'
            });
        }

        if (editTaskTab) {
            steps.push({
                element: editTaskTab,
                intro: "For a complete task, make sure to configure the position details and add any relevant images, parts, drawings, documents, and tools before moving on.",
                position: 'bottom'
            });
        }

        if (container) {
            steps.push({
                element: container,
                intro: "That's it! Remember to work left-to-right: Problem → Solution → Task → Edit Task. <strong>CLICK </strong>'Done' to finish the tour.",
                position: 'right',
                onShow: function () {
                    if (problemTab) {
                        window.setTimeout(() => {
                            showBootstrapTab(problemTab);
                            restoreSearchResults();
                            closeProblemAccordions();
                            refreshTour(250);
                        }, 100);
                    }
                }
            });
        }

        return steps;
    }

    function updateCustomStepCounter(tourInstance) {
        const tooltipText = document.querySelector('.introjs-tooltiptext');
        if (!tooltipText || !tourInstance || !tourInstance._options || !tourInstance._options.steps) {
            return;
        }

        const existingCounter = document.getElementById('custom-step-counter');
        if (existingCounter) {
            existingCounter.remove();
        }

        const currentStep = tourInstance._currentStep + 1;
        const totalSteps = tourInstance._options.steps.length;

        const counterDiv = document.createElement('div');
        counterDiv.id = 'custom-step-counter';

        const stepSpan = document.createElement('span');
        stepSpan.textContent = currentStep;

        const ofSpan = document.createElement('span');
        ofSpan.textContent = ' of ';

        const totalSpan = document.createElement('span');
        totalSpan.textContent = totalSteps;

        counterDiv.appendChild(stepSpan);
        counterDiv.appendChild(ofSpan);
        counterDiv.appendChild(totalSpan);

        tooltipText.appendChild(counterDiv);
    }

    function normalizeTooltipButtons() {
        const skipButton = document.querySelector('.introjs-skipbutton');
        const tooltipButtons = document.querySelector('.introjs-tooltipbuttons');

        if (!tooltipButtons) {
            return;
        }

        const outsideSkipButton = document.querySelector('body > .introjs-skipbutton');
        if (outsideSkipButton) {
            outsideSkipButton.remove();
        }

        if (skipButton && !tooltipButtons.contains(skipButton)) {
            tooltipButtons.appendChild(skipButton);
        }

        tooltipButtons.style.display = 'flex';
        tooltipButtons.style.justifyContent = 'space-between';
        tooltipButtons.style.alignItems = 'center';
        tooltipButtons.style.width = '100%';

        const backButton = document.querySelector('.introjs-prevbutton');
        const nextButton = document.querySelector('.introjs-nextbutton');
        const doneButton = document.querySelector('.introjs-donebutton');

        if (backButton) {
            backButton.style.order = '0';
            backButton.style.marginRight = 'auto';
            tooltipButtons.insertBefore(backButton, tooltipButtons.firstChild);
        }

        if (skipButton) {
            skipButton.classList.add('btn', 'btn-danger', 'btn-sm');
            skipButton.style.order = '1';
            skipButton.style.margin = '0 10px';
        }

        const rightButton = doneButton || nextButton;
        if (rightButton) {
            rightButton.style.order = '2';
            rightButton.style.marginLeft = 'auto';
            tooltipButtons.appendChild(rightButton);
        }
    }

    function cleanupTourState() {
        restoreSearchResults();
        closeProblemAccordions();
        setTourActive(false);
        activeTour = null;
    }

    tourButton.addEventListener('click', function () {
        setTourActive(true);

        const tour = introJs();
        activeTour = tour;

        tour.setOptions({
            steps: defineTourSteps(),
            nextLabel: 'Next →',
            prevLabel: '← Back',
            skipLabel: 'Skip tour',
            doneLabel: 'Done',
            hideNext: false,
            hidePrev: false,
            exitOnOverlayClick: false,
            showStepNumbers: false,
            keyboardNavigation: true,
            showButtons: true,
            showBullets: true,
            scrollToElement: true,
            scrollPadding: 30,
            disableInteraction: false,
            tooltipPosition: 'auto',
            positionPrecedence: ['bottom', 'top', 'right', 'left'],
            tooltipClass: 'custom-tooltip',
            skipButtonClass: 'centered-skip'
        });

        tour.onafterchange(function () {
            window.setTimeout(() => {
                updateCustomStepCounter(this);
                normalizeTooltipButtons();
                refreshTour(50);
            }, 80);
        });

        tour.onexit(function () {
            cleanupTourState();
        });

        tour.oncomplete(function () {
            cleanupTourState();
        });

        tour.start();
    });
});