// tour.js - Enhanced version with detailed Problem and Solutions tab guidance
document.addEventListener('DOMContentLoaded', function () {
    // Get the tour button
    const tourButton = document.getElementById('startTourBtn');

    // Store original states for restoration
    let originalSearchResultsHTML = null;
    let originalSearchResultsDisplay = null;

    // Define tour steps
    function defineTourSteps() {
        const steps = [];

        // Welcome step
        steps.push({
            element: document.querySelector('.container'),
            intro: "<div class='intro-left-text'>Welcome to the PST Troubleshooting System! This tour will guide you through each part of the " +
                " interface, starting with the Problem tab.</div>",
            position: 'auto'
        });

        // ===== PROBLEM TAB =====
        const problemTab = document.getElementById('problem-tab');
        if (problemTab) {
            steps.push({
                element: problemTab,
                intro: "Start here in the <strong>Problem tab</strong>. This is where you search for existing problems or create new ones.",
                position: 'Right'
            });

            // Problem Accordion Overview
            const problemAccordion = document.getElementById('problemAccordion');
            if (problemAccordion) {
                steps.push({
                    element: problemAccordion,
                    intro: "This accordion contains two sections: one for searching existing problems and another for creating new problems.",
                    position: 'Right'
                });
            }

            // Search Problem Section Header
            const searchAccordionHeader = document.getElementById('headingSearchProblem');
            if (searchAccordionHeader) {
                steps.push({
                    element: searchAccordionHeader,
                    intro: "<strong>Click </strong> this section to expand the <strong>Search Problem by Position</strong> form.",
                    position: 'right',
                    onShow: function() {
                        // Make sure both accordions are closed initially
                        const collapseSearch = bootstrap.Collapse.getInstance(document.getElementById('collapseSearchProblem')) ||
                                              new bootstrap.Collapse(document.getElementById('collapseSearchProblem'), {toggle: false});
                        const collapseNew = bootstrap.Collapse.getInstance(document.getElementById('collapseNewProblem')) ||
                                           new bootstrap.Collapse(document.getElementById('collapseNewProblem'), {toggle: false});

                        collapseSearch.hide();
                        collapseNew.hide();

                        // Hide search results if they were shown in a previous step
                        const searchResults = document.getElementById('pst_searchResults');
                        if (searchResults) {
                            searchResults.style.display = 'none';
                        }
                    }
                });
            }

            // Search Problem Form
            const searchForm = document.getElementById('searchProblemByPositionForm');
            if (searchForm) {
                steps.push({
                    element: searchForm,
                    intro: "Use this form to search for existing problems based on equipment details.",
                    position: 'right',
                    onShow: function() {
                        // Expand the search accordion when reaching this step
                        const collapseSearch = bootstrap.Collapse.getInstance(document.getElementById('collapseSearchProblem')) ||
                                              new bootstrap.Collapse(document.getElementById('collapseSearchProblem'), {toggle: false});

                        if (!document.getElementById('collapseSearchProblem').classList.contains('show')) {
                            collapseSearch.show();
                        }
                    }
                });
            }

            // Area Dropdown in Search Form
            const areaDropdown = document.getElementById('pst_areaDropdown');
            if (areaDropdown) {
                steps.push({
                    element: areaDropdown,
                    intro: "First, select an <strong>Area</strong> from this dropdown. This is required to start the search process.",
                    position: 'right'
                });
            }

            // Equipment Group Dropdown in Search Form
            const equipmentGroupDropdown = document.getElementById('pst_equipmentGroupDropdown');
            if (equipmentGroupDropdown) {
                steps.push({
                    element: equipmentGroupDropdown,
                    intro: "Next, select an <strong>Equipment Group</strong>. This dropdown becomes available after selecting an Area.",
                    position: 'right'
                });
            }

            // Model Dropdown in Search Form
            const modelDropdown = document.getElementById('pst_modelDropdown');
            if (modelDropdown) {
                steps.push({
                    element: modelDropdown,
                    intro: "Then select a <strong>Model</strong>. This dropdown becomes available after selecting an Equipment Group.",
                    position: 'right'
                });
            }

            // Asset Number Dropdown in Search Form
            const assetNumberDropdown = document.getElementById('pst_assetNumberDropdown');
            if (assetNumberDropdown) {
                steps.push({
                    element: assetNumberDropdown,
                    intro: "The <strong>Asset Number</strong> dropdown becomes available after selecting a Model. This field is optional for searching.",
                    position: 'right'
                });
            }

            // Location Dropdown in Search Form
            const locationDropdown = document.getElementById('pst_locationDropdown');
            if (locationDropdown) {
                steps.push({
                    element: locationDropdown,
                    intro: "The <strong>Location</strong> dropdown becomes available after selecting a Model. This field is optional for searching.",
                    position: 'right'
                });
            }

            // Site Location Dropdown in Search Form
            const siteLocationDropdown = document.getElementById('pst_siteLocationDropdown');
            if (siteLocationDropdown) {
                steps.push({
                    element: siteLocationDropdown,
                    intro: "Select a <strong>Site Location</strong> if needed. You can also create a new Site Location by selecting 'New Site Location...'",
                    position: 'right'
                });
            }

            // Search Button
            const searchButton = document.getElementById('searchProblemByPositionBtn');
            if (searchButton) {
                steps.push({
                    element: searchButton,
                    intro: "After filling out the search criteria, <strong>CLICK </strong> this button to find matching problems.",
                    position: 'right',
                    onShow: function() {
                        // Keep the search accordion open when showing the search button
                        const collapseSearch = bootstrap.Collapse.getInstance(document.getElementById('collapseSearchProblem')) ||
                                              new bootstrap.Collapse(document.getElementById('collapseSearchProblem'), {toggle: false});
                        if (!document.getElementById('collapseSearchProblem').classList.contains('show')) {
                            collapseSearch.show();
                        }
                    }
                });
            }

            // Search Results Section - MOVED HERE to appear right after the Search Button
            const searchResults = document.getElementById('pst_searchResults');
            if (searchResults) {
                steps.push({
                    element: searchResults,
                    intro: "After searching, the results will appear here. You can then update an existing problem, edit its solutions, or delete it using these buttons.",
                    position: 'top',
                    onShow: function() {
                        // Hide search accordion when showing search results
                        const collapseSearch = bootstrap.Collapse.getInstance(document.getElementById('collapseSearchProblem')) ||
                                              new bootstrap.Collapse(document.getElementById('collapseSearchProblem'), {toggle: false});
                        const collapseNew = bootstrap.Collapse.getInstance(document.getElementById('collapseNewProblem')) ||
                                           new bootstrap.Collapse(document.getElementById('collapseNewProblem'), {toggle: false});

                        collapseSearch.hide();
                        collapseNew.hide();

                        // Save original state
                        originalSearchResultsHTML = searchResults.innerHTML;
                        originalSearchResultsDisplay = searchResults.style.display;

                        // Add placeholder example problem with buttons
                        const resultsList = document.getElementById('pst_positionResultsList');
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

                        // Make search results visible for the tour
                        searchResults.style.display = 'block';
                    }
                });
            }

            // Get the New Problem Form element reference early
            const newProblemForm = document.getElementById('newProblemForm');

            // New Problem Section Header - NOW AFTER search results
            const newProblemHeader = document.getElementById('headingNewProblem');
            if (newProblemHeader) {
                steps.push({
                    element: newProblemHeader,
                    intro: "If you can't find an existing problem in the search results, <strong>CLICK </strong>here to expand the <strong>New Problem Form</strong>.",
                    position: 'right',
                    onShow: function() {
                        // Hide search results when moving to New Problem form
                        const searchResults = document.getElementById('pst_searchResults');
                        if (searchResults) {
                            searchResults.style.display = 'none';
                        }

                        // Make sure the New Problem accordion is expanded
                        const collapseNew = bootstrap.Collapse.getInstance(document.getElementById('collapseNewProblem')) ||
                                          new bootstrap.Collapse(document.getElementById('collapseNewProblem'), {toggle: false});

                        // Show the accordion
                        collapseNew.show();

                        // Add a small delay to ensure accordion is fully expanded before scrolling
                        setTimeout(() => {
                            // Scroll to make sure the form is visible
                            if (newProblemForm) {
                                newProblemForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            }
                        }, 400); // Adjust timing to match Bootstrap's collapse animation duration
                    }
                });
            }

            // New Problem Form Step
            if (newProblemForm) {
                steps.push({
                    element: newProblemForm,
                    intro: "Use this form to create a new problem when one doesn't already exist.",
                    position: 'left',
                    onShow: function() {
                        // Double-check that the accordion is still open
                        const collapseNew = bootstrap.Collapse.getInstance(document.getElementById('collapseNewProblem')) ||
                                          new bootstrap.Collapse(document.getElementById('collapseNewProblem'), {toggle: false});

                        if (!document.getElementById('collapseNewProblem').classList.contains('show')) {
                            collapseNew.show();
                        }

                        // Ensure the form is scrolled into view
                        newProblemForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                });
            }

            // Problem Name Input
            const problemName = document.getElementById('problemName');
            if (problemName) {
                steps.push({
                    element: problemName,
                    intro: "Enter a descriptive <strong>Name</strong> for the problem. This should be concise but informative.",
                    position: 'right'
                });
            }

            // Problem Description Textarea
            const problemDescription = document.getElementById('problemDescription');
            if (problemDescription) {
                steps.push({
                    element: problemDescription,
                    intro: "Provide a detailed <strong>Description</strong> of the problem, including any relevant symptoms or conditions.",
                    position: 'right'
                });
            }

            // New Area Dropdown in New Problem Form
            const newAreaDropdown = document.getElementById('new_pst_areaDropdown');
            if (newAreaDropdown) {
                steps.push({
                    element: newAreaDropdown,
                    intro: "Select the <strong>Area</strong> where the problem occurs.",
                    position: 'right'
                });
            }

            // New Equipment Group Dropdown in New Problem Form
            const newEquipmentGroupDropdown = document.getElementById('new_pst_equipmentGroupDropdown');
            if (newEquipmentGroupDropdown) {
                steps.push({
                    element: newEquipmentGroupDropdown,
                    intro: "Select the <strong>Equipment Group</strong> associated with the problem.",
                    position: 'right'
                });
            }

            // Create Problem Button
            const createButton = newProblemForm.querySelector('button[type="submit"]');
            if (createButton) {
                steps.push({
                    element: createButton,
                    intro: "After filling out all required fields, <strong>CLICK </strong>this button to create the new problem.",
                    position: 'right'
                });
            }
        }

        // ===== SOLUTIONS TAB =====
        const solutionTab = document.getElementById('solution-tab');
        if (solutionTab) {
            steps.push({
                element: solutionTab,
                intro: "After finding or creating a problem, <strong> CLICK </strong> on the <strong>Solutions tab</strong> to manage solutions.",
                position: 'right',
                onShow: function() {
                    // Switch to the Solutions tab
                    solutionTab.click();

                    // Restore search results to original state
                    const searchResults = document.getElementById('pst_searchResults');
                    if (searchResults && originalSearchResultsHTML !== null) {
                        searchResults.innerHTML = originalSearchResultsHTML;
                        searchResults.style.display = originalSearchResultsDisplay;
                    }
                }
            });

            // Problem Solutions Header
            const selectedProblemName = document.getElementById('selected-problem-name');
            if (selectedProblemName) {
                steps.push({
                    element: selectedProblemName,
                    intro: "This header shows which problem you're currently working with. It will display the name of the selected problem.",
                    position: 'right'
                });
            }

            // Related Solutions Dropdown
            const existingSolutions = document.getElementById('existing_solutions');
            if (existingSolutions) {
                steps.push({
                    element: existingSolutions,
                    intro: "This list shows all existing solutions for the selected problem. You can:<br>• <strong>Click</strong> on a solution to select it<br>• <strong>Double-click</strong> a solution to view its tasks<br>• Hold <strong>Ctrl</strong> or <strong>Shift</strong> to select multiple solutions for removal",
                    position: 'right'
                });
            }

            // New Solution Name Field
            const newSolutionName = document.getElementById('new_solution_name');
            if (newSolutionName) {
                steps.push({
                    element: newSolutionName,
                    intro: "Enter a descriptive <strong>name</strong> for your new solution here. Choose a clear, concise name that describes the approach to solving the problem.",
                    position: 'right'
                });
            }

            // New Solution Description Field
            const newSolutionDescription = document.getElementById('new_solution_description');
            if (newSolutionDescription) {
                steps.push({
                    element: newSolutionDescription,
                    intro: "Provide a detailed <strong>description</strong> of your solution here. Include any important context or constraints that apply to this solution.",
                    position: 'right'
                });
            }

            // Add Solution Button
            const addSolutionBtn = document.getElementById('addSolutionBtn');
            if (addSolutionBtn) {
                steps.push({
                    element: addSolutionBtn,
                    intro: "After entering a name and description, <strong>CLICK </strong>this button to <strong>add the new solution</strong> to the selected problem.",
                    position: 'right'
                });
            }

            // Remove Selected Solutions Button
            const removeSolutionsBtn = document.getElementById('removeSolutionsBtn');
            if (removeSolutionsBtn) {
                steps.push({
                    element: removeSolutionsBtn,
                    intro: "Select one or more solutions from the list above, then <strong>CLICK </strong>this button to <strong>remove</strong> them. You'll be prompted to confirm the deletion.",
                    position: 'left'
                });
            }
        }

        // ===== TASKS TAB =====
        const taskTab = document.getElementById('task-tab');
        if (taskTab) {
            steps.push({
                element: taskTab,
                intro: "Once you've selected a solution, <strong>CLICK </strong>on the <strong>Tasks tab</strong> to manage tasks for that solution.",
                position: 'right',
                onShow: function() {
                    // Switch to the Tasks tab
                    taskTab.click();
                }
            });

            const existingTasks = document.getElementById('existing_tasks');
            if (existingTasks) {
                steps.push({
                    element: existingTasks,
                    intro: "This list shows all tasks for the selected solution. Double-<strong>CLICK </strong>a task to edit it in detail.",
                    position: 'right'
                });
            }

            const newTaskName = document.getElementById('new_task_name');
            if (newTaskName) {
                steps.push({
                    element: newTaskName,
                    intro: "Add new tasks by providing a name and description, then clicking 'Add Task'.",
                    position: 'right'
                });
            }
        }

        // Enhanced Task Edit portion for the tour.js file
        // ===== EDIT TASK TAB =====
        const editTaskTab = document.getElementById('edit-task-tab');
        if (editTaskTab) {
            steps.push({
                element: editTaskTab,
                intro: "The <strong>Edit Task tab</strong> is where you configure all aspects of a task. You'll access this after creating a task or selecting an existing one to edit.",
                position: 'right',
                onShow: function() {
                    // Switch to the Edit Task tab
                    editTaskTab.click();
                }
            });

            // Task Name and Description
            const taskName = document.getElementById('pst_task_edit_task_name');
            if (taskName) {
                steps.push({
                    element: taskName,
                    intro: "Enter a clear, concise <strong>Task Name</strong> that describes what this task accomplishes. Good names help technicians quickly understand the task's purpose.",
                    position: 'right'
                });
            }

            const taskDescription = document.getElementById('pst_task_edit_task_description');
            if (taskDescription) {
                steps.push({
                    element: taskDescription,
                    intro: "Provide a detailed <strong>Task Description</strong> with step-by-step instructions or important context. Be thorough but clear in your explanation.",
                    position: 'right'
                });
            }

            const updateTaskBtn = document.getElementById('updateTaskDetailsBtn');
            if (updateTaskBtn) {
                steps.push({
                    element: updateTaskBtn,
                    intro: "After editing the name or description, <strong>CLICK</strong> this button to save those changes.",
                    position: 'right'
                });
            }

            // Tab Navigation
            const editTaskSubTabs = document.getElementById('editTaskSubTabs');
            if (editTaskSubTabs) {
                steps.push({
                    element: editTaskSubTabs,
                    intro: "These sub-tabs organize different types of information for your task. You'll need to work through each tab to fully configure the task.",
                    position: 'bottom'
                });
            }

            // TASK DETAILS TAB
            const taskDetailsTab = document.getElementById('task-details-tab');
            if (taskDetailsTab) {
                steps.push({
                    element: taskDetailsTab,
                    intro: "The <strong>Task Details</strong> tab is where you define position information - the specific equipment locations this task applies to.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Task Details tab
                        taskDetailsTab.click();
                    }
                });
            }

            const addPositionBtn = document.getElementById('addPositionBtn');
            if (addPositionBtn) {
                steps.push({
                    element: addPositionBtn,
                    intro: "<strong>CLICK</strong> this button to add a new position. You can add multiple positions if the task applies to different equipment locations.",
                    position: 'right',
                    onShow: function() {
                        // Make sure at least one position is visible for the tour, even if empty
                        if (positionsContainer && positionsContainer.children.length === 0) {
                            // Trigger a click on the Add Position button
                            addPositionBtn.click();
                        }
                    }
                });
            }

            // If a position is added, explain its fields
            // We'll target these by class since the IDs include unique identifiers
            setTimeout(() => {
                const areaDropdown = document.querySelector('.areaDropdown');
                if (areaDropdown) {
                    steps.push({
                        element: areaDropdown,
                        intro: "For each position, first select the <strong>Area</strong> where the equipment is located. This is the highest level in the equipment hierarchy.",
                        position: 'right'
                    });
                }

                const equipmentGroupDropdown = document.querySelector('.equipmentGroupDropdown');
                if (equipmentGroupDropdown) {
                    steps.push({
                        element: equipmentGroupDropdown,
                        intro: "Next, select the <strong>Equipment Group</strong>. This dropdown populates based on your Area selection.",
                        position: 'right'
                    });
                }

                const modelDropdown = document.querySelector('.modelDropdown');
                if (modelDropdown) {
                    steps.push({
                        element: modelDropdown,
                        intro: "Then select the equipment <strong>Model</strong>. This dropdown populates based on your Equipment Group selection.",
                        position: 'right'
                    });
                }

                const assetNumberInput = document.querySelector('.assetNumberInput');
                if (assetNumberInput) {
                    steps.push({
                        element: assetNumberInput,
                        intro: "Select an <strong>Asset Number</strong> if applicable. This identifies a specific piece of equipment within the model type.",
                        position: 'right'
                    });
                }

                const locationInput = document.querySelector('.locationInput');
                if (locationInput) {
                    steps.push({
                        element: locationInput,
                        intro: "Select a <strong>Location</strong> to specify where on the equipment this task takes place.",
                        position: 'right'
                    });
                }

                const assembliesDropdown = document.querySelector('.assembliesDropdown');
                if (assembliesDropdown) {
                    steps.push({
                        element: assembliesDropdown,
                        intro: "If applicable, select a <strong>Subassembly</strong> to specify which subcomponent the task involves.",
                        position: 'right'
                    });
                }

                const subassembliesDropdown = document.querySelector('.subassembliesDropdown');
                if (subassembliesDropdown) {
                    steps.push({
                        element: subassembliesDropdown,
                        intro: "Further refine the location by selecting a <strong>Component Assembly</strong> within the subassembly.",
                        position: 'right'
                    });
                }

                const assemblyViewsDropdown = document.querySelector('.assemblyViewsDropdown');
                if (assemblyViewsDropdown) {
                    steps.push({
                        element: assemblyViewsDropdown,
                        intro: "If needed, select an <strong>Assembly View</strong> to specify a particular view or configuration of the component.",
                        position: 'right'
                    });
                }

                const siteLocationDropdown = document.querySelector('.siteLocationDropdown');
                if (siteLocationDropdown) {
                    steps.push({
                        element: siteLocationDropdown,
                        intro: "Select a <strong>Site Location</strong> to indicate the physical location where this equipment is installed.",
                        position: 'right'
                    });
                }

                const savePositionBtn = document.querySelector('.savePositionBtn');
                if (savePositionBtn) {
                    steps.push({
                        element: savePositionBtn,
                        intro: "After configuring the position details, <strong>CLICK</strong> this button to save the position information.",
                        position: 'right'
                    });
                }
            }, 500); // Small delay to allow for DOM updates

            // IMAGES TAB
            const imagesTab = document.getElementById('task-images-tab');
            if (imagesTab) {
                steps.push({
                    element: imagesTab,
                    intro: "The <strong>Images</strong> tab allows you to associate relevant images with this task, such as equipment photos or visual guides.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Images tab
                        imagesTab.click();
                    }
                });
            }

            const imagesSelect = document.getElementById('pst_task_edit_task_images');
            if (imagesSelect) {
                steps.push({
                    element: imagesSelect,
                    intro: "This searchable dropdown lets you find and select images. You can type to search and select multiple images for the task.",
                    position: 'right'
                });
            }

            const saveImagesBtn = document.getElementById('saveImagesBtn');
            if (saveImagesBtn) {
                steps.push({
                    element: saveImagesBtn,
                    intro: "After selecting images, <strong>CLICK</strong> this button to save the image associations to the task.",
                    position: 'right'
                });
            }

            const selectedImages = document.getElementById('pst_task_edit_selected_images');
            if (selectedImages) {
                steps.push({
                    element: selectedImages,
                    intro: "This area displays all images currently associated with the task. Each image has a 'Remove' button to delete associations if needed.",
                    position: 'right'
                });
            }

            // PARTS TAB
            const partsTab = document.getElementById('task-parts-tab');
            if (partsTab) {
                steps.push({
                    element: partsTab,
                    intro: "The <strong>Parts</strong> tab lets you associate parts that are needed or affected by this task. This helps technicians know which parts to have on hand.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Parts tab
                        partsTab.click();
                    }
                });
            }

            const partsSelect = document.getElementById('pst_task_edit_task_parts');
            if (partsSelect) {
                steps.push({
                    element: partsSelect,
                    intro: "This searchable dropdown lets you find and select parts by part number or name. You can select multiple parts for the task.",
                    position: 'right'
                });
            }

            const savePartsBtn = document.getElementById('savePartsBtn');
            if (savePartsBtn) {
                steps.push({
                    element: savePartsBtn,
                    intro: "After selecting parts, <strong>CLICK</strong> this button to save the part associations to the task.",
                    position: 'right'
                });
            }

            const selectedParts = document.getElementById('pst_task_edit_selected_parts');
            if (selectedParts) {
                steps.push({
                    element: selectedParts,
                    intro: "This area displays all parts currently associated with the task, showing part numbers and names for quick reference.",
                    position: 'right'
                });
            }

            // DRAWINGS TAB
            const drawingsTab = document.getElementById('task-drawings-tab');
            if (drawingsTab) {
                steps.push({
                    element: drawingsTab,
                    intro: "The <strong>Drawings</strong> tab lets you associate technical drawings that are relevant to completing this task.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Drawings tab
                        drawingsTab.click();
                    }
                });
            }

            const drawingsSelect = document.getElementById('pst_task_edit_task_drawings');
            if (drawingsSelect) {
                steps.push({
                    element: drawingsSelect,
                    intro: "This searchable dropdown lets you find and select technical drawings by number or name. You can select multiple drawings.",
                    position: 'right'
                });
            }

            const saveDrawingsBtn = document.getElementById('saveDrawingsBtn');
            if (saveDrawingsBtn) {
                steps.push({
                    element: saveDrawingsBtn,
                    intro: "After selecting drawings, <strong>CLICK</strong> this button to save the drawing associations to the task.",
                    position: 'right'
                });
            }

            const selectedDrawings = document.getElementById('pst_task_edit_selected_drawings');
            if (selectedDrawings) {
                steps.push({
                    element: selectedDrawings,
                    intro: "This area displays all drawings currently associated with the task, showing drawing numbers and names for quick reference.",
                    position: 'right'
                });
            }

            // DOCUMENTS TAB
            const documentsTab = document.getElementById('task-documents-tab');
            if (documentsTab) {
                steps.push({
                    element: documentsTab,
                    intro: "The <strong>Documents</strong> tab lets you associate reference documents needed for completing this task, such as manuals or procedures.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Documents tab
                        documentsTab.click();
                    }
                });
            }

            const documentsSelect = document.getElementById('pst_task_edit_task_documents');
            if (documentsSelect) {
                steps.push({
                    element: documentsSelect,
                    intro: "This searchable dropdown lets you find and select documents by title or content. You can select multiple documents.",
                    position: 'right'
                });
            }

            const saveDocumentsBtn = document.getElementById('saveDocumentsBtn');
            if (saveDocumentsBtn) {
                steps.push({
                    element: saveDocumentsBtn,
                    intro: "After selecting documents, <strong>CLICK</strong> this button to save the document associations to the task.",
                    position: 'right'
                });
            }

            const selectedDocuments = document.getElementById('pst_task_edit_selected_documents');
            if (selectedDocuments) {
                steps.push({
                    element: selectedDocuments,
                    intro: "This area displays all documents currently associated with the task, showing document titles for quick reference.",
                    position: 'right'
                });
            }

            // TOOLS TAB
            const toolsTab = document.getElementById('task-tools-tab');
            if (toolsTab) {
                steps.push({
                    element: toolsTab,
                    intro: "The <strong>Tools</strong> tab lets you specify what tools are required to complete this task. This helps technicians prepare properly.",
                    position: 'bottom',
                    onShow: function() {
                        // Switch to Tools tab
                        toolsTab.click();
                    }
                });
            }

            const toolsSelect = document.getElementById('pst_task_edit_task_tools');
            if (toolsSelect) {
                steps.push({
                    element: toolsSelect,
                    intro: "This searchable dropdown lets you find and select tools by name or type. You can select multiple tools that are needed for this task.",
                    position: 'right'
                });
            }

            const saveToolsBtn = document.getElementById('saveToolsBtn');
            if (saveToolsBtn) {
                steps.push({
                    element: saveToolsBtn,
                    intro: "After selecting tools, <strong>CLICK</strong> this button to save the tool associations to the task.",
                    position: 'right'
                });
            }

            const selectedTools = document.getElementById('pst_task_edit_selected_tools');
            if (selectedTools) {
                steps.push({
                    element: selectedTools,
                    intro: "This area displays all tools currently associated with the task, showing tool names and types for quick reference.",
                    position: 'right'
                });
            }

            // Final Task Edit Guidance
            steps.push({
                element: editTaskTab,
                intro: "For a complete task, make sure to configure the position details and add any relevant images, parts, drawings, documents, and tools before moving on.",
                position: 'bottom'
            });
        }

        // Conclusion
        steps.push({
            element: document.querySelector('.container'),
            intro: "That's it! Remember to work left-to-right: Problem → Solution → Task → Edit Task. <strong>CLICK </strong>'Done' to finish the tour.",
            position: 'right',
            onShow: function() {
                // Return to the Problem tab for a fresh start
                if (problemTab) {
                    problemTab.click();
                }

                // Restore search results to original state
                const searchResults = document.getElementById('pst_searchResults');
                if (searchResults && originalSearchResultsHTML !== null) {
                    searchResults.innerHTML = originalSearchResultsHTML;
                    searchResults.style.display = originalSearchResultsDisplay;
                }

                // Close any open accordions
                const collapseSearch = bootstrap.Collapse.getInstance(document.getElementById('collapseSearchProblem')) ||
                                      new bootstrap.Collapse(document.getElementById('collapseSearchProblem'), {toggle: false});
                const collapseNew = bootstrap.Collapse.getInstance(document.getElementById('collapseNewProblem')) ||
                                   new bootstrap.Collapse(document.getElementById('collapseNewProblem'), {toggle: false});

                collapseSearch.hide();
                collapseNew.hide();
            }
        });

        return steps;
    }

    // Initialize the tour when the button is clicked
    tourButton.addEventListener('click', function() {
        const tour = introJs();

        // Configure the tour
        tour.setOptions({
            steps: defineTourSteps(),
            nextLabel: 'Next →',
            prevLabel: '← Back',
            skipLabel: 'Skip tour',
            doneLabel: 'Done',
            hideNext: false,
            hidePrev: false,
            exitOnOverlayClick: false,
            showStepNumbers: false, // Change to false to disable the default counter
            keyboardNavigation: true,
            showButtons: true,
            showBullets: true,
            scrollToElement: true,
            disableInteraction: false,
            tooltipPosition: 'auto',
            positionPrecedence: ['bottom', 'top', 'right', 'left'],
            tooltipClass: 'custom-tooltip',
            skipButtonClass: 'centered-skip'
        });

        // Handle button positioning and step counter
        tour.onafterchange(function() {
            // Run after a small delay to ensure DOM is updated
            setTimeout(() => {
                // Current step is 0-based, so add 1 for display
                const currentStep = this._currentStep + 1;
                const totalSteps = this._options.steps.length;

                // Find the tooltip text container
                const tooltipText = document.querySelector('.introjs-tooltiptext');

                // Check if our custom counter already exists and remove it if so
                const existingCounter = document.getElementById('custom-step-counter');
                if (existingCounter) {
                    existingCounter.remove();
                }

                // Create our custom counter
                if (tooltipText) {
                    const counterDiv = document.createElement('div');
                    counterDiv.id = 'custom-step-counter';
                    counterDiv.style.textAlign = 'center';
                    counterDiv.style.marginTop = '10px';
                    counterDiv.style.paddingTop = '5px';
                    counterDiv.style.borderTop = '1px solid rgba(255,255,255,0.2)';

                    // Create separate spans for each part to ensure correct order
                    const stepSpan = document.createElement('span');
                    stepSpan.textContent = currentStep;

                    const ofSpan = document.createElement('span');
                    ofSpan.textContent = ' of ';

                    const totalSpan = document.createElement('span');
                    totalSpan.textContent = totalSteps;

                    // Clear and append in the correct order
                    counterDiv.innerHTML = '';
                    counterDiv.appendChild(stepSpan);
                    counterDiv.appendChild(ofSpan);
                    counterDiv.appendChild(totalSpan);

                    // Append to tooltip
                    tooltipText.appendChild(counterDiv);
                }

                // Button positioning logic
                const skipButton = document.querySelector('.introjs-skipbutton');
                const tooltipButtons = document.querySelector('.introjs-tooltipbuttons');

                if (skipButton && tooltipButtons) {
                    // First, try to remove any skip button that might be outside the tooltip
                    const outsideSkipButton = document.querySelector('body > .introjs-skipbutton');
                    if (outsideSkipButton) {
                        outsideSkipButton.remove();
                    }

                    // If skip button is not already inside the tooltip buttons, move it there
                    if (!tooltipButtons.contains(skipButton)) {
                        tooltipButtons.appendChild(skipButton);
                    }

                    // Style the skip button
                    skipButton.classList.add('btn', 'btn-danger', 'btn-sm');

                    // Set up the flexbox layout for buttons
                    tooltipButtons.style.display = 'flex';
                    tooltipButtons.style.justifyContent = 'space-between';
                    tooltipButtons.style.width = '100%';

                    // Position Back button on the left
                    const backButton = document.querySelector('.introjs-prevbutton');
                    if (backButton) {
                        backButton.style.order = '0';
                        backButton.style.marginRight = 'auto';
                        // Make sure it's first in DOM order too
                        tooltipButtons.insertBefore(backButton, tooltipButtons.firstChild);
                    }

                    // Position Skip button in the center
                    skipButton.style.order = '1';
                    skipButton.style.margin = '0 10px';

                    // Position Next button on the right
                    const nextButton = document.querySelector('.introjs-nextbutton');
                    if (nextButton) {
                        nextButton.style.order = '2';
                        nextButton.style.marginLeft = 'auto';
                        // Make sure it's last in DOM order too
                        tooltipButtons.appendChild(nextButton);
                    }
                }
            }, 50); // Small delay to ensure DOM is updated
        });

        // Clean up function to restore original state when tour ends
        tour.onexit(function() {
            // Restore search results to original state
            const searchResults = document.getElementById('pst_searchResults');
            if (searchResults && originalSearchResultsHTML !== null) {
                searchResults.innerHTML = originalSearchResultsHTML;
                searchResults.style.display = originalSearchResultsDisplay;
            }
        });

        // Start the tour
        tour.start();
    });
});